# modified from https://github.com/feng-yufei/shared_debugging_code/blob/main/train_t2s.py
import os

if "_CUDA_VISIBLE_DEVICES" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["_CUDA_VISIBLE_DEVICES"]
import argparse
import inspect
import logging
import math
import platform
import sys
from pathlib import Path

import torch
from AR.data.data_module import Text2SemanticDataModule
from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from AR.utils.io import load_yaml_config
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger  # WandbLogger
from pytorch_lightning.strategies import DDPStrategy

logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
torch.set_float32_matmul_precision("high")
from collections import OrderedDict

from AR.utils import get_newest_ckpt
from process_ckpt import my_save

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.runtime_status import clear_activity, publish_activity


class RuntimeStatusCallback(Callback):
    def __init__(self, total_epochs):
        super().__init__()
        self.total_epochs = int(total_epochs)

    @staticmethod
    def _metric_value(value):
        if value is None:
            return None
        try:
            if hasattr(value, "detach"):
                value = value.detach()
            if hasattr(value, "float"):
                value = value.float()
            if hasattr(value, "item"):
                value = value.item()
            return float(value)
        except (TypeError, ValueError, RuntimeError):
            return None

    def _publish(self, trainer, batch_idx=0):
        if not trainer.is_global_zero:
            return

        total_steps = None
        try:
            estimated = float(trainer.estimated_stepping_batches)
            if math.isfinite(estimated) and estimated > 0:
                total_steps = int(estimated)
        except (TypeError, ValueError, OverflowError):
            pass

        step = int(trainer.global_step)
        progress = (
            min(step / total_steps, 1.0)
            if total_steps
            else None
        )
        loss = self._metric_value(
            trainer.callback_metrics.get("total_loss")
        )
        accuracy = self._metric_value(
            trainer.callback_metrics.get("top_3_acc")
        )
        metric = (
            f"top-3 acc {accuracy:.2%}"
            if accuracy is not None
            else None
        )
        publish_activity(
            task="GPT / S1 학습",
            stage="training",
            progress=progress,
            step=step,
            total_steps=total_steps,
            epoch=int(trainer.current_epoch) + 1,
            total_epochs=self.total_epochs,
            loss=loss,
            metric=metric,
        )

    def on_train_start(self, trainer, pl_module):
        self._publish(trainer)

    def on_train_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
    ):
        try:
            total_batches = int(trainer.num_training_batches)
        except (TypeError, ValueError, OverflowError):
            total_batches = 0
        interval = max(1, total_batches // 100) if total_batches else 5
        if batch_idx % interval == 0 or (
            total_batches and batch_idx + 1 >= total_batches
        ):
            self._publish(trainer, batch_idx=batch_idx)


class my_model_ckpt(ModelCheckpoint):
    def __init__(
        self,
        config,
        if_save_latest,
        if_save_every_weights,
        half_weights_save_dir,
        exp_name,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.if_save_latest = if_save_latest
        self.if_save_every_weights = if_save_every_weights
        self.half_weights_save_dir = half_weights_save_dir
        self.exp_name = exp_name
        self.config = config
        self.low_disk_save = (
            os.environ.get("GPT_SOVITS_LOW_DISK_SAVE") == "1"
        )

    def on_train_epoch_end(self, trainer, pl_module):
        # if not self._should_skip_saving_checkpoint(trainer) and self._should_save_on_train_epoch_end(trainer):
        if self._should_save_on_train_epoch_end(trainer):
            monitor_candidates = self._monitor_candidates(trainer)
            if self._every_n_epochs >= 1 and (trainer.current_epoch + 1) % self._every_n_epochs == 0:
                if self.low_disk_save:
                    logging.info(
                        "[low-disk] skipping full Lightning resume checkpoint; "
                        "saving compact inference weights only"
                    )
                else:
                    if (
                        self.if_save_latest == True
                    ):  ####如果设置只保存最后一个ckpt，在保存下一个ckpt后要清理掉之前的所有ckpt
                        to_clean = list(os.listdir(self.dirpath))
                    self._save_topk_checkpoint(trainer, monitor_candidates)
                    if self.if_save_latest == True:
                        for name in to_clean:
                            try:
                                os.remove("%s/%s" % (self.dirpath, name))
                            except:
                                pass

                # In low-disk mode compact weights are mandatory because
                # full Lightning checkpoints are intentionally omitted.
                save_compact_weights = (
                    self.if_save_every_weights == True
                    or self.low_disk_save
                )
                if save_compact_weights:
                    to_save_od = OrderedDict()
                    to_save_od["weight"] = OrderedDict()
                    lightning_module = (
                        trainer.strategy._lightning_module
                    )
                    if hasattr(
                        lightning_module,
                        "inference_state_dict",
                    ):
                        dictt = lightning_module.inference_state_dict()
                    else:
                        dictt = lightning_module.state_dict()
                    for key in dictt:
                        to_save_od["weight"][key] = (
                            dictt[key].detach().half().cpu()
                        )
                    to_save_od["config"] = self.config
                    to_save_od["info"] = "GPT-e%s" % (trainer.current_epoch + 1)

                    if os.environ.get("LOCAL_RANK", "0") == "0":
                        if self.low_disk_save and self.if_save_latest:
                            weight_dir = Path(self.half_weights_save_dir)
                            for old_weight in weight_dir.glob(
                                f"{self.exp_name}-e*.ckpt"
                            ):
                                try:
                                    old_weight.unlink()
                                except OSError:
                                    logging.warning(
                                        "[low-disk] failed to remove old weight: %s",
                                        old_weight,
                                    )

                        my_save(
                            to_save_od,
                            "%s/%s-e%s.ckpt"
                            % (
                                self.half_weights_save_dir,
                                self.exp_name,
                                trainer.current_epoch + 1,
                            ),
                        )

            if not self.low_disk_save:
                self._save_last_checkpoint(trainer, monitor_candidates)


def main(args):
    config = load_yaml_config(args.config_file)

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_dir = output_dir / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(config["train"]["seed"], workers=True)
    ckpt_callback: ModelCheckpoint = my_model_ckpt(
        config=config,
        if_save_latest=config["train"]["if_save_latest"],
        if_save_every_weights=config["train"]["if_save_every_weights"],
        half_weights_save_dir=config["train"]["half_weights_save_dir"],
        exp_name=config["train"]["exp_name"],
        save_top_k=-1,
        monitor="top_3_acc",
        mode="max",
        save_on_train_epoch_end=True,
        every_n_epochs=config["train"]["save_every_n_epoch"],
        dirpath=ckpt_dir,
    )
    logger = TensorBoardLogger(name=output_dir.stem, save_dir=output_dir)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["USE_LIBUV"] = "0"
    trainer: Trainer = Trainer(
        max_epochs=config["train"]["epochs"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        # val_check_interval=9999999999999999999999,###不要验证
        # check_val_every_n_epoch=None,
        limit_val_batches=0,
        devices=-1 if torch.cuda.is_available() else 1,
        benchmark=False,
        fast_dev_run=False,
        strategy=DDPStrategy(process_group_backend="nccl" if platform.system() != "Windows" else "gloo")
        if torch.cuda.is_available()
        else "auto",
        precision=config["train"]["precision"],
        logger=logger,
        num_sanity_val_steps=0,
        callbacks=[
            ckpt_callback,
            RuntimeStatusCallback(config["train"]["epochs"]),
        ],
        use_distributed_sampler=False,  # 非常简单的修改，但解决了采用自定义的 bucket_sampler 下训练步数不一致的问题！
    )

    model: Text2SemanticLightningModule = Text2SemanticLightningModule(config, output_dir)

    data_module: Text2SemanticDataModule = Text2SemanticDataModule(
        config,
        train_semantic_path=config["train_semantic_path"],
        train_phoneme_path=config["train_phoneme_path"],
        # dev_semantic_path=args.dev_semantic_path,
        # dev_phoneme_path=args.dev_phoneme_path
    )

    try:
        # 使用正则表达式匹配文件名中的数字部分，并按数字大小进行排序
        newest_ckpt_name = get_newest_ckpt(os.listdir(ckpt_dir))
        ckpt_path = ckpt_dir / newest_ckpt_name
    except Exception:
        ckpt_path = None
    print("ckpt_path:", ckpt_path)
    fit_kwargs = {"ckpt_path": ckpt_path}
    # PyTorch 2.6 changed torch.load's default to weights_only=True, and
    # Lightning 2.6+ exposes that choice through Trainer.fit(). GPT-SoVITS
    # resume checkpoints are full Lightning checkpoints produced locally by
    # this training run, so they require weights_only=False. Keep compatibility
    # with Lightning versions whose Trainer.fit() predates this argument.
    if "weights_only" in inspect.signature(trainer.fit).parameters:
        fit_kwargs["weights_only"] = False
    try:
        trainer.fit(model, data_module, **fit_kwargs)
    except BaseException:
        if os.environ.get("LOCAL_RANK", "0") == "0":
            publish_activity(
                task="GPT / S1 학습",
                stage="failed",
                state="failed",
                metric="학습 프로세스가 오류 또는 중지로 종료되었습니다.",
            )
        raise
    else:
        if os.environ.get("LOCAL_RANK", "0") == "0":
            clear_activity(reason="gpt_finished")


# srun --gpus-per-node=1 --ntasks-per-node=1 python train.py --path-to-configuration configurations/default.yaml
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config_file",
        type=str,
        default="configs/s1longer.yaml",
        help="path of config file",
    )
    # args for dataset
    # parser.add_argument('--train_semantic_path',type=str,default='/data/docker/liujing04/gpt-vits/fine_tune_dataset/xuangou/6-name2semantic.tsv')
    # parser.add_argument('--train_phoneme_path', type=str, default='/data/docker/liujing04/gpt-vits/fine_tune_dataset/xuangou/2-name2text.txt')

    # parser.add_argument('--dev_semantic_path', type=str, default='dump_mix/semantic_dev.tsv')
    # parser.add_argument('--dev_phoneme_path', type=str, default='dump_mix/phoneme_dev.npy')
    # parser.add_argument('--output_dir',type=str,default='/data/docker/liujing04/gpt-vits/fine_tune_dataset/xuangou/logs_s1',help='directory to save the results')
    # parser.add_argument('--output_dir',type=str,default='/liujing04/gpt_logs/s1/xuangou_ft',help='directory to save the results')

    args = parser.parse_args()
    logging.info(str(args))
    main(args)
