import warnings

warnings.filterwarnings("ignore")
import os
import sys
import copy
import json
import math
import time
from pathlib import Path

import utils

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.runtime_status import clear_activity, publish_activity
from tools.grokking import (
    deterministic_holdout_indices,
    repeat_indices_to_minimum,
    gradfilter_ema,
    parameter_l2_norm,
    resolve_grokking_config,
)

hps = utils.get_hparams(stage=2)
os.environ["version"] = hps.model.version
os.environ["CUDA_VISIBLE_DEVICES"] = hps.train.gpu_numbers.replace("-", ",")
import logging

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

logging.getLogger("matplotlib").setLevel(logging.INFO)
logging.getLogger("h5py").setLevel(logging.INFO)
logging.getLogger("numba").setLevel(logging.INFO)
from collections import OrderedDict as od
from random import randint

from module import commons
from module.data_utils import (
    DistributedBucketSampler,
    TextAudioSpeakerCollateV3,
    TextAudioSpeakerLoaderV3,
    TextAudioSpeakerCollateV4,
    TextAudioSpeakerLoaderV4,
)
from module.models import (
    SynthesizerTrnV3 as SynthesizerTrn,
)
from peft import LoraConfig, get_peft_model
from process_ckpt import savee

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = False
###反正A100fp32更快，那试试tf32吧
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("medium")  # 最低精度但最快（也就快一丁点），对于结果造成不了影响
# from config import pretrained_s2G,pretrained_s2D
global_step = 0

device = "cpu"  # cuda以外的设备，等mps优化后加入


def main():
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
    else:
        n_gpus = 1
    if n_gpus <= 1:
        run(0, n_gpus, hps)
        return

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(randint(20000, 55555))

    mp.spawn(
        run,
        nprocs=n_gpus,
        args=(
            n_gpus,
            hps,
        ),
    )


def run(rank, n_gpus, hps):
    use_ddp = torch.cuda.is_available() and n_gpus > 1
    succeeded = False
    try:
        if use_ddp:
            dist.init_process_group(
                backend="gloo" if os.name == "nt" else "nccl",
                init_method="env://?use_libuv=False",
                world_size=n_gpus,
                rank=rank,
            )
        _run(rank, n_gpus, hps, use_ddp)
        succeeded = True
    except BaseException:
        if rank == 0:
            publish_activity(
                task="SoVITS LoRA 학습",
                stage="failed",
                state="failed",
                metric="학습 프로세스가 오류 또는 중지로 종료되었습니다.",
            )
        raise
    finally:
        if rank == 0 and succeeded:
            clear_activity(reason="sovits_lora_finished")
        if use_ddp and dist.is_initialized():
            dist.destroy_process_group()


def _split_cfm_dataset(dataset, cfg):
    # V3/V4/V5 loaders expose names and bucket lengths in parallel arrays.
    train_ids, val_ids = deterministic_holdout_indices(
        [row[0] for row in dataset.audiopaths_sid_text],
        fraction=float(cfg["validation_fraction"]),
        min_validation_samples=int(cfg["min_validation_samples"]),
        seed=int(cfg["seed"]),
    )
    def view(indices):
        result = copy.copy(dataset)
        result.audiopaths_sid_text = [dataset.audiopaths_sid_text[i] for i in indices]
        result.lengths = [dataset.lengths[i] for i in indices]
        return result
    return view(repeat_indices_to_minimum(train_ids, 100)), view(val_ids)


@torch.no_grad()
def _validate_cfm(model, loader, rank, cfg, sample_losses=None):
    # CFM samples both prompt length and noise/time. Compare exactly the
    # same held-out utterances and two deterministic draws at every epoch.
    dev = torch.device("cuda", rank) if torch.cuda.is_available() else torch.device("cpu")
    was_training = model.training
    model.eval()
    values = []
    try:
        devices = [rank] if dev.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            for index, batch in enumerate(loader):
                if index >= int(cfg["max_validation_batches"]):
                    break
                batch = [tensor.to(dev) for tensor in batch]
                for draw in range(2):
                    seed = int(cfg["seed"]) + 1009 * index + 37 * draw
                    torch.random.default_generator.manual_seed(seed)
                    if dev.type == "cuda":
                        torch.cuda.manual_seed(seed)
                    with autocast(enabled=False):
                        value = float(model(*batch, use_grad_ckpt=False).detach().float().item())
                    if not math.isfinite(value):
                        raise RuntimeError("Non-finite held-out CFM loss")
                    values.append(value)
                if sample_losses is not None:
                    sample_losses.append(sum(values[-2:]) / 2)
    finally:
        model.train(was_training)
    if not values:
        raise RuntimeError("Empty CFM validation set")
    return sum(values) / len(values)


def _save_cfm_best(model, hps, epoch):
    weights = od(
        (key, value.detach().half().cpu())
        for key, value in model.state_dict().items()
        if key not in no_grad_names
    )
    Path(hps.save_weight_dir).mkdir(parents=True, exist_ok=True)
    result = savee(
        weights, hps.name + "_grok_cfm_best", epoch, global_step, hps,
        model_version=hps.model.version, lora_rank=lora_rank,
    )
    if result != "Success.":
        raise RuntimeError("CFM best export failed: " + str(result))


def _run(rank, n_gpus, hps, use_ddp):
    global global_step, no_grad_names, save_root, lora_rank
    if rank == 0:
        logger = utils.get_logger(hps.data.exp_dir)
        logger.info(hps)
        # utils.check_git_hash(hps.s2_ckpt_dir)
        writer = SummaryWriter(log_dir=hps.s2_ckpt_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.s2_ckpt_dir, "eval"))

    torch.manual_seed(hps.train.seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

    TextAudioSpeakerLoader = TextAudioSpeakerLoaderV3 if hps.model.version == "v3" else TextAudioSpeakerLoaderV4
    TextAudioSpeakerCollate = TextAudioSpeakerCollateV3 if hps.model.version == "v3" else TextAudioSpeakerCollateV4
    train_dataset = TextAudioSpeakerLoader(hps.data)  ########
    grokking = resolve_grokking_config(hps.train, "s2")
    base_epochs = int(hps.train.epochs)
    eval_dataset = None
    if grokking["enabled"]:
        if int(grokking["max_validation_batches"]) < 1:
            raise ValueError("CFM probe requires at least one validation batch")
        train_dataset, eval_dataset = _split_cfm_dataset(train_dataset, grokking)
        metrics_path = Path(hps.data.exp_dir) / "grokking_s2_cfm_metrics.jsonl"
        best_path = Path(hps.save_weight_dir) / (hps.name + "_grok_cfm_best.pth")
        if metrics_path.exists() or best_path.exists():
            raise RuntimeError("CFM grokking requires a fresh experiment name/output directory")
        if not hps.train.pretrained_s2G or not os.path.isfile(hps.train.pretrained_s2G):
            raise RuntimeError("CFM grokking requires an upstream pretrained S2 checkpoint")
        fixed_epochs = int(os.environ.get("GPT_SOVITS_GROKKING_S2_FIXED_EPOCHS", "0"))
        if fixed_epochs < 0:
            raise ValueError("fixed S2 epoch budget must be nonnegative")
        hps.train.epochs = fixed_epochs or (base_epochs + 4 * int(grokking["s2_post_fit_epochs"]))
        if rank == 0:
            logger.info("[grokking:CFM] version=%s train=%s heldout=%s hard_max=%s",
                        hps.model.version, len(train_dataset), len(eval_dataset), hps.train.epochs)
    train_sampler = DistributedBucketSampler(
        train_dataset,
        hps.train.batch_size,
        [
            32,
            300,
            400,
            500,
            600,
            700,
            800,
            900,
            1000,
            # 1100,
            # 1200,
            # 1300,
            # 1400,
            # 1500,
            # 1600,
            # 1700,
            # 1800,
            # 1900,
        ],
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True,
    )
    collate_fn = TextAudioSpeakerCollate()
    worker_count = 4
    loader_kwargs = dict(
        num_workers=worker_count,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
        batch_sampler=train_sampler,
    )
    if worker_count > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    train_loader = DataLoader(
        train_dataset,
        **loader_kwargs,
    )
    eval_loader = None
    train_probe_loader = None
    if eval_dataset is not None and rank == 0:
        eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False,
                                 num_workers=0, collate_fn=collate_fn)
        # Use the same deterministic evaluation mode and draw count for
        # train and held-out probes. Select unique names, never repeated
        # padding entries from small datasets.
        seen = set()
        indices = []
        limit = min(len(eval_dataset), int(grokking["max_validation_batches"]))
        for i, row in enumerate(train_dataset.audiopaths_sid_text):
            if row[0] not in seen:
                seen.add(row[0])
                indices.append(i)
                if len(indices) >= limit:
                    break
        train_probe_loader = DataLoader(
            torch.utils.data.Subset(train_dataset, indices),
            batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn,
        )
    save_root = "%s/logs_s2_%s_lora_%s" % (hps.data.exp_dir, hps.model.version, hps.train.lora_rank)
    if grokking["enabled"]:
        save_root += "_grok_cfm"
    os.makedirs(save_root, exist_ok=True)
    lora_rank = int(hps.train.lora_rank)
    lora_config = LoraConfig(
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        r=lora_rank,
        lora_alpha=lora_rank,
        init_lora_weights=True,
    )

    def get_model(hps):
        return SynthesizerTrn(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model,
        )

    def get_optim(net_g):
        return torch.optim.AdamW(
            filter(lambda p: p.requires_grad, net_g.parameters()),  ###默认所有层lr一致
            hps.train.learning_rate,
            betas=hps.train.betas,
            eps=hps.train.eps,
            weight_decay=float(grokking["s2_weight_decay"]) if grokking["enabled"] else 0.01,
        )

    def model2cuda(net_g, rank):
        if torch.cuda.is_available():
            net_g = net_g.cuda(rank)
            if use_ddp:
                net_g = DDP(net_g, device_ids=[rank], find_unused_parameters=True)
        else:
            net_g = net_g.to(device)
        return net_g

    try:  # 如果能加载自动resume
        if grokking["enabled"]:
            # Start from generic pretrained weights only, never a prior fit.
            raise FileNotFoundError("fresh CFM research run")
        net_g = get_model(hps)
        net_g.cfm = get_peft_model(net_g.cfm, lora_config)
        net_g = model2cuda(net_g, rank)
        optim_g = get_optim(net_g)
        # _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g,load_opt=0)
        _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(save_root, "G_*.pth"),
            net_g,
            optim_g,
        )
        epoch_str += 1
        global_step = (epoch_str - 1) * len(train_loader)
    except:  # 如果首次不能加载，加载pretrain
        # traceback.print_exc()
        epoch_str = 1
        global_step = 0
        net_g = get_model(hps)
        if (
            hps.train.pretrained_s2G != ""
            and hps.train.pretrained_s2G != None
            and os.path.exists(hps.train.pretrained_s2G)
        ):
            if rank == 0:
                logger.info("loaded pretrained %s" % hps.train.pretrained_s2G)
            print(
                "loaded pretrained %s" % hps.train.pretrained_s2G,
                net_g.load_state_dict(
                    torch.load(hps.train.pretrained_s2G, map_location="cpu", weights_only=False)["weight"],
                    strict=False,
                ),
            )
        net_g.cfm = get_peft_model(net_g.cfm, lora_config)
        net_g = model2cuda(net_g, rank)
        optim_g = get_optim(net_g)

    no_grad_names = set()
    for name, param in net_g.named_parameters():
        if not param.requires_grad:
            no_grad_names.add(name.replace("module.", ""))
            # print(name, "not requires_grad")
    # print(no_grad_names)
    # os._exit(233333)

    if grokking["enabled"]:
        scheduler_g = torch.optim.lr_scheduler.LambdaLR(optim_g, lr_lambda=lambda _: 1.0)
    else:
        scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay, last_epoch=-1)
        for _ in range(epoch_str):
            scheduler_g.step()

    scaler = GradScaler(enabled=hps.train.fp16_run)

    net_d = optim_d = scheduler_d = None
    print("start training from epoch %s" % epoch_str)
    if rank == 0:
        total_steps = int(hps.train.epochs) * len(train_loader)
        publish_activity(
            task="SoVITS LoRA 학습",
            stage="starting",
            progress=min(global_step / total_steps, 1.0)
            if total_steps
            else None,
            step=global_step,
            total_steps=total_steps,
            epoch=epoch_str,
            total_epochs=int(hps.train.epochs),
        )
    grokfast_state = {}
    best_train = float("inf")
    best_val = float("inf")
    last_improvement = epoch_str - 1
    fit_epoch = None
    fit_val = None
    pre_fit_best = None
    if grokking["enabled"] and rank == 0:
        model = net_g.module if hasattr(net_g, "module") else net_g
        val_samples = []
        train_samples = []
        baseline = _validate_cfm(model, eval_loader, rank, grokking, val_samples)
        train_baseline = _validate_cfm(model, train_probe_loader, rank, grokking, train_samples)
        (Path(hps.data.exp_dir) / "grokking_s2_baseline.json").write_text(
            json.dumps({"epoch": 0, "val_cfm": baseline,
                        "train_eval_cfm": train_baseline,
                        "val_sample_losses": val_samples,
                        "train_eval_sample_losses": train_samples}, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        logger.info("[grokking:CFM] pretrained baseline %.8f", baseline)
    if grokking["enabled"] and use_ddp:
        dist.barrier()
    for epoch in range(epoch_str, hps.train.epochs + 1):
        epoch_start = time.monotonic()
        train_cfm = train_and_evaluate(
            rank, epoch, hps, [net_g, net_d], [optim_g, optim_d],
            [scheduler_g, scheduler_d], scaler, [train_loader, None],
            logger if rank == 0 else None,
            [writer, writer_eval] if rank == 0 else None,
            grokking=grokking, grokfast_state=grokfast_state,
        )
        scheduler_g.step()
        if grokking["enabled"]:
            stop = False
            if rank == 0:
                model = net_g.module if hasattr(net_g, "module") else net_g
                publish_activity(task="SoVITS LoRA 학습", stage="validation", epoch=epoch,
                                 total_epochs=int(hps.train.epochs))
                val_samples = []
                train_samples = []
                val_cfm = _validate_cfm(model, eval_loader, rank, grokking, val_samples)
                train_eval_cfm = _validate_cfm(model, train_probe_loader, rank, grokking, train_samples)
                # Online training losses use random draws and evolving weights.
                # Detect a plateau on the fixed, eval-mode train probe instead.
                if train_eval_cfm < best_train * (1 - float(grokking["s2_plateau_relative_delta"])):
                    best_train = train_eval_cfm
                    last_improvement = epoch
                    fit_epoch = fit_val = pre_fit_best = None
                if fit_epoch is None and epoch - last_improvement >= int(grokking["s2_plateau_patience"]):
                    fit_epoch = epoch
                    fit_val = val_cfm
                    pre_fit_best = best_val
                age = None if fit_epoch is None else epoch - fit_epoch
                candidate = bool(
                    age is not None and age >= 2 and fit_val > 0
                    and val_cfm <= 0.95 * fit_val and val_cfm < pre_fit_best
                    and train_eval_cfm <= best_train * 1.05
                )
                if val_cfm < best_val:
                    _save_cfm_best(model, hps, epoch)
                    best_val = val_cfm
                record = {
                    "epoch": epoch, "train_cfm": train_cfm, "val_cfm": val_cfm,
                    "train_eval_cfm": train_eval_cfm,
                    "val_sample_losses": val_samples,
                    "train_eval_sample_losses": train_samples,
                    "fit_metric": "train_eval_cfm",
                    "epoch_seconds": time.monotonic() - epoch_start,
                    "best_val_cfm": best_val, "fit_epoch": fit_epoch,
                    "fit_val_cfm": fit_val, "post_fit_age": age,
                    "delayed_improvement_candidate": candidate,
                    "trainable_parameter_l2": parameter_l2_norm(model),
                    "grokfast": bool(grokking["grokfast_enabled"]),
                    "weight_decay": float(grokking["s2_weight_decay"]),
                }
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, allow_nan=False) + "\n")
                logger.info("[grokking:CFM] %s", record)
                stop = bool(not fixed_epochs and age is not None and age >= int(grokking["s2_post_fit_epochs"])
                            and epoch >= base_epochs)
            if use_ddp:
                flag = torch.tensor(int(stop), device=torch.device("cuda", rank))
                dist.broadcast(flag, src=0)
                stop = bool(flag.item())
            if stop:
                break
    print("training done")


def train_and_evaluate(rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers,
                       grokking=None, grokfast_state=None):
    net_g, net_d = nets
    optim_g, optim_d = optims
    # scheduler_g, scheduler_d = schedulers
    train_loader, eval_loader = loaders
    if writers is not None:
        writer, writer_eval = writers

    train_loader.batch_sampler.set_epoch(epoch)
    global global_step

    net_g.train()
    probe = bool(grokking and grokking["enabled"])
    if probe:
        stats = torch.zeros(2, dtype=torch.float64, device=next(net_g.parameters()).device)
    for batch_idx, (ssl, spec, mel, ssl_lengths, spec_lengths, text, text_lengths, mel_lengths) in enumerate(
        tqdm(train_loader)
    ):
        if torch.cuda.is_available():
            spec, spec_lengths = (
                spec.cuda(
                    rank,
                    non_blocking=True,
                ),
                spec_lengths.cuda(
                    rank,
                    non_blocking=True,
                ),
            )
            mel, mel_lengths = mel.cuda(rank, non_blocking=True), mel_lengths.cuda(rank, non_blocking=True)
            ssl = ssl.cuda(rank, non_blocking=True)
            ssl.requires_grad = False
            text, text_lengths = (
                text.cuda(
                    rank,
                    non_blocking=True,
                ),
                text_lengths.cuda(
                    rank,
                    non_blocking=True,
                ),
            )
        else:
            spec, spec_lengths = spec.to(device), spec_lengths.to(device)
            mel, mel_lengths = mel.to(device), mel_lengths.to(device)
            ssl = ssl.to(device)
            ssl.requires_grad = False
            text, text_lengths = text.to(device), text_lengths.to(device)

        with autocast(enabled=hps.train.fp16_run):
            cfm_loss = net_g(
                ssl,
                spec,
                mel,
                ssl_lengths,
                spec_lengths,
                text,
                text_lengths,
                mel_lengths,
                use_grad_ckpt=hps.train.grad_ckpt,
            )
            loss_gen_all = cfm_loss
        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        if probe and grokking["grokfast_enabled"]:
            gradfilter_ema(
                net_g.module if hasattr(net_g, "module") else net_g,
                grokfast_state,
                alpha=float(grokking["grokfast_alpha"]),
                lamb=float(grokking["grokfast_lambda"]),
            )
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
        scaler.step(optim_g)
        scaler.update()
        if probe:
            stats[0] += loss_gen_all.detach().double() * ssl.shape[0]
            stats[1] += ssl.shape[0]

        if rank == 0:
            status_interval = max(1, len(train_loader) // 100)
            if (
                batch_idx % status_interval == 0
                or batch_idx + 1 == len(train_loader)
            ):
                total_steps = int(hps.train.epochs) * len(train_loader)
                completed_step = global_step + 1
                publish_activity(
                    task="SoVITS LoRA 학습",
                    stage="training",
                    progress=min(completed_step / total_steps, 1.0)
                    if total_steps
                    else None,
                    step=completed_step,
                    total_steps=total_steps,
                    epoch=epoch,
                    total_epochs=int(hps.train.epochs),
                    loss=float(loss_gen_all.detach().float().item()),
                    metric=f"LoRA rank {lora_rank}",
                    extra={"batch": batch_idx + 1, "total_batches": len(train_loader)},
                )

            if global_step % hps.train.log_interval == 0:
                lr = optim_g.param_groups[0]["lr"]
                losses = [cfm_loss]
                logger.info("Train Epoch: {} [{:.0f}%]".format(epoch, 100.0 * batch_idx / len(train_loader)))
                logger.info([x.item() for x in losses] + [global_step, lr])

                scalar_dict = {"loss/g/total": loss_gen_all, "learning_rate": lr, "grad_norm_g": grad_norm_g}
                utils.summarize(
                    writer=writer,
                    global_step=global_step,
                    scalars=scalar_dict,
                )

        global_step += 1
    if probe:
        # The research path exports only its best held-out compact weights.
        # Avoid polluting the regular training checkpoint namespace.
        if dist.is_initialized():
            dist.all_reduce(stats)
        total, count = stats.tolist()
        if count <= 0 or not math.isfinite(total):
            raise RuntimeError("Invalid CFM training statistics")
        return total / count
    if epoch % hps.train.save_every_epoch == 0 and rank == 0:
        low_disk_save = os.environ.get("GPT_SOVITS_LOW_DISK_SAVE") == "1"

        if low_disk_save:
            logger.info(
                "[low-disk] skipping full model+optimizer resume checkpoint; "
                "saving compact inference weights only"
            )
        elif hps.train.if_save_latest == 0:
            utils.save_checkpoint(
                net_g,
                optim_g,
                hps.train.learning_rate,
                epoch,
                os.path.join(save_root, "G_{}.pth".format(global_step)),
            )
        else:
            utils.save_checkpoint(
                net_g,
                optim_g,
                hps.train.learning_rate,
                epoch,
                os.path.join(save_root, "G_{}.pth".format(233333333333)),
            )

        # In low-disk mode a compact weight is mandatory because the large
        # resume checkpoint above is intentionally omitted.
        save_compact_weights = (
            hps.train.if_save_every_weights == True or low_disk_save
        )
        if rank == 0 and save_compact_weights:
            if hasattr(net_g, "module"):
                ckpt = net_g.module.state_dict()
            else:
                ckpt = net_g.state_dict()
            sim_ckpt = od()
            for key in ckpt:
                # if "cfm"not in key:
                #     print(key)
                if key not in no_grad_names:
                    sim_ckpt[key] = ckpt[key].half().cpu()

            if low_disk_save and hps.train.if_save_latest:
                weight_dir = Path(hps.save_weight_dir)
                for old_weight in weight_dir.glob(
                    f"{hps.name}_e*_s*_l{lora_rank}.pth"
                ):
                    try:
                        old_weight.unlink()
                    except OSError:
                        logger.warning(
                            "[low-disk] failed to remove old weight: %s",
                            old_weight,
                        )

            logger.info(
                "saving ckpt %s_e%s:%s"
                % (
                    hps.name,
                    epoch,
                    savee(
                        sim_ckpt,
                        hps.name + "_e%s_s%s_l%s" % (epoch, global_step, lora_rank),
                        epoch,
                        global_step,
                        hps,
                        model_version=hps.model.version,
                        lora_rank=lora_rank,
                    ),
                )
            )

    if rank == 0:
        logger.info("====> Epoch: {}".format(epoch))


if __name__ == "__main__":
    main()
