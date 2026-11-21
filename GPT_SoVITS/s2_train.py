import warnings

warnings.filterwarnings("ignore")
import copy
import json
import os
import sys
from pathlib import Path

import utils

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.grokking import (
    deterministic_holdout_indices,
    gradfilter_ema,
    parameter_l2_norm,
    repeat_indices_to_minimum,
    resolve_grokking_config,
)
from tools.runtime_status import clear_activity, publish_activity

hps = utils.get_hparams(stage=2)
os.environ["CUDA_VISIBLE_DEVICES"] = hps.train.gpu_numbers.replace("-", ",")
import logging

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

logging.getLogger("matplotlib").setLevel(logging.INFO)
logging.getLogger("h5py").setLevel(logging.INFO)
logging.getLogger("numba").setLevel(logging.INFO)
from random import randint

from module import commons
from module.data_utils import (
    DistributedBucketSampler,
    TextAudioSpeakerCollate,
    TextAudioSpeakerLoader,
)
from module.losses import discriminator_loss, feature_loss, generator_loss, kl_loss
from module.mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from module.models import (
    MultiPeriodDiscriminator,
    SynthesizerTrn,
)
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
                task="SoVITS 학습",
                stage="failed",
                state="failed",
                metric="학습 프로세스가 오류 또는 중지로 종료되었습니다.",
            )
        raise
    finally:
        if rank == 0 and succeeded:
            clear_activity(reason="sovits_finished")
        if use_ddp and dist.is_initialized():
            dist.destroy_process_group()


def _split_s2_for_grokking(dataset, grokking):
    names = [item[0] for item in dataset.audiopaths_sid_text]
    train_indices, val_indices = deterministic_holdout_indices(
        names,
        fraction=float(grokking["validation_fraction"]),
        min_validation_samples=int(grokking["min_validation_samples"]),
        seed=int(grokking["seed"]),
    )
    train_indices = repeat_indices_to_minimum(train_indices, minimum=100)

    eval_dataset = copy.copy(dataset)
    eval_dataset.audiopaths_sid_text = [
        dataset.audiopaths_sid_text[index] for index in val_indices
    ]
    eval_dataset.lengths = [dataset.lengths[index] for index in val_indices]
    eval_dataset.val = True

    dataset.audiopaths_sid_text = [
        dataset.audiopaths_sid_text[index] for index in train_indices
    ]
    dataset.lengths = [dataset.lengths[index] for index in train_indices]

    print(
        "[grokking:S2] held-out split "
        f"train={len(dataset)} val={len(eval_dataset)}"
    )
    return eval_dataset


def _move_s2_batch(data, rank, model_version):
    if model_version in {"v2Pro", "v2ProPlus"}:
        (
            ssl,
            ssl_lengths,
            spec,
            spec_lengths,
            y,
            y_lengths,
            text,
            text_lengths,
            sv_emb,
        ) = data
    else:
        (
            ssl,
            ssl_lengths,
            spec,
            spec_lengths,
            y,
            y_lengths,
            text,
            text_lengths,
        ) = data
        sv_emb = None

    target = rank if torch.cuda.is_available() else device
    spec = spec.cuda(rank) if torch.cuda.is_available() else spec.to(target)
    spec_lengths = (
        spec_lengths.cuda(rank)
        if torch.cuda.is_available()
        else spec_lengths.to(target)
    )
    y = y.cuda(rank) if torch.cuda.is_available() else y.to(target)
    y_lengths = (
        y_lengths.cuda(rank)
        if torch.cuda.is_available()
        else y_lengths.to(target)
    )
    ssl = ssl.cuda(rank) if torch.cuda.is_available() else ssl.to(target)
    text = text.cuda(rank) if torch.cuda.is_available() else text.to(target)
    text_lengths = (
        text_lengths.cuda(rank)
        if torch.cuda.is_available()
        else text_lengths.to(target)
    )
    if sv_emb is not None:
        sv_emb = (
            sv_emb.cuda(rank)
            if torch.cuda.is_available()
            else sv_emb.to(target)
        )
    return (
        ssl,
        spec,
        spec_lengths,
        y,
        y_lengths,
        text,
        text_lengths,
        sv_emb,
    )


def evaluate_grokking(
    rank,
    hps,
    generator,
    eval_loader,
    max_batches,
    seed,
):
    """Held-out reconstruction metric for the acoustic model.

    The metric intentionally avoids the discriminator so it measures generator
    generalization rather than GAN game balance.
    """

    model = generator.module if hasattr(generator, "module") else generator
    was_training = model.training
    model.eval()
    losses = []
    rng_devices = [rank] if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=rng_devices, enabled=True):
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed(int(seed))
        with torch.no_grad():
            for batch_idx, data in enumerate(eval_loader):
                if batch_idx >= int(max_batches):
                    break
                (
                    ssl,
                    spec,
                    spec_lengths,
                    _y,
                    _y_lengths,
                    text,
                    text_lengths,
                    sv_emb,
                ) = _move_s2_batch(data, rank, hps.model.version)

                if hps.model.version in {"v2Pro", "v2ProPlus"}:
                    outputs = model(
                        ssl,
                        spec,
                        spec_lengths,
                        text,
                        text_lengths,
                        sv_emb,
                    )
                else:
                    outputs = model(
                        ssl,
                        spec,
                        spec_lengths,
                        text,
                        text_lengths,
                    )

                y_hat = outputs[0]
                ids_slice = outputs[2]
                mel = spec_to_mel_torch(
                    spec,
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax,
                )
                y_mel = commons.slice_segments(
                    mel,
                    ids_slice,
                    hps.train.segment_size // hps.data.hop_length,
                )
                y_hat_mel = mel_spectrogram_torch(
                    y_hat.squeeze(1),
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.hop_length,
                    hps.data.win_length,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax,
                )
                losses.append(
                    float(
                        F.l1_loss(y_mel.float(), y_hat_mel.float())
                        .detach()
                        .cpu()
                        .item()
                    )
                )

    if was_training:
        model.train()
    if not losses:
        return None
    return sum(losses) / len(losses)


def _write_s2_grokking_metric(hps, record):
    path = Path(hps.data.exp_dir) / "grokking_s2_metrics.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _run(rank, n_gpus, hps, use_ddp):
    global global_step
    if rank == 0:
        logger = utils.get_logger(hps.data.exp_dir)
        logger.info(hps)
        # utils.check_git_hash(hps.s2_ckpt_dir)
        writer = SummaryWriter(log_dir=hps.s2_ckpt_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.s2_ckpt_dir, "eval"))

    torch.manual_seed(hps.train.seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

    grokking = resolve_grokking_config(hps.train, "s2")
    train_dataset = TextAudioSpeakerLoader(hps.data, version=hps.model.version)
    eval_dataset = None
    if grokking["enabled"]:
        eval_dataset = _split_s2_for_grokking(train_dataset, grokking)

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
            1100,
            1200,
            1300,
            1400,
            1500,
            1600,
            1700,
            1800,
            1900,
        ],
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True,
    )
    collate_fn = TextAudioSpeakerCollate(version=hps.model.version)
    train_loader = DataLoader(
        train_dataset,
        num_workers=4,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
        batch_sampler=train_sampler,
        persistent_workers=True,
        prefetch_factor=2,
    )
    eval_loader = None
    if grokking["enabled"] and eval_dataset is not None:
        eval_loader = DataLoader(
            eval_dataset,
            num_workers=0,
            shuffle=False,
            batch_size=1,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )

    net_g = (
        SynthesizerTrn(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model,
        ).cuda(rank)
        if torch.cuda.is_available()
        else SynthesizerTrn(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model,
        ).to(device)
    )

    net_d = (
        MultiPeriodDiscriminator(hps.model.use_spectral_norm, version=hps.model.version).cuda(rank)
        if torch.cuda.is_available()
        else MultiPeriodDiscriminator(hps.model.use_spectral_norm, version=hps.model.version).to(device)
    )
    for name, param in net_g.named_parameters():
        if not param.requires_grad:
            print(name, "not requires_grad")

    te_p = list(map(id, net_g.enc_p.text_embedding.parameters()))
    et_p = list(map(id, net_g.enc_p.encoder_text.parameters()))
    mrte_p = list(map(id, net_g.enc_p.mrte.parameters()))
    base_params = filter(
        lambda p: id(p) not in te_p + et_p + mrte_p and p.requires_grad,
        net_g.parameters(),
    )

    # te_p=net_g.enc_p.text_embedding.parameters()
    # et_p=net_g.enc_p.encoder_text.parameters()
    # mrte_p=net_g.enc_p.mrte.parameters()

    s2_weight_decay = (
        float(grokking["s2_weight_decay"])
        if grokking["enabled"]
        else 0.01
    )
    optim_g = torch.optim.AdamW(
        # filter(lambda p: p.requires_grad, net_g.parameters()),###默认所有层lr一致
        [
            {"params": base_params, "lr": hps.train.learning_rate},
            {
                "params": net_g.enc_p.text_embedding.parameters(),
                "lr": hps.train.learning_rate * hps.train.text_low_lr_rate,
            },
            {
                "params": net_g.enc_p.encoder_text.parameters(),
                "lr": hps.train.learning_rate * hps.train.text_low_lr_rate,
            },
            {
                "params": net_g.enc_p.mrte.parameters(),
                "lr": hps.train.learning_rate * hps.train.text_low_lr_rate,
            },
        ],
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
        weight_decay=s2_weight_decay,
    )
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    if use_ddp:
        net_g = DDP(net_g, device_ids=[rank], find_unused_parameters=True)
        net_d = DDP(net_d, device_ids=[rank], find_unused_parameters=True)
    elif not torch.cuda.is_available():
        net_g = net_g.to(device)
        net_d = net_d.to(device)

    resumed_existing_experiment = False
    if not grokking["enabled"]:
        try:  # normal mode keeps the original automatic resume behaviour
            _, _, _, epoch_str = utils.load_checkpoint(
                utils.latest_checkpoint_path(
                    "%s/logs_s2_%s"
                    % (hps.data.exp_dir, hps.model.version),
                    "D_*.pth",
                ),
                net_d,
                optim_d,
            )
            if rank == 0:
                logger.info("loaded D")
            _, _, _, epoch_str = utils.load_checkpoint(
                utils.latest_checkpoint_path(
                    "%s/logs_s2_%s"
                    % (hps.data.exp_dir, hps.model.version),
                    "G_*.pth",
                ),
                net_g,
                optim_g,
            )
            epoch_str += 1
            global_step = (epoch_str - 1) * len(train_loader)
            resumed_existing_experiment = True
        except Exception:
            resumed_existing_experiment = False

    if not resumed_existing_experiment:
        # A grokking probe must not resume a checkpoint that may already have
        # seen the held-out utterances. Generic pretrained weights are fine;
        # the current experiment-specific train/validation split starts fresh.
        epoch_str = 1
        global_step = 0
        if grokking["enabled"] and rank == 0:
            logger.info(
                "[grokking:S2] ignoring existing experiment checkpoints "
                "to prevent held-out leakage"
            )
        if (
            hps.train.pretrained_s2G != ""
            and hps.train.pretrained_s2G is not None
            and os.path.exists(hps.train.pretrained_s2G)
        ):
            if rank == 0:
                logger.info("loaded pretrained %s" % hps.train.pretrained_s2G)
            model_g = net_g.module if hasattr(net_g, "module") else net_g
            print(
                "loaded pretrained %s" % hps.train.pretrained_s2G,
                model_g.load_state_dict(
                    torch.load(
                        hps.train.pretrained_s2G,
                        map_location="cpu",
                        weights_only=False,
                    )["weight"],
                    strict=False,
                ),
            )
        if (
            hps.train.pretrained_s2D != ""
            and hps.train.pretrained_s2D is not None
            and os.path.exists(hps.train.pretrained_s2D)
        ):
            if rank == 0:
                logger.info("loaded pretrained %s" % hps.train.pretrained_s2D)
            model_d = net_d.module if hasattr(net_d, "module") else net_d
            print(
                "loaded pretrained %s" % hps.train.pretrained_s2D,
                model_d.load_state_dict(
                    torch.load(
                        hps.train.pretrained_s2D,
                        map_location="cpu",
                        weights_only=False,
                    )["weight"],
                    strict=False,
                ),
            )

    # scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)
    # scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g,
        gamma=hps.train.lr_decay,
        last_epoch=-1,
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d,
        gamma=hps.train.lr_decay,
        last_epoch=-1,
    )
    for _ in range(epoch_str):
        scheduler_g.step()
        scheduler_d.step()

    scaler = GradScaler(enabled=hps.train.fp16_run)
    grokfast_state = {}
    base_epochs = int(hps.train.epochs)
    effective_epochs = base_epochs
    if grokking["enabled"]:
        post_fit_epochs = int(grokking["s2_post_fit_epochs"])
        plateau_patience = int(grokking["s2_plateau_patience"])
        hard_extension = max(
            post_fit_epochs * 4,
            post_fit_epochs + plateau_patience,
        )
        effective_epochs = base_epochs + hard_extension
        if rank == 0:
            logger.info(
                "[grokking:S2] enabled: base_epochs=%s hard_max=%s "
                "post_fit=%s plateau_patience=%s "
                "generator_weight_decay=%s grokfast=%s",
                base_epochs,
                effective_epochs,
                post_fit_epochs,
                plateau_patience,
                s2_weight_decay,
                grokking["grokfast_enabled"],
            )
    else:
        post_fit_epochs = 0
        plateau_patience = 0

    print("start training from epoch %s" % epoch_str)
    if rank == 0:
        total_steps = effective_epochs * len(train_loader)
        publish_activity(
            task="SoVITS 학습",
            stage="starting",
            progress=min(global_step / total_steps, 1.0)
            if total_steps
            else None,
            step=global_step,
            total_steps=total_steps,
            epoch=epoch_str,
            total_epochs=effective_epochs,
        )
    best_grokking_val = float("inf")
    best_train_mel = float("inf")
    last_significant_train_improvement = int(epoch_str) - 1
    s2_fit_epoch = None
    for epoch in range(epoch_str, effective_epochs + 1):
        if rank == 0:
            epoch_train_mel = train_and_evaluate(
                rank,
                epoch,
                hps,
                [net_g, net_d],
                [optim_g, optim_d],
                [scheduler_g, scheduler_d],
                scaler,
                # [train_loader, eval_loader], logger, [writer, writer_eval])
                [train_loader, eval_loader],
                logger,
                [writer, writer_eval],
                grokking=grokking,
                grokfast_state=grokfast_state,
                effective_epochs=effective_epochs,
            )
        else:
            epoch_train_mel = train_and_evaluate(
                rank,
                epoch,
                hps,
                [net_g, net_d],
                [optim_g, optim_d],
                [scheduler_g, scheduler_d],
                scaler,
                [train_loader, eval_loader],
                None,
                None,
                grokking=grokking,
                grokfast_state=grokfast_state,
                effective_epochs=effective_epochs,
            )
        if grokking["enabled"] and epoch_train_mel is not None:
            relative_delta = float(
                grokking["s2_plateau_relative_delta"]
            )
            threshold = best_train_mel * (1.0 - relative_delta)
            if epoch_train_mel < threshold:
                best_train_mel = epoch_train_mel
                last_significant_train_improvement = epoch
                s2_fit_epoch = None
            elif (
                s2_fit_epoch is None
                and epoch - last_significant_train_improvement
                >= plateau_patience
            ):
                s2_fit_epoch = epoch

        scheduler_g.step()
        scheduler_d.step()

        if use_ddp:
            dist.barrier()
        if rank == 0 and grokking["enabled"] and eval_loader is not None:
            val_mel = evaluate_grokking(
                rank,
                hps,
                net_g,
                eval_loader,
                grokking["max_validation_batches"],
                seed=int(grokking["seed"]),
            )
            model_g = net_g.module if hasattr(net_g, "module") else net_g
            record = {
                "epoch": epoch,
                "global_step": global_step,
                "train_mel_l1": epoch_train_mel,
                "val_mel_l1": val_mel,
                "fit_epoch": s2_fit_epoch,
                "post_fit_age": (
                    None
                    if s2_fit_epoch is None
                    else epoch - s2_fit_epoch
                ),
                "parameter_l2": parameter_l2_norm(model_g),
                "generator_weight_decay": s2_weight_decay,
                "grokfast": bool(grokking["grokfast_enabled"]),
            }
            _write_s2_grokking_metric(hps, record)
            logger.info("[grokking:S2] %s", record)
            if val_mel is not None and val_mel < best_grokking_val:
                best_grokking_val = val_mel
                ckpt = model_g.state_dict()
                logger.info(
                    "saving grokking best: %s",
                    savee(
                        ckpt,
                        hps.name + "_grok_best",
                        epoch,
                        global_step,
                        hps,
                        model_version=(
                            None
                            if hps.model.version not in {"v2Pro", "v2ProPlus"}
                            else hps.model.version
                        ),
                    ),
                )
        if use_ddp:
            dist.barrier()

        if (
            grokking["enabled"]
            and s2_fit_epoch is not None
            and epoch >= base_epochs
            and epoch - s2_fit_epoch >= post_fit_epochs
        ):
            if rank == 0:
                logger.info(
                    "[grokking:S2] post-fit window complete: "
                    "fit_epoch=%s stop_epoch=%s",
                    s2_fit_epoch,
                    epoch,
                )
            break
    print("training done")


def train_and_evaluate(
    rank,
    epoch,
    hps,
    nets,
    optims,
    schedulers,
    scaler,
    loaders,
    logger,
    writers,
    grokking=None,
    grokfast_state=None,
    effective_epochs=None,
):
    net_g, net_d = nets
    optim_g, optim_d = optims
    # scheduler_g, scheduler_d = schedulers
    train_loader, eval_loader = loaders
    if writers is not None:
        writer, writer_eval = writers

    train_loader.batch_sampler.set_epoch(epoch)
    global global_step

    net_g.train()
    net_d.train()
    epoch_mel_sum = 0.0
    epoch_mel_count = 0
    for batch_idx, data in enumerate(tqdm(train_loader)):
        if hps.model.version in {"v2Pro", "v2ProPlus"}:
            ssl, ssl_lengths, spec, spec_lengths, y, y_lengths, text, text_lengths, sv_emb = data
        else:
            ssl, ssl_lengths, spec, spec_lengths, y, y_lengths, text, text_lengths = data
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
            y, y_lengths = (
                y.cuda(
                    rank,
                    non_blocking=True,
                ),
                y_lengths.cuda(
                    rank,
                    non_blocking=True,
                ),
            )
            ssl = ssl.cuda(rank, non_blocking=True)
            ssl.requires_grad = False
            # ssl_lengths = ssl_lengths.cuda(rank, non_blocking=True)
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
            if hps.model.version in {"v2Pro", "v2ProPlus"}:
                sv_emb = sv_emb.cuda(rank, non_blocking=True)
        else:
            spec, spec_lengths = spec.to(device), spec_lengths.to(device)
            y, y_lengths = y.to(device), y_lengths.to(device)
            ssl = ssl.to(device)
            ssl.requires_grad = False
            # ssl_lengths = ssl_lengths.cuda(rank, non_blocking=True)
            text, text_lengths = text.to(device), text_lengths.to(device)
            if hps.model.version in {"v2Pro", "v2ProPlus"}:
                sv_emb = sv_emb.to(device)
        with autocast(enabled=hps.train.fp16_run):
            if hps.model.version in {"v2Pro", "v2ProPlus"}:
                (y_hat, kl_ssl, ids_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q), stats_ssl) = net_g(
                    ssl, spec, spec_lengths, text, text_lengths, sv_emb
                )
            else:
                (
                    y_hat,
                    kl_ssl,
                    ids_slice,
                    x_mask,
                    z_mask,
                    (z, z_p, m_p, logs_p, m_q, logs_q),
                    stats_ssl,
                ) = net_g(ssl, spec, spec_lengths, text, text_lengths)

            mel = spec_to_mel_torch(
                spec,
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )
            y_mel = commons.slice_segments(mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1),
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )

            y = commons.slice_segments(y, ids_slice * hps.data.hop_length, hps.train.segment_size)  # slice

            # Discriminator
            y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
            with autocast(enabled=False):
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                    y_d_hat_r,
                    y_d_hat_g,
                )
                loss_disc_all = loss_disc
        optim_d.zero_grad()
        scaler.scale(loss_disc_all).backward()
        scaler.unscale_(optim_d)
        grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
        scaler.step(optim_d)

        with autocast(enabled=hps.train.fp16_run):
            # Generator
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
            with autocast(enabled=False):
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                epoch_mel_sum += (
                    float(loss_mel.detach().float().item())
                    / max(float(hps.train.c_mel), 1.0e-12)
                )
                epoch_mel_count += 1
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl

                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, losses_gen = generator_loss(y_d_hat_g)
                loss_gen_all = loss_gen + loss_fm + loss_mel + kl_ssl * 1 + loss_kl

        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        if (
            grokking
            and grokking["enabled"]
            and grokking["grokfast_enabled"]
        ):
            model_g = net_g.module if hasattr(net_g, "module") else net_g
            gradfilter_ema(
                model_g,
                grokfast_state,
                alpha=float(grokking["grokfast_alpha"]),
                lamb=float(grokking["grokfast_lambda"]),
            )
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
        scaler.step(optim_g)
        scaler.update()

        if rank == 0:
            status_interval = max(1, len(train_loader) // 100)
            if (
                batch_idx % status_interval == 0
                or batch_idx + 1 == len(train_loader)
            ):
                total_epochs = int(effective_epochs or hps.train.epochs)
                total_steps = total_epochs * len(train_loader)
                completed_step = global_step + 1
                publish_activity(
                    task="SoVITS 학습",
                    stage="training",
                    progress=min(completed_step / total_steps, 1.0)
                    if total_steps
                    else None,
                    step=completed_step,
                    total_steps=total_steps,
                    epoch=epoch,
                    total_epochs=int(effective_epochs or hps.train.epochs),
                    loss=float(loss_gen_all.detach().float().item()),
                    metric=(
                        f"mel {float(loss_mel.detach().float().item()):.4f}"
                    ),
                )

            if global_step % hps.train.log_interval == 0:
                lr = optim_g.param_groups[0]["lr"]
                losses = [loss_disc, loss_gen, loss_fm, loss_mel, kl_ssl, loss_kl]
                logger.info(
                    "Train Epoch: {} [{:.0f}%]".format(
                        epoch,
                        100.0 * batch_idx / len(train_loader),
                    )
                )
                logger.info([x.item() for x in losses] + [global_step, lr])

                scalar_dict = {
                    "loss/g/total": loss_gen_all,
                    "loss/d/total": loss_disc_all,
                    "learning_rate": lr,
                    "grad_norm_d": grad_norm_d,
                    "grad_norm_g": grad_norm_g,
                }
                scalar_dict.update(
                    {
                        "loss/g/fm": loss_fm,
                        "loss/g/mel": loss_mel,
                        "loss/g/kl_ssl": kl_ssl,
                        "loss/g/kl": loss_kl,
                    }
                )

                # scalar_dict.update({"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)})
                # scalar_dict.update({"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)})
                # scalar_dict.update({"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)})
                image_dict = None
                try:  ###Some people installed the wrong version of matplotlib.
                    image_dict = {
                        "slice/mel_org": utils.plot_spectrogram_to_numpy(
                            y_mel[0].data.cpu().numpy(),
                        ),
                        "slice/mel_gen": utils.plot_spectrogram_to_numpy(
                            y_hat_mel[0].data.cpu().numpy(),
                        ),
                        "all/mel": utils.plot_spectrogram_to_numpy(
                            mel[0].data.cpu().numpy(),
                        ),
                        "all/stats_ssl": utils.plot_spectrogram_to_numpy(
                            stats_ssl[0].data.cpu().numpy(),
                        ),
                    }
                except:
                    pass
                if image_dict:
                    utils.summarize(
                        writer=writer,
                        global_step=global_step,
                        images=image_dict,
                        scalars=scalar_dict,
                    )
                else:
                    utils.summarize(
                        writer=writer,
                        global_step=global_step,
                        scalars=scalar_dict,
                    )
        global_step += 1
    if epoch % hps.train.save_every_epoch == 0 and rank == 0:
        if hps.train.if_save_latest == 0:
            utils.save_checkpoint(
                net_g,
                optim_g,
                hps.train.learning_rate,
                epoch,
                os.path.join(
                    "%s/logs_s2_%s" % (hps.data.exp_dir, hps.model.version),
                    "G_{}.pth".format(global_step),
                ),
            )
            utils.save_checkpoint(
                net_d,
                optim_d,
                hps.train.learning_rate,
                epoch,
                os.path.join(
                    "%s/logs_s2_%s" % (hps.data.exp_dir, hps.model.version),
                    "D_{}.pth".format(global_step),
                ),
            )
        else:
            utils.save_checkpoint(
                net_g,
                optim_g,
                hps.train.learning_rate,
                epoch,
                os.path.join(
                    "%s/logs_s2_%s" % (hps.data.exp_dir, hps.model.version),
                    "G_{}.pth".format(233333333333),
                ),
            )
            utils.save_checkpoint(
                net_d,
                optim_d,
                hps.train.learning_rate,
                epoch,
                os.path.join(
                    "%s/logs_s2_%s" % (hps.data.exp_dir, hps.model.version),
                    "D_{}.pth".format(233333333333),
                ),
            )
        if rank == 0 and hps.train.if_save_every_weights == True:
            if hasattr(net_g, "module"):
                ckpt = net_g.module.state_dict()
            else:
                ckpt = net_g.state_dict()
            logger.info(
                "saving ckpt %s_e%s:%s"
                % (
                    hps.name,
                    epoch,
                    savee(
                        ckpt,
                        hps.name + "_e%s_s%s" % (epoch, global_step),
                        epoch,
                        global_step,
                        hps,
                        model_version=None if hps.model.version not in {"v2Pro", "v2ProPlus"} else hps.model.version,
                    ),
                )
            )

    stats_device = (
        torch.device(f"cuda:{rank}")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    mel_stats = torch.tensor(
        [epoch_mel_sum, float(epoch_mel_count)],
        device=stats_device,
        dtype=torch.float64,
    )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(mel_stats, op=dist.ReduceOp.SUM)
    epoch_train_mel = (
        float((mel_stats[0] / mel_stats[1]).item())
        if mel_stats[1].item() > 0
        else None
    )

    if rank == 0:
        logger.info(
            "====> Epoch: %s train_mel_l1=%s",
            epoch,
            epoch_train_mel,
        )
    return epoch_train_mel


def evaluate(hps, generator, eval_loader, writer_eval):
    generator.eval()
    generator_to_infer = generator.module if hasattr(generator, "module") else generator
    image_dict = {}
    audio_dict = {}
    print("Evaluating ...")
    with torch.no_grad():
        for batch_idx, (
            ssl,
            ssl_lengths,
            spec,
            spec_lengths,
            y,
            y_lengths,
            text,
            text_lengths,
        ) in enumerate(eval_loader):
            print(111)
            if torch.cuda.is_available():
                spec, spec_lengths = spec.cuda(), spec_lengths.cuda()
                y, y_lengths = y.cuda(), y_lengths.cuda()
                ssl = ssl.cuda()
                text, text_lengths = text.cuda(), text_lengths.cuda()
            else:
                spec, spec_lengths = spec.to(device), spec_lengths.to(device)
                y, y_lengths = y.to(device), y_lengths.to(device)
                ssl = ssl.to(device)
                text, text_lengths = text.to(device), text_lengths.to(device)
            for test in [0, 1]:
                y_hat, mask, *_ = generator_to_infer.infer(
                    ssl,
                    spec,
                    spec_lengths,
                    text,
                    text_lengths,
                    test=test,
                )
                y_hat_lengths = mask.sum([1, 2]).long() * hps.data.hop_length

                mel = spec_to_mel_torch(
                    spec,
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax,
                )
                y_hat_mel = mel_spectrogram_torch(
                    y_hat.squeeze(1).float(),
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.hop_length,
                    hps.data.win_length,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax,
                )
                image_dict.update(
                    {
                        f"gen/mel_{batch_idx}_{test}": utils.plot_spectrogram_to_numpy(
                            y_hat_mel[0].cpu().numpy(),
                        ),
                    }
                )
                audio_dict.update(
                    {
                        f"gen/audio_{batch_idx}_{test}": y_hat[0, :, : y_hat_lengths[0]],
                    },
                )
                image_dict.update(
                    {
                        f"gt/mel_{batch_idx}": utils.plot_spectrogram_to_numpy(mel[0].cpu().numpy()),
                    },
                )
                audio_dict.update({f"gt/audio_{batch_idx}": y[0, :, : y_lengths[0]]})

        # y_hat, mask, *_ = generator.module.infer(ssl, spec_lengths, speakers, y=None)
        # audio_dict.update({
        #     f"gen/audio_{batch_idx}_style_pred": y_hat[0, :, :]
        # })

    utils.summarize(
        writer=writer_eval,
        global_step=global_step,
        images=image_dict,
        audios=audio_dict,
        audio_sampling_rate=hps.data.sampling_rate,
    )
    generator.train()


if __name__ == "__main__":
    main()
