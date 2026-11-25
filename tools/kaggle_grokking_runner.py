"""Headless Kaggle S1/S2 ablation. Invoked from webui.py before building Gradio.

All data stays in Kaggle. Only configs, metrics and text logs are bundled.
Each run gets its own training output and never resumes a checkpoint.
"""
from __future__ import annotations

import json
import math
import statistics
import os
import signal
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = {
    "stage": "s1",
    "seed": 1234,
    "batch_size": 2,
    "experiment": "s1_weight_decay_long_v2",
    "base_epochs": 120,
    "fixed_epochs": 120,
    "post_fit_epochs": 100,
    "validation_fraction": 0.10,
    "min_validation_samples": 8,
    "learning_rate": 1e-4,
    "weight_decay": 0.01,
    "grokfast_alpha": 0.98,
    "grokfast_lambda": 0.5,
    "max_training_seconds": 5400,
    "bert_model": "google-bert/bert-base-multilingual-cased",
    "arms": [
        {"id": "A_wd001", "grokfast": False, "weight_decay": 0.01},
        {"id": "B_wd1", "grokfast": False, "weight_decay": 1.0},
    ],
}


S2_SETTINGS = {
    **SETTINGS,
    "stage": "s2",
    "experiment": "s2_v5dev_cfm_grokfast_long_v2",
    "batch_size": 1,
    "base_epochs": 30,
    "fixed_epochs": 30,
    "post_fit_epochs": 8,
    "lora_rank": 16,
    "max_validation_batches": 32,
    "arms": [
        {"id": "A_cfm_off", "grokfast": False, "weight_decay": 0.01},
        {"id": "B_cfm_on", "grokfast": True, "weight_decay": 0.01},
    ],
}
# Quality-first controlled comparison: same split, LoRA rank and inference;
# no experimental gradient amplification. Vary only the acoustic learning rate.
QUALITY_SETTINGS = {
    **S2_SETTINGS,
    "experiment": "v5dev_quality_lr_v1",
    "base_epochs": 8,
    "fixed_epochs": 8,
    "arms": [
        {"id": "A_lr100", "grokfast": False, "weight_decay": 0.01, "learning_rate": 1e-4},
        {"id": "B_lr030", "grokfast": False, "weight_decay": 0.01, "learning_rate": 3e-5},
    ],
}
# Independent lanes share preprocessing, never share optimizer or writable weights.
SESSION_SETTINGS = {
    **QUALITY_SETTINGS,
    "experiment": "quality_parallel_session_v1",
    "fixed_epochs": 8,
    "base_epochs": 8,
    "max_training_seconds": 7200,
    "arms": [
        {"id": "S1_v_o", "gpu": 0, "stage": "s1", "grokfast": False, "adapter": "lora_v_o"},
        {"id": "S2_rank16", "gpu": 1, "stage": "s2", "grokfast": False, "lora_rank": 16},
        {"id": "S1_qkv_o", "gpu": 0, "stage": "s1", "grokfast": False, "adapter": "lora_qkv_o"},
        {"id": "S2_rank32", "gpu": 1, "stage": "s2", "grokfast": False, "lora_rank": 32},
    ],
}
PROFILES = {"s1": SETTINGS, "s2": S2_SETTINGS, "s2_quality": QUALITY_SETTINGS,
            "quality_session": SESSION_SETTINGS,
            "quality_reuse": {**SESSION_SETTINGS, "experiment": "v5dev_multi_reference_reuse", "fixed_epochs": 0, "base_epochs": 0},
            "quality_mix": {**SESSION_SETTINGS, "experiment": "v5dev_mix_and_cfg", "fixed_epochs": 0, "base_epochs": 0},
            "quality_soft": {**SESSION_SETTINGS, "experiment": "v5dev_soft_semantic_and_mix_refinement", "fixed_epochs": 0, "base_epochs": 0},
            "quality_general": {**SESSION_SETTINGS, "experiment": "v5dev_multiseed_prompt_validation", "fixed_epochs": 0, "base_epochs": 0},
            "s2_listening": {**QUALITY_SETTINGS, "experiment": "v5dev_reuse_listening", "fixed_epochs": 0, "base_epochs": 0}}
DEFAULT_PROFILE = "s2"


def _write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _read_metrics(path):
    rows = []
    for file in sorted(path.glob("grokking_*metrics*.jsonl")):
        for line in file.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                # Preserve malformed/non-finite values in the raw log, not the summary.
                if isinstance(row, dict):
                    rows.append(json.loads(json.dumps(row, allow_nan=False)))
            except (ValueError, TypeError):
                continue
    return rows


def _summarize(rows):
    metric = "val_cfm" if rows and "val_cfm" in rows[0] else "val_loss"
    valid = [r for r in rows if isinstance(r.get(metric), (int, float))]
    return {
        "epochs_recorded": len(rows),
        "last": rows[-1] if rows else None,
        "best": min(valid, key=lambda r: r[metric]) if valid else None,
        "candidate_epochs": [r["epoch"] for r in rows if r.get("delayed_improvement") or r.get("delayed_improvement_candidate")],
    }


def _collect_baseline(out, arm_report, arm_state, stage="s1"):
    path = out / f"grokking_{stage}_baseline.json"
    if path.is_file():
        try:
            baseline = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        shutil.copyfile(path, arm_report / path.name)
        arm_state["pretrained_baseline"] = baseline


KST = timezone(timedelta(hours=9), "KST")


def _remaining_seconds(rows, total_epochs, elapsed_in_epoch=0.0, observed_durations=()):
    """Estimate from recent measured full epochs (training + validation)."""
    remaining = max(0, total_epochs - len(rows))
    if remaining == 0:
        return 0.0
    measured = [r.get("epoch_seconds") for r in rows]
    durations = [float(v) for v in measured if isinstance(v, (int, float))
                 and math.isfinite(v) and v > 0]
    if not durations:
        durations = [float(v) for v in observed_durations
                     if isinstance(v, (int, float)) and math.isfinite(v) and v > 0]
    if not durations:
        return None
    typical = statistics.median(durations[-5:])
    # Do not claim zero remaining for an epoch that is still running.
    current = max(typical * 0.1, typical - max(0.0, elapsed_in_epoch))
    return (remaining - 1) * typical + current


def _eta_text(seconds, wall_now, budget_seconds):
    cutoff = max(0.0, budget_seconds)
    if seconds is None:
        return "ETA 계산 중 (완료 Epoch 대기)"
    limited = seconds > cutoff
    effective = min(seconds, cutoff)
    minutes = math.ceil(effective / 60)
    hours, minutes = divmod(minutes, 60)
    duration = f"{hours}h {minutes:02d}m" if hours else f"{minutes}m"
    finish = datetime.fromtimestamp(wall_now + effective, KST).strftime("%m/%d %H:%M KST")
    if limited:
        return f"시간 제한 종료 예상 {finish} (남은 {duration}; 목표 Epoch 미도달 예상)"
    return f"완료 예상 {finish} (남은 약 {duration})"


def _progress_line(arm_id, rows, process, arm_report, settings, gpu_status, eta=None):
    try:
        status = json.loads((arm_report / "status.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        status = {}
    if process.poll() is not None:
        return f"{arm_id}: completed epochs={len(rows)} exit={process.poll()}"
    epoch = status.get("epoch", len(rows) + 1)
    stage = status.get("stage", "initializing")
    batch = status.get("batch")
    batches = status.get("total_batches")
    detail = f" batch {batch}/{batches}" if batch is not None and batches else ""
    gpu = f" | {gpu_status}" if gpu_status else ""
    val = rows[-1].get("val_cfm", rows[-1].get("val_loss")) if rows else None
    metric = f" | last val={val:.6f}" if isinstance(val, (float, int)) else ""
    estimate = f" | {eta}" if eta else ""
    return f"{arm_id}: epoch {epoch}/{settings['fixed_epochs']} [{stage}]{detail}{metric}{gpu}{estimate}"


def _gpu_status():
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"], text=True, timeout=5,
            stderr=subprocess.DEVNULL,
        )
        result = {}
        for line in output.splitlines():
            index, util, used, total = [v.strip() for v in line.split(",")]
            result[int(index)] = f"GPU {util}% | VRAM {used}/{total} MiB"
        return result
    except (OSError, subprocess.SubprocessError, ValueError):
        return {}


def _stop(process):
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


MAX_ARCHIVE_LOG_BYTES = 8 * 1024 * 1024
DISK_RESERVE_BYTES = 2 * 1024**3


def _bundle(report, archive):
    """Keep complete logs in tmp, publish bounded diagnostics atomically."""
    allowed = {".json", ".jsonl", ".yaml", ".log", ".txt"}
    archive.parent.mkdir(parents=True, exist_ok=True)
    staging = archive.with_name("." + archive.name + ".part")
    try:
        with zipfile.ZipFile(staging, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for path in sorted(report.rglob("*")):
                if not path.is_file() or path.is_symlink() or path.suffix not in allowed:
                    continue
                name = path.relative_to(report).as_posix()
                size = path.stat().st_size
                if path.suffix in {".log", ".txt"} and size > MAX_ARCHIVE_LOG_BYTES:
                    half = MAX_ARCHIVE_LOG_BYTES // 2
                    with path.open("rb") as handle:
                        head = handle.read(half)
                        handle.seek(max(half, size - half))
                        tail = handle.read(half)
                    notice = f"\n[archive truncated: original {size} bytes; full log in tmp]\n".encode()
                    z.writestr(name, head + notice + tail)
                else:
                    z.write(path, name)
        os.replace(staging, archive)
    finally:
        staging.unlink(missing_ok=True)


def _disk_state():
    return {name: {"free_bytes": shutil.disk_usage(path).free,
                   "total_bytes": shutil.disk_usage(path).total}
            for name, path in (("tmp", "/kaggle/tmp"), ("working", "/kaggle/working"))}


def _require_tmp_space(state):
    state["disk"] = _disk_state()
    if state["disk"]["tmp"]["free_bytes"] < DISK_RESERVE_BYTES:
        raise RuntimeError("Kaggle tmp free space below 2 GiB reserve; stopping safely")


def _arm_env(arm, gpu, status_path, settings=SETTINGS):
    env = os.environ.copy()
    # Remove inherited experiment overrides; the manifest is the authority.
    for key in list(env):
        if key.startswith(("GPT_SOVITS_GROKKING", "GPT_SOVITS_GROKFAST")):
            del env[key]
    env.update({
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "_CUDA_VISIBLE_DEVICES": str(gpu),
        "GPT_SOVITS_GROKKING": "1",
        "GPT_SOVITS_GROKKING_S1": "1" if settings["stage"] == "s1" else "0",
        "GPT_SOVITS_GROKKING_S2": "1" if settings["stage"] == "s2" else "0",
        "GPT_SOVITS_GROKFAST": "1" if arm["grokfast"] else "0",
        "GPT_SOVITS_GROKFAST_ALPHA": str(settings["grokfast_alpha"]),
        "GPT_SOVITS_GROKFAST_LAMBDA": str(settings["grokfast_lambda"]),
        "GPT_SOVITS_GROKKING_SEED": str(settings["seed"]),
        "GPT_SOVITS_GROKKING_VALIDATION_FRACTION": str(settings["validation_fraction"]),
        "GPT_SOVITS_GROKKING_MIN_VALIDATION_SAMPLES": str(settings["min_validation_samples"]),
        "GPT_SOVITS_GROKKING_S1_LR": str(settings["learning_rate"]),
        "GPT_SOVITS_GROKKING_S1_WEIGHT_DECAY": str(arm.get("weight_decay", settings["weight_decay"])),
        "GPT_SOVITS_GROKKING_S1_FIXED_EPOCHS": str(settings["fixed_epochs"]),
        "GPT_SOVITS_GROKKING_S2_FIXED_EPOCHS": str(settings["fixed_epochs"]),
        "GPT_SOVITS_GROKKING_S2_WEIGHT_DECAY": str(arm.get("weight_decay", settings["weight_decay"])),
        "GPT_SOVITS_GROKKING_S2_POST_FIT_EPOCHS": str(settings["post_fit_epochs"]),
        "GPT_SOVITS_GROKKING_MAX_VALIDATION_BATCHES": str(settings.get("max_validation_batches", 32)),
        "GPT_SOVITS_GROKKING_S1_POST_FIT_EPOCHS": str(settings["post_fit_epochs"]),
        "GPT_SOVITS_GROKKING_S1_KEEP_DPO": "0",
        "GPT_SOVITS_GROKKING_S1_OPTIMIZER": "adamw",
        "GPT_SOVITS_LOW_DISK_SAVE": "1",
        "GPT_SOVITS_RUNTIME_STATUS_PATH": str(status_path),
        "PYTHONUNBUFFERED": "1",
        "hz": "25hz",
        "version": "v5dev",
    })
    return env


def _config(preprocessed, output, weights, arm, pretrained, text_features, settings=SETTINGS):
    import yaml
    data = yaml.safe_load((ROOT / "GPT_SoVITS/configs/s1longer-v2.yaml").read_text(encoding="utf-8"))
    data["train"].update({
        "seed": settings["seed"], "epochs": settings["base_epochs"],
        "batch_size": settings["batch_size"], "precision": "32-true",
        "if_dpo": False, "s1_adapter": arm.get("adapter", "full"),
        "s1_lora_rank": 8, "s1_lora_alpha": 16,
        "save_every_n_epoch": 100000,
        "if_save_every_weights": False, "if_save_latest": True,
        "half_weights_save_dir": str(weights), "exp_name": arm["id"],
    })
    data["data"]["num_workers"] = 2
    data.update({
        "pretrained_s1": str(pretrained),
        "text_features": text_features,
        "train_semantic_path": str(preprocessed / "6-name2semantic.tsv"),
        "train_phoneme_path": str(preprocessed / "2-name2text.txt"),
        "output_dir": str(output),
    })
    return data


def _s2_config(preprocessed, output, weights, arm, pretrained, gpu, settings=S2_SETTINGS):
    data = json.loads((ROOT / "GPT_SoVITS/configs/s2.json").read_text(encoding="utf-8"))
    # Separate writable experiment paths; share only immutable preprocessed inputs.
    for name in ("2-name2text.txt", "4-cnhubert", "5-wav32k"):
        source = preprocessed / name
        if not source.exists():
            raise RuntimeError(f"Missing S2 input: {source}")
        (output / name).symlink_to(source, target_is_directory=source.is_dir())
    data["train"].update({
        "seed": settings["seed"], "epochs": settings["base_epochs"],
        "batch_size": settings["batch_size"], "learning_rate": arm.get("learning_rate", settings["learning_rate"]),
        "fp16_run": False, "grad_ckpt": True, "lora_rank": arm.get("lora_rank", settings["lora_rank"]),
        "pretrained_s2G": str(pretrained), "pretrained_s2D": "",
        "gpu_numbers": str(gpu), "if_save_latest": True,
        "if_save_every_weights": False, "save_every_epoch": 100000,
    })
    data["model"]["version"] = "v5dev"
    data["data"]["exp_dir"] = str(output)
    data.update({"s2_ckpt_dir": str(output), "save_weight_dir": str(weights),
                 "name": arm["id"], "version": "v5dev"})
    return data


def _launchable_jobs(pending, busy):
    """Return at most one queued job per currently idle GPU, in lane order."""
    reserved = set(busy)
    ready = []
    for gpu, arm in pending:
        if gpu not in reserved:
            ready.append((gpu, arm))
            reserved.add(gpu)
    return ready


def _dispatch_jobs(pending, busy, *, work_stealing=False):
    """Prefer the declared lane, then use any idle compatible T4.

    Return (original queue entry, actual device). Never assign two jobs to
    the same device, and remove the original entry after successful launch.
    """
    ready = _launchable_jobs(pending, busy)
    assigned = [(job, job[0]) for job in ready]
    if work_stealing:
        idle = sorted({0, 1} - set(busy) - {gpu for _, gpu in assigned})
        waiting = [job for job in pending if job not in ready]
        assigned.extend(zip(waiting, idle))
    return assigned


def run(ui):
    import torch
    import yaml
    profile = os.environ.get("GPT_SOVITS_RESEARCH_PROFILE", DEFAULT_PROFILE)
    if profile not in PROFILES:
        raise ValueError(f"Unknown research profile: {profile}")
    settings = dict(PROFILES[profile])
    # Explicit remote smoke overrides; ordinary notebook profiles are unchanged.
    epochs = int(os.environ.get("GPT_SOVITS_RESEARCH_EPOCHS", settings["fixed_epochs"]))
    seconds = int(os.environ.get("GPT_SOVITS_RESEARCH_MAX_SECONDS", settings["max_training_seconds"]))
    if not ((epochs == 0 if profile in ("s2_listening", "quality_reuse", "quality_mix", "quality_soft", "quality_general") else 1 <= epochs <= 120) and 60 <= seconds <= 14400):
        raise ValueError("Research budget out of range")
    settings.update(base_epochs=epochs, fixed_epochs=epochs, max_training_seconds=seconds)
    run_id = os.environ.get("GPT_SOVITS_RESEARCH_RUN_ID") or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    work = Path("/kaggle/tmp/grokking_runs") / run_id
    work.mkdir(parents=True, exist_ok=False)
    shared_report = os.environ.get("GPT_SOVITS_RESEARCH_REPORT_ROOT")
    bundle_root = Path(shared_report) if shared_report else work / "report"
    report = bundle_root / "training" if shared_report else bundle_root
    report.mkdir(parents=True, exist_ok=False)
    archive = Path("/kaggle/working/grokking_result.zip")
    exp = "grok_preprocess_" + run_id
    processes = {}
    arm_settings = {}
    state = {"run_id": run_id, "status": "starting", "settings": settings, "arms": {}}
    _write_json(report / "summary.json", state)
    try:
        _require_tmp_space(state)
        state["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        state["torch_version"] = torch.__version__
        state["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        if torch.cuda.device_count() != 2:
            raise RuntimeError("This paired experiment requires exactly 2 visible GPUs. Select GPU T4 x2.")
        source = os.environ.get("GPT_SOVITS_DEFAULT_TRAINING_LIST", "")
        wav_root = os.environ["GPT_SOVITS_DEFAULT_DATAS_DIR"]
        s1 = (ROOT / ui.pretrained_gpt_name["v5dev"]).resolve()
        s2 = (ROOT / ui.pretrained_sovits_name["v5dev"]).resolve()
        if not s1.is_file() or not s2.is_file():
            raise RuntimeError("Generic V5dev pretrained weights missing")
        state["status"] = "preprocessing"
        _write_json(report / "summary.json", state)
        gpu_string = "0-1"
        completed = False
        with (report / "preprocess.log").open("w", encoding="utf-8", buffering=1) as log:
            # Reuse upstream canonical a/b/c, including automatic .list generation.
            # Emotion labeling/Controller are not involved in this controlled S1 probe.
            for result in ui.open1abc(
                "v5dev", source, wav_root, exp, gpu_string, gpu_string, gpu_string,
                ui.default_bert_path, True, settings["bert_model"],
                ui.default_cnhubert_path, str(s2),
            ):
                status = str(result[0]) if result else ""
                print(status, flush=True)
                log.write(status + "\n")
                completed = status == ui.process_info(ui.process_name_1abc, "finish")
        if not completed:
            raise RuntimeError("Canonical a/b/c preprocessing did not finish")
        preprocessed = (ROOT / ui.exp_root / exp).resolve()
        stages = ("gpt", "sovits") if profile == "quality_session" else ("gpt" if profile == "s1" else "sovits",)
        for stage in stages:
            artifact_error = ui._training_artifact_error(preprocessed, stage, "v5dev")
            if artifact_error:
                raise RuntimeError(artifact_error)
        feature_path = preprocessed / "text_feature_config.json"
        features = json.loads(feature_path.read_text(encoding="utf-8"))
        if not features.get("multilingual_bert"):
            raise RuntimeError("BERT was requested but feature marker does not confirm it")
        _require_tmp_space(state)
        if profile in ("s2_quality", "s2_listening"):
            from tools.kaggle_quality_listening import choose_reference
            # Fail before spending GPU epochs if the listening prompt is unavailable.
            state["reference_counts"] = {}
            choose_reference(preprocessed, state["reference_counts"])
        if profile in ("quality_reuse", "quality_mix", "quality_soft", "quality_general"):
            import re
            source_kernel = os.environ.get("GPT_SOVITS_RESEARCH_SOURCE_KERNEL", "")
            if not re.fullmatch(r"pymood1/gpt-sovits-research-[0-9]+", source_kernel):
                raise ValueError("Invalid private source kernel")
            slug = source_kernel.split("/")[-1]
            required = ("S1_v_o-grok-best.ckpt", "S1_qkv_o-grok-best.ckpt",
                        "S2_rank16_grok_cfm_best.pth", "S2_rank32_grok_cfm_best.pth")
            candidates = [p.parent for p in Path("/kaggle/input").rglob(required[0])
                          if slug in p.parts and p.parent.name == "private-candidates"
                          and all((p.parent / name).is_file() for name in required)]
            if len(candidates) != 1:
                raise FileNotFoundError("Unique four prior private candidates not mounted")
            for name in required:
                (work / name).symlink_to(candidates[0] / name)
            for name in ("2-name2text.txt", "5-wav32k"):
                (work / name).symlink_to(preprocessed / name)
            state.update(status="completed", reused_checkpoints=4, training_epochs=0)
            return 0
        if profile == "s2_listening":
            # Mount prior private Kaggle output; no optimizer/trainer is started.
            import re
            source_kernel = os.environ.get("GPT_SOVITS_RESEARCH_SOURCE_KERNEL", "")
            if not re.fullmatch(r"pymood1/gpt-sovits-research-[0-9]+", source_kernel):
                raise ValueError("Invalid private source kernel")
            input_root = Path("/kaggle/input")
            slug = source_kernel.split("/")[-1]
            candidates = [p for p in input_root.rglob("A_lr100.pth")
                          if slug in p.parts and p.parent.name == "private-listening"
                          and (p.parent / "B_lr030.pth").is_file()]
            if len(candidates) != 1:
                raise FileNotFoundError("Unique prior private checkpoints not mounted")
            for arm in settings["arms"]:
                arm_id = arm["id"]
                out = work / arm_id
                out.mkdir()
                weights = out / "weights"
                weights.mkdir()
                for name in ("2-name2text.txt", "5-wav32k"):
                    (out / name).symlink_to(preprocessed / name)
                (weights / (arm_id + "_grok_cfm_best.pth")).symlink_to(candidates[0].parent / (arm_id + ".pth"))
            state["status"] = "completed"
            state["reused_checkpoints"] = 2
            state["training_epochs"] = 0
            return 0
        state["status"] = "training"
        pending = [(arm.get("gpu", index), arm) for index, arm in enumerate(settings["arms"])]
        observations = {}
        def launch(gpu, arm):
            arm_id = arm["id"]
            stage = arm.get("stage", settings["stage"])
            local = {**settings, "stage": stage}
            if profile == "quality_session" and stage == "s1":
                local["batch_size"] = SETTINGS["batch_size"]
            arm_settings[arm_id] = local
            out = work / arm_id
            out.mkdir()
            weight_dir = out / "weights"
            weight_dir.mkdir()
            arm_report = report / arm_id
            arm_report.mkdir()
            if stage == "s1":
                cfg = _config(preprocessed, out, weight_dir, arm, s1, features, local)
                cfg_path = arm_report / "config.yaml"
                cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
                cmd = [sys.executable, "-u", "GPT_SoVITS/s1_train.py", "--config_file", str(cfg_path)]
            else:
                cfg = _s2_config(preprocessed, out, weight_dir, arm, s2, gpu, local)
                cfg_path = arm_report / "config.json"
                _write_json(cfg_path, cfg)
                cmd = [sys.executable, "-u", "GPT_SoVITS/s2_train_v3_lora.py", "--config", str(cfg_path)]
            log = (arm_report / "train.log").open("w", encoding="utf-8", buffering=1)
            try:
                process = subprocess.Popen(
                    cmd,
                    cwd=ROOT, env=_arm_env(arm, gpu, arm_report / "status.json", local),
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                )
            except BaseException:
                log.close()
                raise
            processes[arm_id] = (process, log, out, arm_report)
            state["arms"][arm_id] = {"gpu": gpu, "stage": stage, "status": "running"}
            observations[arm_id] = {"count": 0, "seen_at": time.monotonic(), "durations": []}

        deadline = time.monotonic() + settings["max_training_seconds"]
        while pending or any(p.poll() is None for p, *_ in processes.values()):
            # Each idle GPU prefers its lane, then may take another lane's job; completed jobs remain
            # in the report. A new job cannot overwrite another job's output.
            busy = {state["arms"][key]["gpu"] for key, (p, *_) in processes.items() if p.poll() is None}
            for entry, gpu in _dispatch_jobs(pending, busy, work_stealing=profile == "quality_session"):
                if time.monotonic() < deadline:
                    launch(gpu, entry[1])
                    pending.remove(entry)
            state["pending_arms"] = len(pending)
            _require_tmp_space(state)
            gpu_status = _gpu_status()
            now = time.monotonic()
            wall_now = time.time()
            budget_seconds = max(0.0, deadline - now)
            active_estimates = []
            failed = False
            for arm_id, (process, _, out, arm_report) in processes.items():
                rows = _read_metrics(out)
                state["arms"][arm_id].update(_summarize(rows))
                _collect_baseline(out, arm_report, state["arms"][arm_id], arm_settings[arm_id]["stage"])
                code = process.poll()
                state["arms"][arm_id]["returncode"] = code
                observation = observations[arm_id]
                if len(rows) > observation["count"]:
                    if observation["count"] > 0:
                        observation["durations"].append(
                            (now - observation["seen_at"]) / (len(rows) - observation["count"])
                        )
                    observation["count"] = len(rows)
                    observation["seen_at"] = now
                eta = None
                if code is None:
                    seconds = _remaining_seconds(
                        rows, arm_settings[arm_id]["fixed_epochs"], now - observation["seen_at"],
                        observation["durations"],
                    )
                    active_estimates.append(seconds)
                    eta = _eta_text(seconds, wall_now, budget_seconds)
                    state["arms"][arm_id]["estimated_remaining_seconds"] = seconds
                else:
                    failed = failed or code != 0
                    state["arms"][arm_id].pop("estimated_remaining_seconds", None)
                print(_progress_line(arm_id, rows, process, arm_report, arm_settings[arm_id],
                                     gpu_status.get(state["arms"][arm_id]["gpu"]), eta), flush=True)
                for path in out.glob("grokking_*metrics*.jsonl"):
                    shutil.copyfile(path, arm_report / path.name)
            overall = (max(active_estimates) if not pending and active_estimates
                       and all(v is not None for v in active_estimates) else None)
            state["estimated_remaining_seconds"] = overall
            cutoff = datetime.fromtimestamp(wall_now + budget_seconds, KST).strftime("%m/%d %H:%M KST")
            if failed:
                print(f"[전체] 일부 모델 실패 | 안전 제한 {cutoff}", flush=True)
            else:
                print(f"[전체] {_eta_text(overall, wall_now, budget_seconds)} | 안전 제한 {cutoff}", flush=True)
            _write_json(report / "summary.json", state)
            _bundle(bundle_root, archive)
            if time.monotonic() >= deadline:
                state["status"] = "time_limit"
                break
            time.sleep(20)
        if state["status"] != "time_limit":
            state["status"] = "completed" if all(p.poll() == 0 for p, *_ in processes.values()) else "training_failed"
    except BaseException:
        state["status"] = "failed"
        (report / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        for arm_id, (process, log, out, arm_report) in processes.items():
            _stop(process)
            log.close()
            for path in out.glob("grokking_*metrics*.jsonl"):
                shutil.copyfile(path, arm_report / path.name)
            rows = _read_metrics(out)
            state["arms"][arm_id].update(_summarize(rows))
            _collect_baseline(out, arm_report, state["arms"][arm_id], arm_settings[arm_id]["stage"])
            state["arms"][arm_id]["returncode"] = process.returncode
            state["arms"][arm_id].pop("estimated_remaining_seconds", None)
            state["arms"][arm_id]["status"] = "completed" if process.returncode == 0 and rows else "incomplete"
        if state["status"] == "completed" and not all(a["status"] == "completed" for a in state["arms"].values()):
            state["status"] = "missing_metrics"
        if profile == "quality_session":
            # Save compact candidates even if another lane failed. No audio or
            # checkpoints enter the publicly recovered metrics archive.
            destination = Path("/kaggle/working/private-candidates")
            destination.mkdir(parents=True, exist_ok=True)
            state["candidate_exports"] = 0
            for key, (_, _, out, _) in processes.items():
                stage = arm_settings[key]["stage"]
                name = key + ("-grok-best.ckpt" if stage == "s1" else "_grok_cfm_best.pth")
                source = out / "weights" / name
                if source.is_file():
                    shutil.copyfile(source, destination / name)
                    state["candidate_exports"] += 1
        state.pop("estimated_remaining_seconds", None)
        state["disk"] = _disk_state()
        _write_json(report / "summary.json", state)
        _bundle(bundle_root, archive)
        print("RESULT ZIP:", archive, flush=True)
    return 0 if state["status"] == "completed" else 1
