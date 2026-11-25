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


def _config(preprocessed, output, weights, arm, pretrained, text_features):
    import yaml
    data = yaml.safe_load((ROOT / "GPT_SoVITS/configs/s1longer-v2.yaml").read_text(encoding="utf-8"))
    data["train"].update({
        "seed": SETTINGS["seed"], "epochs": SETTINGS["base_epochs"],
        "batch_size": SETTINGS["batch_size"], "precision": "32-true",
        "if_dpo": False, "s1_adapter": "full",
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
        "batch_size": settings["batch_size"], "learning_rate": settings["learning_rate"],
        "fp16_run": False, "grad_ckpt": True, "lora_rank": settings["lora_rank"],
        "pretrained_s2G": str(pretrained), "pretrained_s2D": "",
        "gpu_numbers": str(gpu), "if_save_latest": True,
        "if_save_every_weights": False, "save_every_epoch": 100000,
    })
    data["model"]["version"] = "v5dev"
    data["data"]["exp_dir"] = str(output)
    data.update({"s2_ckpt_dir": str(output), "save_weight_dir": str(weights),
                 "name": arm["id"], "version": "v5dev"})
    return data


def run(ui):
    import torch
    import yaml
    profile = os.environ.get("GPT_SOVITS_RESEARCH_PROFILE", DEFAULT_PROFILE)
    if profile not in ("s1", "s2"):
        raise ValueError(f"Unknown research profile: {profile}")
    settings = SETTINGS if profile == "s1" else S2_SETTINGS
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
        artifact_error = ui._training_artifact_error(preprocessed, "gpt" if profile == "s1" else "sovits", "v5dev")
        if artifact_error:
            raise RuntimeError(artifact_error)
        feature_path = preprocessed / "text_feature_config.json"
        features = json.loads(feature_path.read_text(encoding="utf-8"))
        if not features.get("multilingual_bert"):
            raise RuntimeError("BERT was requested but feature marker does not confirm it")
        _require_tmp_space(state)
        state["status"] = "training"
        for gpu, arm in enumerate(settings["arms"]):
            arm_id = arm["id"]
            out = work / arm_id
            out.mkdir()
            weight_dir = out / "weights"
            weight_dir.mkdir()
            arm_report = report / arm_id
            arm_report.mkdir()
            if profile == "s1":
                cfg = _config(preprocessed, out, weight_dir, arm, s1, features)
                cfg_path = arm_report / "config.yaml"
                cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
                cmd = [sys.executable, "-u", "GPT_SoVITS/s1_train.py", "--config_file", str(cfg_path)]
            else:
                cfg = _s2_config(preprocessed, out, weight_dir, arm, s2, gpu, settings)
                cfg_path = arm_report / "config.json"
                _write_json(cfg_path, cfg)
                cmd = [sys.executable, "-u", "GPT_SoVITS/s2_train_v3_lora.py", "--config", str(cfg_path)]
            log = (arm_report / "train.log").open("w", encoding="utf-8", buffering=1)
            try:
                process = subprocess.Popen(
                    cmd,
                    cwd=ROOT, env=_arm_env(arm, gpu, arm_report / "status.json", settings),
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                )
            except BaseException:
                log.close()
                raise
            processes[arm_id] = (process, log, out, arm_report)
            state["arms"][arm_id] = {"gpu": gpu, "status": "running"}
        deadline = time.monotonic() + settings["max_training_seconds"]
        observations = {arm_id: {"count": 0, "seen_at": time.monotonic(), "durations": []}
                        for arm_id in processes}
        while any(p.poll() is None for p, *_ in processes.values()):
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
                _collect_baseline(out, arm_report, state["arms"][arm_id], profile)
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
                        rows, settings["fixed_epochs"], now - observation["seen_at"],
                        observation["durations"],
                    )
                    active_estimates.append(seconds)
                    eta = _eta_text(seconds, wall_now, budget_seconds)
                    state["arms"][arm_id]["estimated_remaining_seconds"] = seconds
                else:
                    failed = failed or code != 0
                    state["arms"][arm_id].pop("estimated_remaining_seconds", None)
                print(_progress_line(arm_id, rows, process, arm_report, settings,
                                     gpu_status.get(state["arms"][arm_id]["gpu"]), eta), flush=True)
                for path in out.glob("grokking_*metrics*.jsonl"):
                    shutil.copyfile(path, arm_report / path.name)
            overall = (max(active_estimates) if active_estimates
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
            _collect_baseline(out, arm_report, state["arms"][arm_id], profile)
            state["arms"][arm_id]["returncode"] = process.returncode
            state["arms"][arm_id].pop("estimated_remaining_seconds", None)
            state["arms"][arm_id]["status"] = "completed" if process.returncode == 0 and rows else "incomplete"
        if state["status"] == "completed" and not all(a["status"] == "completed" for a in state["arms"].values()):
            state["status"] = "missing_metrics"
        state.pop("estimated_remaining_seconds", None)
        state["disk"] = _disk_state()
        _write_json(report / "summary.json", state)
        _bundle(bundle_root, archive)
        print("RESULT ZIP:", archive, flush=True)
    return 0 if state["status"] == "completed" else 1
