"""Headless Kaggle S1 ablation. Invoked from webui.py before building Gradio.

All data stays in Kaggle. Only configs, metrics and text logs are bundled.
Each run gets its own training output and never resumes a checkpoint.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = {
    "seed": 1234,
    "batch_size": 2,
    "base_epochs": 8,
    "post_fit_epochs": 4,
    "validation_fraction": 0.10,
    "min_validation_samples": 8,
    "learning_rate": 1e-4,
    "weight_decay": 0.01,
    "grokfast_alpha": 0.98,
    "grokfast_lambda": 0.5,
    "max_training_seconds": 5400,
    "bert_model": "google-bert/bert-base-multilingual-cased",
    "arms": [{"id": "A_off", "grokfast": False}, {"id": "B_on", "grokfast": True}],
}


def _write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _read_metrics(path):
    rows = []
    for file in sorted(path.glob("grokking_s1_metrics_*.jsonl")):
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
    valid = [r for r in rows if isinstance(r.get("val_loss"), (int, float))]
    return {
        "epochs_recorded": len(rows),
        "last": rows[-1] if rows else None,
        "best": min(valid, key=lambda r: r["val_loss"]) if valid else None,
        "candidate_epochs": [r["epoch"] for r in rows if r.get("delayed_improvement")],
    }


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


def _bundle(report, archive):
    allowed = {".json", ".jsonl", ".yaml", ".log", ".txt"}
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(report.rglob("*")):
            if path.is_file() and path.suffix in allowed:
                z.write(path, path.relative_to(report))


def _arm_env(arm, gpu, status_path):
    env = os.environ.copy()
    # Remove inherited experiment overrides; the manifest is the authority.
    for key in list(env):
        if key.startswith(("GPT_SOVITS_GROKKING", "GPT_SOVITS_GROKFAST")):
            del env[key]
    env.update({
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "_CUDA_VISIBLE_DEVICES": str(gpu),
        "GPT_SOVITS_GROKKING": "1",
        "GPT_SOVITS_GROKKING_S1": "1",
        "GPT_SOVITS_GROKKING_S2": "0",
        "GPT_SOVITS_GROKFAST": "1" if arm["grokfast"] else "0",
        "GPT_SOVITS_GROKFAST_ALPHA": str(SETTINGS["grokfast_alpha"]),
        "GPT_SOVITS_GROKFAST_LAMBDA": str(SETTINGS["grokfast_lambda"]),
        "GPT_SOVITS_GROKKING_SEED": str(SETTINGS["seed"]),
        "GPT_SOVITS_GROKKING_VALIDATION_FRACTION": str(SETTINGS["validation_fraction"]),
        "GPT_SOVITS_GROKKING_MIN_VALIDATION_SAMPLES": str(SETTINGS["min_validation_samples"]),
        "GPT_SOVITS_GROKKING_S1_LR": str(SETTINGS["learning_rate"]),
        "GPT_SOVITS_GROKKING_S1_WEIGHT_DECAY": str(SETTINGS["weight_decay"]),
        "GPT_SOVITS_GROKKING_S1_POST_FIT_EPOCHS": str(SETTINGS["post_fit_epochs"]),
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


def run(ui):
    import torch
    import yaml
    import shutil
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    report = Path("/kaggle/working/grokking_reports") / run_id
    report.mkdir(parents=True, exist_ok=False)
    work = Path("/kaggle/tmp/grokking_runs") / run_id
    work.mkdir(parents=True, exist_ok=False)
    archive = Path("/kaggle/working") / ("grokking_" + run_id + ".zip")
    exp = "grok_preprocess_" + run_id
    processes = {}
    state = {"run_id": run_id, "status": "starting", "settings": SETTINGS, "arms": {}}
    _write_json(report / "summary.json", state)
    try:
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
                ui.default_bert_path, True, SETTINGS["bert_model"],
                ui.default_cnhubert_path, str(s2),
            ):
                status = str(result[0]) if result else ""
                print(status, flush=True)
                log.write(status + "\n")
                completed = status == ui.process_info(ui.process_name_1abc, "finish")
        if not completed:
            raise RuntimeError("Canonical a/b/c preprocessing did not finish")
        preprocessed = (ROOT / ui.exp_root / exp).resolve()
        artifact_error = ui._training_artifact_error(preprocessed, "gpt", "v5dev")
        if artifact_error:
            raise RuntimeError(artifact_error)
        feature_path = preprocessed / "text_feature_config.json"
        features = json.loads(feature_path.read_text(encoding="utf-8"))
        if not features.get("multilingual_bert"):
            raise RuntimeError("BERT was requested but feature marker does not confirm it")
        state["status"] = "training"
        for gpu, arm in enumerate(SETTINGS["arms"]):
            arm_id = arm["id"]
            out = work / arm_id
            out.mkdir()
            weight_dir = out / "weights"
            weight_dir.mkdir()
            arm_report = report / arm_id
            arm_report.mkdir()
            cfg = _config(preprocessed, out, weight_dir, arm, s1, features)
            cfg_path = arm_report / "config.yaml"
            cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
            log = (arm_report / "train.log").open("w", encoding="utf-8", buffering=1)
            try:
                process = subprocess.Popen(
                    [sys.executable, "-u", "GPT_SoVITS/s1_train.py", "--config_file", str(cfg_path)],
                    cwd=ROOT, env=_arm_env(arm, gpu, arm_report / "status.json"),
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                )
            except BaseException:
                log.close()
                raise
            processes[arm_id] = (process, log, out, arm_report)
            state["arms"][arm_id] = {"gpu": gpu, "status": "running"}
        deadline = time.monotonic() + SETTINGS["max_training_seconds"]
        while any(p.poll() is None for p, *_ in processes.values()):
            for arm_id, (process, _, out, arm_report) in processes.items():
                rows = _read_metrics(out)
                state["arms"][arm_id].update(_summarize(rows))
                state["arms"][arm_id]["returncode"] = process.poll()
                print(arm_id, "epoch", len(rows), "exit", process.poll(), flush=True)
                for path in out.glob("grokking_s1_metrics_*.jsonl"):
                    shutil.copyfile(path, arm_report / path.name)
            _write_json(report / "summary.json", state)
            _bundle(report, archive)
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
            for path in out.glob("grokking_s1_metrics_*.jsonl"):
                shutil.copyfile(path, arm_report / path.name)
            rows = _read_metrics(out)
            state["arms"][arm_id].update(_summarize(rows))
            state["arms"][arm_id]["returncode"] = process.returncode
            state["arms"][arm_id]["status"] = "completed" if process.returncode == 0 and rows else "incomplete"
        if state["status"] == "completed" and not all(a["status"] == "completed" for a in state["arms"].values()):
            state["status"] = "missing_metrics"
        _write_json(report / "summary.json", state)
        _bundle(report, archive)
        print("RESULT ZIP:", archive, flush=True)
    return 0 if state["status"] == "completed" else 1
