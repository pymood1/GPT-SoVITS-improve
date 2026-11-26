"""Private Kaggle entry point. The Actions launcher prepends a pinned REQUEST."""
import json
import math
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

REPO = "https://github.com/pymood1/GPT-SoVITS-improve.git"
ROOT = Path("/kaggle/tmp/GPT-SoVITS")
WORKING = Path("/kaggle/working")
SAFE_RESULT = WORKING / "remote-result.json"


def number(v):
    return v if type(v) in (int, float) and math.isfinite(v) else None


def collect(report, expected_sha):
    # Raw ZIP stays private in Kaggle. Never forward paths, text, audio, or logs.
    archive = WORKING / "grokking_result.zip"
    if not archive.is_file():
        return
    with zipfile.ZipFile(archive) as z:
        run = json.loads(z.read("run.json"))
        if run.get("git_commit") != expected_sha:
            return
        report["bootstrap_status"] = run.get("status") if run.get("status") in ("running", "completed", "failed") else "unknown"
        if "training/summary.json" not in z.namelist():
            return
        summary = json.loads(z.read("training/summary.json"))
        histories = {}
        for key in ("A_cfm_off", "B_cfm_on", "A_wd001", "B_wd1", "A_lr100", "B_lr030", "S1_v_o", "S1_qkv_o", "S2_rank16", "S2_rank32"):
            prefix = "training/" + key + "/grokking_"
            rows = []
            for name in z.namelist():
                if name.startswith(prefix) and name.endswith(".jsonl") and "metrics" in name:
                    for line in z.read(name).decode("utf-8").splitlines():
                        try:
                            row = json.loads(line)
                            if isinstance(row, dict):
                                rows.append(row)
                        except ValueError:
                            continue
            histories[key] = rows
    if summary.get("git_commit") != expected_sha:
        return
    counts = summary.get("reference_counts")
    if isinstance(counts, dict):
        report["reference_counts"] = {k: number(counts.get(k)) for k in
                                      ("rows", "readable", "duration_eligible", "text_eligible", "korean_eligible")}
    if summary.get("reused_checkpoints") in (2, 4) and summary.get("training_epochs") == 0:
        report.update(reused_checkpoints=summary["reused_checkpoints"], training_epochs=0)
    if type(summary.get("candidate_exports")) is int:
        report["candidate_exports"] = number(summary["candidate_exports"])
    status = summary.get("status")
    report["training_status"] = status if status in ("starting", "preprocessing", "training", "completed", "failed", "time_limit", "training_failed", "missing_metrics") else "unknown"
    if status == "failed":
        phase = summary.get("failure_phase")
        report["failure_phase"] = phase if phase in ("disk_preflight", "pretrained_setup", "preprocessing", "artifact_validation", "candidate_mount", "candidate_link") else "unknown"
        error = summary.get("error_type")
        report["training_error_type"] = error if error in ("RuntimeError", "FileNotFoundError", "KeyError", "ValueError", "OSError", "PermissionError", "JSONDecodeError", "CalledProcessError", "ImportError", "ModuleNotFoundError") else "other"
        if type(summary.get("candidate_mount_matches")) is int:
            report["candidate_mount_matches"] = number(summary["candidate_mount_matches"])

    report["gpus"] = [v if v in ("Tesla T4", "NVIDIA Tesla T4") else "other" for v in summary.get("gpus", [])]
    fields = ("epoch", "val_loss", "train_cfm", "val_cfm", "train_eval_cfm", "epoch_seconds",
              "train_top3", "val_top3", "trainable_parameter_l2", "fit_epoch")
    def metrics(v):
        return {k: number(v[k]) for k in fields if k in v} if isinstance(v, dict) else {}
    report["arms"] = {}
    for key in ("A_cfm_off", "B_cfm_on", "A_wd001", "B_wd1", "A_lr100", "B_lr030", "S1_v_o", "S1_qkv_o", "S2_rank16", "S2_rank32"):
        arm = summary.get("arms", {}).get(key)
        if not isinstance(arm, dict):
            continue
        report["arms"][key] = {
            "epochs_recorded": number(arm.get("epochs_recorded")),
            "returncode": number(arm.get("returncode")),
            "baseline": metrics(arm.get("pretrained_baseline")),
            "best": metrics(arm.get("best")),
            "last": metrics(arm.get("last")),
            "history": [metrics(row) for row in histories.get(key, [])],
            "candidate_epochs": [number(v) for v in arm.get("candidate_epochs", []) if type(v) is int],
        }


def main():
    req = REQUEST
    sha = req["code_sha"]
    profile = req["profile"]
    epochs = req["epochs"]
    budget = req["max_training_seconds"]
    report = {"code_sha": sha, "profile": profile, "epochs": epochs, "status": "failed", "stage": "clone"}
    started = time.monotonic()
    WORKING.mkdir(parents=True, exist_ok=True)
    ROOT.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(["git", "init", str(ROOT)], check=True)
        subprocess.run(["git", "-C", str(ROOT), "fetch", "--depth=1", REPO, sha], check=True)
        subprocess.run(["git", "-C", str(ROOT), "checkout", "--detach", "FETCH_HEAD"], check=True)
        actual = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
        if actual != sha:
            raise RuntimeError("Unexpected commit")
        report["stage"] = "bootstrap"
        env = os.environ.copy()
        env.update(GPT_SOVITS_RESEARCH_PROFILE=profile,
                   GPT_SOVITS_RESEARCH_EPOCHS=str(epochs),
                   GPT_SOVITS_RESEARCH_MAX_SECONDS=str(budget),
                   GPT_SOVITS_RESEARCH_SOURCE_KERNEL=req.get("source_kernel", ""),
                   PYTHONUNBUFFERED="1")
        # No GitHub credential is injected into this kernel.
        proc = subprocess.run([sys.executable, "-u", str(ROOT / "tools/kaggle_grokking_bootstrap.py")],
                              cwd=ROOT, env=env)
        report["bootstrap_exit"] = proc.returncode
        report["stage"] = "collect"
        collect(report, sha)
        arms = report.get("arms", {})
        expected = {"s2": ("A_cfm_off", "B_cfm_on"), "s1": ("A_wd001", "B_wd1"),
                    "s2_quality": ("A_lr100", "B_lr030"), "s2_listening": (),
                    "quality_session": ("S1_v_o", "S1_qkv_o", "S2_rank16", "S2_rank32"), "quality_reuse": (), "quality_mix": (), "quality_soft": (), "quality_general": (), "quality_style": ()}[profile]
        if (proc.returncode == 0 and report.get("training_status") == "completed"
                and len(report.get("gpus", [])) == 2
                and (profile != "s2_listening" or report.get("reused_checkpoints") == 2)
                and (profile != "quality_session" or report.get("candidate_exports") == 4)
                and (profile not in ("quality_reuse", "quality_mix", "quality_soft", "quality_general", "quality_style") or report.get("reused_checkpoints") == 4)
                and all(arms.get(k, {}).get("epochs_recorded") == epochs
                        and arms[k].get("returncode") == 0 for k in expected)):
            report.update(status="completed", stage="finished")
            if profile in ("s2_quality", "s2_listening", "quality_session", "quality_reuse", "quality_mix", "quality_soft", "quality_general", "quality_style"):
                report["stage"] = "listening"
                with zipfile.ZipFile(WORKING / "grokking_result.zip") as z:
                    run_id = json.loads(z.read("run.json"))["run_id"]
                import re
                if not re.fullmatch(r"\d{8}T\d{6}_\d{6}Z", run_id):
                    raise ValueError("Invalid run ID")
                output = WORKING / "private-listening"
                env["PYTHONPATH"] = os.pathsep.join([str(ROOT), str(ROOT / "GPT_SoVITS")])
                env["HF_HOME"] = "/kaggle/tmp/huggingface"
                env["CUDA_VISIBLE_DEVICES"] = "0"
                env["version"] = "v5dev"
                env["language"] = "ko_KR"
                from importlib.util import spec_from_file_location, module_from_spec
                spec = spec_from_file_location("kaggle_launcher", ROOT / "tools/kaggle_launch_webui.py")
                launcher = module_from_spec(spec)
                spec.loader.exec_module(launcher)
                launcher.apply_nvidia_library_path(env)
                if req.get("deploy", False):
                    report.update(status="failed", stage="deployment")
                    proc = subprocess.run(
                        [sys.executable, "-u", str(ROOT / "tools/practical_v5.py"),
                         "--repo", str(ROOT), "--export-work", str(Path("/kaggle/tmp/grokking_runs") / run_id),
                         "--output", str(WORKING)], cwd=ROOT, env=env, timeout=1800)
                    report["deployment_exit"] = proc.returncode
                    raw = json.loads((WORKING / "deploy-status.json").read_text())
                    report["deployment"] = {
                        "status": raw.get("status") if raw.get("status") in ("completed", "failed") else "unknown",
                        **{k: number(raw.get(k)) for k in ("smoke_count", "file_count", "archive_bytes")},
                    }
                    if proc.returncode == 0 and raw.get("status") == "completed" and raw.get("smoke_count") == 3:
                        report.update(status="completed", stage="finished")
                    return
                # Only safe aggregate metrics are exported by this launcher.
                try:
                    proc = subprocess.run([sys.executable, "-u", str(ROOT / "tools/kaggle_quality_listening.py"),
                                           str(Path("/kaggle/tmp/grokking_runs") / run_id), str(output)],
                                          cwd=ROOT, env=env, timeout=9600 if profile in ("quality_mix", "quality_soft", "quality_general", "quality_style") else 5400 if profile == "quality_reuse" else 1500 if profile == "quality_session" else 900)
                    report["listening_exit"] = proc.returncode
                    path = output / "listening-metrics.json"
                    if path.is_file():
                        raw = json.loads(path.read_text())
                        report["listening"] = {"status": raw.get("status") if raw.get("status") in ("completed", "failed") else "unknown",
                                               "models": {}}
                        counts = raw.get("reference_counts")
                        if isinstance(counts, dict):
                            report["listening"]["reference_counts"] = {k: number(counts.get(k)) for k in
                                                                       ("rows", "readable", "duration_eligible", "text_eligible", "korean_eligible")}
                        evaluation = raw.get("evaluation")
                        if isinstance(evaluation, dict):
                            report["listening"]["evaluation"] = {
                                k: evaluation[k] for k in ("asr_status", "speaker_status")
                                if evaluation.get(k) in ("completed", "failed", "unavailable")
                            }
                            for stage, allowed in (("asr", ("cuda_int8_float32", "cpu_int8")),
                                                   ("speaker", ("cuda_fp32", "cpu_fp32"))):
                                backend = evaluation.get(stage + "_backend")
                                if backend in allowed:
                                    report["listening"]["evaluation"][stage + "_backend"] = backend
                                for suffix in ("_seconds", "_calibration_seconds", "_calibration_count", "_cpu_calibration_seconds", "_gpu_calibration_seconds"):
                                    value = number(evaluation.get(stage + suffix))
                                    if value is not None and value >= 0:
                                        report["listening"]["evaluation"][stage + suffix] = value
                        selected = raw.get("selected", {})
                        if isinstance(selected, dict) and selected.get("s1") in ("S1_v_o", "S1_qkv_o") and selected.get("s2") in ("S2_rank16", "S2_rank32"):
                            report["listening"]["selected"] = {"s1": selected["s1"], "s2": selected["s2"]}
                        for label in ("base", "A_lr100", "B_lr030", "s1_only", "s2_only", "combined", "rank16", "rank32", "mix025", "mix050", "mix075", "mix125", "cfg100", "cfg160", "soft000", "soft005", "soft010", "soft025", "soft050", "soft100", "soft000b", "mix065", "mix085", "gen_base", "gen_hard", "gen_soft010", "gen_hardb", "gen_soft010b", "gen_soft025", "style_base", "style_base_mean", "style_base_sv", "style_lora", "style_lora_mean", "style_lora_sv"):
                            v = raw.get("models", {}).get(label, {})
                            if v:
                                report["listening"]["models"][label] = {
                                    "status": v.get("status") if v.get("status") in ("completed", "failed") else "unknown",
                                    "stage": v.get("stage") if v.get("stage") in ("model_load", "reference", "synthesis", "output_validation", "finished") else "unknown",
                                    "observed_sample_rate": number(v.get("observed_sample_rate")),
                                    "samples": [{k: number(row.get(k)) for k in ("index", "reference_index", "text_index", "seed", "sample_rate", "seconds", "rms", "peak", "cer", "char_errors", "char_count", "speaker_cosine", "speaker_holdout_cosine")}
                                                for row in v.get("samples", [])],
                                }
                except Exception as exc:
                    report["listening_error_type"] = type(exc).__name__
                report["stage"] = "finished"

    except Exception as exc:
        # Error class is safe; exception messages and traceback stay in private logs.
        report["error_type"] = type(exc).__name__
    finally:
        report["elapsed_seconds"] = round(time.monotonic() - started, 2)
        SAFE_RESULT.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        print("REMOTE_RESULT", report["status"], report["stage"], flush=True)
    # Save a diagnostic output even when model training fails. Actions checks status separately.


if __name__ == "__main__":
    main()
