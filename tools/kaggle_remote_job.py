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
    if summary.get("git_commit") != expected_sha:
        return
    status = summary.get("status")
    report["training_status"] = status if status in ("starting", "preprocessing", "training", "completed", "failed", "time_limit", "training_failed", "missing_metrics") else "unknown"
    report["gpus"] = [v if v in ("Tesla T4", "NVIDIA Tesla T4") else "other" for v in summary.get("gpus", [])]
    fields = ("epoch", "val_loss", "train_cfm", "val_cfm", "train_eval_cfm", "epoch_seconds",
              "train_top3", "val_top3", "trainable_parameter_l2", "fit_epoch")
    def metrics(v):
        return {k: number(v[k]) for k in fields if k in v} if isinstance(v, dict) else {}
    report["arms"] = {}
    for key in ("A_cfm_off", "B_cfm_on", "A_wd001", "B_wd1"):
        arm = summary.get("arms", {}).get(key)
        if not isinstance(arm, dict):
            continue
        report["arms"][key] = {
            "epochs_recorded": number(arm.get("epochs_recorded")),
            "returncode": number(arm.get("returncode")),
            "baseline": metrics(arm.get("pretrained_baseline")),
            "best": metrics(arm.get("best")),
            "last": metrics(arm.get("last")),
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
                   PYTHONUNBUFFERED="1")
        # No GitHub credential is injected into this kernel.
        proc = subprocess.run([sys.executable, "-u", str(ROOT / "tools/kaggle_grokking_bootstrap.py")],
                              cwd=ROOT, env=env)
        report["bootstrap_exit"] = proc.returncode
        report["stage"] = "collect"
        collect(report, sha)
        arms = report.get("arms", {})
        expected = ("A_cfm_off", "B_cfm_on") if profile == "s2" else ("A_wd001", "B_wd1")
        if (proc.returncode == 0 and report.get("training_status") == "completed"
                and len(report.get("gpus", [])) == 2
                and all(arms.get(k, {}).get("epochs_recorded") == epochs
                        and arms[k].get("returncode") == 0 for k in expected)):
            report.update(status="completed", stage="finished")
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
