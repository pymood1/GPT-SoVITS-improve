"""Kaggle bootstrap: install, preflight, run, and retain diagnostics on failure."""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.kaggle_grokking_runner import _bundle, _disk_state, DISK_RESERVE_BYTES


def run_logged(args, log_path, env):
    """Retain the full stage log and stream new diagnostics to the notebook."""
    print(f"[stage] {log_path.stem} starting", flush=True)
    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        proc = subprocess.Popen(args, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        last = 0.0
        try:
            with log_path.open("r", encoding="utf-8", errors="replace") as reader:
                while proc.poll() is None:
                    chunk = reader.read()
                    if chunk:
                        print(chunk, end="" if chunk.endswith("\n") else "\n", flush=True)
                    now = time.monotonic()
                    if now - last >= 30:
                        print(f"[{log_path.stem}] running; {log_path.stat().st_size} log bytes", flush=True)
                        last = now
                    time.sleep(2)
                chunk = reader.read()
                if chunk:
                    print(chunk, end="" if chunk.endswith("\n") else "\n", flush=True)
        except BaseException:
            proc.terminate()
            raise
        code = proc.wait()
    if code:
        with log_path.open("rb") as log:
            log.seek(max(0, log_path.stat().st_size - 16000))
            tail = log.read().decode("utf-8", errors="replace")
        print(f"\\n[FAILED STAGE] {log_path.stem}; exit={code}\\n{tail}", flush=True)
        raise RuntimeError(f"{log_path.stem} exited {code}; full log: {log_path}")
    print(f"[stage] {log_path.stem} completed", flush=True)


def main():
    if not Path("/kaggle/input").is_dir():
        raise RuntimeError("Run in Kaggle, with GPU T4 x2 and Internet enabled")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = Path("/kaggle/tmp/grokking_bootstrap") / run_id
    output.mkdir(parents=True, exist_ok=False)
    archive = Path("/kaggle/working") / f"grokking_bootstrap_{run_id}.zip"
    env = os.environ.copy()
    env.update({"GPT_SOVITS_KAGGLE_GROKKING": "1", "PYTHONUNBUFFERED": "1",
                "HF_HOME": "/kaggle/tmp/huggingface", "TMPDIR": "/kaggle/tmp"})
    # Record only code identity and stage logs, not environment variables or tokens.
    try:
        disk = _disk_state()
        print("[disk]", disk, flush=True)
        if disk["tmp"]["free_bytes"] < DISK_RESERVE_BYTES:
            raise RuntimeError("Insufficient Kaggle tmp free space")
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        (output / "git_commit.txt").write_text(commit + "\n", encoding="utf-8")
        run_logged(["nvidia-smi", "-L"], output / "gpu.log", env)
        setup = ROOT / "tools/colab_fast_setup.py"
        original = setup.read_bytes()
        source = original.decode("utf-8").replace("/content/", "/kaggle/tmp/")
        marker = Path("/kaggle/tmp/gpt_sovits_grokking_install.sha256")
        requirements = (ROOT / "requirements.txt").read_bytes()
        digest = hashlib.sha256(original + requirements + sys.version.encode()).hexdigest()
        required = [
            ROOT / "GPT_SoVITS/pretrained_models/s1v3.ckpt",
            ROOT / "GPT_SoVITS/pretrained_models/gsv-v5-pretrained/s2Gv5dev.pth",
            ROOT / "GPT_SoVITS/pretrained_models/gsv-v5-pretrained/vocoder.pth",
        ]
        reusable = marker.is_file() and marker.read_text().strip() == digest and all(p.is_file() for p in required)
        if reusable:
            (output / "setup.log").write_text("Reusing verified installation in this Kaggle session.\n")
        else:
            try:
                setup.write_text(source, encoding="utf-8")
                run_logged([sys.executable, "-u", str(setup)], output / "setup.log", env)
                marker.write_text(digest)
            finally:
                setup.write_bytes(original)
            # Download archives are disposable; keep installed models and HF cache.
            shutil.rmtree("/kaggle/tmp/gpt_sovits_downloads", ignore_errors=True)
        print("[disk after setup]", _disk_state(), flush=True)
        run_logged([sys.executable, "-u", str(ROOT / "tools/kaggle_launch_webui.py")],
                   output / "launcher.log", env)
        return 0
    except BaseException:
        error = traceback.format_exc()
        (output / "error.txt").write_text(error, encoding="utf-8")
        print(error, flush=True)
        raise
    finally:
        _bundle(output, archive)
        print("BOOTSTRAP DIAGNOSTICS:", archive, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
