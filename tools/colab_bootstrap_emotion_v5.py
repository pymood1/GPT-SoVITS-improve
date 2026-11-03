#!/usr/bin/env python3
"""Prepare a Colab checkout of GPT-SoVITS with the emotion-v5 branch.

The checkout keeps the official repository as ``origin`` and adds the user's
fork as ``emotion-patch``. It refuses to overwrite a dirty worktree.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

UPSTREAM_URL = "https://github.com/RVC-Boss/GPT-SoVITS.git"
PATCH_URL = "https://github.com/pymood1/GPT-SoVITS-improve.git"
PATCH_BRANCH = "emotion-v5"


def run(cmd, cwd=None, check=True):
    print("+", " ".join(str(x) for x in cmd))
    return subprocess.run(cmd, cwd=cwd, check=check, text=True)


def capture(cmd, cwd=None):
    return subprocess.check_output(cmd, cwd=cwd, text=True).strip()


def ensure_clean_repo(target: Path) -> None:
    status = capture(["git", "status", "--porcelain"], cwd=target)
    if status:
        raise RuntimeError(
            f"Refusing to modify dirty checkout at {target}. "
            "Commit/stash changes first."
        )


def ensure_remote(target: Path, name: str, url: str) -> None:
    remotes = capture(["git", "remote"], cwd=target).splitlines()
    if name not in remotes:
        run(["git", "remote", "add", name, url], cwd=target)
        return
    current = capture(["git", "remote", "get-url", name], cwd=target)
    if current != url:
        run(["git", "remote", "set-url", name, url], cwd=target)


def verify(target: Path) -> None:
    required = [
        target / "GPT_SoVITS/AR/models/emotion_control.py",
        target / "GPT_SoVITS/AR/models/t2s_model.py",
        target / "GPT_SoVITS/TTS_infer_pack/TTS.py",
        target / "tools/emotion_dataset_builder.py",
    ]
    missing = [
        str(path.relative_to(target))
        for path in required
        if not path.exists()
    ]
    if missing:
        raise RuntimeError(
            f"Patch verification failed; missing files: {missing}"
        )

    run(
        [
            sys.executable,
            "-m",
            "py_compile",
            "GPT_SoVITS/AR/models/emotion_control.py",
            "GPT_SoVITS/AR/models/t2s_model.py",
            "GPT_SoVITS/TTS_infer_pack/TTS.py",
            "tools/emotion_dataset_builder.py",
        ],
        cwd=target,
    )

    t2s_text = (
        target / "GPT_SoVITS/AR/models/t2s_model.py"
    ).read_text(encoding="utf-8")
    tts_text = (
        target / "GPT_SoVITS/TTS_infer_pack/TTS.py"
    ).read_text(encoding="utf-8")

    markers = [
        "sample_with_emotion_preference",
        "emotion_controller = None",
        "emotion_enabled",
    ]
    joined = t2s_text + "\n" + tts_text
    absent = [marker for marker in markers if marker not in joined]
    if absent:
        raise RuntimeError(
            f"Patch verification failed; markers absent: {absent}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="/content/GPT-SoVITS")
    parser.add_argument("--upstream", default=UPSTREAM_URL)
    parser.add_argument("--patch-repo", default=PATCH_URL)
    parser.add_argument("--branch", default=PATCH_BRANCH)
    args = parser.parse_args()

    target = Path(args.target).expanduser().resolve()
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", args.upstream, str(target)])
    elif not (target / ".git").exists():
        raise RuntimeError(
            f"Target exists but is not a git checkout: {target}"
        )

    ensure_clean_repo(target)
    ensure_remote(target, "origin", args.upstream)
    ensure_remote(target, "emotion-patch", args.patch_repo)

    run(["git", "fetch", "origin", "main"], cwd=target)
    run(["git", "fetch", "emotion-patch", args.branch], cwd=target)
    run(
        [
            "git",
            "checkout",
            "-B",
            args.branch,
            f"emotion-patch/{args.branch}",
        ],
        cwd=target,
    )

    verify(target)
    commit = capture(["git", "rev-parse", "HEAD"], cwd=target)
    print(f"\n[OK] GPT-SoVITS emotion branch ready: {commit}")
    print(f"[OK] Path: {target}")
    print(
        "[NOTE] Dependency/model installation is intentionally left "
        "to the official GPT-SoVITS setup."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
