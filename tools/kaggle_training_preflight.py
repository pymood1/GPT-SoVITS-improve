#!/usr/bin/env python3
"""Kaggle training preflight for GPT-SoVITS Improve.

This script does not train anything. It verifies that the Kaggle runtime is
ready for S1 / SoVITS / Emotion Controller experiments and runs only fast
synthetic self-tests.

Example:
    python tools/kaggle_training_preflight.py \
      --training-list /kaggle/tmp/audio_input/training.list \
      --datas-dir /kaggle/input/datasets/pymood1/endfield/datas
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = Path("/kaggle/working/gpt_sovits_output")
DEFAULT_EMOTION_ROOT = Path("/kaggle/tmp/emotion_opt")

KEY_FILES = (
    "webui.py",
    "GPT_SoVITS/s1_train.py",
    "GPT_SoVITS/AR/models/t2s_lightning_module.py",
    "GPT_SoVITS/AR/modules/activation.py",
    "GPT_SoVITS/module/cache_dit.py",
    "GPT_SoVITS/module/models.py",
    "GPT_SoVITS/f5_tts/model/backbones/dit.py",
    "GPT_SoVITS/TTS_infer_pack/TTS.py",
    "GPT_SoVITS/inference_webui.py",
    "GPT_SoVITS/inference_webui_fast.py",
    "tools/asr/fasterwhisper_asr.py",
    "tools/asr/funasr_asr.py",
    "tools/emotion_label_refiner.py",
    "tools/build_emotion_review_set.py",
    "tools/evaluate_emotion_review.py",
    "tools/calibrate_emotion_seed_thresholds.py",
)

SELF_TESTS = (
    "tools/s1_lora_selftest.py",
    "tools/emotion_labeling_selftest.py",
)

PRETRAINED_CANDIDATES = {
    "s1v3": (
        "GPT_SoVITS/pretrained_models/s1v3.ckpt",
        "GPT_SoVITS/pretrained_models/s1v3.ckpt",
    ),
    "s2Gv5dev": (
        "GPT_SoVITS/pretrained_models/s2Gv5dev.pth",
        "GPT_SoVITS/pretrained_models/gsv-v5-pretrained/s2Gv5dev.pth",
    ),
    "vocoder": (
        "GPT_SoVITS/pretrained_models/vocoder.pth",
        "GPT_SoVITS/pretrained_models/gsv-v5-pretrained/vocoder.pth",
    ),
}


class PreflightError(RuntimeError):
    pass


def human_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    amount = float(value)
    for unit in units:
        if abs(amount) < 1024.0 or unit == units[-1]:
            return f"{amount:.2f} {unit}"
        amount /= 1024.0
    return f"{amount:.2f} TiB"


def check_dataset(
    training_list: Path,
    datas_dir: Path,
) -> dict[str, Any]:
    if not training_list.is_file():
        raise PreflightError(
            f"training.list not found: {training_list}"
        )
    if not datas_dir.is_dir():
        raise PreflightError(f"datas dir not found: {datas_dir}")

    lines = training_list.read_text(
        encoding="utf-8-sig",
    ).splitlines()
    sample_count = 0
    missing: list[str] = []
    malformed: list[int] = []

    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        parts = line.split("|")
        if len(parts) < 4:
            malformed.append(line_number)
            continue
        sample_count += 1
        audio = Path(parts[0].replace("\\", "/"))
        if not audio.is_file():
            candidate = datas_dir / audio.name
            if not candidate.is_file():
                missing.append(str(audio))

    if malformed:
        raise PreflightError(
            "malformed training.list rows: "
            + ", ".join(map(str, malformed[:10]))
        )
    if missing:
        raise PreflightError(
            f"{len(missing)} audio files are missing; "
            f"first={missing[0]}"
        )
    if sample_count < 1:
        raise PreflightError("training.list has no usable samples")

    return {
        "samples": sample_count,
        "training_list": str(training_list),
        "datas_dir": str(datas_dir),
    }


def find_pretrained() -> dict[str, str | None]:
    found: dict[str, str | None] = {}
    for name, candidates in PRETRAINED_CANDIDATES.items():
        resolved = None
        for relative in candidates:
            path = ROOT / relative
            if path.is_file():
                resolved = str(path)
                break
        found[name] = resolved
    return found


def run_py_compile() -> list[str]:
    compiled: list[str] = []
    for relative in KEY_FILES:
        path = ROOT / relative
        if not path.is_file():
            raise PreflightError(
                f"required source file missing: {relative}"
            )
        subprocess.run(
            [sys.executable, "-m", "py_compile", str(path)],
            cwd=ROOT,
            check=True,
        )
        compiled.append(relative)
    return compiled


def run_self_tests() -> list[str]:
    passed: list[str] = []
    for relative in SELF_TESTS:
        path = ROOT / relative
        if not path.is_file():
            raise PreflightError(
                f"self-test missing: {relative}"
            )
        print(f"\n[self-test] {relative}", flush=True)
        subprocess.run(
            [sys.executable, "-u", str(path)],
            cwd=ROOT,
            check=True,
        )
        passed.append(relative)
    return passed


def gpu_summary() -> dict[str, Any]:
    count = int(torch.cuda.device_count())
    return {
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": count,
        "devices": [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "capability": ".".join(
                    map(
                        str,
                        torch.cuda.get_device_capability(index),
                    )
                ),
            }
            for index in range(count)
        ],
    }


def disk_summary(path: Path) -> dict[str, Any]:
    path.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(path)
    return {
        "path": str(path),
        "total": human_bytes(usage.total),
        "used": human_bytes(usage.used),
        "free": human_bytes(usage.free),
    }


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-list", required=True)
    parser.add_argument("--datas-dir", required=True)
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
    )
    parser.add_argument(
        "--emotion-root",
        default=str(DEFAULT_EMOTION_ROOT),
    )
    parser.add_argument(
        "--skip-self-tests",
        action="store_true",
    )
    return parser


def main() -> int:
    args = make_parser().parse_args()
    try:
        training_list = Path(
            args.training_list
        ).expanduser().resolve()
        datas_dir = Path(args.datas_dir).expanduser().resolve()
        output_root = Path(
            args.output_root
        ).expanduser().resolve()
        emotion_root = Path(
            args.emotion_root
        ).expanduser().resolve()

        output_root.mkdir(parents=True, exist_ok=True)
        emotion_root.mkdir(parents=True, exist_ok=True)

        print("[preflight] dataset", flush=True)
        dataset = check_dataset(training_list, datas_dir)

        print("[preflight] Python compile", flush=True)
        compiled = run_py_compile()

        gpus = gpu_summary()
        if not gpus["cuda_available"]:
            raise PreflightError(
                "CUDA is not available. Enable a Kaggle GPU accelerator."
            )

        pretrained = find_pretrained()
        missing_pretrained = [
            name
            for name, path in pretrained.items()
            if path is None
        ]
        if missing_pretrained:
            raise PreflightError(
                "missing pretrained files after setup: "
                + ", ".join(missing_pretrained)
            )

        tests = []
        if not args.skip_self_tests:
            tests = run_self_tests()

        report = {
            "status": "PASS",
            "python": sys.version.split()[0],
            "dataset": dataset,
            "gpu": gpus,
            "pretrained": pretrained,
            "compiled": compiled,
            "self_tests": tests,
            "disk": {
                "tmp": disk_summary(Path("/kaggle/tmp")),
                "working": disk_summary(Path("/kaggle/working")),
            },
            "paths": {
                "repo": str(ROOT),
                "output_root": str(output_root),
                "emotion_root": str(emotion_root),
            },
            "recommended_gpu_policy": (
                "GPT/S1 and SoVITS independent jobs: one GPU each when "
                "running them concurrently; avoid DDP/model-parallel for "
                "these modest independent jobs."
                if gpus["cuda_device_count"] >= 2
                else "Single GPU runtime: run GPT/S1 and SoVITS sequentially."
            ),
            "next": [
                "Launch webui.py from the Kaggle notebook.",
                "Keep Full fine-tuning as the baseline.",
                "For S1 retention experiment, select V + Attention Out LoRA, rank=8, alpha=16.",
                "Do not reuse an Emotion Controller across different S1 checkpoints; retrain per S1.",
            ],
        }
        print("\n" + json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ))
        return 0
    except (
        PreflightError,
        OSError,
        ValueError,
        RuntimeError,
        subprocess.CalledProcessError,
    ) as exc:
        print(
            f"\n[FATAL] {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
