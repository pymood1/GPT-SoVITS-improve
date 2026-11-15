#!/usr/bin/env python3
"""Prepare Kaggle data, run preflight, then launch the GPT-SoVITS WebUI."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.kaggle_dataset import prepare_kaggle_dataset


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        default="",
        help=(
            "Optional Kaggle dataset root. When omitted, the repo discovers "
            "pymood1/w-voice-file first and otherwise scans /kaggle/input."
        ),
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
    )
    return parser


def main() -> int:
    args = make_parser().parse_args()

    dataset = prepare_kaggle_dataset(
        args.dataset_root or None,
    )
    datas_dir = dataset["datas_dir"]
    training_list = dataset["training_list"]

    print(
        "[dataset] mode        :",
        dataset["mode"],
        flush=True,
    )
    print(
        "[dataset] datas       :",
        datas_dir,
        flush=True,
    )
    print(
        "[dataset] training.list:",
        training_list or "<auto-generate>",
        flush=True,
    )
    print(
        "[dataset] audio files :",
        dataset["audio_count"],
        flush=True,
    )

    output_root = Path("/kaggle/working/gpt_sovits_output")
    emotion_root = Path("/kaggle/tmp/emotion_opt")
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "GPT_weights_v5dev").mkdir(
        parents=True,
        exist_ok=True,
    )
    (output_root / "SoVITS_weights_v5dev").mkdir(
        parents=True,
        exist_ok=True,
    )
    (output_root / "emotion_controllers").mkdir(
        parents=True,
        exist_ok=True,
    )
    emotion_root.mkdir(parents=True, exist_ok=True)

    if not args.skip_preflight:
        preflight = [
            sys.executable,
            "-u",
            str(ROOT / "tools" / "kaggle_training_preflight.py"),
            "--datas-dir",
            datas_dir,
            "--output-root",
            str(output_root),
            "--emotion-root",
            str(emotion_root),
        ]
        if training_list:
            preflight.extend(
                [
                    "--training-list",
                    training_list,
                ]
            )

        print(
            "[preflight] 자동 실행",
            flush=True,
        )
        subprocess.run(
            preflight,
            cwd=ROOT,
            check=True,
        )

    env = os.environ.copy()
    python_path = [
        str(ROOT),
        str(ROOT / "GPT_SoVITS"),
    ]
    if env.get("PYTHONPATH"):
        python_path.append(env["PYTHONPATH"])

    env.update(
        {
            "PYTHONPATH": os.pathsep.join(python_path),
            "language": "ko_KR",
            "version": "v5dev",
            "is_share": "True",
            "HF_HOME": "/kaggle/tmp/huggingface",
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
            "HF_HUB_DISABLE_UPDATE_CHECK": "1",
            "HF_HUB_DOWNLOAD_TIMEOUT": "120",
            "HF_XET_HIGH_PERFORMANCE": "1",
            "GPT_SOVITS_LOW_DISK_SAVE": "1",
            "GPT_SOVITS_OUTPUT_ROOT": str(output_root),
            "GPT_SOVITS_EMOTION_WORK_ROOT": str(emotion_root),
            "GPT_SOVITS_DEFAULT_TRAINING_LIST": training_list,
            "GPT_SOVITS_DEFAULT_DATAS_DIR": datas_dir,
            "GPT_SOVITS_KAGGLE_DATASET_ROOT": dataset["source_root"],
            "TMPDIR": "/kaggle/tmp",
            "PYTHONUNBUFFERED": "1",
            "GRADIO_ANALYTICS_ENABLED": "False",
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        }
    )

    print("[run] 파인튜닝 WebUI 시작 — 한국어", flush=True)
    print(
        "[run] 데이터셋 준비/Preflight/WebUI 경로 주입은 Git 코드가 관리합니다.",
        flush=True,
    )

    process = subprocess.run(
        [
            sys.executable,
            "-u",
            "webui.py",
            "ko_KR",
        ],
        cwd=ROOT,
        env=env,
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"webui.py failed with exit code {process.returncode}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
