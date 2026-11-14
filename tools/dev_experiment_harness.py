"""Temporary helpers for the Kaggle one-click development experiment.

This module is intentionally small and removable. It does not train models
itself; webui.py continues to call the existing preprocessing/training
pipelines. Remove this file together with DEV_EXPERIMENT_PRESET.yaml and the
temporary WebUI section once the final training direction is selected.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

import yaml

from tools.emotion_training_pipeline import (
    latest_gpt_checkpoint_for_experiment,
)


ALLOWED_S1_ADAPTERS = {"full", "lora_v_o", "lora_qkv_o"}


def slug(value: object) -> str:
    text = re.sub(r"[^0-9A-Za-z._-]+", "_", str(value or "").strip())
    return text.strip("._-") or "case"


def load_preset(path: Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise RuntimeError(f"개발 실험 preset이 없습니다: {path}")

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError("개발 실험 preset 형식이 올바르지 않습니다.")

    version = str(data.get("version") or "").strip()
    if version != "v5dev":
        raise RuntimeError(
            "현재 원클릭 실험 하네스는 v5dev만 지원합니다. "
            f"preset version={version!r}"
        )

    training = data.get("training")
    if not isinstance(training, dict):
        raise RuntimeError("preset.training이 없습니다.")

    cases = training.get("cases")
    if not isinstance(cases, list) or not cases:
        raise RuntimeError("preset.training.cases가 비어 있습니다.")

    seen = set()
    for case in cases:
        if not isinstance(case, dict):
            raise RuntimeError("각 training case는 mapping이어야 합니다.")
        case_id = slug(case.get("id"))
        if case_id in seen:
            raise RuntimeError(f"중복 case id: {case_id}")
        seen.add(case_id)

        adapter = str(case.get("adapter") or "").strip()
        if adapter not in ALLOWED_S1_ADAPTERS:
            raise RuntimeError(f"지원하지 않는 S1 adapter: {adapter!r}")

    return data


def available_gpu_ids(gpu_ids, default_gpu: int) -> list[int]:
    values = sorted({int(item) for item in gpu_ids})
    return values or [int(default_gpu)]


def preprocess_gpu_string(gpu_ids: list[int]) -> str:
    return "-".join(str(item) for item in gpu_ids)


def case_gpu(case: dict, index: int, gpu_ids: list[int]) -> str:
    raw = str(case.get("gpu") or "auto").strip().lower()
    if raw == "auto":
        return str(gpu_ids[index % len(gpu_ids)])

    try:
        requested = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"잘못된 개발 실험 GPU 설정: {raw!r}") from exc

    if requested not in gpu_ids:
        raise RuntimeError(
            f"preset GPU {requested}를 사용할 수 없습니다. "
            f"available={gpu_ids}"
        )
    return str(requested)


def clear_s1_training_state(
    experiment_root: Path,
    experiment_name: str,
    version: str,
    weight_root: Path,
) -> None:
    """Force a dev case to start from pretrained S1 instead of resuming."""
    experiment_dir = Path(experiment_root) / experiment_name
    logs_dir = experiment_dir / f"logs_s1_{version}"
    if logs_dir.exists():
        shutil.rmtree(logs_dir)

    resume_marker = experiment_dir / f"training_resume_s1_{version}.json"
    if resume_marker.exists():
        resume_marker.unlink()

    weight_root = Path(weight_root)
    if weight_root.is_dir():
        for checkpoint in weight_root.glob(f"{experiment_name}-e*.ckpt"):
            checkpoint.unlink()


def snapshot_s1_checkpoint(
    weight_root: Path,
    experiment_name: str,
    preset_id: str,
    case_id: str,
) -> Path:
    weight_root = Path(weight_root)
    source = latest_gpt_checkpoint_for_experiment(
        weight_root,
        experiment_name,
    )
    if source is None or not source.is_file():
        raise RuntimeError(
            "GPT 학습은 끝났지만 결과 체크포인트를 찾지 못했습니다: "
            f"{experiment_name}"
        )

    suffix = source.name[len(experiment_name) :]
    destination = weight_root / (
        f"{experiment_name}__{slug(preset_id)}__{slug(case_id)}{suffix}"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    shutil.copy2(source, destination)
    return destination


def output_root(gpt_weight_root: Path) -> Path:
    configured = os.environ.get("GPT_SOVITS_OUTPUT_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(gpt_weight_root).resolve().parent


def write_manifest(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
