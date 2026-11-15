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
import wave
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


def _wav_duration_seconds(path: Path) -> float | None:
    try:
        with wave.open(str(path), "rb") as handle:
            rate = int(handle.getframerate())
            frames = int(handle.getnframes())
    except (OSError, wave.Error):
        return None
    if rate <= 0:
        return None
    return frames / rate


def _resolve_list_audio(
    audio_value: str,
    list_path: Path,
    wav_root: Path,
) -> Path | None:
    raw = Path(str(audio_value).replace("\\", "/")).expanduser()
    candidates = []
    if raw.is_absolute():
        candidates.extend(
            [
                raw,
                wav_root / raw.name,
            ]
        )
    else:
        candidates.extend(
            [
                list_path.parent / raw,
                wav_root / raw,
                wav_root / raw.name,
            ]
        )

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    matches = list(wav_root.rglob(raw.name))
    if len(matches) == 1 and matches[0].is_file():
        return matches[0].resolve()
    return None


def export_listening_reference(
    list_path: Path,
    wav_root: Path,
    output_dir: Path,
) -> dict:
    """Persist one deterministic inference reference outside temporary data."""

    list_path = Path(list_path).expanduser().resolve()
    wav_root = Path(wav_root).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()

    if not list_path.is_file():
        raise RuntimeError(
            f"추론용 reference를 고를 .list가 없습니다: {list_path}"
        )
    if not wav_root.is_dir():
        raise RuntimeError(
            f"추론용 reference WAV 폴더가 없습니다: {wav_root}"
        )

    rows = []
    for index, raw_line in enumerate(
        list_path.read_text(encoding="utf-8-sig").splitlines()
    ):
        if not raw_line.strip():
            continue
        parts = raw_line.split("|", 3)
        if len(parts) < 4:
            continue

        audio = _resolve_list_audio(
            parts[0],
            list_path,
            wav_root,
        )
        if audio is None:
            continue

        duration = _wav_duration_seconds(audio)
        rows.append(
            {
                "index": index,
                "audio": audio,
                "speaker": parts[1].strip(),
                "language": parts[2].strip(),
                "text": parts[3].strip(),
                "duration_seconds": duration,
            }
        )

    if not rows:
        raise RuntimeError(
            "자동 생성된 .list에서 추론용 reference WAV를 찾지 못했습니다."
        )

    def score(item):
        duration = item["duration_seconds"]
        text = item["text"]
        if duration is None:
            band = 3
            distance = 999.0
        elif 3.0 <= duration <= 10.0:
            band = 0
            distance = abs(duration - 6.0)
        elif 1.5 <= duration <= 15.0:
            band = 1
            distance = abs(duration - 6.0)
        else:
            band = 2
            distance = abs(duration - 6.0)

        text_penalty = 0 if 2 <= len(text) <= 120 else 1
        return (
            band,
            text_penalty,
            distance,
            item["index"],
        )

    chosen = min(rows, key=score)
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = chosen["audio"].suffix.lower() or ".wav"
    copied_audio = output_dir / f"reference{suffix}"
    shutil.copy2(chosen["audio"], copied_audio)

    text_path = output_dir / "reference.txt"
    text_path.write_text(
        chosen["text"] + "\n",
        encoding="utf-8",
    )

    payload = {
        "source_audio": str(chosen["audio"]),
        "audio": str(copied_audio),
        "text_file": str(text_path),
        "text": chosen["text"],
        "language": chosen["language"],
        "speaker": chosen["speaker"],
        "duration_seconds": chosen["duration_seconds"],
        "selection": "prefer_3_to_10_seconds_closest_to_6",
    }
    (output_dir / "reference.json").write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return payload


def controller_for_gpt(
    root: Path,
    gpt_checkpoint: Path,
) -> Path | None:
    """Return the dev controller paired with one generated GPT checkpoint."""
    root = Path(root)
    try:
        target = Path(gpt_checkpoint).expanduser().resolve()
    except OSError:
        return None

    manifest_root = root / "dev_experiments"
    if not manifest_root.is_dir():
        return None

    for manifest_path in manifest_root.glob("*/manifest.json"):
        try:
            payload = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            continue

        for item in payload.get("results") or []:
            try:
                candidate = Path(
                    str(item.get("gpt_checkpoint") or "")
                ).expanduser().resolve()
            except OSError:
                continue
            if candidate != target:
                continue

            controller = str(item.get("controller") or "").strip()
            if not controller:
                return None
            controller_path = Path(controller).expanduser().resolve()
            if controller_path.is_file():
                return controller_path
            return None

    return None

def controller_bypass_for_gpt(
    root: Path,
    gpt_checkpoint: Path,
) -> bool:
    """Allow controllerless inference only for explicit dev acoustic ablations."""

    root = Path(root)
    try:
        target = Path(gpt_checkpoint).expanduser().resolve()
    except OSError:
        return False

    manifest_root = root / "dev_experiments"
    if not manifest_root.is_dir():
        return False

    for manifest_path in manifest_root.glob("*/manifest.json"):
        try:
            payload = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            continue

        if not bool(payload.get("acoustic_ablation", False)):
            continue

        for item in payload.get("results") or []:
            try:
                candidate = Path(
                    str(item.get("gpt_checkpoint") or "")
                ).expanduser().resolve()
            except OSError:
                continue

            if candidate != target:
                continue

            return not bool(item.get("controller_enabled", True))

    return False


def write_manifest(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
