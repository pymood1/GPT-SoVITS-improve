"""Small orchestration helpers for the V5 emotion-controller training UI.

This module does not implement training itself. It only connects the existing
canonical dataset, V5 preprocessing, reference-prompt export, and controller
trainer so the WebUI can run the pipeline without duplicating model logic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import yaml


class EmotionPipelineError(RuntimeError):
    pass


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EmotionPipelineError(
                    f"{path}:{line_number}: invalid JSON"
                ) from exc

            if not isinstance(row, dict):
                raise EmotionPipelineError(
                    f"{path}:{line_number}: expected a JSON object"
                )

            rows.append(row)

    return rows


def load_dataset_report(dataset_dir: Path) -> dict:
    report_path = dataset_dir / "report.json"

    if not report_path.is_file():
        raise EmotionPipelineError(
            f"dataset report not found: {report_path}"
        )

    report = json.loads(report_path.read_text(encoding="utf-8"))

    if not isinstance(report, dict):
        raise EmotionPipelineError("dataset report must be a JSON object")

    return report


def validate_emotion_dataset(dataset_dir: Path) -> dict:
    metadata_path = dataset_dir / "metadata.jsonl"
    train_list = dataset_dir / "train.list"

    if not metadata_path.is_file():
        raise EmotionPipelineError(
            f"metadata.jsonl not found: {metadata_path}"
        )

    if not train_list.is_file():
        raise EmotionPipelineError(
            f"train.list not found: {train_list}"
        )

    report = load_dataset_report(dataset_dir)

    usable_groups = int(
        report.get("pair_groups_with_multiple_emotions", 0)
    )

    if usable_groups <= 0:
        raise EmotionPipelineError(
            "No emotion pair group contains at least two emotions. "
            "Add explicit pair_group_id, pair_source, and emotion_intended "
            "metadata in the sidecar JSONL."
        )

    return report


def _reference_duration_seconds(audio_path: Path) -> float | None:
    try:
        import soundfile as sf

        info = sf.info(str(audio_path))
    except Exception:
        return None

    if not info.samplerate:
        return None

    return float(info.frames) / float(info.samplerate)


def _reference_candidates(
    rows: Iterable[dict],
    dataset_dir: Path,
) -> dict[str, list[tuple[int, float, str]]]:
    candidates: dict[str, list[tuple[int, float, str]]] = {}

    for row in rows:
        if row.get("split") != "train":
            continue

        if row.get("pair_group_id"):
            continue

        if not row.get("tts_usable", True):
            continue

        speaker_uid = str(row.get("speaker_uid", "")).strip()
        canonical_wav = str(row.get("canonical_wav", "")).strip()

        if not speaker_uid or not canonical_wav:
            continue

        audio_path = dataset_dir / "wav" / canonical_wav
        duration = _reference_duration_seconds(audio_path)

        if duration is None or not 3.0 <= duration <= 10.0:
            continue

        emotion = str(row.get("emotion_intended") or "").lower()
        priority = 0 if emotion in {"", "neutral"} else 1

        candidates.setdefault(speaker_uid, []).append(
            (priority, abs(duration - 6.0), canonical_wav)
        )

    return candidates


def create_reference_map(
    metadata_path: Path,
    output_path: Path,
) -> dict[str, str]:
    metadata_path = metadata_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    dataset_dir = metadata_path.parent

    rows = read_jsonl(metadata_path)

    speakers = {
        str(row.get("speaker_uid", "")).strip()
        for row in rows
        if row.get("split") == "train"
        and str(row.get("speaker_uid", "")).strip()
    }

    candidates = _reference_candidates(rows, dataset_dir)
    reference_map: dict[str, str] = {}

    missing: list[str] = []

    for speaker_uid in sorted(speakers):
        speaker_candidates = candidates.get(speaker_uid, [])

        if not speaker_candidates:
            missing.append(speaker_uid)
            continue

        speaker_candidates.sort()
        reference_map[speaker_uid] = speaker_candidates[0][2]

    if missing:
        raise EmotionPipelineError(
            "No 3-10 second train reference outside emotion pair groups for: "
            + ", ".join(missing)
        )

    if not reference_map:
        raise EmotionPipelineError(
            "No valid train reference could be selected"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            reference_map,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return reference_map


def write_v5_tts_config(
    output_path: Path,
    version: str,
    s1_checkpoint: Path,
    s2_checkpoint: Path,
    bert_dir: Path,
    cnhubert_dir: Path,
    device: str = "cuda",
    is_half: bool = True,
) -> Path:
    if version not in {"v5dev", "v5turbo"}:
        raise EmotionPipelineError(
            f"unsupported V5 version: {version}"
        )

    output_path = output_path.expanduser().resolve()

    config = {
        "custom": {
            "bert_base_path": str(bert_dir.expanduser().resolve()),
            "cnhuhbert_base_path": str(
                cnhubert_dir.expanduser().resolve()
            ),
            "device": device,
            "is_half": bool(is_half),
            "t2s_weights_path": str(
                s1_checkpoint.expanduser().resolve()
            ),
            "version": version,
            "vits_weights_path": str(
                s2_checkpoint.expanduser().resolve()
            ),
        }
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(
            config,
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    return output_path


def latest_feature_build(output_root: Path) -> Path:
    output_root = output_root.expanduser().resolve()

    builds = [
        path
        for path in output_root.glob("build_*")
        if path.is_dir() and (path / "build.json").is_file()
    ]

    if not builds:
        raise EmotionPipelineError(
            f"no valid feature build found under: {output_root}"
        )

    return max(
        builds,
        key=lambda path: (path / "build.json").stat().st_mtime_ns,
    )


def format_dataset_summary(report: dict) -> str:
    samples = int(report.get("samples", 0))
    usable = int(report.get("emotion_pair_usable", 0))
    groups = int(report.get("pair_groups_with_multiple_emotions", 0))
    emotions = report.get("emotions", {})

    return (
        f"samples={samples}, "
        f"emotion_pair_usable={usable}, "
        f"pair_groups={groups}, "
        f"emotions={emotions}"
    )
