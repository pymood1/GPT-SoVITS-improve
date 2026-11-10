#!/usr/bin/env python3
"""Build emotion-pair sidecar metadata from an existing GPT-SoVITS dataset.

The tool deliberately starts with the most conservative automatic pairing rule:
same speaker + same language + exact normalized transcript. It predicts an
utterance-level emotion for each audio file, keeps only confident supported
labels for training, reserves a 3-10 second reference per usable speaker, and
writes explicit pair provenance for the existing emotion dataset builder.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    from tools.emotion_dataset_builder import (
        canonical_text,
        parse_list_line,
        resolve_audio_path,
        stable_hash,
    )
except ModuleNotFoundError:
    from emotion_dataset_builder import (
        canonical_text,
        parse_list_line,
        resolve_audio_path,
        stable_hash,
    )


DEFAULT_MODEL = "iic/emotion2vec_plus_base"
DEFAULT_EMOTIONS = ("neutral", "happy", "sad", "angry")


class AutoEmotionError(RuntimeError):
    pass


def normalize_emotion_label(label: str) -> str:
    value = str(label or "").strip().lower()

    aliases = (
        ("angry", ("angry", "生气")),
        ("disgusted", ("disgusted", "disgust", "厌恶")),
        ("fearful", ("fearful", "fear", "恐惧")),
        ("happy", ("happy", "开心")),
        ("neutral", ("neutral", "中立")),
        ("other", ("other", "其他")),
        ("sad", ("sad", "难过")),
        ("surprised", ("surprised", "surprise", "吃惊")),
        ("unknown", ("unknown", "<unk>", "unk")),
    )

    for normalized, names in aliases:
        if any(name in value for name in names):
            return normalized

    return value or "unknown"


def audio_duration_seconds(path: Path) -> float | None:
    try:
        import soundfile as sf

        info = sf.info(str(path))
        if not info.samplerate:
            return None
        return float(info.frames) / float(info.samplerate)
    except Exception:
        return None


def prediction_cache_key(path: Path, model_name: str) -> str:
    stat = path.stat()
    return stable_hash(
        model_name,
        str(path),
        str(stat.st_size),
        str(stat.st_mtime_ns),
    )


def load_prediction_cache(path: Path) -> dict[str, dict[str, Any]]:
    cached: dict[str, dict[str, Any]] = {}

    if not path.is_file():
        return cached

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            key = str(row.get("cache_key") or "")
            if key:
                cached[key] = row

    return cached


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            )


def parse_prediction(result: Any) -> tuple[str, float, dict[str, float]]:
    if isinstance(result, list):
        if not result:
            raise AutoEmotionError("emotion model returned an empty result")
        result = result[0]

    if not isinstance(result, dict):
        raise AutoEmotionError(
            f"unexpected emotion result type: {type(result)!r}"
        )

    labels = result.get("labels") or []
    scores = result.get("scores") or []

    if not labels or len(labels) != len(scores):
        raise AutoEmotionError(
            "emotion result does not contain aligned labels/scores"
        )

    normalized_scores: dict[str, float] = {}

    for raw_label, raw_score in zip(labels, scores):
        label = normalize_emotion_label(raw_label)
        score = float(raw_score)
        normalized_scores[label] = max(
            score,
            normalized_scores.get(label, 0.0),
        )

    if not normalized_scores:
        raise AutoEmotionError("emotion result contains no usable scores")

    label, confidence = max(
        normalized_scores.items(),
        key=lambda item: item[1],
    )
    return label, float(confidence), normalized_scores


def load_dataset_rows(
    list_path: Path,
    wav_root: Path | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    with list_path.open("r", encoding="utf-8-sig") as handle:
        lines = [
            line.rstrip("\r\n")
            for line in handle
            if line.strip()
        ]

    for line_number, line in enumerate(lines, 1):
        raw_wav, speaker, language, text = parse_list_line(
            line,
            line_number,
        )
        resolved = resolve_audio_path(
            raw_wav,
            list_path,
            wav_root,
        )

        rows.append(
            {
                "index": len(rows),
                "raw_wav": raw_wav,
                "resolved_path": resolved,
                "speaker": speaker,
                "language": language,
                "text": text,
                "canonical_text": canonical_text(text),
                "duration": audio_duration_seconds(resolved),
            }
        )

    if not rows:
        raise AutoEmotionError("the GPT-SoVITS .list file is empty")

    return rows


def load_emotion_model(model_name: str, device: str):
    try:
        from funasr import AutoModel
    except ImportError as exc:
        raise AutoEmotionError(
            "FunASR is required for automatic emotion labeling"
        ) from exc

    kwargs: dict[str, Any] = {
        "model": model_name,
        "disable_update": True,
    }

    if device and device != "auto":
        kwargs["device"] = device

    print(
        f"[model] loading {model_name} device={device or 'auto'}",
        flush=True,
    )
    return AutoModel(**kwargs)


def predict_emotions(
    rows: list[dict[str, Any]],
    model_name: str,
    device: str,
    cache_path: Path,
    refresh: bool,
) -> tuple[int, int]:
    cache = {} if refresh else load_prediction_cache(cache_path)
    model = None
    reused = 0
    predicted = 0

    if refresh and cache_path.exists():
        cache_path.unlink()

    for number, row in enumerate(rows, 1):
        audio_path = row["resolved_path"]
        cache_key = prediction_cache_key(audio_path, model_name)
        cached = cache.get(cache_key)

        if cached is not None:
            row["emotion_predicted"] = cached.get(
                "emotion_predicted",
                "unknown",
            )
            row["emotion_predicted_confidence"] = float(
                cached.get("emotion_predicted_confidence", 0.0)
            )
            row["emotion_scores"] = cached.get("emotion_scores", {})
            reused += 1
            print(
                f"[emotion] {number}/{len(rows)} reuse "
                f"{row['emotion_predicted']} "
                f"{row['emotion_predicted_confidence']:.3f} "
                f"{audio_path.name}",
                flush=True,
            )
            continue

        if model is None:
            model = load_emotion_model(model_name, device)

        try:
            result = model.generate(
                str(audio_path),
                granularity="utterance",
                extract_embedding=False,
            )
            label, confidence, scores = parse_prediction(result)
        except Exception as exc:
            label = "unknown"
            confidence = 0.0
            scores = {}
            row["emotion_prediction_error"] = str(exc)
            print(
                f"[emotion] {number}/{len(rows)} ERROR "
                f"{audio_path.name}: {exc}",
                file=sys.stderr,
                flush=True,
            )
        else:
            cache_row = {
                "cache_key": cache_key,
                "resolved_path": str(audio_path),
                "emotion_predicted": label,
                "emotion_predicted_confidence": confidence,
                "emotion_scores": scores,
                "model": model_name,
            }
            append_jsonl(cache_path, cache_row)
            cache[cache_key] = cache_row
            predicted += 1

        row["emotion_predicted"] = label
        row["emotion_predicted_confidence"] = confidence
        row["emotion_scores"] = scores

        print(
            f"[emotion] {number}/{len(rows)} "
            f"{label} {confidence:.3f} {audio_path.name}",
            flush=True,
        )

    return reused, predicted


def prepare_pair_candidates(
    rows: list[dict[str, Any]],
    allowed_emotions: set[str],
    min_confidence: float,
) -> tuple[
    dict[tuple[str, str, str], list[dict[str, Any]]],
    set[tuple[str, str, str]],
]:
    groups: dict[
        tuple[str, str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)

    for row in rows:
        predicted = str(
            row.get("emotion_predicted") or "unknown"
        )
        confidence = float(
            row.get("emotion_predicted_confidence") or 0.0
        )

        accepted = (
            predicted
            if predicted in allowed_emotions
            and confidence >= min_confidence
            else None
        )
        row["accepted_emotion"] = accepted

        key = (
            row["speaker"],
            row["language"],
            row["canonical_text"],
        )
        row["group_key"] = key
        groups[key].append(row)

    potential: set[tuple[str, str, str]] = set()

    for key, group_rows in groups.items():
        emotions = {
            row["accepted_emotion"]
            for row in group_rows
            if row["accepted_emotion"]
        }
        if len(emotions) >= 2:
            potential.add(key)

    return groups, potential


def reference_rank(
    row: dict[str, Any],
    inside_potential_group: bool,
) -> tuple[int, int, float, float]:
    emotion = row.get("accepted_emotion")
    confidence = float(
        row.get("emotion_predicted_confidence") or 0.0
    )
    duration = float(row.get("duration") or 0.0)

    group_penalty = 1 if inside_potential_group else 0

    if emotion == "neutral":
        emotion_priority = 0
    elif emotion is None:
        emotion_priority = 1
    else:
        emotion_priority = 2

    return (
        group_penalty,
        emotion_priority,
        abs(duration - 6.0),
        -confidence,
    )


def group_emotions_after_removal(
    group_rows: list[dict[str, Any]],
    removed_index: int | None,
) -> set[str]:
    return {
        row["accepted_emotion"]
        for row in group_rows
        if row["accepted_emotion"]
        and row["index"] != removed_index
    }


def choose_references(
    rows: list[dict[str, Any]],
    groups: dict[
        tuple[str, str, str],
        list[dict[str, Any]],
    ],
    potential_groups: set[tuple[str, str, str]],
    min_duration: float,
    max_duration: float,
) -> tuple[dict[str, int], list[str]]:
    potential_speakers = {
        key[0]
        for key in potential_groups
    }
    references: dict[str, int] = {}
    missing: list[str] = []

    by_speaker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_speaker[row["speaker"]].append(row)

    for speaker in sorted(potential_speakers):
        options: list[dict[str, Any]] = []

        for row in by_speaker[speaker]:
            duration = row.get("duration")
            if (
                duration is None
                or duration < min_duration
                or duration > max_duration
            ):
                continue

            key = row["group_key"]
            if key in potential_groups:
                remaining = group_emotions_after_removal(
                    groups[key],
                    row["index"],
                )
                if len(remaining) < 2:
                    continue

            options.append(row)

        if not options:
            missing.append(speaker)
            continue

        options.sort(
            key=lambda row: reference_rank(
                row,
                row["group_key"] in potential_groups,
            )
        )
        references[speaker] = options[0]["index"]

    return references, missing


def assign_pairs(
    groups: dict[
        tuple[str, str, str],
        list[dict[str, Any]],
    ],
    potential_groups: set[tuple[str, str, str]],
    references: dict[str, int],
) -> dict[tuple[str, str, str], str]:
    assigned: dict[tuple[str, str, str], str] = {}

    for key in sorted(potential_groups):
        speaker, language, text = key
        reference_index = references.get(speaker)

        if reference_index is None:
            continue

        remaining_emotions = group_emotions_after_removal(
            groups[key],
            reference_index,
        )

        if len(remaining_emotions) < 2:
            continue

        pair_id = "auto_exact_" + stable_hash(
            "auto_exact_text",
            speaker,
            language,
            text,
        )[:16]
        assigned[key] = pair_id

    return assigned


def build_sidecar(args: argparse.Namespace) -> dict[str, Any]:
    list_path = Path(args.list).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    wav_root = (
        Path(args.wav_root).expanduser().resolve()
        if args.wav_root
        else None
    )

    if not list_path.is_file():
        raise AutoEmotionError(
            f"GPT-SoVITS .list file not found: {list_path}"
        )

    allowed_emotions = {
        normalize_emotion_label(item)
        for item in str(args.emotions).split(",")
        if item.strip()
    }

    if len(allowed_emotions) < 2:
        raise AutoEmotionError(
            "at least two supported emotions must be enabled"
        )

    rows = load_dataset_rows(list_path, wav_root)

    cache_path = (
        Path(args.cache).expanduser().resolve()
        if args.cache
        else output_path.with_suffix(
            output_path.suffix + ".predictions.jsonl"
        )
    )

    reused, predicted = predict_emotions(
        rows,
        args.model,
        args.device,
        cache_path,
        args.refresh,
    )

    groups, potential_groups = prepare_pair_candidates(
        rows,
        allowed_emotions,
        args.min_confidence,
    )

    references, speakers_without_reference = choose_references(
        rows,
        groups,
        potential_groups,
        args.reference_min_seconds,
        args.reference_max_seconds,
    )

    pair_groups = assign_pairs(
        groups,
        potential_groups,
        references,
    )

    sidecar_rows: list[dict[str, Any]] = []
    paired_samples = 0

    reference_indices = set(references.values())

    for row in rows:
        key = row["group_key"]
        pair_group_id = pair_groups.get(key)
        is_reference = row["index"] in reference_indices
        accepted_emotion = row.get("accepted_emotion")

        if pair_group_id and not is_reference and accepted_emotion:
            pair_source = "auto_exact_text"
            paired_samples += 1
        else:
            pair_group_id = None
            pair_source = "none"

        sidecar_row = {
            "wav": row["raw_wav"],
            "split": "train",
            "emotion_intended": accepted_emotion,
            "emotion_predicted": row.get(
                "emotion_predicted",
                "unknown",
            ),
            "emotion_predicted_confidence": float(
                row.get("emotion_predicted_confidence") or 0.0
            ),
            "pair_group_id": pair_group_id,
            "pair_source": pair_source,
            "transcript_source": "existing_list",
            "transcript_verified": False,
            "auto_emotion_model": args.model,
            "auto_reference_reserved": is_reference,
        }

        if row.get("emotion_prediction_error"):
            sidecar_row["emotion_prediction_error"] = row[
                "emotion_prediction_error"
            ]

        sidecar_rows.append(sidecar_row)

    write_jsonl(output_path, sidecar_rows)

    prediction_counts = Counter(
        str(row.get("emotion_predicted") or "unknown")
        for row in rows
    )
    accepted_counts = Counter(
        str(row["accepted_emotion"])
        for row in rows
        if row.get("accepted_emotion")
    )

    report = {
        "list": str(list_path),
        "output": str(output_path),
        "prediction_cache": str(cache_path),
        "model": args.model,
        "device": args.device,
        "min_confidence": args.min_confidence,
        "allowed_emotions": sorted(allowed_emotions),
        "samples": len(rows),
        "prediction_reused": reused,
        "prediction_new": predicted,
        "prediction_counts": dict(prediction_counts),
        "accepted_counts": dict(accepted_counts),
        "exact_text_groups": len(groups),
        "potential_pair_groups": len(potential_groups),
        "pair_groups": len(pair_groups),
        "paired_samples": paired_samples,
        "reference_reserved": {
            speaker: rows[index]["raw_wav"]
            for speaker, index in references.items()
        },
        "speakers_without_safe_reference": speakers_without_reference,
        "notes": [
            (
                "Automatic pairs require same speaker, same language, "
                "and exact normalized transcript."
            ),
            (
                "Only supported predictions above min_confidence receive "
                "emotion_intended."
            ),
            (
                "A 3-10 second reference is reserved outside each speaker's "
                "usable pair samples."
            ),
            (
                "Low-confidence and unsupported emotion predictions remain "
                "in the sidecar but are not used as emotion pairs."
            ),
        ],
    }

    report_path = output_path.with_suffix(
        output_path.suffix + ".report.json"
    )
    report_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )

    return report


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--list",
        required=True,
        help="Existing GPT-SoVITS .list file",
    )
    parser.add_argument(
        "--wav-root",
        help="Optional audio root for relative .list paths",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output sidecar JSONL path",
    )
    parser.add_argument(
        "--cache",
        help="Optional prediction-cache JSONL path",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="FunASR/ModelScope emotion model",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="FunASR device, for example auto, cuda:0, or cpu",
    )
    parser.add_argument(
        "--emotions",
        default=",".join(DEFAULT_EMOTIONS),
        help="Comma-separated emotions allowed for controller training",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.45,
        help="Minimum top-class probability used as emotion_intended",
    )
    parser.add_argument(
        "--reference-min-seconds",
        type=float,
        default=3.0,
    )
    parser.add_argument(
        "--reference-max-seconds",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Ignore and rebuild the prediction cache",
    )
    return parser


def main() -> int:
    args = make_parser().parse_args()

    if not 0.0 <= args.min_confidence <= 1.0:
        print(
            "[FATAL] --min-confidence must be between 0 and 1",
            file=sys.stderr,
        )
        return 2

    try:
        build_sidecar(args)
    except (
        AutoEmotionError,
        OSError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
