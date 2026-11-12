#!/usr/bin/env python3
"""Conservative ingest for existing GPT-SoVITS wav + .list datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from tools.preprocess_contract import (
    build_dataset_fingerprint,
    file_sha256,
)


LANGUAGE_ALIASES = {
    "ZH": "zh", "zh": "zh",
    "JP": "ja", "jp": "ja", "JA": "ja", "ja": "ja",
    "EN": "en", "en": "en", "En": "en",
    "KO": "ko", "Ko": "ko", "ko": "ko",
    "YUE": "yue", "Yue": "yue", "yue": "yue",
}
VALID_SPLITS = {"train", "val", "test"}
VALID_PAIR_SOURCES = {"recording_manifest", "manual", "auto_exact_text", "none", None}


class DatasetError(RuntimeError):
    pass


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(*parts: str) -> str:
    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()


def canonical_text(text: str) -> str:
    # Keep punctuation. Only Unicode/whitespace are normalized for exact grouping.
    return " ".join(unicodedata.normalize("NFC", text).strip().split())


def normalize_language(language: str) -> str:
    if language not in LANGUAGE_ALIASES:
        raise DatasetError(
            f"Unsupported GPT-SoVITS training language label: {language!r}. "
            f"Expected one of {sorted(set(LANGUAGE_ALIASES.values()))} or a known alias."
        )
    return LANGUAGE_ALIASES[language]


def parse_list_line(line: str, line_number: int) -> Tuple[str, str, str, str]:
    parts = line.split("|", 3)
    if len(parts) != 4:
        raise DatasetError(
            f"Line {line_number}: expected 4 pipe-separated fields. "
            "The wav path, speaker and language fields must not contain '|'."
        )
    wav, speaker, language, text = parts
    if not wav.strip() or not speaker.strip() or not language.strip() or not text.strip():
        raise DatasetError(f"Line {line_number}: empty wav/speaker/language/text field")
    return wav.strip(), speaker.strip(), language.strip(), text


def resolve_audio_path(raw_path: str, list_path: Path, wav_root: Optional[Path]) -> Path:
    # Older Colab/Windows-generated .list files may contain paths such as
    # "\\content\\drive\\MyDrive\\...". On POSIX, backslashes are ordinary
    # characters, so normalize them before Path resolution.
    normalized_path = raw_path
    if os.name != "nt" and "\\" in normalized_path:
        normalized_path = normalized_path.replace("\\", "/")
        if normalized_path.startswith("/content/") is False and normalized_path.startswith("content/"):
            normalized_path = "/" + normalized_path
    candidate = Path(normalized_path).expanduser()
    if candidate.is_absolute() and candidate.exists():
        return candidate.resolve()

    if wav_root is not None:
        rooted = (wav_root / candidate).expanduser()
        if rooted.exists():
            return rooted.resolve()
        by_name = wav_root / candidate.name
        if by_name.exists():
            return by_name.resolve()

    relative = list_path.parent / candidate
    if relative.exists():
        return relative.resolve()

    raise DatasetError(f"Audio file not found for .list entry: {raw_path}")


def load_sidecar(path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    if path is None:
        return {}
    result: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            key = item.get("wav") or item.get("original_path")
            if not key:
                raise DatasetError(
                    f"Sidecar line {line_number}: missing 'wav' or 'original_path'"
                )
            key = str(key)
            if key in result:
                raise DatasetError(f"Sidecar duplicate key: {key}")
            result[key] = item
    return result


def portable_basename(path_text: str) -> str:
    return str(path_text).replace("\\", "/").rsplit("/", 1)[-1]


def sidecar_for(
    sidecar: Dict[str, Dict[str, Any]],
    raw_path: str,
    resolved_path: Path,
    *,
    basename_is_unique: bool,
) -> Dict[str, Any]:
    for key in (raw_path, str(resolved_path), resolved_path.as_posix()):
        if key in sidecar:
            return dict(sidecar[key])

    if basename_is_unique:
        for key in (portable_basename(raw_path), resolved_path.name):
            if key in sidecar:
                return dict(sidecar[key])

    return {}


def stage_audio(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        return

    if mode == "copy":
        shutil.copy2(source, destination)
        return
    if mode == "symlink":
        try:
            destination.symlink_to(source)
            return
        except OSError:
            shutil.copy2(source, destination)
            return
    raise DatasetError(f"Unknown stage mode: {mode}")


def validate_pair_metadata(item: Dict[str, Any]) -> None:
    pair_group_id = item.get("pair_group_id")
    pair_source = item.get("pair_source", "none")
    if pair_source not in VALID_PAIR_SOURCES:
        raise DatasetError(f"Invalid pair_source: {pair_source!r}")
    if pair_group_id and pair_source in {None, "none"}:
        raise DatasetError(
            "pair_group_id requires explicit pair_source such as 'manual', "
            "'recording_manifest', or 'auto_exact_text'"
        )
    if pair_source not in {None, "none"} and not pair_group_id:
        raise DatasetError("Explicit pair_source requires pair_group_id")


def check_split_leakage(items: Iterable[Dict[str, Any]]) -> List[str]:
    errors: List[str] = []
    for field in ("source_id", "text_group_id", "pair_group_id"):
        ownership: Dict[str, set] = defaultdict(set)
        for item in items:
            value = item.get(field)
            if value:
                ownership[str(value)].add(item["split"])
        for value, splits in ownership.items():
            if len(splits) > 1:
                errors.append(f"{field}={value} crosses splits: {sorted(splits)}")
    return errors


def unique_canonical_name(file_hash: str, suffix: str, used: Dict[str, str]) -> str:
    suffix = suffix.lower() or ".wav"
    for prefix_len in (16, 20, 24, 32, 64):
        name = f"a_{file_hash[:prefix_len]}{suffix}"
        owner = used.get(name)
        if owner is None or owner == file_hash:
            used[name] = file_hash
            return name
    raise DatasetError("Unable to generate a collision-free canonical audio name")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def build_dataset(args: argparse.Namespace) -> None:
    list_path = Path(args.list).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    wav_root = Path(args.wav_root).expanduser().resolve() if args.wav_root else None
    sidecar_path = (
        Path(args.sidecar).expanduser().resolve()
        if args.sidecar
        else None
    )
    sidecar = load_sidecar(sidecar_path)

    if not list_path.is_file():
        raise DatasetError(f".list file not found: {list_path}")

    if output_dir.exists():
        if not output_dir.is_dir():
            raise DatasetError(
                f"output path is not a directory: {output_dir}"
            )
        if any(output_dir.iterdir()):
            raise DatasetError(
                f"output directory is not empty: {output_dir}"
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    staged_dir = output_dir / "wav"
    staged_dir.mkdir(parents=True, exist_ok=True)

    used_names: Dict[str, str] = {}
    by_audio_hash: Dict[str, Dict[str, Any]] = {}
    records: List[Dict[str, Any]] = []
    canonical_list_lines: Dict[str, List[str]] = {s: [] for s in VALID_SPLITS}

    with list_path.open("r", encoding="utf-8-sig") as handle:
        raw_lines = [line.rstrip("\r\n") for line in handle if line.strip()]

    raw_basenames = [
        portable_basename(parse_list_line(line, index)[0])
        for index, line in enumerate(raw_lines, 1)
    ]
    basename_counts = Counter(raw_basenames)
    ambiguous_sidecar_basenames = sorted(
        name
        for name, count in basename_counts.items()
        if count > 1 and name in sidecar
    )
    if ambiguous_sidecar_basenames:
        raise DatasetError(
            "Basename-only sidecar keys are ambiguous; use the exact .list "
            "path for: "
            + ", ".join(ambiguous_sidecar_basenames[:20])
        )

    for line_number, line in enumerate(raw_lines, 1):
        raw_wav, speaker, raw_language, text = parse_list_line(line, line_number)
        language = normalize_language(raw_language)
        resolved = resolve_audio_path(raw_wav, list_path, wav_root)
        if not resolved.is_file():
            raise DatasetError(f"Not a regular audio file: {resolved}")

        file_hash = sha256_file(resolved)
        metadata = sidecar_for(
            sidecar,
            raw_wav,
            resolved,
            basename_is_unique=(
                basename_counts[portable_basename(raw_wav)] == 1
            ),
        )
        validate_pair_metadata(metadata)

        split = str(metadata.get("split", "train")).strip().lower()
        if split not in VALID_SPLITS:
            raise DatasetError(f"Invalid split {split!r} for {raw_wav}")

        normalized = canonical_text(text)
        speaker_uid = (
            str(metadata.get("speaker_uid") or "").strip()
            or f"{args.dataset_id}:{speaker}"
        )
        text_group_id = stable_hash(speaker_uid, language, normalized)
        source_id = metadata.get("source_id")
        if split in {"val", "test"} and not source_id:
            raise DatasetError(
                f"{raw_wav}: {split} samples require explicit source_id "
                "so evaluation lineage is known"
            )
        pair_group_id = metadata.get("pair_group_id")
        pair_source = metadata.get("pair_source", "none")
        raw_emotion = (
            metadata.get("emotion_intended")
            or metadata.get("emotion")
        )
        emotion_intended = (
            str(raw_emotion).strip().lower()
            if raw_emotion is not None and str(raw_emotion).strip()
            else None
        )
        emotion_predicted = metadata.get("emotion_predicted")

        canonical_name = unique_canonical_name(
            file_hash, resolved.suffix, used_names
        )
        staged_path = staged_dir / canonical_name

        identity = {
            "speaker_uid": speaker_uid,
            "language": language,
            "text": text,
            "split": split,
            "source_id": source_id,
            "session_id": metadata.get("session_id"),
            "pair_group_id": pair_group_id,
            "pair_source": pair_source,
            "emotion_intended": emotion_intended,
        }
        previous = by_audio_hash.get(file_hash)
        if previous is not None:
            if identity != previous["identity"]:
                raise DatasetError(
                    "The same audio bytes appear with conflicting metadata: "
                    f"{resolved} vs {previous['path']}"
                )
            # Exact duplicate: preserve one canonical sample only.
            continue
        by_audio_hash[file_hash] = {
            "identity": identity,
            "path": str(resolved),
        }

        stage_audio(resolved, staged_path, args.stage_mode)

        record = {
            "audio_uid": file_hash,
            "canonical_wav": canonical_name,
            "file_sha256": file_hash,
            "original_path": str(resolved),
            "original_list_path": raw_wav,
            "speaker": speaker,
            "speaker_uid": speaker_uid,
            "language": language,
            "text": text,
            "canonical_text": normalized,
            "text_group_id": text_group_id,
            "source_id": source_id,
            "session_id": metadata.get("session_id"),
            "lineage_known": bool(source_id),
            "pair_group_id": pair_group_id,
            "pair_source": pair_source,
            "emotion_intended": emotion_intended,
            "emotion_predicted": emotion_predicted,
            "emotion_predicted_confidence": metadata.get(
                "emotion_predicted_confidence"
            ),
            "emotion_relative_scores": metadata.get(
                "emotion_relative_scores"
            ),
            "emotion_relative_score": metadata.get(
                "emotion_relative_score"
            ),
            "emotion_relative_confidence": metadata.get(
                "emotion_relative_confidence"
            ),
            "emotion_is_default": bool(
                metadata.get("emotion_is_default", False)
            ),
            "auto_emotion_calibration": metadata.get(
                "auto_emotion_calibration"
            ),
            "transcript_source": metadata.get(
                "transcript_source", "existing_list"
            ),
            "transcript_verified": bool(
                metadata.get("transcript_verified", False)
            ),
            "split": split,
            "tts_usable": True,
            "emotion_pair_usable": bool(
                pair_group_id and emotion_intended
            ),
        }
        records.append(record)
        canonical_list_lines[split].append(
            f"{staged_path.as_posix()}|{speaker}|{language}|{text}"
        )

    pair_consistency: Dict[str, set] = defaultdict(set)
    pair_emotions: Dict[str, set] = defaultdict(set)
    for item in records:
        pair_group_id = item.get("pair_group_id")
        if not pair_group_id:
            continue
        pair_consistency[str(pair_group_id)].add(
            (
                item["speaker_uid"],
                item["language"],
                item["canonical_text"],
            )
        )
        if item.get("emotion_intended"):
            pair_emotions[str(pair_group_id)].add(item["emotion_intended"])

    invalid_pair_groups = [
        pair_group_id
        for pair_group_id, identities in pair_consistency.items()
        if len(identities) != 1
    ]
    if invalid_pair_groups:
        raise DatasetError(
            "pair_group_id must refer to one speaker/language/exact text only: "
            + ", ".join(invalid_pair_groups[:20])
        )

    leakage = check_split_leakage(records)
    if leakage:
        formatted = "\n  - ".join(leakage[:50])
        raise DatasetError(f"Split leakage detected:\n  - {formatted}")

    write_jsonl(output_dir / "metadata.jsonl", records)

    for split, lines in canonical_list_lines.items():
        if lines:
            (output_dir / f"{split}.list").write_text(
                "\n".join(lines) + "\n", encoding="utf-8"
            )

    list_fingerprint = hashlib.sha256()
    for split in sorted(canonical_list_lines):
        for line in sorted(canonical_list_lines[split]):
            list_fingerprint.update(
                split.encode("utf-8")
                + b"\0"
                + line.encode("utf-8")
                + b"\n"
            )

    source_fingerprint = build_dataset_fingerprint(
        list_path,
        wav_root,
    )
    report = {
        "dataset_id": args.dataset_id,
        "source_provenance": {
            "dataset_digest": source_fingerprint["digest"],
            "list_sha256": source_fingerprint["list_sha256"],
            "wav_root": source_fingerprint.get("wav_root", ""),
            "sidecar_sha256": (
                file_sha256(sidecar_path)
                if sidecar_path is not None and sidecar_path.is_file()
                else None
            ),
        },
        "dataset_version": args.dataset_version,
        "samples": len(records),
        "splits": dict(Counter(item["split"] for item in records)),
        "languages": dict(Counter(item["language"] for item in records)),
        "speakers": dict(
            Counter(item["speaker_uid"] for item in records)
        ),
        "emotions": dict(
            Counter(
                item["emotion_intended"] or "unlabeled"
                for item in records
            )
        ),
        "emotion_pair_usable": sum(
            bool(item["emotion_pair_usable"]) for item in records
        ),
        "pair_groups": len(pair_consistency),
        "pair_groups_with_multiple_emotions": sum(
            len(emotions) >= 2 for emotions in pair_emotions.values()
        ),
        "lineage_known": sum(
            bool(item["lineage_known"]) for item in records
        ),
        "list_manifest_sha256": list_fingerprint.hexdigest(),
        "notes": [
            "Existing .list transcripts are preserved verbatim.",
            "Punctuation is preserved for text_group_id.",
            "Emotion controller training consumes per-sample emotion labels; "
            "exact-text pair groups are optional legacy metadata.",
            "Legacy samples without explicit split default to train.",
        ],
    }
    (output_dir / "report.json").write_text(
        json.dumps(
            report, ensure_ascii=False, indent=2, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--list", required=True, help="Existing GPT-SoVITS .list file"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output canonical dataset directory",
    )
    parser.add_argument(
        "--dataset-id",
        required=True,
        help="Stable ID used to namespace speakers",
    )
    parser.add_argument(
        "--dataset-version",
        default="v1",
        help="Human-readable dataset version",
    )
    parser.add_argument(
        "--wav-root",
        help="Optional root used to resolve relative/basename wav paths",
    )
    parser.add_argument(
        "--sidecar",
        help=(
            "Optional JSONL with explicit source/session/emotion/pair/split "
            "metadata"
        ),
    )
    parser.add_argument(
        "--stage-mode",
        choices=("copy", "symlink"),
        default="symlink",
        help=(
            "How unique audio names are staged "
            "(symlink falls back to copy)"
        ),
    )
    return parser


def main() -> int:
    args = make_parser().parse_args()
    output_dir = Path(args.output).expanduser().resolve()
    output_existed = output_dir.exists()
    output_was_empty = (
        output_dir.is_dir()
        and not any(output_dir.iterdir())
        if output_existed
        else False
    )

    try:
        build_dataset(args)
    except (DatasetError, OSError, json.JSONDecodeError) as exc:
        if (
            (not output_existed or output_was_empty)
            and output_dir.is_dir()
        ):
            shutil.rmtree(output_dir, ignore_errors=True)
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
