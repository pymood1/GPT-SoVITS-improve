"""Dataset/preprocessing provenance contracts.

The goal is to prevent silent reuse of stale preprocessing artifacts when the
source list, source audio, model paths, or version changes between runs.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping


CONTRACT_VERSION = 1


def _normalize_path(value: str | os.PathLike[str] | None) -> str:
    if value is None:
        return ""
    return str(Path(value).expanduser().resolve())


def _file_stat_payload(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {
            "path": str(path.resolve()),
            "exists": False,
        }
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "exists": True,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def prepare_training_resume_state(
    marker_path: str | os.PathLike[str],
    resume_paths: list[str | os.PathLike[str]],
    identity: Mapping[str, Any],
) -> bool:
    """Keep optimizer/full checkpoints only when their training identity matches."""

    marker = Path(marker_path)
    payload = {
        "contract_version": CONTRACT_VERSION,
        "identity": dict(identity),
    }
    payload["digest"] = _json_digest(payload)

    current = read_stage_contract(marker)
    if current and current.get("digest") == payload["digest"]:
        return False

    for value in resume_paths:
        path = Path(value)
        if not path.exists():
            continue
        if path.is_dir():
            import shutil

            shutil.rmtree(path)
        else:
            path.unlink()

    write_stage_contract(marker, payload)
    return True


def quick_path_signature(
    path: str | os.PathLike[str],
) -> str:
    """Fast path signature using names, sizes, and mtimes.

    This intentionally avoids hashing multi-GB model contents on every UI
    action while still detecting ordinary in-place model replacements.
    """

    target = Path(path).expanduser().resolve()
    if not target.exists():
        return "missing"

    if target.is_file():
        stat = target.stat()
        payload = {
            "kind": "file",
            "name": target.name,
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }
        return _json_digest(payload)

    entries = []
    for item in sorted(
        (entry for entry in target.rglob("*") if entry.is_file()),
        key=lambda entry: entry.relative_to(target).as_posix(),
    ):
        stat = item.stat()
        entries.append(
            {
                "path": item.relative_to(target).as_posix(),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )
    return _json_digest(
        {
            "kind": "directory",
            "entries": entries,
        }
    )


def file_sha256(
    path: str | os.PathLike[str],
    chunk_size: int = 1024 * 1024,
) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _json_digest(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def build_dataset_fingerprint(
    list_path: str | os.PathLike[str],
    wav_root: str | os.PathLike[str] | None = None,
    *,
    prefer_list_relative_audio: bool = False,
) -> dict[str, Any]:
    """Fingerprint .list content plus referenced audio file identities.

    The default wav_root behavior matches official GPT-SoVITS preprocessing:
    resolve by basename. Canonical emotion ingest can opt into its actual
    relative-path-first resolver.
    """

    list_path = Path(list_path).expanduser().resolve()
    if not list_path.is_file():
        raise FileNotFoundError(list_path)

    wav_root_path = None
    if wav_root:
        wav_root_path = Path(wav_root).expanduser().resolve()

    list_bytes = list_path.read_bytes()
    rows = []
    invalid_lines = []
    basename_counts: dict[str, int] = {}

    for line_number, raw_line in enumerate(
        list_bytes.decode("utf-8-sig").splitlines(),
        1,
    ):
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split("|", 3)
        if len(parts) != 4 or any(not part.strip() for part in parts):
            invalid_lines.append(line_number)
            continue

        wav_field = parts[0].strip()
        speaker = parts[1].strip()
        language = parts[2].strip()
        text = parts[3].strip()
        portable_wav_field = (
            wav_field.replace("\\", "/")
            if os.name != "nt"
            else wav_field
        )
        if (
            prefer_list_relative_audio
            and os.name != "nt"
            and portable_wav_field.startswith("content/")
        ):
            portable_wav_field = "/" + portable_wav_field
        wav_path = Path(portable_wav_field).expanduser()
        if wav_root_path is not None:
            if prefer_list_relative_audio:
                rooted_path = wav_root_path / wav_path
                basename_path = wav_root_path / wav_path.name
                list_relative_path = list_path.parent / wav_path
                if wav_path.is_absolute() and wav_path.exists():
                    wav_path = wav_path.resolve()
                elif rooted_path.exists():
                    wav_path = rooted_path.resolve()
                elif basename_path.exists():
                    wav_path = basename_path.resolve()
                elif list_relative_path.exists():
                    wav_path = list_relative_path.resolve()
                else:
                    wav_path = rooted_path.resolve()
            else:
                wav_path = wav_root_path / wav_path.name
        elif not wav_path.is_absolute():
            wav_path = (list_path.parent / wav_path).resolve()

        basename = wav_path.name
        basename_counts[basename] = basename_counts.get(basename, 0) + 1

        rows.append(
            {
                "line": line_number,
                "wav_field": wav_field,
                "basename": basename,
                "speaker": speaker,
                "language": language,
                "text": text,
                "audio": _file_stat_payload(wav_path),
            }
        )

    payload = {
        "contract_version": CONTRACT_VERSION,
        "list_path": str(list_path),
        "list_sha256": hashlib.sha256(list_bytes).hexdigest(),
        "wav_root": (
            str(wav_root_path)
            if wav_root_path is not None
            else ""
        ),
        "rows": rows,
        "invalid_lines": invalid_lines,
        "duplicate_basenames": sorted(
            name
            for name, count in basename_counts.items()
            if count > 1
        ),
    }
    if prefer_list_relative_audio:
        payload["wav_resolution"] = "list_relative_then_basename"

    text_identity = [
        {
            "basename": row["basename"],
            "language": row["language"],
            "text": row["text"],
        }
        for row in sorted(rows, key=lambda item: item["basename"])
    ]
    audio_identity = [
        {
            "basename": row["basename"],
            "exists": bool(row["audio"].get("exists")),
            "size": row["audio"].get("size"),
            "mtime_ns": row["audio"].get("mtime_ns"),
        }
        for row in sorted(rows, key=lambda item: item["basename"])
    ]
    payload["text_digest"] = _json_digest(
        {
            "rows": text_identity,
            "invalid_lines": invalid_lines,
            "duplicate_basenames": payload["duplicate_basenames"],
        }
    )
    payload["audio_digest"] = _json_digest(
        {
            "rows": audio_identity,
            "invalid_lines": invalid_lines,
            "duplicate_basenames": payload["duplicate_basenames"],
        }
    )
    payload["digest"] = _json_digest(payload)
    return payload


def validate_dataset_fingerprint(
    fingerprint: Mapping[str, Any],
) -> list[str]:
    errors: list[str] = []

    invalid_lines = list(fingerprint.get("invalid_lines") or [])
    if invalid_lines:
        errors.append(
            "invalid .list line(s): "
            + ", ".join(str(value) for value in invalid_lines[:20])
        )

    missing = [
        row
        for row in fingerprint.get("rows", [])
        if not bool(row.get("audio", {}).get("exists"))
    ]
    if missing:
        errors.append(
            "missing audio file(s): "
            + ", ".join(
                str(row.get("wav_field") or row.get("line"))
                for row in missing[:20]
            )
        )

    duplicates = list(fingerprint.get("duplicate_basenames") or [])
    if duplicates:
        errors.append(
            "duplicate basename(s) are unsafe because preprocessing keys "
            "features by basename: "
            + ", ".join(str(value) for value in duplicates[:20])
        )

    if not fingerprint.get("rows"):
        errors.append("dataset has no valid .list rows")

    return errors


def stage_version_key(stage: str, version: str) -> str:
    stage_name = str(stage)
    version_name = str(version)
    if stage_name == "1b":
        if version_name in {"standard", "pro-sv"}:
            return version_name
        return "pro-sv" if "Pro" in version_name else "standard"
    return version_name


def validate_stage_dataset(
    stage: str,
    fingerprint: Mapping[str, Any],
) -> list[str]:
    errors = validate_dataset_fingerprint(fingerprint)
    if str(stage) == "1a":
        errors = [
            error
            for error in errors
            if not error.startswith("missing audio file(s):")
        ]
    return errors


def build_stage_contract(
    *,
    stage: str,
    list_path: str | os.PathLike[str],
    wav_root: str | os.PathLike[str] | None,
    version: str,
    settings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    dataset = build_dataset_fingerprint(list_path, wav_root)
    normalized_settings: dict[str, Any] = {}
    for key, value in dict(settings or {}).items():
        if isinstance(value, Path):
            normalized_settings[str(key)] = _normalize_path(value)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            normalized_settings[str(key)] = value
        else:
            normalized_settings[str(key)] = str(value)

    stage_name = str(stage)
    normalized_version = stage_version_key(stage_name, str(version))
    if stage_name == "1a":
        dependency_kind = "text"
        dependency_digest = dataset["text_digest"]
    elif stage_name in {"1b", "1c"}:
        dependency_kind = "audio"
        dependency_digest = dataset["audio_digest"]
    else:
        dependency_kind = "full"
        dependency_digest = dataset["digest"]

    payload = {
        "contract_version": CONTRACT_VERSION,
        "stage": stage_name,
        "version": normalized_version,
        "dataset": dataset,
        "dependency_kind": dependency_kind,
        "dataset_dependency_digest": dependency_digest,
        "settings": normalized_settings,
    }
    payload["digest"] = _json_digest(
        {
            "contract_version": CONTRACT_VERSION,
            "stage": stage_name,
            "version": normalized_version,
            "dependency_kind": dependency_kind,
            "dataset_dependency_digest": dependency_digest,
            "settings": normalized_settings,
        }
    )
    return payload


def stage_matches_dataset(
    contract: Mapping[str, Any],
    dataset: Mapping[str, Any],
) -> bool:
    kind = str(contract.get("dependency_kind") or "full")
    if kind == "text":
        current = dataset.get("text_digest")
    elif kind == "audio":
        current = dataset.get("audio_digest")
    else:
        current = dataset.get("digest")
    return bool(
        current
        and str(contract.get("dataset_dependency_digest") or "") == str(current)
    )


def read_stage_contract(path: str | os.PathLike[str]) -> dict[str, Any] | None:
    path = Path(path)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def contract_matches(
    path: str | os.PathLike[str],
    expected: Mapping[str, Any],
) -> bool:
    current = read_stage_contract(path)
    return bool(
        current
        and current.get("digest")
        and current.get("digest") == expected.get("digest")
    )


def write_stage_contract(
    path: str | os.PathLike[str],
    contract: Mapping[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temp.write_text(
        json.dumps(
            dict(contract),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temp, path)


def stage_contract_path(
    exp_dir: str | os.PathLike[str],
    stage: str,
) -> Path:
    safe_stage = str(stage).replace("/", "_").replace("\\", "_")
    return Path(exp_dir) / f"preprocess_contract_{safe_stage}.json"


def contracts_share_dataset(
    *contracts: Mapping[str, Any] | None,
) -> bool:
    valid = [contract for contract in contracts if contract]
    if len(valid) < 2:
        return True
    digests = {
        str(contract.get("dataset", {}).get("digest") or "")
        for contract in valid
    }
    return len(digests) == 1 and "" not in digests
