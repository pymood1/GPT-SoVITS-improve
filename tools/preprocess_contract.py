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
) -> dict[str, Any]:
    """Fingerprint .list content plus referenced audio file identities."""

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
        wav_path = Path(wav_field).expanduser()
        if wav_root_path is not None:
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

    payload = {
        "contract_version": CONTRACT_VERSION,
        "stage": str(stage),
        "version": str(version),
        "dataset": dataset,
        "settings": normalized_settings,
    }
    payload["digest"] = _json_digest(payload)
    return payload


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
