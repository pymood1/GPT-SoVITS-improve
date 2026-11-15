#!/usr/bin/env python3
"""Discover and normalize a Kaggle GPT-SoVITS training dataset.

The Kaggle notebook should not need to know whether a dataset already has a
training.list. This module finds datas/, optionally normalizes an existing list,
and otherwise leaves list creation to the WebUI automatic preprocessing flow.
"""

from __future__ import annotations

import json
import os
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Any

AUDIO_EXTENSIONS = frozenset(
    {
        ".wav",
        ".flac",
        ".mp3",
        ".m4a",
        ".ogg",
        ".opus",
        ".aac",
        ".wma",
    }
)

DEFAULT_KAGGLE_INPUT = Path("/kaggle/input")
DEFAULT_RUNTIME_ROOT = Path("/kaggle/tmp/audio_input")
PREFERRED_DATASET_ROOTS = (
    Path("/kaggle/input/datasets/pymood1/w-voice-file"),
    Path("/kaggle/input/w-voice-file"),
)


class KaggleDatasetError(RuntimeError):
    pass


def audio_files(path: Path) -> list[Path]:
    return [
        item
        for item in sorted(Path(path).rglob("*"))
        if (
            item.is_file()
            and item.suffix.lower() in AUDIO_EXTENSIONS
        )
    ]


def has_audio_files(path: Path) -> bool:
    return bool(audio_files(path))


def _direct_dataset(root: Path) -> tuple[Path, Path] | None:
    root = Path(root)
    if not root.is_dir():
        return None

    if root.name == "datas" and has_audio_files(root):
        return root.parent, root

    datas = root / "datas"
    if datas.is_dir() and has_audio_files(datas):
        return root, datas

    candidates = [
        path
        for path in root.rglob("datas")
        if path.is_dir() and has_audio_files(path)
    ]
    if len(candidates) == 1:
        return candidates[0].parent, candidates[0]
    if len(candidates) > 1:
        preview = ", ".join(str(path) for path in candidates[:5])
        raise KaggleDatasetError(
            "여러 datas/ 후보를 찾았습니다. "
            "GPT_SOVITS_KAGGLE_DATASET_ROOT를 지정하세요: "
            + preview
        )
    return None


def _archive_files(root: Path) -> list[Path]:
    found: list[Path] = []
    for pattern in ("*.zip", "*.tar", "*.tar.gz", "*.tgz"):
        found.extend(Path(root).rglob(pattern))
    return sorted(set(found))


def _safe_destination(base: Path, member_name: str) -> Path:
    base = base.resolve()
    destination = (base / member_name).resolve()
    try:
        destination.relative_to(base)
    except ValueError as exc:
        raise KaggleDatasetError(
            f"압축파일에 안전하지 않은 경로가 있습니다: {member_name}"
        ) from exc
    return destination


def _extract_archive(archive: Path, output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    name = archive.name.lower()
    if name.endswith(".zip"):
        with zipfile.ZipFile(archive) as handle:
            for member in handle.infolist():
                _safe_destination(output_dir, member.filename)
            handle.extractall(output_dir)
        return

    with tarfile.open(archive, "r:*") as handle:
        for member in handle.getmembers():
            _safe_destination(output_dir, member.name)
        handle.extractall(output_dir)


def _candidate_roots(
    explicit_root: str | os.PathLike[str] | None,
    kaggle_input: Path,
) -> list[Path]:
    candidates: list[Path] = []

    env_root = str(
        os.environ.get("GPT_SOVITS_KAGGLE_DATASET_ROOT", "")
    ).strip()
    for value in (explicit_root, env_root):
        if value:
            path = Path(value).expanduser()
            if path not in candidates:
                candidates.append(path)

    for path in PREFERRED_DATASET_ROOTS:
        if path not in candidates:
            candidates.append(path)

    return candidates


def discover_dataset(
    explicit_root: str | os.PathLike[str] | None = None,
    *,
    kaggle_input: str | os.PathLike[str] = DEFAULT_KAGGLE_INPUT,
    runtime_root: str | os.PathLike[str] = DEFAULT_RUNTIME_ROOT,
) -> tuple[Path, Path]:
    kaggle_input = Path(kaggle_input).expanduser()
    runtime_root = Path(runtime_root).expanduser()
    runtime_root.mkdir(parents=True, exist_ok=True)

    for root in _candidate_roots(explicit_root, kaggle_input):
        if not root.exists():
            continue
        direct = _direct_dataset(root)
        if direct is not None:
            return direct

        archives = _archive_files(root)
        if archives:
            extract_root = runtime_root / "extracted"
            _extract_archive(archives[0], extract_root)
            direct = _direct_dataset(extract_root)
            if direct is not None:
                return direct

    discovered = [
        path
        for path in kaggle_input.rglob("datas")
        if path.is_dir() and has_audio_files(path)
    ]
    if len(discovered) == 1:
        return discovered[0].parent, discovered[0]
    if not discovered:
        raise KaggleDatasetError(
            "Kaggle input에서 음성이 들어 있는 datas/를 찾지 못했습니다."
        )

    preview = ", ".join(str(path) for path in discovered[:5])
    raise KaggleDatasetError(
        "Kaggle input에 datas/ 후보가 여러 개 있습니다. "
        "GPT_SOVITS_KAGGLE_DATASET_ROOT를 지정하세요: "
        + preview
    )


def normalize_training_list(
    source_list: Path,
    datas_dir: Path,
    output_list: Path,
) -> Path:
    lines = source_list.read_text(
        encoding="utf-8-sig",
    ).splitlines()
    fixed: list[str] = []
    missing: list[str] = []

    for line in lines:
        if not line.strip():
            continue

        parts = line.split("|")
        if len(parts) < 4:
            raise KaggleDatasetError(
                f"잘못된 training.list 행: {line}"
            )

        old_audio = Path(parts[0].replace("\\", "/"))
        candidate = datas_dir / old_audio.name

        if not candidate.is_file():
            matches = list(datas_dir.rglob(old_audio.name))
            if len(matches) == 1:
                candidate = matches[0]
            else:
                missing.append(old_audio.name)
                continue

        parts[0] = str(candidate.resolve())
        fixed.append("|".join(parts))

    if missing:
        preview = ", ".join(missing[:10])
        raise KaggleDatasetError(
            "training.list에 있지만 datas/에서 찾지 못한 파일이 "
            f"{len(missing)}개 있습니다: {preview}"
        )
    if not fixed:
        raise KaggleDatasetError("training.list가 비어 있습니다.")

    output_list.parent.mkdir(parents=True, exist_ok=True)
    output_list.write_text(
        "\n".join(fixed) + "\n",
        encoding="utf-8",
    )
    return output_list.resolve()


def prepare_kaggle_dataset(
    explicit_root: str | os.PathLike[str] | None = None,
    *,
    kaggle_input: str | os.PathLike[str] = DEFAULT_KAGGLE_INPUT,
    runtime_root: str | os.PathLike[str] = DEFAULT_RUNTIME_ROOT,
) -> dict[str, Any]:
    runtime_root = Path(runtime_root).expanduser().resolve()
    runtime_root.mkdir(parents=True, exist_ok=True)

    source_root, datas_dir = discover_dataset(
        explicit_root,
        kaggle_input=kaggle_input,
        runtime_root=runtime_root,
    )
    source_list = source_root / "training.list"

    training_list = ""
    mode = "auto_list"
    if source_list.is_file():
        training_list = str(
            normalize_training_list(
                source_list,
                datas_dir,
                runtime_root / "training.list",
            )
        )
        mode = "provided_list"

    result = {
        "mode": mode,
        "source_root": str(source_root.resolve()),
        "datas_dir": str(datas_dir.resolve()),
        "training_list": training_list,
        "audio_count": len(audio_files(datas_dir)),
    }

    manifest = runtime_root / "dataset.json"
    manifest.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    result["manifest"] = str(manifest)
    return result
