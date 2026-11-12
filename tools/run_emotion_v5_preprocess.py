#!/usr/bin/env python3
"""Run official GPT-SoVITS V5 1A/1B/1C preprocessing in an isolated build dir.

The official preprocessing scripts are executed unchanged. This wrapper only:
- validates the canonical .list,
- fingerprints dataset/model/code inputs,
- chooses an isolated build_<id> output directory,
- merges the single-GPU part files,
- records build.json provenance.

This prevents stale 2/3/4/5/6 preprocessing artifacts from being silently
reused after data/model/code changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "GPT_SoVITS"))


class PreprocessError(RuntimeError):
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


def sha256_tree(path: Path) -> str:
    if not path.is_dir():
        raise PreprocessError(f"model directory not found: {path}")
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise PreprocessError(f"model directory is empty: {path}")
    for item in files:
        relative = item.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(item).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def preprocessing_code_digest() -> str:
    """Hash only code that can change preprocessing outputs.

    UI/CSS/docs commits must not force an expensive feature rebuild.
    """

    targets = [
        ROOT / "tools" / "run_emotion_v5_preprocess.py",
        ROOT / "tools" / "my_utils.py",
        ROOT / "GPT_SoVITS" / "prepare_datasets" / "1-get-text.py",
        ROOT / "GPT_SoVITS" / "prepare_datasets" / "2-get-hubert-wav32k.py",
        ROOT / "GPT_SoVITS" / "prepare_datasets" / "3-get-semantic.py",
        ROOT / "GPT_SoVITS" / "feature_extractor" / "cnhubert.py",
        ROOT / "GPT_SoVITS" / "module" / "models.py",
        ROOT / "GPT_SoVITS" / "process_ckpt.py",
        ROOT / "GPT_SoVITS" / "utils.py",
        ROOT / "GPT_SoVITS" / "text",
    ]
    digest = hashlib.sha256()
    for target in targets:
        if not target.exists():
            raise PreprocessError(
                f"preprocessing code dependency not found: {target}"
            )
        relative = target.relative_to(ROOT).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        if target.is_dir():
            digest.update(sha256_tree(target).encode("ascii"))
        else:
            digest.update(sha256_file(target).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()



def validate_list(path: Path) -> tuple[str, list[str]]:
    if not path.is_file():
        raise PreprocessError(f"train list not found: {path}")
    basenames: Dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig") as handle:
        rows = [line.rstrip("\r\n") for line in handle if line.strip()]
    if not rows:
        raise PreprocessError("train list is empty")

    for line_number, line in enumerate(rows, 1):
        parts = line.split("|", 3)
        if len(parts) != 4:
            raise PreprocessError(
                f"{path}:{line_number}: expected 4 pipe-separated fields"
            )
        wav_path = Path(parts[0]).expanduser()
        if not wav_path.is_absolute():
            wav_path = (path.parent / wav_path).resolve()
        if not wav_path.is_file():
            raise PreprocessError(
                f"{path}:{line_number}: audio not found: {parts[0]}"
            )
        basename = wav_path.name
        previous = basenames.get(basename)
        if previous is not None and previous != str(wav_path):
            raise PreprocessError(
                f"basename collision before official preprocessing: {basename}"
            )
        if previous is not None:
            raise PreprocessError(
                f"duplicate audio basename in train list: {basename}"
            )
        basenames[basename] = str(wav_path)

    audio_digest = hashlib.sha256()
    for basename, wav_path_text in sorted(basenames.items()):
        wav_path = Path(wav_path_text)
        audio_digest.update(basename.encode("utf-8"))
        audio_digest.update(b"\0")
        audio_digest.update(sha256_file(wav_path).encode("ascii"))
        audio_digest.update(b"\n")
    return audio_digest.hexdigest(), sorted(basenames)


def _read_named_rows(
    path: Path,
    *,
    header_prefix: str | None = None,
) -> dict[str, str]:
    if not path.is_file():
        raise PreprocessError(f"preprocessing output missing: {path}")

    rows: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.rstrip("\r\n")
            if not line:
                continue
            if (
                line_number == 1
                and header_prefix is not None
                and line.startswith(header_prefix)
            ):
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                raise PreprocessError(
                    f"{path}:{line_number}: malformed preprocessing row"
                )
            name, payload = parts
            if not name or not payload.strip():
                raise PreprocessError(
                    f"{path}:{line_number}: empty name/payload"
                )
            if name in rows:
                raise PreprocessError(
                    f"{path}:{line_number}: duplicate item {name!r}"
                )
            rows[name] = payload
    return rows


def validate_complete_outputs(
    output_dir: Path,
    expected_names: Iterable[str],
) -> None:
    expected = set(expected_names)
    if not expected:
        raise PreprocessError("no expected preprocessing items")

    text_rows = _read_named_rows(output_dir / "2-name2text.txt")
    semantic_rows = _read_named_rows(
        output_dir / "6-name2semantic.tsv",
        header_prefix="item_name\t",
    )

    problems: list[str] = []

    for label, actual in (
        ("text", set(text_rows)),
        ("semantic", set(semantic_rows)),
    ):
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        if missing:
            problems.append(
                f"{label} missing {len(missing)}: "
                + ", ".join(missing[:10])
            )
        if extra:
            problems.append(
                f"{label} has unexpected {len(extra)}: "
                + ", ".join(extra[:10])
            )

    missing_hubert = sorted(
        name
        for name in expected
        if (
            not (output_dir / "4-cnhubert" / f"{name}.pt").is_file()
            or (output_dir / "4-cnhubert" / f"{name}.pt").stat().st_size == 0
        )
    )
    if missing_hubert:
        problems.append(
            f"CNHuBERT missing {len(missing_hubert)}: "
            + ", ".join(missing_hubert[:10])
        )

    missing_wav32k = sorted(
        name
        for name in expected
        if (
            not (output_dir / "5-wav32k" / name).is_file()
            or (output_dir / "5-wav32k" / name).stat().st_size == 0
        )
    )
    if missing_wav32k:
        problems.append(
            f"32k wav missing {len(missing_wav32k)}: "
            + ", ".join(missing_wav32k[:10])
        )

    if problems:
        raise PreprocessError(
            "official preprocessing produced an incomplete build:\n  - "
            + "\n  - ".join(problems)
        )


def run_script(script: str, env: Dict[str, str]) -> None:
    command = [sys.executable, "-s", script]
    print("+", " ".join(command))
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
    )
    if completed.returncode != 0:
        raise PreprocessError(
            f"official preprocessing failed: {script} "
            f"(exit={completed.returncode})"
        )


def merge_outputs(output_dir: Path) -> None:
    text_part = output_dir / "2-name2text-0.txt"
    text_final = output_dir / "2-name2text.txt"
    if not text_part.is_file():
        raise PreprocessError(
            f"official 1A did not produce {text_part}"
        )
    if text_final.exists():
        raise PreprocessError(
            f"unexpected pre-existing final text feature: {text_final}"
        )
    text_part.replace(text_final)

    semantic_part = output_dir / "6-name2semantic-0.tsv"
    semantic_final = output_dir / "6-name2semantic.tsv"
    if not semantic_part.is_file():
        raise PreprocessError(
            f"official 1C did not produce {semantic_part}"
        )
    if semantic_final.exists():
        raise PreprocessError(
            f"unexpected pre-existing final semantic feature: {semantic_final}"
        )
    content = semantic_part.read_text(encoding="utf-8").strip("\n")
    semantic_final.write_text(
        "item_name\tsemantic_audio\n"
        + (content + "\n" if content else ""),
        encoding="utf-8",
    )
    semantic_part.unlink()


def expected_outputs(output_dir: Path) -> Iterable[Path]:
    return (
        output_dir / "2-name2text.txt",
        output_dir / "3-bert",
        output_dir / "4-cnhubert",
        output_dir / "5-wav32k",
        output_dir / "6-name2semantic.tsv",
        output_dir / "build.json",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-list", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--bert-dir", required=True)
    parser.add_argument("--cnhubert-dir", required=True)
    parser.add_argument("--pretrained-s2g", required=True)
    parser.add_argument(
        "--s1-checkpoint",
        help=(
            "Exact S1 checkpoint whose text-feature contract should be "
            "used. Recommended for controller preprocessing."
        ),
    )
    parser.add_argument(
        "--s2-config",
        default="GPT_SoVITS/configs/s2.json",
    )
    parser.add_argument(
        "--version",
        choices=("v5dev", "v5turbo"),
        default="v5dev",
    )
    parser.add_argument("--gpu", default="0")
    parser.add_argument(
        "--full-precision",
        action="store_true",
        help="Disable the official half-precision preprocessing path",
    )
    args = parser.parse_args()

    train_list = Path(args.train_list).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    bert_dir = Path(args.bert_dir).expanduser().resolve()
    cnhubert_dir = Path(args.cnhubert_dir).expanduser().resolve()
    s2_path = Path(args.pretrained_s2g).expanduser().resolve()
    s1_path = (
        Path(args.s1_checkpoint).expanduser().resolve()
        if args.s1_checkpoint
        else None
    )
    s2_config = Path(args.s2_config)
    if not s2_config.is_absolute():
        s2_config = (ROOT / s2_config).resolve()

    print("[fingerprint] referenced audio files")
    audio_manifest_hash, expected_names = validate_list(train_list)
    if not s2_path.is_file():
        raise PreprocessError(f"S2 checkpoint not found: {s2_path}")
    if not s2_config.is_file():
        raise PreprocessError(f"S2 config not found: {s2_config}")

    if s1_path is not None:
        import torch

        from text.multilingual_bert import (
            ALIGNMENT_VERSION,
            checkpoint_multilingual_bert_config,
        )

        if not s1_path.is_file():
            raise PreprocessError(
                f"S1 checkpoint not found: {s1_path}"
            )
        print("[fingerprint] S1 text feature contract")
        s1_payload = torch.load(
            s1_path,
            map_location="cpu",
            weights_only=False,
        )
        if (
            not isinstance(s1_payload, dict)
            or not isinstance(s1_payload.get("config"), dict)
        ):
            raise PreprocessError(
                "S1 checkpoint must contain a config mapping"
            )
        use_multilingual_bert, multilingual_bert_model = (
            checkpoint_multilingual_bert_config(
                s1_payload["config"]
            )
        )
        s1_hash = sha256_file(s1_path)
        del s1_payload
    else:
        use_multilingual_bert = False
        multilingual_bert_model = (
            "google-bert/bert-base-multilingual-cased"
        )
        s1_hash = None

    text_features = {
        "multilingual_bert": use_multilingual_bert,
        "multilingual_bert_alignment": (
            ALIGNMENT_VERSION
            if use_multilingual_bert
            else None
        ),
        "multilingual_bert_model": (
            multilingual_bert_model
            if use_multilingual_bert
            else None
        ),
    }

    print("[fingerprint] train.list")
    list_hash = sha256_file(train_list)
    print("[fingerprint] V5 S2")
    s2_hash = sha256_file(s2_path)
    print("[fingerprint] BERT directory")
    bert_hash = sha256_tree(bert_dir)
    print("[fingerprint] CNHuBERT directory")
    cnhubert_hash = sha256_tree(cnhubert_dir)
    print("[fingerprint] preprocessing code")
    preprocess_code_hash = preprocessing_code_digest()
    s2_config_hash = sha256_file(s2_config)

    identity = {
        "train_list_sha256": list_hash,
        "audio_manifest_sha256": audio_manifest_hash,
        "s2_sha256": s2_hash,
        "s1_sha256": s1_hash,
        "text_features": text_features,
        "bert_sha256": bert_hash,
        "cnhubert_sha256": cnhubert_hash,
        "s2_config_sha256": s2_config_hash,
        "preprocess_code_sha256": preprocess_code_hash,
        "version": args.version,
        "is_half": not args.full_precision,
    }
    identity_bytes = json.dumps(
        identity,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    build_id = hashlib.sha256(identity_bytes).hexdigest()[:16]
    output_dir = output_root / f"build_{build_id}"
    manifest_path = output_dir / "build.json"

    if output_dir.exists():
        recover_reason = None

        if manifest_path.is_file():
            try:
                existing = json.loads(
                    manifest_path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError) as exc:
                recover_reason = (
                    f"invalid build manifest: {type(exc).__name__}: {exc}"
                )
            else:
                if existing.get("identity") != identity:
                    raise PreprocessError(
                        "build identity mismatch in existing directory: "
                        f"{output_dir}"
                    )

                missing = [
                    str(path)
                    for path in expected_outputs(output_dir)
                    if not path.exists()
                ]
                if missing:
                    recover_reason = (
                        "missing outputs: " + ", ".join(missing)
                    )
                else:
                    try:
                        validate_complete_outputs(
                            output_dir,
                            expected_names,
                        )
                    except PreprocessError as exc:
                        recover_reason = str(exc)
                    else:
                        print(
                            f"[OK] exact preprocessing build already exists: "
                            f"{output_dir}"
                        )
                        print(output_dir)
                        return 0
        else:
            recover_reason = "build.json is missing"

        print(
            f"[recover] removing incomplete preprocessing build: "
            f"{recover_reason or output_dir}",
            file=sys.stderr,
        )
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()

    # Official preprocessing scripts import modules as top-level packages
    # (for example: text.cleaner, module.models, utils, tools.my_utils).
    # When they are executed by file path, Python puts only the script
    # directory on sys.path, so explicitly expose both the repository root
    # and GPT_SoVITS package root to subprocesses.
    pythonpath_entries = [
        str(ROOT / "GPT_SoVITS"),
        str(ROOT),
    ]
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)

    env.update(
        {
            "inp_text": str(train_list),
            # Canonical builder emits absolute paths. Empty inp_wav_dir tells
            # the official scripts to use those paths directly.
            "inp_wav_dir": "",
            "exp_name": f"emotion_{build_id}",
            "i_part": "0",
            "all_parts": "1",
            "_CUDA_VISIBLE_DEVICES": str(args.gpu),
            "opt_dir": str(output_dir),
            "bert_pretrained_dir": str(bert_dir),
            "use_multilingual_bert": str(use_multilingual_bert),
            "multilingual_bert_model": multilingual_bert_model,
            "cnhubert_base_dir": str(cnhubert_dir),
            "pretrained_s2G": str(s2_path),
            "s2config_path": str(s2_config),
            "is_half": str(not args.full_precision),
            "version": args.version,
        }
    )

    try:
        run_script(
            "GPT_SoVITS/prepare_datasets/1-get-text.py",
            env,
        )
        run_script(
            "GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py",
            env,
        )
        run_script(
            "GPT_SoVITS/prepare_datasets/3-get-semantic.py",
            env,
        )
        merge_outputs(output_dir)
        validate_complete_outputs(
            output_dir,
            expected_names,
        )

        manifest = {
            "format_version": 1,
            "build_id": build_id,
            "feature_dir": str(output_dir),
            "train_list": str(train_list),
            "pretrained_s2g": str(s2_path),
            "bert_dir": str(bert_dir),
            "cnhubert_dir": str(cnhubert_dir),
            "s2_config": str(s2_config),
            "identity": identity,
        }
        manifest_temp = manifest_path.with_name(
            f".{manifest_path.name}.tmp-{os.getpid()}"
        )
        manifest_temp.write_text(
            json.dumps(
                manifest,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(manifest_temp, manifest_path)
    except Exception:
        # The parent WebUI already retains subprocess logs. Remove partial
        # artifacts so retrying the same dataset does not require manual
        # directory cleanup after a failure or interrupted child process.
        shutil.rmtree(output_dir, ignore_errors=True)
        raise

    print(f"[OK] fresh official V5 preprocessing build: {output_dir}")
    print(f"[OK] manifest: {manifest_path}")
    print(output_dir)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        PreprocessError,
        OSError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        raise SystemExit(2)
