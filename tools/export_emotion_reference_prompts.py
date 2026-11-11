#!/usr/bin/env python3
"""Export fixed reference semantic prompts exactly as GPT-SoVITS TTS does.

Unlike ordinary 1C preprocessing, TTS._set_prompt_semantic appends 0.3 seconds
of silence before extracting the reference semantic prefix. This helper reuses
that official path and stores only the resulting semantic token tensors.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "GPT_SoVITS"))

from AR.models.emotion_control import sha256_file  # noqa: E402
from TTS_infer_pack.TTS import TTS, TTS_Config  # noqa: E402
from TTS_infer_pack.text_segmentation_method import splits  # noqa: E402
from text.multilingual_bert import ALIGNMENT_VERSION  # noqa: E402


class ReferenceExportError(RuntimeError):
    pass


def load_metadata(path: Path) -> dict:
    result = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            name = item.get("canonical_wav")
            if not name:
                raise ReferenceExportError(
                    f"{path}:{line_number}: missing canonical_wav"
                )
            if name in result:
                raise ReferenceExportError(
                    f"duplicate canonical_wav: {name}"
                )
            result[name] = item
    return result


def resolve_reference_audio(
    metadata_path: Path,
    item: dict,
) -> Path:
    canonical = metadata_path.parent / "wav" / item["canonical_wav"]
    if canonical.exists():
        return canonical.resolve()
    original = Path(item["original_path"]).expanduser()
    if original.exists():
        return original.resolve()
    raise ReferenceExportError(
        f"reference audio not found: {item['canonical_wav']}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tts-config", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--reference-map", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    metadata_path = Path(args.metadata).resolve()
    metadata = load_metadata(metadata_path)
    reference_map = json.loads(
        Path(args.reference_map).read_text(encoding="utf-8")
    )
    if not isinstance(reference_map, dict):
        raise ReferenceExportError(
            "reference map must be a JSON object"
        )

    config = TTS_Config(str(Path(args.tts_config).resolve()))
    if config.version not in {"v5dev", "v5turbo"}:
        raise ReferenceExportError(
            f"TTS config must select V5, got {config.version!r}"
        )
    tts = TTS(config)

    references = {}
    for speaker_uid, name in reference_map.items():
        if name not in metadata:
            raise ReferenceExportError(
                f"reference {name!r} missing from metadata"
            )
        item = metadata[name]
        if item.get("speaker_uid") != speaker_uid:
            raise ReferenceExportError(
                f"{name!r} does not belong to {speaker_uid!r}"
            )
        if item.get("split") != "train":
            raise ReferenceExportError(
                f"{name!r} must be train-split"
            )
        audio_path = resolve_reference_audio(
            metadata_path,
            item,
        )
        tts.set_ref_audio(str(audio_path))
        prompt_semantic = tts.prompt_cache["prompt_semantic"]
        if prompt_semantic is None or prompt_semantic.numel() == 0:
            raise ReferenceExportError(
                f"empty prompt semantic for {name!r}"
            )

        prompt_text = str(item["text"]).strip("\n")
        if not prompt_text:
            raise ReferenceExportError(
                f"empty prompt text for {name!r}"
            )
        prompt_lang = str(item["language"])
        if prompt_text[-1] not in splits:
            prompt_text += "。" if prompt_lang != "en" else "."
        phones, bert_features, norm_text = (
            tts.text_preprocessor.segment_and_extract_feature_for_text(
                prompt_text,
                prompt_lang,
                tts.configs.version,
            )
        )
        if phones is None or len(phones) == 0:
            raise ReferenceExportError(
                f"empty prompt phones for {name!r}"
            )
        if bert_features is None:
            bert_features = torch.zeros(
                1024, len(phones), dtype=torch.float32
            )

        references[name] = {
            "semantic": prompt_semantic.detach().cpu().long(),
            "phones": torch.as_tensor(phones).reshape(-1).cpu().long(),
            "bert": bert_features.detach().cpu().float(),
            "prompt_text": prompt_text,
            "norm_text": norm_text,
        }
        print(
            f"[OK] {speaker_uid}: {name} -> "
            f"{references[name]['semantic'].numel()} semantic tokens, "
            f"{references[name]['phones'].numel()} phones"
        )

    s2_checkpoint = str(
        Path(tts.configs.vits_weights_path).expanduser().resolve()
    )
    if not Path(s2_checkpoint).is_file():
        raise ReferenceExportError(
            f"V5 S2 checkpoint not found: {s2_checkpoint}"
        )
    payload = {
        "format_version": 3,
        "tts_version": config.version,
        "text_features": {
            "multilingual_bert": bool(
                tts.use_multilingual_bert
            ),
            "multilingual_bert_alignment": (
                ALIGNMENT_VERSION
                if tts.use_multilingual_bert
                else None
            ),
            "multilingual_bert_model": (                tts.multilingual_bert_model
                if tts.use_multilingual_bert
                else None
            ),
        },
        "s2_checkpoint": s2_checkpoint,
        "s2_sha256": sha256_file(s2_checkpoint),
        "references": references,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temp_output = output.with_name(
        f".{output.name}.tmp-{os.getpid()}"
    )
    try:
        torch.save(payload, temp_output)
        os.replace(temp_output, output)
    finally:
        if temp_output.exists():
            temp_output.unlink()
    print(f"[OK] saved {output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        ReferenceExportError,
        OSError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        raise SystemExit(2)
