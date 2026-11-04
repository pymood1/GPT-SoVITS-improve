#!/usr/bin/env python3
"""Source-level regression checks for training/API guard fixes."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    text = (ROOT / path).read_text(encoding="utf-8")
    ast.parse(text, filename=path)
    return text


def main() -> int:
    s1_data = read("GPT_SoVITS/AR/data/dataset.py")
    s2_data = read("GPT_SoVITS/module/data_utils.py")
    lightning = read("GPT_SoVITS/AR/models/t2s_lightning_module.py")
    api = read("api_v2.py")
    streaming = read("GPT_SoVITS/TTS_infer_pack/streaming.py")
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")

    if 'if leng == 0:' not in s1_data:
        raise AssertionError("S1 zero-sample guard missing")
    if s2_data.count('if leng == 0:') < 4:
        raise AssertionError("Not all S2 loaders guard zero matching samples")
    if s2_data.count('Too few usable S2 training samples') < 4:
        raise AssertionError("Not all S2 loaders guard post-filter undersized data")

    if 'batch_idx > 0 and batch_idx % 4 == 0' in lightning:
        raise AssertionError("Old off-by-one accumulation condition still present")
    for marker in (
        'self._grad_accum_count += 1',
        'self._grad_accum_count >= self.grad_accum_steps',
        'def on_train_epoch_end(self):',
    ):
        if marker not in lightning:
            raise AssertionError(f"Gradient accumulation marker missing: {marker}")

    if 'status_code=501' not in api or 'gpt_streaming is temporarily disabled' not in api:
        raise AssertionError("API does not explicitly reject unwired gpt_streaming")
    if 'validate_streaming_options' in api:
        raise AssertionError("API still advertises/validates unwired semantic streaming")
    if 'not wired into API v2 yet' not in streaming:
        raise AssertionError("Semantic streaming module is not marked experimental")

    for requirement in (
        "fastapi[standard]>=0.115.2,<0.116",
        "starlette>=0.40,<0.47",
    ):
        if requirement not in requirements:
            raise AssertionError(
                f"Gradio 4 compatibility requirement missing: {requirement}"
            )

    print("[OK] training/API guard checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
