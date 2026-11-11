"""Optional multilingual BERT features for non-Chinese GPT-SoVITS text.

Legacy GPT-SoVITS checkpoints were trained with Chinese RoBERTa features for
Chinese and all-zero 1024-D features for other languages. This module provides
an opt-in path for S1 checkpoints that are explicitly fine-tuned with
multilingual features.

The default encoder is multilingual BERT base (768-D). Its contextual token
features are mapped monotonically to phone positions and padded to the existing
1024-D GPT-SoVITS BERT feature slot, so the S1 architecture itself does not
change.
"""

from __future__ import annotations

from typing import Sequence

import torch
from transformers import AutoModel, AutoTokenizer

DEFAULT_MULTILINGUAL_BERT = "google-bert/bert-base-multilingual-cased"
MULTILINGUAL_BERT_LANGUAGES = frozenset({"en", "ko"})
GPT_SOVITS_BERT_DIM = 1024
ALIGNMENT_VERSION = "token_offset_nearest_v1"


def pad_hidden_to_gpt_dimension(
    feature: torch.Tensor,
    output_dim: int = GPT_SOVITS_BERT_DIM,
) -> torch.Tensor:
    """Pad [T,H] multilingual features to GPT-SoVITS' fixed 1024-D slot."""
    if feature.ndim != 2:
        raise ValueError(
            f"multilingual BERT feature must be [T,H], got {tuple(feature.shape)}"
        )
    hidden_dim = int(feature.shape[-1])
    if hidden_dim <= 0:
        raise ValueError("multilingual BERT hidden dimension must be positive")
    if hidden_dim > output_dim:
        raise ValueError(
            "multilingual BERT hidden dimension exceeds GPT-SoVITS BERT slot: "
            f"{hidden_dim} > {output_dim}"
        )
    if hidden_dim == output_dim:
        return feature

    padded = feature.new_zeros((feature.shape[0], output_dim))
    padded[:, :hidden_dim] = feature
    return padded


def align_token_features_to_phones(
    token_features: torch.Tensor,
    offsets: Sequence[Sequence[int]],
    *,
    text_length: int,
    phone_count: int,
) -> torch.Tensor:
    """Map contextual token features monotonically to phone positions.

    Fast-tokenizer offsets keep the mapping tied to text order without requiring
    every language-specific G2P frontend to expose a word2ph table. Each phone
    position is assigned the contextual token whose character-span center is
    nearest to the phone's monotonic text position.
    """
    if phone_count <= 0:
        raise ValueError("phone_count must be positive")
    if token_features.ndim != 2:
        raise ValueError(
            f"token_features must be [T,H], got {tuple(token_features.shape)}"
        )
    if len(offsets) != token_features.shape[0]:
        raise ValueError(
            "offset/token count mismatch: "
            f"{len(offsets)} != {token_features.shape[0]}"
        )

    valid_indices: list[int] = []
    centers: list[float] = []
    for index, pair in enumerate(offsets):
        if len(pair) != 2:
            raise ValueError(f"invalid tokenizer offset at index {index}: {pair!r}")
        start, end = int(pair[0]), int(pair[1])
        if end <= start:
            continue
        valid_indices.append(index)
        centers.append((start + end) / 2.0)

    if not valid_indices:
        raise ValueError("multilingual tokenizer produced no text-bearing tokens")

    span = max(int(text_length), 1)
    selected: list[torch.Tensor] = []
    for phone_index in range(phone_count):
        target = ((phone_index + 0.5) / phone_count) * span
        nearest = min(
            range(len(centers)),
            key=lambda item: abs(centers[item] - target),
        )
        selected.append(token_features[valid_indices[nearest]])

    return torch.stack(selected, dim=0)


class MultilingualBertFeatureExtractor:
    """Frozen multilingual text encoder used only for explicit S1 opt-in."""

    def __init__(
        self,
        model_name_or_path: str = DEFAULT_MULTILINGUAL_BERT,
        *,
        device: torch.device | str = "cpu",
        is_half: bool = False,
    ) -> None:
        self.model_name_or_path = str(model_name_or_path)
        self.device = torch.device(device)
        self.is_half = bool(
            is_half
            and self.device.type == "cuda"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            use_fast=True,
        )
        if not getattr(self.tokenizer, "is_fast", False):
            raise RuntimeError(
                "multilingual BERT requires a fast tokenizer with offset mapping"
            )

        self.model = AutoModel.from_pretrained(
            self.model_name_or_path,
        ).eval()
        hidden_size = int(getattr(self.model.config, "hidden_size", 0))
        if not 0 < hidden_size <= GPT_SOVITS_BERT_DIM:
            raise RuntimeError(
                "unsupported multilingual BERT hidden size: "
                f"{hidden_size}; expected <= {GPT_SOVITS_BERT_DIM}"
            )
        self.hidden_size = hidden_size
        self.set_device(self.device, is_half=self.is_half)

    def set_device(
        self,
        device: torch.device | str,
        *,
        is_half: bool | None = None,
    ) -> None:
        self.device = torch.device(device)
        if is_half is not None:
            self.is_half = bool(
                is_half
                and self.device.type == "cuda"
            )
        dtype = torch.float16 if self.is_half else torch.float32
        self.model = self.model.to(device=self.device, dtype=dtype).eval()

    @torch.no_grad()
    def extract(
        self,
        text: str,
        *,
        phone_count: int,
    ) -> torch.Tensor:
        text = str(text)
        if not text:
            raise ValueError("multilingual BERT text must not be empty")
        if phone_count <= 0:
            raise ValueError("phone_count must be positive")

        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=512,
        )
        offsets = encoded.pop("offset_mapping")[0].tolist()
        inputs = {
            key: value.to(self.device)
            for key, value in encoded.items()
        }
        result = self.model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = result.hidden_states
        if hidden_states is None or len(hidden_states) < 3:
            raise RuntimeError(
                "multilingual BERT did not return enough hidden states"
            )

        # Match GPT-SoVITS Chinese BERT convention: third layer from the end.
        token_features = hidden_states[-3][0].float().cpu()
        token_features = pad_hidden_to_gpt_dimension(token_features)
        phone_features = align_token_features_to_phones(
            token_features,
            offsets,
            text_length=len(text),
            phone_count=phone_count,
        )
        if phone_features.shape != (phone_count, GPT_SOVITS_BERT_DIM):
            raise RuntimeError(
                "unexpected multilingual phone feature shape: "
                f"{tuple(phone_features.shape)}"
            )
        return phone_features.transpose(0, 1).contiguous()


def checkpoint_multilingual_bert_config(config: dict) -> tuple[bool, str]:
    """Read and validate the opt-in feature contract from an S1 config."""
    text_features = config.get("text_features") or {}
    enabled = bool(text_features.get("multilingual_bert", False))
    model_name = str(
        text_features.get(
            "multilingual_bert_model",
            DEFAULT_MULTILINGUAL_BERT,
        )
    ).strip() or DEFAULT_MULTILINGUAL_BERT

    alignment_version = text_features.get(
        "multilingual_bert_alignment",
        ALIGNMENT_VERSION if enabled else None,
    )
    if enabled and alignment_version != ALIGNMENT_VERSION:
        raise ValueError(
            "Unsupported multilingual BERT alignment contract: "
            f"{alignment_version!r}; expected {ALIGNMENT_VERSION!r}"
        )

    return enabled, model_name
