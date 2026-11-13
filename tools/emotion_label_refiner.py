#!/usr/bin/env python3
"""Cross-fitted emotion compatibility critic for pseudo-label validation.

The critic uses frozen emotion2vec+ embeddings to ask whether an accepted
speaker-relative pseudo-label is geometrically compatible with the audio.
It never overwrites emotion_intended. Accepted samples are scored OOF; samples
that were not training seeds are scored by a final critic fitted on all seeds.

This implements Phase C and the gated Phase D described in
ADVERSARIAL_EMOTION_REFINER_DESIGN.md. Critic-only remains the default. The
bounded residual refiner can run only after the critic passes its OOF gate.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.auto_emotion_sidecar import (  # noqa: E402
    DEFAULT_MODEL,
    load_dataset_rows,
    load_emotion_model,
    load_prediction_cache,
    prediction_cache_key,
    prediction_diagnostics,
)
from tools.emotion_dataset_builder import stable_hash  # noqa: E402
from tools.emotion_training_pipeline import (  # noqa: E402
    EmotionPipelineError,
    stratified_group_kfold_indices,
)

NON_TARGET = {"", "neutral", "default", "unlabeled", "unknown", "other"}
EMBEDDING_CACHE_VERSION = 1


class CriticError(RuntimeError):
    pass


@dataclass(frozen=True)
class CriticConfig:
    emotions: tuple[str, ...]
    pca_dim: int = 32
    rank: int = 4
    temperature: float = 0.10
    ranking_margin: float = 0.20
    steps: int = 300
    learning_rate: float = 3e-3
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    seed: int = 1234
    max_folds: int = 5


@dataclass(frozen=True)
class RefinerConfig:
    delta_max: float = 1.0
    steps: int = 300
    learning_rate: float = 2e-3
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    teacher_weight: float = 0.35
    critic_weight: float = 0.50


class ResidualEmotionRefiner(nn.Module):
    """Small zero-effect MLP that can only adjust teacher logits boundedly."""

    def __init__(
        self,
        input_dim: int,
        class_count: int,
        *,
        delta_max: float,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(32, 16),
            nn.SiLU(),
            nn.Linear(16, class_count),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.delta_max = float(delta_max)

    def forward(
        self,
        features: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        delta = self.delta_max * torch.tanh(self.net(features))
        return teacher_logits + delta


class PrototypeMatchCritic(nn.Module):
    """Small residual metric head initialized from class centroids."""

    def __init__(
        self,
        feature_dim: int,
        class_count: int,
        *,
        rank: int,
        temperature: float,
        prototypes: torch.Tensor,
    ) -> None:
        super().__init__()
        if prototypes.shape != (class_count, feature_dim):
            raise CriticError(
                "prototype shape mismatch: "
                f"{tuple(prototypes.shape)} != "
                f"{(class_count, feature_dim)}"
            )
        self.down = nn.Linear(feature_dim, rank, bias=False)
        self.up = nn.Linear(rank, feature_dim, bias=False)
        self.prototypes = nn.Parameter(prototypes.clone())
        self.temperature = float(temperature)

        nn.init.normal_(self.down.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.up.weight)

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        hidden = features + self.up(self.down(features))
        return F.normalize(hidden, dim=-1, eps=1e-6)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.encode(features)
        prototypes = F.normalize(
            self.prototypes,
            dim=-1,
            eps=1e-6,
        )
        return (
            hidden @ prototypes.transpose(0, 1)
        ) / max(self.temperature, 1e-4)


def normalize_label(value: Any) -> str:
    return str(value or "").strip().lower()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def load_sidecar(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise CriticError(f"sidecar not found: {path}")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CriticError(
                    f"{path}:{line_number}: invalid JSON"
                ) from exc
            if not isinstance(row, dict) or not row.get("wav"):
                raise CriticError(
                    f"{path}:{line_number}: expected object with wav"
                )
            rows.append(row)

    if not rows:
        raise CriticError("sidecar is empty")
    return rows


def normalize_review_confidence(value: Any) -> str:
    normalized = normalize_label(value)
    aliases = {
        "high": {"high", "h", "높음", "높다", "3"},
        "medium": {"medium", "mid", "m", "중간", "보통", "2"},
        "low": {"low", "l", "낮음", "낮다", "1"},
    }
    for canonical, values in aliases.items():
        if normalized in values:
            return canonical
    return normalized


def seed_label_for_row(row: dict[str, Any]) -> str:
    override = normalize_label(row.get("_seed_label_override"))
    if override:
        return override
    return normalize_label(row.get("emotion_intended"))


def apply_human_review_seeds(
    rows: list[dict[str, Any]],
    review_path: Path | None,
    *,
    accepted_confidences: set[str],
) -> int:
    """Use reviewed labels as seed anchors without overwriting auto labels."""
    if review_path is None:
        return 0

    review_rows = load_sidecar(review_path)
    review_map = {
        str(row.get("wav") or ""): row
        for row in review_rows
        if str(row.get("wav") or "")
    }
    applied = 0
    for row in rows:
        review = review_map.get(str(row.get("wav") or ""))
        if review is None:
            continue

        human_label = normalize_label(
            review.get("human_emotion")
        )
        confidence = normalize_review_confidence(
            review.get("human_confidence")
        )
        row["emotion_human_label"] = human_label or None
        row["emotion_human_confidence"] = confidence or None
        row["emotion_human_secondary_label"] = (
            normalize_label(
                review.get("human_secondary_emotion")
            )
            or None
        )
        if (
            human_label
            and human_label not in NON_TARGET
            and confidence in accepted_confidences
        ):
            row["_seed_label_override"] = human_label
            row["_seed_source"] = "human_review"
            applied += 1

    return applied


def sidecar_by_wav(
    rows: Sequence[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row["wav"])
        if key in result:
            raise CriticError(f"duplicate sidecar wav key: {key}")
        result[key] = dict(row)
    return result


def prepare_rows(
    *,
    list_path: Path,
    wav_root: Path | None,
    sidecar_path: Path,
    prediction_cache_path: Path | None,
    model_name: str,
) -> list[dict[str, Any]]:
    dataset_rows = load_dataset_rows(list_path, wav_root)
    sidecar_map = sidecar_by_wav(load_sidecar(sidecar_path))
    prediction_cache = (
        load_prediction_cache(prediction_cache_path)
        if prediction_cache_path is not None
        else {}
    )

    result: list[dict[str, Any]] = []
    for dataset_row in dataset_rows:
        raw_wav = str(dataset_row["raw_wav"])
        sidecar = sidecar_map.get(raw_wav)
        if sidecar is None:
            raise CriticError(
                f"sidecar has no row for .list entry: {raw_wav}"
            )

        row = dict(sidecar)
        row["_resolved_path"] = dataset_row["resolved_path"]
        row["_text_group_id"] = stable_hash(
            str(dataset_row["speaker"]),
            str(dataset_row["language"]),
            str(dataset_row["canonical_text"]),
        )

        scores = row.get("emotion_raw_scores")
        if not isinstance(scores, dict) or not scores:
            scores = None
            if prediction_cache:
                key = prediction_cache_key(
                    dataset_row["resolved_path"],
                    model_name,
                )
                cached = prediction_cache.get(key)
                if cached is not None:
                    cached_scores = cached.get("emotion_scores")
                    if isinstance(cached_scores, dict):
                        scores = cached_scores
            if not scores:
                raise CriticError(
                    "raw emotion scores are missing for "
                    f"{raw_wav}; regenerate the sidecar with the current "
                    "auto_emotion_sidecar.py or pass --prediction-cache"
                )
            row["emotion_raw_scores"] = scores

        diagnostics = prediction_diagnostics(scores)
        for key, value in diagnostics.items():
            row.setdefault(key, value)

        result.append(row)

    return result


def load_embedding_cache(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "version": EMBEDDING_CACHE_VERSION,
            "model": None,
            "items": {},
        }

    payload = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(payload, dict):
        raise CriticError("embedding cache must be a mapping")
    if int(payload.get("version", -1)) != EMBEDDING_CACHE_VERSION:
        raise CriticError(
            "embedding cache version mismatch; use a new cache path"
        )
    if not isinstance(payload.get("items"), dict):
        raise CriticError("embedding cache has no items mapping")
    return payload


def save_embedding_cache(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.tmp-{os.getpid()}"
    )
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_embedding(result: Any) -> torch.Tensor:
    if isinstance(result, list):
        if not result:
            raise CriticError("emotion model returned an empty result")
        result = result[0]
    if not isinstance(result, dict):
        raise CriticError(
            f"unexpected embedding result type: {type(result)!r}"
        )

    feats = result.get("feats")
    if feats is None:
        raise CriticError(
            "emotion model returned no feats with extract_embedding=True"
        )

    tensor = torch.as_tensor(feats, dtype=torch.float32)
    if tensor.numel() == 0:
        raise CriticError("emotion embedding is empty")
    if tensor.ndim == 0:
        tensor = tensor.reshape(1)
    elif tensor.ndim > 1:
        tensor = tensor.reshape(-1, tensor.shape[-1]).mean(dim=0)
    if tensor.ndim != 1:
        raise CriticError(
            f"expected pooled 1-D embedding, got {tuple(tensor.shape)}"
        )
    if not bool(torch.isfinite(tensor).all()):
        raise CriticError("emotion embedding contains NaN/Inf")
    return tensor.contiguous()


def ensure_embeddings(
    rows: Sequence[dict[str, Any]],
    *,
    model_name: str,
    device: str,
    cache_path: Path,
) -> torch.Tensor:
    payload = load_embedding_cache(cache_path)
    cache_model = payload.get("model")
    if cache_model not in {None, model_name}:
        raise CriticError(
            f"embedding cache model mismatch: "
            f"{cache_model!r} != {model_name!r}"
        )
    payload["model"] = model_name
    items = payload["items"]

    model = None
    embeddings: list[torch.Tensor] = []
    changed = False

    for number, row in enumerate(rows, 1):
        audio_path = Path(row["_resolved_path"])
        key = prediction_cache_key(audio_path, model_name)
        cached = items.get(key)

        if cached is None:
            if model is None:
                model = load_emotion_model(model_name, device)
            raw = model.generate(
                str(audio_path),
                granularity="utterance",
                extract_embedding=True,
            )
            embedding = parse_embedding(raw)
            items[key] = embedding.to(
                dtype=torch.float16,
                device="cpu",
            )
            changed = True
            print(
                f"[embedding] {number}/{len(rows)} new "
                f"dim={embedding.numel()} {audio_path.name}",
                flush=True,
            )
        else:
            embedding = torch.as_tensor(
                cached,
                dtype=torch.float32,
            ).flatten()

        if embeddings and embedding.numel() != embeddings[0].numel():
            raise CriticError(
                "emotion embedding dimensions are inconsistent: "
                f"{embedding.numel()} != {embeddings[0].numel()}"
            )
        embeddings.append(embedding.float().cpu())

    if changed:
        save_embedding_cache(cache_path, payload)

    return torch.stack(embeddings)


def select_seed_indices(
    rows: Sequence[dict[str, Any]],
    emotions: Sequence[str],
    *,
    max_ood: float,
    max_entropy: float,
    min_raw_margin: float,
    min_relative_score: float,
    min_relative_confidence: float,
    min_relative_gap: float,
    min_relative_z: float,
    min_relative_z_gap: float,
    min_seed_per_class: int,
) -> tuple[list[int], tuple[str, ...]]:
    requested = tuple(emotions)
    requested_set = set(requested)
    provisional: list[int] = []

    for index, row in enumerate(rows):
        label = seed_label_for_row(row)
        if label not in requested_set:
            continue
        if row.get("emotion_prediction_error"):
            continue
        if normalize_label(row.get("_seed_source")) == "human_review":
            provisional.append(index)
            continue
        if normalize_label(row.get("emotion_raw_top1")) in {
            "other",
            "unknown",
        }:
            continue
        if safe_float(row.get("emotion_ood_mass"), 1.0) > max_ood:
            continue
        if safe_float(row.get("emotion_entropy"), 1.0) > max_entropy:
            continue
        if (
            safe_float(row.get("emotion_top1_margin"), 0.0)
            < min_raw_margin
        ):
            continue
        if (
            safe_float(row.get("emotion_relative_score"), 0.0)
            < min_relative_score
        ):
            continue
        if (
            safe_float(
                row.get("emotion_relative_confidence"),
                0.0,
            )
            < min_relative_confidence
        ):
            continue
        if (
            safe_float(row.get("emotion_relative_gap"), 0.0)
            < min_relative_gap
        ):
            continue
        if (
            safe_float(row.get("emotion_relative_z"), 0.0)
            < min_relative_z
        ):
            continue
        if (
            safe_float(row.get("emotion_relative_z_gap"), 0.0)
            < min_relative_z_gap
        ):
            continue
        provisional.append(index)

    counts = Counter(
        seed_label_for_row(rows[index])
        for index in provisional
    )
    active = tuple(
        emotion
        for emotion in requested
        if counts.get(emotion, 0) >= min_seed_per_class
    )
    if len(active) < 2:
        raise CriticError(
            "need at least two classes with "
            f">={min_seed_per_class} critic seeds; got {dict(counts)}"
        )

    active_set = set(active)
    selected = [
        index
        for index in provisional
        if seed_label_for_row(rows[index]) in active_set
    ]
    return selected, active


def fit_pca(
    embeddings: torch.Tensor,
    requested_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if embeddings.ndim != 2:
        raise CriticError("PCA expects a 2-D embedding matrix")

    mean = embeddings.mean(dim=0, keepdim=True)
    centered = embeddings - mean
    actual_dim = min(
        int(requested_dim),
        centered.shape[0] - 1,
        centered.shape[1],
    )
    if actual_dim < 2:
        raise CriticError("not enough seed samples to fit PCA")

    _, _, vh = torch.linalg.svd(
        centered,
        full_matrices=False,
    )
    return mean.squeeze(0), vh[:actual_dim].contiguous()


def transform_pca(
    embeddings: torch.Tensor,
    mean: torch.Tensor,
    components: torch.Tensor,
) -> torch.Tensor:
    reduced = (
        embeddings - mean
    ) @ components.transpose(0, 1)
    return F.normalize(reduced, dim=-1, eps=1e-6)


def initialize_prototypes(
    features: torch.Tensor,
    labels: torch.Tensor,
    class_count: int,
) -> torch.Tensor:
    prototypes = []
    for class_id in range(class_count):
        mask = labels == class_id
        if not bool(mask.any()):
            raise CriticError(
                f"no critic seed for class {class_id}"
            )
        center = features[mask].mean(dim=0)
        prototypes.append(
            F.normalize(center, dim=-1, eps=1e-6)
        )
    return torch.stack(prototypes)


def balanced_class_weights(
    labels: torch.Tensor,
    class_count: int,
) -> torch.Tensor:
    counts = torch.bincount(
        labels,
        minlength=class_count,
    ).float().clamp_min(1.0)
    return (
        float(labels.numel())
        / (class_count * counts)
    )


def hardest_negative_ranking_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    positive = logits.gather(
        1,
        labels[:, None],
    ).squeeze(1)
    negative = logits.clone()
    negative.scatter_(
        1,
        labels[:, None],
        float("-inf"),
    )
    hardest = negative.max(dim=1).values
    return F.relu(
        float(margin) - positive + hardest
    ).mean()


def train_critic(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    config: CriticConfig,
    run_seed: int,
) -> PrototypeMatchCritic:
    class_count = len(config.emotions)
    prototypes = initialize_prototypes(
        features,
        labels,
        class_count,
    )

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(run_seed)
        critic = PrototypeMatchCritic(
            feature_dim=features.shape[1],
            class_count=class_count,
            rank=config.rank,
            temperature=config.temperature,
            prototypes=prototypes,
        )

    weights = balanced_class_weights(
        labels,
        class_count,
    )
    optimizer = torch.optim.AdamW(
        critic.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    critic.train()
    for _ in range(config.steps):
        optimizer.zero_grad(set_to_none=True)
        logits = critic(features)
        loss = (
            F.cross_entropy(
                logits,
                labels,
                weight=weights,
            )
            + 0.25
            * hardest_negative_ranking_loss(
                logits,
                labels,
                config.ranking_margin,
            )
        )
        loss.backward()
        nn.utils.clip_grad_norm_(
            critic.parameters(),
            config.grad_clip,
        )
        optimizer.step()

    critic.eval()
    return critic


def binary_auc(
    scores: torch.Tensor,
    positives: torch.Tensor,
) -> float | None:
    positive = scores[positives]
    negative = scores[~positives]
    if positive.numel() == 0 or negative.numel() == 0:
        return None

    comparisons = positive[:, None] - negative[None, :]
    wins = (comparisons > 0).float().mean()
    ties = (comparisons == 0).float().mean()
    return float((wins + 0.5 * ties).item())


def macro_auc(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_count: int,
) -> float | None:
    probabilities = F.softmax(logits, dim=-1)
    values = []
    for class_id in range(class_count):
        value = binary_auc(
            probabilities[:, class_id],
            labels == class_id,
        )
        if value is not None:
            values.append(value)
    if not values:
        return None
    return sum(values) / len(values)


def macro_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_count: int,
) -> float:
    predictions = logits.argmax(dim=-1)
    values = []
    for class_id in range(class_count):
        mask = labels == class_id
        if bool(mask.any()):
            values.append(
                float(
                    (
                        predictions[mask]
                        == labels[mask]
                    )
                    .float()
                    .mean()
                    .item()
                )
            )
    return sum(values) / max(len(values), 1)


def probability_summary(
    logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    probabilities = F.softmax(logits, dim=-1)
    values, indices = torch.topk(
        probabilities,
        k=min(2, probabilities.shape[1]),
        dim=-1,
    )
    top1 = values[:, 0]
    top2 = (
        values[:, 1]
        if values.shape[1] > 1
        else torch.zeros_like(top1)
    )
    return indices[:, 0], top1, top1 - top2


def seed_labels_and_groups(
    rows: Sequence[dict[str, Any]],
    seed_indices: Sequence[int],
    emotions: Sequence[str],
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    emotion_to_id = {
        emotion: index
        for index, emotion in enumerate(emotions)
    }
    labels = torch.tensor(
        [
            emotion_to_id[seed_label_for_row(rows[index])]
            for index in seed_indices
        ],
        dtype=torch.long,
    )
    group_samples = [
        {
            "name": str(rows[index].get("wav") or index),
            "source_id": rows[index].get("source_id"),
            "text_group_id": (
                rows[index].get("text_group_id")
                or rows[index].get("_text_group_id")
            ),
            "pair_group_id": rows[index].get("pair_group_id"),
        }
        for index in seed_indices
    ]
    return labels, group_samples


def build_refiner_inputs(
    rows: Sequence[dict[str, Any]],
    emotions: Sequence[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return compact evidence features and teacher logits for all rows."""
    features: list[list[float]] = []
    teacher_logits: list[list[float]] = []

    for row in rows:
        raw_scores = row.get("emotion_raw_scores") or {}
        raw_values = [
            max(safe_float(raw_scores.get(emotion), 0.0), 1e-6)
            for emotion in emotions
        ]
        all_score_total = sum(
            max(safe_float(value, 0.0), 0.0)
            for value in raw_scores.values()
        )
        active_raw_total = sum(raw_values)
        conditional = [
            value / max(active_raw_total, 1e-12)
            for value in raw_values
        ]
        neutral_probability = (
            max(
                safe_float(raw_scores.get("neutral"), 0.0),
                0.0,
            )
            / max(all_score_total, 1e-12)
        )
        active_probability_mass = (
            active_raw_total / max(all_score_total, 1e-12)
        )
        teacher_row = [
            math.log(max(value, 1e-6))
            for value in conditional
        ]

        relative = row.get("emotion_relative_scores") or {}
        relative_z = row.get("emotion_relative_z_scores") or {}
        feature_row = list(teacher_row)
        feature_row.extend(
            safe_float(relative.get(emotion), 0.0)
            for emotion in emotions
        )
        feature_row.extend(
            safe_float(relative_z.get(emotion), 0.0)
            for emotion in emotions
        )
        feature_row.extend(
            [
                safe_float(row.get("emotion_entropy"), 1.0),
                safe_float(row.get("emotion_top1_margin"), 0.0),
                safe_float(row.get("emotion_ood_mass"), 1.0),
                float(neutral_probability),
                float(active_probability_mass),
                safe_float(row.get("emotion_relative_gap"), 0.0),
                safe_float(row.get("emotion_relative_z_gap"), 0.0),
                safe_float(
                    row.get("emotion_relative_confidence"),
                    0.0,
                ),
            ]
        )
        features.append(feature_row)
        teacher_logits.append(teacher_row)

    feature_tensor = torch.tensor(features, dtype=torch.float32)
    teacher_tensor = torch.tensor(
        teacher_logits,
        dtype=torch.float32,
    )
    if not bool(torch.isfinite(feature_tensor).all()):
        raise CriticError("refiner features contain NaN/Inf")
    if not bool(torch.isfinite(teacher_tensor).all()):
        raise CriticError("teacher logits contain NaN/Inf")
    return feature_tensor, teacher_tensor


def standardize_features(
    train_features: torch.Tensor,
    other_features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = train_features.mean(dim=0, keepdim=True)
    std = train_features.std(
        dim=0,
        unbiased=False,
        keepdim=True,
    ).clamp_min(1e-3)
    return (
        (train_features - mean) / std,
        (other_features - mean) / std,
        mean.squeeze(0),
        std.squeeze(0),
    )


def js_distribution_loss(
    left_logits: torch.Tensor,
    right_logits: torch.Tensor,
) -> torch.Tensor:
    left = F.softmax(left_logits, dim=-1).clamp_min(1e-8)
    right = F.softmax(right_logits, dim=-1).clamp_min(1e-8)
    middle = 0.5 * (left + right)
    return 0.5 * (
        F.kl_div(
            torch.log(middle),
            left,
            reduction="batchmean",
        )
        + F.kl_div(
            torch.log(middle),
            right,
            reduction="batchmean",
        )
    )


def train_refiner(
    features: torch.Tensor,
    teacher_logits: torch.Tensor,
    critic_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    class_count: int,
    config: RefinerConfig,
    run_seed: int,
) -> ResidualEmotionRefiner:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(run_seed)
        model = ResidualEmotionRefiner(
            input_dim=features.shape[1],
            class_count=class_count,
            delta_max=config.delta_max,
        )

    weights = balanced_class_weights(labels, class_count)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    teacher_probs = F.softmax(
        teacher_logits.detach(),
        dim=-1,
    ).clamp_min(1e-8)
    critic_logits = critic_logits.detach()

    model.train()
    for _ in range(config.steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(features, teacher_logits)
        ce = F.cross_entropy(
            logits,
            labels,
            weight=weights,
        )
        teacher_kl = F.kl_div(
            F.log_softmax(logits, dim=-1),
            teacher_probs,
            reduction="batchmean",
        )
        critic_js = js_distribution_loss(
            logits,
            critic_logits,
        )
        loss = (
            ce
            + config.teacher_weight * teacher_kl
            + config.critic_weight * critic_js
        )
        loss.backward()
        nn.utils.clip_grad_norm_(
            model.parameters(),
            config.grad_clip,
        )
        optimizer.step()

    model.eval()
    return model


def critic_gate_passed(
    metrics: dict[str, Any],
    *,
    min_macro_accuracy: float,
    min_macro_auc: float,
) -> bool:
    macro_accuracy = safe_float(
        metrics.get("oof_macro_accuracy"),
        0.0,
    )
    macro_auc_value = metrics.get("oof_macro_auc")
    macro_auc_score = (
        safe_float(macro_auc_value, 0.0)
        if macro_auc_value is not None
        else 0.0
    )
    return (
        macro_accuracy >= min_macro_accuracy
        and macro_auc_score >= min_macro_auc
    )


def crossfit_refiner(
    rows: Sequence[dict[str, Any]],
    embeddings: torch.Tensor,
    seed_indices: Sequence[int],
    *,
    critic_config: CriticConfig,
    refiner_config: RefinerConfig,
) -> tuple[torch.Tensor, dict[str, Any]]:
    labels, group_samples = seed_labels_and_groups(
        rows,
        seed_indices,
        critic_config.emotions,
    )
    try:
        folds = stratified_group_kfold_indices(
            group_samples,
            labels.tolist(),
            max_folds=critic_config.max_folds,
            seed=critic_config.seed,
        )
    except EmotionPipelineError as exc:
        raise CriticError(str(exc)) from exc

    all_refiner_features, all_teacher_logits = build_refiner_inputs(
        rows,
        critic_config.emotions,
    )
    oof_logits = torch.full(
        (len(seed_indices), len(critic_config.emotions)),
        float("nan"),
        dtype=torch.float32,
    )

    for fold_index, (train_local, val_local) in enumerate(folds):
        train_global = [seed_indices[index] for index in train_local]
        val_global = [seed_indices[index] for index in val_local]
        train_local_tensor = torch.tensor(
            train_local,
            dtype=torch.long,
        )
        val_local_tensor = torch.tensor(
            val_local,
            dtype=torch.long,
        )
        train_labels = labels.index_select(
            0,
            train_local_tensor,
        )

        pca_mean, pca_components = fit_pca(
            embeddings[train_global],
            critic_config.pca_dim,
        )
        train_critic_features = transform_pca(
            embeddings[train_global],
            pca_mean,
            pca_components,
        )
        val_critic_features = transform_pca(
            embeddings[val_global],
            pca_mean,
            pca_components,
        )
        critic = train_critic(
            train_critic_features,
            train_labels,
            config=critic_config,
            run_seed=(
                critic_config.seed
                + 1009 * (fold_index + 1)
            ),
        )
        with torch.no_grad():
            critic_train_logits = critic(
                train_critic_features
            )
            _ = critic(val_critic_features)

        train_refiner_raw = all_refiner_features[train_global]
        val_refiner_raw = all_refiner_features[val_global]
        (
            train_refiner_features,
            val_refiner_features,
            _,
            _,
        ) = standardize_features(
            train_refiner_raw,
            val_refiner_raw,
        )

        model = train_refiner(
            train_refiner_features,
            all_teacher_logits[train_global],
            critic_train_logits,
            train_labels,
            class_count=len(critic_config.emotions),
            config=refiner_config,
            run_seed=(
                critic_config.seed
                + 3001 * (fold_index + 1)
            ),
        )
        with torch.no_grad():
            val_logits = model(
                val_refiner_features,
                all_teacher_logits[val_global],
            )
        oof_logits.index_copy_(
            0,
            val_local_tensor,
            val_logits,
        )

    if not bool(torch.isfinite(oof_logits).all()):
        raise CriticError(
            "refiner cross-fit did not produce complete OOF logits"
        )

    predictions = oof_logits.argmax(dim=-1)
    teacher_seed_logits = all_teacher_logits[list(seed_indices)]
    teacher_macro_accuracy = macro_accuracy(
        teacher_seed_logits,
        labels,
        len(critic_config.emotions),
    )
    teacher_macro_auc = macro_auc(
        teacher_seed_logits,
        labels,
        len(critic_config.emotions),
    )
    refiner_macro_accuracy = macro_accuracy(
        oof_logits,
        labels,
        len(critic_config.emotions),
    )
    refiner_macro_auc = macro_auc(
        oof_logits,
        labels,
        len(critic_config.emotions),
    )
    metrics = {
        "folds": len(folds),
        "seed_samples": len(seed_indices),
        "oof_ce": float(
            F.cross_entropy(oof_logits, labels).item()
        ),
        "oof_accuracy": float(
            (predictions == labels).float().mean().item()
        ),
        "oof_macro_accuracy": refiner_macro_accuracy,
        "oof_macro_auc": refiner_macro_auc,
        "teacher_seed_ce": float(
            F.cross_entropy(
                teacher_seed_logits,
                labels,
            ).item()
        ),
        "teacher_seed_macro_accuracy": teacher_macro_accuracy,
        "teacher_seed_macro_auc": teacher_macro_auc,
        "refiner_minus_teacher_macro_accuracy": (
            float(refiner_macro_accuracy)
            - float(teacher_macro_accuracy)
        ),
        "refiner_minus_teacher_macro_auc": (
            float(refiner_macro_auc) - float(teacher_macro_auc)
            if refiner_macro_auc is not None
            and teacher_macro_auc is not None
            else None
        ),
    }
    return oof_logits, metrics


def fit_final_refiner(
    rows: Sequence[dict[str, Any]],
    seed_indices: Sequence[int],
    final_critic_logits: torch.Tensor,
    *,
    critic_config: CriticConfig,
    refiner_config: RefinerConfig,
) -> dict[str, Any]:
    emotion_to_id = {
        emotion: index
        for index, emotion in enumerate(critic_config.emotions)
    }
    labels = torch.tensor(
        [
            emotion_to_id[seed_label_for_row(rows[index])]
            for index in seed_indices
        ],
        dtype=torch.long,
    )
    all_features, all_teacher_logits = build_refiner_inputs(
        rows,
        critic_config.emotions,
    )
    seed_features = all_features[list(seed_indices)]
    mean = seed_features.mean(dim=0, keepdim=True)
    std = seed_features.std(
        dim=0,
        unbiased=False,
        keepdim=True,
    ).clamp_min(1e-3)
    normalized_all = (all_features - mean) / std

    model = train_refiner(
        normalized_all[list(seed_indices)],
        all_teacher_logits[list(seed_indices)],
        final_critic_logits[list(seed_indices)],
        labels,
        class_count=len(critic_config.emotions),
        config=refiner_config,
        run_seed=critic_config.seed + 77771,
    )
    with torch.no_grad():
        all_logits = model(
            normalized_all,
            all_teacher_logits,
        )
    return {
        "model": model,
        "feature_mean": mean.squeeze(0),
        "feature_std": std.squeeze(0),
        "all_logits": all_logits,
    }


def crossfit_critic(
    rows: Sequence[dict[str, Any]],
    embeddings: torch.Tensor,
    seed_indices: Sequence[int],
    *,
    config: CriticConfig,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    labels, group_samples = seed_labels_and_groups(
        rows,
        seed_indices,
        config.emotions,
    )

    try:
        folds = stratified_group_kfold_indices(
            group_samples,
            labels.tolist(),
            max_folds=config.max_folds,
            seed=config.seed,
        )
    except EmotionPipelineError as exc:
        raise CriticError(str(exc)) from exc

    oof_logits = torch.full(
        (len(seed_indices), len(config.emotions)),
        float("nan"),
        dtype=torch.float32,
    )
    prototype_oof_logits = torch.full_like(
        oof_logits,
        float("nan"),
    )

    for fold_index, (train_local, val_local) in enumerate(folds):
        train_global = [
            seed_indices[index]
            for index in train_local
        ]
        val_global = [
            seed_indices[index]
            for index in val_local
        ]
        train_labels = labels[
            torch.tensor(train_local, dtype=torch.long)
        ]

        pca_mean, pca_components = fit_pca(
            embeddings[train_global],
            config.pca_dim,
        )
        train_features = transform_pca(
            embeddings[train_global],
            pca_mean,
            pca_components,
        )
        val_features = transform_pca(
            embeddings[val_global],
            pca_mean,
            pca_components,
        )

        prototype_centers = initialize_prototypes(
            train_features,
            train_labels,
            len(config.emotions),
        )
        with torch.no_grad():
            prototype_logits = (
                val_features
                @ F.normalize(
                    prototype_centers,
                    dim=-1,
                    eps=1e-6,
                ).transpose(0, 1)
            ) / max(config.temperature, 1e-4)
        prototype_oof_logits.index_copy_(
            0,
            torch.tensor(val_local, dtype=torch.long),
            prototype_logits,
        )

        critic = train_critic(
            train_features,
            train_labels,
            config=config,
            run_seed=config.seed + 1009 * (fold_index + 1),
        )
        with torch.no_grad():
            val_logits = critic(val_features)
        oof_logits.index_copy_(
            0,
            torch.tensor(val_local, dtype=torch.long),
            val_logits,
        )

    if not bool(torch.isfinite(oof_logits).all()):
        raise CriticError(
            "critic cross-fit did not produce complete OOF logits"
        )
    if not bool(torch.isfinite(prototype_oof_logits).all()):
        raise CriticError(
            "prototype baseline did not produce complete OOF logits"
        )

    predictions = oof_logits.argmax(dim=-1)
    metrics = {
        "folds": len(folds),
        "seed_samples": len(seed_indices),
        "oof_ce": float(
            F.cross_entropy(
                oof_logits,
                labels,
            ).item()
        ),
        "oof_accuracy": float(
            (
                predictions == labels
            )
            .float()
            .mean()
            .item()
        ),
        "oof_macro_accuracy": macro_accuracy(
            oof_logits,
            labels,
            len(config.emotions),
        ),
        "oof_macro_auc": macro_auc(
            oof_logits,
            labels,
            len(config.emotions),
        ),
        "prototype_oof_ce": float(
            F.cross_entropy(
                prototype_oof_logits,
                labels,
            ).item()
        ),
        "prototype_oof_macro_accuracy": macro_accuracy(
            prototype_oof_logits,
            labels,
            len(config.emotions),
        ),
        "prototype_oof_macro_auc": macro_auc(
            prototype_oof_logits,
            labels,
            len(config.emotions),
        ),
    }
    metrics["critic_minus_prototype_macro_accuracy"] = (
        float(metrics["oof_macro_accuracy"])
        - float(metrics["prototype_oof_macro_accuracy"])
    )
    prototype_auc = metrics["prototype_oof_macro_auc"]
    critic_auc = metrics["oof_macro_auc"]
    metrics["critic_minus_prototype_macro_auc"] = (
        float(critic_auc) - float(prototype_auc)
        if critic_auc is not None and prototype_auc is not None
        else None
    )
    return oof_logits, labels, metrics


def fit_final_critic(
    embeddings: torch.Tensor,
    seed_indices: Sequence[int],
    rows: Sequence[dict[str, Any]],
    *,
    config: CriticConfig,
) -> dict[str, Any]:
    emotion_to_id = {
        emotion: index
        for index, emotion in enumerate(config.emotions)
    }
    labels = torch.tensor(
        [
            emotion_to_id[seed_label_for_row(rows[index])]
            for index in seed_indices
        ],
        dtype=torch.long,
    )

    pca_mean, pca_components = fit_pca(
        embeddings[list(seed_indices)],
        config.pca_dim,
    )
    all_features = transform_pca(
        embeddings,
        pca_mean,
        pca_components,
    )
    critic = train_critic(
        all_features[list(seed_indices)],
        labels,
        config=config,
        run_seed=config.seed + 99991,
    )
    with torch.no_grad():
        all_logits = critic(all_features)

    return {
        "pca_mean": pca_mean,
        "pca_components": pca_components,
        "critic": critic,
        "all_logits": all_logits,
    }


def prototype_geometry(
    critic: PrototypeMatchCritic,
    emotions: Sequence[str],
) -> dict[str, Any]:
    prototypes = F.normalize(
        critic.prototypes.detach().float().cpu(),
        dim=-1,
        eps=1e-6,
    )
    matrix = prototypes @ prototypes.transpose(0, 1)
    class_count = len(emotions)
    max_off_diagonal = -1.0
    closest_pair: tuple[str, str] | None = None
    rows: dict[str, dict[str, float]] = {}

    for i, emotion_i in enumerate(emotions):
        rows[emotion_i] = {}
        for j, emotion_j in enumerate(emotions):
            value = float(matrix[i, j].item())
            rows[emotion_i][emotion_j] = value
            if i != j and value > max_off_diagonal:
                max_off_diagonal = value
                closest_pair = (emotion_i, emotion_j)

    return {
        "cosine_matrix": rows,
        "max_off_diagonal": (
            max_off_diagonal
            if class_count > 1
            else None
        ),
        "closest_pair": (
            list(closest_pair)
            if closest_pair is not None
            else None
        ),
    }


def apply_critic_outputs(
    rows: Sequence[dict[str, Any]],
    *,
    seed_indices: Sequence[int],
    oof_logits: torch.Tensor,
    final_logits: torch.Tensor,
    emotions: Sequence[str],
    reject_ood: float,
    max_neutral_entropy: float,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    seed_position = {
        global_index: local_index
        for local_index, global_index in enumerate(seed_indices)
    }
    emotion_set = set(emotions)
    output: list[dict[str, Any]] = []
    states: Counter[str] = Counter()

    for index, original in enumerate(rows):
        row = {
            key: value
            for key, value in original.items()
            if not str(key).startswith("_")
        }

        if index in seed_position:
            local = seed_position[index]
            logits = oof_logits[local : local + 1]
            oof = True
        else:
            logits = final_logits[index : index + 1]
            oof = False

        ids, top_probability, margin = probability_summary(logits)
        label = str(emotions[int(ids.item())])
        row["emotion_critic_label"] = label
        row["emotion_critic_max_probability"] = float(
            top_probability.item()
        )
        row["emotion_critic_margin"] = float(margin.item())
        row["emotion_critic_oof"] = bool(oof)

        intended = normalize_label(row.get("emotion_intended"))
        seed_target = (
            seed_label_for_row(original)
            if index in seed_position
            else ""
        )
        raw_top1 = normalize_label(row.get("emotion_raw_top1"))
        ood_mass = safe_float(
            row.get("emotion_ood_mass"),
            1.0,
        )
        entropy = safe_float(
            row.get("emotion_entropy"),
            1.0,
        )
        candidate = None

        if row.get("emotion_prediction_error"):
            state = "failed"
        elif (
            index not in seed_position
            and (
                raw_top1 in {"other", "unknown"}
                or ood_mass >= reject_ood
            )
        ):
            state = "ood"
        elif index in seed_position and seed_target in emotion_set:
            state = (
                "accepted_core"
                if label == seed_target
                else "conflict"
            )
            candidate = (
                seed_target
                if state == "accepted_core"
                else None
            )
        elif intended in emotion_set:
            state = (
                "critic_confirmed_candidate"
                if label == intended
                else "conflict"
            )
            candidate = (
                intended
                if state == "critic_confirmed_candidate"
                else None
            )
        elif intended in {"neutral", "default"}:
            state = (
                "neutral_candidate"
                if (
                    raw_top1 == "neutral"
                    and entropy <= max_neutral_entropy
                )
                else "ambiguous"
            )
        else:
            state = "ambiguous"

        row["emotion_label_state_v3"] = state
        row["emotion_label_quality_v3"] = (
            "high"
            if state == "accepted_core"
            else (
                "candidate"
                if state in {
                    "neutral_candidate",
                    "critic_confirmed_candidate",
                }
                else "reject"
            )
        )
        if state == "neutral_candidate":
            candidate = "neutral"
        row["emotion_v3_candidate"] = candidate
        row["auto_emotion_refiner"] = "prototype_critic_oof_v1"

        states[state] += 1
        output.append(row)

    return output, dict(sorted(states.items()))


def apply_refiner_outputs(
    rows: Sequence[dict[str, Any]],
    *,
    seed_indices: Sequence[int],
    critic_oof_logits: torch.Tensor,
    critic_final_logits: torch.Tensor,
    refiner_oof_logits: torch.Tensor,
    refiner_final_logits: torch.Tensor,
    emotions: Sequence[str],
    reject_ood: float,
    max_neutral_entropy: float,
    min_critic_margin: float,
    min_refiner_margin: float,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    seed_position = {
        global_index: local_index
        for local_index, global_index in enumerate(seed_indices)
    }
    emotion_set = set(emotions)
    output: list[dict[str, Any]] = []
    states: Counter[str] = Counter()

    for index, original in enumerate(rows):
        row = {
            key: value
            for key, value in original.items()
            if not str(key).startswith("_")
        }
        if index in seed_position:
            local = seed_position[index]
            critic_logits = critic_oof_logits[local : local + 1]
            refiner_logits = refiner_oof_logits[local : local + 1]
            oof = True
        else:
            critic_logits = critic_final_logits[index : index + 1]
            refiner_logits = refiner_final_logits[index : index + 1]
            oof = False

        critic_ids, critic_probability, critic_margin = (
            probability_summary(critic_logits)
        )
        refiner_ids, refiner_probability, refiner_margin = (
            probability_summary(refiner_logits)
        )
        critic_label = str(emotions[int(critic_ids.item())])
        refiner_label = str(emotions[int(refiner_ids.item())])

        row["emotion_critic_label"] = critic_label
        row["emotion_critic_max_probability"] = float(
            critic_probability.item()
        )
        row["emotion_critic_margin"] = float(
            critic_margin.item()
        )
        row["emotion_critic_oof"] = bool(oof)
        row["emotion_refiner_label"] = refiner_label
        row["emotion_refiner_max_probability"] = float(
            refiner_probability.item()
        )
        row["emotion_refiner_margin"] = float(
            refiner_margin.item()
        )
        row["emotion_refiner_oof"] = bool(oof)

        intended = normalize_label(row.get("emotion_intended"))
        seed_target = (
            seed_label_for_row(original)
            if index in seed_position
            else ""
        )
        raw_top1 = normalize_label(row.get("emotion_raw_top1"))
        ood_mass = safe_float(
            row.get("emotion_ood_mass"),
            1.0,
        )
        entropy = safe_float(
            row.get("emotion_entropy"),
            1.0,
        )
        consensus = (
            critic_label == refiner_label
            and float(critic_margin.item()) >= min_critic_margin
            and float(refiner_margin.item()) >= min_refiner_margin
        )

        if row.get("emotion_prediction_error"):
            state = "failed"
            candidate = None
        elif (
            index not in seed_position
            and (
                raw_top1 in {"other", "unknown"}
                or ood_mass >= reject_ood
            )
        ):
            state = "ood"
            candidate = None
        elif index in seed_position and seed_target in emotion_set:
            if consensus and critic_label == seed_target:
                state = "accepted_core"
                candidate = seed_target
            else:
                state = "conflict"
                candidate = None
        elif intended in emotion_set:
            if consensus and critic_label == intended:
                state = "refiner_confirmed_candidate"
                candidate = intended
            else:
                state = "conflict"
                candidate = None
        elif intended in {"neutral", "default"}:
            if raw_top1 == "neutral":
                state = (
                    "neutral_candidate"
                    if entropy <= max_neutral_entropy
                    else "ambiguous"
                )
                candidate = (
                    "neutral"
                    if state == "neutral_candidate"
                    else None
                )
            elif consensus and raw_top1 in emotion_set:
                state = "rescued_candidate"
                candidate = refiner_label
            else:
                state = "ambiguous"
                candidate = None
        elif consensus and raw_top1 in emotion_set:
            state = "rescued_candidate"
            candidate = refiner_label
        else:
            state = "ambiguous"
            candidate = None

        row["emotion_label_state_v3"] = state
        row["emotion_label_quality_v3"] = (
            "high"
            if state == "accepted_core"
            else (
                "candidate"
                if state in {
                    "neutral_candidate",
                    "refiner_confirmed_candidate",
                    "rescued_candidate",
                }
                else "reject"
            )
        )
        row["emotion_v3_candidate"] = candidate
        row["auto_emotion_refiner"] = (
            "critic_guided_residual_oof_v1"
        )
        states[state] += 1
        output.append(row)

    return output, dict(sorted(states.items()))


def materialize_training_sidecar(
    rows: Sequence[dict[str, Any]],
    *,
    policy: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Create a derived sidecar without mutating the baseline pseudo-labels."""
    allowed_states = {"accepted_core"}
    if policy in {"core+confirmed", "all-candidates"}:
        allowed_states.update(
            {
                "critic_confirmed_candidate",
                "refiner_confirmed_candidate",
            }
        )
    if policy == "all-candidates":
        allowed_states.add("rescued_candidate")

    output: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for original in rows:
        row = dict(original)
        state = normalize_label(
            row.get("emotion_label_state_v3")
        )
        candidate = normalize_label(
            row.get("emotion_v3_candidate")
        )

        if state in allowed_states and candidate not in NON_TARGET:
            row["emotion_intended"] = candidate
            if "emotion" in row:
                row["emotion"] = candidate
            row["emotion_is_default"] = False
            counts[f"target:{candidate}"] += 1
        elif state == "neutral_candidate":
            row["emotion_intended"] = "neutral"
            if "emotion" in row:
                row["emotion"] = "neutral"
            row["emotion_is_default"] = True
            counts["neutral"] += 1
        else:
            row["emotion_intended"] = None
            if "emotion" in row:
                row["emotion"] = None
            row["emotion_is_default"] = False
            counts[f"excluded:{state or 'unknown'}"] += 1

        row["emotion_training_policy_v3"] = policy
        output.append(row)

    return output, dict(sorted(counts.items()))


def write_jsonl(
    path: Path,
    rows: Sequence[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.tmp-{os.getpid()}"
    )
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def save_artifact(
    path: Path,
    *,
    config: CriticConfig,
    model_name: str,
    final_result: dict[str, Any],
    metrics: dict[str, Any],
    refiner_config: RefinerConfig | None = None,
    refiner_result: dict[str, Any] | None = None,
    refiner_metrics: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    critic = final_result["critic"]
    payload = {
        "format_version": 1,
        "kind": "prototype_critic_oof_v1",
        "teacher_model": model_name,
        "config": asdict(config),
        "pca_mean": final_result["pca_mean"].cpu(),
        "pca_components": final_result["pca_components"].cpu(),
        "critic_state_dict": {
            key: value.detach().cpu()
            for key, value in critic.state_dict().items()
        },
        "metrics": metrics,
    }
    if (
        refiner_config is not None
        and refiner_result is not None
        and refiner_metrics is not None
    ):
        refiner_model = refiner_result["model"]
        payload.update(
            {
                "kind": "critic_guided_residual_oof_v1",
                "refiner_config": asdict(refiner_config),
                "refiner_state_dict": {
                    key: value.detach().cpu()
                    for key, value in refiner_model.state_dict().items()
                },
                "refiner_feature_mean": refiner_result[
                    "feature_mean"
                ].cpu(),
                "refiner_feature_std": refiner_result[
                    "feature_std"
                ].cpu(),
                "refiner_metrics": refiner_metrics,
            }
        )
    temporary = path.with_name(
        f".{path.name}.tmp-{os.getpid()}"
    )
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", required=True)
    parser.add_argument("--sidecar", required=True)
    parser.add_argument(
        "--review",
        help=(
            "Optional review JSONL. High-confidence reviewed non-neutral "
            "labels become seed anchors without overwriting emotion_intended."
        ),
    )
    parser.add_argument(
        "--human-seed-confidences",
        default="high",
        help="Comma-separated human confidence values accepted as seed anchors",
    )
    parser.add_argument("--prediction-cache")
    parser.add_argument("--embedding-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--artifact")
    parser.add_argument(
        "--training-sidecar",
        help=(
            "Optional derived sidecar for Controller training. The baseline "
            "sidecar is never modified."
        ),
    )
    parser.add_argument(
        "--training-policy",
        choices=("core", "core+confirmed", "all-candidates"),
        default="core",
    )
    parser.add_argument("--wav-root")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--emotions",
        default="happy,angry,disgusted,surprised",
    )

    parser.add_argument(
        "--mode",
        choices=("critic", "refiner"),
        default="critic",
        help=(
            "critic is the safe default. refiner is automatically blocked "
            "unless the critic OOF gate passes."
        ),
    )

    parser.add_argument("--pca-dim", type=int, default=32)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--ranking-margin", type=float, default=0.20)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-folds", type=int, default=5)

    parser.add_argument("--seed-max-ood", type=float, default=0.35)
    parser.add_argument("--seed-max-entropy", type=float, default=0.80)
    parser.add_argument("--seed-min-raw-margin", type=float, default=0.05)
    parser.add_argument(
        "--seed-min-relative-score",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--seed-min-relative-confidence",
        type=float,
        default=0.35,
    )
    parser.add_argument(
        "--seed-min-relative-gap",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--seed-min-relative-z",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--seed-min-relative-z-gap",
        type=float,
        default=0.0,
    )
    parser.add_argument("--min-seed-per-class", type=int, default=3)

    parser.add_argument("--reject-ood", type=float, default=0.50)
    parser.add_argument(
        "--max-neutral-entropy",
        type=float,
        default=0.65,
    )

    parser.add_argument("--refiner-delta-max", type=float, default=1.0)
    parser.add_argument("--refiner-steps", type=int, default=300)
    parser.add_argument(
        "--refiner-learning-rate",
        type=float,
        default=2e-3,
    )
    parser.add_argument(
        "--refiner-teacher-weight",
        type=float,
        default=0.35,
    )
    parser.add_argument(
        "--refiner-critic-weight",
        type=float,
        default=0.50,
    )
    parser.add_argument(
        "--refiner-min-critic-macro-accuracy",
        type=float,
        default=0.35,
    )
    parser.add_argument(
        "--refiner-min-critic-macro-auc",
        type=float,
        default=0.60,
    )
    parser.add_argument(
        "--refiner-min-macro-accuracy",
        type=float,
        default=0.35,
    )
    parser.add_argument(
        "--refiner-min-macro-auc",
        type=float,
        default=0.60,
    )
    parser.add_argument(
        "--refiner-min-critic-margin",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--refiner-min-refiner-margin",
        type=float,
        default=0.15,
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.pca_dim < 2:
        raise CriticError("--pca-dim must be >= 2")
    if args.rank < 1:
        raise CriticError("--rank must be >= 1")
    if args.temperature <= 0:
        raise CriticError("--temperature must be positive")
    if args.steps < 1:
        raise CriticError("--steps must be >= 1")
    if args.learning_rate <= 0:
        raise CriticError("--learning-rate must be positive")
    if args.max_folds < 2:
        raise CriticError("--max-folds must be >= 2")
    if args.min_seed_per_class < 2:
        raise CriticError("--min-seed-per-class must be >= 2")
    if args.seed_min_relative_z < 0:
        raise CriticError("--seed-min-relative-z must be >= 0")
    if args.seed_min_relative_z_gap < 0:
        raise CriticError("--seed-min-relative-z-gap must be >= 0")

    if args.refiner_delta_max <= 0:
        raise CriticError("--refiner-delta-max must be positive")
    if args.refiner_steps < 1:
        raise CriticError("--refiner-steps must be >= 1")
    if args.refiner_learning_rate <= 0:
        raise CriticError("--refiner-learning-rate must be positive")
    if args.refiner_teacher_weight < 0:
        raise CriticError("--refiner-teacher-weight must be >= 0")
    if args.refiner_critic_weight < 0:
        raise CriticError("--refiner-critic-weight must be >= 0")

    for name in (
        "seed_max_ood",
        "seed_max_entropy",
        "seed_min_raw_margin",
        "seed_min_relative_confidence",
        "reject_ood",
        "max_neutral_entropy",
        "refiner_min_critic_macro_accuracy",
        "refiner_min_critic_macro_auc",
        "refiner_min_macro_accuracy",
        "refiner_min_macro_auc",
        "refiner_min_critic_margin",
        "refiner_min_refiner_margin",
    ):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            raise CriticError(
                f"--{name.replace('_', '-')} must be between 0 and 1"
            )


def main() -> int:
    args = make_parser().parse_args()
    try:
        validate_args(args)

        requested_emotions = tuple(
            value.strip().lower()
            for value in args.emotions.split(",")
            if value.strip()
        )
        if (
            len(requested_emotions) < 2
            or len(set(requested_emotions))
            != len(requested_emotions)
        ):
            raise CriticError(
                "--emotions must contain at least two unique labels"
            )
        if any(
            label in NON_TARGET
            for label in requested_emotions
        ):
            raise CriticError(
                "--emotions must contain only non-neutral target labels"
            )

        list_path = Path(args.list).expanduser().resolve()
        sidecar_path = Path(args.sidecar).expanduser().resolve()
        review_path = (
            Path(args.review).expanduser().resolve()
            if args.review
            else None
        )
        wav_root = (
            Path(args.wav_root).expanduser().resolve()
            if args.wav_root
            else None
        )
        prediction_cache_path = (
            Path(args.prediction_cache).expanduser().resolve()
            if args.prediction_cache
            else None
        )
        embedding_cache_path = Path(
            args.embedding_cache
        ).expanduser().resolve()
        output_path = Path(args.output).expanduser().resolve()
        artifact_path = (
            Path(args.artifact).expanduser().resolve()
            if args.artifact
            else output_path.with_suffix(".critic.pt")
        )
        training_sidecar_path = (
            Path(args.training_sidecar).expanduser().resolve()
            if args.training_sidecar
            else None
        )

        rows = prepare_rows(
            list_path=list_path,
            wav_root=wav_root,
            sidecar_path=sidecar_path,
            prediction_cache_path=prediction_cache_path,
            model_name=args.model,
        )
        human_seed_confidences = {
            normalize_review_confidence(value)
            for value in args.human_seed_confidences.split(",")
            if normalize_review_confidence(value)
        }
        if not human_seed_confidences:
            raise CriticError(
                "--human-seed-confidences must contain at least one value"
            )
        human_seed_count = apply_human_review_seeds(
            rows,
            review_path,
            accepted_confidences=human_seed_confidences,
        )
        if human_seed_count:
            print(
                f"[human seeds] applied={human_seed_count} "
                f"confidences={sorted(human_seed_confidences)}",
                flush=True,
            )

        embeddings = ensure_embeddings(
            rows,
            model_name=args.model,
            device=args.device,
            cache_path=embedding_cache_path,
        )

        seeds, active_emotions = select_seed_indices(
            rows,
            requested_emotions,
            max_ood=args.seed_max_ood,
            max_entropy=args.seed_max_entropy,
            min_raw_margin=args.seed_min_raw_margin,
            min_relative_score=args.seed_min_relative_score,
            min_relative_confidence=(
                args.seed_min_relative_confidence
            ),
            min_relative_gap=args.seed_min_relative_gap,
            min_relative_z=args.seed_min_relative_z,
            min_relative_z_gap=args.seed_min_relative_z_gap,
            min_seed_per_class=args.min_seed_per_class,
        )
        config = CriticConfig(
            emotions=active_emotions,
            pca_dim=args.pca_dim,
            rank=args.rank,
            temperature=args.temperature,
            ranking_margin=args.ranking_margin,
            steps=args.steps,
            learning_rate=args.learning_rate,
            seed=args.seed,
            max_folds=args.max_folds,
        )

        seed_counts = Counter(
            seed_label_for_row(rows[index])
            for index in seeds
        )
        print(
            "[seed] "
            + ", ".join(
                f"{emotion}={seed_counts[emotion]}"
                for emotion in config.emotions
            ),
            flush=True,
        )

        oof_logits, _, metrics = crossfit_critic(
            rows,
            embeddings,
            seeds,
            config=config,
        )
        print(
            "[critic OOF] "
            + " ".join(
                f"{key}={value}"
                for key, value in metrics.items()
            ),
            flush=True,
        )

        final_result = fit_final_critic(
            embeddings,
            seeds,
            rows,
            config=config,
        )

        refiner_config = None
        refiner_result = None
        refiner_metrics = None
        if args.mode == "refiner":
            if not critic_gate_passed(
                metrics,
                min_macro_accuracy=(
                    args.refiner_min_critic_macro_accuracy
                ),
                min_macro_auc=args.refiner_min_critic_macro_auc,
            ):
                raise CriticError(
                    "bounded refiner blocked: critic OOF gate failed "
                    f"(macro_accuracy={metrics.get('oof_macro_accuracy')}, "
                    f"macro_auc={metrics.get('oof_macro_auc')})"
                )

            refiner_config = RefinerConfig(
                delta_max=args.refiner_delta_max,
                steps=args.refiner_steps,
                learning_rate=args.refiner_learning_rate,
                teacher_weight=args.refiner_teacher_weight,
                critic_weight=args.refiner_critic_weight,
            )
            refiner_oof_logits, refiner_metrics = (
                crossfit_refiner(
                    rows,
                    embeddings,
                    seeds,
                    critic_config=config,
                    refiner_config=refiner_config,
                )
            )
            if not critic_gate_passed(
                refiner_metrics,
                min_macro_accuracy=args.refiner_min_macro_accuracy,
                min_macro_auc=args.refiner_min_macro_auc,
            ):
                raise CriticError(
                    "bounded refiner blocked: refiner OOF gate failed "
                    f"(macro_accuracy="
                    f"{refiner_metrics.get('oof_macro_accuracy')}, "
                    f"macro_auc={refiner_metrics.get('oof_macro_auc')})"
                )
            refiner_result = fit_final_refiner(
                rows,
                seeds,
                final_result["all_logits"],
                critic_config=config,
                refiner_config=refiner_config,
            )
            output_rows, state_counts = apply_refiner_outputs(
                rows,
                seed_indices=seeds,
                critic_oof_logits=oof_logits,
                critic_final_logits=final_result["all_logits"],
                refiner_oof_logits=refiner_oof_logits,
                refiner_final_logits=refiner_result["all_logits"],
                emotions=config.emotions,
                reject_ood=args.reject_ood,
                max_neutral_entropy=args.max_neutral_entropy,
                min_critic_margin=args.refiner_min_critic_margin,
                min_refiner_margin=args.refiner_min_refiner_margin,
            )
            kind = "critic_guided_residual_oof_v1"
            print(
                "[refiner OOF] "
                + " ".join(
                    f"{key}={value}"
                    for key, value in refiner_metrics.items()
                ),
                flush=True,
            )
        else:
            output_rows, state_counts = apply_critic_outputs(
                rows,
                seed_indices=seeds,
                oof_logits=oof_logits,
                final_logits=final_result["all_logits"],
                emotions=config.emotions,
                reject_ood=args.reject_ood,
                max_neutral_entropy=args.max_neutral_entropy,
            )
            kind = "prototype_critic_oof_v1"

        write_jsonl(output_path, output_rows)

        training_counts = None
        if training_sidecar_path is not None:
            training_rows, training_counts = (
                materialize_training_sidecar(
                    output_rows,
                    policy=args.training_policy,
                )
            )
            write_jsonl(
                training_sidecar_path,
                training_rows,
            )
            print(
                "[training sidecar] "
                f"policy={args.training_policy} "
                f"path={training_sidecar_path}",
                flush=True,
            )

        report = {
            "kind": kind,
            "mode": args.mode,
            "teacher_model": args.model,
            "samples": len(rows),
            "embedding_dim": int(embeddings.shape[1]),
            "seed_samples": len(seeds),
            "seed_counts": dict(sorted(seed_counts.items())),
            "human_seed_count": human_seed_count,
            "human_seed_confidences": sorted(
                human_seed_confidences
            ),
            "active_emotions": list(config.emotions),
            "state_counts": state_counts,
            "prototype_geometry": prototype_geometry(
                final_result["critic"],
                config.emotions,
            ),
            "training_sidecar": (
                str(training_sidecar_path)
                if training_sidecar_path is not None
                else None
            ),
            "training_policy": (
                args.training_policy
                if training_sidecar_path is not None
                else None
            ),
            "training_counts": training_counts,
            "metrics": metrics,
            "refiner_metrics": refiner_metrics,
            "config": asdict(config),
            "refiner_config": (
                asdict(refiner_config)
                if refiner_config is not None
                else None
            ),
            "notes": [
                (
                    "emotion_intended is never overwritten; critic/refiner "
                    "fields and v3 states are additive diagnostics."
                ),
                (
                    "Seed samples are evaluated OOF. Non-seed samples use "
                    "final models fitted on all accepted seeds."
                ),
                (
                    "Refiner mode is blocked automatically unless the critic "
                    "passes the configured OOF macro-accuracy/AUC gate."
                ),
                (
                    "Candidate states are not automatic Controller targets. "
                    "A derived training sidecar is created only when "
                    "--training-sidecar is explicitly supplied."
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
        save_artifact(
            artifact_path,
            config=config,
            model_name=args.model,
            final_result=final_result,
            metrics=metrics,
            refiner_config=refiner_config,
            refiner_result=refiner_result,
            refiner_metrics=refiner_metrics,
        )

        print(
            json.dumps(
                report,
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
    except (
        CriticError,
        EmotionPipelineError,
        OSError,
        ValueError,
        RuntimeError,
    ) as exc:
        print(
            f"[FATAL] {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
