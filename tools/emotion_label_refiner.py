#!/usr/bin/env python3
"""Cross-fitted emotion compatibility critic for pseudo-label validation.

The critic uses frozen emotion2vec+ embeddings to ask whether an accepted
speaker-relative pseudo-label is geometrically compatible with the audio.
It never overwrites emotion_intended. Accepted samples are scored OOF; samples
that were not training seeds are scored by a final critic fitted on all seeds.

This is Phase C of ADVERSARIAL_EMOTION_REFINER_DESIGN.md. The bounded refiner
should only be added after this critic proves informative out-of-fold.
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
    min_seed_per_class: int,
) -> tuple[list[int], tuple[str, ...]]:
    requested = tuple(emotions)
    requested_set = set(requested)
    provisional: list[int] = []

    for index, row in enumerate(rows):
        label = normalize_label(row.get("emotion_intended"))
        if label not in requested_set:
            continue
        if row.get("emotion_prediction_error"):
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
        provisional.append(index)

    counts = Counter(
        normalize_label(rows[index].get("emotion_intended"))
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
        if normalize_label(rows[index].get("emotion_intended"))
        in active_set
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


def crossfit_critic(
    rows: Sequence[dict[str, Any]],
    embeddings: torch.Tensor,
    seed_indices: Sequence[int],
    *,
    config: CriticConfig,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    emotion_to_id = {
        emotion: index
        for index, emotion in enumerate(config.emotions)
    }
    labels = torch.tensor(
        [
            emotion_to_id[
                normalize_label(
                    rows[index].get("emotion_intended")
                )
            ]
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
    }
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
            emotion_to_id[
                normalize_label(
                    rows[index].get("emotion_intended")
                )
            ]
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
        raw_top1 = normalize_label(row.get("emotion_raw_top1"))
        ood_mass = safe_float(
            row.get("emotion_ood_mass"),
            1.0,
        )
        entropy = safe_float(
            row.get("emotion_entropy"),
            1.0,
        )

        if row.get("emotion_prediction_error"):
            state = "failed"
        elif (
            raw_top1 in {"other", "unknown"}
            or ood_mass >= reject_ood
        ):
            state = "ood"
        elif intended in emotion_set:
            state = (
                "accepted_core"
                if label == intended
                else "conflict"
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
                if state == "neutral_candidate"
                else "reject"
            )
        )
        row["emotion_v3_candidate"] = (
            intended
            if state == "accepted_core"
            else None
        )
        row["auto_emotion_refiner"] = "prototype_critic_oof_v1"

        states[state] += 1
        output.append(row)

    return output, dict(sorted(states.items()))


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
    parser.add_argument("--prediction-cache")
    parser.add_argument("--embedding-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--artifact")
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
    parser.add_argument("--min-seed-per-class", type=int, default=3)

    parser.add_argument("--reject-ood", type=float, default=0.50)
    parser.add_argument(
        "--max-neutral-entropy",
        type=float,
        default=0.65,
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

    for name in (
        "seed_max_ood",
        "seed_max_entropy",
        "seed_min_raw_margin",
        "seed_min_relative_confidence",
        "reject_ood",
        "max_neutral_entropy",
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

        rows = prepare_rows(
            list_path=list_path,
            wav_root=wav_root,
            sidecar_path=sidecar_path,
            prediction_cache_path=prediction_cache_path,
            model_name=args.model,
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
            normalize_label(
                rows[index].get("emotion_intended")
            )
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
        output_rows, state_counts = apply_critic_outputs(
            rows,
            seed_indices=seeds,
            oof_logits=oof_logits,
            final_logits=final_result["all_logits"],
            emotions=config.emotions,
            reject_ood=args.reject_ood,
            max_neutral_entropy=args.max_neutral_entropy,
        )
        write_jsonl(output_path, output_rows)

        report = {
            "kind": "prototype_critic_oof_v1",
            "teacher_model": args.model,
            "samples": len(rows),
            "embedding_dim": int(embeddings.shape[1]),
            "seed_samples": len(seeds),
            "seed_counts": dict(sorted(seed_counts.items())),
            "active_emotions": list(config.emotions),
            "state_counts": state_counts,
            "metrics": metrics,
            "config": asdict(config),
            "notes": [
                (
                    "emotion_intended is never overwritten; critic fields and "
                    "v3 states are additive diagnostics."
                ),
                (
                    "Seed samples are evaluated OOF. Non-seed samples use a "
                    "final critic fitted on all accepted seeds."
                ),
                (
                    "Do not implement/enable the bounded refiner unless this "
                    "critic is informative OOF."
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
