"""Small orchestration helpers for the V5 emotion-controller training UI.

This module does not implement training itself. It only connects the existing
canonical dataset, V5 preprocessing, reference-prompt export, and controller
trainer so the WebUI can run the pipeline without duplicating model logic.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import yaml


class EmotionPipelineError(RuntimeError):
    pass


def controller_artifact_paths(final_output: Path) -> dict[str, Path]:
    final_path = Path(final_output)
    candidate = final_path.with_name(
        f"{final_path.stem}.candidate{final_path.suffix}"
    )
    failed = final_path.with_name(
        f"{final_path.stem}.failed{final_path.suffix}"
    )
    candidate_diagnostics = candidate.with_name(
        f"{candidate.stem}.diagnostics.json"
    )
    final_diagnostics = final_path.with_name(
        f"{final_path.stem}.diagnostics.json"
    )
    failed_diagnostics = failed.with_name(
        f"{failed.stem}.diagnostics.json"
    )
    return {
        "final": final_path,
        "candidate": candidate,
        "failed": failed,
        "candidate_diagnostics": candidate_diagnostics,
        "final_diagnostics": final_diagnostics,
        "failed_diagnostics": failed_diagnostics,
    }


def latest_gpt_checkpoint_for_experiment(
    weight_root: Path,
    experiment_name: str,
) -> Path | None:
    """Return the most recently written half-weight checkpoint for one experiment."""
    root = Path(weight_root)
    name = str(experiment_name or "").strip()
    if not name or not root.is_dir():
        return None

    prefix = f"{name}-e"
    candidates = []
    for path in root.glob(f"{name}-e*.ckpt"):
        stem = path.stem
        if not stem.startswith(prefix):
            continue
        suffix = stem[len(prefix):]
        if not suffix.isdigit():
            continue
        candidates.append((int(suffix), path))

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: (
            item[1].stat().st_mtime_ns,
            item[0],
        )
    )
    return candidates[-1][1]


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EmotionPipelineError(
                    f"{path}:{line_number}: invalid JSON"
                ) from exc

            if not isinstance(row, dict):
                raise EmotionPipelineError(
                    f"{path}:{line_number}: expected a JSON object"
                )

            rows.append(row)

    return rows


def load_dataset_report(dataset_dir: Path) -> dict:
    report_path = dataset_dir / "report.json"

    if not report_path.is_file():
        raise EmotionPipelineError(
            f"dataset report not found: {report_path}"
        )

    report = json.loads(report_path.read_text(encoding="utf-8"))

    if not isinstance(report, dict):
        raise EmotionPipelineError("dataset report must be a JSON object")

    return report


NON_CONTROLLER_EMOTIONS = {
    "",
    "neutral",
    "default",
    "unlabeled",
    "unknown",
    "other",
}

DEFAULT_MIN_SAMPLES_PER_EMOTION = 3



def _connected_sample_groups(
    samples: Sequence[dict],
) -> list[list[int]]:
    """Connect samples that share provenance/text/pair identities.

    Internal controller validation must not split correlated examples across
    train/validation. The canonical dataset already treats source_id,
    text_group_id and pair_group_id as leakage-sensitive identities; reuse the
    same rule here. Samples with no such metadata remain independent.
    """
    parent = list(range(len(samples)))
    owner: dict[str, int] = {}

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for index, sample in enumerate(samples):
        tokens = []
        for field in ("source_id", "text_group_id", "pair_group_id"):
            value = str(sample.get(field) or "").strip()
            if value:
                tokens.append(f"{field}:{value}")

        if not tokens:
            tokens.append(
                "sample:"
                + str(sample.get("name") or index)
            )

        for token in tokens:
            previous = owner.get(token)
            if previous is None:
                owner[token] = index
            else:
                union(index, previous)

    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(samples)):
        groups[find(index)].append(index)

    return [
        sorted(indices)
        for _, indices in sorted(
            groups.items(),
            key=lambda item: min(item[1]),
        )
    ]


def split_train_validation_samples(
    samples: Sequence[dict],
    emotion_names: Sequence[str],
    *,
    val_fraction: float,
    seed: int,
    min_train_samples: int = 2,
) -> tuple[list[dict], list[dict]]:
    """Create a deterministic, leakage-aware held-out split.

    Samples connected by source_id, text_group_id or pair_group_id are kept in
    the same side of the split. This prevents exact-text or same-source leakage
    from inflating validation quality.
    """
    if not 0.0 < val_fraction < 1.0:
        raise EmotionPipelineError("val_fraction must be in (0,1)")
    if min_train_samples < 1:
        raise EmotionPipelineError("min_train_samples must be >= 1")

    samples = [dict(sample) for sample in samples]
    emotion_names = [
        str(emotion).strip().lower()
        for emotion in emotion_names
        if str(emotion).strip()
    ]
    emotion_set = set(emotion_names)

    total_counts = {emotion: 0 for emotion in emotion_names}
    for sample in samples:
        emotion = str(sample.get("emotion") or "").strip().lower()
        if emotion in total_counts:
            total_counts[emotion] += 1

    target_val = {}
    for emotion in emotion_names:
        count = total_counts[emotion]
        if count < min_train_samples + 1:
            raise EmotionPipelineError(
                f"Emotion {emotion!r} needs at least "
                f"{min_train_samples + 1} usable samples for validation"
            )
        requested = max(1, int(round(count * val_fraction)))
        target_val[emotion] = min(
            requested,
            count - min_train_samples,
        )

    groups = _connected_sample_groups(samples)
    group_rows = []
    for indices in groups:
        counts = {emotion: 0 for emotion in emotion_names}
        for index in indices:
            emotion = str(
                samples[index].get("emotion") or ""
            ).strip().lower()
            if emotion in counts:
                counts[emotion] += 1
        group_rows.append(
            {
                "indices": indices,
                "counts": counts,
            }
        )

    split_rng = random.Random(seed + 104729)
    split_rng.shuffle(group_rows)

    val_counts = {emotion: 0 for emotion in emotion_names}
    train_counts = dict(total_counts)
    selected_groups: list[dict] = []

    while any(
        val_counts[emotion] < target_val[emotion]
        for emotion in emotion_names
    ):
        best_index = None
        best_score = None

        for index, group in enumerate(group_rows):
            counts = group["counts"]
            if any(
                train_counts[emotion] - counts[emotion]
                < min_train_samples
                for emotion in emotion_names
            ):
                continue

            gain = sum(
                min(
                    counts[emotion],
                    max(
                        target_val[emotion] - val_counts[emotion],
                        0,
                    ),
                )
                for emotion in emotion_names
            )
            if gain <= 0:
                continue

            excess = sum(
                max(
                    val_counts[emotion]
                    + counts[emotion]
                    - target_val[emotion],
                    0,
                )
                for emotion in emotion_names
            )
            group_size = len(group["indices"])
            # Prefer groups that satisfy the most unmet class targets while
            # adding the least excess validation data.
            score = (gain, -excess, -group_size)
            if best_score is None or score > best_score:
                best_score = score
                best_index = index

        if best_index is None:
            unmet = [
                emotion
                for emotion in emotion_names
                if val_counts[emotion] < target_val[emotion]
            ]
            raise EmotionPipelineError(
                "Leakage-safe validation split is impossible while keeping "
                f"{min_train_samples} train samples per class. "
                "Correlated groups are too large for: "
                + ", ".join(unmet)
            )

        chosen = group_rows.pop(best_index)
        selected_groups.append(chosen)
        for emotion in emotion_names:
            val_counts[emotion] += chosen["counts"][emotion]
            train_counts[emotion] -= chosen["counts"][emotion]

    val_indices = {
        index
        for group in selected_groups
        for index in group["indices"]
    }
    train_samples = [
        sample
        for index, sample in enumerate(samples)
        if index not in val_indices
        and str(sample.get("emotion") or "").strip().lower()
        in emotion_set
    ]
    val_samples = [
        sample
        for index, sample in enumerate(samples)
        if index in val_indices
        and str(sample.get("emotion") or "").strip().lower()
        in emotion_set
    ]

    if not train_samples or not val_samples:
        raise EmotionPipelineError("train/validation split is empty")

    return train_samples, val_samples


def stratified_kfold_indices(
    labels: Sequence[int],
    *,
    max_folds: int = 5,
    seed: int,
) -> list[tuple[list[int], list[int]]]:
    """Build deterministic stratified folds without sklearn."""
    if max_folds < 2:
        raise EmotionPipelineError("max_folds must be >= 2")
    if len(labels) < 2:
        raise EmotionPipelineError("need at least two samples for k-fold")

    by_label: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        by_label[int(label)].append(index)

    if len(by_label) < 2:
        raise EmotionPipelineError(
            "representation probe needs at least two classes"
        )

    min_class_count = min(len(indices) for indices in by_label.values())
    fold_count = min(max_folds, min_class_count)
    if fold_count < 2:
        raise EmotionPipelineError(
            "representation probe needs at least two samples per class"
        )

    split_rng = random.Random(seed + 32452843)
    fold_validation: list[list[int]] = [
        [] for _ in range(fold_count)
    ]

    for label in sorted(by_label):
        indices = list(by_label[label])
        split_rng.shuffle(indices)
        for position, index in enumerate(indices):
            fold_validation[position % fold_count].append(index)

    all_indices = list(range(len(labels)))
    folds: list[tuple[list[int], list[int]]] = []
    for validation in fold_validation:
        validation = sorted(validation)
        validation_set = set(validation)
        training = [
            index for index in all_indices
            if index not in validation_set
        ]
        folds.append((training, validation))

    return folds


def stratified_group_kfold_indices(
    samples: Sequence[dict],
    labels: Sequence[int],
    *,
    max_folds: int = 5,
    seed: int,
) -> list[tuple[list[int], list[int]]]:
    """Build deterministic stratified folds without leaking correlated groups."""
    if len(samples) != len(labels):
        raise EmotionPipelineError("sample/label count mismatch for group k-fold")
    if max_folds < 2:
        raise EmotionPipelineError("max_folds must be >= 2")
    if len(labels) < 2:
        raise EmotionPipelineError("need at least two samples for group k-fold")

    labels = [int(label) for label in labels]
    class_ids = sorted(set(labels))
    if len(class_ids) < 2:
        raise EmotionPipelineError(
            "controller OOF needs at least two classes"
        )

    groups = _connected_sample_groups(samples)
    groups_per_class = {class_id: 0 for class_id in class_ids}
    group_counts = []

    for indices in groups:
        counts = {class_id: 0 for class_id in class_ids}
        present = set()
        for index in indices:
            label = labels[index]
            counts[label] += 1
            present.add(label)
        for class_id in present:
            groups_per_class[class_id] += 1
        group_counts.append((indices, counts))

    fold_count = min(
        max_folds,
        min(groups_per_class.values(), default=0),
    )
    if fold_count < 2:
        raise EmotionPipelineError(
            "controller OOF needs at least two independent leakage groups "
            "per class"
        )

    total_by_class = {
        class_id: sum(label == class_id for label in labels)
        for class_id in class_ids
    }

    def assignment_score(
        fold_counts: list[dict[int, int]],
    ) -> float:
        score = 0.0
        for class_id in class_ids:
            total = max(total_by_class[class_id], 1)
            ratios = [
                fold[class_id] / total
                for fold in fold_counts
            ]
            mean = sum(ratios) / len(ratios)
            score += sum(
                (value - mean) ** 2
                for value in ratios
            ) / len(ratios)
        return score

    # Greedy stratified-group assignment can have tie-sensitive outcomes.
    # Retry deterministic shuffles and keep the first complete allocation.
    for retry in range(32):
        split_rng = random.Random(
            seed + 32452843 + retry * 1009
        )
        ordered = list(group_counts)
        split_rng.shuffle(ordered)
        ordered.sort(
            key=lambda item: (
                -max(
                    (
                        item[1][class_id]
                        / max(total_by_class[class_id], 1)
                    )
                    for class_id in class_ids
                ),
                -len(item[0]),
            )
        )

        fold_counts = [
            {class_id: 0 for class_id in class_ids}
            for _ in range(fold_count)
        ]
        fold_groups: list[list[list[int]]] = [
            [] for _ in range(fold_count)
        ]

        for indices, counts in ordered:
            best_fold = None
            best_score = None
            for fold_index in range(fold_count):
                trial = [
                    dict(values)
                    for values in fold_counts
                ]
                for class_id in class_ids:
                    trial[fold_index][class_id] += counts[class_id]

                balance = assignment_score(trial)
                fold_size = sum(trial[fold_index].values())
                score = (balance, fold_size, fold_index)
                if best_score is None or score < best_score:
                    best_score = score
                    best_fold = fold_index

            for class_id in class_ids:
                fold_counts[best_fold][class_id] += counts[class_id]
            fold_groups[best_fold].append(indices)

        folds: list[tuple[list[int], list[int]]] = []
        valid = True
        all_indices = set(range(len(labels)))
        for fold_index in range(fold_count):
            validation = sorted(
                index
                for group in fold_groups[fold_index]
                for index in group
            )
            validation_set = set(validation)
            training = sorted(all_indices - validation_set)

            val_labels = {labels[index] for index in validation}
            train_labels = {labels[index] for index in training}
            if (
                not validation
                or not training
                or val_labels != set(class_ids)
                or train_labels != set(class_ids)
            ):
                valid = False
                break
            folds.append((training, validation))

        if valid:
            return folds

    raise EmotionPipelineError(
        "Could not build leakage-safe stratified OOF folds from the current "
        "source/text/pair groups"
    )


def run_mean_pool_linear_probe(
    features,
    labels: Sequence[int],
    emotion_names: Sequence[str],
    *,
    seed: int,
    max_folds: int = 5,
    permutations: int = 3,
    steps: int = 120,
    method: str = "mean_pool_linear_stratified_cv_v2",
) -> dict:
    """Cross-validated linear probe with OOF metrics and permutation baseline.

    The probe is intentionally lightweight and uses only torch. It reports
    out-of-fold CE, accuracy, macro accuracy and one-vs-rest macro AUC so tiny
    validation folds are not reduced to one noisy percentage.
    """
    import torch
    import torch.nn.functional as F

    labels = [int(label) for label in labels]
    class_count = len(emotion_names)
    sample_count = len(labels)

    counts = {
        str(emotion_names[index]): labels.count(index)
        for index in range(class_count)
    }
    report = {
        "method": method,
        "samples": sample_count,
        "class_counts": counts,
        "max_folds": int(max_folds),
        "permutations": int(permutations),
        "steps": int(steps),
        "chance_accuracy": (
            1.0 / class_count if class_count else 0.0
        ),
        "chance_ce": (
            float(torch.log(torch.tensor(float(class_count))).item())
            if class_count > 0
            else None
        ),
    }

    if (
        class_count < 2
        or sample_count != int(features.shape[0])
        or min(counts.values(), default=0) < 2
    ):
        report.update(
            {
                "status": "insufficient_data",
                "verdict": "unavailable",
            }
        )
        return report

    x = features.detach().to(
        device="cpu",
        dtype=torch.float32,
    )
    if x.ndim != 2 or not torch.isfinite(x).all():
        report.update(
            {
                "status": "invalid_features",
                "verdict": "unavailable",
            }
        )
        return report

    def _macro_accuracy(logits, target) -> float:
        prediction = logits.argmax(dim=-1)
        per_class = []
        for class_id in range(class_count):
            mask = target == class_id
            if bool(mask.any()):
                per_class.append(
                    (prediction[mask] == target[mask])
                    .float()
                    .mean()
                )
        if not per_class:
            return 0.0
        return float(torch.stack(per_class).mean().item())

    def _binary_auc(scores, positive_mask) -> float | None:
        positive = scores[positive_mask]
        negative = scores[~positive_mask]
        if positive.numel() == 0 or negative.numel() == 0:
            return None
        comparisons = positive[:, None] - negative[None, :]
        wins = (comparisons > 0).float().mean()
        ties = (comparisons == 0).float().mean()
        return float((wins + 0.5 * ties).item())

    def _macro_auc(logits, target) -> float | None:
        probs = torch.softmax(logits, dim=-1)
        values = []
        for class_id in range(class_count):
            auc = _binary_auc(
                probs[:, class_id],
                target == class_id,
            )
            if auc is not None:
                values.append(auc)
        if not values:
            return None
        return sum(values) / len(values)

    def _evaluate(label_values: list[int], run_seed: int):
        folds = stratified_kfold_indices(
            label_values,
            max_folds=max_folds,
            seed=run_seed,
        )
        fold_scores: list[float] = []
        y_all = torch.tensor(label_values, dtype=torch.long)
        oof_logits = torch.full(
            (sample_count, class_count),
            float("nan"),
            dtype=torch.float32,
        )

        for fold_index, (train_idx, val_idx) in enumerate(folds):
            train_index = torch.tensor(train_idx, dtype=torch.long)
            val_index = torch.tensor(val_idx, dtype=torch.long)

            x_train = x.index_select(0, train_index)
            x_val = x.index_select(0, val_index)
            y_train = y_all.index_select(0, train_index)
            y_val = y_all.index_select(0, val_index)

            mean = x_train.mean(dim=0, keepdim=True)
            std = x_train.std(
                dim=0,
                unbiased=False,
                keepdim=True,
            ).clamp_min(1e-3)
            x_train = (x_train - mean) / std
            x_val = (x_val - mean) / std

            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(
                    run_seed + 1009 * (fold_index + 1)
                )
                classifier = torch.nn.Linear(
                    x.shape[1],
                    class_count,
                )

            class_counts = torch.bincount(
                y_train,
                minlength=class_count,
            ).float().clamp_min(1.0)
            class_weights = (
                float(y_train.numel())
                / (class_count * class_counts)
            )

            optimizer = torch.optim.AdamW(
                classifier.parameters(),
                lr=3e-2,
                weight_decay=5e-2,
            )

            classifier.train()
            for _ in range(steps):
                optimizer.zero_grad(set_to_none=True)
                logits = classifier(x_train)
                loss = F.cross_entropy(
                    logits,
                    y_train,
                    weight=class_weights,
                )
                loss.backward()
                optimizer.step()

            classifier.eval()
            with torch.no_grad():
                val_logits = classifier(x_val)
                oof_logits.index_copy_(0, val_index, val_logits)
                fold_scores.append(
                    _macro_accuracy(
                        val_logits,
                        y_val,
                    )
                )

        if not bool(torch.isfinite(oof_logits).all()):
            raise EmotionPipelineError(
                "linear probe did not produce complete OOF predictions"
            )

        fold_values = torch.tensor(
            fold_scores,
            dtype=torch.float32,
        )
        oof_prediction = oof_logits.argmax(dim=-1)
        return {
            "fold_scores": fold_scores,
            "fold_macro_accuracy_mean": float(
                fold_values.mean().item()
            ),
            "fold_macro_accuracy_std": float(
                fold_values.std(unbiased=False).item()
            ),
            "oof_ce": float(
                F.cross_entropy(oof_logits, y_all).item()
            ),
            "oof_accuracy": float(
                (oof_prediction == y_all).float().mean().item()
            ),
            "oof_macro_accuracy": _macro_accuracy(
                oof_logits,
                y_all,
            ),
            "oof_macro_auc": _macro_auc(
                oof_logits,
                y_all,
            ),
        }

    try:
        folds = stratified_kfold_indices(
            labels,
            max_folds=max_folds,
            seed=seed,
        )
    except EmotionPipelineError:
        report.update(
            {
                "status": "insufficient_data",
                "verdict": "unavailable",
            }
        )
        return report

    real = _evaluate(labels, seed)

    permutation_means: list[float] = []
    permutation_ces: list[float] = []
    for permutation_index in range(max(permutations, 0)):
        permuted = list(labels)
        permutation_rng = random.Random(
            seed + 49979687 + permutation_index
        )
        permutation_rng.shuffle(permuted)
        perm = _evaluate(
            permuted,
            seed + 86028121 + permutation_index,
        )
        permutation_means.append(perm["oof_macro_accuracy"])
        permutation_ces.append(perm["oof_ce"])

    if permutation_means:
        perm_tensor = torch.tensor(
            permutation_means,
            dtype=torch.float32,
        )
        permutation_mean = float(perm_tensor.mean().item())
        permutation_std = float(
            perm_tensor.std(unbiased=False).item()
        )
        permutation_ce_mean = float(
            torch.tensor(
                permutation_ces,
                dtype=torch.float32,
            ).mean().item()
        )
    else:
        permutation_mean = report["chance_accuracy"]
        permutation_std = 0.0
        permutation_ce_mean = float(report["chance_ce"])

    chance = float(report["chance_accuracy"])
    delta = real["oof_macro_accuracy"] - permutation_mean

    if (
        real["oof_macro_accuracy"] >= chance + 0.10
        and delta >= 0.10
    ):
        verdict = "informative"
    elif (
        real["oof_macro_accuracy"] <= chance + 0.05
        or delta <= 0.05
    ):
        verdict = "weak"
    else:
        verdict = "uncertain"

    report.update(
        {
            "status": "ok",
            "folds": len(folds),
            "fold_macro_accuracy": real["fold_scores"],
            "macro_accuracy_mean": real[
                "fold_macro_accuracy_mean"
            ],
            "macro_accuracy_std": real[
                "fold_macro_accuracy_std"
            ],
            "oof_ce": real["oof_ce"],
            "oof_accuracy": real["oof_accuracy"],
            "oof_macro_accuracy": real["oof_macro_accuracy"],
            "oof_macro_auc": real["oof_macro_auc"],
            "permutation_macro_accuracy_mean": permutation_mean,
            "permutation_macro_accuracy_std": permutation_std,
            "permutation_ce_mean": permutation_ce_mean,
            "delta_vs_permutation": delta,
            "verdict": verdict,
            "note": (
                "Separability is diagnostic only; text or another "
                "correlated cue can also score above chance."
            ),
        }
    )
    return report


def update_early_stop_state(
    *,
    current_loss: float,
    best_loss: float,
    validations_without_improvement: int,
    min_delta: float,
) -> tuple[float, int, bool]:
    """Update validation-loss early-stop state deterministically."""
    improved = current_loss < best_loss - min_delta
    if improved:
        return current_loss, 0, True
    return (
        best_loss,
        validations_without_improvement + 1,
        False,
    )

def validate_emotion_dataset(dataset_dir: Path) -> dict:
    metadata_path = dataset_dir / "metadata.jsonl"
    train_list = dataset_dir / "train.list"

    if not metadata_path.is_file():
        raise EmotionPipelineError(
            f"metadata.jsonl not found: {metadata_path}"
        )

    if not train_list.is_file():
        raise EmotionPipelineError(
            f"train.list not found: {train_list}"
        )

    report = load_dataset_report(dataset_dir)
    detected = detect_pair_emotions(metadata_path)

    if len(detected) < 2:
        raise EmotionPipelineError(
            "Controller training needs at least two non-neutral emotion "
            "labels. Neutral/default is the unmodified GPT-SoVITS baseline."
        )

    return report


def detect_pair_emotions(
    metadata_path: Path,
    min_samples_per_emotion: int = DEFAULT_MIN_SAMPLES_PER_EMOTION,
) -> list[str]:
    """Return trainable per-sample emotions with enough usable train samples."""
    if min_samples_per_emotion < 1:
        raise ValueError("min_samples_per_emotion must be >= 1")

    rows = read_jsonl(metadata_path)
    counts: dict[str, int] = {}

    for row in rows:
        if row.get("split") != "train":
            continue
        if not row.get("tts_usable", True):
            continue

        emotion = str(
            row.get("emotion_intended") or ""
        ).strip().lower()
        if emotion in NON_CONTROLLER_EMOTIONS:
            continue
        counts[emotion] = counts.get(emotion, 0) + 1

    detected = {
        emotion
        for emotion, count in counts.items()
        if count >= min_samples_per_emotion
    }

    excluded = {
        emotion: count
        for emotion, count in counts.items()
        if count < min_samples_per_emotion
    }
    if excluded:
        print(
            "[emotions] excluded low-sample classes: "
            + ", ".join(
                f"{emotion}={count}"
                for emotion, count in sorted(excluded.items())
            )
            + f" (min={min_samples_per_emotion})"
        )

    preferred_order = [
        "happy",
        "sad",
        "angry",
        "disgusted",
        "fearful",
        "surprised",
    ]

    ordered = [
        emotion
        for emotion in preferred_order
        if emotion in detected
    ]
    ordered.extend(
        sorted(detected - set(ordered))
    )

    if len(ordered) < 2:
        raise EmotionPipelineError(
            "Fewer than two trainable non-neutral emotions have enough "
            f"samples (min={min_samples_per_emotion})"
        )

    return ordered


def _reference_duration_seconds(audio_path: Path) -> float | None:
    try:
        import soundfile as sf

        info = sf.info(str(audio_path))
    except Exception:
        return None

    if not info.samplerate:
        return None

    return float(info.frames) / float(info.samplerate)


def _reference_candidates(
    rows: Iterable[dict],
    dataset_dir: Path,
    trainable_emotions: set[str] | None = None,
    min_samples_per_emotion: int = DEFAULT_MIN_SAMPLES_PER_EMOTION,
) -> dict[str, list[tuple[int, float, str]]]:
    rows = list(rows)
    candidates: dict[str, list[tuple[int, float, str]]] = {}

    controller_counts: dict[tuple[str, str], int] = {}
    for row in rows:
        if (
            row.get("split") != "train"
            or not row.get("tts_usable", True)
        ):
            continue
        speaker_uid = str(row.get("speaker_uid") or "").strip()
        emotion = str(
            row.get("emotion_intended") or ""
        ).strip().lower()
        if not speaker_uid:
            continue
        if emotion in NON_CONTROLLER_EMOTIONS:
            continue
        if (
            trainable_emotions is not None
            and emotion not in trainable_emotions
        ):
            continue
        key = (speaker_uid, emotion)
        controller_counts[key] = controller_counts.get(key, 0) + 1

    for row in rows:
        if row.get("split") != "train":
            continue

        if row.get("pair_group_id"):
            continue

        if not row.get("tts_usable", True):
            continue

        speaker_uid = str(row.get("speaker_uid", "")).strip()
        canonical_wav = str(row.get("canonical_wav", "")).strip()

        if not speaker_uid or not canonical_wav:
            continue

        emotion = str(
            row.get("emotion_intended") or ""
        ).strip().lower()

        if (
            trainable_emotions is not None
            and emotion not in NON_CONTROLLER_EMOTIONS
            and emotion not in trainable_emotions
        ):
            continue

        # A fixed reference is excluded from controller supervision. Never
        # consume the only available sample of a trainable emotion.
        if (
            emotion not in NON_CONTROLLER_EMOTIONS
            and (
                trainable_emotions is None
                or emotion in trainable_emotions
            )
            and controller_counts.get((speaker_uid, emotion), 0)
            <= min_samples_per_emotion
        ):
            continue

        audio_path = dataset_dir / "wav" / canonical_wav
        duration = _reference_duration_seconds(audio_path)

        if duration is None or not 3.0 <= duration <= 10.0:
            continue

        priority = (
            0
            if emotion in NON_CONTROLLER_EMOTIONS
            else 1
        )

        candidates.setdefault(speaker_uid, []).append(
            (priority, abs(duration - 6.0), canonical_wav)
        )

    return candidates


def create_reference_map(
    metadata_path: Path,
    output_path: Path,
    trainable_emotions: Iterable[str] | None = None,
    min_samples_per_emotion: int = DEFAULT_MIN_SAMPLES_PER_EMOTION,
) -> dict[str, str]:
    metadata_path = metadata_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    dataset_dir = metadata_path.parent

    rows = read_jsonl(metadata_path)
    selected_emotions = (
        {
            str(emotion).strip().lower()
            for emotion in trainable_emotions
            if str(emotion).strip()
        }
        if trainable_emotions is not None
        else None
    )

    speakers = {
        str(row.get("speaker_uid") or "").strip()
        for row in rows
        if (
            row.get("split") == "train"
            and row.get("tts_usable", True)
            and str(row.get("speaker_uid") or "").strip()
            and str(
                row.get("emotion_intended") or ""
            ).strip().lower() not in NON_CONTROLLER_EMOTIONS
            and (
                selected_emotions is None
                or str(
                    row.get("emotion_intended") or ""
                ).strip().lower() in selected_emotions
            )
        )
    }

    candidates = _reference_candidates(
        rows,
        dataset_dir,
        trainable_emotions=selected_emotions,
        min_samples_per_emotion=min_samples_per_emotion,
    )
    reference_map: dict[str, str] = {}

    missing: list[str] = []

    for speaker_uid in sorted(speakers):
        speaker_candidates = candidates.get(speaker_uid, [])

        if not speaker_candidates:
            missing.append(speaker_uid)
            continue

        speaker_candidates.sort()
        reference_map[speaker_uid] = speaker_candidates[0][2]

    if missing:
        raise EmotionPipelineError(
            "No usable 3-10 second train reference for: "
            + ", ".join(missing)
        )

    if not reference_map:
        raise EmotionPipelineError(
            "No valid train reference could be selected"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            reference_map,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return reference_map


def write_v5_tts_config(
    output_path: Path,
    version: str,
    s1_checkpoint: Path,
    s2_checkpoint: Path,
    bert_dir: Path,
    cnhubert_dir: Path,
    device: str = "cuda",
    is_half: bool = True,
) -> Path:
    if version not in {"v5dev", "v5turbo"}:
        raise EmotionPipelineError(
            f"unsupported V5 version: {version}"
        )

    output_path = output_path.expanduser().resolve()

    config = {
        "custom": {
            "bert_base_path": str(bert_dir.expanduser().resolve()),
            "cnhuhbert_base_path": str(
                cnhubert_dir.expanduser().resolve()
            ),
            "device": device,
            "is_half": bool(is_half),
            "t2s_weights_path": str(
                s1_checkpoint.expanduser().resolve()
            ),
            "version": version,
            "vits_weights_path": str(
                s2_checkpoint.expanduser().resolve()
            ),
        }
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(
            config,
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    return output_path


def latest_feature_build(output_root: Path) -> Path:
    output_root = output_root.expanduser().resolve()

    builds = [
        path
        for path in output_root.glob("build_*")
        if path.is_dir() and (path / "build.json").is_file()
    ]

    if not builds:
        raise EmotionPipelineError(
            f"no valid feature build found under: {output_root}"
        )

    return max(
        builds,
        key=lambda path: (path / "build.json").stat().st_mtime_ns,
    )


def format_dataset_summary(report: dict) -> str:
    samples = int(report.get("samples", 0))
    emotions = report.get("emotions", {})

    return (
        f"samples={samples}, "
        f"emotions={emotions}"
    )
