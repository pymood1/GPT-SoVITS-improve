#!/usr/bin/env python3
"""Fast synthetic smoke tests for emotion labeling utilities.

No FunASR model or real audio is required. This checks only deterministic
math/data-flow: uncertainty diagnostics, review selection and grouped OOF
critic learning on an intentionally separable synthetic embedding dataset.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.auto_emotion_sidecar import prediction_diagnostics  # noqa: E402
from tools.calibrate_emotion_seed_thresholds import calibrate  # noqa: E402
from tools.build_emotion_review_set import select_review_rows  # noqa: E402
from tools.emotion_label_refiner import (  # noqa: E402
    CriticConfig,
    RefinerConfig,
    apply_critic_outputs,
    apply_refiner_outputs,
    crossfit_critic,
    crossfit_refiner,
    fit_final_critic,
    fit_final_refiner,
    materialize_training_sidecar,
    select_seed_indices,
)
from tools.evaluate_emotion_review import evaluate_system  # noqa: E402


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_prediction_diagnostics() -> None:
    diagnostics = prediction_diagnostics(
        {
            "happy": 0.60,
            "surprised": 0.20,
            "neutral": 0.10,
            "other": 0.06,
            "unknown": 0.04,
        }
    )
    require(
        diagnostics["emotion_raw_top1"] == "happy",
        "raw top-1 mismatch",
    )
    require(
        abs(
            float(diagnostics["emotion_top1_margin"])
            - 0.40
        )
        < 1e-6,
        "top-1 margin mismatch",
    )
    require(
        abs(float(diagnostics["emotion_ood_mass"]) - 0.10)
        < 1e-6,
        "OOD mass mismatch",
    )
    entropy = float(diagnostics["emotion_entropy"])
    require(
        0.0 < entropy < 1.0,
        "normalized entropy must be inside (0,1)",
    )


def synthetic_sidecar_rows() -> list[dict]:
    rows = []
    labels = [
        "happy",
        "angry",
        "disgusted",
        "surprised",
    ]
    for label in labels:
        for sample_index in range(4):
            entropy = 0.15 + 0.15 * sample_index
            margin = 0.70 - 0.12 * sample_index
            ood = 0.02 + 0.02 * sample_index
            rows.append(
                {
                    "wav": f"{label}_{sample_index}.wav",
                    "emotion_intended": label,
                    "emotion_predicted": label,
                    "emotion_raw_top1": label,
                    "emotion_raw_top2": "neutral",
                    "emotion_predicted_confidence": (
                        0.8 - 0.05 * sample_index
                    ),
                    "emotion_entropy": entropy,
                    "emotion_top1_margin": margin,
                    "emotion_ood_mass": ood,
                    "emotion_relative_score": (
                        1.0 - 0.1 * sample_index
                    ),
                    "emotion_relative_confidence": (
                        0.75 - 0.05 * sample_index
                    ),
                    "emotion_relative_gap": (
                        0.8 - 0.15 * sample_index
                    ),
                    "emotion_raw_scores": {
                        label: 0.75,
                        "neutral": 0.10,
                        "other": 0.05,
                        "unknown": 0.02,
                    },
                }
            )

    rows.append(
        {
            "wav": "neutral_boundary.wav",
            "emotion_intended": "neutral",
            "emotion_predicted": "happy",
            "emotion_raw_top1": "happy",
            "emotion_raw_top2": "neutral",
            "emotion_entropy": 0.92,
            "emotion_top1_margin": 0.02,
            "emotion_ood_mass": 0.04,
            "emotion_relative_score": 0.24,
            "emotion_relative_confidence": 0.34,
            "emotion_relative_gap": 0.01,
            "emotion_raw_scores": {
                "happy": 0.28,
                "neutral": 0.26,
                "other": 0.04,
                "unknown": 0.02,
            },
        }
    )
    rows.append(
        {
            "wav": "ood.wav",
            "emotion_intended": "neutral",
            "emotion_predicted": "unknown",
            "emotion_raw_top1": "unknown",
            "emotion_raw_top2": "other",
            "emotion_entropy": 0.70,
            "emotion_top1_margin": 0.10,
            "emotion_ood_mass": 0.70,
            "emotion_relative_score": 0.05,
            "emotion_relative_confidence": 0.10,
            "emotion_relative_gap": 0.01,
            "emotion_raw_scores": {
                "unknown": 0.45,
                "other": 0.25,
                "neutral": 0.10,
            },
        }
    )
    return rows


def test_review_selector() -> None:
    rows = synthetic_sidecar_rows()
    selected, report = select_review_rows(
        rows,
        limit=12,
        anchors_per_class=1,
        boundaries_per_class=1,
        rare_class_max=4,
        ood_cases=2,
        neutral_cases=2,
    )
    require(selected, "review selector returned no rows")
    require(
        len(selected) <= 12,
        "review selector exceeded limit",
    )
    reasons = {
        reason
        for row in selected
        for reason in row["review_reasons"]
    }
    require(
        any(
            reason.startswith("anchor:")
            for reason in reasons
        ),
        "review selector did not include an anchor",
    )
    require(
        "high_ood" in reasons,
        "review selector did not include OOD sample",
    )
    require(
        report["selected"] == len(selected),
        "review report count mismatch",
    )


def synthetic_critic_problem() -> tuple[
    list[dict],
    torch.Tensor,
]:
    torch.manual_seed(20261113)
    emotions = (
        "happy",
        "angry",
        "disgusted",
        "surprised",
    )
    rows = []
    embeddings = []

    dim = 16
    samples_per_class = 8
    for class_id, emotion in enumerate(emotions):
        center = torch.zeros(dim)
        center[class_id] = 5.0
        center[4 + class_id] = 2.0

        for sample_id in range(samples_per_class):
            embedding = (
                center
                + 0.12 * torch.randn(dim)
            )
            embeddings.append(embedding)
            rows.append(
                {
                    "wav": (
                        f"{emotion}_{sample_id}.wav"
                    ),
                    "emotion_intended": emotion,
                    "emotion_raw_top1": emotion,
                    "emotion_ood_mass": 0.02,
                    "emotion_entropy": 0.20,
                    "_text_group_id": (
                        f"{emotion}:synthetic:{sample_id}"
                    ),
                    "pair_group_id": None,
                    "source_id": None,
                }
            )

    return rows, torch.stack(embeddings)


def test_crossfit_critic() -> None:
    rows, embeddings = synthetic_critic_problem()
    emotions = (
        "happy",
        "angry",
        "disgusted",
        "surprised",
    )
    config = CriticConfig(
        emotions=emotions,
        pca_dim=8,
        rank=2,
        temperature=0.20,
        ranking_margin=0.20,
        steps=120,
        learning_rate=3e-3,
        seed=1234,
        max_folds=4,
    )
    seed_indices = list(range(len(rows)))

    oof_logits, _, metrics = crossfit_critic(
        rows,
        embeddings,
        seed_indices,
        config=config,
    )
    macro_accuracy = float(
        metrics["oof_macro_accuracy"]
    )
    macro_auc = float(
        metrics["oof_macro_auc"] or 0.0
    )
    require(
        macro_accuracy >= 0.85,
        (
            "synthetic critic macro accuracy too low: "
            f"{macro_accuracy:.3f}"
        ),
    )
    require(
        macro_auc >= 0.95,
        (
            "synthetic critic macro AUC too low: "
            f"{macro_auc:.3f}"
        ),
    )

    final_result = fit_final_critic(
        embeddings,
        seed_indices,
        rows,
        config=config,
    )
    output_rows, states = apply_critic_outputs(
        rows,
        seed_indices=seed_indices,
        oof_logits=oof_logits,
        final_logits=final_result["all_logits"],
        emotions=emotions,
        reject_ood=0.50,
        max_neutral_entropy=0.65,
    )
    accepted = sum(
        row["emotion_label_state_v3"]
        == "accepted_core"
        for row in output_rows
    )
    require(
        accepted / len(output_rows) >= 0.85,
        (
            "synthetic critic accepted_core ratio "
            "is unexpectedly low"
        ),
    )
    require(
        states.get("accepted_core", 0) == accepted,
        "critic state report mismatch",
    )


def test_crossfit_refiner() -> None:
    rows, embeddings = synthetic_critic_problem()
    emotions = (
        "happy",
        "angry",
        "disgusted",
        "surprised",
    )
    critic_config = CriticConfig(
        emotions=emotions,
        pca_dim=8,
        rank=2,
        temperature=0.20,
        ranking_margin=0.20,
        steps=100,
        learning_rate=3e-3,
        seed=4321,
        max_folds=4,
    )
    refiner_config = RefinerConfig(
        delta_max=1.0,
        steps=100,
        learning_rate=2e-3,
        teacher_weight=0.35,
        critic_weight=0.50,
    )
    seed_indices = list(range(len(rows)))

    critic_oof, _, critic_metrics = crossfit_critic(
        rows,
        embeddings,
        seed_indices,
        config=critic_config,
    )
    require(
        float(critic_metrics["oof_macro_accuracy"]) >= 0.85,
        "synthetic prerequisite critic is unexpectedly weak",
    )

    refiner_oof, refiner_metrics = crossfit_refiner(
        rows,
        embeddings,
        seed_indices,
        critic_config=critic_config,
        refiner_config=refiner_config,
    )
    require(
        float(refiner_metrics["oof_macro_accuracy"]) >= 0.85,
        (
            "synthetic refiner macro accuracy too low: "
            f"{refiner_metrics['oof_macro_accuracy']:.3f}"
        ),
    )
    require(
        float(refiner_metrics["oof_macro_auc"] or 0.0) >= 0.95,
        (
            "synthetic refiner macro AUC too low: "
            f"{refiner_metrics['oof_macro_auc']:.3f}"
        ),
    )

    final_critic = fit_final_critic(
        embeddings,
        seed_indices,
        rows,
        config=critic_config,
    )
    final_refiner = fit_final_refiner(
        rows,
        seed_indices,
        final_critic["all_logits"],
        critic_config=critic_config,
        refiner_config=refiner_config,
    )
    output_rows, states = apply_refiner_outputs(
        rows,
        seed_indices=seed_indices,
        critic_oof_logits=critic_oof,
        critic_final_logits=final_critic["all_logits"],
        refiner_oof_logits=refiner_oof,
        refiner_final_logits=final_refiner["all_logits"],
        emotions=emotions,
        reject_ood=0.50,
        max_neutral_entropy=0.65,
        min_critic_margin=0.05,
        min_refiner_margin=0.05,
    )
    accepted = sum(
        row["emotion_label_state_v3"]
        == "accepted_core"
        for row in output_rows
    )
    require(
        accepted / len(output_rows) >= 0.85,
        "synthetic refiner accepted_core ratio is too low",
    )
    require(
        states.get("accepted_core", 0) == accepted,
        "refiner state report mismatch",
    )
    require(
        all(
            row.get("emotion_refiner_oof") is True
            for row in output_rows
        ),
        "seed-only synthetic refiner rows must be OOF",
    )


def test_reviewed_seed_override() -> None:
    rows, embeddings = synthetic_critic_problem()
    rows[0]["emotion_intended"] = "neutral"
    rows[0]["_seed_label_override"] = "happy"
    rows[0]["_seed_source"] = "human_review"

    seeds, active = select_seed_indices(
        rows,
        ("happy", "angry", "disgusted", "surprised"),
        max_ood=0.35,
        max_entropy=0.80,
        min_raw_margin=0.0,
        min_relative_score=0.0,
        min_relative_confidence=0.0,
        min_relative_gap=0.0,
        min_relative_z=0.0,
        min_relative_z_gap=0.0,
        min_seed_per_class=3,
    )
    require(
        0 in seeds,
        "high-confidence reviewed seed override was not selected",
    )
    require(
        "happy" in active,
        "reviewed happy seed class disappeared",
    )

    config = CriticConfig(
        emotions=active,
        pca_dim=8,
        rank=2,
        temperature=0.20,
        ranking_margin=0.20,
        steps=100,
        learning_rate=3e-3,
        seed=777,
        max_folds=4,
    )
    oof_logits, _, _ = crossfit_critic(
        rows,
        embeddings,
        seeds,
        config=config,
    )
    final_result = fit_final_critic(
        embeddings,
        seeds,
        rows,
        config=config,
    )
    output_rows, _ = apply_critic_outputs(
        rows,
        seed_indices=seeds,
        oof_logits=oof_logits,
        final_logits=final_result["all_logits"],
        emotions=active,
        reject_ood=0.50,
        max_neutral_entropy=0.65,
    )
    require(
        output_rows[0]["emotion_intended"] == "neutral",
        "human anchor must not overwrite baseline emotion_intended",
    )
    require(
        output_rows[0]["emotion_v3_candidate"] == "happy",
        "reviewed seed candidate must use the human anchor label",
    )
    require(
        output_rows[0]["emotion_label_state_v3"]
        == "accepted_core",
        "reviewed seed should be evaluated against its anchor label",
    )


def test_training_sidecar_materialization() -> None:
    rows = [
        {
            "wav": "core.wav",
            "emotion_intended": "happy",
            "emotion_label_state_v3": "accepted_core",
            "emotion_v3_candidate": "happy",
        },
        {
            "wav": "confirmed.wav",
            "emotion_intended": "angry",
            "emotion_label_state_v3": "refiner_confirmed_candidate",
            "emotion_v3_candidate": "angry",
        },
        {
            "wav": "neutral.wav",
            "emotion_intended": "neutral",
            "emotion_label_state_v3": "neutral_candidate",
            "emotion_v3_candidate": None,
        },
        {
            "wav": "conflict.wav",
            "emotion_intended": "surprised",
            "emotion_label_state_v3": "conflict",
            "emotion_v3_candidate": None,
        },
    ]

    core, counts = materialize_training_sidecar(
        rows,
        policy="core",
    )
    require(
        core[0]["emotion_intended"] == "happy",
        "accepted_core must remain a target",
    )
    require(
        core[1]["emotion_intended"] is None,
        "confirmed candidate must not enter core policy",
    )
    require(
        core[2]["emotion_intended"] == "neutral",
        "neutral_candidate must remain neutral",
    )
    require(
        core[3]["emotion_intended"] is None,
        "conflict must be excluded",
    )
    require(
        counts.get("target:happy", 0) == 1,
        "core training count mismatch",
    )

    confirmed, _ = materialize_training_sidecar(
        rows,
        policy="core+confirmed",
    )
    require(
        confirmed[1]["emotion_intended"] == "angry",
        "confirmed policy must include confirmed candidates",
    )


def test_threshold_calibrator() -> None:
    rows = []
    labels = ("happy", "angry")
    for label in labels:
        for index in range(4):
            rows.append(
                {
                    "human_emotion": label,
                    "emotion_intended": label,
                    "emotion_raw_top1": label,
                    "emotion_ood_mass": 0.05,
                    "emotion_entropy": 0.20,
                    "emotion_top1_margin": 0.60,
                    "emotion_relative_score": 0.80,
                    "emotion_relative_confidence": 0.70,
                    "emotion_relative_gap": 0.50,
                    "emotion_relative_z": 1.5,
                    "emotion_relative_z_gap": 0.70,
                }
            )

    report = calibrate(
        rows,
        emotions=labels,
        grids={
            "max_ood": (0.10, 0.30),
            "max_entropy": (0.40, 0.80),
            "min_raw_margin": (0.10,),
            "min_relative_score": (0.25,),
            "min_relative_confidence": (0.35,),
            "min_relative_gap": (0.10,),
            "min_relative_z": (0.50,),
            "min_relative_z_gap": (0.25,),
        },
        min_coverage=0.50,
        min_accepted=4,
        min_predicted_classes=2,
        top_n=5,
    )
    require(
        report["best"]["selective_accuracy"] == 1.0,
        "synthetic threshold calibration should be perfect",
    )
    require(
        "--seed-max-ood" in report["recommended_cli"],
        "calibrator did not emit seed CLI recommendation",
    )


def test_review_evaluator() -> None:
    rows = [
        {
            "human_emotion": "happy",
            "emotion_intended": "happy",
        },
        {
            "human_emotion": "happy",
            "emotion_intended": "angry",
        },
        {
            "human_emotion": "angry",
            "emotion_intended": "angry",
        },
    ]
    report = evaluate_system(
        rows,
        name="synthetic",
        predictor=lambda row: row.get(
            "emotion_intended"
        ),
    )
    require(
        math.isclose(
            float(report["selective_accuracy"]),
            2.0 / 3.0,
            rel_tol=1e-6,
        ),
        "review evaluator accuracy mismatch",
    )


def main() -> int:
    tests = [
        test_prediction_diagnostics,
        test_review_selector,
        test_crossfit_critic,
        test_crossfit_refiner,
        test_reviewed_seed_override,
        test_training_sidecar_materialization,
        test_threshold_calibrator,
        test_review_evaluator,
    ]
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")
    print(
        f"[PASS] emotion labeling self-test: "
        f"{len(tests)} checks"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
