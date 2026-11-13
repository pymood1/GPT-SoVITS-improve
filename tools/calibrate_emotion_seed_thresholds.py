#!/usr/bin/env python3
"""Calibrate conservative auto-emotion seed gates on a reviewed subset.

This tool does not modify the sidecar or defaults. It searches a modest grid of
uncertainty / speaker-relative thresholds and ranks configurations by precision
first, then selective accuracy and coverage.

The review subset is intentionally information-dense rather than IID. Treat
the result as a target-domain calibration suggestion, not an unbiased estimate
of full-dataset accuracy.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.evaluate_emotion_review import (  # noqa: E402
    evaluate_system,
    load_jsonl,
    merge_review_with_sidecar,
    normalize_label,
    select_reviewed_rows,
)


class CalibrationError(RuntimeError):
    pass


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def parse_float_grid(value: str, *, name: str) -> tuple[float, ...]:
    values: list[float] = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        try:
            number = float(item)
        except ValueError as exc:
            raise CalibrationError(
                f"{name}: invalid float {item!r}"
            ) from exc
        if not math.isfinite(number):
            raise CalibrationError(
                f"{name}: non-finite value {item!r}"
            )
        values.append(number)
    if not values:
        raise CalibrationError(f"{name}: grid is empty")
    return tuple(sorted(set(values)))


def prediction_for_thresholds(
    row: dict[str, Any],
    *,
    emotions: set[str],
    max_ood: float,
    max_entropy: float,
    min_raw_margin: float,
    min_relative_score: float,
    min_relative_confidence: float,
    min_relative_gap: float,
    min_relative_z: float,
    min_relative_z_gap: float,
) -> str | None:
    label = normalize_label(row.get("emotion_intended"))
    if label not in emotions:
        return None
    if row.get("emotion_prediction_error"):
        return None
    if normalize_label(row.get("emotion_raw_top1")) in {
        "other",
        "unknown",
    }:
        return None
    if safe_float(row.get("emotion_ood_mass"), 1.0) > max_ood:
        return None
    if safe_float(row.get("emotion_entropy"), 1.0) > max_entropy:
        return None
    if (
        safe_float(row.get("emotion_top1_margin"), 0.0)
        < min_raw_margin
    ):
        return None
    if (
        safe_float(row.get("emotion_relative_score"), 0.0)
        < min_relative_score
    ):
        return None
    if (
        safe_float(row.get("emotion_relative_confidence"), 0.0)
        < min_relative_confidence
    ):
        return None
    if (
        safe_float(row.get("emotion_relative_gap"), 0.0)
        < min_relative_gap
    ):
        return None
    if (
        safe_float(row.get("emotion_relative_z"), 0.0)
        < min_relative_z
    ):
        return None
    if (
        safe_float(row.get("emotion_relative_z_gap"), 0.0)
        < min_relative_z_gap
    ):
        return None
    return label


def metric_or_negative(value: Any) -> float:
    if value is None:
        return -1.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return -1.0


def calibrate(
    rows: list[dict[str, Any]],
    *,
    emotions: tuple[str, ...],
    grids: dict[str, tuple[float, ...]],
    min_coverage: float,
    min_accepted: int,
    top_n: int,
) -> dict[str, Any]:
    emotion_set = set(emotions)
    candidates: list[dict[str, Any]] = []

    grid_names = (
        "max_ood",
        "max_entropy",
        "min_raw_margin",
        "min_relative_score",
        "min_relative_confidence",
        "min_relative_gap",
        "min_relative_z",
        "min_relative_z_gap",
    )
    combinations = itertools.product(
        *(grids[name] for name in grid_names)
    )

    for values in combinations:
        params = dict(zip(grid_names, values))
        report = evaluate_system(
            rows,
            name="calibration_candidate",
            predictor=lambda row, p=params: prediction_for_thresholds(
                row,
                emotions=emotion_set,
                **p,
            ),
        )
        if report["covered"] < min_accepted:
            continue
        if float(report["coverage"]) < min_coverage:
            continue

        predicted_counts: Counter[str] = Counter()
        for row in rows:
            label = prediction_for_thresholds(
                row,
                emotions=emotion_set,
                **params,
            )
            if label:
                predicted_counts[label] += 1

        candidates.append(
            {
                "params": params,
                "covered": report["covered"],
                "coverage": report["coverage"],
                "selective_accuracy": report[
                    "selective_accuracy"
                ],
                "macro_precision": report["macro_precision"],
                "macro_recall": report["macro_recall"],
                "macro_f1": report["macro_f1"],
                "predicted_class_counts": dict(
                    sorted(predicted_counts.items())
                ),
                "predicted_class_count": len(predicted_counts),
            }
        )

    if not candidates:
        raise CalibrationError(
            "no threshold combination satisfies the requested "
            "minimum coverage/accepted constraints"
        )

    candidates.sort(
        key=lambda item: (
            -metric_or_negative(item["macro_precision"]),
            -metric_or_negative(item["selective_accuracy"]),
            -int(item["predicted_class_count"]),
            -float(item["coverage"]),
            -metric_or_negative(item["macro_f1"]),
        )
    )
    top = candidates[:top_n]
    best = top[0]

    params = best["params"]
    recommended_cli = " ".join(
        [
            f"--seed-max-ood {params['max_ood']}",
            f"--seed-max-entropy {params['max_entropy']}",
            (
                "--seed-min-raw-margin "
                f"{params['min_raw_margin']}"
            ),
            (
                "--seed-min-relative-score "
                f"{params['min_relative_score']}"
            ),
            (
                "--seed-min-relative-confidence "
                f"{params['min_relative_confidence']}"
            ),
            (
                "--seed-min-relative-gap "
                f"{params['min_relative_gap']}"
            ),
            (
                "--seed-min-relative-z "
                f"{params['min_relative_z']}"
            ),
            (
                "--seed-min-relative-z-gap "
                f"{params['min_relative_z_gap']}"
            ),
        ]
    )

    return {
        "reviewed_samples": len(rows),
        "emotions": list(emotions),
        "grid_combinations_considered": math.prod(
            len(grids[name])
            for name in grid_names
        ),
        "valid_candidates": len(candidates),
        "best": best,
        "recommended_cli": recommended_cli,
        "top_candidates": top,
        "notes": [
            (
                "The review set is actively selected and therefore not an "
                "IID estimate of dataset-wide accuracy."
            ),
            (
                "Ranking is precision-first. Do not promote thresholds to "
                "defaults until downstream Controller OOF/listening also "
                "improves."
            ),
            (
                "A tiny reviewed subset can overfit threshold selection. "
                "Prefer broad plateaus of similar candidates over one sharp "
                "winner."
            ),
        ],
    }


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", required=True)
    parser.add_argument("--sidecar", required=True)
    parser.add_argument("--output")
    parser.add_argument(
        "--emotions",
        default="happy,angry,disgusted,surprised",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--min-accepted",
        type=int,
        default=6,
    )
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument(
        "--include-low-confidence",
        action="store_true",
    )

    parser.add_argument(
        "--grid-max-ood",
        default="0.15,0.25,0.35,0.50",
    )
    parser.add_argument(
        "--grid-max-entropy",
        default="0.55,0.65,0.75,0.85",
    )
    parser.add_argument(
        "--grid-min-raw-margin",
        default="0.05,0.10,0.20",
    )
    parser.add_argument(
        "--grid-min-relative-score",
        default="0.20,0.25,0.35",
    )
    parser.add_argument(
        "--grid-min-relative-confidence",
        default="0.30,0.35,0.45",
    )
    parser.add_argument(
        "--grid-min-relative-gap",
        default="0.0,0.10,0.25",
    )
    parser.add_argument(
        "--grid-min-relative-z",
        default="0.0,0.50,1.0,1.5",
    )
    parser.add_argument(
        "--grid-min-relative-z-gap",
        default="0.0,0.25,0.50",
    )
    return parser


def main() -> int:
    args = make_parser().parse_args()
    try:
        if not 0.0 <= args.min_coverage <= 1.0:
            raise CalibrationError(
                "--min-coverage must be between 0 and 1"
            )
        if args.min_accepted < 1:
            raise CalibrationError("--min-accepted must be >= 1")
        if args.top_n < 1:
            raise CalibrationError("--top-n must be >= 1")

        emotions = tuple(
            normalize_label(value)
            for value in args.emotions.split(",")
            if normalize_label(value)
        )
        if len(emotions) < 2 or len(set(emotions)) != len(emotions):
            raise CalibrationError(
                "--emotions must contain at least two unique labels"
            )

        review_rows = load_jsonl(
            Path(args.review).expanduser().resolve()
        )
        sidecar_rows = load_jsonl(
            Path(args.sidecar).expanduser().resolve()
        )
        merged = merge_review_with_sidecar(
            review_rows,
            sidecar_rows,
        )
        reviewed = select_reviewed_rows(
            merged,
            ignore_labels={
                "",
                "ambiguous",
                "unknown",
                "other",
                "unlabeled",
            },
            include_low_confidence=args.include_low_confidence,
        )

        grids = {
            "max_ood": parse_float_grid(
                args.grid_max_ood,
                name="--grid-max-ood",
            ),
            "max_entropy": parse_float_grid(
                args.grid_max_entropy,
                name="--grid-max-entropy",
            ),
            "min_raw_margin": parse_float_grid(
                args.grid_min_raw_margin,
                name="--grid-min-raw-margin",
            ),
            "min_relative_score": parse_float_grid(
                args.grid_min_relative_score,
                name="--grid-min-relative-score",
            ),
            "min_relative_confidence": parse_float_grid(
                args.grid_min_relative_confidence,
                name="--grid-min-relative-confidence",
            ),
            "min_relative_gap": parse_float_grid(
                args.grid_min_relative_gap,
                name="--grid-min-relative-gap",
            ),
            "min_relative_z": parse_float_grid(
                args.grid_min_relative_z,
                name="--grid-min-relative-z",
            ),
            "min_relative_z_gap": parse_float_grid(
                args.grid_min_relative_z_gap,
                name="--grid-min-relative-z-gap",
            ),
        }
        report = calibrate(
            reviewed,
            emotions=emotions,
            grids=grids,
            min_coverage=args.min_coverage,
            min_accepted=args.min_accepted,
            top_n=args.top_n,
        )

        output = json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        print(output)
        if args.output:
            output_path = Path(
                args.output
            ).expanduser().resolve()
            output_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            output_path.write_text(
                output + "\n",
                encoding="utf-8",
            )
    except (
        CalibrationError,
        OSError,
        ValueError,
    ) as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
