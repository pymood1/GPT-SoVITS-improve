#!/usr/bin/env python3
"""Evaluate auto emotion labels against a small human-reviewed subset.

The evaluator is deliberately separate from training. It measures selective
coverage and class precision/recall for raw emotion2vec, current v2 relative
labels, the OOF critic and the conservative v3 state policy.

Expected review rows are produced by build_emotion_review_set.py. Fill at
least human_emotion; unreviewed rows are ignored.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable


class EvaluationError(RuntimeError):
    pass


DEFAULT_IGNORE = {"", "ambiguous", "unknown", "other", "unlabeled"}


def normalize_label(value: Any) -> str:
    return str(value or "").strip().lower()


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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise EvaluationError(f"file not found: {path}")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EvaluationError(
                    f"{path}:{line_number}: invalid JSON"
                ) from exc
            if not isinstance(row, dict):
                raise EvaluationError(
                    f"{path}:{line_number}: expected a JSON object"
                )
            rows.append(row)

    if not rows:
        raise EvaluationError(f"no JSONL rows in {path}")
    return rows


def merge_review_with_sidecar(
    review_rows: list[dict[str, Any]],
    sidecar_rows: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    if sidecar_rows is None:
        return [dict(row) for row in review_rows]

    sidecar: dict[str, dict[str, Any]] = {}
    for row in sidecar_rows:
        wav = str(row.get("wav") or "")
        if not wav:
            continue
        if wav in sidecar:
            raise EvaluationError(
                f"duplicate wav in sidecar: {wav}"
            )
        sidecar[wav] = row

    review_override_fields = {
        "review_rank",
        "review_priority",
        "review_reasons",
        "human_emotion",
        "human_confidence",
        "human_secondary_emotion",
        "human_notes",
    }

    merged = []
    for row in review_rows:
        wav = str(row.get("wav") or "")
        if not wav:
            raise EvaluationError("review row is missing wav")
        combined = dict(sidecar.get(wav, {}))
        combined["wav"] = wav
        for field in review_override_fields:
            if field in row:
                combined[field] = row[field]
        merged.append(combined)
    return merged


def macro_average(
    per_class: dict[str, dict[str, float | int | None]],
    key: str,
) -> float | None:
    values = [
        float(metrics[key])
        for metrics in per_class.values()
        if metrics.get(key) is not None
    ]
    if not values:
        return None
    return sum(values) / len(values)


def evaluate_system(
    rows: list[dict[str, Any]],
    *,
    name: str,
    predictor: Callable[[dict[str, Any]], str | None],
) -> dict[str, Any]:
    human_labels = sorted(
        {
            normalize_label(row["human_emotion"])
            for row in rows
        }
    )

    confusion: dict[str, Counter[str]] = {
        label: Counter()
        for label in human_labels
    }
    covered = 0
    correct = 0

    for row in rows:
        truth = normalize_label(row["human_emotion"])
        predicted = predictor(row)
        predicted = (
            normalize_label(predicted)
            if predicted is not None
            else ""
        )
        if not predicted:
            confusion[truth]["<abstain>"] += 1
            continue
        covered += 1
        confusion[truth][predicted] += 1
        if predicted == truth:
            correct += 1

    per_class: dict[str, dict[str, float | int | None]] = {}
    for label in human_labels:
        tp = confusion[label][label]
        fn = sum(confusion[label].values()) - tp
        fp = sum(
            confusion[other][label]
            for other in human_labels
            if other != label
        )
        support = sum(confusion[label].values())

        precision = (
            tp / (tp + fp)
            if tp + fp > 0
            else None
        )
        recall = (
            tp / support
            if support > 0
            else None
        )
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if (
                precision is not None
                and recall is not None
                and precision + recall > 0
            )
            else None
        )
        per_class[label] = {
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    confusion_json = {
        truth: dict(sorted(counts.items()))
        for truth, counts in confusion.items()
    }
    total = len(rows)
    return {
        "name": name,
        "reviewed_samples": total,
        "covered": covered,
        "abstained": total - covered,
        "coverage": covered / max(total, 1),
        "selective_accuracy": (
            correct / covered
            if covered > 0
            else None
        ),
        "overall_accuracy_with_abstain_wrong": (
            correct / max(total, 1)
        ),
        "macro_precision": macro_average(
            per_class,
            "precision",
        ),
        "macro_recall": macro_average(
            per_class,
            "recall",
        ),
        "macro_f1": macro_average(
            per_class,
            "f1",
        ),
        "per_class": per_class,
        "confusion": confusion_json,
        "known_human_labels": human_labels,
    }


def select_reviewed_rows(
    rows: list[dict[str, Any]],
    *,
    ignore_labels: set[str],
    include_low_confidence: bool,
) -> list[dict[str, Any]]:
    selected = []
    for row in rows:
        human = normalize_label(row.get("human_emotion"))
        if not human or human in ignore_labels:
            continue
        confidence = normalize_review_confidence(
            row.get("human_confidence")
        )
        if (
            not include_low_confidence
            and confidence in {"low", "낮음", "1"}
        ):
            continue
        selected.append(row)

    if not selected:
        raise EvaluationError(
            "no reviewed rows remain after filtering; "
            "fill human_emotion in the review JSONL"
        )
    return selected


def v3_selective_prediction(
    row: dict[str, Any],
) -> str | None:
    state = normalize_label(
        row.get("emotion_label_state_v3")
    )
    if state == "accepted_core":
        value = normalize_label(
            row.get("emotion_v3_candidate")
            or row.get("emotion_intended")
        )
        return value or None
    if state == "neutral_candidate":
        return "neutral"
    return None


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", required=True)
    parser.add_argument(
        "--sidecar",
        help=(
            "Optional latest sidecar/critic JSONL. If provided, its model "
            "fields are merged with human fields from --review by wav."
        ),
    )
    parser.add_argument("--output")
    parser.add_argument(
        "--ignore-human-labels",
        default="ambiguous,unknown,other,unlabeled",
    )
    parser.add_argument(
        "--include-low-confidence",
        action="store_true",
    )
    return parser


def main() -> int:
    args = make_parser().parse_args()
    try:
        review_path = Path(args.review).expanduser().resolve()
        review_rows = load_jsonl(review_path)
        sidecar_rows = (
            load_jsonl(
                Path(args.sidecar).expanduser().resolve()
            )
            if args.sidecar
            else None
        )
        rows = merge_review_with_sidecar(
            review_rows,
            sidecar_rows,
        )
        ignore = {
            normalize_label(value)
            for value in args.ignore_human_labels.split(",")
        } | {""}
        reviewed = select_reviewed_rows(
            rows,
            ignore_labels=ignore,
            include_low_confidence=args.include_low_confidence,
        )

        systems: list[
            tuple[
                str,
                Callable[[dict[str, Any]], str | None],
            ]
        ] = [
            (
                "raw_emotion2vec",
                lambda row: normalize_label(
                    row.get("emotion_raw_top1")
                    or row.get("emotion_predicted")
                )
                or None,
            ),
            (
                "speaker_relative_v2",
                lambda row: normalize_label(
                    row.get("emotion_intended")
                )
                or None,
            ),
        ]
        if any(
            row.get("emotion_critic_label")
            for row in reviewed
        ):
            systems.append(
                (
                    "critic",
                    lambda row: normalize_label(
                        row.get("emotion_critic_label")
                    )
                    or None,
                )
            )
        if any(
            row.get("emotion_refiner_label")
            for row in reviewed
        ):
            systems.append(
                (
                    "bounded_refiner",
                    lambda row: normalize_label(
                        row.get("emotion_refiner_label")
                    )
                    or None,
                )
            )
        if any(
            row.get("emotion_label_state_v3")
            for row in reviewed
        ):
            systems.append(
                (
                    "v3_selective",
                    v3_selective_prediction,
                )
            )

        evaluations = {
            name: evaluate_system(
                reviewed,
                name=name,
                predictor=predictor,
            )
            for name, predictor in systems
        }

        human_counts = Counter(
            normalize_label(row["human_emotion"])
            for row in reviewed
        )
        confidence_counts = Counter(
            normalize_review_confidence(
                row.get("human_confidence")
            )
            or "unspecified"
            for row in reviewed
        )
        report = {
            "review_file": str(review_path),
            "sidecar_file": (
                str(
                    Path(
                        args.sidecar
                    ).expanduser().resolve()
                )
                if args.sidecar
                else None
            ),
            "reviewed_samples": len(reviewed),
            "human_label_counts": dict(
                sorted(human_counts.items())
            ),
            "human_confidence_counts": dict(
                sorted(confidence_counts.items())
            ),
            "systems": evaluations,
            "notes": [
                (
                    "v3_selective may abstain. Compare selective accuracy/"
                    "precision together with coverage rather than accuracy "
                    "alone. Critic/refiner standalone rows do not abstain."
                ),
                (
                    "Human labels are calibration evidence, not assumed "
                    "perfect ground truth; low-confidence human rows are "
                    "excluded by default."
                ),
            ],
        }

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
        EvaluationError,
        OSError,
        ValueError,
    ) as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
