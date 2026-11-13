#!/usr/bin/env python3
"""Select a small, information-dense emotion review set from a sidecar JSONL.

The selector never changes labels. It ranks samples that are most useful for
human calibration: class anchors, class boundaries, rare classes, OOD cases,
raw-vs-relative disagreement and suspicious neutral/default assignments.

Example:
    python tools/build_emotion_review_set.py \
      --sidecar emotion_sidecar.jsonl \
      --output emotion_review.jsonl \
      --limit 30
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

NON_TARGET = {"", "neutral", "default", "unlabeled", "unknown", "other"}


class ReviewSetError(RuntimeError):
    pass


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ReviewSetError(f"sidecar not found: {path}")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ReviewSetError(
                    f"{path}:{line_number}: invalid JSON"
                ) from exc
            if not isinstance(row, dict):
                raise ReviewSetError(
                    f"{path}:{line_number}: expected a JSON object"
                )
            if not row.get("wav"):
                raise ReviewSetError(
                    f"{path}:{line_number}: missing wav"
                )
            rows.append(row)

    if not rows:
        raise ReviewSetError("sidecar is empty")
    return rows


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def normalized_label(value: Any) -> str:
    return str(value or "").strip().lower()


def uncertainty_score(row: dict[str, Any]) -> float:
    """Higher means more useful as a boundary/uncertainty review case."""
    entropy = min(
        max(safe_float(row.get("emotion_entropy"), 1.0), 0.0),
        1.0,
    )
    raw_margin = min(
        max(safe_float(row.get("emotion_top1_margin"), 0.0), 0.0),
        1.0,
    )
    relative_gap = max(
        safe_float(row.get("emotion_relative_gap"), 0.0),
        0.0,
    )
    gap_certainty = 1.0 - math.exp(-relative_gap)
    return (
        0.50 * entropy
        + 0.30 * (1.0 - raw_margin)
        + 0.20 * (1.0 - gap_certainty)
    )


def anchor_score(row: dict[str, Any]) -> float:
    """Higher means a cleaner candidate for a human-confirmed class anchor."""
    entropy = min(
        max(safe_float(row.get("emotion_entropy"), 1.0), 0.0),
        1.0,
    )
    raw_margin = min(
        max(safe_float(row.get("emotion_top1_margin"), 0.0), 0.0),
        1.0,
    )
    ood = min(
        max(safe_float(row.get("emotion_ood_mass"), 0.0), 0.0),
        1.0,
    )
    relative_gap = max(
        safe_float(row.get("emotion_relative_gap"), 0.0),
        0.0,
    )
    relative_certainty = 1.0 - math.exp(-relative_gap)
    relative_confidence = min(
        max(
            safe_float(
                row.get("emotion_relative_confidence"),
                0.0,
            ),
            0.0,
        ),
        1.0,
    )
    return (
        0.25 * (1.0 - entropy)
        + 0.20 * raw_margin
        + 0.20 * (1.0 - ood)
        + 0.20 * relative_certainty
        + 0.15 * relative_confidence
    )


def add_reason(
    candidates: dict[int, dict[str, Any]],
    index: int,
    reason: str,
    weight: float,
) -> None:
    item = candidates.setdefault(
        index,
        {
            "reasons": [],
            "priority": 0.0,
        },
    )
    if reason not in item["reasons"]:
        item["reasons"].append(reason)
    item["priority"] += float(weight)


def select_review_rows(
    rows: list[dict[str, Any]],
    *,
    limit: int,
    anchors_per_class: int,
    boundaries_per_class: int,
    rare_class_max: int,
    ood_cases: int,
    neutral_cases: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if limit < 1:
        raise ReviewSetError("--limit must be >= 1")

    accepted = [
        normalized_label(row.get("emotion_intended"))
        for row in rows
    ]
    class_counts = Counter(
        label for label in accepted
        if label not in NON_TARGET
    )

    by_class: dict[str, list[int]] = defaultdict(list)
    neutral_indices: list[int] = []
    for index, label in enumerate(accepted):
        if label in NON_TARGET:
            if label in {"neutral", "default"}:
                neutral_indices.append(index)
            continue
        by_class[label].append(index)

    candidates: dict[int, dict[str, Any]] = {}

    # Review must include clean exemplars as well as edge cases; otherwise a
    # future prototype/calibration step would be anchored only by ambiguity.
    for label, indices in sorted(by_class.items()):
        ordered = sorted(
            indices,
            key=lambda idx: (
                -anchor_score(rows[idx]),
                str(rows[idx].get("wav") or ""),
            ),
        )
        for index in ordered[:anchors_per_class]:
            add_reason(
                candidates,
                index,
                f"anchor:{label}",
                2.0 + anchor_score(rows[index]),
            )

        boundary_order = sorted(
            indices,
            key=lambda idx: (
                -uncertainty_score(rows[idx]),
                str(rows[idx].get("wav") or ""),
            ),
        )
        for index in boundary_order[:boundaries_per_class]:
            add_reason(
                candidates,
                index,
                f"boundary:{label}",
                2.4 + uncertainty_score(rows[index]),
            )

    # A rare class is disproportionately valuable to verify because a handful
    # of wrong pseudo-labels can define the entire controller direction.
    rare_labels = {
        label
        for label, count in class_counts.items()
        if count <= rare_class_max
    }
    for label in sorted(rare_labels):
        for index in by_class[label]:
            add_reason(
                candidates,
                index,
                f"rare_class:{label}",
                3.0,
            )

    # OOD/open-set evidence should be reviewed before a refiner is allowed to
    # rescue an ambiguous sample into a known class.
    ood_order = sorted(
        range(len(rows)),
        key=lambda idx: (
            -safe_float(rows[idx].get("emotion_ood_mass"), 0.0),
            str(rows[idx].get("wav") or ""),
        ),
    )
    for index in ood_order[:ood_cases]:
        ood = safe_float(rows[index].get("emotion_ood_mass"), 0.0)
        raw_top1 = normalized_label(rows[index].get("emotion_raw_top1"))
        if ood > 0.0 or raw_top1 in {"unknown", "other"}:
            add_reason(
                candidates,
                index,
                "high_ood",
                2.5 + min(max(ood, 0.0), 1.0),
            )

    # Absolute teacher top-1 and speaker-relative label answer different
    # questions. Their disagreement is not automatically an error; it is an
    # information-dense review target.
    for index, row in enumerate(rows):
        intended = normalized_label(row.get("emotion_intended"))
        raw_top1 = normalized_label(row.get("emotion_raw_top1"))
        if (
            intended
            and intended not in {"default", "unlabeled"}
            and raw_top1
            and intended != raw_top1
        ):
            add_reason(
                candidates,
                index,
                "raw_vs_relative_disagreement",
                2.2,
            )

    # v2 maps every non-accepted sample to neutral/default. Review a mix of
    # clean-looking neutral anchors and suspicious neutral assignments.
    if neutral_indices and neutral_cases > 0:
        clean_neutral = sorted(
            neutral_indices,
            key=lambda idx: (
                -(
                    1.0
                    - min(
                        max(
                            safe_float(
                                rows[idx].get("emotion_entropy"),
                                1.0,
                            ),
                            0.0,
                        ),
                        1.0,
                    )
                ),
                str(rows[idx].get("wav") or ""),
            ),
        )
        suspicious_neutral = sorted(
            neutral_indices,
            key=lambda idx: (
                -uncertainty_score(rows[idx]),
                -safe_float(
                    rows[idx].get("emotion_relative_score"),
                    0.0,
                ),
                str(rows[idx].get("wav") or ""),
            ),
        )
        clean_count = max(1, neutral_cases // 2)
        suspicious_count = max(1, neutral_cases - clean_count)
        for index in clean_neutral[:clean_count]:
            add_reason(
                candidates,
                index,
                "neutral_anchor",
                1.8 + anchor_score(rows[index]),
            )
        for index in suspicious_neutral[:suspicious_count]:
            add_reason(
                candidates,
                index,
                "neutral_boundary",
                2.3 + uncertainty_score(rows[index]),
            )

    ranked = sorted(
        candidates.items(),
        key=lambda item: (
            -item[1]["priority"],
            str(rows[item[0]].get("wav") or ""),
        ),
    )[:limit]

    selected: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    for rank, (index, meta) in enumerate(ranked, 1):
        row = rows[index]
        reasons = list(meta["reasons"])
        reason_counts.update(reasons)
        selected.append(
            {
                "review_rank": rank,
                "review_priority": round(
                    float(meta["priority"]),
                    6,
                ),
                "review_reasons": reasons,
                "wav": row.get("wav"),
                "emotion_intended": row.get("emotion_intended"),
                "emotion_predicted": row.get("emotion_predicted"),
                "emotion_predicted_confidence": row.get(
                    "emotion_predicted_confidence"
                ),
                "emotion_raw_top1": row.get("emotion_raw_top1"),
                "emotion_raw_top2": row.get("emotion_raw_top2"),
                "emotion_entropy": row.get("emotion_entropy"),
                "emotion_top1_margin": row.get("emotion_top1_margin"),
                "emotion_ood_mass": row.get("emotion_ood_mass"),
                "emotion_relative_score": row.get(
                    "emotion_relative_score"
                ),
                "emotion_relative_confidence": row.get(
                    "emotion_relative_confidence"
                ),
                "emotion_relative_gap": row.get(
                    "emotion_relative_gap"
                ),
                "emotion_raw_scores": row.get(
                    "emotion_raw_scores",
                    {},
                ),
                "human_emotion": None,
                "human_confidence": None,
                "human_secondary_emotion": None,
                "human_notes": "",
            }
        )

    report = {
        "samples_total": len(rows),
        "selected": len(selected),
        "limit": limit,
        "accepted_class_counts": dict(sorted(class_counts.items())),
        "rare_classes": sorted(rare_labels),
        "selection_reason_counts": dict(sorted(reason_counts.items())),
        "notes": [
            (
                "The selector never changes emotion labels; it only chooses "
                "information-dense samples for optional human calibration."
            ),
            (
                "Review includes both clean anchors and ambiguous/OOD cases "
                "so future prototype or threshold calibration is not based "
                "only on edge cases."
            ),
        ],
    }
    return selected, report


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
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


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sidecar", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--anchors-per-class", type=int, default=2)
    parser.add_argument("--boundaries-per-class", type=int, default=2)
    parser.add_argument("--rare-class-max", type=int, default=8)
    parser.add_argument("--ood-cases", type=int, default=6)
    parser.add_argument("--neutral-cases", type=int, default=4)
    return parser


def main() -> int:
    args = make_parser().parse_args()
    try:
        if args.anchors_per_class < 0:
            raise ReviewSetError(
                "--anchors-per-class must be >= 0"
            )
        if args.boundaries_per_class < 0:
            raise ReviewSetError(
                "--boundaries-per-class must be >= 0"
            )
        if args.rare_class_max < 0:
            raise ReviewSetError("--rare-class-max must be >= 0")
        if args.ood_cases < 0:
            raise ReviewSetError("--ood-cases must be >= 0")
        if args.neutral_cases < 0:
            raise ReviewSetError("--neutral-cases must be >= 0")

        sidecar_path = Path(args.sidecar).expanduser().resolve()
        output_path = Path(args.output).expanduser().resolve()
        rows = load_jsonl(sidecar_path)
        selected, report = select_review_rows(
            rows,
            limit=args.limit,
            anchors_per_class=args.anchors_per_class,
            boundaries_per_class=args.boundaries_per_class,
            rare_class_max=args.rare_class_max,
            ood_cases=args.ood_cases,
            neutral_cases=args.neutral_cases,
        )
        write_jsonl(output_path, selected)

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
        print(
            json.dumps(
                report,
                ensure_ascii=False,
                indent=2,
            )
        )
    except (ReviewSetError, OSError, ValueError) as exc:
        print(
            f"[FATAL] {exc}",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
