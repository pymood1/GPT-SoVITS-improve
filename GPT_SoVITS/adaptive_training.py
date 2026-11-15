"""Small shared helpers for optional adaptive GPT/SoVITS training."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Iterable, Sequence


def deterministic_holdout_names(
    names: Iterable[str],
    fraction: float = 0.10,
    seed: int = 233333,
    min_unique: int = 8,
    max_holdout: int = 8,
) -> set[str]:
    """Return a stable file-level validation split.

    Very small few-shot datasets keep all samples for training rather than
    sacrificing speaker evidence to an unreliable validation estimate.
    """

    unique = sorted({str(name) for name in names})
    if len(unique) < int(min_unique):
        return set()

    fraction = min(max(float(fraction), 0.0), 0.5)
    if fraction <= 0.0:
        return set()

    desired = max(1, int(round(len(unique) * fraction)))
    desired = min(desired, int(max_holdout), len(unique) - 2)

    def score(name: str) -> bytes:
        payload = f"{int(seed)}\0{name}".encode("utf-8")
        return hashlib.sha256(payload).digest()

    ranked = sorted(unique, key=score)
    return set(ranked[:desired])


@dataclass
class AdaptiveEpochState:
    min_epochs: int
    max_epochs: int
    patience: int = 3
    min_relative_improvement: float = 0.005

    best_loss: float = math.inf
    best_epoch: int = 0
    bad_epochs: int = 0

    def update(self, epoch: int, loss: float) -> tuple[bool, bool]:
        """Return (improved, should_stop)."""

        epoch = int(epoch)
        loss = float(loss)
        if not math.isfinite(loss):
            self.bad_epochs += 1
            return False, epoch >= self.min_epochs and self.bad_epochs >= self.patience

        if math.isfinite(self.best_loss):
            threshold = self.best_loss * (
                1.0 - float(self.min_relative_improvement)
            )
            improved = loss < threshold
        else:
            improved = True

        if improved:
            self.best_loss = loss
            self.best_epoch = epoch
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1

        should_stop = (
            epoch >= int(self.min_epochs)
            and self.bad_epochs >= int(self.patience)
        )
        return improved, should_stop
