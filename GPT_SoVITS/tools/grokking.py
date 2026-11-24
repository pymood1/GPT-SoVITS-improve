"""Utilities for controlled grokking / delayed-generalization experiments.

This module is intentionally opt-in. The normal GPT-SoVITS training path is
unchanged unless GPT_SOVITS_GROKKING=1 (or the stage-specific override) is set.
"""

from __future__ import annotations

import math
import os
import random
from typing import Dict, Mapping, MutableMapping, Sequence, Tuple, List

import torch


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw is not None else float(default)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw is not None else int(default)


def _cfg_get(config, key, default=None):
    if hasattr(config, "get"):
        return config.get(key, default)
    try:
        return config[key]
    except (KeyError, TypeError, AttributeError):
        return getattr(config, key, default)


def resolve_grokking_config(train_config: Mapping, stage: str) -> Dict[str, object]:
    """Resolve grokking settings from config and environment.

    stage must be 's1' or 's2'. Stage-specific environment variables override
    the global switch so S1 and S2 can be probed independently.
    """

    stage = stage.lower().strip()
    if stage not in {"s1", "s2"}:
        raise ValueError(f"unsupported grokking stage: {stage!r}")

    nested = _cfg_get(train_config, "grokking", {})
    if nested is None:
        nested = {}

    global_enabled = _env_bool(
        "GPT_SOVITS_GROKKING",
        bool(_cfg_get(nested, "enabled", False)),
    )
    stage_enabled = _env_bool(
        f"GPT_SOVITS_GROKKING_{stage.upper()}",
        global_enabled,
    )

    cfg = {
        "enabled": stage_enabled,
        "validation_fraction": _env_float(
            "GPT_SOVITS_GROKKING_VALIDATION_FRACTION",
            float(_cfg_get(nested, "validation_fraction", 0.10)),
        ),
        "min_validation_samples": _env_int(
            "GPT_SOVITS_GROKKING_MIN_VALIDATION_SAMPLES",
            int(_cfg_get(nested, "min_validation_samples", 8)),
        ),
        "seed": _env_int(
            "GPT_SOVITS_GROKKING_SEED",
            int(_cfg_get(nested, "seed", _cfg_get(train_config, "seed", 1234))),
        ),
        "grokfast_enabled": _env_bool(
            "GPT_SOVITS_GROKFAST",
            bool(_cfg_get(nested, "grokfast_enabled", True)),
        ),
        "grokfast_alpha": _env_float(
            "GPT_SOVITS_GROKFAST_ALPHA",
            float(_cfg_get(nested, "grokfast_alpha", 0.98)),
        ),
        "grokfast_lambda": _env_float(
            "GPT_SOVITS_GROKFAST_LAMBDA",
            float(_cfg_get(nested, "grokfast_lambda", 0.5)),
        ),
        "memorization_top3": _env_float(
            "GPT_SOVITS_GROKKING_MEMORIZATION_TOP3",
            float(_cfg_get(nested, "memorization_top3", 0.97)),
        ),
        "s1_post_fit_epochs": _env_int(
            "GPT_SOVITS_GROKKING_S1_POST_FIT_EPOCHS",
            int(_cfg_get(nested, "s1_post_fit_epochs", 20)),
        ),
        "s1_optimizer": os.environ.get(
            "GPT_SOVITS_GROKKING_S1_OPTIMIZER",
            str(_cfg_get(nested, "s1_optimizer", "adamw")),
        ).strip().lower(),
        "s1_keep_dpo": _env_bool(
            "GPT_SOVITS_GROKKING_S1_KEEP_DPO",
            bool(_cfg_get(nested, "s1_keep_dpo", False)),
        ),
        "s1_lr": _env_float(
            "GPT_SOVITS_GROKKING_S1_LR",
            float(_cfg_get(nested, "s1_lr", 1.0e-4)),
        ),
        "s1_weight_decay": _env_float(
            "GPT_SOVITS_GROKKING_S1_WEIGHT_DECAY",
            float(_cfg_get(nested, "s1_weight_decay", 1.0e-2)),
        ),
        "s2_post_fit_epochs": _env_int(
            "GPT_SOVITS_GROKKING_S2_POST_FIT_EPOCHS",
            int(_cfg_get(nested, "s2_post_fit_epochs", 8)),
        ),
        "s2_plateau_patience": _env_int(
            "GPT_SOVITS_GROKKING_S2_PLATEAU_PATIENCE",
            int(_cfg_get(nested, "s2_plateau_patience", 3)),
        ),
        "s2_plateau_relative_delta": _env_float(
            "GPT_SOVITS_GROKKING_S2_PLATEAU_RELATIVE_DELTA",
            float(_cfg_get(nested, "s2_plateau_relative_delta", 0.01)),
        ),
        "s2_weight_decay": _env_float(
            "GPT_SOVITS_GROKKING_S2_WEIGHT_DECAY",
            float(_cfg_get(nested, "s2_weight_decay", 1.0e-2)),
        ),
        "max_validation_batches": _env_int(
            "GPT_SOVITS_GROKKING_MAX_VALIDATION_BATCHES",
            int(_cfg_get(nested, "max_validation_batches", 32)),
        ),
    }

    fraction = float(cfg["validation_fraction"])
    if not 0.0 < fraction < 0.5:
        raise ValueError("grokking validation_fraction must be in (0, 0.5)")
    if int(cfg["min_validation_samples"]) < 1:
        raise ValueError("grokking min_validation_samples must be >= 1")
    if not 0.0 <= float(cfg["grokfast_alpha"]) < 1.0:
        raise ValueError("grokfast_alpha must be in [0, 1)")
    if float(cfg["grokfast_lambda"]) < 0.0:
        raise ValueError("grokfast_lambda must be non-negative")
    if int(cfg["s1_post_fit_epochs"]) < 1 or int(cfg["s2_post_fit_epochs"]) < 1:
        raise ValueError("post-fit epoch budgets must be positive")
    if int(cfg["s2_plateau_patience"]) < 1:
        raise ValueError("s2_plateau_patience must be positive")
    if not 0.0 <= float(cfg["s2_plateau_relative_delta"]) < 1.0:
        raise ValueError("s2_plateau_relative_delta must be in [0, 1)")
    if float(cfg["s1_lr"]) <= 0.0:
        raise ValueError("s1 grokking learning rate must be positive")
    if float(cfg["s1_weight_decay"]) < 0.0 or float(cfg["s2_weight_decay"]) < 0.0:
        raise ValueError("grokking weight decay must be non-negative")
    if str(cfg["s1_optimizer"]) not in {"adamw", "scaled_adam"}:
        raise ValueError("s1 grokking optimizer must be 'adamw' or 'scaled_adam'")
    return cfg


def deterministic_holdout_indices(
    item_names: Sequence[str],
    fraction: float,
    min_validation_samples: int,
    seed: int,
) -> Tuple[List[int], List[int]]:
    """Split unique item identities into deterministic train/validation indices."""

    first_index: Dict[str, int] = {}
    for idx, name in enumerate(item_names):
        first_index.setdefault(str(name), idx)
    unique_indices = list(first_index.values())
    if len(unique_indices) < 3:
        raise ValueError("need at least 3 unique samples for a held-out split")

    rng = random.Random(int(seed))
    rng.shuffle(unique_indices)

    requested = max(
        int(min_validation_samples),
        int(round(len(unique_indices) * float(fraction))),
    )
    val_count = min(max(1, requested), len(unique_indices) - 1)

    val_indices = sorted(unique_indices[:val_count])
    train_indices = sorted(unique_indices[val_count:])
    return train_indices, val_indices


def repeat_indices_to_minimum(indices: Sequence[int], minimum: int = 100) -> List[int]:
    values = list(indices)
    if not values:
        raise ValueError("cannot repeat an empty index set")
    if len(values) >= minimum:
        return values
    repeats = int(math.ceil(float(minimum) / len(values)))
    return (values * repeats)[:minimum]


@torch.no_grad()
def parameter_l2_norm(module: torch.nn.Module) -> float:
    total = None
    for parameter in module.parameters():
        if not parameter.requires_grad:
            continue
        value = parameter.detach().float().pow(2).sum()
        total = value if total is None else total + value
    if total is None:
        return 0.0
    return float(total.sqrt().cpu().item())


@torch.no_grad()
def gradfilter_ema(
    module: torch.nn.Module,
    state: MutableMapping[str, torch.Tensor] | None = None,
    *,
    alpha: float = 0.98,
    lamb: float = 0.5,
) -> MutableMapping[str, torch.Tensor]:
    """Amplify the slow EMA component of gradients (Grokfast-style)."""

    if state is None:
        state = {}

    alpha = float(alpha)
    lamb = float(lamb)
    for name, parameter in module.named_parameters():
        grad = parameter.grad
        if not parameter.requires_grad or grad is None:
            continue
        current = grad.detach()
        previous = state.get(name)
        if previous is None or previous.shape != current.shape or previous.device != current.device:
            ema = current.clone()
        else:
            ema = previous.mul(alpha).add(current, alpha=1.0 - alpha)
        state[name] = ema.detach()
        grad.add_(ema, alpha=lamb)
    return state


class GrokkingPhaseTracker:
    """Track conservative S1 delayed-generalization candidates, not proof."""

    def __init__(self, memorization_top3: float = 0.97):
        self.memorization_top3 = float(memorization_top3)
        self.memorization_epoch = None
        self.memorization_val_loss = None
        self.memorization_val_top3 = None
        self.pre_fit_best_loss = None
        self.best_val_loss = None
        self.best_val_epoch = None
        self.best_val_top3 = None

    def update(
        self,
        *,
        epoch: int,
        train_top3: float | None,
        val_loss: float | None,
        val_top3: float | None,
    ) -> Dict[str, object]:
        # Missing/non-finite measurements cannot establish a fit point.
        valid = all(
            value is not None and math.isfinite(float(value))
            for value in (train_top3, val_loss, val_top3)
        )
        valid = valid and val_loss > 0 and 0 <= train_top3 <= 1 and 0 <= val_top3 <= 1
        if valid and self.memorization_epoch is None:
            if train_top3 >= self.memorization_top3:
                self.memorization_epoch = int(epoch)
                self.memorization_val_loss = float(val_loss)
                self.memorization_val_top3 = float(val_top3)
                self.pre_fit_best_loss = self.best_val_loss

        if val_loss is not None and math.isfinite(float(val_loss)) and val_loss > 0:
            if self.best_val_loss is None or float(val_loss) < self.best_val_loss:
                self.best_val_loss = float(val_loss)
                self.best_val_epoch = int(epoch)
                self.best_val_top3 = val_top3

        post_fit_age = (
            None if self.memorization_epoch is None
            else int(epoch) - int(self.memorization_epoch)
        )
        # Do not compare the fit-point loss against a historical best
        # from BEFORE memorization. Require a current, meaningful gain
        # in both metrics while the training fit still holds.
        delayed_improvement = bool(
            valid
            and post_fit_age is not None
            and post_fit_age >= 2
            and train_top3 >= self.memorization_top3
            and val_loss <= 0.95 * self.memorization_val_loss
            and val_top3 >= self.memorization_val_top3 + 0.05
            and (self.pre_fit_best_loss is None or val_loss < self.pre_fit_best_loss)
        )

        return {
            "memorization_epoch": self.memorization_epoch,
            "memorization_val_loss": self.memorization_val_loss,
            "memorization_val_top3": self.memorization_val_top3,
            "pre_fit_best_loss": self.pre_fit_best_loss,
            "post_fit_age": post_fit_age,
            "best_val_loss": self.best_val_loss,
            "best_val_epoch": self.best_val_epoch,
            "best_val_top3": self.best_val_top3,
            "delayed_improvement": delayed_improvement,
        }
