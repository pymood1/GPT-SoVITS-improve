import os
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "GPT_SoVITS"))

from tools.grokking import (
    GrokkingPhaseTracker,
    deterministic_holdout_indices,
    gradfilter_ema,
    repeat_indices_to_minimum,
    resolve_grokking_config,
)


class GrokkingUtilsTests(unittest.TestCase):
    def test_holdout_is_deterministic_and_deduplicated(self):
        names = ["a", "b", "c", "d", "a", "b"]
        train1, val1 = deterministic_holdout_indices(names, 0.25, 1, 1234)
        train2, val2 = deterministic_holdout_indices(names, 0.25, 1, 1234)
        self.assertEqual((train1, val1), (train2, val2))
        selected = train1 + val1
        self.assertEqual(len(selected), 4)
        self.assertEqual(len(set(selected)), 4)
        self.assertTrue(set(train1).isdisjoint(val1))

    def test_repeat_indices_never_leaks_validation(self):
        train = [0, 2, 4]
        repeated = repeat_indices_to_minimum(train, 10)
        self.assertEqual(len(repeated), 10)
        self.assertTrue(set(repeated).issubset(set(train)))

    def test_env_enables_stage(self):
        old = os.environ.get("GPT_SOVITS_GROKKING_S1")
        try:
            os.environ["GPT_SOVITS_GROKKING_S1"] = "1"
            cfg = resolve_grokking_config({"seed": 7}, "s1")
            self.assertTrue(cfg["enabled"])
            self.assertEqual(cfg["seed"], 7)
        finally:
            if old is None:
                os.environ.pop("GPT_SOVITS_GROKKING_S1", None)
            else:
                os.environ["GPT_SOVITS_GROKKING_S1"] = old

    def test_config_resolver_accepts_hparams_like_objects(self):
        class HParamsLike:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

            def __getitem__(self, key):
                return getattr(self, key)

        cfg = resolve_grokking_config(HParamsLike(seed=19), "s2")
        self.assertEqual(cfg["seed"], 19)
        self.assertEqual(cfg["s2_weight_decay"], 0.01)

    def test_grokfast_amplifies_stable_gradient(self):
        model = torch.nn.Linear(2, 1, bias=False)
        model.weight.grad = torch.ones_like(model.weight)
        state = gradfilter_ema(model, None, alpha=0.9, lamb=0.5)
        self.assertTrue(
            torch.allclose(
                model.weight.grad,
                torch.full_like(model.weight, 1.5),
            )
        )
        model.weight.grad = torch.ones_like(model.weight)
        gradfilter_ema(model, state, alpha=0.9, lamb=0.5)
        self.assertTrue(
            torch.allclose(
                model.weight.grad,
                torch.full_like(model.weight, 1.5),
            )
        )

    def test_tracker_marks_post_fit_improvement(self):
        tracker = GrokkingPhaseTracker(memorization_top3=0.9)
        tracker.update(
            epoch=2,
            train_top3=0.91,
            val_loss=2.0,
            val_top3=0.2,
        )
        result = tracker.update(
            epoch=5,
            train_top3=0.99,
            val_loss=1.5,
            val_top3=0.4,
        )
        self.assertEqual(result["memorization_epoch"], 2)
        self.assertEqual(result["post_fit_age"], 3)
        self.assertTrue(result["delayed_improvement"])


    def test_tracker_rejects_historical_best_before_fit(self):
        tracker = GrokkingPhaseTracker(memorization_top3=0.9)
        tracker.update(epoch=1, train_top3=0.5, val_loss=1.0, val_top3=0.5)
        tracker.update(epoch=2, train_top3=0.95, val_loss=2.0, val_top3=0.2)
        result = tracker.update(
            epoch=5, train_top3=0.96, val_loss=2.1, val_top3=0.2
        )
        self.assertFalse(result["delayed_improvement"])
        result = tracker.update(
            epoch=6, train_top3=0.96, val_loss=1.5, val_top3=0.4
        )
        self.assertFalse(result["delayed_improvement"])

    def test_tracker_requires_meaningful_delayed_joint_improvement(self):
        tracker = GrokkingPhaseTracker(memorization_top3=0.9)
        tracker.update(epoch=2, train_top3=0.95, val_loss=2.0, val_top3=0.2)
        for epoch, train, loss, acc in (
            (3, 0.96, 1.5, 0.4),  # too soon
            (4, 0.96, 1.99, 0.4),  # tiny loss change
            (5, 0.96, 1.5, 0.21),  # no accuracy gain
            (6, 0.80, 1.5, 0.4),  # training fit lost
            (7, 0.96, float("nan"), 0.4),
        ):
            with self.subTest(epoch=epoch):
                self.assertFalse(tracker.update(
                    epoch=epoch, train_top3=train, val_loss=loss, val_top3=acc
                )["delayed_improvement"])
        self.assertTrue(tracker.update(
            epoch=8, train_top3=0.96, val_loss=1.5, val_top3=0.4
        )["delayed_improvement"])

    def test_tracker_does_not_establish_fit_on_missing_validation(self):
        tracker = GrokkingPhaseTracker(memorization_top3=0.9)
        result = tracker.update(
            epoch=1, train_top3=0.95, val_loss=None, val_top3=None
        )
        self.assertIsNone(result["memorization_epoch"])
        self.assertFalse(result["delayed_improvement"])


if __name__ == "__main__":
    unittest.main()
