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


if __name__ == "__main__":
    unittest.main()
