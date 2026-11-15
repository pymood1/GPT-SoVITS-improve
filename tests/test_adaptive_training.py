import ast
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GPT_ROOT = ROOT / "GPT_SoVITS"
if str(GPT_ROOT) not in sys.path:
    sys.path.insert(0, str(GPT_ROOT))

from adaptive_training import (
    AdaptiveEpochState,
    deterministic_holdout_names,
)


class AdaptiveTrainingTest(unittest.TestCase):
    def test_tiny_few_shot_dataset_keeps_all_samples_for_training(self):
        names = [f"item-{index}" for index in range(11)]
        self.assertEqual(
            deterministic_holdout_names(names),
            set(),
        )

    def test_validation_split_is_stable_and_has_at_least_two_items(self):
        names = [f"item-{index}" for index in range(20)]
        first = deterministic_holdout_names(names, seed=1234)
        second = deterministic_holdout_names(names, seed=1234)
        self.assertEqual(first, second)
        self.assertGreaterEqual(len(first), 2)
        self.assertLess(len(first), len(names))

    def test_direct_training_calls_match_webui_function_signatures(self):
        webui_path = ROOT / "webui.py"
        tree = ast.parse(webui_path.read_text(encoding="utf-8"))

        definitions = {}
        for node in tree.body:
            if (
                isinstance(node, ast.FunctionDef)
                and node.name in {"open1Ba", "open1Bb"}
            ):
                definitions[node.name] = len(node.args.args)

        self.assertEqual(set(definitions), {"open1Ba", "open1Bb"})

        checked = 0
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Name):
                continue
            if node.func.id not in definitions:
                continue
            if any(isinstance(arg, ast.Starred) for arg in node.args):
                continue

            self.assertEqual(
                len(node.args),
                definitions[node.func.id],
                (
                    node.func.id,
                    node.lineno,
                    len(node.args),
                    definitions[node.func.id],
                ),
            )
            checked += 1

        self.assertGreaterEqual(checked, 1)

    def test_adaptive_epoch_state_extends_while_improving(self):
        state = AdaptiveEpochState(
            min_epochs=3,
            max_epochs=8,
            patience=2,
            min_relative_improvement=0.01,
        )

        improved, stop = state.update(1, 1.0)
        self.assertTrue(improved)
        self.assertFalse(stop)

        improved, stop = state.update(2, 0.95)
        self.assertTrue(improved)
        self.assertFalse(stop)

        improved, stop = state.update(3, 0.90)
        self.assertTrue(improved)
        self.assertFalse(stop)

        improved, stop = state.update(4, 0.895)
        self.assertFalse(improved)
        self.assertFalse(stop)

        improved, stop = state.update(5, 0.894)
        self.assertFalse(improved)
        self.assertTrue(stop)
        self.assertEqual(state.best_epoch, 3)


if __name__ == "__main__":
    unittest.main()
