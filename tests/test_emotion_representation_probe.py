import math
import unittest

import torch

from tools.emotion_training_pipeline import run_mean_pool_linear_probe


class RepresentationProbeTests(unittest.TestCase):
    def test_separable_features_beat_permutation_baseline(self):
        torch.manual_seed(20261111)
        class_count = 4
        samples_per_class = 12
        feature_dim = 12

        features = []
        labels = []
        for class_id in range(class_count):
            center = torch.zeros(feature_dim)
            center[class_id] = 6.0
            for _ in range(samples_per_class):
                features.append(
                    center + 0.15 * torch.randn(feature_dim)
                )
                labels.append(class_id)

        report = run_mean_pool_linear_probe(
            torch.stack(features),
            labels,
            ["happy", "angry", "disgusted", "surprised"],
            seed=233333,
            max_folds=3,
            permutations=2,
            steps=40,
        )

        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["folds"], 3)
        self.assertGreater(
            report["macro_accuracy_mean"],
            0.90,
        )
        self.assertLess(
            report["oof_ce"],
            math.log(class_count) - 0.50,
        )
        self.assertGreater(report["oof_accuracy"], 0.90)
        self.assertGreater(report["oof_macro_accuracy"], 0.90)
        self.assertIsNotNone(report["oof_macro_auc"])
        self.assertGreater(report["oof_macro_auc"], 0.95)
        self.assertGreater(
            report["delta_vs_permutation"],
            0.30,
        )
        self.assertEqual(report["verdict"], "informative")

    def test_probe_accepts_leakage_group_metadata(self):
        torch.manual_seed(20261112)
        features = []
        labels = []
        samples = []

        for group_index in range(4):
            for class_id, emotion in enumerate(("happy", "sad")):
                center = torch.zeros(6)
                center[class_id] = 4.0
                features.append(
                    center + 0.05 * torch.randn(6)
                )
                labels.append(class_id)
                samples.append(
                    {
                        "name": f"{emotion}_{group_index}.wav",
                        "text_group_id": f"text-{group_index}",
                    }
                )

        report = run_mean_pool_linear_probe(
            torch.stack(features),
            labels,
            ["happy", "sad"],
            samples=samples,
            seed=233333,
            max_folds=3,
            permutations=1,
            steps=30,
        )

        self.assertEqual(report["status"], "ok")
        self.assertTrue(report["grouped_validation"])
        self.assertEqual(report["folds"], 3)
        self.assertTrue(math.isfinite(report["oof_ce"]))


    def test_probe_handles_observed_four_class_distribution(self):
        torch.manual_seed(20261113)
        counts = [38, 13, 7, 35]
        emotion_names = [
            "happy",
            "angry",
            "disgusted",
            "surprised",
        ]
        features = []
        labels = []
        samples = []

        sample_index = 0
        for class_id, count in enumerate(counts):
            center = torch.zeros(8)
            center[class_id] = 2.0
            for _ in range(count):
                features.append(
                    center + 0.10 * torch.randn(8)
                )
                labels.append(class_id)
                samples.append(
                    {
                        "name": f"sample_{sample_index}.wav",
                        "source_id": f"source_{sample_index}",
                    }
                )
                sample_index += 1

        report = run_mean_pool_linear_probe(
            torch.stack(features),
            labels,
            emotion_names,
            samples=samples,
            seed=233333,
            max_folds=5,
            permutations=1,
            steps=2,
        )

        self.assertEqual(report["status"], "ok")
        self.assertGreaterEqual(report["folds"], 2)
        self.assertTrue(math.isfinite(report["oof_ce"]))
        self.assertTrue(
            math.isfinite(report["oof_macro_accuracy"])
        )

    def test_probe_reports_insufficient_per_class_data(self):
        report = run_mean_pool_linear_probe(
            torch.randn(5, 8),
            [0, 0, 1, 1, 2],
            ["happy", "angry", "surprised"],
            seed=233333,
            max_folds=5,
            permutations=1,
            steps=5,
        )

        self.assertEqual(report["status"], "insufficient_data")
        self.assertEqual(report["verdict"], "unavailable")


if __name__ == "__main__":
    unittest.main()
