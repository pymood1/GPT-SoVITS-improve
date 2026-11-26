import math
import unittest
from tools.quality_reference_style import style_weights, reference_groups
from tools.kaggle_quality_listening import STYLE_LABELS, merge_mix_reports


class ReferenceStyleTests(unittest.TestCase):
    def test_weights_are_normalized_and_prefer_similar_speaker(self):
        values = style_weights([.9, .7, .5], "sv")
        self.assertAlmostEqual(sum(values), 1)
        self.assertGreater(values[0], values[1])
        self.assertGreater(values[1], values[2])
        self.assertEqual(style_weights([.9, .7], "uniform"), [.5, .5])
        for bad in ([], [float("nan")], [2]):
            with self.assertRaises(ValueError):
                style_weights(bad, "sv")
        with self.assertRaises(ValueError):
            style_weights([.9], "sv", temperature=0)

    def test_reference_groups_are_disjoint(self):
        refs = [(str(i), "text", "auto") for i in range(12)]
        def choose(folder, count, exclude=()):
            return [r for r in refs if r[0] not in exclude][:count]
        prompt, aux, score = reference_groups(choose, None)
        self.assertEqual([r[0] for r in prompt], ["3", "4", "5"])
        self.assertEqual([r[0] for r in aux], ["6", "7", "8"])
        self.assertEqual([r[0] for r in score], ["9", "10", "11"])

    def test_replication_excludes_previous_references(self):
        refs = [(str(i), "text", "auto") for i in range(21)]
        def choose(folder, count, exclude=()):
            return [r for r in refs if r[0] not in exclude][:count]
        original = reference_groups(choose, None)
        repeated = reference_groups(choose, None, replication=True)
        self.assertEqual([r[0] for group in repeated for r in group],
                         [str(i) for i in range(12, 21)])
        self.assertFalse({r[0] for group in original for r in group}
                         & {r[0] for group in repeated for r in group})

    def test_style_report_requires_disjoint_reference_scores(self):
        reports = [{"status": "completed", "reference_counts": {},
                    "evaluation": {"asr_status": "completed", "speaker_status": "completed"},
                    "models": {label: {"status": "completed", "samples": [
                        {"index": i, "cer": 0., "speaker_cosine": .8, "speaker_holdout_cosine": .79}
                        for i in range(1, 19)]} for label in labels}}
                   for labels in STYLE_LABELS]
        self.assertEqual(merge_mix_reports(reports, STYLE_LABELS, 18)["status"], "completed")
        del reports[1]["models"]["style_lora_sv"]["samples"][0]["speaker_holdout_cosine"]
        self.assertEqual(merge_mix_reports(reports, STYLE_LABELS, 18)["status"], "failed")


if __name__ == "__main__":
    unittest.main()
