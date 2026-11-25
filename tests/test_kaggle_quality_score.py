import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from tools.kaggle_quality_score import calibration_indices, equivalent
from tools.kaggle_quality_listening import apply_score_stage, merge_mix_reports, GENERAL_LABELS, run_score_stage


class QualityScoreTests(unittest.TestCase):
    def test_calibration_per_candidate_and_tail(self):
        self.assertEqual(calibration_indices([{"label": x} for x in ["a", "a", "b", "b"]]), [0, 2, 3])
        self.assertEqual(calibration_indices([]), [])

    def test_strict_asr_and_tolerant_fp32_speaker_gate(self):
        self.assertTrue(equivalent("asr", {"char_errors": 0}, {"char_errors": 0}))
        self.assertFalse(equivalent("asr", {"char_errors": 0}, {"char_errors": 1}))
        self.assertTrue(equivalent("speaker", {"speaker_cosine": .8}, {"speaker_cosine": .80001}))
        self.assertFalse(equivalent("speaker", {"speaker_cosine": .8}, {"speaker_cosine": .81}))
        self.assertFalse(equivalent("speaker", {"speaker_cosine": float("nan")}, {"speaker_cosine": .8}))

    def test_gpu_failure_retries_cpu_and_ignores_stale_results(self):
        import json
        from types import SimpleNamespace
        with TemporaryDirectory() as directory:
            output = Path(directory)
            request = {"samples": [{"label": "a", "index": 1}]}
            (output / "asr-scores.json").write_text('{"scores": [{}]}')
            calls = []
            def mock_run(args, **kwargs):
                device = args[-2]
                calls.append(device)
                self.assertFalse((output / "asr-scores.json").exists())
                if device == "cuda":
                    return SimpleNamespace(returncode=-6)
                (output / "asr-scores.json").write_text(json.dumps({
                    "scores": [{"char_errors": 0, "char_count": 10, "cer": 0}],
                    "backend": "cpu_int8", "seconds": 1,
                    "calibration_count": 0, "calibration_seconds": 0}))
                return SimpleNamespace(returncode=0)
            with patch("subprocess.run", side_effect=mock_run):
                result = run_score_stage(output, "asr", request)
            self.assertEqual(calls, ["cuda", "cpu"])
            models = {"a": {"samples": [{"index": 1}]}}
            evaluation = {}
            apply_score_stage(models, evaluation, "asr", request, result)
            self.assertEqual(evaluation["asr_backend"], "cpu_int8")
            self.assertEqual(models["a"]["samples"][0]["cer"], 0)

    def test_mixed_backends_rejected(self):
        reports = [{"status": "completed", "reference_counts": {},
            "evaluation": {"asr_status": "completed", "speaker_status": "completed",
                           "asr_backend": backend, "speaker_backend": "cpu_fp32"},
            "models": {label: {"status": "completed", "samples": [
                {"index": i, "cer": 0., "speaker_cosine": .8} for i in range(1, 19)]}
                for label in labels}} for backend, labels in
                zip(("cpu_int8", "cuda_int8_float32"), GENERAL_LABELS)]
        self.assertEqual(merge_mix_reports(reports, GENERAL_LABELS, 18)["status"], "failed")
        reports[1]["evaluation"]["asr_backend"] = "cpu_int8"
        self.assertEqual(merge_mix_reports(reports, GENERAL_LABELS, 18)["status"], "completed")


if __name__ == "__main__":
    unittest.main()
