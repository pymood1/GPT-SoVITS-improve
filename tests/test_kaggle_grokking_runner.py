import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from tools.kaggle_grokking_runner import _arm_env, _bundle, _read_metrics, _summarize


class KaggleRunnerTests(unittest.TestCase):
    def test_arm_environment_isolated(self):
        with patch.dict(os.environ, {"GPT_SOVITS_GROKFAST": "1", "GPT_SOVITS_GROKKING_S1_LR": "9"}):
            a = _arm_env({"grokfast": False}, 0, Path("a.json"))
            b = _arm_env({"grokfast": True}, 1, Path("b.json"))
        self.assertEqual(a["CUDA_VISIBLE_DEVICES"], "0")
        self.assertEqual(b["CUDA_VISIBLE_DEVICES"], "1")
        self.assertEqual(a["GPT_SOVITS_GROKFAST"], "0")
        self.assertEqual(b["GPT_SOVITS_GROKFAST"], "1")
        self.assertEqual(a["GPT_SOVITS_GROKKING_S1_LR"], b["GPT_SOVITS_GROKKING_S1_LR"])
        self.assertNotEqual(a["GPT_SOVITS_RUNTIME_STATUS_PATH"], b["GPT_SOVITS_RUNTIME_STATUS_PATH"])

    def test_weight_decay_profile_changes_only_decay_and_gpu(self):
        from tools.kaggle_grokking_runner import SETTINGS
        a, b = SETTINGS["arms"]
        ea = _arm_env(a, 0, Path("a.json"))
        eb = _arm_env(b, 1, Path("b.json"))
        self.assertEqual(ea["GPT_SOVITS_GROKKING_S1_WEIGHT_DECAY"], "0.01")
        self.assertEqual(eb["GPT_SOVITS_GROKKING_S1_WEIGHT_DECAY"], "1.0")
        self.assertEqual(ea["GPT_SOVITS_GROKFAST"], "0")
        self.assertEqual(eb["GPT_SOVITS_GROKFAST"], "0")
        self.assertEqual(ea["GPT_SOVITS_GROKKING_S1_FIXED_EPOCHS"], "120")
        self.assertEqual(eb["GPT_SOVITS_GROKKING_S1_FIXED_EPOCHS"], "120")

    def test_partial_metrics_and_best(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            path = p / "grokking_s1_metrics_s1234.jsonl"
            path.write_text(
                json.dumps({"epoch": 1, "val_loss": 3.0}) + "\n"
                + json.dumps({"epoch": 2, "val_loss": 2.0, "delayed_improvement": True}) + "\n"
                + '{"epoch":3,"val_loss":NaN}\n'
                + '{"epoch":4',
                encoding="utf-8",
            )
            s = _summarize(_read_metrics(p))
            self.assertEqual(s["epochs_recorded"], 2)
            self.assertEqual(s["best"]["epoch"], 2)
            self.assertEqual(s["candidate_epochs"], [2])

    def test_bundle_excludes_audio_and_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = root / "report"
            report.mkdir()
            for name in ("summary.json", "train.log", "voice.wav", "model.ckpt"):
                (report / name).write_text("test")
            target = root / "result.zip"
            _bundle(report, target)
            with zipfile.ZipFile(target) as archive:
                self.assertEqual(set(archive.namelist()), {"summary.json", "train.log"})

    def test_bundle_replaces_snapshot_and_truncates_only_large_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = root / "report"
            report.mkdir()
            target = root / "export" / "result.zip"
            (report / "train.log").write_bytes(b"AAAABBBBCCCCDDDD")
            (report / "metrics.jsonl").write_bytes(b"01234567890123456789")
            with patch("tools.kaggle_grokking_runner.MAX_ARCHIVE_LOG_BYTES", 8):
                _bundle(report, target)
            with zipfile.ZipFile(target) as archive:
                log = archive.read("train.log")
                self.assertTrue(log.startswith(b"AAAA"))
                self.assertTrue(log.endswith(b"DDDD"))
                self.assertIn(b"archive truncated", log)
                self.assertEqual(archive.read("metrics.jsonl"), b"01234567890123456789")
            self.assertEqual((report / "train.log").read_bytes(), b"AAAABBBBCCCCDDDD")
            (report / "train.log").write_text("final")
            _bundle(report, target)
            with zipfile.ZipFile(target) as archive:
                self.assertEqual(archive.read("train.log"), b"final")
            self.assertEqual(list(target.parent.iterdir()), [target])

    def test_failed_snapshot_keeps_previous_zip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = root / "report"
            report.mkdir()
            (report / "summary.json").write_text("{}")
            target = root / "result.zip"
            _bundle(report, target)
            previous = target.read_bytes()
            with patch("tools.kaggle_grokking_runner.zipfile.ZipFile.write", side_effect=OSError("disk")):
                with self.assertRaises(OSError):
                    _bundle(report, target)
            self.assertEqual(target.read_bytes(), previous)
            self.assertFalse(target.with_name("." + target.name + ".part").exists())



if __name__ == "__main__":
    unittest.main()
