import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.auto_emotion_sidecar import assign_speaker_relative_labels
from tools.emotion_training_pipeline import create_reference_map
from tools.run_emotion_v5_preprocess import (
    PreprocessError,
    validate_complete_outputs,
)


class PreprocessCompletenessTests(unittest.TestCase):
    def _make_complete_build(self, root: Path):
        names = ["a.wav", "b.wav"]
        (root / "4-cnhubert").mkdir(parents=True)
        (root / "5-wav32k").mkdir(parents=True)
        (root / "2-name2text.txt").write_text(
            "a.wav\tp1 p2\t1 1\talpha\n"
            "b.wav\tp3 p4\t1 1\tbeta\n",
            encoding="utf-8",
        )
        (root / "6-name2semantic.tsv").write_text(
            "item_name\tsemantic_audio\n"
            "a.wav\t1 2 3\n"
            "b.wav\t4 5 6\n",
            encoding="utf-8",
        )
        for name in names:
            (root / "4-cnhubert" / f"{name}.pt").write_bytes(b"x")
            (root / "5-wav32k" / name).write_bytes(b"x")
        return names

    def test_complete_build_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            names = self._make_complete_build(root)
            validate_complete_outputs(root, names)

    def test_missing_per_item_feature_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            names = self._make_complete_build(root)
            (root / "4-cnhubert" / "b.wav.pt").unlink()
            with self.assertRaises(PreprocessError):
                validate_complete_outputs(root, names)


class ReferenceSelectionTests(unittest.TestCase):
    def _write_metadata(self, root: Path, rows):
        path = root / "metadata.jsonl"
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
        return path

    def test_unlabeled_speaker_does_not_require_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "wav").mkdir()
            metadata = self._write_metadata(
                root,
                [
                    {
                        "canonical_wav": "a_neutral.wav",
                        "speaker_uid": "speaker-a",
                        "split": "train",
                        "tts_usable": True,
                        "emotion_intended": "neutral",
                    },
                    {
                        "canonical_wav": "a_happy.wav",
                        "speaker_uid": "speaker-a",
                        "split": "train",
                        "tts_usable": True,
                        "emotion_intended": "happy",
                    },
                    {
                        "canonical_wav": "a_sad.wav",
                        "speaker_uid": "speaker-a",
                        "split": "train",
                        "tts_usable": True,
                        "emotion_intended": "sad",
                    },
                    {
                        "canonical_wav": "b.wav",
                        "speaker_uid": "speaker-b",
                        "split": "train",
                        "tts_usable": True,
                        "emotion_intended": None,
                    },
                ],
            )
            output = root / "reference_map.json"
            with patch(
                "tools.emotion_training_pipeline."
                "_reference_duration_seconds",
                return_value=5.0,
            ):
                result = create_reference_map(metadata, output)

            self.assertEqual(result, {"speaker-a": "a_neutral.wav"})

    def test_singleton_emotion_is_not_consumed_as_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "wav").mkdir()
            metadata = self._write_metadata(
                root,
                [
                    {
                        "canonical_wav": "happy_only.wav",
                        "speaker_uid": "speaker-a",
                        "split": "train",
                        "tts_usable": True,
                        "emotion_intended": "happy",
                    },
                    {
                        "canonical_wav": "sad_1.wav",
                        "speaker_uid": "speaker-a",
                        "split": "train",
                        "tts_usable": True,
                        "emotion_intended": "sad",
                    },
                    {
                        "canonical_wav": "sad_2.wav",
                        "speaker_uid": "speaker-a",
                        "split": "train",
                        "tts_usable": True,
                        "emotion_intended": "sad",
                    },
                ],
            )
            output = root / "reference_map.json"
            with patch(
                "tools.emotion_training_pipeline."
                "_reference_duration_seconds",
                return_value=5.0,
            ):
                result = create_reference_map(metadata, output)

            self.assertIn(result["speaker-a"], {"sad_1.wav", "sad_2.wav"})
            self.assertNotEqual(result["speaker-a"], "happy_only.wav")


class AutoEmotionCalibrationTests(unittest.TestCase):
    def test_prediction_failure_is_not_used_as_neutral_or_baseline(self):
        rows = [
            {
                "speaker": "speaker-a",
                "emotion_scores": {
                    "neutral": 0.2,
                    "happy": 0.7,
                    "sad": 0.1,
                },
            },
            {
                "speaker": "speaker-a",
                "emotion_scores": {},
                "emotion_prediction_error": "decode failed",
            },
        ]

        baselines = assign_speaker_relative_labels(
            rows,
            {"neutral", "happy", "sad"},
            min_confidence=0.0,
            min_relative_score=0.0,
        )

        self.assertTrue(baselines["speaker-a"])
        self.assertIsNone(rows[1]["accepted_emotion"])
        self.assertEqual(rows[1]["emotion_relative_scores"], {})
        self.assertFalse(rows[1]["emotion_is_default"])


if __name__ == "__main__":
    unittest.main()
