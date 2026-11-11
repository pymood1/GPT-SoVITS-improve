import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.auto_emotion_sidecar import assign_speaker_relative_labels
from tools.emotion_dataset_builder import parse_list_line, sidecar_for
from tools.emotion_training_pipeline import (
    create_reference_map,
    detect_pair_emotions,
    split_train_validation_samples,
    update_early_stop_state,
)
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



class DatasetParsingTests(unittest.TestCase):
    def test_transcript_may_contain_pipe_character(self):
        wav, speaker, language, text_value = parse_list_line(
            "audio.wav|speaker|ko|앞부분|뒷부분",
            1,
        )
        self.assertEqual(wav, "audio.wav")
        self.assertEqual(speaker, "speaker")
        self.assertEqual(language, "ko")
        self.assertEqual(text_value, "앞부분|뒷부분")

    def test_unique_basename_sidecar_fallback(self):
        sidecar = {
            "audio.wav": {
                "emotion_intended": "happy",
            }
        }
        matched = sidecar_for(
            sidecar,
            "/old/location/audio.wav",
            Path("/new/location/audio.wav"),
            basename_is_unique=True,
        )
        self.assertEqual(matched["emotion_intended"], "happy")

        ambiguous = sidecar_for(
            sidecar,
            "/old/location/audio.wav",
            Path("/new/location/audio.wav"),
            basename_is_unique=False,
        )
        self.assertEqual(ambiguous, {})


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

    def test_reference_preserves_minimum_trainable_class_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "wav").mkdir()
            metadata = self._write_metadata(
                root,
                [
                    {
                        "canonical_wav": "neutral_ref.wav",
                        "speaker_uid": "speaker-a",
                        "split": "train",
                        "tts_usable": True,
                        "emotion_intended": "neutral",
                    },
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
                    {
                        "canonical_wav": "sad_3.wav",
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
                result = create_reference_map(
                    metadata,
                    output,
                    trainable_emotions={"sad"},
                )

            self.assertEqual(
                result["speaker-a"],
                "neutral_ref.wav",
            )

    def test_low_sample_emotion_is_not_auto_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = []

            for emotion in ("happy", "surprised"):
                for index in range(3):
                    rows.append(
                        {
                            "canonical_wav": (
                                f"{emotion}_{index}.wav"
                            ),
                            "speaker_uid": "speaker-a",
                            "split": "train",
                            "tts_usable": True,
                            "emotion_intended": emotion,
                        }
                    )

            rows.append(
                {
                    "canonical_wav": "fearful_1.wav",
                    "speaker_uid": "speaker-a",
                    "split": "train",
                    "tts_usable": True,
                    "emotion_intended": "fearful",
                }
            )
            rows.extend(
                [
                    {
                        "canonical_wav": "angry_1.wav",
                        "speaker_uid": "speaker-a",
                        "split": "train",
                        "tts_usable": True,
                        "emotion_intended": "angry",
                    },
                    {
                        "canonical_wav": "angry_2.wav",
                        "speaker_uid": "speaker-a",
                        "split": "train",
                        "tts_usable": True,
                        "emotion_intended": "angry",
                    },
                ]
            )

            metadata = self._write_metadata(root, rows)

            self.assertEqual(
                detect_pair_emotions(metadata),
                ["happy", "surprised"],
            )



class ValidationPolicyTests(unittest.TestCase):
    def test_stratified_split_keeps_two_train_samples_per_emotion(self):
        samples = []
        for emotion, count in (
            ("happy", 10),
            ("angry", 5),
            ("disgusted", 3),
        ):
            for index in range(count):
                samples.append(
                    {
                        "name": f"{emotion}_{index}.wav",
                        "emotion": emotion,
                    }
                )

        train, val = split_train_validation_samples(
            samples,
            ["happy", "angry", "disgusted"],
            val_fraction=0.20,
            seed=233333,
            min_train_samples=2,
        )

        train_counts = {}
        val_counts = {}
        for sample in train:
            train_counts[sample["emotion"]] = (
                train_counts.get(sample["emotion"], 0) + 1
            )
        for sample in val:
            val_counts[sample["emotion"]] = (
                val_counts.get(sample["emotion"], 0) + 1
            )

        self.assertEqual(train_counts["happy"], 8)
        self.assertEqual(val_counts["happy"], 2)
        self.assertEqual(train_counts["angry"], 4)
        self.assertEqual(val_counts["angry"], 1)
        self.assertEqual(train_counts["disgusted"], 2)
        self.assertEqual(val_counts["disgusted"], 1)

        train2, val2 = split_train_validation_samples(
            samples,
            ["happy", "angry", "disgusted"],
            val_fraction=0.20,
            seed=233333,
            min_train_samples=2,
        )
        self.assertEqual(train, train2)
        self.assertEqual(val, val2)

    def test_early_stop_state_uses_min_delta_and_patience_counter(self):
        best, bad, improved = update_early_stop_state(
            current_loss=1.30,
            best_loss=float("inf"),
            validations_without_improvement=0,
            min_delta=0.001,
        )
        self.assertTrue(improved)
        self.assertEqual(bad, 0)
        self.assertAlmostEqual(best, 1.30)

        best, bad, improved = update_early_stop_state(
            current_loss=1.2995,
            best_loss=best,
            validations_without_improvement=bad,
            min_delta=0.001,
        )
        self.assertFalse(improved)
        self.assertEqual(bad, 1)
        self.assertAlmostEqual(best, 1.30)

        best, bad, improved = update_early_stop_state(
            current_loss=1.295,
            best_loss=best,
            validations_without_improvement=bad,
            min_delta=0.001,
        )
        self.assertTrue(improved)
        self.assertEqual(bad, 0)
        self.assertAlmostEqual(best, 1.295)


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
