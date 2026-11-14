import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.auto_emotion_sidecar import assign_speaker_relative_labels
from tools.emotion_dataset_builder import parse_list_line, sidecar_for
from tools.emotion_training_pipeline import (
    controller_artifact_paths,
    create_reference_map,
    detect_pair_emotions,
    is_managed_finetuned_gpt,
    latest_gpt_checkpoint_for_experiment,
    paired_controller_for_gpt,
    paired_controller_path,
    split_train_validation_samples,
    stratified_group_kfold_indices,
    stratified_kfold_indices,
    update_early_stop_state,
)
from tools.run_emotion_v5_preprocess import (
    PreprocessError,
    effective_half_precision,
    validate_complete_outputs,
)


class EmotionFeatureDependencyTests(unittest.TestCase):
    def test_feature_build_depends_on_s1_text_contract_not_s1_weights(self):
        root = Path(__file__).resolve().parents[1]
        preprocess = (
            root / "tools" / "run_emotion_v5_preprocess.py"
        ).read_text(encoding="utf-8")
        trainer = (
            root / "tools" / "train_emotion_controller.py"
        ).read_text(encoding="utf-8")

        self.assertIn('"text_features": text_features', preprocess)
        self.assertNotIn('"s1_sha256": s1_hash', preprocess)
        self.assertNotIn(
            "feature build was generated for a different S1 checkpoint",
            trainer,
        )
        self.assertIn(
            "feature build text-feature contract does not match S1",
            trainer,
        )


class PreprocessPrecisionTests(unittest.TestCase):
    def test_half_precision_requires_cuda_and_no_full_precision_flag(self):
        self.assertTrue(
            effective_half_precision(
                full_precision=False,
                cuda_available=True,
            )
        )
        self.assertFalse(
            effective_half_precision(
                full_precision=False,
                cuda_available=False,
            )
        )
        self.assertFalse(
            effective_half_precision(
                full_precision=True,
                cuda_available=True,
            )
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


    def test_required_bert_feature_is_rejected_when_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            names = self._make_complete_build(root)

            with self.assertRaises(PreprocessError):
                validate_complete_outputs(
                    root,
                    names,
                    required_bert_names={"a.wav"},
                )



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

    def test_reference_can_use_independent_emotional_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "wav").mkdir()
            rows = [
                {
                    "canonical_wav": f"happy_{index}.wav",
                    "speaker_uid": "speaker-a",
                    "split": "train",
                    "tts_usable": True,
                    "emotion_intended": "happy",
                    "text_group_id": f"happy-text-{index}",
                }
                for index in range(4)
            ]
            metadata = self._write_metadata(root, rows)
            output = root / "reference_map.json"

            with patch(
                "tools.emotion_training_pipeline."
                "_reference_duration_seconds",
                return_value=5.0,
            ):
                result = create_reference_map(
                    metadata,
                    output,
                    trainable_emotions={"happy"},
                    min_samples_per_emotion=3,
                )

            self.assertEqual(
                result["speaker-a"],
                "happy_0.wav",
            )

    def test_reference_avoids_controller_text_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "wav").mkdir()
            rows = [
                {
                    "canonical_wav": "neutral_shared.wav",
                    "speaker_uid": "speaker-a",
                    "split": "train",
                    "tts_usable": True,
                    "emotion_intended": "neutral",
                    "text_group_id": "shared-text",
                },
                {
                    "canonical_wav": "happy_shared.wav",
                    "speaker_uid": "speaker-a",
                    "split": "train",
                    "tts_usable": True,
                    "emotion_intended": "happy",
                    "text_group_id": "shared-text",
                },
                {
                    "canonical_wav": "neutral_unique.wav",
                    "speaker_uid": "speaker-a",
                    "split": "train",
                    "tts_usable": True,
                    "emotion_intended": "neutral",
                    "text_group_id": "unique-text",
                },
                {
                    "canonical_wav": "happy_2.wav",
                    "speaker_uid": "speaker-a",
                    "split": "train",
                    "tts_usable": True,
                    "emotion_intended": "happy",
                    "text_group_id": "happy-2",
                },
                {
                    "canonical_wav": "happy_3.wav",
                    "speaker_uid": "speaker-a",
                    "split": "train",
                    "tts_usable": True,
                    "emotion_intended": "happy",
                    "text_group_id": "happy-3",
                },
            ]
            metadata = self._write_metadata(root, rows)
            output = root / "reference_map.json"

            with patch(
                "tools.emotion_training_pipeline."
                "_reference_duration_seconds",
                return_value=5.0,
            ):
                result = create_reference_map(
                    metadata,
                    output,
                    trainable_emotions={"happy"},
                )

            self.assertEqual(
                result["speaker-a"],
                "neutral_unique.wav",
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



class ControllerArtifactPromotionTests(unittest.TestCase):
    def test_candidate_and_failed_paths_do_not_overwrite_final_controller(self):
        final = Path("/tmp/controller.pt")
        paths = controller_artifact_paths(final)
        self.assertEqual(paths["final"], final)
        self.assertNotEqual(paths["candidate"], final)
        self.assertNotEqual(paths["failed"], final)
        self.assertTrue(str(paths["candidate"]).endswith(".candidate.pt"))
        self.assertTrue(str(paths["failed"]).endswith(".failed.pt"))
        self.assertTrue(
            str(paths["candidate_diagnostics"]).endswith(
                ".candidate.diagnostics.json"
            )
        )


class ControllerPromotionIntegrationTests(unittest.TestCase):
    def test_webui_promotes_only_passed_candidate(self):
        webui = (
            Path(__file__).resolve().parents[1] / "webui.py"
        ).read_text(encoding="utf-8")
        start = webui.index("def run_emotion_training_pipeline(")
        end = webui.index("process_name_subfix", start)
        block = webui[start:end]
        self.assertIn("controller_artifact_paths(", block)
        self.assertIn('diagnosis.get("passed")', block)
        self.assertIn(
            'os.replace(\n                candidate_output,\n'
            '                controller_artifacts["final"],',
            block,
        )
        self.assertIn('controller_artifacts["failed"]', block)
        pass_pos = block.index("latest_emotion_controller_path =")
        fail_pos = block.index('failed_output = controller_artifacts["failed"]')
        self.assertLess(pass_pos, fail_pos)


class GptControllerPairingTests(unittest.TestCase):
    def test_managed_gpt_uses_deterministic_paired_controller_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            weight_root = root / "GPT_weights_v5dev"
            weight_root.mkdir()
            checkpoint = weight_root / "demo-e15.ckpt"
            checkpoint.write_bytes(b"gpt")

            expected = (
                root
                / "emotion_controllers"
                / "paired"
                / "GPT_weights_v5dev"
                / "demo-e15_emotion_controller.pt"
            )
            self.assertTrue(
                is_managed_finetuned_gpt(checkpoint)
            )
            self.assertEqual(
                paired_controller_path(
                    checkpoint,
                    output_root=root,
                ),
                expected,
            )
            self.assertIsNone(
                paired_controller_for_gpt(
                    checkpoint,
                    output_root=root,
                )
            )

            expected.parent.mkdir(parents=True)
            expected.write_bytes(b"controller")
            self.assertEqual(
                paired_controller_for_gpt(
                    checkpoint,
                    output_root=root,
                ),
                expected,
            )

    def test_pretrained_checkpoint_is_not_managed_finetune(self):
        checkpoint = Path(
            "GPT_SoVITS/pretrained_models/s1v3.ckpt"
        )
        self.assertFalse(
            is_managed_finetuned_gpt(checkpoint)
        )

    def test_webui_pairs_controller_with_gpt_training_and_inference(self):
        root = Path(__file__).resolve().parents[1]
        webui = (root / "webui.py").read_text(encoding="utf-8")
        train_start = webui.index("def open1Bb(")
        train_end = webui.index("def close1Bb(", train_start)
        train_block = webui[train_start:train_end]

        self.assertIn(
            "controller_output = paired_controller_path(gpt_checkpoint)",
            train_block,
        )
        self.assertIn(
            "run_emotion_training_pipeline(",
            train_block,
        )
        self.assertIn(
            "auto_create_sidecar=True",
            train_block,
        )

        infer_start = webui.index("def change_tts_inference(")
        infer_end = webui.index("def close_tts_inference(", infer_start)
        infer_block = webui[infer_start:infer_end]
        self.assertIn(
            "paired_controller_for_gpt(",
            infer_block,
        )
        self.assertNotIn(
            "latest_emotion_controller_path",
            infer_block,
        )

    def test_inference_webui_requires_pair_for_managed_gpt_switch(self):
        root = Path(__file__).resolve().parents[1]
        source = (
            root / "GPT_SoVITS" / "inference_webui.py"
        ).read_text(encoding="utf-8")
        start = source.index("def change_gpt_weights_webui(")
        end = source.index(
            'os.environ["HF_ENDPOINT"]',
            start,
        )
        block = source[start:end]

        self.assertIn(
            "_paired_controller_for_gpt_path(",
            block,
        )
        self.assertIn(
            "is_managed_finetuned_gpt(",
            block,
        )
        self.assertIn(
            "대응 Emotion Controller가 필수",
            block,
        )

class GptCheckpointSelectionTests(unittest.TestCase):
    def test_latest_checkpoint_is_selected_for_exact_experiment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old_high = root / "demo-e15.ckpt"
            old_high.write_bytes(b"old")
            newer_low = root / "demo-e5.ckpt"
            newer_low.write_bytes(b"new")
            (root / "demo-other-e99.ckpt").write_bytes(b"x")

            old_ns = old_high.stat().st_mtime_ns
            newer_ns = max(newer_low.stat().st_mtime_ns, old_ns + 1_000_000)
            import os
            os.utime(old_high, ns=(old_ns, old_ns))
            os.utime(newer_low, ns=(newer_ns, newer_ns))

            selected = latest_gpt_checkpoint_for_experiment(
                root,
                "demo",
            )

            self.assertEqual(selected, newer_low)

    def test_missing_experiment_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "other-e10.ckpt").write_bytes(b"x")
            self.assertIsNone(
                latest_gpt_checkpoint_for_experiment(root, "demo")
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

    def test_holdout_keeps_same_text_group_on_one_side(self):
        samples = []
        for group_index in range(3):
            for emotion in ("happy", "sad"):
                samples.append(
                    {
                        "name": f"{emotion}_{group_index}.wav",
                        "emotion": emotion,
                        "text_group_id": f"text-{group_index}",
                    }
                )

        train, val = split_train_validation_samples(
            samples,
            ["happy", "sad"],
            val_fraction=0.34,
            seed=233333,
            min_train_samples=2,
        )

        train_groups = {
            sample["text_group_id"] for sample in train
        }
        val_groups = {
            sample["text_group_id"] for sample in val
        }
        self.assertFalse(train_groups & val_groups)
        self.assertEqual(
            {sample["emotion"] for sample in val},
            {"happy", "sad"},
        )
        self.assertEqual(len(train), 4)
        self.assertEqual(len(val), 2)

    def test_grouped_split_recovers_from_greedy_dead_end(self):
        # The highest-gain group alone makes the next greedy choice
        # impossible, but groups 1 and 3 form a feasible held-out split.
        group_counts = [
            (1, 3, 0),
            (1, 1, 0),
            (1, 1, 1),
            (0, 3, 1),
            (0, 0, 1),
        ]
        emotions = ("happy", "sad", "angry")
        samples = []
        for group_index, counts in enumerate(group_counts):
            for emotion, count in zip(emotions, counts):
                for sample_index in range(count):
                    samples.append(
                        {
                            "name": (
                                f"group{group_index}_{emotion}_{sample_index}.wav"
                            ),
                            "emotion": emotion,
                            "text_group_id": f"group-{group_index}",
                        }
                    )

        train, val = split_train_validation_samples(
            samples,
            emotions,
            val_fraction=0.2,
            seed=233333,
            min_train_samples=2,
        )
        self.assertFalse(
            {item["text_group_id"] for item in train}
            & {item["text_group_id"] for item in val}
        )
        for emotion in emotions:
            self.assertGreaterEqual(
                sum(item["emotion"] == emotion for item in train),
                2,
            )
            self.assertGreaterEqual(
                sum(item["emotion"] == emotion for item in val),
                1,
            )
        self.assertEqual(len(train) + len(val), len(samples))

    def test_controller_group_kfold_does_not_leak_text_groups(self):
        samples = []
        labels = []
        for group_index in range(4):
            for class_id, emotion in enumerate(("happy", "sad")):
                samples.append(
                    {
                        "name": f"{emotion}_{group_index}.wav",
                        "emotion": emotion,
                        "text_group_id": f"text-{group_index}",
                    }
                )
                labels.append(class_id)

        folds = stratified_group_kfold_indices(
            samples,
            labels,
            max_folds=3,
            seed=233333,
        )
        self.assertEqual(len(folds), 3)

        validation_union = []
        for train_idx, val_idx in folds:
            validation_union.extend(val_idx)
            train_groups = {
                samples[index]["text_group_id"]
                for index in train_idx
            }
            val_groups = {
                samples[index]["text_group_id"]
                for index in val_idx
            }
            self.assertFalse(train_groups & val_groups)
            self.assertEqual(
                {labels[index] for index in val_idx},
                {0, 1},
            )
            self.assertEqual(
                {labels[index] for index in train_idx},
                {0, 1},
            )

        self.assertEqual(
            sorted(validation_union),
            list(range(len(samples))),
        )
        self.assertEqual(
            folds,
            stratified_group_kfold_indices(
                samples,
                labels,
                max_folds=3,
                seed=233333,
            ),
        )

    def test_group_kfold_falls_back_to_feasible_two_folds(self):
        # There are three independent groups per class, but no way to
        # allocate every class into all three folds. Two folds are viable.
        group_emotions = [
            ("happy", "sad", "angry"),
            ("happy", "sad"),
            ("happy", "angry"),
            ("sad", "angry"),
        ]
        emotions = ("happy", "sad", "angry")
        samples = []
        labels = []
        for group_index, present in enumerate(group_emotions):
            for emotion in present:
                samples.append(
                    {
                        "name": f"group{group_index}_{emotion}.wav",
                        "text_group_id": f"group-{group_index}",
                    }
                )
                labels.append(emotions.index(emotion))

        folds = stratified_group_kfold_indices(
            samples,
            labels,
            max_folds=5,
            seed=233333,
        )
        self.assertEqual(len(folds), 2)
        all_val = []
        for train_idx, val_idx in folds:
            all_val.extend(val_idx)
            self.assertEqual(
                {labels[i] for i in train_idx},
                {0, 1, 2},
            )
            self.assertEqual(
                {labels[i] for i in val_idx},
                {0, 1, 2},
            )
            self.assertFalse(
                {samples[i]["text_group_id"] for i in train_idx}
                & {samples[i]["text_group_id"] for i in val_idx}
            )
        self.assertEqual(sorted(all_val), list(range(len(samples))))

    def test_probe_kfold_is_stratified_and_deterministic(self):
        labels = (
            [0] * 8
            + [1] * 5
            + [2] * 3
        )
        folds = stratified_kfold_indices(
            labels,
            max_folds=5,
            seed=233333,
        )
        self.assertEqual(len(folds), 3)

        validation_union = []
        for train_idx, val_idx in folds:
            self.assertTrue(train_idx)
            self.assertTrue(val_idx)
            validation_union.extend(val_idx)
            val_labels = {labels[index] for index in val_idx}
            self.assertEqual(val_labels, {0, 1, 2})

        self.assertEqual(
            sorted(validation_union),
            list(range(len(labels))),
        )
        self.assertEqual(
            folds,
            stratified_kfold_indices(
                labels,
                max_folds=5,
                seed=233333,
            ),
        )

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
