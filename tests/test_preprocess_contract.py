import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from tools.preprocess_contract import (
    build_dataset_fingerprint,
    build_stage_contract,
    contract_matches,
    prepare_training_resume_state,
    quick_path_signature,
    stage_matches_dataset,
    stage_version_key,
    validate_dataset_fingerprint,
    validate_stage_dataset,
    write_stage_contract,
)


class PreprocessContractTests(unittest.TestCase):
    def _dataset(self, root: Path):
        wav_dir = root / "wav"
        wav_dir.mkdir()
        (wav_dir / "a.wav").write_bytes(b"aaa")
        (wav_dir / "b.wav").write_bytes(b"bbb")
        list_path = root / "training.list"
        list_path.write_text(
            "a.wav|spk|ko|hello\n"
            "b.wav|spk|en|world\n",
            encoding="utf-8",
        )
        return list_path, wav_dir

    def test_dataset_fingerprint_changes_when_text_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            list_path, wav_dir = self._dataset(root)
            before = build_dataset_fingerprint(list_path, wav_dir)
            list_path.write_text(
                "a.wav|spk|ko|changed\n"
                "b.wav|spk|en|world\n",
                encoding="utf-8",
            )
            after = build_dataset_fingerprint(list_path, wav_dir)
            self.assertNotEqual(before["digest"], after["digest"])

    def test_dataset_fingerprint_changes_when_audio_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            list_path, wav_dir = self._dataset(root)
            before = build_dataset_fingerprint(list_path, wav_dir)
            time.sleep(0.002)
            (wav_dir / "a.wav").write_bytes(b"changed-audio")
            after = build_dataset_fingerprint(list_path, wav_dir)
            self.assertNotEqual(before["digest"], after["digest"])

    def test_windows_style_list_path_uses_portable_basename(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wav_dir = root / "wav"
            wav_dir.mkdir()
            (wav_dir / "voice.wav").write_bytes(b"audio")
            list_path = root / "training.list"
            list_path.write_text(
                "C:\\dataset\\voice.wav|spk|ko|hello\n",
                encoding="utf-8",
            )
            fingerprint = build_dataset_fingerprint(
                list_path,
                wav_dir,
            )
            self.assertEqual(
                validate_dataset_fingerprint(fingerprint),
                [],
            )
            self.assertEqual(
                fingerprint["rows"][0]["basename"],
                "voice.wav",
            )

    def test_canonical_fingerprint_tracks_relative_audio_under_wav_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wav_dir = root / "wav"
            nested_dir = wav_dir / "nested"
            nested_dir.mkdir(parents=True)
            (wav_dir / "voice.wav").write_bytes(b"basename-copy")
            nested_audio = nested_dir / "voice.wav"
            nested_audio.write_bytes(b"canonical-audio")

            list_path = root / "training.list"
            list_path.write_text(
                "nested/voice.wav|spk|ko|hello\n",
                encoding="utf-8",
            )

            ordinary = build_dataset_fingerprint(
                list_path,
                wav_dir,
            )
            canonical_before = build_dataset_fingerprint(
                list_path,
                wav_dir,
                prefer_list_relative_audio=True,
            )
            self.assertEqual(
                Path(ordinary["rows"][0]["audio"]["path"]),
                (wav_dir / "voice.wav").resolve(),
            )
            self.assertEqual(
                Path(canonical_before["rows"][0]["audio"]["path"]),
                nested_audio.resolve(),
            )

            nested_audio.write_bytes(b"canonical-audio-changed")
            canonical_after = build_dataset_fingerprint(
                list_path,
                wav_dir,
                prefer_list_relative_audio=True,
            )
            self.assertNotEqual(
                canonical_before["digest"],
                canonical_after["digest"],
            )


    def test_1a_validation_does_not_require_audio_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            list_path, wav_dir = self._dataset(root)
            (wav_dir / "a.wav").unlink()
            (wav_dir / "b.wav").unlink()
            fingerprint = build_dataset_fingerprint(
                list_path,
                wav_dir,
            )
            self.assertEqual(
                validate_stage_dataset("1a", fingerprint),
                [],
            )
            self.assertTrue(
                any(
                    "missing audio" in error
                    for error in validate_stage_dataset(
                        "1b",
                        fingerprint,
                    )
                )
            )

    def test_fingerprint_reports_missing_audio(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            list_path, wav_dir = self._dataset(root)
            (wav_dir / "b.wav").unlink()
            fingerprint = build_dataset_fingerprint(
                list_path,
                wav_dir,
            )
            errors = validate_dataset_fingerprint(fingerprint)
            self.assertTrue(
                any("missing audio" in error for error in errors)
            )

    def test_fingerprint_rejects_duplicate_output_basenames(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wav_dir = root / "wav"
            wav_dir.mkdir()
            (wav_dir / "same.wav").write_bytes(b"audio")
            list_path = root / "training.list"
            list_path.write_text(
                "folder_a/same.wav|spk|ko|one\n"
                "folder_b/same.wav|spk|ko|two\n",
                encoding="utf-8",
            )
            fingerprint = build_dataset_fingerprint(
                list_path,
                wav_dir,
            )
            errors = validate_dataset_fingerprint(fingerprint)
            self.assertTrue(
                any("duplicate basename" in error for error in errors)
            )

    def test_text_change_invalidates_1a_but_not_audio_stages(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            list_path, wav_dir = self._dataset(root)
            before_1a = build_stage_contract(
                stage="1a",
                list_path=list_path,
                wav_root=wav_dir,
                version="v5dev",
                settings={"bert": "same"},
            )
            before_1b = build_stage_contract(
                stage="1b",
                list_path=list_path,
                wav_root=wav_dir,
                version="v5dev",
                settings={"cnhubert": "same"},
            )
            before_1c = build_stage_contract(
                stage="1c",
                list_path=list_path,
                wav_root=wav_dir,
                version="v5dev",
                settings={"s2": "same", "upstream_1b": "same"},
            )

            list_path.write_text(
                "a.wav|different-speaker|ko|changed text\n"
                "b.wav|spk|en|world\n",
                encoding="utf-8",
            )

            after_1a = build_stage_contract(
                stage="1a",
                list_path=list_path,
                wav_root=wav_dir,
                version="v5dev",
                settings={"bert": "same"},
            )
            after_1b = build_stage_contract(
                stage="1b",
                list_path=list_path,
                wav_root=wav_dir,
                version="v5dev",
                settings={"cnhubert": "same"},
            )
            after_1c = build_stage_contract(
                stage="1c",
                list_path=list_path,
                wav_root=wav_dir,
                version="v5dev",
                settings={"s2": "same", "upstream_1b": "same"},
            )

            self.assertNotEqual(before_1a["digest"], after_1a["digest"])
            self.assertEqual(before_1b["digest"], after_1b["digest"])
            self.assertEqual(before_1c["digest"], after_1c["digest"])

    def test_audio_change_invalidates_audio_stages_but_not_1a(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            list_path, wav_dir = self._dataset(root)
            before_1a = build_stage_contract(
                stage="1a",
                list_path=list_path,
                wav_root=wav_dir,
                version="v5dev",
                settings={"bert": "same"},
            )
            before_1b = build_stage_contract(
                stage="1b",
                list_path=list_path,
                wav_root=wav_dir,
                version="v5dev",
                settings={"cnhubert": "same"},
            )
            time.sleep(0.002)
            (wav_dir / "a.wav").write_bytes(b"changed audio bytes")
            current = build_dataset_fingerprint(list_path, wav_dir)
            after_1a = build_stage_contract(
                stage="1a",
                list_path=list_path,
                wav_root=wav_dir,
                version="v5dev",
                settings={"bert": "same"},
            )
            after_1b = build_stage_contract(
                stage="1b",
                list_path=list_path,
                wav_root=wav_dir,
                version="v5dev",
                settings={"cnhubert": "same"},
            )

            self.assertEqual(before_1a["digest"], after_1a["digest"])
            self.assertNotEqual(before_1b["digest"], after_1b["digest"])
            self.assertTrue(stage_matches_dataset(after_1b, current))
            self.assertFalse(stage_matches_dataset(before_1b, current))

    def test_speaker_only_change_does_not_invalidate_preprocessing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            list_path, wav_dir = self._dataset(root)
            before = {
                stage: build_stage_contract(
                    stage=stage,
                    list_path=list_path,
                    wav_root=wav_dir,
                    version="v5dev",
                    settings={},
                )
                for stage in ("1a", "1b")
            }
            list_path.write_text(
                "a.wav|new-speaker|ko|hello\n"
                "b.wav|another-speaker|en|world\n",
                encoding="utf-8",
            )
            after = {
                stage: build_stage_contract(
                    stage=stage,
                    list_path=list_path,
                    wav_root=wav_dir,
                    version="v5dev",
                    settings={},
                )
                for stage in ("1a", "1b")
            }
            self.assertEqual(before["1a"]["digest"], after["1a"]["digest"])
            self.assertEqual(before["1b"]["digest"], after["1b"]["digest"])

    def test_quick_model_signature_detects_in_place_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "model"
            model_dir.mkdir()
            weight = model_dir / "weights.bin"
            weight.write_bytes(b"first")
            before = quick_path_signature(model_dir)
            time.sleep(0.002)
            weight.write_bytes(b"second version")
            after = quick_path_signature(model_dir)
            self.assertNotEqual(before, after)

    def test_resume_state_is_preserved_only_for_same_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            marker_path = root / "resume.json"
            resume_dir = root / "ckpt"
            resume_dir.mkdir()
            (resume_dir / "old.ckpt").write_text("old", encoding="utf-8")

            reset = prepare_training_resume_state(
                marker_path,
                [resume_dir],
                {"dataset": "A", "base": "one"},
            )
            self.assertTrue(reset)
            self.assertFalse(resume_dir.exists())

            resume_dir.mkdir()
            (resume_dir / "same.ckpt").write_text("same", encoding="utf-8")
            reset = prepare_training_resume_state(
                marker_path,
                [resume_dir],
                {"dataset": "A", "base": "one"},
            )
            self.assertFalse(reset)
            self.assertTrue((resume_dir / "same.ckpt").is_file())

            reset = prepare_training_resume_state(
                marker_path,
                [resume_dir],
                {"dataset": "B", "base": "one"},
            )
            self.assertTrue(reset)
            self.assertFalse(resume_dir.exists())

    def test_1b_contract_reuses_compatible_standard_versions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            list_path, wav_dir = self._dataset(root)
            v4 = build_stage_contract(
                stage="1b",
                list_path=list_path,
                wav_root=wav_dir,
                version="v4",
                settings={"cnhubert": "same"},
            )
            v5 = build_stage_contract(
                stage="1b",
                list_path=list_path,
                wav_root=wav_dir,
                version="v5dev",
                settings={"cnhubert": "same"},
            )
            self.assertEqual(v4["digest"], v5["digest"])

    def test_1b_version_key_is_idempotent(self):
        self.assertEqual(stage_version_key("1b", "standard"), "standard")
        self.assertEqual(stage_version_key("1b", "pro-sv"), "pro-sv")

    def test_1b_version_key_only_distinguishes_pro_features(self):
        self.assertEqual(
            stage_version_key("1b", "v4"),
            stage_version_key("1b", "v5dev"),
        )
        self.assertNotEqual(
            stage_version_key("1b", "v5dev"),
            stage_version_key("1b", "v2Pro"),
        )
        self.assertEqual(stage_version_key("1a", "v4"), "v4")
        self.assertEqual(stage_version_key("1c", "v5dev"), "v5dev")

    def test_stage_contract_detects_model_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            list_path, wav_dir = self._dataset(root)
            first = build_stage_contract(
                stage="1c",
                list_path=list_path,
                wav_root=wav_dir,
                version="v5dev",
                settings={"pretrained_s2G": "first.pth"},
            )
            second = build_stage_contract(
                stage="1c",
                list_path=list_path,
                wav_root=wav_dir,
                version="v5dev",
                settings={"pretrained_s2G": "second.pth"},
            )
            self.assertNotEqual(first["digest"], second["digest"])

    def test_contract_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            list_path, wav_dir = self._dataset(root)
            expected = build_stage_contract(
                stage="1a",
                list_path=list_path,
                wav_root=wav_dir,
                version="v5dev",
                settings={"multilingual_bert": True},
            )
            marker = root / "1a.json"
            write_stage_contract(marker, expected)
            self.assertTrue(contract_matches(marker, expected))

            changed = build_stage_contract(
                stage="1a",
                list_path=list_path,
                wav_root=wav_dir,
                version="v5dev",
                settings={"multilingual_bert": False},
            )
            self.assertFalse(contract_matches(marker, changed))


class WebUiDatasetContractIntegrationTests(unittest.TestCase):
    def test_training_validates_current_dataset_contracts(self):
        webui = (ROOT / "webui.py").read_text(encoding="utf-8")
        self.assertIn("build_stage_contract", webui)
        self.assertIn("stage_contract_path", webui)
        self.assertIn("stage_matches_dataset", webui)
        self.assertIn("inp_text,\n    inp_wav_dir,", webui)

    def test_hubert_nan_retry_uses_a_snapshot(self):
        script = (
            ROOT
            / "GPT_SoVITS"
            / "prepare_datasets"
            / "2-get-hubert-wav32k.py"
        ).read_text(encoding="utf-8")

        self.assertIn("retry_items = list(nan_fails)", script)
        self.assertIn("nan_fails.clear()", script)
        self.assertIn("for wav in retry_items:", script)
        self.assertNotIn("for wav in nan_fails:", script)
        self.assertIn("raise SystemExit(2)", script)


    def test_s1_switch_detaches_emotion_controller_before_loading(self):
        tts = (
            ROOT
            / "GPT_SoVITS"
            / "TTS_infer_pack"
            / "TTS.py"
        ).read_text(encoding="utf-8")
        tts_start = tts.index("    def init_t2s_weights(")
        tts_end = tts.index("    def init_vocoder(", tts_start)
        tts_switch = tts[tts_start:tts_end]
        self.assertLess(
            tts_switch.index("clear_emotion_controller()"),
            tts_switch.index("torch.load(weights_path"),
        )

        webui = (ROOT / "GPT_SoVITS" / "inference_webui.py").read_text(
            encoding="utf-8"
        )
        webui_start = webui.index("def change_gpt_weights(")
        webui_end = webui.index("\nchange_gpt_weights(gpt_path)", webui_start)
        webui_switch = webui[webui_start:webui_end]
        self.assertLess(
            webui_switch.index("clear_emotion_controller()"),
            webui_switch.index("torch.load(gpt_path"),
        )


    def test_one_click_stop_terminates_delegated_preprocess_stage(self):
        webui = (ROOT / "webui.py").read_text(encoding="utf-8")
        start = webui.index("def close1abcd(")
        end = webui.index("def switch_version(", start)
        block = webui[start:end]
        self.assertIn("close1abc()", block)
        self.assertNotIn("if ps1abc:", block)

    def test_model_paths_are_validated_before_cache_invalidation(self):
        webui = (ROOT / "webui.py").read_text(encoding="utf-8")
        for function_name, path_var in (
            ("open1a", "bert_model_path"),
            ("open1b", "cnhubert_model_path"),
        ):
            start = webui.index(f"def {function_name}(")
            end = webui.index("\ndef ", start + 5)
            block = webui[start:end]
            self.assertIn(path_var, block)
            self.assertLess(
                block.index(f"if {path_var} is None"),
                block.index("_ensure_stage_contract"),
            )

    def test_one_click_preprocessing_reuses_canonical_stages(self):
        webui = (ROOT / "webui.py").read_text(encoding="utf-8")
        start = webui.index("def open1abc(")
        end = webui.index("def close1abc(", start)
        block = webui[start:end]
        self.assertIn("open1a,", block)
        self.assertIn("open1b,", block)
        self.assertIn("open1c,", block)
        self.assertNotIn("1-get-text.py", block)
        self.assertNotIn("2-get-hubert-wav32k.py", block)
        self.assertNotIn("3-get-semantic.py", block)

    def test_preprocess_and_gpt_training_receive_explicit_version(self):
        webui = (ROOT / "webui.py").read_text(encoding="utf-8")
        open1a_start = webui.index("def open1a(")
        open1a_end = webui.index("def close1a(", open1a_start)
        self.assertIn(
            "def open1a(\n    version,",
            webui[open1a_start:open1a_end],
        )

        gpt_start = webui.index("def open1Bb(")
        gpt_end = webui.index("def close1Bb(", gpt_start)
        self.assertIn(
            "def open1Bb(\n    version,",
            webui[gpt_start:gpt_end],
        )

        click_start = webui.index("button1Bb_open.click(")
        click_end = webui.index("version_checkbox.change(", click_start)
        self.assertIn("version_checkbox,", webui[click_start:click_end])

    def test_training_and_preprocessing_are_mutually_exclusive(self):
        webui = (ROOT / "webui.py").read_text(encoding="utf-8")
        self.assertIn("def _training_process_running(", webui)
        self.assertIn("def _preprocessing_process_running(", webui)

        for function_name in ("open1a", "open1b", "open1c"):
            start = webui.index(f"def {function_name}(")
            end = webui.index("\ndef ", start + 5)
            block = webui[start:end]
            self.assertIn("_training_process_running()", block)

        for function_name in ("open1Ba", "open1Bb"):
            start = webui.index(f"def {function_name}(")
            end = webui.index("\ndef ", start + 5)
            block = webui[start:end]
            self.assertIn("_preprocessing_process_running()", block)

    def test_standard_sovits_resume_identity_tracks_text_lr_rate(self):
        webui = (ROOT / "webui.py").read_text(encoding="utf-8")
        start = webui.index("def open1Ba(")
        end = webui.index("def close1Ba(", start)
        block = webui[start:end]
        self.assertIn('"text_low_lr_rate": (', block)
        self.assertIn("if version not in v3v4set", block)


    def test_training_checks_only_real_artifacts_and_provenance(self):
        webui = (ROOT / "webui.py").read_text(encoding="utf-8")
        for function_name in ("open1Ba", "open1Bb"):
            start = webui.index(f"def {function_name}(")
            end = webui.index("\ndef ", start + 5)
            block = webui[start:end]
            self.assertIn("_training_artifact_error", block)
            self.assertIn(
                "_validate_training_dataset_contract",
                block,
            )
            self.assertNotIn("check_for_existance(", block)
            self.assertNotIn("check_details(", block)

    def test_gpt_training_requires_contract_bert_artifacts(self):
        webui = (ROOT / "webui.py").read_text(encoding="utf-8")
        start = webui.index("def _training_artifact_error(")
        end = webui.index("def _emotion_dataset_slug(", start)
        block = webui[start:end]
        self.assertIn('required_bert_languages = {"zh"}', block)
        self.assertIn('required_bert_languages.update({"en", "ko"})', block)
        self.assertIn("language_by_name", block)
        self.assertIn("BERT feature가 없는 GPT 학습 샘플", block)


    def test_gpt_training_rejects_stale_1c_upstream(self):
        webui = (ROOT / "webui.py").read_text(encoding="utf-8")
        start = webui.index("def _validate_training_dataset_contract(")
        end = webui.index("def _training_artifact_error(", start)
        block = webui[start:end]
        self.assertIn("upstream_1b_digest", block)
        self.assertIn('stage == "1c"', block)

    def test_semantic_extraction_requires_matching_hubert_stage(self):
        webui = (ROOT / "webui.py").read_text(encoding="utf-8")
        start = webui.index("def open1c(")
        end = webui.index("def close1c(", start)
        block = webui[start:end]
        self.assertIn('("1b",)', block)
        self.assertIn("1B dependency mismatch", block)

    def test_emotion_preprocess_cache_ignores_unrelated_git_head(self):
        script = (
            ROOT / "tools" / "run_emotion_v5_preprocess.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"preprocess_code_sha256"', script)
        self.assertNotIn('"git_commit": code_commit', script)

    def test_emotion_reuse_checks_source_provenance(self):
        webui = (ROOT / "webui.py").read_text(encoding="utf-8")
        builder = (
            ROOT / "tools" / "emotion_dataset_builder.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"source_provenance"', builder)
        self.assertIn("current_fingerprint = build_dataset_fingerprint", webui)
        self.assertIn("current_sidecar_sha", webui)

    def test_sequential_training_binding_has_one_s2_input(self):
        webui = (ROOT / "webui.py").read_text(encoding="utf-8")
        start = webui.index("button_train_sequence_open.click(")
        end = webui.index("button_train_sequence_close.click(", start)
        block = webui[start:end]
        self.assertEqual(block.count("pretrained_s2G,"), 1)

    def test_dpo_batch_is_not_halved_twice(self):
        webui = (ROOT / "webui.py").read_text(encoding="utf-8")
        data_module = (
            ROOT / "GPT_SoVITS" / "AR" / "data" / "data_module.py"
        ).read_text(encoding="utf-8")
        self.assertIn('self.config["train"].get("if_dpo", False)', data_module)
        self.assertNotIn("[GPT/DPO]", webui)
        self.assertNotIn("original_batch_size = batch_size", webui)


if __name__ == "__main__":
    unittest.main()
