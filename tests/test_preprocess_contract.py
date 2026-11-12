import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from tools.preprocess_contract import (
    build_dataset_fingerprint,
    build_stage_contract,
    contract_matches,
    validate_dataset_fingerprint,
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
        self.assertIn("contracts_share_dataset", webui)
        self.assertIn("inp_text,\n    inp_wav_dir,", webui)

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

    def test_semantic_extraction_requires_matching_hubert_stage(self):
        webui = (ROOT / "webui.py").read_text(encoding="utf-8")
        start = webui.index("def open1c(")
        end = webui.index("def close1c(", start)
        block = webui[start:end]
        self.assertIn('("1b",)', block)
        self.assertIn("1B dependency mismatch", block)

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
