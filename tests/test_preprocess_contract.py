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
