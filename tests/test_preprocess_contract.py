import tempfile
import time
import unittest
from pathlib import Path

from tools.preprocess_contract import (
    build_dataset_fingerprint,
    build_stage_contract,
    contract_matches,
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


if __name__ == "__main__":
    unittest.main()
