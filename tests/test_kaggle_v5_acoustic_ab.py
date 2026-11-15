import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

from tools.kaggle_v5_acoustic_ab import link_assets, select_reference, TEST_TEXT


class LauncherTests(unittest.TestCase):
    def test_reference_uses_exact_korean_transcript_and_duration_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "datas").mkdir()
            for name in ("a.wav", "b.wav"):
                (root / "datas" / name).write_bytes(b"fixture")
            listing = root / "training.list"
            listing.write_text("a.wav|speaker|KO|짧은 참조입니다.\nb.wav|speaker|KO|정확한 참조 전사입니다.", encoding="utf-8")
            def info(path):
                return SimpleNamespace(duration=2 if Path(path).name == "a.wav" else 6)
            audio, text = select_reference([listing], info)
            self.assertEqual(Path(audio).name, "b.wav")
            self.assertEqual(text, "정확한 참조 전사입니다.")

    def test_reference_without_metadata_does_not_invent_transcript(self):
        self.assertEqual(select_reference([], lambda p: None), ("", ""))
        self.assertNotIn("[", TEST_TEXT)

    @unittest.skipIf(__import__("os").name == "nt", "Kaggle symlink behavior is Linux-only")
    def test_asset_links_preserve_existing_target_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, target = Path(tmp) / "source", Path(tmp) / "target"
            relative = Path("GPT_SoVITS/pretrained_models/a.pth")
            (source / relative).parent.mkdir(parents=True)
            (target / relative).parent.mkdir(parents=True)
            (source / relative).write_bytes(b"source")
            (target / relative).write_bytes(b"user")
            link_assets(source, target)
            self.assertEqual((target / relative).read_bytes(), b"user")


if __name__ == "__main__":
    unittest.main()
