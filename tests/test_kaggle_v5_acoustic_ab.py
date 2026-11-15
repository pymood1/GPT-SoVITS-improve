import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

from tools.kaggle_v5_acoustic_ab import (
    TEST_TEXT,
    _has_pretrained,
    _runtime_environment,
    link_assets,
    select_reference,
)


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
            audio, text, used = select_reference([listing], info)
            self.assertEqual(Path(audio).name, "b.wav")
            self.assertEqual(text, "정확한 참조 전사입니다.")
            self.assertEqual(Path(used), listing.resolve())

    def test_reference_without_metadata_does_not_invent_transcript(self):
        self.assertEqual(select_reference([], lambda p: None), ("", "", ""))
        self.assertNotIn("[", TEST_TEXT)

    def test_pretrained_detection_requires_complete_base_asset_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pretrained = root / "GPT_SoVITS/pretrained_models"
            required = [
                pretrained / "s1v3.ckpt",
                pretrained / "gsv-v5-pretrained/s2Gv5dev.pth",
                pretrained / "gsv-v5-pretrained/vocoder.pth",
                pretrained / "chinese-hubert-base/config.json",
                pretrained / "chinese-roberta-wwm-ext-large/config.json",
            ]
            for path in required:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"fixture")

            self.assertTrue(_has_pretrained(root))
            (pretrained / "gsv-v5-pretrained/vocoder.pth").unlink()
            self.assertFalse(_has_pretrained(root))

    def test_runtime_environment_uses_existing_korean_reference_only(self):
        repo = Path("/kaggle/working/GPT-SoVITS-improve-diagnostic")
        output = Path("/kaggle/working/v5_acoustic_ab")
        env = _runtime_environment(
            repo,
            output,
            "/kaggle/tmp/audio_input/datas/ref.wav",
            "기존 데이터 전사입니다.",
        )
        self.assertEqual(env["GPT_SOVITS_AB_REFERENCE_LANGUAGE"], "all_ko")
        self.assertEqual(env["GPT_SOVITS_AB_REFERENCE_TEXT"], "기존 데이터 전사입니다.")
        self.assertEqual(env["GPT_SOVITS_AB_REFERENCE"], "/kaggle/tmp/audio_input/datas/ref.wav")
        self.assertNotIn("HF_TOKEN", env)

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
