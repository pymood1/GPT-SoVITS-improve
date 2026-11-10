"""Regression coverage for the jieba backend used by Chinese tone sandhi."""
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "GPT_SoVITS" / "text" / "tone_sandhi.py"


def load_tone_sandhi():
    spec = importlib.util.spec_from_file_location("_tone_sandhi_regression", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ToneSandhiBackendTests(unittest.TestCase):
    def check_tones(self, module):
        sandhi = module.ToneSandhi()
        self.assertEqual("".join(sandhi._split_word("你好")), "你好")
        self.assertEqual(
            sandhi.modified_tone("你好", "l", ["i3", "ao3"]),
            ["i2", "ao3"],
        )

    def test_fallback_without_jieba_fast(self):
        # Mirrors Python >=3.13 requirements: jieba-py exports "jieba".
        with patch.dict(sys.modules, {"jieba_fast": None}):
            module = load_tone_sandhi()
            self.check_tones(module)

    def test_fast_backend_remains_preferred_when_installed(self):
        if importlib.util.find_spec("jieba_fast") is None:
            self.skipTest("jieba_fast is not installed on this Python version")
        import jieba_fast
        module = load_tone_sandhi()
        self.assertIs(module.jieba, jieba_fast)
        self.check_tones(module)

    def test_missing_both_backends_reports_import_error(self):
        with patch.dict(sys.modules, {"jieba_fast": None, "jieba": None}):
            with self.assertRaises(ImportError):
                load_tone_sandhi()


if __name__ == "__main__":
    unittest.main()
