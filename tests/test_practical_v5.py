import json
import tempfile
import unittest
import wave
from pathlib import Path

from tools.practical_v5 import mounted_sources


class PracticalMountTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.candidate = "pymood1/gpt-sovits-research-1"
        self.reference = "pymood1/gpt-sovits-research-2"
        self.weights = self.root / "gpt-sovits-research-1/private-candidates/S2_rank32_grok_cfm_best.pth"
        self.weights.parent.mkdir(parents=True)
        self.weights.write_bytes(b"07fixture")
        self.refs = self.root / "gpt-sovits-research-2/private-listening/gpu0"
        self.refs.mkdir(parents=True)
        for i in range(1, 4):
            with wave.open(str(self.refs / f"reference-{i:02d}.wav"), "wb") as f:
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(8000)
                f.writeframes(bytes([i, 0]) * 32000)
            (self.refs / f"reference-{i:02d}.json").write_text(
                json.dumps({"text": "test", "language": "auto"}), encoding="utf-8")

    def test_pinned_mount(self):
        weight, prompts = mounted_sources(self.root, self.candidate, self.reference)
        self.assertEqual(weight, self.weights)
        self.assertEqual(len(prompts), 3)

    def test_wrong_header(self):
        self.weights.write_bytes(b"PKfixture")
        with self.assertRaises(ValueError):
            mounted_sources(self.root, self.candidate, self.reference)

    def test_duplicate_audio(self):
        (self.refs / "reference-02.wav").write_bytes((self.refs / "reference-01.wav").read_bytes())
        with self.assertRaises(ValueError):
            mounted_sources(self.root, self.candidate, self.reference)

    def test_missing_transcript(self):
        (self.refs / "reference-03.json").unlink()
        with self.assertRaises(FileNotFoundError):
            mounted_sources(self.root, self.candidate, self.reference)

    def test_no_fallback_to_other_kernel(self):
        with self.assertRaises(FileNotFoundError):
            mounted_sources(self.root, self.candidate, "pymood1/gpt-sovits-research-9")

    def test_ambiguous_checkpoint(self):
        other = self.root / "another/gpt-sovits-research-1/private-candidates" / self.weights.name
        other.parent.mkdir(parents=True)
        other.write_bytes(b"07fixture")
        with self.assertRaises(FileNotFoundError):
            mounted_sources(self.root, self.candidate, self.reference)


if __name__ == "__main__":
    unittest.main()
