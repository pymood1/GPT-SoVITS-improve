import hashlib
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

from tools.colab_v5_acoustic_ab import (
    find_completed_suite,
    run_headless,
    runtime_environment,
    verified_download,
)


class ColabAcousticABTests(unittest.TestCase):
    def test_runtime_environment_selects_only_pretrained_v5dev_assets(self):
        repo = Path("/content/GPT-SoVITS-acoustic-ab")
        output = Path("/content/v5_acoustic_ab")
        reference = Path("/content/v5_acoustic_ab/reference.wav")

        env = runtime_environment(repo, output, reference)

        self.assertEqual(env["version"], "v5dev")
        self.assertEqual(env["GPT_SOVITS_V5_AB_REQUIRE_PRETRAINED"], "1")
        self.assertEqual(env["gpt_path"], str(repo / "GPT_SoVITS/pretrained_models/s1v3.ckpt"))
        self.assertEqual(
            env["sovits_path"],
            str(repo / "GPT_SoVITS/pretrained_models/gsv-v5-pretrained/s2Gv5dev.pth"),
        )
        self.assertEqual(env["GPT_SOVITS_V5_ACOUSTIC_AB_DIR"], str(output))
        self.assertEqual(env["GPT_SOVITS_AB_REFERENCE"], str(reference))
        self.assertNotIn("emotion_controller_path", env)

    def test_headless_run_consumes_generator_with_english_reference_and_korean_target(self):
        calls = []

        def get_tts_wav(*args, **kwargs):
            calls.append((args, kwargs))
            yield 48000, b"ordinary-output"

        webui = SimpleNamespace(
            dict_language={"English label": "en", "Korean label": "all_ko"},
            get_tts_wav=get_tts_wav,
        )

        yielded = run_headless(webui, "/content/reference.wav")

        self.assertEqual(yielded, 1)
        self.assertEqual(len(calls), 1)
        args, kwargs = calls[0]
        self.assertEqual(args[0], "/content/reference.wav")
        self.assertEqual(args[1], "I had that curiosity beside me at this moment.")
        self.assertEqual(args[2], "English label")
        self.assertEqual(args[4], "Korean label")
        self.assertEqual(kwargs["sample_steps"], 32)
        self.assertEqual(kwargs["cfg_rate"], 1.30)
        self.assertFalse(kwargs["use_cuda_graph"])
        self.assertFalse(kwargs["use_flash_attention"])

    def test_verified_download_rejects_content_that_does_not_match_pinned_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.wav"
            target = Path(tmp) / "target.wav"
            source.write_bytes(b"known reference")
            digest = hashlib.sha256(source.read_bytes()).hexdigest()

            verified_download(source.as_uri(), target, digest)
            self.assertEqual(target.read_bytes(), b"known reference")

            target.unlink()
            with self.assertRaisesRegex(RuntimeError, "SHA-256"):
                verified_download(source.as_uri(), target, "0" * 64)
            self.assertFalse(target.exists())

    def test_completed_suite_ignores_partial_and_failed_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            partial = output / "100"
            failed = output / "200"
            complete = output / "300"
            for folder in (partial, failed, complete):
                folder.mkdir()
            (partial / "report.json").write_text(json.dumps({"suite": True}), encoding="utf-8")
            (failed / "report.json").write_text(
                json.dumps({"suite": True, "completed": True, "failure": "boom"}), encoding="utf-8"
            )
            (complete / "report.json").write_text(
                json.dumps({"suite": True, "completed": True, "failure": None}), encoding="utf-8"
            )

            self.assertEqual(find_completed_suite(output), complete)


if __name__ == "__main__":
    unittest.main()
