import hashlib
import json
import sys
import tempfile
from pathlib import Path
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

from tools.colab_v5_acoustic_ab import (
    _run,
    ensure_nltk_frontend_resources,
    find_completed_suite,
    run_headless,
    runtime_environment,
    select_korean_reference,
    verified_download,
)


class ColabAcousticABTests(unittest.TestCase):
    def test_child_failure_keeps_stderr_in_exception_and_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "child.log"
            command = [
                sys.executable,
                "-c",
                "import sys; print('actual child cause', file=sys.stderr); raise SystemExit(7)",
            ]

            with self.assertRaisesRegex(RuntimeError, "exit code 7[\\s\\S]*actual child cause"):
                _run(command, log_path=log_path)

            self.assertIn("actual child cause", log_path.read_text(encoding="utf-8"))

    def test_notebook_bootstrap_supplies_source_location_to_legacy_loader(self):
        notebook = json.loads(
            Path("notebooks/GPT_SoVITS_V5_Acoustic_AB_Colab.ipynb").read_text(encoding="utf-8")
        )
        code = next("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")
        legacy_script = b"def run_colab_all(): return globals().get('__file__')"
        response = SimpleNamespace(read=lambda: legacy_script)
        namespace = {}

        with mock.patch("urllib.request.urlopen", return_value=response):
            exec(compile(code, "colab-cell", "exec"), namespace)

        self.assertEqual(namespace["result_folder"], namespace["url"])

    def test_raw_bootstrap_without_dunder_file_is_treated_as_external_loader(self):
        script = Path("tools/colab_v5_acoustic_ab.py")
        scope = {"__name__": "colab_v5_ab_bootstrap"}

        exec(compile(script.read_bytes(), str(script), "exec"), scope)

        self.assertFalse(scope["_running_from_colab_checkout"]())

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
        self.assertEqual(env["GPT_SOVITS_AB_REFERENCE_TEXT"], "I had that curiosity beside me at this moment.")
        self.assertEqual(env["GPT_SOVITS_AB_REFERENCE_LANGUAGE"], "en")
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

    def test_headless_run_accepts_korean_reference_without_english_frontend(self):
        calls = []

        def get_tts_wav(*args, **kwargs):
            calls.append((args, kwargs))
            yield 48000, b"ordinary-output"

        webui = SimpleNamespace(
            dict_language={"Korean label": "all_ko"},
            get_tts_wav=get_tts_wav,
        )

        yielded = run_headless(
            webui,
            "/content/korean.wav",
            "안녕하세요. 참조 음성입니다.",
            "all_ko",
        )

        self.assertEqual(yielded, 1)
        args, _ = calls[0]
        self.assertEqual(args[1], "안녕하세요. 참조 음성입니다.")
        self.assertEqual(args[2], "Korean label")
        self.assertEqual(args[4], "Korean label")

    def test_select_korean_reference_prefers_valid_three_to_ten_second_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "sample.wav"
            audio.write_bytes(b"fixture")
            listing = root / "training.list"
            listing.write_text(
                f"{audio}|speaker|ko|정확한 한국어 전사문\n",
                encoding="utf-8",
            )

            selected = select_korean_reference(
                [listing],
                lambda _: SimpleNamespace(duration=5.5),
            )

        self.assertIsNotNone(selected)
        path, transcript = selected
        self.assertEqual(path.name, "sample.wav")
        self.assertEqual(transcript, "정확한 한국어 전사문")

    def test_nltk_frontend_preflight_installs_only_missing_resources_and_runs_pos_tag(self):
        nltk = ModuleType("nltk")
        corpus = ModuleType("nltk.corpus")
        installed = set()
        downloads = []
        pos_calls = []

        resource_for_package = {
            "averaged_perceptron_tagger_eng": "taggers/averaged_perceptron_tagger_eng",
            "cmudict": "corpora/cmudict",
            "averaged_perceptron_tagger": "taggers/averaged_perceptron_tagger",
        }

        def find(resource):
            if resource not in installed:
                raise LookupError(resource)
            return resource

        def download(package, quiet=True):
            downloads.append(package)
            installed.add(resource_for_package[package])
            return True

        nltk.data = SimpleNamespace(find=find)
        nltk.download = download
        nltk.pos_tag = lambda words: pos_calls.append(tuple(words)) or [(word, "NN") for word in words]
        corpus.cmudict = SimpleNamespace(dict=lambda: {"frontend": [["F"]]})

        with mock.patch.dict(sys.modules, {"nltk": nltk, "nltk.corpus": corpus}):
            ensure_nltk_frontend_resources()

        self.assertEqual(downloads, ["averaged_perceptron_tagger_eng", "cmudict"])
        self.assertEqual(pos_calls, [("frontend", "check")])

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
