"""CPU-only regression checks for notebook setup and launcher boundaries."""

import ast
from importlib.metadata import PackageNotFoundError, version as pkg_version
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import kaggle_launch_webui as launcher


def setup_functions():
    """Load helpers without running installation or model downloads."""
    tree = ast.parse((ROOT / "tools/colab_fast_setup.py").read_text(encoding="utf-8"))
    helpers = ast.Module(
        body=[node for node in tree.body if isinstance(node, ast.FunctionDef)],
        type_ignores=[],
    )
    namespace = {"os": os, "Path": Path, "pkg_version": pkg_version, "PackageNotFoundError": PackageNotFoundError}
    exec(compile(helpers, "setup_helpers", "exec"), namespace)
    return namespace


class StartupTests(unittest.TestCase):
    def test_torchcodec_matches_existing_torch_and_stack_is_constrained(self):
        helpers = setup_functions()
        self.assertTrue("runtime_constraints" in helpers)
        versions = {"torch": "2.6.0+cu124", "torchaudio": "2.6.0+cu124", "torchvision": "0.21.0+cu124"}
        for torch_minor, codec in [(6, "0.2"), (7, "0.5"), (8, "0.7"), (9, "0.9"), (10, "0.10"), (11, "0.11")]:
            versions["torch"] = f"2.{torch_minor}.0+cu124"
            constraints = helpers["runtime_constraints"](versions.__getitem__)
            self.assertIn(f"torch=={versions['torch']}", constraints)
            self.assertIn(f"torchaudio=={versions['torchaudio']}", constraints)
            self.assertIn(f"torchvision=={versions['torchvision']}", constraints)
            self.assertIn(f"torchcodec=={codec}.*", constraints)

    @unittest.skipUnless(
        os.environ.get("GPT_SOVITS_TEST_NLTK_DOWNLOAD") == "1",
        "opt-in check downloads public NLTK resources",
    )
    def test_nltk_resources_are_prepared_and_real_english_tagger_works(self):
        helpers = setup_functions()
        self.assertTrue("prepare_nltk" in helpers)
        import nltk
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ):
            old_paths = nltk.data.path[:]
            try:
                nltk.data.path[:] = [tmp]
                helpers["prepare_nltk"](Path(tmp))
                self.assertEqual(os.environ["NLTK_DATA"].split(os.pathsep)[0], tmp)
                self.assertTrue(nltk.pos_tag(["hello", "world"]))
                self.assertTrue(nltk.corpus.cmudict.words())
                helpers["prepare_nltk"](Path(tmp))
            finally:
                nltk.data.path[:] = old_paths

    def test_nltk_download_failure_is_not_ignored(self):
        helpers = setup_functions()
        import nltk
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ), patch.object(nltk.data, "find", side_effect=LookupError("missing")), patch.object(nltk, "download", side_effect=ValueError("network unavailable")):
            old_paths = nltk.data.path[:]
            try:
                with self.assertRaisesRegex(ValueError, "network unavailable"):
                    helpers["prepare_nltk"](Path(tmp))
            finally:
                nltk.data.path[:] = old_paths

    def test_unknown_torch_pairing_fails_without_replacing_runtime(self):
        with self.assertRaisesRegex(RuntimeError, "No verified TorchCodec pairing"):
            setup_functions()["runtime_constraints"](lambda name: "9.0.0")

    def test_preflight_and_webui_receive_same_runtime_environment(self):
        calls = []
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = {"mode": "auto_list", "datas_dir": tmp, "training_list": "", "audio_count": 1, "source_root": tmp}
            with patch.object(launcher, "ROOT", root), patch.object(launcher, "prepare_kaggle_dataset", return_value=dataset), patch.object(launcher.Path, "mkdir"), patch.object(launcher, "apply_nvidia_library_path", side_effect=lambda env: env.update(LD_LIBRARY_PATH="fixture-cuda") or []), patch.object(launcher.subprocess, "run", side_effect=lambda *a, **kw: calls.append(kw) or type("Result", (), {"returncode": 0})()), patch.object(sys, "argv", ["launcher"]):
                launcher.main()
        self.assertEqual(len(calls), 2)
        self.assertIn("env", calls[0])
        self.assertEqual(calls[0]["env"], calls[1]["env"])
        self.assertEqual(calls[0]["env"]["LD_LIBRARY_PATH"], "fixture-cuda")
        self.assertEqual(calls[0]["env"]["NLTK_DATA"].split(os.pathsep)[0], str(root / "runtime/nltk_data"))

    def test_failed_preflight_prevents_webui_launch(self):
        import subprocess
        dataset = {"mode": "auto_list", "datas_dir": "fixture", "training_list": "", "audio_count": 1, "source_root": "fixture"}
        with patch.object(launcher, "prepare_kaggle_dataset", return_value=dataset), patch.object(launcher.Path, "mkdir"), patch.object(launcher.subprocess, "run", side_effect=subprocess.CalledProcessError(2, "preflight")) as run, patch.object(sys, "argv", ["launcher"]):
            with self.assertRaises(subprocess.CalledProcessError):
                launcher.main()
        self.assertEqual(run.call_count, 1)

    def test_preflight_reports_cublas_error_before_launch(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("preflight_fixture", ROOT / "tools/kaggle_training_preflight.py")
        module = importlib.util.module_from_spec(spec)
        cuda = SimpleNamespace(device_count=lambda: 1, is_available=lambda: True)
        torch = SimpleNamespace(cuda=cuda, float32="fp32", float16="fp16")
        with patch.dict(sys.modules, {"torch": torch}):
            spec.loader.exec_module(module)
        torch.ones = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("libcublas.so.12 missing"))
        with self.assertRaisesRegex(module.PreflightError, "CUDA/cuBLAS check failed on GPU 0"):
            module.gpu_summary()

    def test_notebook_and_embedded_verification_compile(self):
        notebook = json.loads((ROOT / "GPT_SoVITS_Improve_Kaggle_Fast_Minimal.ipynb").read_text(encoding="utf-8"))
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), cell["id"], "exec")
        tree = ast.parse((ROOT / "tools/colab_fast_setup.py").read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in {"smoke", "verify_script"} for t in node.targets):
                compile(ast.literal_eval(node.value), "setup-child", "exec")


if __name__ == "__main__":
    unittest.main()
