import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class GradioLayoutRegressionTests(unittest.TestCase):
    def test_main_blocks_use_full_width(self):
        webui = (ROOT / "webui.py").read_text(encoding="utf-8")
        self.assertRegex(
            webui,
            r"with gr\.Blocks\([^\n]*fill_width=True",
        )

    def test_no_direct_row_inside_row(self):
        webui = (ROOT / "webui.py").read_text(encoding="utf-8")
        lines = webui.splitlines()
        direct_nested_rows = []

        for index, line in enumerate(lines):
            match = re.match(r"^(\s*)with gr\.Row\(", line)
            if not match:
                continue
            parent_indent = len(match.group(1))

            for child_index in range(index + 1, len(lines)):
                child = lines[child_index]
                if not child.strip():
                    continue
                child_indent = len(child) - len(child.lstrip())
                if child_indent <= parent_indent:
                    break
                if child.strip().startswith("with gr.Row("):
                    direct_nested_rows.append(
                        (index + 1, child_index + 1)
                    )
                break

        self.assertEqual(direct_nested_rows, [])

    def test_training_errors_are_persistent(self):
        webui = (ROOT / "webui.py").read_text(encoding="utf-8")
        self.assertIn("def _persistent_gradio_errors(", webui)
        self.assertIn("duration=None", webui)
        for function_name in (
            "open1a",
            "open1b",
            "open1c",
            "open1abc",
            "open1abcd",
            "open1Ba",
            "open1Bb",
            "run_auto_emotion_sidecar",
        ):
            self.assertIn(
                f"@_persistent_gradio_errors\ndef {function_name}(",
                webui,
            )
        self.assertIn(
            "@_persistent_gradio_errors\ndef run_gpt_then_sovits(",
            webui,
        )
        self.assertIn(
            "@_persistent_gradio_errors\ndef run_emotion_training_pipeline(",
            webui,
        )
        self.assertIn("show_error=True", webui)

    def test_emotion_ui_stream_is_throttled(self):
        webui = (ROOT / "webui.py").read_text(encoding="utf-8")
        start = webui.index("def _stream_emotion_command(")
        end = webui.index("\ndef stop_emotion_training(", start)
        block = webui[start:end]
        self.assertIn("last_ui_update = 0.0", block)
        self.assertIn("now - last_ui_update >= 0.25", block)
        self.assertIn("pending_ui_update", block)
        self.assertIn('"[probe]"', block)

    def test_multi_gpu_training_defaults_are_split(self):
        webui = (ROOT / "webui.py").read_text(encoding="utf-8")
        self.assertIn("default_gpt_training_gpu", webui)
        self.assertIn("default_sovits_training_gpu", webui)
        self.assertIn("sorted(GPU_INDEX)", webui)
        self.assertIn(
            "value=default_gpt_training_gpu",
            webui,
        )
        self.assertIn(
            "value=default_sovits_training_gpu",
            webui,
        )

    def test_parallel_training_uses_disjoint_single_gpus(self):
        webui = (ROOT / "webui.py").read_text(encoding="utf-8")
        start = webui.index("def _single_training_gpu(")
        end = webui.index("\ndef stop_gpt_then_sovits()", start)
        block = webui[start:end]
        self.assertIn("len(tokens) != 1", block)
        self.assertIn("gpt_gpu != sovits_gpu", block)
        self.assertIn("if len(tokens) != 1:", block)
        self.assertIn("threading.Thread(", block)
        self.assertIn('name="gpt-training"', block)
        self.assertIn('name="sovits-training"', block)
        self.assertIn("time.sleep(0.25)", block)
        self.assertIn("result[3]", block)
        self.assertIn("result[4]", block)
        self.assertNotIn("sovits_update = result[2]", block)

    def test_training_children_use_independent_status_slots(self):
        webui = (ROOT / "webui.py").read_text(encoding="utf-8")
        self.assertIn(
            'child_env[STATUS_PATH_ENV] = str(status_slot_path("gpt"))',
            webui,
        )
        self.assertIn(
            'child_env[STATUS_PATH_ENV] = str(status_slot_path("sovits"))',
            webui,
        )
        self.assertIn(
            'child_env[STATUS_PATH_ENV] = str(status_slot_path("emotion"))',
            webui,
        )

    def test_header_is_responsive(self):
        assets = (ROOT / "tools" / "assets.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('class="app-header-links"', assets)
        self.assertIn(".app-header-links", assets)
        self.assertIn("flex-wrap: wrap", assets)
        self.assertIn("@media (max-width: 900px)", assets)


if __name__ == "__main__":
    unittest.main()
