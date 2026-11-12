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
        self.assertIn("@_persistent_gradio_errors\ndef open1abcd(", webui)
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
        self.assertIn(
            "value=default_gpt_training_gpu",
            webui,
        )
        self.assertIn(
            "value=default_sovits_training_gpu",
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
