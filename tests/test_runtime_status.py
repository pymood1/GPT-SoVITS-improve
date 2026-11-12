import json
import tempfile
import time
import unittest
from pathlib import Path

from tools.runtime_status import (
    clear_activity,
    publish_activity,
    read_activity,
    render_status_html,
    build_runtime_status_payload,
)


ROOT = Path(__file__).resolve().parents[1]


class RuntimeStatusTests(unittest.TestCase):
    def test_activity_round_trip_and_clear(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "runtime_status.json"
            publish_activity(
                path=path,
                task="Emotion Controller",
                stage="training",
                progress=0.625,
                step=312,
                total_steps=500,
                loss=1.287,
            )
            state = read_activity(path=path, stale_after=60.0)
            self.assertEqual(state["task"], "Emotion Controller")
            self.assertEqual(state["stage"], "training")
            self.assertAlmostEqual(state["progress"], 0.625)
            self.assertEqual(state["step"], 312)
            self.assertEqual(state["total_steps"], 500)
            self.assertAlmostEqual(state["loss"], 1.287)

            clear_activity(path=path)
            self.assertEqual(
                read_activity(path=path, stale_after=60.0)["state"],
                "idle",
            )

    def test_stale_activity_degrades_to_idle(self):
        for state_name in ("running", "failed"):
            with self.subTest(state=state_name):
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / "runtime_status.json"
                    path.write_text(
                        json.dumps(
                            {
                                "state": state_name,
                                "task": "old task",
                                "updated_at": time.time() - 120,
                            }
                        ),
                        encoding="utf-8",
                    )
                    state = read_activity(
                        path=path,
                        stale_after=15.0,
                    )
                    self.assertEqual(state["state"], "idle")
                    self.assertEqual(state["reason"], "stale")

    def test_render_escapes_text_and_shows_progress(self):
        hardware = {
            "cpu_percent": 42.0,
            "ram_percent": 61.0,
            "ram_used_gib": 7.3,
            "ram_total_gib": 12.0,
            "gpu_available": True,
            "gpu_name": "<T4>",
            "gpu_percent": 78.0,
            "vram_used_gib": 11.2,
            "vram_total_gib": 15.0,
            "vram_percent": 74.7,
            "gpu_temperature_c": 66.0,
        }
        activity = {
            "state": "running",
            "task": "<script>alert(1)</script>",
            "stage": "training",
            "progress": 0.68,
            "step": 340,
            "total_steps": 500,
            "loss": 1.292,
            "metric": "val 1.357",
        }
        html = render_status_html(hardware, activity)
        self.assertIn("status-dashboard", html)
        self.assertIn("68%", html)
        self.assertIn("340 / 500", html)
        self.assertIn("1.292", html)
        self.assertIn("&lt;script&gt;", html)
        self.assertNotIn("<script>alert", html)
        self.assertIn("&lt;T4&gt;", html)

    def test_training_paths_publish_runtime_status(self):
        required = {
            "GPT_SoVITS/s1_train.py": "publish_activity",
            "GPT_SoVITS/s2_train.py": "publish_activity",
            "GPT_SoVITS/s2_train_v3_lora.py": "publish_activity",
            "tools/train_emotion_controller.py": "publish_activity",
        }
        for relative, marker in required.items():
            text = (ROOT / relative).read_text(encoding="utf-8")
            self.assertIn(marker, text, relative)
            self.assertIn("clear_activity", text, relative)

    def test_runtime_payload_is_json(self):
        payload = json.loads(build_runtime_status_payload())
        self.assertIn("hardware", payload)
        self.assertIn("activity", payload)
        self.assertIn("cpu_percent", payload["hardware"])
        self.assertIn("state", payload["activity"])

    def test_webui_patches_static_dashboard_without_rerender(self):
        webui = (ROOT / "webui.py").read_text(encoding="utf-8")
        self.assertIn("runtime_status_monitor = gr.HTML(", webui)
        self.assertIn('elem_id="runtime-status-monitor"', webui)
        self.assertIn("runtime_status_payload = gr.Textbox(", webui)
        self.assertIn("runtime_status_timer = gr.Timer(value=2", webui)
        self.assertIn("runtime_status_timer.tick(", webui)
        self.assertIn("outputs=[runtime_status_payload]", webui)
        self.assertNotIn("outputs=[runtime_status_monitor]", webui)
        self.assertIn("runtime_status_payload.change(", webui)
        self.assertIn("js=runtime_status_patch_js", webui)
        self.assertIn("queue=False", webui)


if __name__ == "__main__":
    unittest.main()
