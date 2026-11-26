import unittest
from unittest.mock import patch
from tools.kaggle_live_status import parse_progress, kernel_status, cli


class SafeProgressTests(unittest.TestCase):
    def test_per_worker_and_legacy_counts(self):
        data = """
[stage] launcher starting
[quality] gen_base: completed 18 samples
1400.0s | [gsv-progress] gpu0 asr all 9/54
[gsv-progress] gpu1 speaker all 3/54
[gsv-progress] gpu0 asr all 10/54
"""
        result = parse_progress(data)
        self.assertEqual(result["completed_synthesis"], {"gen_base": 18})
        self.assertEqual(result["workers"]["gpu0"], dict(phase="asr", label="all", done=10, total=54))
        self.assertEqual(result["workers"]["gpu1"]["phase"], "speaker")

    def test_reject_private_text_and_malformed_events(self):
        result = parse_progress("""
transcript: [gsv-progress] gpu0 asr all 9/54
[gsv-progress] gpu1 asr all 900/54
[gsv-progress] gpu1 private all 1/2
[gsv-progress] gpu1 asr private_name 1/2
[gsv-progress] gpu0 asr all 1/2 secret
[quality] private_name: completed 18 samples
[quality] gen_base: completed 999999 samples
""")
        self.assertEqual(result, {"workers": {}, "completed_synthesis": {}})

    def test_status(self):
        self.assertEqual(kernel_status('kernel has status "KernelWorkerStatus.RUNNING"'), "running")
        self.assertEqual(kernel_status("arbitrary unknown output"), "unknown")
        self.assertEqual(kernel_status("KernelWorkerStatus.COMPLETE"), "complete")

    @patch("tools.kaggle_live_status.subprocess.run")
    def test_cli_does_not_return_failed_command_output(self, run):
        run.return_value.returncode = 1
        run.return_value.stdout = "private output"
        self.assertEqual(cli(["logs", "fake"]), (False, ""))


if __name__ == "__main__":
    unittest.main()
