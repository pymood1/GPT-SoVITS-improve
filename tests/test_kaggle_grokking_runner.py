import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from tools.kaggle_grokking_runner import _arm_env, _bundle, _read_metrics, _summarize


class KaggleRunnerTests(unittest.TestCase):
    def test_quality_candidate_selection_excludes_failed_and_nonfinite_results(self):
        from tools.kaggle_quality_listening import pick_candidate
        arms = {"a": {"returncode": 0, "best": {"val_loss": 2.0}},
                "b": {"returncode": 0, "best": {"val_loss": 1.0}},
                "c": {"returncode": 1, "best": {"val_loss": 0.0}},
                "d": {"returncode": 0, "best": {"val_loss": float("nan")}}}
        self.assertEqual(pick_candidate(arms, ("a", "b", "c", "d"), "val_loss"), "b")
        with self.assertRaises(RuntimeError):
            pick_candidate(arms, ("c", "d"), "val_loss")

    def test_parallel_session_lanes_reuse_idle_gpu_without_overlap(self):
        from tools.kaggle_grokking_runner import SESSION_SETTINGS, _launchable_jobs
        pending = [(arm["gpu"], arm) for arm in SESSION_SETTINGS["arms"]]
        first = _launchable_jobs(pending, set())
        self.assertEqual([arm["id"] for _, arm in first], ["S1_v_o", "S2_rank16"])
        remaining = [job for job in pending if job not in first]
        self.assertEqual(_launchable_jobs(remaining, {0, 1}), [])
        self.assertEqual([arm["id"] for _, arm in _launchable_jobs(remaining, {1})], ["S1_qkv_o"])
        self.assertEqual([arm["id"] for _, arm in _launchable_jobs(remaining, {0})], ["S2_rank32"])
        self.assertEqual(len({arm["id"] for _, arm in pending}), 4)

    def test_work_stealing_prefers_lane_and_uses_idle_device(self):
        from tools.kaggle_grokking_runner import _dispatch_jobs, SESSION_SETTINGS
        pending = [(arm["gpu"], arm) for arm in SESSION_SETTINGS["arms"]]
        first = _dispatch_jobs(pending, set(), work_stealing=True)
        self.assertEqual([(job[1]["id"], gpu) for job, gpu in first],
                         [("S1_v_o", 0), ("S2_rank16", 1)])
        remaining = [job for job in pending if job not in [j for j, _ in first]]
        self.assertEqual(_dispatch_jobs(remaining, {0, 1}, work_stealing=True), [])
        rank32 = remaining[-1:]
        self.assertEqual(_dispatch_jobs(rank32, {1}, work_stealing=True),
                         [(rank32[0], 0)])
        self.assertEqual(_dispatch_jobs(rank32, {1}, work_stealing=False), [])
        same_lane = [(1, {"id": "a"}), (1, {"id": "b"}), (1, {"id": "c"})]
        scheduled = _dispatch_jobs(same_lane, set(), work_stealing=True)
        self.assertEqual(scheduled, [(same_lane[0], 1), (same_lane[1], 0)])
        self.assertEqual(len({gpu for _, gpu in scheduled}), len(scheduled))

    def test_mix_workers_require_complete_disjoint_reports(self):
        from tools.kaggle_quality_listening import MIX_LABELS, merge_mix_reports
        reports = [
            {"status": "completed", "reference_counts": {"rows": 272},
             "evaluation": {"asr_status": "completed", "speaker_status": "completed"},
             "models": {label: {"status": "completed", "samples": [
                 {"index": i, "cer": 0., "speaker_cosine": .8} for i in range(1, 16)
             ]} for label in labels}}
            for labels in MIX_LABELS
        ]
        result = merge_mix_reports(reports)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(result["models"]), 8)
        reports[0]["models"]["mix025"]["samples"].pop()
        self.assertEqual(merge_mix_reports(reports)["status"], "failed")
        self.assertEqual(merge_mix_reports(reports[:1])["status"], "failed")

    def test_arm_environment_isolated(self):
        with patch.dict(os.environ, {"GPT_SOVITS_GROKFAST": "1", "GPT_SOVITS_GROKKING_S1_LR": "9"}):
            a = _arm_env({"grokfast": False}, 0, Path("a.json"))
            b = _arm_env({"grokfast": True}, 1, Path("b.json"))
        self.assertEqual(a["CUDA_VISIBLE_DEVICES"], "0")
        self.assertEqual(b["CUDA_VISIBLE_DEVICES"], "1")
        self.assertEqual(a["GPT_SOVITS_GROKFAST"], "0")
        self.assertEqual(b["GPT_SOVITS_GROKFAST"], "1")
        self.assertEqual(a["GPT_SOVITS_GROKKING_S1_LR"], b["GPT_SOVITS_GROKKING_S1_LR"])
        self.assertNotEqual(a["GPT_SOVITS_RUNTIME_STATUS_PATH"], b["GPT_SOVITS_RUNTIME_STATUS_PATH"])

    def test_weight_decay_profile_changes_only_decay_and_gpu(self):
        from tools.kaggle_grokking_runner import SETTINGS
        a, b = SETTINGS["arms"]
        ea = _arm_env(a, 0, Path("a.json"))
        eb = _arm_env(b, 1, Path("b.json"))
        self.assertEqual(ea["GPT_SOVITS_GROKKING_S1_WEIGHT_DECAY"], "0.01")
        self.assertEqual(eb["GPT_SOVITS_GROKKING_S1_WEIGHT_DECAY"], "1.0")
        self.assertEqual(ea["GPT_SOVITS_GROKFAST"], "0")
        self.assertEqual(eb["GPT_SOVITS_GROKFAST"], "0")
        self.assertEqual(ea["GPT_SOVITS_GROKKING_S1_FIXED_EPOCHS"], "120")
        self.assertEqual(eb["GPT_SOVITS_GROKKING_S1_FIXED_EPOCHS"], "120")

    def test_s2_profile_isolates_stage_and_config(self):
        from tools.kaggle_grokking_runner import S2_SETTINGS, _s2_config, DEFAULT_PROFILE
        self.assertEqual(DEFAULT_PROFILE, "s2")
        a, b = S2_SETTINGS["arms"]
        ea = _arm_env(a, 0, Path("a.json"), S2_SETTINGS)
        eb = _arm_env(b, 1, Path("b.json"), S2_SETTINGS)
        self.assertEqual(ea["GPT_SOVITS_GROKKING_S1"], "0")
        self.assertEqual(ea["GPT_SOVITS_GROKKING_S2"], "1")
        self.assertEqual(ea["GPT_SOVITS_GROKFAST"], "0")
        self.assertEqual(eb["GPT_SOVITS_GROKFAST"], "1")
        self.assertEqual(ea["GPT_SOVITS_GROKKING_S2_WEIGHT_DECAY"], eb["GPT_SOVITS_GROKKING_S2_WEIGHT_DECAY"])
        self.assertEqual(ea["GPT_SOVITS_GROKKING_S2_FIXED_EPOCHS"], "30")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            (source / "2-name2text.txt").write_text("test")
            (source / "4-cnhubert").mkdir()
            (source / "5-wav32k").mkdir()
            for gpu, arm in enumerate((a, b)):
                out = root / arm["id"]
                out.mkdir()
                cfg = _s2_config(source, out, out / "weights", arm, root / "base.pth", gpu)
                self.assertEqual(cfg["model"]["version"], "v5dev")
                self.assertFalse(cfg["train"]["fp16_run"])
                self.assertEqual(cfg["train"]["gpu_numbers"], str(gpu))
                self.assertEqual(cfg["data"]["exp_dir"], str(out))
                self.assertEqual((out / "4-cnhubert").resolve(), source / "4-cnhubert")
            self.assertFalse((source / "config.json").exists())

    def test_character_error_normalizes_spacing_punctuation_and_unicode(self):
        from tools.kaggle_quality_listening import character_errors
        self.assertEqual(character_errors("안녕하세요!", "안녕 하세요."), (0, 5))
        self.assertEqual(character_errors("가나다", "가다"), (1, 3))
        self.assertEqual(character_errors("가나다", "가나마"), (1, 3))
        self.assertEqual(character_errors("가", "가나다"), (2, 1))
        self.assertEqual(character_errors("안녕", ""), (2, 2))
        self.assertEqual(character_errors("Ａ", "a"), (0, 1))

    def test_quality_reference_prefers_korean_but_accepts_other_languages(self):
        import wave
        from tools.kaggle_quality_listening import choose_reference, ReferenceUnavailable
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "5-wav32k"
            audio.mkdir()
            def wav(name, seconds):
                with wave.open(str(audio / name), "wb") as out:
                    out.setnchannels(1)
                    out.setsampwidth(2)
                    out.setframerate(16000)
                    out.writeframes(b"\x00\x00" * int(seconds * 16000))
            wav("en.wav", 5)
            wav("ko.wav", 9)
            wav("short.wav", 2)
            lines = root / "2-name2text.txt"
            lines.write_text("en.wav\tx\tNone\tHello there.\nko.wav\tx\tNone\t안녕하세요.\nshort.wav\tx\tNone\t짧아요.\n", encoding="utf-8")
            counts = {}
            path, _, lang = choose_reference(root, counts)
            self.assertEqual((path.name, lang), ("ko.wav", "all_ko"))
            self.assertEqual(counts["duration_eligible"], 2)
            (audio / "ko.wav").unlink()
            path, _, lang = choose_reference(root)
            self.assertEqual((path.name, lang), ("en.wav", "auto"))
            (audio / "en.wav").unlink()
            with self.assertRaises(ReferenceUnavailable):
                choose_reference(root)

    def test_quality_profile_compares_learning_rate_only(self):
        from tools.kaggle_grokking_runner import QUALITY_SETTINGS, _s2_config, _arm_env
        a, b = QUALITY_SETTINGS["arms"]
        self.assertFalse(a["grokfast"])
        self.assertFalse(b["grokfast"])
        self.assertEqual(a["weight_decay"], b["weight_decay"])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            (source / "2-name2text.txt").write_text("test")
            (source / "4-cnhubert").mkdir()
            (source / "5-wav32k").mkdir()
            for gpu, arm in enumerate((a, b)):
                out = root / arm["id"]
                out.mkdir()
                cfg = _s2_config(source, out, out / "weights", arm, root / "base.pth", gpu, QUALITY_SETTINGS)
                self.assertEqual(cfg["train"]["learning_rate"], arm["learning_rate"])
                self.assertEqual(cfg["train"]["epochs"], 8)
                env = _arm_env(arm, gpu, out / "status.json", QUALITY_SETTINGS)
                self.assertEqual(env["GPT_SOVITS_GROKFAST"], "0")
                self.assertEqual(env["GPT_SOVITS_GROKKING_S2_FIXED_EPOCHS"], "8")

    def test_progress_line_displays_current_batch_and_gpu(self):
        from tools.kaggle_grokking_runner import _progress_line, S2_SETTINGS
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp)
            (report / "status.json").write_text(json.dumps({
                "epoch": 1, "stage": "training", "batch": 95, "total_batches": 245,
            }))
            process = Mock()
            process.poll.return_value = None
            line = _progress_line("A", [], process, report, S2_SETTINGS, "GPU 88%")
            self.assertIn("epoch 1/30", line)
            self.assertIn("batch 95/245", line)
            self.assertIn("GPU 88%", line)
            process.poll.return_value = 0
            self.assertIn("exit=0", _progress_line("A", [], process, report, S2_SETTINGS, None))

    def test_eta_uses_recent_epoch_durations_and_partial_epoch(self):
        from tools.kaggle_grokking_runner import _remaining_seconds
        rows = [{"epoch_seconds": x} for x in (900, 100, 110, 120, 130, 140)]
        self.assertEqual(_remaining_seconds(rows, 10, 20), 460)
        self.assertEqual(_remaining_seconds(rows, 10, 999), 372)
        self.assertIsNone(_remaining_seconds([], 30))
        self.assertIsNone(_remaining_seconds([{"epoch_seconds": float("nan")}], 30))
        self.assertEqual(_remaining_seconds([{}, {}], 4, 10, [60]), 110)
        self.assertEqual(_remaining_seconds(rows, 6), 0)

    def test_eta_kst_and_safety_cutoff(self):
        from tools.kaggle_grokking_runner import _eta_text
        from datetime import datetime, timezone
        now = datetime(2026, 11, 25, 14, 50, tzinfo=timezone.utc).timestamp()
        self.assertIn("11/26 00:10 KST", _eta_text(1200, now, 5400))
        self.assertIn("20m", _eta_text(1200, now, 5400))
        self.assertIn("11/25 23:55 KST", _eta_text(1200, now, 300))
        self.assertIn("목표 Epoch 미도달", _eta_text(1200, now, 300))
        self.assertIn("계산 중", _eta_text(None, now, 5400))

    def test_cfm_metrics_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p / "grokking_s2_cfm_metrics.jsonl").write_text(
                '{"epoch":1,"val_cfm":0.8}\n'
                '{"epoch":2,"val_cfm":0.5,"delayed_improvement_candidate":true}\n'
            )
            s = _summarize(_read_metrics(p))
            self.assertEqual(s["best"]["epoch"], 2)
            self.assertEqual(s["candidate_epochs"], [2])

    def test_partial_metrics_and_best(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            path = p / "grokking_s1_metrics_s1234.jsonl"
            path.write_text(
                json.dumps({"epoch": 1, "val_loss": 3.0}) + "\n"
                + json.dumps({"epoch": 2, "val_loss": 2.0, "delayed_improvement": True}) + "\n"
                + '{"epoch":3,"val_loss":NaN}\n'
                + '{"epoch":4',
                encoding="utf-8",
            )
            s = _summarize(_read_metrics(p))
            self.assertEqual(s["epochs_recorded"], 2)
            self.assertEqual(s["best"]["epoch"], 2)
            self.assertEqual(s["candidate_epochs"], [2])

    def test_bundle_excludes_audio_and_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = root / "report"
            report.mkdir()
            for name in ("summary.json", "train.log", "voice.wav", "model.ckpt"):
                (report / name).write_text("test")
            target = root / "result.zip"
            _bundle(report, target)
            with zipfile.ZipFile(target) as archive:
                self.assertEqual(set(archive.namelist()), {"summary.json", "train.log"})

    def test_bundle_replaces_snapshot_and_truncates_only_large_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = root / "report"
            report.mkdir()
            target = root / "export" / "result.zip"
            (report / "train.log").write_bytes(b"AAAABBBBCCCCDDDD")
            (report / "metrics.jsonl").write_bytes(b"01234567890123456789")
            with patch("tools.kaggle_grokking_runner.MAX_ARCHIVE_LOG_BYTES", 8):
                _bundle(report, target)
            with zipfile.ZipFile(target) as archive:
                log = archive.read("train.log")
                self.assertTrue(log.startswith(b"AAAA"))
                self.assertTrue(log.endswith(b"DDDD"))
                self.assertIn(b"archive truncated", log)
                self.assertEqual(archive.read("metrics.jsonl"), b"01234567890123456789")
            self.assertEqual((report / "train.log").read_bytes(), b"AAAABBBBCCCCDDDD")
            (report / "train.log").write_text("final")
            _bundle(report, target)
            with zipfile.ZipFile(target) as archive:
                self.assertEqual(archive.read("train.log"), b"final")
            self.assertEqual(list(target.parent.iterdir()), [target])

    def test_failed_snapshot_keeps_previous_zip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = root / "report"
            report.mkdir()
            (report / "summary.json").write_text("{}")
            target = root / "result.zip"
            _bundle(report, target)
            previous = target.read_bytes()
            with patch("tools.kaggle_grokking_runner.zipfile.ZipFile.write", side_effect=OSError("disk")):
                with self.assertRaises(OSError):
                    _bundle(report, target)
            self.assertEqual(target.read_bytes(), previous)
            self.assertFalse(target.with_name("." + target.name + ".part").exists())


    def test_single_archive_replaces_previous_run_without_mixing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "grokking_result.zip"
            old = root / "old"
            old.mkdir()
            (old / "summary.json").write_text('{"run_id": "old"}')
            _bundle(old, target)
            current = root / "current"
            (current / "bootstrap").mkdir(parents=True)
            (current / "training").mkdir()
            (current / "run.json").write_text('{"run_id": "new"}')
            (current / "bootstrap" / "setup.log").write_text("setup")
            (current / "training" / "summary.json").write_text("training")
            _bundle(current, target)
            with zipfile.ZipFile(target) as archive:
                self.assertEqual(set(archive.namelist()), {
                    "run.json", "bootstrap/setup.log", "training/summary.json"
                })
                self.assertEqual(json.loads(archive.read("run.json"))["run_id"], "new")

    def test_legacy_archive_migration_preserves_unrelated_files(self):
        from tools.kaggle_grokking_bootstrap import _archive_legacy_results
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            working = root / "working"
            working.mkdir()
            backup = root / "backup"
            old = {"grokking_20261125T013234_805522Z.zip",
                   "grokking_bootstrap_20261125T013054_087532Z.zip"}
            keep = {"grokking_result.zip", "grokking_notes.zip", "user.zip"}
            for name in old | keep:
                (working / name).write_text(name)
            _archive_legacy_results(working, backup)
            self.assertEqual({p.name for p in working.iterdir()}, keep)
            self.assertEqual({p.name for p in backup.iterdir()}, old)
            for name in old:
                self.assertEqual((backup / name).read_text(), name)




if __name__ == "__main__":
    unittest.main()
