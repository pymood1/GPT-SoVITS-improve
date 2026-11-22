import contextlib
import io
import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'GPT_SoVITS'))
from AR.models import emotion_control as ec


class SteeringDiagnosticsTests(unittest.TestCase):
    def _trace_records(self, controller=None):
        torch.manual_seed(65)
        controller = controller or ec.LowRankEmotionPreference(8, 4, rank=4)
        controller._steering_trace_identity = {
            'controller_sha256': 'c' * 64,
            'bound_s1_sha256': 'a' * 64,
        }
        weight = torch.randn(11, 8)
        records = []
        with patch.object(ec, 'STEERING_TRACE', True):
            with contextlib.redirect_stderr(io.StringIO()) as log:
                for strength in (1.0, 2.0):
                    for _ in range(2):
                        state = ec.EmotionSamplingState(torch.tensor([1.9, 0.0]))
                        for _ in range(2):
                            _, _, state = ec.sample_with_emotion_preference(
                                torch.zeros(2, 11), torch.randn(2, 8), None,
                                weight, controller, torch.tensor([0, 1]),
                                strength=strength, state=state, eos_id=10, top_k=11,
                            )
                records = [json.loads(line) for line in log.getvalue().splitlines()]
        return records

    def test_trace_conditions_identity_and_multiple_generation_summary(self):
        from tools.diagnose_emotion_steering import summarize_trace
        records = self._trace_records()
        self.assertEqual(len({record['generation'] for record in records}), 4)
        self.assertEqual(records[0]['rows'][0]['emotion_id'], 0)
        self.assertEqual(records[0]['rows'][0]['strength'], 1.0)
        self.assertEqual(records[-1]['rows'][0]['strength'], 2.0)
        report = summarize_trace(records, {
            'controller_sha256': 'c' * 64, 's1_sha256': 'a' * 64,
            's1_pairing': 'verified', 'emotion_order': ['a', 'b', 'c', 'd'],
        })
        self.assertEqual(report['sampler_generations'], 4)
        self.assertEqual(report['sequences'], 8)
        self.assertEqual(len(report['groups']), 4)
        for group in report['groups']:
            self.assertEqual(group['sequences'], 2)
            self.assertEqual(group['observed_row_steps'], 4)
            expected = 1.0 if group['emotion_id'] == 0 else 0.0
            self.assertEqual(group['budget_disabled_fraction_pooled'], expected)
            if expected == 1.0:
                self.assertIsNone(group['mean_abs_delta_active_support_weighted'])
                self.assertIsNone(group['total_variation_active_row_step_mean'])
            else:
                self.assertIsNotNone(group['mean_abs_delta_active_support_weighted'])
        with self.assertRaisesRegex(ValueError, 'identity|SHA|pair'):
            summarize_trace(records, {
                'controller_sha256': 'b' * 64, 's1_sha256': 'a' * 64,
                's1_pairing': 'verified', 'emotion_order': ['a', 'b', 'c', 'd'],
            })

    def test_constant_tilt_has_zero_effective_probability_change(self):
        controller = ec.LowRankEmotionPreference(8, 4, rank=4)
        with patch.object(controller, 'preference_scores', return_value=torch.full((2, 11), 0.03)):
            records = self._trace_records(controller)
        row = records[0]['rows'][1]
        self.assertAlmostEqual(row['mean_abs_delta'], 0.03, places=6)
        self.assertLess(row['total_variation'], 1e-6)

    def test_trace_summary_cli_and_incomplete_log_detection(self):
        from tools.diagnose_emotion_steering import main
        records = self._trace_records()
        static = {
            'controller_sha256': 'c' * 64, 's1_sha256': 'a' * 64,
            's1_pairing': 'verified', 'emotion_order': ['a', 'b', 'c', 'd'],
        }
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            log_path = folder / 'trace.log'
            static_path = folder / 'static.json'
            output = folder / 'summary.json'
            # Drop first step of first generation, retain unrelated stderr text.
            lines = ['ordinary runtime message'] + [json.dumps(row) for row in records[1:]]
            log_path.write_text('\n'.join(lines), encoding='utf-8')
            static_path.write_text(json.dumps(static), encoding='utf-8')
            with contextlib.redirect_stdout(io.StringIO()):
                main(['--trace-log', str(log_path), '--static-report', str(static_path),
                      '--output', str(output)])
            report = json.loads(output.read_text())
            self.assertEqual(report['incomplete_trace_generations'], 1)
            self.assertGreater(len(report['warnings']), 0)

    def test_observer_failure_never_changes_sample(self):
        torch.manual_seed(91)
        controller = ec.LowRankEmotionPreference(8, 4, rank=4)
        logits, hidden, weight = torch.randn(1, 11), torch.randn(1, 8), torch.randn(11, 8)
        def run(flag):
            torch.manual_seed(89)
            with patch.object(ec, 'STEERING_TRACE', flag):
                return ec.sample_with_emotion_preference(
                    logits.clone(), hidden, None, weight, controller, 0,
                )
        baseline = run(False)
        with patch.object(ec, '_log_steering_step', side_effect=OSError('closed stream')):
            observed = run(True)
        self.assertTrue(torch.equal(baseline[0], observed[0]))
        self.assertTrue(torch.equal(baseline[1], observed[1]))

    def test_off_on_equality_across_modes_and_safety_cases(self):
        cases = [
            ('hidden_direction', 1.0, 2.0, 0.35, 5, 10),
            ('legacy_low_rank', 2.0, 0.0, 0.35, 5, None),
            ('hidden_direction', 0.0, 2.0, 0.35, 5, 10),
            ('hidden_direction', 2.0, 2.0, 0.0, 5, 10),
            ('hidden_direction', 2.0, 2.0, 0.35, 1, 10),
        ]
        for mode, strength, budget, clamp, top_k, eos_id in cases:
            with self.subTest(mode=mode, strength=strength, top_k=top_k, clamp=clamp):
                torch.manual_seed(78)
                controller = ec.LowRankEmotionPreference(
                    8, 4, rank=4, steering_mode=mode,
                    prefix_log_ratio_budget=budget, max_log_odds_shift=clamp,
                )
                with torch.no_grad():
                    controller.u_proj.normal_()
                logits = torch.randn(2, 11)
                hidden, weight = torch.randn(2, 8), torch.randn(11, 8)
                results = []
                for enabled in (False, True):
                    torch.manual_seed(943)
                    state = ec.EmotionSamplingState(torch.tensor([-1.9, 0.0]))
                    with patch.object(ec, 'STEERING_TRACE', enabled):
                        with contextlib.redirect_stderr(io.StringIO()) as log:
                            sample, probs, state = ec.sample_with_emotion_preference(
                                logits.clone(), hidden, None, weight, controller,
                                torch.tensor([0, 2]), strength=strength, eos_id=eos_id,
                                top_k=top_k, state=state,
                            )
                    results.append((
                        sample, probs, state.cumulative_log_ratio, torch.get_rng_state(),
                    ))
                    if enabled:
                        record = json.loads(log.getvalue())
                        if strength == 0 or clamp == 0 or top_k == 1:
                            self.assertEqual(record['eligible_row_steps'], 0)
                            self.assertEqual(record['budget_disabled_row_steps'], 0)
                for before, after in zip(*results):
                    self.assertTrue(torch.equal(before, after))

    def test_cli_static_then_loaded_trace_then_summary(self):
        from tools.diagnose_emotion_steering import main
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            s1, path = folder / 's1.ckpt', folder / 'controller.pt'
            weight = torch.randn(19, 8)
            torch.save({'weight': {'model.ar_predict_layer.weight': weight}}, s1)
            ec.save_emotion_controller_checkpoint(
                ec.LowRankEmotionPreference(8, 4, rank=4), str(path),
                emotion_names=['happy', 'angry', 'disgusted', 'surprised'],
                extra={'s1_sha256': ec.sha256_file(str(s1))},
            )
            static_path, trace_path = folder / 'static.json', folder / 'trace.log'
            summary_path = folder / 'summary.json'
            with contextlib.redirect_stdout(io.StringIO()):
                main(['--controller', str(path), '--s1-checkpoint', str(s1),
                      '--output', str(static_path)])
            with patch.object(ec, 'STEERING_TRACE', True):
                loaded, _ = ec.load_emotion_controller_checkpoint(str(path))
                with contextlib.redirect_stderr(io.StringIO()) as log:
                    for emotion_id in range(4):
                        for strength in (1.0, 2.0):
                            for _ in range(2):
                                ec.sample_with_emotion_preference(
                                    torch.randn(1, 19), torch.randn(1, 8), None,
                                    weight, loaded, emotion_id, strength=strength,
                                )
            trace_path.write_text(log.getvalue(), encoding='utf-8')
            with contextlib.redirect_stdout(io.StringIO()):
                main(['--trace-log', str(trace_path), '--static-report', str(static_path),
                      '--output', str(summary_path)])
            report = json.loads(summary_path.read_text())
            self.assertEqual(len(report['groups']), 8)
            self.assertEqual(report['sampler_generations'], 16)
            self.assertTrue(all(group['sequences'] == 2 for group in report['groups']))

    def test_summary_duplicate_records_and_sampling_mismatch(self):
        from tools.diagnose_emotion_steering import summarize_trace
        records = self._trace_records()
        static = {
            'controller_sha256': 'c' * 64, 's1_sha256': 'a' * 64,
            's1_pairing': 'verified', 'emotion_order': ['a', 'b', 'c', 'd'],
        }
        report = summarize_trace([*records, records[0]], static)
        self.assertEqual(report['duplicate_records_ignored'], 1)
        altered = json.loads(json.dumps(records))
        altered[-1]['runtime']['sampling']['temperature'] = 0.5
        with self.assertRaisesRegex(ValueError, 'settings differ'):
            summarize_trace(altered, static)

    def test_half_precision_filtered_trace_metrics_are_finite(self):
        torch.manual_seed(74)
        controller = ec.LowRankEmotionPreference(8, 4, rank=4).half()
        logits, hidden = torch.randn(2, 11).half(), torch.randn(2, 8).half()
        weight = torch.randn(11, 8).half()
        results = []
        for enabled in (False, True):
            torch.manual_seed(84)
            with patch.object(ec, 'STEERING_TRACE', enabled):
                with contextlib.redirect_stderr(io.StringIO()) as log:
                    sample, probs, state = ec.sample_with_emotion_preference(
                        logits.clone(), hidden, None, weight, controller, 0,
                        eos_id=10, top_k=5,
                    )
            results.append((sample, probs, state.cumulative_log_ratio, torch.get_rng_state()))
        self.assertTrue(log.getvalue(), 'FP16 filtered trace must not disappear silently')
        record = json.loads(log.getvalue())
        for row in record['rows']:
            self.assertTrue(torch.isfinite(torch.tensor(row['total_variation'])))
            self.assertTrue(torch.isfinite(torch.tensor(
                row['base_weighted_abs_log_probability_ratio']
            )))
        for before, after in zip(*results):
            self.assertTrue(torch.equal(before, after))

    def test_budget_trace_uses_the_same_state_dtype_as_the_sampler(self):
        controller = ec.LowRankEmotionPreference(8, 4, rank=4).half()
        logits = torch.zeros(1, 11).half()
        hidden, weight = torch.zeros(1, 8).half(), torch.ones(11, 8).half()
        before = torch.tensor([1.3001], dtype=torch.float32)
        expected = bool((
            controller.prefix_log_ratio_budget - before.half().abs()
            < 2 * controller.max_log_odds_shift
        ).item())
        with patch.object(ec, 'STEERING_TRACE', True):
            with contextlib.redirect_stderr(io.StringIO()) as log:
                ec.sample_with_emotion_preference(
                    logits, hidden, None, weight, controller, 0,
                    state=ec.EmotionSamplingState(before), top_k=11,
                )
        record = json.loads(log.getvalue())
        self.assertEqual(record['rows'][0]['budget_disabled'], expected)

    def test_environment_flag_is_read_at_fresh_process_start(self):
        for setting in (None, '0', '1'):
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='')
            env.pop('GPT_SOVITS_STEERING_TRACE', None)
            if setting is not None:
                env['GPT_SOVITS_STEERING_TRACE'] = setting
            code = (
                "import sys; sys.path.insert(0, 'GPT_SoVITS'); "
                "from AR.models.emotion_control import STEERING_TRACE; "
                "print(STEERING_TRACE)"
            )
            result = subprocess.run(
                [sys.executable, '-B', '-c', code], cwd=ROOT, env=env,
                text=True, capture_output=True, check=True, timeout=30,
            )
            self.assertEqual(result.stdout.strip(), str(setting == '1'))

    def test_exception_fail_open_is_not_counted_as_budget_cutoff(self):
        from tools.diagnose_emotion_steering import summarize_trace
        controller = ec.LowRankEmotionPreference(8, 4, rank=4)
        records = self._trace_records(controller)
        with patch.object(ec, 'STEERING_TRACE', True):
            with patch.object(controller, 'preference_scores', side_effect=ValueError('test')):
                with contextlib.redirect_stderr(io.StringIO()) as log:
                    ec.sample_with_emotion_preference(
                        torch.zeros(2, 11), torch.zeros(2, 8), None,
                        torch.randn(11, 8), controller, 0, top_k=11, eos_id=10,
                    )
        failure = json.loads(log.getvalue())
        self.assertEqual(failure['eligible_row_steps'], 0)
        self.assertEqual(failure['budget_disabled_row_steps'], 0)
        report = summarize_trace([*records, failure], {
            'controller_sha256': 'c' * 64, 's1_sha256': 'a' * 64,
            's1_pairing': 'verified', 'emotion_order': ['a', 'b', 'c', 'd'],
        })
        self.assertEqual(report['exception_fail_open_steps'], 1)

    def test_report_never_overwrites_input_artifact(self):
        from tools.diagnose_emotion_steering import write_report
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'input.json'
            path.write_text('preserved', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'overwrite'):
                write_report({'test': 1}, path, [path])
            self.assertEqual(path.read_text(), 'preserved')

    def test_static_distribution_and_context_independence(self):
        from tools.diagnose_emotion_steering import diagnose
        torch.manual_seed(42)
        controller = ec.LowRankEmotionPreference(16, 4, rank=4)
        weight = torch.randn(31, 16)
        report = diagnose(controller, weight, ['happy', 'angry', 'disgusted', 'surprised'])
        self.assertEqual(len(report['emotions']), 4)
        self.assertEqual(len(report['emotions'][0]['strengths']), 4)
        self.assertTrue(report['context_independent'])
        self.assertEqual(len(report['correlations']), 4)
        self.assertLessEqual(report['emotions'][0]['strengths'][0]['delta_abs']['max'], 0.35 + 1e-6)
        controller.steering_mode = 'legacy_low_rank'
        with self.assertRaisesRegex(ValueError, 'hidden_direction'):
            diagnose(controller, weight, ['a', 'b', 'c', 'd'])

    def test_flag_off_and_on_preserve_sampling_and_rng(self):
        torch.manual_seed(123)
        controller = ec.LowRankEmotionPreference(8, 4, rank=4)
        logits, hidden, weight = torch.randn(2, 11), torch.randn(2, 8), torch.randn(11, 8)
        def run(enabled):
            with patch.object(ec, 'STEERING_TRACE', enabled, create=True):
                with contextlib.redirect_stderr(io.StringIO()) as log:
                    torch.manual_seed(912)
                    state = ec.EmotionSamplingState(torch.tensor([1.9, 0.0]))
                    sample, probs, state = ec.sample_with_emotion_preference(
                        logits.clone(), hidden, None, weight, controller, 1,
                        state=state, eos_id=10, top_k=5,
                    )
                    return sample, probs, state, torch.get_rng_state(), log.getvalue()
        off, on = run(False), run(True)
        # Pre-instrumentation sampler arithmetic, with the identical seed.
        torch.manual_seed(912)
        base = ec.logits_to_probs(logits=logits.clone(), previous_tokens=None, top_k=5)
        expected_state = ec.EmotionSamplingState(torch.tensor([1.9, 0.0]))
        scores = controller.preference_scores(hidden, weight, 1, vocab_size=11)
        expected_probs, expected_state = controller.tilt_probabilities(
            base, scores, 1.0, 10, expected_state)
        expected_sample = ec.multinomial_sample_one_no_sync(expected_probs)
        p0 = base.gather(1, expected_sample.long()).squeeze(1).clamp_min(1e-12)
        pe = expected_probs.gather(1, expected_sample.long()).squeeze(1).clamp_min(1e-12)
        expected_state.cumulative_log_ratio = (
            expected_state.cumulative_log_ratio + pe.log() - p0.log()
        )
        self.assertTrue(torch.equal(off[0], expected_sample))
        self.assertTrue(torch.equal(off[1], expected_probs))
        self.assertTrue(torch.equal(off[3], torch.get_rng_state()))
        self.assertTrue(torch.equal(
            off[2].cumulative_log_ratio, expected_state.cumulative_log_ratio,
        ))
        for index in (0, 1, 3):
            self.assertTrue(torch.equal(off[index], on[index]))
        self.assertTrue(torch.equal(off[2].cumulative_log_ratio, on[2].cumulative_log_ratio))
        self.assertEqual(off[4], '')
        record = json.loads(on[4].strip())
        self.assertEqual(record['budget_disabled_row_steps'], 1)
        self.assertEqual(record['eligible_row_steps'], 2)
        self.assertIn('cumulative_log_ratio', record)
        self.assertFalse(hasattr(off[2], '_steering_trace'))
        with patch.object(ec, 'STEERING_TRACE', True):
            on[2].index_select(torch.tensor([1]))
        self.assertEqual(on[2]._steering_trace['row_ids'], [1])
        self.assertEqual(len(on[2]._steering_trace['final']), 2)

    def test_cli_synthetic_checkpoints(self):
        from tools.diagnose_emotion_steering import main
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            s1 = folder / 's1.ckpt'
            torch.save({'weight': {'model.ar_predict_layer.weight': torch.randn(19, 8)}}, s1)
            controller = ec.LowRankEmotionPreference(8, 4, rank=4)
            path = folder / 'controller.pt'
            ec.save_emotion_controller_checkpoint(
                controller, str(path), emotion_names=['a', 'b', 'c', 'd'],
                extra={'s1_sha256': ec.sha256_file(str(s1))},
            )
            output = folder / 'report.json'
            with contextlib.redirect_stdout(io.StringIO()):
                main([
                    '--controller', str(path), '--s1-checkpoint', str(s1),
                    '--output', str(output),
                ])
            self.assertEqual(json.loads(output.read_text())['s1_pairing'], 'verified')

    def test_missing_or_mismatched_pairing_is_rejected(self):
        from tools.diagnose_emotion_steering import main
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            s1 = folder / 's1.ckpt'
            torch.save({'weight': {'model.ar_predict_layer.weight': torch.randn(19, 8)}}, s1)
            controller = ec.LowRankEmotionPreference(8, 4, rank=4)
            for extra in ({}, {'s1_sha256': '0' * 64}):
                with self.subTest(extra=extra):
                    path = folder / 'controller.pt'
                    output = folder / 'report.json'
                    ec.save_emotion_controller_checkpoint(
                        controller, str(path),
                        emotion_names=['a', 'b', 'c', 'd'], extra=extra,
                    )
                    with self.assertRaisesRegex(ValueError, 'S1|pairing'):
                        main(['--controller', str(path), '--s1-checkpoint', str(s1),
                              '--output', str(output)])
                    self.assertFalse(output.exists())

    def test_known_statistics_and_undefined_correlations(self):
        from tools.diagnose_emotion_steering import abs_stats, correlations
        values = torch.tensor([0.0, -0.1, 0.2, -0.4], dtype=torch.float64)
        stats = abs_stats(values)
        self.assertAlmostEqual(stats['mean'], 0.175)
        self.assertAlmostEqual(stats['p50'], 0.15)
        self.assertAlmostEqual(stats['p90'], 0.34)
        self.assertAlmostEqual(stats['p99'], 0.394)
        self.assertAlmostEqual(stats['max'], 0.4)
        matrix = correlations(torch.stack([values, -values, torch.ones_like(values)]))
        self.assertAlmostEqual(matrix[0][1], -1.0)
        self.assertIsNone(matrix[0][2])

    def test_diagnose_preserves_controller_dtype_and_training_state(self):
        from tools.diagnose_emotion_steering import diagnose
        controller = ec.LowRankEmotionPreference(8, 4, rank=4).double().train()
        before = {key: value.clone() for key, value in controller.state_dict().items()}
        diagnose(controller, torch.randn(19, 8), ['a', 'b', 'c', 'd'])
        self.assertTrue(controller.training)
        for key, value in controller.state_dict().items():
            self.assertEqual(value.dtype, before[key].dtype)
            self.assertTrue(torch.equal(value, before[key]))


if __name__ == '__main__':
    unittest.main()
