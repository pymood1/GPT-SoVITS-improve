"""CPU fixtures test source execution/measurement, not acoustic quality."""
import unittest
import ast
import tempfile
import sys
import types
from pathlib import Path
from unittest.mock import patch
import torch
from tools.v5_acoustic_ab import compile_webui_acoustic, compile_engine, compare_traces


class AcousticABTests(unittest.TestCase):
    def test_production_routes_short_and_multichunk_with_identical_rng(self):
        from tools import v5_acoustic_ab as ab
        root = Path(ab.__file__).resolve().parents[1]
        helper_source = (root / "GPT_SoVITS/module/models_v5.py").read_text(encoding="utf-8")
        tree = ast.parse(helper_source)
        helper_ns = {"torch": torch}
        nodes = [n for n in tree.body if isinstance(n, ast.Assign)
                 or isinstance(n, ast.FunctionDef) and n.name == "synthesize_v5_mel"]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "<real V5 helper>", "exec"), helper_ns)
        class CFM(torch.nn.Module):
            in_channels = 100
            noise_temperature = .875
            def __init__(self):
                super().__init__()
                self.calls = 0
                self.stateful = False
            def inference(self, mu, lengths, prompt, steps, **kwargs):
                x = torch.randn(mu.shape[0], 100, mu.shape[1], dtype=mu.dtype) * self.noise_temperature
                self.calls += 1
                return x + mu.mean(-1).unsqueeze(1) + (self.calls if self.stateful else 0)
        class Model(torch.nn.Module):
            version = "v5dev"
            def __init__(self):
                super().__init__()
                self.cfm = CFM()
            def decode_encp(self, tokens, phones, refer, ge=None, speed=1):
                features = tokens.reshape(1, 1, -1).float().repeat(1, 4, 2) / 1000
                return features, torch.ones(1, 4, 1)
        class Vocoder(torch.nn.Module):
            def forward(self, mel):
                return mel.mean(1, keepdim=True)
        raw = torch.ones(1, 32000)
        fake_audio = types.SimpleNamespace(load=lambda p: (raw.clone(), 32000),
                                           save=lambda *a, **kw: None)
        mel = lambda x, **kwargs: x.new_ones((1, 100, 100))
        ns = dict(torch=torch, torchaudio=fake_audio, model_version="v5dev",
                  V5_VERSIONS={"v5dev", "v5turbo"}, vq_model=Model().eval(),
                  hifigan_model=Vocoder().eval(), hifigan_version="v5", device="cpu",
                  dtype=torch.float32, is_half=False,
                  hps=types.SimpleNamespace(data=types.SimpleNamespace(
                      sampling_rate=32000, filter_length=1280, hop_length=320, win_length=1280)),
                  get_spepc=lambda *a: (torch.ones(1, 641, 100), raw.clone()),
                  mel_fn_v4=mel, mel_spectrogram_torch=mel,
                  spectrogram_torch=lambda *a, **kw: torch.ones(1, 641, 100),
                  norm_spec=lambda x: (x + 12) / 14 * 2 - 1,
                  denorm_spec=lambda x: (x + 1) / 2 * 14 - 12,
                  synthesize_v5_mel=helper_ns["synthesize_v5_mel"])
        inputs = dict(pred_semantic=torch.arange(400).reshape(1, 1, -1),
                      prompt=torch.arange(20).reshape(1, -1), phones1=[1, 2], phones2=[3, 4],
                      ref_wav_path="fixture.wav", speed=1., sample_steps=32, cfg_rate=1.3)
        source = (root / "GPT_SoVITS/TTS_infer_pack/TTS.py").read_text(encoding="utf-8")
        before = torch.random.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, torchaudio=fake_audio), \
                patch.object(ab, "official_sources", return_value=({"GPT_SoVITS/TTS_infer_pack/TTS.py": source}, {})), \
                patch.object(ab, "checkpoint_bindings", return_value={}):
            report = ab._run_sequence(ns, inputs, tmp)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        for case in report["cases"].values():
            self.assertIsNone(case["repeat_control"]["first_difference"])
            self.assertIsNone(case["ab"]["first_difference"])
        self.assertEqual(len(report["cases"]), 1)
        self.assertNotIn("forward", ns["hifigan_model"].__dict__)
        self.assertNotIn("inference", ns["vq_model"].cfm.__dict__)
        # A single deliberately changed frontend factor must be located at mel2,
        # rather than reported as RNG drift or hidden by the common helper.
        changed = source.replace("mel2 = norm_spec(mel2)", "mel2 = norm_spec(mel2) + 0.25")
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, torchaudio=fake_audio), \
                patch.object(ab, "official_sources", return_value=({"GPT_SoVITS/TTS_infer_pack/TTS.py": changed}, {})), \
                patch.object(ab, "checkpoint_bindings", return_value={}):
            report = ab._run_sequence(ns, inputs, tmp)
        for case in report["cases"].values():
            self.assertIsNone(case["repeat_control"]["first_difference"])
            self.assertEqual(case["ab"]["first_difference"], "mel2[1]")
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, torchaudio=fake_audio), \
                patch.object(ab, "official_sources", return_value=({"GPT_SoVITS/TTS_infer_pack/TTS.py": source}, {})), \
                patch.object(ab, "checkpoint_bindings", return_value={}), \
                patch.object(ab, "save_float_wav", side_effect=OSError("fixture write failure")):
            with self.assertRaisesRegex(OSError, "fixture write failure"):
                ab._run_sequence(ns, inputs, tmp)
            self.assertTrue(list(Path(tmp).glob("*/report.json")))
        self.assertNotIn("forward", ns["hifigan_model"].__dict__)
        self.assertNotIn("inference", ns["vq_model"].cfm.__dict__)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        # Full fresh/order/precision suite must retain fixed semantics and noise.
        ns["_ab_factory"] = lambda fp32=False: (Model().eval(), Vocoder().eval())
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, torchaudio=fake_audio), \
                patch.object(ab, "official_sources", return_value=({"GPT_SoVITS/TTS_infer_pack/TTS.py": source}, {})), \
                patch.object(ab, "checkpoint_bindings", return_value={}):
            suite = ab.run_from_webui(ns, inputs, tmp)
        for case in suite["comparisons"].values():
            for name, comparison in case.items():
                if isinstance(comparison, dict):
                    self.assertIsNone(comparison["first_difference"], name)
        self.assertEqual(len(suite["sequences"]), 5)
        def stateful_factory(fp32=False):
            model = Model().eval()
            model.cfm.stateful = True
            return model, Vocoder().eval()
        ns["_ab_factory"] = stateful_factory
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, torchaudio=fake_audio), \
                patch.object(ab, "official_sources", return_value=({"GPT_SoVITS/TTS_infer_pack/TTS.py": source}, {})), \
                patch.object(ab, "checkpoint_bindings", return_value={}):
            suite = ab.run_from_webui(ns, inputs, tmp)
        case = suite["comparisons"]["full"]
        self.assertIsNotNone(case["forward_A_repeat"]["first_difference"])
        self.assertIsNotNone(case["B_order_effect"]["first_difference"])
        self.assertIsNone(case["cold_A_B"]["first_difference"])

    def test_nonfinite_difference_is_json_safe(self):
        import json
        result = compare_traces({"mel2": [torch.tensor([float("nan")])]},
                                {"mel2": [torch.ones(1)]})
        self.assertEqual(result["first_difference"], "mel2[0]")
        json.dumps(result, allow_nan=False)

    def test_raw_float_wav_preserves_out_of_range_samples(self):
        from tools.v5_acoustic_ab import save_float_wav
        from scipy.io.wavfile import read
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "raw.wav"
            save_float_wav(path, torch.tensor([-2., 0., 2.]))
            sr, values = read(path)
        self.assertEqual(sr, 48000)
        self.assertEqual(values.tolist(), [-2., 0., 2.])

    def test_real_branch_selection_and_capture(self):
        source = '''
def get_tts_wav():
    for item in prepared_texts:
        if model_version not in v3v4set:
            raise AssertionError("legacy branch")
        else:
            fea_ref, ge = model.decode_encp()
            mel2 = fea_ref + 1
            cfm_res = mel2 * 2
            audio = cfm_res[0]
'''
        class Model:
            def decode_encp(self):
                return torch.ones(1, 2, 3), torch.zeros(1)
        trace = {}
        fn = compile_webui_acoustic(source, {"model": Model()}, trace)
        self.assertTrue(torch.equal(fn(), torch.full((2, 3), 4.0)))
        self.assertEqual(trace["fea_ref"][0].shape, (1, 2, 3))

    def test_trace_detects_first_value_and_dtype_difference(self):
        a = {"fea_ref": [torch.ones(2)], "mel2": [torch.ones(2)]}
        b = {"fea_ref": [torch.ones(2)], "mel2": [torch.zeros(2)]}
        result = compare_traces(a, b)
        self.assertEqual(result["first_difference"], "mel2[0]")
        b["mel2"] = [torch.ones(2, dtype=torch.float64)]
        self.assertEqual(compare_traces(a, b)["first_difference"], "mel2[0]")

    def test_capture_is_snapshot_and_nonfinite_is_visible(self):
        from tools.v5_acoustic_ab import record, tensor_stats
        trace = {}
        x = torch.ones(2)
        record(trace, "ge", x)
        x.zero_()
        self.assertEqual(trace["ge"][0].sum(), 2)
        self.assertEqual(tensor_stats(torch.tensor([float("nan"), 1.0]))["nonfinite"], 1)

    def test_run_sequence_uses_pinned_ref_spec_without_fork_only_keyword(self):
        from tools import v5_acoustic_ab as ab
        source = Path(ab.__file__).read_text(encoding="utf-8")
        self.assertIn('methods["_get_ref_spec"](engine, base["ref_wav_path"])', source)
        self.assertNotIn(
            'methods["_get_ref_spec"](engine, base["ref_wav_path"], cache_raw=True)',
            source,
        )

    def test_official_method_is_executed_not_reimplemented(self):
        source = '''
class TTS:
    def using_vocoder_synthesis(self, semantic_tokens, phones, speed=1., sample_steps=None, cfg_rate=None):
        fea_todo = semantic_tokens + phones
        return fea_todo * self.factor
'''
        trace = {}
        method = compile_engine(source, {}, trace)["using_vocoder_synthesis"]
        from types import SimpleNamespace
        result = method(SimpleNamespace(factor=3), torch.ones(2), torch.ones(2))
        self.assertTrue(torch.equal(result, torch.full((2,), 6.)))
        self.assertIn("fea_todo", trace)


if __name__ == "__main__":
    unittest.main()
