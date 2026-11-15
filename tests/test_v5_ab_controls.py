import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
from tools.v5_ab_controls import (
    FreshAcousticFactory, initial_noise_override, precision_context,
    parameter_profile, buffer_snapshot, changed_buffers,
)


class CFM(torch.nn.Module):
    def inference(self, mu, lengths, prompt, steps, temperature=.875):
        x = torch.randn(mu.shape, dtype=mu.dtype) * temperature
        return x + mu


class TinyS2(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([.123456789]))
        self.cfm = CFM()
        for key, value in dict(noise_temperature=.875, use_static_cache=True,
                               use_conditioner_cache=True, use_step_embedding=False,
                               cfg_drop_text=False, dbcache_spec=None).items():
            setattr(self.cfm, key, value)


class TinyVocoder(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([.234567891]))

    def remove_weight_norm(self):
        pass


class ControlsTests(unittest.TestCase):
    def test_fresh_instances_do_not_share_mutable_weights_and_fp32_reads_file(self):
        s2, voc = TinyS2(), TinyVocoder()
        raw = {k: v.detach().clone() for k, v in s2.state_dict().items()}
        original_float = raw["weight"].clone()
        s2.half()
        voc.half()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "GPT_SoVITS/pretrained_models/gsv-v5-pretrained/vocoder.pth"
            path.parent.mkdir(parents=True)
            torch.save(TinyVocoder().state_dict(), path)
            ns = dict(vq_model=s2, hifigan_model=voc, device="cpu", sovits_path_global="fixture.pth",
                      load_sovits_new=lambda p: {"weight": raw},
                      SynthesizerTrnV3=TinyS2, Generator=TinyVocoder,
                      hps=SimpleNamespace(data=SimpleNamespace(filter_length=4, hop_length=2, n_speakers=1),
                                          train=SimpleNamespace(segment_size=4), model={}))
            factory = FreshAcousticFactory(ns, root)
            first, _ = factory()
            first.weight.data.add_(1)
            second, _ = factory()
            full, _ = factory(fp32=True)
            self.assertTrue(torch.equal(second.weight, s2.weight))
            self.assertEqual(second.weight.dtype, torch.float16)
            self.assertEqual(full.weight.dtype, torch.float32)
            self.assertTrue(torch.equal(full.weight, original_float))
            self.assertFalse(torch.equal(full.weight, s2.weight.float()))
            factory.close()
        self.assertEqual(s2.weight.dtype, torch.float16)

    def test_initial_scaled_noise_is_exact_across_dtype_and_restored_on_error(self):
        cfm = CFM()
        mu = torch.zeros(4)
        fixed = torch.tensor([.125, -.375, .5, -.75], dtype=torch.float16)
        with self.assertRaisesRegex(RuntimeError, "fixture"):
            with initial_noise_override(cfm, fixed):
                output = cfm.inference(mu, None, None, 32)
                self.assertTrue(torch.equal(output, fixed.float()))
                raise RuntimeError("fixture")
        self.assertNotIn("inference", cfm.__dict__)

    def test_fp32_context_disables_cpu_amp_and_tf32_then_restores(self):
        previous = torch.backends.cuda.matmul.allow_tf32
        with torch.autocast("cpu", dtype=torch.bfloat16):
            self.assertTrue(torch.is_autocast_enabled("cpu"))
            with precision_context(True):
                self.assertFalse(torch.is_autocast_enabled("cpu"))
                self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
            self.assertTrue(torch.is_autocast_enabled("cpu"))
        self.assertEqual(torch.backends.cuda.matmul.allow_tf32, previous)

    def test_buffer_mutation_and_dtype_profile_are_observable(self):
        module = TinyS2().half()
        module.register_buffer("cache_value", torch.zeros(1))
        before = buffer_snapshot(module)
        module.cache_value.add_(1)
        self.assertEqual(changed_buffers(module, before), ["cache_value"])
        self.assertIn("torch.float16@cpu", parameter_profile(module))


if __name__ == "__main__":
    unittest.main()
