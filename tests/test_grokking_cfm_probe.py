"""CPU-only source-level CFM research smoke, without loading full TTS weights."""
import ast
import copy
import math
import os
import sys
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.cuda.amp import autocast
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "GPT_SoVITS"))
sys.path.insert(0, str(ROOT))
from tools.grokking import deterministic_holdout_indices, repeat_indices_to_minimum, gradfilter_ema


def load_selected(path, names, namespace):
    # Trainer entry points parse CLI config at import. Execute only the
    # actual functions/classes under test, not a copied implementation.
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
             and n.name in names]
    assert {n.name for n in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class TinyDataset(Dataset):
    def __init__(self, names):
        self.audiopaths_sid_text = [(name, [1]) for name in names]
        self.lengths = [100 + ord(name) for name in names]

    def __len__(self):
        return len(self.audiopaths_sid_text)

    def __getitem__(self, i):
        return torch.tensor([float(ord(self.audiopaths_sid_text[i][0][0])) / 100])


class TinyCFM(nn.Module):
    def __init__(self, in_channels, dit):
        super().__init__()
        self.estimator = dit
        self.criterion = nn.MSELoss()


class TinyEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 3)

    def forward(self, xt, prompt, lengths, t, text0, use_grad_ckpt):
        return self.linear((xt + prompt + text0).transpose(1, 2))


class RandomValidationModel(nn.Module):
    def forward(self, *batch, use_grad_ckpt):
        assert not self.training and not torch.is_grad_enabled()
        assert not use_grad_ckpt
        return batch[0].mean() + torch.rand(())


class CFMProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.helpers = load_selected(
            "GPT_SoVITS/s2_train_v3_lora.py", {"_split_cfm_dataset", "_validate_cfm"},
            dict(torch=torch, copy=copy, math=math, autocast=autocast,
                 deterministic_holdout_indices=deterministic_holdout_indices,
                 repeat_indices_to_minimum=repeat_indices_to_minimum),
        )
        cls.CFMV5 = load_selected(
            "GPT_SoVITS/module/models_v5.py", {"CFMV5"},
            dict(torch=torch, os=os, CFM=TinyCFM, V5_TEMPERATURE=0.875),
        )["CFMV5"]
        cls.cfg = dict(validation_fraction=0.2, min_validation_samples=2,
                       seed=1234, max_validation_batches=3)

    def test_cfm_split_is_rank_order_independent(self):
        a = TinyDataset(list("aabbcdef"))
        b = TinyDataset(list("fedcbbaa"))
        split = self.helpers["_split_cfm_dataset"]
        tr_a, va_a = split(a, self.cfg)
        tr_b, va_b = split(b, self.cfg)
        names = lambda d: [x[0] for x in d.audiopaths_sid_text]
        self.assertEqual(names(tr_a), names(tr_b))
        self.assertEqual(names(va_a), names(va_b))
        self.assertEqual(len(tr_a), 100)
        self.assertEqual(len(va_a), 2)
        self.assertTrue(set(names(tr_a)).isdisjoint(names(va_a)))
        self.assertEqual(len(a), 8)
        self.assertEqual(tr_a.lengths, [100 + ord(n) for n in names(tr_a)])

    @unittest.skipIf(torch.cuda.is_available(), "CPU-only smoke")
    def test_cfm_validation_reproducible_and_restores_rng_and_mode(self):
        dataset = TinyDataset(list("abcdef"))
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        model = RandomValidationModel()
        validate = self.helpers["_validate_cfm"]
        before = torch.random.get_rng_state().clone()
        losses_a, losses_b = [], []
        first = validate(model, loader, 0, self.cfg, losses_a)
        second = validate(model, loader, 0, self.cfg, losses_b)
        self.assertEqual(first, second)
        self.assertEqual(losses_a, losses_b)
        self.assertEqual(len(losses_a), self.cfg["max_validation_batches"])
        self.assertAlmostEqual(first, sum(losses_a) / len(losses_a))
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertTrue(model.training)

    def test_model_mix_effective_delta_and_invalid_pairs(self):
        from tools.quality_model_mix import mix_s1_state, scale_s2_lora
        a = torch.tensor([[1., 2., 3.], [2., 1., 4.]])
        b = torch.tensor([[3., 2.], [1., 1.]])
        weights = {"cfm.q.lora_A.default.weight": a, "cfm.q.lora_B.default.weight": b}
        for scale in (0., .25, .75, 1., 1.5):
            out = scale_s2_lora(weights, 2, scale)
            self.assertTrue(torch.allclose(out["cfm.q.lora_B.default.weight"] @ out["cfm.q.lora_A.default.weight"], scale * (b @ a)))
        self.assertTrue(torch.equal(weights["cfm.q.lora_B.default.weight"], b))
        with self.assertRaises(ValueError):
            scale_s2_lora({"cfm.q.lora_A.default.weight": a}, 2, .5)
        self.assertTrue(torch.equal(mix_s1_state({"w": a}, {"w": a + 2}, .5)["w"], a + 1))
        with self.assertRaises(ValueError):
            mix_s1_state({"w": a}, {"w": b}, .5)

    def test_v5_mix_export_keeps_version_header_and_frozen_state(self):
        from io import BytesIO
        from tempfile import TemporaryDirectory
        from tools.quality_model_mix import export_scaled_v5_checkpoint
        source_data = {"lora_rank": 2, "config": {"version": "v5dev"},
                       "weight": {"cfm.q.lora_A.default.weight": torch.eye(2),
                                  "cfm.q.lora_B.default.weight": torch.ones(2, 2),
                                  "quantizer.buffer": torch.tensor([1., 2.])}}
        with TemporaryDirectory() as folder:
            original, mixed = Path(folder) / "source.pth", Path(folder) / "mixed.pth"
            buffer = BytesIO()
            torch.save(source_data, buffer)
            original.write_bytes(b"07" + buffer.getvalue()[2:])
            before = original.read_bytes()
            export_scaled_v5_checkpoint(original, mixed, .75)
            self.assertEqual(original.read_bytes(), before)
            data = mixed.read_bytes()
            self.assertEqual(data[:2], b"07")
            result = torch.load(BytesIO(b"PK" + data[2:]), map_location="cpu", weights_only=True)
            self.assertEqual(result["lora_rank"], 2)
            self.assertTrue(torch.equal(result["weight"]["quantizer.buffer"], source_data["weight"]["quantizer.buffer"]))
            self.assertTrue(torch.equal(result["weight"]["cfm.q.lora_B.default.weight"], torch.full((2, 2), .75)))
            with self.assertRaises(ValueError):
                export_scaled_v5_checkpoint(original, original, .75)

    def test_soft_semantic_onehot_matches_projected_hard_decode(self):
        from module.soft_semantics import decode_soft_single_layer
        from types import SimpleNamespace
        book = torch.tensor([[1., 2.], [3., 4.], [6., 7.]])
        proj = nn.Linear(2, 3)
        layer = SimpleNamespace(codebook=book, project_out=proj)
        q = SimpleNamespace(vq=SimpleNamespace(layers=[layer]),
                            decode=lambda codes: proj(book[codes[0]]).transpose(1, 2).contiguous())
        codes = torch.tensor([[[0, 2], [1, 0]]])
        p = torch.nn.functional.one_hot(codes[0], 3).float()
        hard = q.decode(codes)
        self.assertTrue(torch.allclose(decode_soft_single_layer(q, codes, p, 1.), hard))
        uniform = torch.full_like(p, 1/3)
        self.assertTrue(torch.equal(decode_soft_single_layer(q, codes, uniform, 0.), hard))
        soft = proj(uniform @ book).transpose(1, 2)
        self.assertTrue(torch.allclose(decode_soft_single_layer(q, codes, uniform, .5), (hard + soft) / 2))
        with self.assertRaises(ValueError):
            decode_soft_single_layer(q, codes, p * .5)
        with self.assertRaises(ValueError):
            decode_soft_single_layer(q, codes, torch.cat([p, p[..., :1]], -1))

    def test_tts_pcm_full_scale_and_output_rate_pause(self):
        import numpy as np
        from types import SimpleNamespace
        from typing import List, Tuple
        tree = ast.parse((ROOT / "GPT_SoVITS/TTS_infer_pack/TTS.py").read_text(encoding="utf-8"))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TTS")
        fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "audio_postprocess")
        ns = dict(torch=torch, np=np, List=List, Tuple=Tuple)
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "<tts postprocess>", "exec"), ns)
        stub = SimpleNamespace(configs=SimpleNamespace(sampling_rate=32000, device="cpu"), precision=torch.float32)
        sr, pcm = ns["audio_postprocess"](stub, [[torch.tensor([1.0, -1.0, 0.5])]], 48000,
                                          split_bucket=False, fragment_interval=0.01)
        self.assertEqual(sr, 48000)
        self.assertEqual(len(pcm), 483)
        self.assertEqual(pcm[:3].tolist(), [32767, -32767, 16384])
        self.assertTrue((pcm[3:] == 0).all())
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.assertRaisesRegex(RuntimeError, "Nonfinite"):
                ns["audio_postprocess"](stub, [[torch.tensor([0.1, value])]], 48000,
                                        split_bucket=False, fragment_interval=0)

    def test_actual_v5_cfm_forward_and_grokfast_optimizer_step(self):
        torch.manual_seed(42)
        model = self.CFMV5(3, TinyEstimator())
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
        x = torch.randn(2, 3, 12)
        mu = torch.randn_like(x)
        lengths = torch.tensor([12, 10])
        prompt = torch.tensor([3, 2])
        state = {}
        losses = []
        for _ in range(32):
            optimizer.zero_grad(set_to_none=True)
            torch.manual_seed(1234)  # same flow target for this optimizer smoke
            loss = model(x, lengths, prompt, mu, False)
            loss.backward()
            gradfilter_ema(model, state)
            optimizer.step()
            losses.append(loss.item())
        self.assertTrue(all(math.isfinite(v) for v in losses))
        self.assertLess(losses[-1], losses[0] * 0.85)
        self.assertTrue(state)
        print("V5_CFM_CPU_SMOKE", losses[0], losses[-1])


if __name__ == "__main__":
    unittest.main()
