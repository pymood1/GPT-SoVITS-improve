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
        first = validate(model, loader, 0, self.cfg)
        second = validate(model, loader, 0, self.cfg)
        self.assertEqual(first, second)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertTrue(model.training)

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
