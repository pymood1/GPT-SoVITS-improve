import math
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
GPT_ROOT = ROOT / "GPT_SoVITS"
if str(GPT_ROOT) not in sys.path:
    sys.path.insert(0, str(GPT_ROOT))

from AR.models.utils import get_batch_logps, make_reject_y
from AR.modules.activation import MultiheadAttention


class S1LowDataDPOTest(unittest.TestCase):
    def test_rejected_sequences_ignore_batch_padding_and_always_change(self):
        torch.manual_seed(20261115)
        padded = torch.tensor(
            [
                [1, 2, 3, 999, 999, 999],
                [4, 5, 6, 7, 999, 999],
            ],
            dtype=torch.long,
        )
        lengths = torch.tensor([3, 4], dtype=torch.long)

        rejected, rejected_lengths = make_reject_y(
            padded,
            lengths,
        )

        self.assertEqual(rejected.shape[0], 2)
        self.assertNotIn(999, rejected.tolist()[0])
        self.assertNotIn(999, rejected.tolist()[1])
        self.assertNotEqual(int(rejected_lengths[0]), 3)
        self.assertNotEqual(int(rejected_lengths[1]), 4)

    def test_length_normalized_logps_ignore_padding(self):
        vocab = 4
        target_logits = torch.zeros(2, 5, vocab)
        reject_logits = torch.zeros(2, 5, vocab)
        target_labels = torch.tensor(
            [
                [0, 1, 3, 2, 1],
                [3, 2, 1, 0, 3],
            ]
        )
        reject_labels = target_labels.flip(1)
        target_lens = torch.tensor([1, 3])
        reject_lens = torch.tensor([2, 4])

        chosen, rejected = get_batch_logps(
            target_logits,
            reject_logits,
            target_labels,
            reject_labels,
            target_lens=target_lens,
            reject_lens=reject_lens,
            average_log_prob=True,
        )

        expected = -math.log(vocab)
        self.assertTrue(
            torch.allclose(
                chosen,
                torch.full_like(chosen, expected),
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                rejected,
                torch.full_like(rejected, expected),
                atol=1e-6,
            )
        )

    def test_speaker_lora_reference_snapshot_stays_frozen(self):
        torch.manual_seed(20261115)
        attention = MultiheadAttention(
            embed_dim=8,
            num_heads=2,
            batch_first=True,
        )
        attention.enable_lora(
            target="v",
            rank=2,
            alpha=4.0,
        )

        with torch.no_grad():
            attention.lora_in_B.normal_(std=0.1)
            attention.lora_out_B.normal_(std=0.1)

        snapshot_in = attention.merged_in_proj_weight().detach().clone()
        snapshot_out = attention.merged_out_proj_weight().detach().clone()
        attention.snapshot_lora_reference()
        self.assertTrue(attention.has_lora_reference)

        with torch.no_grad():
            attention.lora_in_B.add_(0.25)
            attention.lora_out_B.sub_(0.15)

        current_in = attention.merged_in_proj_weight().detach().clone()
        current_out = attention.merged_out_proj_weight().detach().clone()
        self.assertFalse(torch.allclose(snapshot_in, current_in))
        self.assertFalse(torch.allclose(snapshot_out, current_out))

        attention.use_lora_reference(True)
        try:
            reference_in = (
                attention.merged_in_proj_weight().detach().clone()
            )
            reference_out = (
                attention.merged_out_proj_weight().detach().clone()
            )
        finally:
            attention.use_lora_reference(False)

        self.assertTrue(torch.allclose(snapshot_in, reference_in))
        self.assertTrue(torch.allclose(snapshot_out, reference_out))
        self.assertTrue(
            torch.allclose(
                current_in,
                attention.merged_in_proj_weight(),
            )
        )
        self.assertTrue(
            torch.allclose(
                current_out,
                attention.merged_out_proj_weight(),
            )
        )


if __name__ == "__main__":
    unittest.main()
