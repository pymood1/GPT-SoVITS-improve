import sys
from pathlib import Path
import unittest

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "GPT_SoVITS"))

from AR.models.emotion_control import (  # noqa: E402
    LowRankEmotionPreference,
    vectorized_sequence_relative_score,
    vectorized_sequence_relative_scores,
)


class EmotionTrainingVectorizationTests(unittest.TestCase):
    def _make_case(self):
        torch.manual_seed(20261111)
        time_steps = 19
        hidden_dim = 16
        vocab_size = 29
        num_emotions = 4

        controller = LowRankEmotionPreference(
            hidden_dim=hidden_dim,
            num_emotions=num_emotions,
            rank=5,
            max_log_odds_shift=0.35,
            prefix_log_ratio_budget=0.90,
        )
        with torch.no_grad():
            controller.u_proj.normal_(mean=0.0, std=0.10)

        hidden = torch.randn(time_steps, hidden_dim)
        output_weight = torch.randn(vocab_size, hidden_dim)
        logits = torch.randn(time_steps, vocab_size)
        top_values, top_indices = torch.topk(logits, 7, dim=-1)
        base_probs = torch.zeros_like(logits)
        base_probs.scatter_(
            1,
            top_indices,
            torch.softmax(top_values, dim=-1),
        )
        target_semantic = top_indices[:, 0].tolist()
        valid_mask = [
            index % 5 != 3
            for index in range(time_steps)
        ]
        return (
            controller,
            hidden,
            base_probs,
            target_semantic,
            valid_mask,
            output_weight,
            vocab_size - 1,
        )

    def _legacy_scores(
        self,
        controller,
        hidden,
        base_probs,
        target_semantic,
        valid_mask,
        output_weight,
        eos_id,
    ):
        values = []
        for emotion_id in range(controller.num_emotions):
            score = vectorized_sequence_relative_score(
                controller=controller,
                hidden=hidden,
                base_probs=base_probs,
                target_semantic=target_semantic,
                valid_mask=valid_mask,
                output_weight=output_weight,
                emotion_id=emotion_id,
                eos_id=eos_id,
            )
            self.assertIsNotNone(score)
            values.append(score)
        return torch.stack(values)

    def test_all_emotion_scores_match_legacy_path(self):
        case = self._make_case()
        legacy = self._legacy_scores(*case)
        vectorized = vectorized_sequence_relative_scores(
            controller=case[0],
            hidden=case[1],
            base_probs=case[2],
            target_semantic=case[3],
            valid_mask=case[4],
            output_weight=case[5],
            eos_id=case[6],
        )
        self.assertIsNotNone(vectorized)
        torch.testing.assert_close(
            vectorized,
            legacy,
            rtol=1e-5,
            atol=1e-6,
        )

    def test_all_emotion_gradients_match_legacy_path(self):
        case = self._make_case()
        controller = case[0]

        legacy = self._legacy_scores(*case)
        target = torch.tensor([2], dtype=torch.long)
        legacy_loss = F.cross_entropy(
            4.0 * legacy.unsqueeze(0),
            target,
        )
        legacy_loss.backward()
        legacy_gradients = {
            name: parameter.grad.detach().clone()
            for name, parameter in controller.named_parameters()
        }

        controller.zero_grad(set_to_none=True)
        vectorized = vectorized_sequence_relative_scores(
            controller=case[0],
            hidden=case[1],
            base_probs=case[2],
            target_semantic=case[3],
            valid_mask=case[4],
            output_weight=case[5],
            eos_id=case[6],
        )
        self.assertIsNotNone(vectorized)
        vectorized_loss = F.cross_entropy(
            4.0 * vectorized.unsqueeze(0),
            target,
        )
        vectorized_loss.backward()

        for name, parameter in controller.named_parameters():
            torch.testing.assert_close(
                parameter.grad,
                legacy_gradients[name],
                rtol=2e-5,
                atol=2e-6,
            )


if __name__ == "__main__":
    unittest.main()
