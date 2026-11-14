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


    def test_legacy_simplex_init_is_centered_and_zero_effect(self):
        torch.manual_seed(1234)
        controller = LowRankEmotionPreference(
            hidden_dim=16,
            num_emotions=4,
            rank=8,
            steering_mode="legacy_low_rank",
        )
        coeff = controller.emotion_coeff.weight.detach()

        torch.testing.assert_close(
            coeff.mean(dim=0),
            torch.zeros(8),
            rtol=0.0,
            atol=1e-6,
        )
        torch.testing.assert_close(
            coeff.norm(dim=1),
            torch.ones(4),
            rtol=0.0,
            atol=1e-6,
        )

        gram = coeff @ coeff.transpose(0, 1)
        expected = torch.full((4, 4), -1.0 / 3.0)
        expected.fill_diagonal_(1.0)
        torch.testing.assert_close(
            gram,
            expected,
            rtol=1e-5,
            atol=1e-5,
        )

        hidden = torch.randn(7, 16)
        output_weight = torch.randn(29, 16)
        for emotion_id in range(4):
            scores = controller.preference_scores(
                hidden,
                output_weight,
                emotion_id,
            )
            torch.testing.assert_close(
                scores,
                torch.zeros_like(scores),
                rtol=0.0,
                atol=0.0,
            )

    def test_legacy_simplex_init_gives_u_proj_first_step_gradient(self):
        torch.manual_seed(4321)
        controller = LowRankEmotionPreference(
            hidden_dim=16,
            num_emotions=4,
            rank=8,
            prefix_log_ratio_budget=2.0,
            steering_mode="legacy_low_rank",
        )
        time_steps = 17
        vocab_size = 31
        hidden = torch.randn(time_steps, 16)
        output_weight = torch.randn(vocab_size, 16)
        logits = torch.randn(time_steps, vocab_size)
        top_values, top_indices = torch.topk(logits, 8, dim=-1)
        base_probs = torch.zeros_like(logits)
        base_probs.scatter_(
            1,
            top_indices,
            torch.softmax(top_values, dim=-1),
        )
        target_semantic = top_indices[:, 0]
        valid_mask = torch.ones(time_steps, dtype=torch.bool)

        scores = vectorized_sequence_relative_scores(
            controller=controller,
            hidden=hidden,
            base_probs=base_probs,
            target_semantic=target_semantic,
            valid_mask=valid_mask,
            output_weight=output_weight,
            eos_id=vocab_size - 1,
        )
        self.assertIsNotNone(scores)
        torch.testing.assert_close(
            scores,
            torch.zeros_like(scores),
            rtol=0.0,
            atol=1e-6,
        )

        loss = F.cross_entropy(
            4.0 * scores.unsqueeze(0),
            torch.tensor([2], dtype=torch.long),
        )
        loss.backward()

        self.assertIsNotNone(controller.u_proj.grad)
        self.assertGreater(
            float(controller.u_proj.grad.norm()),
            1e-8,
        )
        self.assertIsNotNone(controller.emotion_coeff.weight.grad)
        torch.testing.assert_close(
            controller.emotion_coeff.weight.grad,
            torch.zeros_like(controller.emotion_coeff.weight.grad),
            rtol=0.0,
            atol=0.0,
        )

    def test_hidden_direction_default_trains_active_direction_parameters(self):
        torch.manual_seed(20261114)
        controller = LowRankEmotionPreference(
            hidden_dim=16,
            num_emotions=4,
            rank=8,
            prefix_log_ratio_budget=2.0,
        )
        self.assertEqual(
            controller.steering_mode,
            "hidden_direction",
        )

        time_steps = 17
        vocab_size = 31
        hidden = torch.randn(time_steps, 16)
        output_weight = torch.randn(vocab_size, 16)
        logits = torch.randn(time_steps, vocab_size)
        top_values, top_indices = torch.topk(logits, 8, dim=-1)
        base_probs = torch.zeros_like(logits)
        base_probs.scatter_(
            1,
            top_indices,
            torch.softmax(top_values, dim=-1),
        )

        scores = vectorized_sequence_relative_scores(
            controller=controller,
            hidden=hidden,
            base_probs=base_probs,
            target_semantic=top_indices[:, 0],
            valid_mask=torch.ones(time_steps, dtype=torch.bool),
            output_weight=output_weight,
            eos_id=vocab_size - 1,
        )
        self.assertIsNotNone(scores)
        self.assertGreater(
            float(scores.detach().abs().max()),
            1e-8,
        )

        loss = F.cross_entropy(
            4.0 * scores.unsqueeze(0),
            torch.tensor([2], dtype=torch.long),
        )
        loss.backward()

        self.assertIsNone(controller.u_proj.grad)
        self.assertIsNotNone(controller.v_proj.grad)
        self.assertGreater(
            float(controller.v_proj.grad.norm()),
            1e-8,
        )
        self.assertIsNotNone(
            controller.emotion_coeff.weight.grad
        )
        self.assertGreater(
            float(controller.emotion_coeff.weight.grad.norm()),
            1e-8,
        )


    def test_single_non_eos_support_has_zero_training_effect(self):
        torch.manual_seed(20261112)
        controller = LowRankEmotionPreference(
            hidden_dim=8,
            num_emotions=4,
            rank=4,
            max_log_odds_shift=0.35,
            prefix_log_ratio_budget=2.0,
        )
        with torch.no_grad():
            controller.u_proj.normal_(mean=0.0, std=0.2)

        time_steps = 5
        vocab_size = 7
        eos_id = vocab_size - 1
        target_token = 2
        hidden = torch.randn(time_steps, 8)
        output_weight = torch.randn(vocab_size, 8)
        base_probs = torch.zeros(time_steps, vocab_size)
        base_probs[:, target_token] = 0.7
        base_probs[:, eos_id] = 0.3
        targets = [target_token] * time_steps
        valid_mask = [True] * time_steps

        scores = vectorized_sequence_relative_scores(
            controller=controller,
            hidden=hidden,
            base_probs=base_probs,
            target_semantic=targets,
            valid_mask=valid_mask,
            output_weight=output_weight,
            eos_id=eos_id,
        )
        self.assertIsNotNone(scores)
        torch.testing.assert_close(
            scores,
            torch.zeros_like(scores),
            rtol=0.0,
            atol=1e-7,
        )

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
            name: (
                None
                if parameter.grad is None
                else parameter.grad.detach().clone()
            )
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
            expected = legacy_gradients[name]
            if expected is None:
                self.assertIsNone(
                    parameter.grad,
                    msg=f"unexpected gradient for {name}",
                )
                continue
            self.assertIsNotNone(
                parameter.grad,
                msg=f"missing gradient for {name}",
            )
            torch.testing.assert_close(
                parameter.grad,
                expected,
                rtol=2e-5,
                atol=2e-6,
            )


if __name__ == "__main__":
    unittest.main()
