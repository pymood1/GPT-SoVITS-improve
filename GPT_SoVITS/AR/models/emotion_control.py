from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

from AR.models.utils import logits_to_probs, multinomial_sample_one_no_sync


@dataclass
class EmotionSamplingState:
    """Per-generation state for conservative autoregressive steering."""

    cumulative_log_ratio: Optional[torch.Tensor] = None

    def ensure_batch(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> None:
        if self.cumulative_log_ratio is None or self.cumulative_log_ratio.numel() != batch_size:
            self.cumulative_log_ratio = torch.zeros(batch_size, device=device, dtype=dtype)
        else:
            self.cumulative_log_ratio = self.cumulative_log_ratio.to(device=device, dtype=dtype)

    def index_select(self, index: torch.Tensor) -> "EmotionSamplingState":
        if self.cumulative_log_ratio is not None:
            self.cumulative_log_ratio = torch.index_select(self.cumulative_log_ratio, 0, index)
        return self


class LowRankEmotionPreference(nn.Module):
    """Tiny S1 readout-space preference head.

    GPT-SoVITS stays frozen. This module only produces a bounded preference
    score which is applied after the original sampler has built its supported
    probability distribution.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_emotions: int,
        rank: int = 8,
        max_log_odds_shift: float = 0.35,
        prefix_log_ratio_budget: float = 2.0,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or num_emotions <= 0 or rank <= 0:
            raise ValueError("hidden_dim, num_emotions and rank must be positive")
        if max_log_odds_shift < 0 or prefix_log_ratio_budget < 0:
            raise ValueError("steering limits must be non-negative")

        self.hidden_dim = int(hidden_dim)
        self.num_emotions = int(num_emotions)
        self.rank = int(rank)
        self.max_log_odds_shift = float(max_log_odds_shift)
        self.prefix_log_ratio_budget = float(prefix_log_ratio_budget)

        # Zero-effect initialization: attaching an untrained controller does not
        # change GPT-SoVITS output probabilities.
        self.v_proj = nn.Parameter(torch.empty(self.hidden_dim, self.rank))
        self.u_proj = nn.Parameter(torch.zeros(self.hidden_dim, self.rank))
        self.emotion_coeff = nn.Embedding(self.num_emotions, self.rank)
        nn.init.normal_(self.v_proj, mean=0.0, std=self.hidden_dim ** -0.5)
        nn.init.normal_(self.emotion_coeff.weight, mean=1.0, std=0.01)

    def preference_scores(
        self,
        hidden_state: torch.Tensor,
        output_weight: torch.Tensor,
        emotion_id: Union[int, torch.Tensor],
        vocab_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Project a low-rank virtual hidden delta through the pretrained readout."""
        if hidden_state.ndim != 2:
            raise ValueError(f"hidden_state must be [B,H], got {tuple(hidden_state.shape)}")
        if hidden_state.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"hidden dim mismatch: controller={self.hidden_dim}, hidden={hidden_state.shape[-1]}"
            )

        batch_size = hidden_state.shape[0]
        if isinstance(emotion_id, int):
            emotion_id = torch.full(
                (batch_size,), emotion_id, dtype=torch.long, device=hidden_state.device
            )
        else:
            emotion_id = emotion_id.to(device=hidden_state.device, dtype=torch.long).reshape(-1)
            if emotion_id.numel() == 1 and batch_size != 1:
                emotion_id = emotion_id.expand(batch_size)
            if emotion_id.numel() != batch_size:
                raise ValueError("emotion_id must be scalar or have one value per batch item")

        if torch.any(emotion_id < 0) or torch.any(emotion_id >= self.num_emotions):
            raise ValueError("emotion_id out of range")

        normalized = F.layer_norm(hidden_state, (self.hidden_dim,))
        low_rank = normalized @ self.v_proj
        low_rank = low_rank * self.emotion_coeff(emotion_id)
        delta_hidden = low_rank @ self.u_proj.transpose(0, 1)

        if vocab_size is None:
            vocab_size = output_weight.shape[0]
        output_weight = output_weight[:vocab_size]
        return delta_hidden @ output_weight.transpose(0, 1)

    def tilt_probabilities(
        self,
        base_probs: torch.Tensor,
        preference_scores: torch.Tensor,
        strength: float,
        eos_id: Optional[int],
        state: Optional[EmotionSamplingState],
    ) -> Tuple[torch.Tensor, EmotionSamplingState]:
        """Apply a support-preserving bounded tilt while keeping EOS mass unchanged."""
        if state is None:
            state = EmotionSamplingState()
        state.ensure_batch(base_probs.shape[0], base_probs.device, base_probs.dtype)

        if strength == 0.0 or self.max_log_odds_shift == 0.0:
            return base_probs, state
        if not torch.isfinite(base_probs).all() or not torch.isfinite(preference_scores).all():
            return base_probs, state

        support = base_probs > 0
        support_count = support.sum(dim=-1)
        if torch.all(support_count <= 1):
            return base_probs, state

        # For d_i in [-kappa, kappa], |log(p_E/p_0)| <= 2*kappa.
        # If the remaining prefix budget cannot absorb one worst-case step,
        # steering fails open to the original distribution for that row.
        if self.prefix_log_ratio_budget > 0:
            remaining = self.prefix_log_ratio_budget - state.cumulative_log_ratio.abs()
            active_rows = remaining >= (2.0 * self.max_log_odds_shift)
        else:
            active_rows = torch.ones(
                base_probs.shape[0], dtype=torch.bool, device=base_probs.device
            )

        if not torch.any(active_rows):
            return base_probs, state

        delta = torch.clamp(
            float(strength) * preference_scores,
            min=-self.max_log_odds_shift,
            max=self.max_log_odds_shift,
        )
        delta = torch.where(support, delta, torch.zeros_like(delta))

        result = base_probs.clone()
        for row in torch.where(active_rows)[0].tolist():
            p0 = base_probs[row]
            row_support = support[row]

            eos_present = (
                eos_id is not None
                and 0 <= eos_id < p0.numel()
                and bool(row_support[eos_id].item())
            )
            eos_mass = p0[eos_id] if eos_present else p0.new_zeros(())

            non_eos_support = row_support.clone()
            if eos_present:
                non_eos_support[eos_id] = False
            if int(non_eos_support.sum().item()) == 0:
                continue

            weighted = torch.zeros_like(p0)
            weighted[non_eos_support] = (
                p0[non_eos_support] * torch.exp(delta[row, non_eos_support])
            )
            denom = weighted[non_eos_support].sum()
            if not torch.isfinite(denom) or denom <= 0:
                continue

            candidate = torch.zeros_like(p0)
            candidate[non_eos_support] = (
                (1.0 - eos_mass) * weighted[non_eos_support] / denom
            )
            if eos_present:
                candidate[eos_id] = eos_mass

            candidate = torch.where(row_support, candidate, torch.zeros_like(candidate))
            total = candidate.sum()
            if not torch.isfinite(total) or total <= 0:
                continue
            # Do not renormalize: the formula above already sums to 1 up to
            # floating-point error, and renormalizing would perturb EOS mass.
            if torch.abs(total - 1.0) > 1e-4:
                continue
            result[row] = candidate

        return result, state


def sample_with_emotion_preference(
    logits: torch.Tensor,
    hidden_state: torch.Tensor,
    previous_tokens: Optional[torch.Tensor],
    output_weight: torch.Tensor,
    controller: LowRankEmotionPreference,
    emotion_id: Union[int, torch.Tensor],
    strength: float = 1.0,
    eos_id: Optional[int] = None,
    state: Optional[EmotionSamplingState] = None,
    **sampling_kwargs,
) -> Tuple[torch.Tensor, torch.Tensor, EmotionSamplingState]:
    """Reuse GPT-SoVITS filtering, then tilt only inside its surviving support."""
    base_probs = logits_to_probs(
        logits=logits,
        previous_tokens=previous_tokens,
        **sampling_kwargs,
    )

    try:
        scores = controller.preference_scores(
            hidden_state=hidden_state,
            output_weight=output_weight,
            emotion_id=emotion_id,
            vocab_size=base_probs.shape[-1],
        )
        probs, state = controller.tilt_probabilities(
            base_probs=base_probs,
            preference_scores=scores,
            strength=strength,
            eos_id=eos_id,
            state=state,
        )
    except (RuntimeError, ValueError, FloatingPointError):
        # Fail open: controller failure must not break ordinary GPT-SoVITS.
        probs = base_probs
        if state is None:
            state = EmotionSamplingState()
        state.ensure_batch(base_probs.shape[0], base_probs.device, base_probs.dtype)

    idx_next = multinomial_sample_one_no_sync(probs)

    chosen = idx_next.long()
    p0 = torch.gather(base_probs, dim=1, index=chosen).squeeze(1).clamp_min(1e-12)
    pe = torch.gather(probs, dim=1, index=chosen).squeeze(1).clamp_min(1e-12)
    state.cumulative_log_ratio = (
        state.cumulative_log_ratio + torch.log(pe) - torch.log(p0)
    )

    return idx_next, probs, state
