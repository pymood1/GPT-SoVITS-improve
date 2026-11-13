from dataclasses import dataclass
import hashlib
from typing import Optional, Sequence, Tuple, Union

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
        self._init_emotion_coeff_simplex()

    def _init_emotion_coeff_simplex(self) -> None:
        """Break class symmetry while preserving exact zero-effect output.

        u_proj starts at zero, so controller output remains exactly zero for
        every emotion. The class coefficients themselves are initialized as a
        centered regular simplex instead of nearly-identical all-ones vectors.
        This gives the first u_proj update an O(1) class-discriminative signal.
        """
        k = self.num_emotions
        r = self.rank

        if k == 1:
            with torch.no_grad():
                self.emotion_coeff.weight.fill_(1.0)
            return

        if r >= k - 1:
            centered = torch.eye(k) - torch.full((k, k), 1.0 / k)
            # Build an orthonormal basis for the sum-zero subspace, then
            # randomly rotate it into the full rank space. Pairwise geometry
            # stays that of a regular simplex while all rank dimensions can
            # participate from the first update.
            basis, _ = torch.linalg.qr(centered[:, :-1], mode="reduced")
            coords = centered @ basis
            rotation, _ = torch.linalg.qr(
                torch.randn(r, k - 1),
                mode="reduced",
            )
            coeff = coords @ rotation.transpose(0, 1)
        else:
            # Exact K-simplex geometry needs rank >= K-1. Fall back to
            # centered, normalized random directions rather than restoring
            # the old near-identical initialization.
            coeff = torch.randn(k, r)
            coeff = coeff - coeff.mean(dim=0, keepdim=True)

        coeff = coeff / coeff.norm(
            dim=1,
            keepdim=True,
        ).clamp_min(1e-12)
        with torch.no_grad():
            self.emotion_coeff.weight.copy_(coeff)

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

    def hidden_emotion_logits(
        self,
        hidden_state: torch.Tensor,
    ) -> torch.Tensor:
        """Classify emotion directly from frozen S1 hidden states.

        The diagnostic probe showed that mean-pooled S1 hidden states contain
        a substantially stronger emotion signal than semantic likelihood
        residuals. Reuse the controller's existing low-rank parameters so the
        primary supervision trains the same basis used by inference, without
        adding a separate classifier head or changing checkpoint format.

        Both V and U projections participate. U starts at zero to preserve the
        controller's exact zero-effect inference initialization, while the V
        branch provides a non-degenerate class signal on the first step.
        """
        if hidden_state.ndim != 2:
            raise ValueError(
                f"hidden_state must be [T,H], got {tuple(hidden_state.shape)}"
            )
        if hidden_state.shape[-1] != self.hidden_dim:
            raise ValueError(
                "hidden dim mismatch: "
                f"controller={self.hidden_dim}, "
                f"hidden={hidden_state.shape[-1]}"
            )
        if hidden_state.shape[0] == 0:
            raise ValueError("hidden_state sequence must be non-empty")

        normalized = F.layer_norm(
            hidden_state,
            (self.hidden_dim,),
        )
        pooled = normalized.mean(dim=0)
        latent_v = pooled @ self.v_proj
        latent_u = pooled @ self.u_proj
        shared_latent = 0.5 * (latent_v + latent_u)
        return shared_latent @ self.emotion_coeff.weight.transpose(0, 1)

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
            if int(non_eos_support.sum().item()) <= 1:
                # With EOS mass fixed, one surviving non-EOS token leaves no
                # probability mass that the controller can redistribute.
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




def vectorized_sequence_relative_scores(
    controller: LowRankEmotionPreference,
    hidden: torch.Tensor,
    base_probs: torch.Tensor,
    target_semantic: Sequence[int] | torch.Tensor,
    valid_mask: Sequence[bool] | torch.Tensor,
    output_weight: torch.Tensor,
    eos_id: int,
) -> Optional[torch.Tensor]:
    """Score every emotion in one GPU pass without changing the objective.

    This is algebraically equivalent to evaluating
    ``vectorized_sequence_relative_score`` once per emotion, but reuses the
    shared layer-normalization, low-rank projection, support masks, and
    readout basis. The prefix budget is evaluated on-device to avoid the
    per-emotion CUDA -> CPU synchronization in the legacy scorer.
    """
    if hidden.ndim != 2 or base_probs.ndim != 2:
        raise ValueError("hidden/base_probs must be rank-2 tensors")
    if hidden.shape[0] != base_probs.shape[0]:
        raise ValueError("hidden/base_probs sequence lengths differ")
    if len(target_semantic) != hidden.shape[0]:
        raise ValueError("target semantic length does not match hidden")
    if len(valid_mask) != hidden.shape[0]:
        raise ValueError("valid mask length does not match hidden")

    vocab_size = base_probs.shape[-1]
    normalized = F.layer_norm(hidden, (controller.hidden_dim,))
    low_rank = normalized @ controller.v_proj

    # ((hV) * e) U^T W^T == ((hV) * e) (WU)^T.
    # WU is shared by every emotion, so compute it only once per sample.
    readout_basis = (
        output_weight[:vocab_size] @ controller.u_proj
    )
    conditioned = (
        low_rank.unsqueeze(1)
        * controller.emotion_coeff.weight.unsqueeze(0)
    )
    preference_scores = torch.matmul(
        conditioned,
        readout_basis.transpose(0, 1),
    )

    support = base_probs > 0
    support_count = support.sum(dim=-1)
    finite_rows = (
        torch.isfinite(base_probs).all(dim=-1).unsqueeze(1)
        & torch.isfinite(preference_scores).all(dim=-1)
    )

    if 0 <= eos_id < vocab_size:
        eos_present = support[:, eos_id]
        eos_mass = base_probs[:, eos_id]
    else:
        eos_present = torch.zeros(
            base_probs.shape[0],
            dtype=torch.bool,
            device=base_probs.device,
        )
        eos_mass = torch.zeros(
            base_probs.shape[0],
            dtype=base_probs.dtype,
            device=base_probs.device,
        )

    non_eos_support = support.clone()
    if 0 <= eos_id < vocab_size:
        non_eos_support[:, eos_id] = False

    delta = torch.clamp(
        preference_scores,
        min=-controller.max_log_odds_shift,
        max=controller.max_log_odds_shift,
    )
    delta = torch.where(
        support.unsqueeze(1),
        delta,
        torch.zeros_like(delta),
    )

    weighted = torch.where(
        non_eos_support.unsqueeze(1),
        base_probs.unsqueeze(1) * torch.exp(delta),
        torch.zeros_like(delta),
    )
    denom = weighted.sum(dim=-1)

    structurally_active = (
        finite_rows
        & (support_count > 1).unsqueeze(1)
        & (non_eos_support.sum(dim=-1) > 1).unsqueeze(1)
        & torch.isfinite(denom)
        & (denom > 0)
    )

    target_ids = torch.as_tensor(
        target_semantic,
        dtype=torch.long,
        device=base_probs.device,
    ).reshape(-1)
    valid_tensor = torch.as_tensor(
        valid_mask,
        dtype=torch.bool,
        device=base_probs.device,
    ).reshape(-1)
    if target_ids.numel() == 0 or valid_tensor.numel() == 0:
        return None

    target_base = base_probs.gather(
        1,
        target_ids.unsqueeze(1),
    ).squeeze(1)
    target_index = target_ids.view(-1, 1, 1).expand(
        -1,
        controller.num_emotions,
        1,
    )
    target_delta = delta.gather(
        2,
        target_index,
    ).squeeze(2)

    target_is_eos = (
        (target_ids == eos_id) & eos_present
    ).unsqueeze(1)
    non_eos_target_prob = (
        (1.0 - eos_mass).unsqueeze(1)
        * target_base.unsqueeze(1)
        * torch.exp(target_delta)
        / denom.clamp_min(1e-30)
    )
    steered_target = torch.where(
        target_is_eos,
        eos_mass.unsqueeze(1),
        non_eos_target_prob,
    )
    steered_target = torch.where(
        structurally_active,
        steered_target,
        target_base.unsqueeze(1),
    )

    raw_ratio = (
        torch.log(steered_target.clamp_min(1e-12))
        - torch.log(target_base.clamp_min(1e-12)).unsqueeze(1)
    )

    valid_2d = valid_tensor.unsqueeze(1).expand_as(raw_ratio)
    if controller.prefix_log_ratio_budget > 0:
        detached_valid_ratios = torch.where(
            valid_2d,
            raw_ratio.detach().to(dtype=torch.float64),
            torch.zeros_like(raw_ratio, dtype=torch.float64),
        )
        prefix_before = (
            torch.cumsum(detached_valid_ratios, dim=0)
            - detached_valid_ratios
        )
        threshold = (
            controller.prefix_log_ratio_budget
            - 2.0 * controller.max_log_odds_shift
        )
        can_continue = prefix_before.abs() <= threshold
        gate = torch.where(
            valid_2d,
            can_continue,
            torch.ones_like(can_continue),
        )
        active_prefix = (
            torch.cumprod(gate.to(dtype=torch.float32), dim=0) > 0.5
        )
        budget_mask = (
            active_prefix & valid_2d
        ).to(dtype=raw_ratio.dtype)
    else:
        budget_mask = valid_2d.to(dtype=raw_ratio.dtype)

    valid_count = valid_tensor.sum().clamp_min(1).to(
        dtype=raw_ratio.dtype
    )
    return (raw_ratio * budget_mask).sum(dim=0) / valid_count

def vectorized_sequence_relative_score(
    controller: LowRankEmotionPreference,
    hidden: torch.Tensor,
    base_probs: torch.Tensor,
    target_semantic: Sequence[int],
    valid_mask: Sequence[bool],
    output_weight: torch.Tensor,
    emotion_id: int,
    eos_id: int,
) -> Optional[torch.Tensor]:
    """Match sequential tilt scoring while vectorizing heavy GPU math."""
    if hidden.ndim != 2 or base_probs.ndim != 2:
        raise ValueError("hidden/base_probs must be rank-2 tensors")
    if hidden.shape[0] != base_probs.shape[0]:
        raise ValueError("hidden/base_probs sequence lengths differ")
    if len(target_semantic) != hidden.shape[0]:
        raise ValueError("target semantic length does not match hidden")
    if len(valid_mask) != hidden.shape[0]:
        raise ValueError("valid mask length does not match hidden")

    preference_scores = controller.preference_scores(
        hidden_state=hidden,
        output_weight=output_weight,
        emotion_id=emotion_id,
        vocab_size=base_probs.shape[-1],
    )

    support = base_probs > 0
    support_count = support.sum(dim=-1)
    finite_rows = (
        torch.isfinite(base_probs).all(dim=-1)
        & torch.isfinite(preference_scores).all(dim=-1)
    )

    if 0 <= eos_id < base_probs.shape[-1]:
        eos_present = support[:, eos_id]
        eos_mass = base_probs[:, eos_id]
    else:
        eos_present = torch.zeros(
            base_probs.shape[0],
            dtype=torch.bool,
            device=base_probs.device,
        )
        eos_mass = torch.zeros(
            base_probs.shape[0],
            dtype=base_probs.dtype,
            device=base_probs.device,
        )

    non_eos_support = support.clone()
    if 0 <= eos_id < base_probs.shape[-1]:
        non_eos_support[:, eos_id] = False

    delta = torch.clamp(
        preference_scores,
        min=-controller.max_log_odds_shift,
        max=controller.max_log_odds_shift,
    )
    delta = torch.where(
        support,
        delta,
        torch.zeros_like(delta),
    )

    weighted = torch.where(
        non_eos_support,
        base_probs * torch.exp(delta),
        torch.zeros_like(base_probs),
    )
    denom = weighted.sum(dim=-1)

    structurally_active = (
        finite_rows
        & (support_count > 1)
        & (non_eos_support.sum(dim=-1) > 1)
        & torch.isfinite(denom)
        & (denom > 0)
    )

    target_ids = torch.tensor(
        target_semantic,
        dtype=torch.long,
        device=base_probs.device,
    )
    target_base = base_probs.gather(
        1,
        target_ids.unsqueeze(1),
    ).squeeze(1)
    target_delta = delta.gather(
        1,
        target_ids.unsqueeze(1),
    ).squeeze(1)

    target_is_eos = (target_ids == eos_id) & eos_present
    non_eos_target_prob = (
        (1.0 - eos_mass)
        * target_base
        * torch.exp(target_delta)
        / denom.clamp_min(1e-30)
    )
    steered_target = torch.where(
        target_is_eos,
        eos_mass,
        non_eos_target_prob,
    )
    steered_target = torch.where(
        structurally_active,
        steered_target,
        target_base,
    )

    raw_ratio = (
        torch.log(steered_target.clamp_min(1e-12))
        - torch.log(target_base.clamp_min(1e-12))
    )

    valid_tensor = torch.tensor(
        valid_mask,
        dtype=torch.bool,
        device=base_probs.device,
    )
    if not torch.any(valid_tensor):
        return None

    budget_mask = torch.zeros(
        len(target_semantic),
        dtype=raw_ratio.dtype,
        device=base_probs.device,
    )
    cumulative = 0.0
    detached_ratios = raw_ratio.detach().cpu().tolist()

    for index, is_valid in enumerate(valid_mask):
        if not is_valid:
            continue

        if controller.prefix_log_ratio_budget > 0:
            remaining = (
                controller.prefix_log_ratio_budget
                - abs(cumulative)
            )
            is_active = (
                remaining
                >= 2.0 * controller.max_log_odds_shift
            )
        else:
            is_active = True

        if is_active:
            budget_mask[index] = 1.0
            cumulative += float(detached_ratios[index])

    ratios = (raw_ratio * budget_mask)[valid_tensor]
    if ratios.numel() == 0:
        return None
    return ratios.mean()

def sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    """Hash a checkpoint so tiny controllers cannot silently bind to another model."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def save_emotion_controller_checkpoint(
    controller: LowRankEmotionPreference,
    path: str,
    emotion_names=None,
    extra=None,
) -> None:
    """Save only the tiny controller; GPT-SoVITS backbone weights stay separate."""
    payload = {
        "format_version": 1,
        "config": {
            "hidden_dim": controller.hidden_dim,
            "num_emotions": controller.num_emotions,
            "rank": controller.rank,
            "max_log_odds_shift": controller.max_log_odds_shift,
            "prefix_log_ratio_budget": controller.prefix_log_ratio_budget,
        },
        "state_dict": controller.state_dict(),
        "emotion_names": list(emotion_names) if emotion_names is not None else None,
        "extra": dict(extra or {}),
    }
    torch.save(payload, path)


def _validate_checkpoint_metadata(value, field: str) -> None:
    """Reject unexpected Python objects in controller checkpoint metadata."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_checkpoint_metadata(item, f"{field}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{field} contains a non-string key")
            _validate_checkpoint_metadata(item, f"{field}.{key}")
        return
    raise ValueError(
        f"{field} contains unsupported metadata type: {type(value).__name__}"
    )


def load_emotion_controller_checkpoint(
    path: str,
    map_location="cpu",
) -> Tuple[LowRankEmotionPreference, dict]:
    """Safely load a tiny controller without deserializing arbitrary Python objects."""
    payload = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("Invalid emotion controller checkpoint")

    allowed_payload_keys = {
        "format_version",
        "config",
        "state_dict",
        "emotion_names",
        "extra",
    }
    unexpected_payload_keys = set(payload) - allowed_payload_keys
    if unexpected_payload_keys:
        raise ValueError(
            "Unexpected emotion controller checkpoint keys: "
            + ", ".join(sorted(unexpected_payload_keys))
        )
    if payload.get("format_version", 1) != 1:
        raise ValueError(
            f"Unsupported emotion controller format_version: "
            f"{payload.get('format_version')!r}"
        )

    raw_config = payload.get("config")
    state_dict = payload.get("state_dict")
    if not isinstance(raw_config, dict) or not isinstance(state_dict, dict):
        raise ValueError(
            "Emotion controller checkpoint must contain config/state_dict mappings"
        )

    allowed_config_keys = {
        "hidden_dim",
        "num_emotions",
        "rank",
        "max_log_odds_shift",
        "prefix_log_ratio_budget",
    }
    unexpected_config_keys = set(raw_config) - allowed_config_keys
    if unexpected_config_keys:
        raise ValueError(
            "Unexpected emotion controller config keys: "
            + ", ".join(sorted(unexpected_config_keys))
        )
    for required_key in ("hidden_dim", "num_emotions"):
        if required_key not in raw_config:
            raise ValueError(
                f"Emotion controller config is missing {required_key!r}"
            )

    expected_state_keys = {
        "v_proj",
        "u_proj",
        "emotion_coeff.weight",
    }
    if set(state_dict) != expected_state_keys:
        raise ValueError(
            "Emotion controller state_dict keys do not match the expected schema"
        )
    if not all(isinstance(value, torch.Tensor) for value in state_dict.values()):
        raise ValueError("Emotion controller state_dict must contain tensors only")

    config = dict(raw_config)
    controller = LowRankEmotionPreference(
        hidden_dim=int(config["hidden_dim"]),
        num_emotions=int(config["num_emotions"]),
        rank=int(config.get("rank", 8)),
        max_log_odds_shift=float(config.get("max_log_odds_shift", 0.35)),
        prefix_log_ratio_budget=float(config.get("prefix_log_ratio_budget", 2.0)),
    )
    controller.load_state_dict(state_dict, strict=True)

    emotion_names = payload.get("emotion_names")
    if emotion_names is not None:
        if (
            not isinstance(emotion_names, list)
            or not all(isinstance(name, str) for name in emotion_names)
            or len(emotion_names) != controller.num_emotions
        ):
            raise ValueError(
                "emotion_names must contain one string per configured emotion"
            )

    extra = payload.get("extra") or {}
    if not isinstance(extra, dict):
        raise ValueError("Emotion controller extra metadata must be a mapping")
    _validate_checkpoint_metadata(extra, "extra")

    metadata = {
        "format_version": payload.get("format_version", 1),
        "emotion_names": emotion_names,
        "extra": extra,
        "config": config,
    }
    return controller, metadata


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
