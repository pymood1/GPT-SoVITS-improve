"""Opt-in checkpoint research; never modify source checkpoints in place."""
from collections import OrderedDict
import math
import re

import torch


def _scale(value):
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 2.0:
        raise ValueError("Research scale must be finite and in [0, 2]")
    return value


def mix_s1_state(base, tuned, scale):
    """Interpolate matching inference state dictionaries, using FP32 arithmetic."""
    scale = _scale(scale)
    if base.keys() != tuned.keys():
        raise ValueError("S1 parameter keys differ")
    output = OrderedDict()
    for name, a in base.items():
        b = tuned[name]
        if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor) or a.shape != b.shape:
            raise ValueError("S1 tensor geometry differs")
        if a.is_floating_point() and b.is_floating_point():
            if not torch.isfinite(a).all() or not torch.isfinite(b).all():
                raise ValueError("Nonfinite S1 weight")
            result = a.clone() if scale == 0 else b.to(a.dtype).clone() if scale == 1 else torch.lerp(a.float(), b.float(), scale).to(a.dtype)
            if not torch.isfinite(result).all():
                raise ValueError("Nonfinite mixed S1 weight")
            output[name] = result
        elif a.dtype == b.dtype and torch.equal(a, b):
            output[name] = a.clone()
        else:
            raise ValueError("Nonfloating S1 state differs")
    return output


def scale_s2_lora(weights, rank, scale):
    """Scale only the B factors of standard PEFT LoRA pairs, keeping rank/alpha.

    With delta = B @ A (alpha/rank = 1 in the current V5 trainer), multiplying
    B by scale multiplies the effective update, not its square. Strictly
    reject extra trainable weights and mismatched pairs; no silent partial mix.
    """
    scale = _scale(scale)
    if type(rank) is not int or rank <= 0:
        raise ValueError("Invalid LoRA rank")
    groups = {}
    for key, value in weights.items():
        match = re.fullmatch(r"(.+)\.lora_([AB])(\.[^.]+)?\.weight", key)
        if not match or not isinstance(value, torch.Tensor) or not value.is_floating_point() or value.ndim != 2:
            raise ValueError("Unsupported non-LoRA weight")
        if not torch.isfinite(value).all():
            raise ValueError("Nonfinite LoRA weight")
        group = (match[1], match[3] or "")
        groups.setdefault(group, {})[match[2]] = key
    if not groups:
        raise ValueError("Empty LoRA state")
    output = OrderedDict()
    for pair in groups.values():
        if set(pair) != {"A", "B"}:
            raise ValueError("Incomplete LoRA pair")
        a, b = weights[pair["A"]], weights[pair["B"]]
        if a.shape[0] != rank or b.shape[1] != rank:
            raise ValueError("LoRA rank mismatch")
        output[pair["A"]] = a.clone()
        scaled = (b.float() * scale).to(b.dtype)
        if not torch.isfinite(scaled).all():
            raise ValueError("Nonfinite scaled LoRA weight")
        output[pair["B"]] = scaled
    return output
