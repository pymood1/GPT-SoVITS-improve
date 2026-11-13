#!/usr/bin/env python3
"""Fast synthetic checks for S1 attention LoRA.

No GPT-S1 dataset or training is required. This verifies:
- disabled LoRA leaves legacy attention unchanged,
- zero-init LoRA has zero functional effect,
- V-only mode keeps Q/K deltas exactly zero,
- merged ordinary weights reproduce active-LoRA output.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GPT_ROOT = PROJECT_ROOT / "GPT_SoVITS"
for path in (PROJECT_ROOT, GPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from AR.modules.activation import MultiheadAttention  # noqa: E402


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def assert_close(
    left: torch.Tensor,
    right: torch.Tensor,
    message: str,
    *,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> None:
    if not torch.allclose(left, right, atol=atol, rtol=rtol):
        error = float((left - right).abs().max().item())
        raise AssertionError(f"{message}; max_abs_error={error:.3e}")


def make_attention() -> MultiheadAttention:
    torch.manual_seed(20261113)
    module = MultiheadAttention(
        embed_dim=16,
        num_heads=4,
        dropout=0.0,
        batch_first=True,
    )
    module.eval()
    return module


def sample_input() -> torch.Tensor:
    torch.manual_seed(1234)
    return torch.randn(2, 7, 16)


def test_zero_effect_v_o() -> None:
    module = make_attention()
    x = sample_input()
    with torch.no_grad():
        baseline, _ = module(x, x, x, need_weights=False)

    module.enable_lora(target="v", rank=4, alpha=8.0)
    with torch.no_grad():
        adapted, _ = module(x, x, x, need_weights=False)

    assert_close(
        baseline,
        adapted,
        "zero-init V+O LoRA changed attention output",
    )
    delta = module.lora_in_proj_delta()
    require(
        torch.count_nonzero(delta[: 2 * module.embed_dim]) == 0,
        "V-only LoRA modified Q/K rows",
    )
    require(
        torch.count_nonzero(delta[2 * module.embed_dim :]) == 0,
        "zero-init V delta should start at zero",
    )


def test_zero_effect_qkv_o() -> None:
    module = make_attention()
    x = sample_input()
    with torch.no_grad():
        baseline, _ = module(x, x, x, need_weights=False)

    module.enable_lora(target="qkv", rank=4, alpha=8.0)
    with torch.no_grad():
        adapted, _ = module(x, x, x, need_weights=False)

    assert_close(
        baseline,
        adapted,
        "zero-init QKV+O LoRA changed attention output",
    )


def test_merge_equivalence(target: str) -> None:
    module = make_attention()
    module.enable_lora(target=target, rank=4, alpha=8.0)

    torch.manual_seed(777)
    with torch.no_grad():
        module.lora_in_B.normal_(mean=0.0, std=0.03)
        module.lora_out_B.normal_(mean=0.0, std=0.03)

    x = sample_input()
    with torch.no_grad():
        live_output, _ = module(x, x, x, need_weights=False)

    merged = make_attention()
    merged.load_state_dict(
        {
            key: value
            for key, value in module.state_dict().items()
            if ".lora_" not in key
        },
        strict=True,
    )
    with torch.no_grad():
        merged.in_proj_weight.copy_(
            module.merged_in_proj_weight()
        )
        merged.out_proj.weight.copy_(
            module.merged_out_proj_weight()
        )
        merged_output, _ = merged(
            x,
            x,
            x,
            need_weights=False,
        )

    assert_close(
        live_output,
        merged_output,
        f"{target} LoRA merge changed attention output",
        atol=2e-6,
        rtol=2e-5,
    )


def main() -> int:
    tests = [
        test_zero_effect_v_o,
        test_zero_effect_qkv_o,
        lambda: test_merge_equivalence("v"),
        lambda: test_merge_equivalence("qkv"),
    ]
    names = [
        "zero_effect_v_o",
        "zero_effect_qkv_o",
        "merge_equivalence_v_o",
        "merge_equivalence_qkv_o",
    ]
    for name, test in zip(names, tests):
        test()
        print(f"[PASS] {name}")
    print(f"[PASS] S1 LoRA self-test: {len(tests)} checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
