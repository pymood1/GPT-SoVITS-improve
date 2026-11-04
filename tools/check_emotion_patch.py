#!/usr/bin/env python3
"""Small invariant checks for the emotion-v5 S1 preference patch."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "GPT_SoVITS"))

from AR.models.emotion_control import (  # noqa: E402
    EmotionSamplingState,
    LowRankEmotionPreference,
    load_emotion_controller_checkpoint,
    save_emotion_controller_checkpoint,
)


def check_zero_effect() -> None:
    controller = LowRankEmotionPreference(
        hidden_dim=8,
        num_emotions=4,
        rank=4,
    )
    hidden = torch.randn(2, 8)
    output_weight = torch.randn(5, 8)
    scores = controller.preference_scores(
        hidden,
        output_weight,
        emotion_id=torch.tensor([0, 1]),
    )
    if not torch.equal(scores, torch.zeros_like(scores)):
        raise AssertionError("Zero-effect initialization was lost")


def check_zero_init_is_trainable() -> None:
    controller = LowRankEmotionPreference(
        hidden_dim=8,
        num_emotions=4,
        rank=4,
    )
    hidden = torch.randn(1, 8)
    output_weight = torch.randn(5, 8)
    base = torch.tensor(
        [[0.40, 0.20, 0.15, 0.15, 0.10]],
        dtype=torch.float32,
    )
    scores = controller.preference_scores(
        hidden,
        output_weight,
        emotion_id=0,
    )
    tilted, _ = controller.tilt_probabilities(
        base_probs=base,
        preference_scores=scores,
        strength=1.0,
        eos_id=None,
        state=None,
    )
    loss = -torch.log(tilted[0, 0].clamp_min(1e-12))
    loss.backward()
    grad = controller.u_proj.grad
    if grad is None or not torch.isfinite(grad).all() or grad.norm() <= 0:
        raise AssertionError(
            "Zero-effect initialization blocks the first training update"
        )


def check_support_and_eos() -> None:
    controller = LowRankEmotionPreference(
        hidden_dim=8,
        num_emotions=4,
        rank=4,
        max_log_odds_shift=0.3,
        prefix_log_ratio_budget=2.0,
    )
    base = torch.tensor([[0.40, 0.20, 0.0, 0.15, 0.25]], dtype=torch.float32)
    scores = torch.tensor([[2.0, -1.0, 9.0, 0.5, -7.0]], dtype=torch.float32)
    tilted, _ = controller.tilt_probabilities(
        base_probs=base,
        preference_scores=scores,
        strength=1.0,
        eos_id=4,
        state=None,
    )

    if tilted[0, 2].item() != 0.0:
        raise AssertionError("Filtered token was resurrected")
    if tilted[0, 4].item() != base[0, 4].item():
        raise AssertionError("EOS probability mass changed")
    if not torch.allclose(tilted.sum(dim=-1), torch.ones(1), atol=1e-6):
        raise AssertionError("Tilted distribution is not normalized")


def check_budget_fail_open() -> None:
    controller = LowRankEmotionPreference(
        hidden_dim=8,
        num_emotions=4,
        rank=4,
        max_log_odds_shift=0.3,
        prefix_log_ratio_budget=1.0,
    )
    base = torch.tensor([[0.6, 0.4]], dtype=torch.float32)
    scores = torch.tensor([[3.0, -3.0]], dtype=torch.float32)
    state = EmotionSamplingState(
        cumulative_log_ratio=torch.tensor([0.8], dtype=torch.float32)
    )
    tilted, _ = controller.tilt_probabilities(
        base_probs=base,
        preference_scores=scores,
        strength=1.0,
        eos_id=None,
        state=state,
    )
    if not torch.equal(tilted, base):
        raise AssertionError("Prefix budget did not fail open")


def check_checkpoint_roundtrip() -> None:
    controller = LowRankEmotionPreference(
        hidden_dim=8,
        num_emotions=4,
        rank=4,
    )
    with torch.no_grad():
        controller.u_proj.normal_(0.0, 0.02)

    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "controller.pt")
        save_emotion_controller_checkpoint(
            controller,
            path,
            emotion_names=["neutral", "happy", "sad", "angry"],
        )
        loaded, metadata = load_emotion_controller_checkpoint(path)

    for key, value in controller.state_dict().items():
        if not torch.equal(value, loaded.state_dict()[key]):
            raise AssertionError(f"Checkpoint mismatch: {key}")
    if metadata["emotion_names"] != ["neutral", "happy", "sad", "angry"]:
        raise AssertionError("Checkpoint metadata mismatch")


def check_checkpoint_schema_rejects_unexpected_metadata() -> None:
    controller = LowRankEmotionPreference(
        hidden_dim=8,
        num_emotions=2,
        rank=4,
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "controller.pt")
        save_emotion_controller_checkpoint(
            controller,
            path,
            emotion_names=["neutral", "happy"],
            extra={"safe": "value"},
        )
        payload = torch.load(path, map_location="cpu", weights_only=True)
        payload["unexpected"] = "reject-me"
        torch.save(payload, path)
        try:
            load_emotion_controller_checkpoint(path)
        except ValueError:
            pass
        else:
            raise AssertionError(
                "Unexpected controller checkpoint fields were accepted"
            )


def check_integration_wiring() -> None:
    root = ROOT
    api = (root / "api_v2.py").read_text(encoding="utf-8")
    webui = (
        root / "GPT_SoVITS/inference_webui.py"
    ).read_text(encoding="utf-8")
    tts = (
        root / "GPT_SoVITS/TTS_infer_pack/TTS.py"
    ).read_text(encoding="utf-8")
    streaming = (
        root / "GPT_SoVITS/TTS_infer_pack/streaming.py"
    ).read_text(encoding="utf-8")

    required_api = (
        "emotion_id: Optional[int] = None",
        "emotion_strength: float = 1.0",
        '@APP.get("/set_emotion_controller")',
        '@APP.get("/unload_emotion_controller")',
    )
    for marker in required_api:
        if marker not in api:
            raise AssertionError(f"API integration marker missing: {marker}")

    required_webui = (
        "load_emotion_controller_webui",
        "emotion_strength_slider",
        "emotion_id=emotion_id if emotion_enabled else None",
        "change_gpt_weights_webui",
        "sovits_path_global",
        'vocoder_family = "v5" if target_version in V5_VERSIONS else "v4"',
        "gsv-%s-pretrained/vocoder.pth",
        "init_hifigan(model_version)",
    )
    for marker in required_webui:
        if marker not in webui:
            raise AssertionError(f"WebUI integration marker missing: {marker}")

    for marker in (
        "expected_s1_hash",
        "expected_s2_hash",
        "semantic_s2_sha256",
        "_invalidate_s2_dependent_state",
        "prompt_version",
        "V5_VERSIONS",
    ):
        if marker not in tts:
            raise AssertionError(f"TTS safety marker missing: {marker}")

    if "gpt_streaming does not yet support" not in streaming:
        raise AssertionError(
            "Streaming path does not explicitly reject unsupported emotion steering"
        )


def main() -> int:
    torch.manual_seed(233333)
    check_zero_effect()
    check_zero_init_is_trainable()
    check_support_and_eos()
    check_budget_fail_open()
    check_checkpoint_roundtrip()
    check_checkpoint_schema_rejects_unexpected_metadata()
    check_integration_wiring()
    print("[OK] emotion-v5 invariant checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
