#!/usr/bin/env python3
"""Compare S1 checkpoints in the same reference-conditioned space used by the emotion controller.

The first --s1 checkpoint is the baseline. For every checkpoint this tool:
- builds fixed-reference -> target teacher-forcing batches exactly like controller training,
- measures hidden emotion separability with the existing grouped linear probe,
- compares target hidden/probability drift against the baseline,
- reports parameter-space drift for embeddings, Q/K/V, attention output, FFN,
  LayerNorm, and semantic readout.

Example:
    python tools/compare_s1_controller_space.py \
      --s1 pretrained=/path/to/s1v3.ckpt \
      --s1 full15=/path/to/Gilberta-e15.ckpt \
      --feature-dir /path/to/features \
      --metadata /path/to/canonical/metadata.jsonl \
      --reference-map /path/to/reference_map.json \
      --reference-prompt-cache /path/to/reference_prompts.pt \
      --emotions happy,angry,disgusted,surprised \
      --version v5dev \
      --output /path/to/s1_space_comparison.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "GPT_SoVITS"))

from AR.models.emotion_control import sha256_file  # noqa: E402
from AR.models.t2s_lightning_module import (  # noqa: E402
    Text2SemanticLightningModule,
)
from text.multilingual_bert import (  # noqa: E402
    MULTILINGUAL_BERT_LANGUAGES,
    checkpoint_multilingual_bert_config,
)
from tools.emotion_training_pipeline import (  # noqa: E402
    run_mean_pool_linear_probe,
)
from tools.train_emotion_controller import (  # noqa: E402
    FeatureStore,
    TrainingDataError,
    build_single_sample_batch,
    build_training_samples,
    load_reference_map,
    load_reference_prompt_cache,
    prepare_controller_sample,
)


class ComparisonError(RuntimeError):
    pass


def parse_checkpoint_spec(value: str) -> Tuple[str, Path]:
    if "=" in value:
        label, raw_path = value.split("=", 1)
        label = label.strip()
        raw_path = raw_path.strip()
    else:
        raw_path = value.strip()
        label = Path(raw_path).stem
    if not label or not raw_path:
        raise argparse.ArgumentTypeError(
            "--s1 must be LABEL=/path/to/checkpoint or /path/to/checkpoint"
        )
    return label, Path(raw_path).expanduser().resolve()


def checkpoint_payload(path: Path) -> dict:
    if not path.is_file():
        raise ComparisonError(f"S1 checkpoint not found: {path}")
    payload = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )
    if (
        not isinstance(payload, dict)
        or not isinstance(payload.get("config"), dict)
        or not isinstance(payload.get("weight"), dict)
    ):
        raise ComparisonError(
            f"S1 checkpoint must contain config + weight mappings: {path}"
        )
    return payload


def text_feature_signature(config: dict) -> dict:
    enabled, model_name = checkpoint_multilingual_bert_config(config)
    return {
        "multilingual_bert": bool(enabled),
        "multilingual_bert_model": model_name if enabled else None,
    }


def required_bert_languages(config: dict) -> set[str]:
    enabled, _ = checkpoint_multilingual_bert_config(config)
    result = {"zh"}
    if enabled:
        result.update(MULTILINGUAL_BERT_LANGUAGES)
    return result


def load_model(
    payload: dict,
    device: torch.device,
    output_stub: Path,
):
    wrapper = Text2SemanticLightningModule(
        payload["config"],
        output_stub,
        is_train=False,
    )
    wrapper.load_state_dict(
        payload["weight"],
        strict=True,
    )
    model = wrapper.model.to(
        device=device,
        dtype=torch.float32,
    ).eval()
    model.requires_grad_(False)
    return model


def cpu_half(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to(
        device="cpu",
        dtype=torch.float16,
    ).clone()


def snapshot_parameter_groups(model) -> Dict[str, List[torch.Tensor]]:
    groups: Dict[str, List[torch.Tensor]] = {
        "text_embedding": [cpu_half(model.ar_text_embedding.weight)],
        "audio_embedding": [cpu_half(model.ar_audio_embedding.weight)],
        "bert_projection": [
            cpu_half(model.bert_proj.weight),
            cpu_half(model.bert_proj.bias),
        ],
        "position_alpha": [
            cpu_half(model.ar_text_position.alpha),
            cpu_half(model.ar_audio_position.alpha),
        ],
        "attention_q": [],
        "attention_k": [],
        "attention_v": [],
        "attention_out": [],
        "ffn": [],
        "layer_norm": [],
        "semantic_readout": [cpu_half(model.ar_predict_layer.weight)],
    }

    hidden_dim = int(model.model_dim)
    for layer in model.h.layers:
        in_proj = layer.self_attn.in_proj_weight.detach()
        if in_proj.shape[0] != 3 * hidden_dim:
            raise ComparisonError(
                "Unexpected S1 attention in_proj_weight shape: "
                f"{tuple(in_proj.shape)}"
            )
        q, k, v = in_proj.split(hidden_dim, dim=0)
        groups["attention_q"].append(cpu_half(q))
        groups["attention_k"].append(cpu_half(k))
        groups["attention_v"].append(cpu_half(v))
        if layer.self_attn.in_proj_bias is not None:
            qb, kb, vb = layer.self_attn.in_proj_bias.detach().split(
                hidden_dim,
                dim=0,
            )
            groups["attention_q"].append(cpu_half(qb))
            groups["attention_k"].append(cpu_half(kb))
            groups["attention_v"].append(cpu_half(vb))

        groups["attention_out"].append(
            cpu_half(layer.self_attn.out_proj.weight)
        )
        if layer.self_attn.out_proj.bias is not None:
            groups["attention_out"].append(
                cpu_half(layer.self_attn.out_proj.bias)
            )

        groups["ffn"].extend(
            [
                cpu_half(layer.linear1.weight),
                cpu_half(layer.linear1.bias),
                cpu_half(layer.linear2.weight),
                cpu_half(layer.linear2.bias),
            ]
        )
        groups["layer_norm"].extend(
            [
                cpu_half(layer.norm1.weight),
                cpu_half(layer.norm1.bias),
                cpu_half(layer.norm2.weight),
                cpu_half(layer.norm2.bias),
            ]
        )

    return groups


def group_drift(
    baseline: Sequence[torch.Tensor],
    current: Sequence[torch.Tensor],
) -> dict:
    if len(baseline) != len(current):
        raise ComparisonError("parameter group structure mismatch")

    dot = 0.0
    base_sq = 0.0
    current_sq = 0.0
    diff_sq = 0.0
    count = 0

    for base_tensor, current_tensor in zip(baseline, current):
        if base_tensor.shape != current_tensor.shape:
            raise ComparisonError(
                "parameter tensor shape mismatch: "
                f"{tuple(base_tensor.shape)} != {tuple(current_tensor.shape)}"
            )
        base = base_tensor.float()
        cur = current_tensor.float()
        delta = cur - base
        dot += float((base * cur).sum().item())
        base_sq += float((base * base).sum().item())
        current_sq += float((cur * cur).sum().item())
        diff_sq += float((delta * delta).sum().item())
        count += base.numel()

    denom = math.sqrt(max(base_sq * current_sq, 1e-30))
    return {
        "parameters": int(count),
        "relative_delta_norm": math.sqrt(diff_sq)
        / max(math.sqrt(base_sq), 1e-15),
        "cosine_similarity": dot / denom,
    }


def js_divergence(
    p: torch.Tensor,
    q: torch.Tensor,
) -> torch.Tensor:
    p = p.float().clamp_min(1e-12)
    q = q.float().clamp_min(1e-12)
    p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    q = q / q.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    m = 0.5 * (p + q)
    return 0.5 * (
        (p * (torch.log(p) - torch.log(m))).sum(dim=-1)
        + (q * (torch.log(q) - torch.log(m))).sum(dim=-1)
    )


def topk_overlap(
    baseline: torch.Tensor,
    current: torch.Tensor,
    k: int,
) -> float:
    k = min(k, baseline.shape[-1], current.shape[-1])
    baseline_ids = torch.topk(baseline, k=k, dim=-1).indices.tolist()
    current_ids = torch.topk(current, k=k, dim=-1).indices.tolist()
    return len(set(baseline_ids) & set(current_ids)) / max(k, 1)


def select_samples(
    samples: Sequence[dict],
    emotion_names: Sequence[str],
    max_per_emotion: int,
    seed: int,
) -> List[dict]:
    if max_per_emotion <= 0:
        return list(samples)

    rng = random.Random(seed)
    selected: List[dict] = []
    for emotion in emotion_names:
        candidates = [
            sample
            for sample in samples
            if sample["emotion"] == emotion
        ]
        rng.shuffle(candidates)
        selected.extend(candidates[:max_per_emotion])

    selected.sort(
        key=lambda item: (
            int(item["emotion_id"]),
            str(item["name"]),
        )
    )
    return selected


def evaluate_checkpoint(
    *,
    label: str,
    path: Path,
    payload: dict,
    model,
    store: FeatureStore,
    samples: Sequence[dict],
    emotion_names: Sequence[str],
    reference_prompt_cache: Dict[str, dict],
    device: torch.device,
    top_k: int,
    top_p: float,
    temperature: float,
    repetition_penalty: float,
    probe_steps: int,
    seed: int,
) -> dict:
    hidden_features: List[torch.Tensor] = []
    mean_probabilities: List[torch.Tensor] = []

    pad_val = int(payload["config"]["data"]["pad_val"])
    for index, sample in enumerate(samples, 1):
        batch = build_single_sample_batch(
            store=store,
            reference_name=sample["reference"],
            target_name=sample["name"],
            device=device,
            pad_val=pad_val,
            reference_prompt_cache=reference_prompt_cache,
        )
        prepared = prepare_controller_sample(
            model=model,
            batch=batch,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )
        hidden = prepared["hidden"].float()
        full_probs = prepared["full_probs"].float()
        if (
            hidden.ndim != 2
            or hidden.shape[0] == 0
            or full_probs.ndim != 2
            or full_probs.shape[0] == 0
            or not bool(torch.isfinite(hidden).all())
            or not bool(torch.isfinite(full_probs).all())
        ):
            raise ComparisonError(
                f"{label}: invalid cached representation for {sample['name']}"
            )

        hidden_features.append(hidden.mean(dim=0).cpu())
        mean_probabilities.append(full_probs.mean(dim=0).cpu())

        if index == 1 or index % 25 == 0 or index == len(samples):
            print(
                f"[{label}] cached {index}/{len(samples)} "
                f"reference-conditioned targets"
            )

    hidden_matrix = torch.stack(hidden_features)
    probability_matrix = torch.stack(mean_probabilities)
    labels = [int(sample["emotion_id"]) for sample in samples]

    probe = run_mean_pool_linear_probe(
        hidden_matrix,
        labels,
        emotion_names,
        samples=samples,
        seed=seed,
        max_folds=5,
        permutations=3,
        steps=probe_steps,
        method="s1_reference_conditioned_hidden_probe",
    )

    return {
        "label": label,
        "path": str(path),
        "sha256": sha256_file(str(path)),
        "samples": len(samples),
        "hidden_features": hidden_matrix,
        "mean_probabilities": probability_matrix,
        "probe": probe,
        "parameter_groups": snapshot_parameter_groups(model),
    }


def serializable_report(result: dict) -> dict:
    return {
        key: value
        for key, value in result.items()
        if key
        not in {
            "hidden_features",
            "mean_probabilities",
            "parameter_groups",
        }
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--s1",
        action="append",
        required=True,
        help=(
            "S1 checkpoint spec. Repeat at least twice. "
            "First checkpoint is the baseline. Format LABEL=/path/model.ckpt"
        ),
    )
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--reference-map", required=True)
    parser.add_argument("--reference-prompt-cache")
    parser.add_argument(
        "--emotions",
        default="happy,angry,disgusted,surprised",
    )
    parser.add_argument("--version", default="v5dev")
    parser.add_argument("--max-samples-per-emotion", type=int, default=0)
    parser.add_argument("--probe-steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.35)
    parser.add_argument("--drift-top-k", type=int, default=20)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    specs = [parse_checkpoint_spec(value) for value in args.s1]
    if len(specs) < 2:
        raise ComparisonError("Repeat --s1 at least twice")
    labels_seen = [label for label, _ in specs]
    if len(set(labels_seen)) != len(labels_seen):
        raise ComparisonError("--s1 labels must be unique")
    if args.probe_steps <= 0:
        raise ComparisonError("--probe-steps must be positive")
    if args.top_k < 2:
        raise ComparisonError("--top-k must be >= 2")
    if not 0.0 < args.top_p <= 1.0:
        raise ComparisonError("--top-p must be in (0,1]")
    if args.temperature <= 0:
        raise ComparisonError("--temperature must be positive")

    emotion_names = [
        value.strip().lower()
        for value in args.emotions.split(",")
        if value.strip()
        and value.strip().lower() not in {"neutral", "default"}
    ]
    if len(emotion_names) < 2:
        raise ComparisonError("Need at least two non-neutral emotions")
    emotion_to_id = {
        name: index
        for index, name in enumerate(emotion_names)
    }

    payloads = {
        label: checkpoint_payload(path)
        for label, path in specs
    }
    baseline_label, baseline_path = specs[0]
    baseline_config = payloads[baseline_label]["config"]
    expected_text_features = text_feature_signature(baseline_config)

    for label, _ in specs[1:]:
        current_signature = text_feature_signature(
            payloads[label]["config"]
        )
        if current_signature != expected_text_features:
            raise ComparisonError(
                f"{label}: text feature config differs from baseline: "
                f"{current_signature} != {expected_text_features}"
            )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ComparisonError("CUDA requested but unavailable")

    feature_dir = Path(args.feature_dir).expanduser().resolve()
    metadata_path = Path(args.metadata).expanduser().resolve()
    store = FeatureStore(
        metadata_path=metadata_path,
        feature_dir=feature_dir,
        version=args.version,
        required_bert_languages=required_bert_languages(
            baseline_config
        ),
    )
    reference_map = load_reference_map(
        Path(args.reference_map).expanduser().resolve(),
        store,
    )
    reference_prompt_cache, reference_cache_metadata = (
        load_reference_prompt_cache(
            Path(args.reference_prompt_cache).expanduser().resolve()
            if args.reference_prompt_cache
            else None
        )
    )

    samples = build_training_samples(
        store,
        emotion_to_id,
        reference_map,
    )
    samples = select_samples(
        samples,
        emotion_names,
        args.max_samples_per_emotion,
        args.seed,
    )
    if not samples:
        raise ComparisonError("No usable reference-conditioned samples")

    counts = {
        emotion: sum(
            sample["emotion"] == emotion
            for sample in samples
        )
        for emotion in emotion_names
    }
    print(
        "[data] "
        + ", ".join(
            f"{name}={counts[name]}"
            for name in emotion_names
        )
    )

    results: Dict[str, dict] = {}
    for label, path in specs:
        print(f"\n=== S1 {label}: {path} ===")
        model = load_model(
            payloads[label],
            device,
            ROOT / "TEMP" / "s1-controller-space-comparison",
        )
        try:
            results[label] = evaluate_checkpoint(
                label=label,
                path=path,
                payload=payloads[label],
                model=model,
                store=store,
                samples=samples,
                emotion_names=emotion_names,
                reference_prompt_cache=reference_prompt_cache,
                device=device,
                top_k=args.top_k,
                top_p=args.top_p,
                temperature=args.temperature,
                repetition_penalty=args.repetition_penalty,
                probe_steps=args.probe_steps,
                seed=args.seed,
            )
        finally:
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    baseline = results[baseline_label]
    report = {
        "baseline": baseline_label,
        "baseline_path": str(baseline_path),
        "emotion_names": emotion_names,
        "sample_counts": counts,
        "samples": len(samples),
        "version": args.version,
        "text_features": expected_text_features,
        "reference_cache_metadata": reference_cache_metadata,
        "checkpoints": {},
    }

    base_hidden = baseline["hidden_features"].float()
    base_probs = baseline["mean_probabilities"].float()

    for label, _ in specs:
        current = results[label]
        hidden = current["hidden_features"].float()
        probs = current["mean_probabilities"].float()

        if hidden.shape != base_hidden.shape:
            raise ComparisonError(
                f"{label}: hidden feature shape differs from baseline"
            )
        if probs.shape != base_probs.shape:
            raise ComparisonError(
                f"{label}: probability shape differs from baseline"
            )

        hidden_cos = F.cosine_similarity(
            hidden,
            base_hidden,
            dim=-1,
            eps=1e-8,
        )
        js = js_divergence(base_probs, probs)
        overlaps = [
            topk_overlap(base_probs[i], probs[i], args.drift_top_k)
            for i in range(probs.shape[0])
        ]

        parameter_drift = {}
        for group_name, base_group in baseline[
            "parameter_groups"
        ].items():
            parameter_drift[group_name] = group_drift(
                base_group,
                current["parameter_groups"][group_name],
            )

        item = serializable_report(current)
        item["drift_from_baseline"] = {
            "hidden_cosine_mean": float(hidden_cos.mean().item()),
            "hidden_cosine_min": float(hidden_cos.min().item()),
            "mean_probability_js": float(js.mean().item()),
            "mean_probability_js_max": float(js.max().item()),
            "topk_overlap_mean": (
                sum(overlaps) / len(overlaps)
                if overlaps
                else None
            ),
            "topk": int(args.drift_top_k),
            "parameter_groups": parameter_drift,
        }
        report["checkpoints"][label] = item

        probe = item["probe"]
        print(
            f"[summary] {label}: "
            f"probe_acc={probe.get('oof_macro_accuracy')} "
            f"probe_auc={probe.get('oof_macro_auc')} "
            f"hidden_cos={item['drift_from_baseline']['hidden_cosine_mean']:.6f} "
            f"JS={item['drift_from_baseline']['mean_probability_js']:.6f}"
        )

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(f".{output.name}.tmp")
    temp.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    temp.replace(output)
    print(f"[OK] saved: {output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        ComparisonError,
        TrainingDataError,
        OSError,
        ValueError,
        RuntimeError,
    ) as exc:
        print(f"[FATAL] {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
