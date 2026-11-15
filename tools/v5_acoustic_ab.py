"""Opt-in, in-process acoustic A/B. Never runs S1 or uploads input/output.

Execute production source bodies with the already loaded model and vocoder.
The upstream TTS source is pinned, not the obsolete upstream general WebUI.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import types
import urllib.request
from pathlib import Path

UPSTREAM = "6d51ffceafbf240145619af4c92408a6e54b7924"
TRACE_NAMES = {
    "refer", "refer_audio_spec", "ref_audio", "fea_ref", "fea_todo", "ge",
    "mel2", "mel", "cfm_res", "audio", "wav_gen",
}


def record(trace, name, value):
    import torch
    if isinstance(value, torch.Tensor):
        trace.setdefault(name, []).append(value.detach().clone())


class Capture(ast.NodeTransformer):
    def visit_Assign(self, node):
        names = [n.id for target in node.targets for n in ast.walk(target)
                 if isinstance(n, ast.Name) and n.id in TRACE_NAMES]
        extra = []
        if "fea_todo" in names and isinstance(node.targets[0], ast.Tuple):
            second = node.targets[0].elts[1]
            if isinstance(second, ast.Name):
                extra = [ast.Expr(ast.Call(ast.Name("_ab_record", ast.Load()),
                                          [ast.Constant("decode_ge"), ast.Name(second.id, ast.Load())], []))]
                names = [n for n in names if n != "ge"]
        return [node] + [ast.Expr(ast.Call(ast.Name("_ab_record", ast.Load()),
                                          [ast.Constant(n), ast.Name(n, ast.Load())], []))
                         for n in names] + extra


def _compile(nodes, namespace, trace):
    ns = dict(namespace)
    ns["_ab_record"] = lambda name, value: record(trace, name, value)
    tree = ast.Module(body=copy.deepcopy(nodes), type_ignores=[])
    tree = Capture().visit(tree)
    exec(compile(ast.fix_missing_locations(tree), "<pinned acoustic source>", "exec"), ns)
    return ns


def compile_webui_acoustic(source, namespace, trace):
    tree = ast.parse(source)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "get_tts_wav")
    # Select the actual top-level legacy/V3/V4/V5 branch in the output loop.
    candidates = [n for n in ast.walk(fn) if isinstance(n, ast.If)
                  and ast.unparse(n.test) == "model_version not in v3v4set"]
    if len(candidates) != 1:
        raise ValueError("WebUI acoustic source layout changed; refusing approximate replay")
    branch = candidates[0].orelse
    wrapper = ast.FunctionDef(
        name="_ab_webui", args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[],
                                           kw_defaults=[], defaults=[]),
        body=branch + [ast.Return(ast.Name("audio", ast.Load()))], decorator_list=[])
    return _compile([wrapper], namespace, trace)["_ab_webui"]


def compile_engine(source, namespace, trace):
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TTS")
    names = {"using_vocoder_synthesis", "_get_ref_spec"}
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    for fn in methods:
        fn.decorator_list = []
        fn.returns = None
        for arg in fn.args.args:
            arg.annotation = None
    return _compile(methods, namespace, trace)


def tensor_stats(x):
    import torch
    y = x.detach().float()
    finite = y[torch.isfinite(y)].double()
    return dict(shape=list(x.shape), dtype=str(x.dtype), device=str(x.device),
                nonfinite=x.numel() - finite.numel(),
                min=float(finite.min()) if finite.numel() else None,
                max=float(finite.max()) if finite.numel() else None,
                mean=float(finite.mean()) if finite.numel() else None,
                std=float(finite.std(unbiased=False)) if finite.numel() else None,
                rms=float(finite.square().mean().sqrt()) if finite.numel() else None)


def save_float_wav(path, audio, sample_rate=48000):
    # torchaudio's TorchCodec writer can ignore encoding/bits_per_sample.
    # scipy preserves the requested IEEE FLOAT32 values, including peaks > 1.
    from scipy.io.wavfile import write
    write(str(path), sample_rate, audio.detach().float().cpu().numpy())


def compare_traces(left, right):
    import torch
    # Name aliases correspond to the same production stage.
    aliases = {"refer": "refer_audio_spec"}
    ordered = ["refer", "fea_ref", "ge", "ref_audio", "mel2", "fea_todo", "decode_ge",
               "cfm_mu", "cfm_lengths", "cfm_prompt", "cfm_noise", "cfm_output",
               "normalized_mel", "denorm_mel", "vocoder_input", "raw_audio"]
    rows = []
    first = None
    first_value = None
    first_metadata = None
    for name in ordered:
        if name not in left:
            continue
        other = name if name in right else aliases.get(name, name)
        a, b = left[name], right.get(other, [])
        for i in range(max(len(a), len(b))):
            label = f"{name}[{i}]"
            row = {"stage": label}
            if i >= len(a) or i >= len(b):
                row.update(equal=False, reason="capture count differs")
                row["value_equal"] = False
            else:
                x, y = a[i], b[i]
                meta = x.shape == y.shape and x.dtype == y.dtype and x.device == y.device
                row.update(left=tensor_stats(x), right=tensor_stats(y),
                           equal=bool(meta and torch.equal(x, y)))
                row["value_equal"] = bool(x.shape == y.shape and torch.equal(x.float().cpu(), y.float().cpu()))
                row["metadata_equal"] = meta
                if x.shape == y.shape:
                    diff = (x.float() - y.float()).abs()
                    row["max_abs_diff"] = (float(diff.max()) if torch.isfinite(diff).all() else None) if diff.numel() else 0.
            rows.append(row)
            if not row["equal"] and first is None:
                first = label
            if not row["value_equal"] and first_value is None:
                first_value = label
            if not row.get("metadata_equal", False) and first_metadata is None:
                first_metadata = label
    return {"first_difference": first, "first_value_difference": first_value,
            "first_metadata_difference": first_metadata, "stages": rows}


def official_sources(root):
    paths = ["GPT_SoVITS/TTS_infer_pack/TTS.py", "GPT_SoVITS/module/models_v5.py",
             "GPT_SoVITS/module/models.py", "GPT_SoVITS/module/mel_processing.py"]
    source, audit = {}, {}
    for path in paths:
        url = f"https://raw.githubusercontent.com/RVC-Boss/GPT-SoVITS/{UPSTREAM}/{path}"
        with urllib.request.urlopen(url, timeout=30) as response:
            data = response.read()
        source[path] = data.decode("utf-8")
        local = (root / path).read_text(encoding="utf-8")
        same = ast.dump(ast.parse(local)) == ast.dump(ast.parse(source[path]))
        audit[path] = {"upstream_sha256": hashlib.sha256(data).hexdigest(),
                       "local_sha256": hashlib.sha256(local.encode()).hexdigest(),
                       "ast_equal": same}
        if path != paths[0] and not same:
            raise ValueError(f"Shared acoustic implementation differs from pinned upstream: {path}")
    return source, audit


def _engine_globals(source, namespace):
    # Execute official mel/normalization/resampling definitions, not fork copies.
    wanted = {"spec_min", "spec_max", "mel_fn", "mel_fn_v4"}
    functions = {"resample", "norm_spec", "denorm_spec", "_resampy_gpu_filter"}
    nodes = []
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name in functions:
            nodes.append(node)
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in wanted
                                                 for t in node.targets):
            nodes.append(node)
    ns = dict(namespace)
    ns["resample_transform_dict"] = {}
    ns["resampy_filter_cache"] = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 "<official mel frontend>", "exec"), ns)
    return ns


def checkpoint_bindings(namespace, root):
    import torch
    result = {}
    for label, path in {
        "s2": namespace["sovits_path_global"],
        "vocoder": root / "GPT_SoVITS/pretrained_models/gsv-v5-pretrained/vocoder.pth",
    }.items():
        digest = hashlib.sha256()
        with open(path, "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        result[label + "_sha256"] = digest.hexdigest()
    expected = torch.load(root / "GPT_SoVITS/pretrained_models/gsv-v5-pretrained/vocoder.pth",
                          map_location="cpu", weights_only=True)
    actual = namespace["hifigan_model"].state_dict()
    mismatch = [key for key, value in actual.items()
                if key not in expected or value.shape != expected[key].shape
                or not torch.equal(value.detach().cpu(), expected[key].to(dtype=value.dtype))]
    result["vocoder_loaded_state_matches_file_after_dtype_cast"] = not mismatch and set(actual) == set(expected)
    result["vocoder_mismatched_keys"] = mismatch[:10]
    result["vocoder_checkpoint_extra_keys"] = sorted(set(expected) - set(actual))[:10]
    expected_s2 = namespace["load_sovits_new"](namespace["sovits_path_global"])["weight"]
    actual_s2 = namespace["vq_model"].state_dict()
    common = sorted(set(expected_s2) & set(actual_s2))
    mismatched_s2 = [key for key in common
                     if actual_s2[key].shape != expected_s2[key].shape
                     or not torch.equal(actual_s2[key].detach().cpu(),
                                        expected_s2[key].to(dtype=actual_s2[key].dtype))]
    result["s2_loaded_common_state_matches_file_after_dtype_cast"] = not mismatched_s2
    result["s2_compared_keys"] = len(common)
    result["s2_mismatched_keys"] = mismatched_s2[:10]
    result["s2_missing_from_runtime_keys"] = sorted(set(expected_s2) - set(actual_s2))[:20]
    result["s2_missing_from_checkpoint_keys"] = sorted(set(actual_s2) - set(expected_s2))[:20]
    return result


def _run_sequence(namespace, inputs, output_root, routes=None, frozen_noise=None):
    """Called only when explicitly enabled; tensors/WAV stay in the runtime."""
    import datetime
    import torch
    import torchaudio

    if namespace["model_version"] not in namespace["V5_VERSIONS"]:
        raise ValueError("V5 diagnostic only")
    root = Path(__file__).resolve().parents[1]
    sources, audit = namespace.get("_ab_sources") or official_sources(root)
    engine_source = sources["GPT_SoVITS/TTS_infer_pack/TTS.py"]
    # No S1 sampling here. Clone the exact semantic and prompt selected by the UI.
    base = dict(namespace)
    base.update(inputs)
    base["pred_semantic"] = inputs["pred_semantic"].detach().clone()
    base["prompt"] = inputs["prompt"].detach().clone()
    if base["pred_semantic"].shape[-1] > 1000:
        raise ValueError("Diagnostic budget: use a target with at most 1000 semantic tokens")
    model = namespace["vq_model"]
    vocoder = namespace["hifigan_model"]
    if vocoder is None or namespace["hifigan_version"] != "v5":
        raise ValueError("Load the actual V5 vocoder before diagnosing")
    if getattr(model, "training", False) or getattr(vocoder, "training", False):
        raise ValueError("Acoustic A/B requires eval-mode S2 and vocoder")
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = Path(output_root) / timestamp
    out.mkdir(parents=True, exist_ok=False)
    report = {"upstream": UPSTREAM, "source_audit": audit, "cases": {},
              "seed": 1234,
              "webui_source_sha256": hashlib.sha256(
                  (root / "GPT_SoVITS/inference_webui.py").read_bytes()).hexdigest(),
              "semantic": tensor_stats(base["pred_semantic"]),
              "sample_steps": base["sample_steps"], "cfg": base["cfg_rate"],
              "model_version": namespace["model_version"], "output_sr": 48000,
              "cfm_runtime": {key: getattr(model.cfm, key, None) for key in
                              ("noise_temperature", "use_static_cache", "use_conditioner_cache",
                               "use_step_embedding", "cfg_drop_text")},
              "grad_enabled": torch.is_grad_enabled(),
              "autocast_cuda": torch.is_autocast_enabled(),
              "reference_features_shared": False,
              "models_shared": True,
              "mel_settings": {"sampling_rate": 32000, "n_fft": 1280,
                               "win_size": 1280, "hop_size": 320, "num_mels": 100,
                               "fmin": 0, "fmax": None, "center": False,
                               "spec_min": -12, "spec_max": 2},
              "listening_performed": False}
    report["s2_audio_config"] = {
        key: getattr(base["hps"].data, key) for key in
        ("sampling_rate", "filter_length", "hop_length", "win_length")}
    report["vocoder_upsample_strides"] = [list(layer.stride) for layer in getattr(vocoder, "ups", [])]
    report["loaded_classes"] = {"s2": type(model).__name__, "cfm": type(model.cfm).__name__,
                                "vocoder": type(vocoder).__name__}
    report["loaded_s2_version"] = getattr(model, "version", None)
    report["torch_version"] = str(torch.__version__)
    report["cudnn_deterministic"] = torch.backends.cudnn.deterministic
    report["cudnn_benchmark"] = torch.backends.cudnn.benchmark
    report["matmul_allow_tf32"] = torch.backends.cuda.matmul.allow_tf32
    if "get_sovits_version_from_path_fast" in namespace:
        report["s2_file_version"] = list(namespace["get_sovits_version_from_path_fast"](
            namespace["sovits_path_global"]))
    if report["vocoder_upsample_strides"]:
        import math
        report["vocoder_sr_derived_from_mel_and_stride"] = (
            32000 / 320 * math.prod(s[0] for s in report["vocoder_upsample_strides"]))
    # Bind the checkpoint bytes actually selected, without exporting their paths.
    report.update(namespace.get("_ab_bindings") or checkpoint_bindings(namespace, root))
    device = torch.device(namespace["device"])
    devices = list(range(torch.cuda.device_count())) if device.type == "cuda" else []
    # Runtime-local replay bundle; contains private derived tensors, never uploaded.
    torch.save({key: (value.detach().cpu() if isinstance(value, torch.Tensor) else value)
                for key, value in inputs.items()}, out / "private_inputs.pt")
    # The short prefix isolates a single chunk; full case preserves every token.
    cases = {"full": base["pred_semantic"]}
    # One case per fresh pair: never carry state from a prefix into the full case.
    original_forward = vocoder.forward
    original_inference = model.cfm.inference
    saved_inference = model.cfm.__dict__.get("inference")
    saved_forward = vocoder.__dict__.get("forward")
    all_traces = {}
    from tools.v5_ab_controls import initial_noise_override, runtime_profile, buffer_snapshot, changed_buffers
    try:
        for case, semantic in cases.items():
            traces, waves = {}, {}
            for route in routes or ("webui_A", "webui_A_repeat", "official_B"):
                trace = {}
                cfm_calls = []
                ns = dict(base, pred_semantic=semantic.clone())
                def forward(mel, *args, **kwargs):
                    record(trace, "vocoder_input", mel)
                    return original_forward(mel, *args, **kwargs)
                vocoder.forward = forward
                def infer(mu, lengths, prompt_mel, steps, *args, **kwargs):
                    record(trace, "cfm_mu", mu)
                    record(trace, "cfm_lengths", lengths)
                    record(trace, "cfm_prompt", prompt_mel)
                    effective_temperature = getattr(model.cfm, "noise_temperature", None)
                    if effective_temperature is None:
                        effective_temperature = kwargs.get("temperature", 0.875)
                    cfm_calls.append({"steps": steps, "cfg": kwargs.get("inference_cfg_rate", 1.3),
                                      "effective_temperature": effective_temperature,
                                      "initial_noise_source": "cold_B_exact_scaled_values" if frozen_noise is not None else "native_dtype_seed_1234"})
                    fixed = None
                    if frozen_noise is not None:
                        fixed = frozen_noise[case][len(cfm_calls) - 1].to(mu)
                        noise = fixed
                    else:
                        with torch.random.fork_rng(devices=devices):
                            noise = torch.randn(
                                [mu.shape[0], model.cfm.in_channels, mu.shape[1]],
                                device=mu.device, dtype=mu.dtype) * effective_temperature
                    record(trace, "cfm_noise", noise)
                    if fixed is None:
                        result = original_inference(mu, lengths, prompt_mel, steps, *args, **kwargs)
                    else:
                        with initial_noise_override(model.cfm, fixed):
                            result = model.cfm.inference(mu, lengths, prompt_mel, steps, *args, **kwargs)
                    record(trace, "cfm_output", result)
                    return result
                model.cfm.inference = infer
                # Capture helper output and denormalized input independently.
                helper = ns["synthesize_v5_mel"]
                denorm = ns["denorm_spec"]
                def synth(*args, **kwargs):
                    value = helper(*args, **kwargs)
                    record(trace, "normalized_mel", value)
                    return value
                def denormalize(value):
                    result = denorm(value)
                    record(trace, "denorm_mel", result)
                    return result
                ns["synthesize_v5_mel"] = synth
                ns["denorm_spec"] = denormalize
                # Restore identical CPU and CUDA RNG state around each execution.
                with torch.random.fork_rng(devices=devices), torch.no_grad():
                    torch.manual_seed(1234)
                    profile = runtime_profile(model, vocoder)
                    buffer_state = buffer_snapshot(model), buffer_snapshot(vocoder)
                    parameter_versions = {n: p._version for n, p in model.named_parameters()}
                    vocoder_versions = {n: p._version for n, p in vocoder.named_parameters()}
                    if route.startswith("webui"):
                        fn = compile_webui_acoustic((root / "GPT_SoVITS/inference_webui.py").read_text(encoding="utf-8"), ns, trace)
                        audio = fn()
                    else:
                        ns = _engine_globals(engine_source, ns)
                        official_denorm = ns["denorm_spec"]
                        def official_denormalize(value):
                            result = official_denorm(value)
                            record(trace, "denorm_mel", result)
                            return result
                        ns["denorm_spec"] = official_denormalize
                        methods = compile_engine(engine_source, ns, trace)
                        cfg = types.SimpleNamespace(
                            version=base["model_version"], device=device,
                            sampling_rate=base["hps"].data.sampling_rate,
                            filter_length=base["hps"].data.filter_length,
                            hop_length=base["hps"].data.hop_length,
                            win_length=base["hps"].data.win_length,
                            is_half=base["is_half"],
                        )
                        engine = types.SimpleNamespace(
                            configs=cfg, precision=base["dtype"], vits_model=model,
                            vocoder=vocoder, is_v2pro=False,
                            prompt_cache={"prompt_semantic": base["prompt"].reshape(-1).clone(),
                                          "phones": list(base["phones1"])})
                        # Execute the pinned upstream API exactly. At UPSTREAM
                        # 6d51ffc, _get_ref_spec(self, path) caches raw_audio/raw_sr
                        # unconditionally and returns (spec, optional_audio).
                        spec = methods["_get_ref_spec"](engine, base["ref_wav_path"])
                        engine.prompt_cache["refer_spec"] = [spec]
                        audio = methods["using_vocoder_synthesis"](
                            engine, semantic.clone(),
                            torch.tensor(base["phones2"], device=device).unsqueeze(0),
                            speed=base["speed"], sample_steps=base["sample_steps"], cfg_rate=base["cfg_rate"])
                record(trace, "raw_audio", audio)
                # Different assignment counts are not different pipeline stages.
                # Compare the final mono/resampled reference passed to mel extraction.
                trace["ref_audio"] = trace.get("ref_audio", [])[-1:]
                required = {"fea_ref", "ge", "fea_todo", "decode_ge", "mel2",
                            "cfm_mu", "cfm_lengths", "cfm_prompt", "cfm_noise", "cfm_output",
                            "normalized_mel", "denorm_mel", "vocoder_input", "raw_audio"}
                missing = sorted(key for key in required if not trace.get(key))
                if missing:
                    raise ValueError(f"Incomplete {route} trace: {missing}")
                waves[route] = audio.detach().float().cpu()
                traces[route] = trace
                trace["_settings"] = cfm_calls
                trace["_runtime"] = {"before": profile, "after": runtime_profile(model, vocoder),
                                      "changed_s2_buffers": changed_buffers(model, buffer_state[0]),
                                      "changed_vocoder_buffers": changed_buffers(vocoder, buffer_state[1])}
                trace["_parameter_mutations"] = [n for n, p in model.named_parameters()
                                                  if p._version != parameter_versions[n]]
                trace["_vocoder_parameter_mutations"] = [n for n, p in vocoder.named_parameters()
                                                          if p._version != vocoder_versions[n]]
                # Same output gain on both routes: raw FLOAT WAV, no normalization.
                save_float_wav(out / f"{case}_{route}.wav", waves[route])
            # Browser-compatible listening copies use ONE common gain across A/B.
            # Keep raw FLOAT WAVs separately for exact, unclipped measurements.
            from scipy.io.wavfile import write
            import numpy as np
            peaks = [float(w.abs().max()) for w in waves.values()]
            listening_gain = 0.98 / max(1., *peaks)
            for route, wave in waves.items():
                pcm = (wave.numpy() * listening_gain).clip(-1., 1.)
                write(str(out / f"{case}_{route}_listen.wav"), 48000,
                      (pcm * 32767).astype(np.int16))
            if "reference_vocoder_probe" not in report:
                # Remove CFM entirely: reconstruct the actual reference mel.
                raw_reference_mel = next(iter(traces.values()))["mel2"][0]
                normalized_reference = base["norm_spec"](raw_reference_mel)
                reconstructed_mel = base["denorm_spec"](normalized_reference).to(base["dtype"])
                with torch.inference_mode():
                    reconstructed_audio = original_forward(reconstructed_mel)[0, 0].float().cpu()
                save_float_wav(out / "reference_vocoder_roundtrip.wav", reconstructed_audio)
                gain = .98 / max(1., float(reconstructed_audio.abs().max()))
                write(str(out / "reference_vocoder_roundtrip_listen.wav"), 48000,
                      (reconstructed_audio.numpy() * gain * 32767).clip(-32767, 32767).astype(np.int16))
                report["reference_vocoder_probe"] = {
                    "bypasses_cfm": True,
                    "normalization_roundtrip_max_abs": float(
                        (base["denorm_spec"](normalized_reference) - raw_reference_mel).abs().max()),
                    "input": tensor_stats(reconstructed_mel),
                    "output": tensor_stats(reconstructed_audio),
                    "listening_gain": gain,
                }
            report["cases"][case] = {
                "repeat_control": (compare_traces(traces["webui_A"], traces["webui_A_repeat"])
                                   if "webui_A_repeat" in traces else None),
                "ab": (compare_traces(traces["webui_A"], traces["official_B"])
                       if "webui_A" in traces and "official_B" in traces else None),
                "semantic_tokens": semantic.shape[-1],
                "listening_common_gain": listening_gain,
                "cfm_call_settings": {r: t["_settings"] for r, t in traces.items()},
                "route_runtime": {r: t["_runtime"] for r, t in traces.items()},
                "parameter_mutations": {r: t["_parameter_mutations"] for r, t in traces.items()},
                "vocoder_parameter_mutations": {r: t["_vocoder_parameter_mutations"] for r, t in traces.items()},
                "normalized_mel": {r: [tensor_stats(x) for x in t.get("normalized_mel", [])]
                                   for r, t in traces.items()},
            }
            if namespace.get("_ab_retain_traces"):
                all_traces[case] = traces
            del traces, waves
    except Exception as exc:
        report["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        raise
    finally:
        if saved_forward is None:
            del vocoder.forward
        else:
            vocoder.forward = saved_forward
        if saved_inference is None:
            del model.cfm.inference
        else:
            model.cfm.inference = saved_inference
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(f"[V5 acoustic A/B] report and private listening WAV: {out}", flush=True)
    if namespace.get("_ab_retain_traces"):
        return report, all_traces, out
    return report


def run_from_webui(namespace, inputs, output_root):
    """Order/state and FP32 controls, fresh checkpoint-backed pairs per sequence."""
    import datetime
    import gc
    import torch
    from tools.v5_ab_controls import FreshAcousticFactory, precision_context, runtime_profile
    root = Path(__file__).resolve().parents[1]
    import os
    if os.environ.get("GPT_SOVITS_V5_AB_REQUIRE_PRETRAINED") == "1":
        expected = root / "GPT_SoVITS/pretrained_models/gsv-v5-pretrained/s2Gv5dev.pth"
        if Path(namespace["sovits_path_global"]).resolve() != expected.resolve():
            raise ValueError("진단에서는 V5dev 기본 SoVITS를 선택하세요.")
    suite = Path(output_root) / datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    suite.mkdir(parents=True, exist_ok=False)
    ns = dict(namespace, _ab_retain_traces=True)
    ns["_ab_sources"] = official_sources(root)
    ns["_ab_bindings"] = checkpoint_bindings(namespace, root)
    original_runtime = runtime_profile(namespace["vq_model"], namespace["hifigan_model"])
    factory = namespace.get("_ab_factory") or FreshAcousticFactory(namespace, root)
    results, captures, folders = {}, {}, {}
    report = {"suite": "v5_acoustic_order_and_precision_v2", "completed": False,
              "listening_performed": False,
              "baseline": "a43dfd1", "sequences": {}, "comparisons": {}, "listening_files": [],
              "checkpoints": ns["_ab_bindings"],
              "sample_steps": inputs["sample_steps"], "cfg": inputs["cfg_rate"],
              "semantic": tensor_stats(inputs["pred_semantic"]),
              "original_loaded_runtime_before_offload": original_runtime,
              "fp32_noise_policy": "exact scaled initial noise from cold_B; not merely the same seed",
              "state_policy": "fresh pair per sequence; shared pair within forward/reverse sequences"}
    pair = None
    cpu_rng = torch.random.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if str(namespace["device"]).startswith("cuda") else None
    try:
        plans = [
            ("forward", ("webui_A", "webui_A_repeat", "official_B")),
            ("reverse", ("official_B", "webui_A")),
            ("cold_A", ("webui_A",)),
            ("cold_B", ("official_B",)),
            ("fp32", ("official_B",)),
        ]
        for sequence, routes in plans:
            print(f"[V5 acoustic A/B] {sequence}: 새 acoustic/vocoder 인스턴스 준비", flush=True)
            pair = factory(fp32=sequence == "fp32")
            active = dict(ns, vq_model=pair[0], hifigan_model=pair[1])
            fixed = None
            if sequence == "fp32":
                active.update(dtype=torch.float32, is_half=False)
                fixed = {case: trace["official_B"]["cfm_noise"]
                         for case, trace in captures["cold_B"].items()}
            with precision_context(sequence == "fp32"):
                result, trace, folder = _run_sequence(active, inputs, suite / sequence, routes, fixed)
            results[sequence], captures[sequence], folders[sequence] = result, trace, folder
            report["sequences"][sequence] = {"report": str((folder / "report.json").relative_to(suite)),
                                              "fresh_instance": True,
                                              "precision": "fp32_from_original_checkpoint" if sequence == "fp32" else "current_loaded_weights_and_dtypes",
                                              "runtime_by_route": result["cases"]["full"]["route_runtime"],
                                              "parameter_mutations": result["cases"]["full"]["parameter_mutations"],
                                              "vocoder_parameter_mutations": result["cases"]["full"]["vocoder_parameter_mutations"],
                                              "cfm_settings": result["cases"]["full"]["cfm_call_settings"]}
            for module in pair:
                module.cpu()
            pair = None
            del active
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        for case in captures["forward"]:
            f, r = captures["forward"][case], captures["reverse"][case]
            a, b = captures["cold_A"][case]["webui_A"], captures["cold_B"][case]["official_B"]
            report["comparisons"][case] = {
                "forward_A_repeat": compare_traces(f["webui_A"], f["webui_A_repeat"]),
                "forward_A_B": compare_traces(f["webui_A"], f["official_B"]),
                "reverse_B_A": compare_traces(r["official_B"], r["webui_A"]),
                "A_order_effect": compare_traces(f["webui_A"], r["webui_A"]),
                "B_order_effect": compare_traces(f["official_B"], r["official_B"]),
                "cold_A_B": compare_traces(a, b),
                "cold_current_vs_fp32_matched_noise": compare_traces(b, captures["fp32"][case]["official_B"]),
            }
            fp_noise = captures["fp32"][case]["official_B"]["cfm_noise"]
            report["comparisons"][case]["fp32_initial_noise_values_identical"] = (
                len(b["cfm_noise"]) == len(fp_noise)
                and all(torch.equal(x.float(), y.float()) for x, y in zip(b["cfm_noise"], fp_noise)))
            # Regenerate suite listening copies with ONE gain across every route/precision.
            from scipy.io.wavfile import read, write
            import numpy as np
            tracks = [(s, route, folders[s] / f"{case}_{route}.wav") for s, routes in plans for route in routes]
            tracks.extend((s, "reference_roundtrip", folders[s] / "reference_vocoder_roundtrip.wav")
                          for s in ("cold_B", "fp32"))
            samples = [(s, route, path, read(path)[1]) for s, route, path in tracks]
            gain = .98 / max(1., *(float(np.max(np.abs(x))) for _, _, _, x in samples))
            for s, route, path, audio in samples:
                target = path.with_name(path.stem + "_listen.wav")
                write(str(target), 48000, (audio * gain * 32767).clip(-32767, 32767).astype(np.int16))
                report["listening_files"].append({"sequence": s, "route": route, "case": case,
                                                   "path": str(target.relative_to(suite))})
            report["comparisons"][case]["common_listening_gain"] = gain
            for sequence in results:
                results[sequence]["cases"][case]["listening_common_gain"] = gain
                if sequence in ("cold_B", "fp32"):
                    results[sequence]["reference_vocoder_probe"]["listening_gain"] = gain
                (folders[sequence] / "report.json").write_text(
                    json.dumps(results[sequence], indent=2, allow_nan=False), encoding="utf-8")
        report["completed"] = True
    except Exception as exc:
        report["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        if pair is not None:
            for module in pair:
                module.cpu()
        if hasattr(factory, "close"):
            factory.close()
        torch.random.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        (suite / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(f"[V5 acoustic A/B SUITE] 완료: {suite}", flush=True)
    return report
