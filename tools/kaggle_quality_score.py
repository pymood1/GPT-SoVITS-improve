"""Isolated scoring subprocess: GPU first, CPU parity gate, no public raw text."""
import gc
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "GPT_SoVITS")]
from tools.kaggle_quality_listening import character_errors, normalized_text, progress


def calibration_indices(samples):
    """One deterministic anchor per candidate, plus last sample."""
    first = {}
    for i, sample in enumerate(samples):
        first.setdefault(sample["label"], i)
    return sorted(set(first.values()) | ({len(samples) - 1} if samples else set()))


def equivalent(stage, a, b):
    if stage == "asr":
        return a == b
    if a.keys() != b.keys():
        return False
    return all(math.isfinite(a[k]) and math.isfinite(b[k]) and abs(a[k] - b[k]) <= 1e-4 for k in a)


def make_scorer(stage, device, request):
    import torch
    if stage == "asr":
        from faster_whisper import WhisperModel
        asr_root = ROOT / "tools/asr/models"
        local = next((p for p in (asr_root / ("faster-whisper-" + name) for name in
                     ("large-v3-turbo", "large-v3", "medium", "small"))
                     if (p / "model.bin").is_file() and (p / "config.json").is_file()), None)
        if local is None:
            raise FileNotFoundError("Cached ASR not found")
        compute = "int8_float32" if device == "cuda" else "int8"
        model = WhisperModel(str(local), device=device, compute_type=compute,
                             cpu_threads=request["cpu_threads"], num_workers=1)
        def score(sample):
            segments, _ = model.transcribe(sample["path"], language="ko", beam_size=5,
                                           condition_on_previous_text=False, vad_filter=False)
            actual = "".join(seg.text for seg in segments)
            errors, count = character_errors(request["texts"][sample["text_index"] - 1], actual)
            # Normalized text is used for the private parity check only.
            return {"char_errors": errors, "char_count": count, "cer": errors / max(1, count),
                    "_normalized": normalized_text(actual)}
        return score
    if stage != "speaker":
        raise ValueError("Unknown score stage")
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    from sv import SV
    sv = SV(device, False)
    def embedding(path):
        audio, sr = sf.read(path, dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        g = math.gcd(int(sr), 16000)
        if sr != 16000:
            audio = resample_poly(audio, 16000 // g, int(sr) // g)
        wav = torch.from_numpy(np.ascontiguousarray(audio)).float().unsqueeze(0).to(device)
        with torch.inference_mode():
            return sv.compute_embedding3(wav).flatten().float()
    refs = [embedding(path) for path in request["references"]]
    heldout = [embedding(path) for path in request.get("heldout_references", [])]
    def score(sample):
        ref = refs[sample["reference_index"] - 1]
        emb = embedding(sample["path"])
        def cosine(other):
            value = float(torch.nn.functional.cosine_similarity(other, emb, dim=0).item())
            if not math.isfinite(value):
                raise ValueError("Nonfinite speaker score")
            return value
        result = {"speaker_cosine": cosine(ref)}
        if heldout:
            result["speaker_holdout_cosine"] = sum(cosine(r) for r in heldout) / len(heldout)
        return result
    return score


def run(stage, device, request):
    import torch
    torch.set_num_threads(request["cpu_threads"])
    started = time.monotonic()
    samples = request["samples"]
    progress(stage, "all", 0, len(samples))
    score = make_scorer(stage, device, request)
    anchors = {}
    calibration_seconds = 0.0
    cpu_calibration_seconds = gpu_calibration_seconds = 0.0
    if device == "cuda":
        t = time.monotonic()
        cpu = make_scorer(stage, "cpu", request)
        for i in calibration_indices(samples):
            measured = time.monotonic()
            a = score(samples[i])
            gpu_calibration_seconds += time.monotonic() - measured
            measured = time.monotonic()
            b = cpu(samples[i])
            cpu_calibration_seconds += time.monotonic() - measured
            if not equivalent(stage, a, b):
                raise RuntimeError("CPU/GPU parity gate failed")
            anchors[i] = a
        del cpu
        gc.collect()
        calibration_seconds = time.monotonic() - t
    results = []
    for i, sample in enumerate(samples):
        result = anchors[i] if i in anchors else score(sample)
        result.pop("_normalized", None)
        results.append(result)
        progress(stage, "all", i + 1, len(samples))
    return {"scores": results, "backend": ("cuda_int8_float32" if device == "cuda" else "cpu_int8")
            if stage == "asr" else ("cuda_fp32" if device == "cuda" else "cpu_fp32"),
            "seconds": round(time.monotonic() - started, 3),
            "calibration_seconds": round(calibration_seconds, 3),
            "calibration_count": len(anchors),
            "cpu_calibration_seconds": round(cpu_calibration_seconds, 3),
            "gpu_calibration_seconds": round(gpu_calibration_seconds, 3)}


if __name__ == "__main__":
    source, stage, device, target = sys.argv[1:]
    request = json.loads(Path(source).read_text(encoding="utf-8"))
    result = run(stage, device, request)
    path = Path(target)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, allow_nan=False), encoding="utf-8")
    tmp.replace(path)
