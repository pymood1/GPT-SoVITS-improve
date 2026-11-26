"""Opt-in V5 research: fuse existing style codes using speaker-verification similarity.

The reference speaker embedding is a weighting signal, not a learned V5 SV adapter.
Keep auxiliary conditioning references disjoint from scoring references.
"""
import math


def style_weights(similarities, mode, temperature=.08):
    if not similarities or any(not math.isfinite(v) or not -1.001 <= v <= 1.001 for v in similarities):
        raise ValueError("Invalid speaker similarity")
    if mode == "uniform":
        return [1 / len(similarities)] * len(similarities)
    if mode != "sv" or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Invalid style weighting mode")
    top = max(similarities)
    weights = [math.exp((v - top) / temperature) for v in similarities]
    total = sum(weights)
    return [w / total for w in weights]


def reference_groups(choose, folder, *, replication=False):
    """Allocate disjoint groups; optionally exclude all first-round references."""
    used = {r[0] for r in choose(folder, count=3)}
    for _ in range(2 if replication else 1):
        groups = []
        for _ in range(3):
            group = choose(folder, count=3, exclude=used)
            paths = {r[0] for r in group}
            if len(group) != 3 or len(paths) != 3 or paths & used:
                raise ValueError("Reference groups must be disjoint")
            groups.append(group)
            used |= paths
    return tuple(groups)


def speaker_weights(prompts, auxiliary):
    import torch
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    from sv import SV
    sv = SV("cpu", False)
    def embedding(path):
        audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        if sr != 16000:
            g = math.gcd(int(sr), 16000)
            audio = resample_poly(audio, 16000 // g, int(sr) // g)
        with torch.inference_mode():
            vec = sv.compute_embedding3(torch.from_numpy(np.ascontiguousarray(audio)).float().unsqueeze(0)).flatten().float()
            if not torch.isfinite(vec).all() or vec.norm() <= 0:
                raise ValueError("Invalid reference speaker embedding")
            return torch.nn.functional.normalize(vec, dim=0)
    aux = [embedding(r[0]) for r in auxiliary]
    result = []
    for ref in prompts:
        v = embedding(ref[0])
        sims = [float(torch.dot(v, a)) for a in aux]
        result.append(style_weights(sims, "sv"))
    return result


def extract_style(tts, entry):
    import torch
    spec = entry[0] if isinstance(entry, tuple) else entry
    spec = spec.to(device=tts.configs.device, dtype=tts.precision)
    mask = torch.ones_like(spec[:, :1, :])
    with torch.no_grad():
        return tts.vits_model.ref_enc(spec[:, :704] * mask, mask)


def blend_style(primary, auxiliary, weights, mix=.5):
    import torch
    if len(auxiliary) != len(weights) or not auxiliary or not 0 <= mix <= 1:
        raise ValueError("Invalid style blend")
    if not all(math.isfinite(w) and w >= 0 for w in weights) or abs(sum(weights) - 1) > 1e-6:
        raise ValueError("Invalid style weights")
    if any(a.shape != primary.shape or not torch.isfinite(a).all() for a in auxiliary) or not torch.isfinite(primary).all():
        raise ValueError("Incompatible reference style tensor")
    if mix == 0:
        return primary
    pooled = sum(a.float() * w for a, w in zip(auxiliary, weights))
    return (primary.float() * (1 - mix) + pooled * mix).to(primary.dtype)
