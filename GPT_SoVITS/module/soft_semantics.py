"""Research-only expected codebook embeddings; no change to default inference."""
import math
import torch


def decode_soft_single_layer(quantizer, codes, probabilities, mix=1.0):
    """Decode [1,B,T] hard IDs and [B,T,V] codebook probabilities.

    Only single-layer RVQ is supported. The vocabulary excludes EOS and must
    match the codebook exactly. Preserve the actual layer's output projection.
    """
    mix = float(mix)
    if not math.isfinite(mix) or not 0 <= mix <= 1:
        raise ValueError("mix must be in [0, 1]")
    if codes.ndim != 3 or codes.shape[0] != 1 or len(quantizer.vq.layers) != 1:
        raise ValueError("Expected single-layer RVQ [1,B,T] codes")
    layer = quantizer.vq.layers[0]
    book = layer.codebook
    if probabilities.ndim != 3 or probabilities.shape[:2] != codes.shape[1:] or probabilities.shape[-1] != book.shape[0]:
        raise ValueError("Probability shape or vocabulary mismatch")
    if codes.dtype not in (torch.int32, torch.int64) or (codes < 0).any() or (codes >= book.shape[0]).any():
        raise ValueError("Invalid hard token IDs")
    if probabilities.device != book.device or codes.device != book.device:
        raise ValueError("All semantic inputs must share the codebook device")
    p = probabilities.float()
    if not torch.isfinite(p).all() or (p < 0).any() or not torch.allclose(p.sum(-1), torch.ones_like(p[..., 0]), atol=1e-4, rtol=1e-4):
        raise ValueError("Invalid normalized probabilities")
    hard = quantizer.decode(codes)
    if mix == 0:
        return hard
    soft = (p @ book.float()).to(book.dtype)
    soft = layer.project_out(soft).transpose(1, 2).contiguous()
    if soft.shape != hard.shape:
        raise ValueError("Soft decode shape mismatch")
    return torch.lerp(hard, soft.to(hard.dtype), mix)
