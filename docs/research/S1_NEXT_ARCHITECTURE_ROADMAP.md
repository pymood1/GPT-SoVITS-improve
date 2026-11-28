# Next architecture candidates — 2026-11-26

Status: CPU prototypes only. Frozen round 15 is the reference; no GPU
submission or checkpoint change is authorized by this document.
Existing V5dev S2/vocoder and production branch remain untouched.

## A. Function-preserving inference export (priority 1)
The round-15 T4 cached speed is 75.79 tokens/s vs original 127.40;
round 12 (SwiGLU, LayerNorm) was 109.43. Isolate the slowdown before
changing more architecture. tools/s1_inference_export.py provides two
independent switches:
- F.rms_norm(x, normalized_shape, weight, eps) + original bias.
  Bias is intentionally retained, as is exact checkpoint epsilon.
- Pack gate and up weights/biases into one 4096-wide linear projection,
  split into two 2048-wide tensors, SiLU(gate)*up, unchanged down.
  Parameter count stays unchanged. This reduces linear calls, not FLOPs.

CPU gates: random inputs, zero and near-zero RMS inputs, nonzero
affine bias, multiple sequence lengths, parameter-count equality,
two-layer static cache logits/top-1 parity and sampled output parity.
FP32 rounding can differ; native RMS need not be fused on every backend.
Do not claim speedup without real T4 measurement.
Next bounded GPU experiment (no training): original round-15 eager,
native RMS only, packed FFN only, both, with alternating order and
warmup; same 24 x <=128 teacher-forced tokens; >=.999 top-1 agreement,
24/24 logit parity, 4/4 fixed-seed sampling, same cache allocation.
Measure prefill, decode, peak memory, per-op CUDA time and end-to-end
waveform RTF separately. If native is insufficient, investigate a
Triton kernel for residual-add + FP32 RMS reduction + affine transform;
T4 compatibility and numerical parity are mandatory. Never use a
different epsilon or silently remove bias. A worthwhile first target is
>=109 tokens/s; speed improvement is a hypothesis, not a promise.

## B. Rank-adaptive modulation SwiGLU (priority 2, new candidate)
Round 12's full-width conversion preserved quality far better than
discarding neurons, but added ~25.2M parameters. The up branch started
from zero weight and unit bias, and was subsequently trained. Its actual
singular spectrum has not been measured.

Keep the 2048-neuron gate and down matrices untouched. Replace only
the up projection with U(Vx)+b, where V is r x 512 and U is 2048 x r.
Initialize via balanced truncated SVD of the learned up matrix, not
random initialization; preserve its learned bias. The new FFN is
down(SiLU(gate(x)) * (U(Vx)+b)). This is a custom research variant of
low-rank factorization, not a claim of novel published architecture.
Full rank retains the function within FP32 rounding, truncation does not.
tools/s1_rank_adaptive_ffn.py implements deterministic CPU FP64 SVD
initialization, frozen gate/down, discarded spectral energy and tests.

Theoretical parameter budget (not measured quality or latency), assuming
all 24 FFNs use the same rank and ignoring any unrelated export metadata:
- Dense up weights: 2048*512 = 1,048,576 per block.
- Rank 64: 163,840 per block; ~21.23M weights saved, ~81.59M total
  against the round-12 dense count, plus any unchanged norm differences.
- Rank 32: 81,920 per block; ~23.20M weights saved, ~79.62M total.
- Rank 128: 327,680 per block; ~17.30M weights saved, ~85.52M total.
Bias count is unchanged. Extra small GEMM can be slower than dense
on T4 even at lower parameter count. Measure instead of assuming.

Stage B0: inspect each layer's SVD energy and activation-weighted
reconstruction on cached training inputs; no training. Rank candidates
32/64/128/256/dense, choose per layer using held-out development gate,
not raw spectral energy alone. Keep dense if no rank passes. Avoid
tuning on fresh waveform test. Stage B1: if useful, short KD updates
only on factors, with original gate/down/attention/norm frozen. Apply
rollback if CE ratio >1.005, KL >.003 or teacher agreement <.95; stricter
selection should require no material regression from round 15. Stop
after two stale epochs, preserve best. No scratch training.
Stage B2: test full stack, cache parity, new matched audio and strict
FP32 CER. Compare original, round 12, round 15 and candidate on same
references/text/seed. Human listening before adoption. Do not merge
rank factorization with packed projection blindly: packed dense gate/up
and low-rank up are alternative kernels requiring separate measurement.

## C. Position and attention modernization (deferred)
Round 8 shows that immediately removing absolute position or swapping in
RoPE is destructive. A prospective homotopy retains original absolute
embedding and adds a continuously interpolated rotary angle on Q/K:
alpha=0 exactly original, small stages toward alpha=1; separately fade
absolute positions after rotary adaptation, not both at once. The teacher
stays frozen, use small LR and rollback, cache and prefix semantics exact.
Do not assume angle interpolation makes training stable.
For later GQA, first measure learned K/V head redundancy, group similar
heads and distill a gradual interpolation from individual to shared K/V
while keeping query heads. At 16->8/4 KV heads, theoretical KV bytes
halve/quarter; actual T4 speed and quality unknown. This is not in current
code and must not be dispatched before baseline waveform quality passes.

## Decision order
1. Finish listening-only recovery and resolve frontend; freeze round 15.
2. Native RMS / packed projection isolated benchmarks without training.
3. Rank-adaptive spectrum audit before any compression training.
4. Only consider RoPE/GQA if quality and speed gates justify it.
No claim of grokking, real waveform superiority, production integration,
or GPU performance from the CPU prototypes.
