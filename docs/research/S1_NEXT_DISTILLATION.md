# S1 Next — independent architecture research (2026-11-26)

This branch is based on d08d895199db1fe8f5894f0775c1a99f103319ac.
The existing V5dev deployment, PR #48 and current Kaggle request are not modified.
No automatic GPU dispatch without an explicit pilot request; no new public audio uploads, no production integration.

## Hypothesis and implementation
S1 Next uses pre-RMSNorm, RoPE, SDPA MHA, SwiGLU and final RMSNorm.
The 512-wide, 24-block, 16-head example is derived from the repository training
configuration; actual dimensions are read from the trusted teacher checkpoint.
For width 512, SwiGLU uses 1,344 intermediate channels (nearest multiple of
64 to 8/3 * 512), versus 2,048 for the two-matrix ReLU MLP.
This controls the dense FFN weight budget. MHA is retained to isolate quality.
Existing codebook IDs, EOS, phoneme IDs and BERT dimensions are unchanged.
Full text-prefix attention and causal audio attention are essential.

The model copies shape-compatible embeddings, BERT projection, readout and
Q/K/V/O matrices. This is NOT function-preserving: changed normalization,
positions and FFNs require distillation. Use random-init controls.
Initial implementation runs unpadded examples, FP32, single-device and AdamW with token-weighted gradient accumulation, warmup and decay. RoPE terms are shared across blocks. Optional content-addressed exact FP32 teacher-logit cache avoids repeat teacher passes.
KV-cache parity is tested. EOS is never forwarded to the acoustic codebook.

## Distillation
Use a trusted local s1v3.ckpt and already-preprocessed, separately partitioned
corpora compatible with that teacher's tokenizer, BERT and codebook.
Each folder needs 2-name2text.txt, 6-name2semantic.tsv and 3-bert/.
Do not use the personal 30-minute corpus to claim universal pretraining.
Prefer speaker-disjoint and recording-disjoint splits, with audio hashes;
the generic trainer enforces distinct IDs and directories; the FLEURS pilot preparer also excludes exact duplicate audio and normalized text and uses the official splits.
Human-curated provenance/permission and split manifests are still required.
Teacher is frozen. Teacher logits and CE targets use exactly the same prefix
and prior ground-truth audio tokens. EOS position is included.
L = (1-alpha) CE + alpha * T^2 KL(teacher || student).
Initial alpha=.5, T=2, AdamW 1e-4, gradient clip 1.
These are starting points, not established optima.

Example (only after runtime/asset preparation, Kaggle or local):
```bash
python tools/train_s1_next.py --teacher /path/to/s1v3.ckpt \
  --train /path/to/train-preprocessed --validation /path/to/val-preprocessed \
  --output /kaggle/working/s1-next-research --max-steps 1000
```
Use --alpha 0 for CE-only control and --random-init for transfer ablation.
The saved best checkpoint is an experimental format, selected by held-out
teacher-forced CE; it is NOT a drop-in production .ckpt.
No resume, DDP, rollout KD, or full S2 integration yet.

## Experiment gates
1. CPU structural tests: prefix mask, causality, cache parity, EOS exclusion,
   post-token hidden alignment, gradient isolation, partial transfer, bridge.
2. Real frozen teacher load; verify checkpoint dimension and output alignment;
   real single-GPU train/eval pilot, log initial and final CE/KL, memory and time.
3. Equal token-budget ablation: old S1 fine-tuning control, Next CE-only,
   Next KD, Next random-init KD. Compare multiple seeds, speaker-disjoint data.
4. Full free-running synthesis with unchanged V5dev: CER, speaker similarity,
   duration/repetition/EOS failures, RTF, listening (blinded if possible).
   Teacher-forced CE alone is not sufficient; include prompts and long text.
5. Only after Next matches baseline, study student-generated prefix
   distillation and intermediate layer supervision. Neither is implemented.

## S1-S2 bridge
The optional SemanticResidualBridge module starts as exact hard-embedding
identity (zero scalar gate). It consumes post-token S1 hidden state, not
the predictor state for the next token. At 25 Hz, project 512 -> 128 -> 768,
add to decoded VQ embeddings before upsampling, preserving old token path.
The isolated FrozenV5Bridge and bridge pilot trainer now use a differentiable path through frozen S2; production integration is not implemented. Test the bridge
separately with a fixed S1, frozen S2, target acoustic supervision via CFM loss,
then compare constrained S2 adaptation, adapter regularization and gate
strength. Use held-out real audio; compare hard-only baseline, bridge-zero,
trained bridge, and shuffled-hidden controls. Distillation alone cannot
guarantee the student knows more acoustic detail than the teacher.

## Efficiency review and bounded real pilot
The production V5dev deployment is unchanged. Public FLEURS (google/fleurs,
CC BY 4.0, revision 3dfbbb3b3cfb7d48550c24defda9876bbb73f8ac, Korean) is used
with official train/validation splits, exact audio/text duplicate exclusion,
64/16 utterances and teacher-compatible BERT. The preparer checks complete
canonical preprocessing instead of silently accepting partial features.
The original teacher's config determines whether multilingual BERT is enabled;
otherwise zero features for Korean are intentional, not an accidental fallback.

GPU0: full-width S1 Next, teacher-forced KD, three epochs/40 optimizer steps/
40k token budget (may exceed by the final indivisible utterance), exact FP32
teacher-cache, token-weighted accumulation, held-out CE/KL/accuracy history.
GPU1: original frozen S1 states and frozen base V5dev, 8/4 examples,
two epochs, paired CFM loss under common RNG seeds. This separates bridge
feasibility from a not-yet-trained S1 Next. Same-utterance reconstruction
with first-second reference is not proof of free-running voice-clone quality.
Cache is on tmp and can be large; no mixed precision is used in this baseline.
KV preallocation, length-bucket batching, DDP, resume, on-policy KD, and
full waveform evaluation are deferred until the first real pilot passes.
Both workers keep independent private logs and only aggregate results are
recovered in Actions. The request workflow runs CPU gates before submission.

## Round 2: matched ablations (2026-11-26)
Round 1 actual T4x2 result: 40,040 training tokens; validation CE 8.76747
to 5.23491 (teacher 3.72837); bridge held-out CFM 0.33241538 to
0.33241297, too small to establish improved quality. The next round uses
256/32 Korean FLEURS utterances, a fresh initialization and identical
seed/order/token budget for KD (alpha .5) versus CE-only (alpha 0).
Each arm has up to 8 epochs, 400 optimizer steps and 400k tokens; the
first stopping condition wins. Cache is shared between sequential arms.
Best-checkpoint greedy rollout on four held-out examples reports cap
rate, repetition and length ratio; it uses a same-utterance semantic
prefix and full text, not an independent-reference waveform evaluation.

The bridge compares the original zero-gate initialization to a zero final
projection with unit fixed scale. Both begin at exact hard-token identity,
but the latter immediately updates the final projection. Both use 24/8
examples, four epochs, identical training noise and evaluation seeds.
Reversed temporal hidden states serve as a negative control, not a
different-speaker or independent-reference control. Paired win rate
and CFM loss are reported. Checkpoints are saved only on held-out gain.
Neither CE nor CFM loss establishes naturalness; full synthesis and an
independent test set remain necessary. Production model remains unchanged.

## Round 3: exposure-bias and conditioning controls
Round 2 (Kaggle 45739171412): matched 400,120-token KD versus CE-only
validation CE 4.4872 versus 4.6696 (teacher 3.5772); greedy cap rates
75% versus 100%; maximum runs 338 versus 320. Zero-projection bridge
CFM 0.274237 versus hard baseline 0.304380; reversal 0.274175, so
the reduction does not establish time-aligned hidden information.

Round 3 mounts only the exact completed private round-2 kernel and checks
the result provenance and checkpoint format/config. Both S1 arms start from
the same KD checkpoint (optimizer restarted), same 384 training examples,
32 validation, and 24 untouched official test examples. Each has up to
600k supervised tokens, 600 updates, 8 epochs. The trajectory arm adds
teacher KL on four sampled student tokens after a ground-truth prefix
every fourth eligible example (weight .5); the baseline continues standard
KD. The supervised token budget is matched, not the compute budget.
No CE target is imposed on off-reference student trajectories.
The selected validation-CE checkpoint is evaluated on the official test
split; six greedy and sampled rollout trials (recent-token penalty 1.35)
use the same stop rule. The baseline arm also evaluates the actual teacher
using its full forward with the shared sampler, not legacy EOS shortcuts.

Bridge arms all use zero projection and matched training noise, compare
aligned teacher hidden states, prompt-only mean (first 25 tokens), and a
constant training-set prompt mean. 32/8/12 train/val/test examples, four
epochs, selection by validation CFM then untouched test evaluation.
Reversed hidden states remain a diagnostic for the aligned arm. A global
mean gain suggests a static correction rather than transferred temporal
information. All reconstruction uses true acoustic codes; no waveform
naturalness or voice clone claim is made. Production remains untouched.

## Round 4: waveform gate (2026-11-26)
Round 3 completed: trajectory KD test CE 4.56138, continued KD 4.58072,
teacher 3.65175. Six sampled rollouts for each student finished below cap;
maximum repeat 3 versus 7. Matched teacher sampled repeat 3, greedy cap
6/6. These tiny diagnostics do not establish TTS quality.
Independent bridge test CFM: hard .300291, aligned .261579, prompt mean
.263640, global mean .262926, reversed aligned .264890.
Static correction accounts for most of the observed improvement.

Round 4 pins the completed round-3 kernel and evaluates 24 attempts,
two previously unused Korean FLEURS references x three distinct target
transcripts x four arms: teacher, trajectory student, teacher+aligned bridge,
teacher+global mean bridge. The first 384 training, 32 validation and 24
previously tested examples are excluded using the same pinned selection.
Both GPUs process one reference. Teacher-derived tokens are reused exactly
between teacher and both bridge arms; aligned states are recomputed from
the same teacher and generated prefix. S2 and vocoder frozen, 32 steps,
CFG 1.30, common acoustic seed, matched top-k 15, recent-token penalty 1.35.
An encoder parity gate compares bridge-disabled features with decode_encp.
Report synthesis failures, cap rates, duration, repetitions, CER, SV cosine
and private listening samples. Timings are diagnostics only: teacher uses
full forward, student cached generation, bridge arms reuse semantic tokens.
No training, production integration or deployment in this round.

## Round 5: repair after waveform failure
Round 4: 24/24 waveform files, teacher CER 2.27%, SV .90879, duration
6.39s; student CER 96.02%, SV .53751, duration .56s. Aligned/global
bridge CER 31.25%/55.68%, SV .84394/.71828. None is deployable.
The student used same-utterance prefixes during earlier training, not
independent references; this is a plausible distribution mismatch, not
a proven sole cause. Round 5 generates 96/12 teacher trajectories from
distinct reference/target utterances in the prior train/validation splits
using the actual TTS frontend and prompt extractor. Target-only KD/CE
excludes reference tokens, includes EOS. Starting checkpoint is round-3
trajectory KD; 12 epochs, LR 5e-5; select by validation KD and free-running
length diagnostics. In parallel, sweep aligned residual scale .05/.15/.35
and global mean .10 with frozen teacher tokens. Compare 36 synthesis
attempts across two new references and three new target transcripts,
excluding first 29 test selections. Scores are exploratory; another
fresh confirmation and listening are required before deployment.

### Round 5 retry: confirmed frontend resource error
The first run (45747069668) completed 20/30 teacher/bridge syntheses, no
student training. The CPU-only diagnostic run 45754800367 recovered the
same LookupError in both workers: NLTK averaged_perceptron_tagger_eng
missing, from English G2P reached by mixed-language text. The four matched
successful cases had teacher CER 3.51%, SV .868737; aligned .35 CER
3.51%, SV .877907. This is preliminary and excludes two failed cases.
Retry the exact round-5 design after verifying nltk.pos_tag and explicitly
setting NLTK_DATA. Stream bounded numerical worker progress and return
sanitized exception signatures in aggregate output. No model or data
selection change, no production integration, no GPU submission by
diagnostic profile.

## Round 6: S1 Next scale-up, two GPU synchronous training
Round 5 retry (45754943705) completed 36/36 waveforms; 95/96 teacher
training trajectories, 12/12 validation. Prompt training reduced initial
EOS probability from .80694 to .01523 at epoch 4 (selected score); student
CER 93.18%, teacher 23.64%, student duration 7.42s, teacher 6.08s.
The checkpoint is undertrained; architecture potential is not established.

Round 6 starts from that selected checkpoint, not from scratch. Generate
1024 prompt/target trajectories and 24 validation trajectories, split
deterministically across two T4 GPUs. Preserve earlier 384/32 source split
and first 34 test selections; reserve six fresh test cases, expand train by
up to 256 eligible nonoverlapping source utterances. Pair count is not the
number of unique source recordings. Cached teacher decoding must agree
with full-forward logits during an eight-token parity preflight; fall back
to full-forward on mismatch. Save teacher cache shards for reuse.

Train one shared student using NCCL DDP, FP32, eight epochs, AdamW 5e-5,
four microexamples per rank per sync (global eight). Normalize gradients
by global target-token count, account for the DDP averaging factor, drop
at most one shuffled indivisible example per epoch. CPU Gloo gradient
parity gate compares the same actual S1 Next update with full-batch
single-model gradient. Rank-zero best checkpoint selected by independent
prompt validation KD/length score, with metrics and optimizer steps.
No bridge training. Compare old/new student and teacher on fresh matched
2-reference x 4-text waveform evaluation (24 attempts). Same S2/vocoder.
Two GPUs replicate weights rather than pooling memory. Actual speedup,
capacity and generalization are to be measured, not assumed.

### Subsequent training convergence policy
Keep the already submitted round-6 commit immutable. Subsequent DDP runs
should pass --until-teacher, a generous epoch ceiling and bounded session
time. Compute both teacher and student token-weighted CE on the exact
same independent-reference validation pseudo-targets, including EOS.
Candidate gate: student CE <= 1.05 * teacher CE, no short or capped
rollouts, mean length ratio 0.9–1.1, three consecutive validation checks.
Save a teacher-gate candidate and a separate latest weights checkpoint.
If time or epoch ceiling is reached, report that convergence was not
achieved; carry the checkpoint and reuse cached data next session.
This gate is necessary but not sufficient for production replacement:
compare CER, speaker similarity and listening on a fresh test corpus,
not the repeatedly consulted validation data. The current code supports
the stop policy; the next Kaggle request must explicitly enable it.

## Round 7: teacher-relative convergence
Round 6 (45760100606) succeeded on both T4s, 1019 accepted training
pairs from 442 source recordings, 24 validation pairs, 1,249,309 tokens,
1024 updates, 1154.65s training/validation and 3243.69s total.
Validation CE minimum 4.6669 (epoch 5), KD minimum 2.5649 (epoch 6),
both worsen by epoch 8. Selected checkpoint epoch 3. On the same eight
independent test syntheses, teacher/prior/new CER 7.53/95.89/95.55%,
speaker cosine .7904/.7404/.6873. No deployment.
Round 7 starts from selected round-6 weights, replays the 1019 cached
training trajectories and fixed 24 validation trajectories. Reconstruct
the exact 442-source selection and generate 1024 new reversed reference/
target pair attempts with disjoint indices; these are novel pairings,
not additional source recordings. Reserve another six test rows after
excluding round-6 data. Two-GPU FP32 DDP, AdamW reset at 3e-5, reduce
LR by half on validation CE+KD plateau (patience 2, floor 1e-6).
Up to 60 epochs or 5400s training budget checked at epoch boundaries;
stop earlier only on three consecutive teacher-relative gate passes.
Save best and latest weights after epochs, evaluate the gate candidate
if attained, otherwise validation-best. Cached original trajectories
remain in round-6 output, new trajectories in round-7 output. CE is on
teacher-generated labels, not ground-truth audio; fresh matched waveform
evaluation remains essential. A budget stop is not convergence.

## Rounds 8–15: measured architecture-preserving transfer (2026-11-26)

This section supersedes the earlier proposed architecture and convergence
policy where they conflict. All reported measurements are from the completed
or partially completed Kaggle runs, not from theoretical estimates.
The 24 repeated development trajectories are for model selection, not a
fresh generalization test. Teacher-forced top-1 agreement is not speech CER.

| Round | Actions run | Actual result |
|---|---|---|
| 7 | 45772036033 | Cancelled to conserve GPU quota; large architecture replacement remained far below teacher. |
| 8 | 45785591042 | Transfer audit: identity exact; direct RoPE 6.52%, direct post-RMS 24.91%, pre-LN 1.61% top-1 agreement. Abrupt surgery is destructive. |
| 9 | 45794687129 | First reduced-width SwiGLU block 97.99% agreement after warm-start distillation. |
| 10 | 45806410140 | Six reduced-width replacements; 90.23% agreement. |
| 11 | 45812067222 | Seven reduced-width replacements, 90.45%; further simultaneous compression did not pass. |
| 12 | 45815400064 | All 24 full-width SwiGLU FFNs; 97.81% agreement, CE 2.844995, KL .0003843. |
| 13 | 45825615128 | Static KV cache parity 24/24, sampled equality 4/4; original 115.58 vs round 12 103.71 tokens/s. |
| 14 | 45828829274 | All 48 block norms fully affine RMS; 95.52% agreement, CE 2.847964, KL .0014566. |
| 15 | 45834021558 | Frozen attention/position/readout; 512 replay pairs, 8 epochs, 623912 tokens; selected epoch 6, 97.47% agreement, CE 2.845899, KL .00077238. |

Round 12 waveform evaluation produced 12/12 samples (3 arms, 2 refs,
2 texts). Round 14 recovered strict FP32 GPU large-v3 ASR on those same
saved samples, no INT8/fp16 fallback: original and round 12 both 2/108
characters erroneous (1.85% CER), round 11 7/108 (6.48%).
Speaker cosine .8024/.8095 for original/12, .8067 for 11.
These are tiny matched samples; do not claim equivalence beyond them.
Round 12 increased parameters from 77,606,402 to 102,821,378 (+32.49%).

Round 15 measured all four decoders on the same 24 cached development
trajectories (2842 target tokens), T4 FP32. Sum of per-example times,
not end-to-end concurrent throughput:

| Decoder | Full seconds | Cached seconds | Cached tokens/s | Cache parity |
|---|---:|---:|---:|---|
| Original | 92.470 | 22.308 | 127.40 | 24/24 |
| Round 12 | 119.971 | 25.972 | 109.43 | 24/24 |
| Round 14 | 122.253 | 38.085 | 74.62 | 24/24 |
| Round 15 | 122.115 | 37.497 | 75.79 | 24/24 |

All four also passed 4/4 matched full/cached sampling trials.
Round 15 is 40.51% lower throughput than original on this test;
the RMS implementation likely introduces extra kernel launches, but
profiling is required. Its 98% development agreement target was not met.
The lowest KL epoch was not selected because the predeclared criterion
prioritized teacher top-1 agreement. No evidence of grokking is established.

Round 15 stopped at waveform evaluation (1196.734s total).
The CPU-only observer run 45842230720 recovered 20 per-sample exceptions
(16 + 4) from English G2P's NLTK POS tagger lookup, not evidence of failed
learned weights or cache arithmetic. Earlier isolated NLTK preflight had
passed. Worker-side repair now re-prioritizes resource paths and exercises
actual English G2P after TTS initialization; per-process fallback downloads
avoid concurrent writes. CPU path tests passed on 45842553746.
Real GPU validation of the repair is pending.
The immutable listening-only run 45840975381 was submitted before the
repair and remains pinned to its original code; it may repeat that failure.
No automatic resubmission, no new training, no deployment.

Checkpoint: private Kaggle kernel pymood1/gpt-sovits-s1-next-45834021558,
s1-rms-polish/student/norm-best.pt; round 12 folded stack and round 14
norm checkpoint are separately mounted. This path is required by recovery;
actual file mounting is checked in the listening job. Never overwrite it.
The general read-only observer can inspect numbered S1 kernels without
allocating GPU; request-only commits skip redundant CPU tests.

## Frozen listening and acoustic isolation (2026-11-27 KST)
Listening recovery 45840975381 succeeded 32/32 (8 per arm), full-forward
FP32, no training, 793.344s. Original/12/14/15 mean durations
5.825/5.89/6.535/6.21s; no cap hits. User listened to four random
files (three round14, one original), reporting ambiguous reference match
and noise. Not a matched blind comparison; no inference about round15
superiority. Preserve all checkpoints.

Acoustic isolation 45849884407 completed 20/20 in 456.338s. Exact
previous references, text01, original frozen S1 and V5dev.
Reference RMS -35.055/-46.730 dBFS; true mel vocoder outputs
-35.803/-47.509 dBFS. Target real -38.676, vocoded -39.649.
Original synthesis RMS -35.395/-44.599; with reference gain
(4.006x/8x) -23.946/-27.880. This is a clear gain effect, not proof of
noise removal. The level-matched listening copies use bounded gain only.

Reference1 true mel min -11.513, oracle-token S2 min -23.905,
original GPT+S2 min -48.924, CFG0 min -13.284. Reference2 baseline
min -12.027, CFG0 -13.426. Static cache off changes summary statistics
only by rounding-scale differences; this does not prove waveform identity.
Vocoder-only reconstruction preserves RMS within ~1 dB for these three
recordings; listening and spectral analysis still required. CFM uses
Euler flow integration and velocity CFG v+g(v-neg), normalization
spec_min=-12, spec_max=2, no final clamp. Extreme lower values could
be sparse and need not be audible; don't infer a causal noise source
from extrema alone.

Follow-up request bc55172ef69608707db51f2f5d64d3ff2d1233a4:
reuse exact saved baseline mel and tokens, two reference workers, no
new S1 generation or training. Re-vocode baseline and clamp [-12,2]
for a controlled listening ablation; measure .1/1 percentile and
out-of-range fraction; compare CFG .5/.8, reference gain with CFG .8,
and gain applied only to CFM prompt mel versus only to encoder
conditioning. Use matched seed. These are research-only controls,
not a proposed production clamp without evidence. 14 cases, 1500s
session cap, no automatic retry. Prior CFG0/1.3 and real recordings
available as comparison files. No GPU precision reduction.

## Acoustic CFG v2: completed (2026-11-27 KST)
Initial request 45855318993 failed at CPU import preflight; no GPU submission.
Retry 45858321277 completed 14/14 in 414.373s. No training. Re-vocoding
saved baseline mel reproduced previous PCM exactly on both references.
Ref1 fraction of mel below -12: baseline CFG1.3 1.8766%, CFG.8 .7208%,
CFG.5 .1461%, full gain CFG.8 .0260%, mel-only gain .00487%,
encoder-only gain .01623%. Ref2 baseline .00161%, gain variants
.00323%/.01290%/.00323%; no universal improvement in this statistic.
Clamping fixed mel to [-12,2] removed excursions by definition, changed
ref1 RMS only +.161dB and ref2 negligibly; does not establish noise gain.
RMS (ref1/ref2) at CFG.8: raw -36.446/-45.316 dBFS, full gain
-25.090/-29.677, mel-only -28.644/-30.847, encoder-only
-32.791/-42.038. Mel-only normalization is an interesting candidate
because it retains original semantic/reference encoder conditioning.
However ref1/2 are only two references, one target; no fresh test.
Use existing level-matched audio for strict FP32 ASR and SV before
choosing candidates. Nine arms x two references, matched gain-only
listening copies, normalized reference WAVs for speaker comparison.
No new synthesis, training, or model modification. A lower mel-outlier
fraction is not itself evidence of lower audible noise.

## Fixed-mel tail factorial (2026-11-27 KST)
Scoring run 45910192924 completed all 18 samples with CT2 CUDA float32
and CUDA FP32 speaker scoring. Aggregate CER / speaker cosine:
raw CFG1.30 .37037/.82373; hard clamp .31481/.82655;
raw CFG.50 .35185/.50339; raw CFG.80 .29630/.74066;
gain CFG.80 .29630/.75127; mel-only gain CFG.80 .37037/.74489;
encoder-only gain CFG.80 .37037/.76241; CFG0 .48148/.30504;
prior CFG1.30 full reference gain .31481/.83617.
These are two references and a single target, not a held-out generalization
claim. Noise reduction is not established by CER, SV or mel outlier count.
Reject CFG lowering as the primary solution; retain CFG1.30 for this test.

Next: reuse the exact saved CFG1.30 raw and full-reference-gain mels
from isolation run 45849884407. Cross raw/gain with identity, hard
clamp [-12,2], and soft tail compression (identity within bounds,
tanh compression of excursions with scale 2). 12 paired waveforms;
no new CFM draw, S1 inference or training. Require identity PCM replay
within one 16-bit LSB. Score all 12 with strict FP32 CER/SV and
gain-invariant quiet-frame and high-frequency descriptive proxies.
No perceptual noise claim from those proxies. Preserve level-matched
audio for blind listening. Compare effects within each reference;
only candidates that preserve intelligibility and speaker identity
are eligible for fresh multi-reference, multi-text held-out trials.
Do not modify production main or checkpoints.

## Literature-first research policy and cross-model pilot (2026-11-27 KST)

Stage 1: evidence assessment before implementation.
- Kim et al., VITS, ICML 2021, https://proceedings.mlr.press/v139/kim21f.html : Mel reconstruction is useful supervision for the paired real waveform, not proof that an arbitrary teacher output is an optimal target.
- Mehta et al., Matcha-TTS, ICASSP 2024, https://arxiv.org/abs/2309.03199 : OT-CFM is a viable acoustic generative objective; this does not validate clipping in GPT-SoVITS.
- Yamamoto et al., Parallel WaveGAN, ICASSP 2020, https://arxiv.org/abs/1910.11480 : multi-resolution spectral objectives can supervise waveform generators, not direct cross-architecture latent alignment.
- Ren et al., FastSpeech 2, ICLR 2021, https://arxiv.org/abs/2006.04558 : teacher Mel targets can lose information; prefer real paired targets where available.
- Le et al., Voicebox, NeurIPS 2023, https://arxiv.org/abs/2306.15687 : preserve intelligibility and speaker similarity alongside acoustic criteria.
- Synthetic speech evaluation review, https://www.jstage.jst.go.jp/article/ast/advpub/0/advpub_e24.12/_article : retain blinded listening; objective proxies alone do not establish naturalness.
Stage 2: architecture and experimental logic.
V2ProPlus and V5 variants have different S2 synthesis routes but repo defaults
share S1 s1v3.ckpt. Compare native WAVs in a common 24 kHz / 80-bin
log-Mel frontend, with RMS matching, boundary-only silence trim,
constrained DTW and unaligned mean-band distance. Report duration ratio
and warping stretch. These are descriptive inter-model distances, not MCD,
ground-truth fidelity, perceptual noise, or MOS. No same-speaker ground-truth
target has been verified in the current FLEURS selection; do not claim it.
A separate verified same-speaker/different-utterance corpus is required for
paired real-target fidelity and future CFM training. No teacher-only
distillation, production change or new model training yet.

Prior fixed-mel factorial 45912359274 completed 12/12, 390.386s;
raw identity CER .37037/SV .82373; raw hard .31481/.82655;
raw soft .31481/.82602; gain identity .31481/.83617;
gain hard .31481/.84564; gain soft .31481/.84107.
Identity PCM replay was exact. Ref1 quiet HF ratio .18467 -> .13981
with gain-hard; ref2 quiet HF ratio .01764 -> .02713, thus not universal.
Reject a blanket noise reduction conclusion; compare per-pair effects.

New pilot: same saved public FLEURS references and target texts 02/03,
not used in tail tuning (but not globally unseen). Four native arms:
pretrained V2ProPlus, V5dev 32 steps/CFG1.3, V5turbo 4 steps/CFG0,
and V5dev reference gain plus denormalized Mel clamp [-12,2].
Same input, sampler settings, seed, full FP32 and S1 checkpoint; native
prompt encoding may differ, and gain can affect S1, so this is end-to-end,
not a fixed-token S2 isolation. Record checkpoint hashes and control asset
revision. 16 audio cases, strict FP32 CER/SV, noise proxies and spectral
distances to each native baseline. Private audio and text, aggregate logs.
One bounded two-T4 job, 2700s cap, mandatory CPU checks, no auto retry.
A candidate must preserve CER and speaker identity and pass blind listening;
similarity to V2ProPlus alone cannot establish superiority. Fresh
same-speaker paired corpus and further validation are prerequisites for
training decisions. 

## Real-target paired evaluation preparation (2026-11-27 KST)
Evidence: NaturalSpeech 2 (ICLR 2024), https://arxiv.org/abs/2304.09116,
uses a different utterance from the same speaker as a synthesis prompt in
its evaluation. The design motivates verified speaker pairing, not a
claim that nearest-Mel synthesis is always most natural.
Data: Zeroth Project, Lucas Jo and Wonkyum Lee, https://www.openslr.org/40/
(CC BY 4.0); https://huggingface.co/datasets/kresnik/zeroth_korean
exposes speaker_id, utterance id, text and 16 kHz audio, test split
457 utterances / 10 speakers. This is ASR read speech, not a studio
high-bandwidth TTS corpus. Pretraining overlap of public checkpoints
is unknown; never describe speakers as certainly unseen.

Implement deterministic private manifest preparation: four distinct
test speakers, one reference and two different real target utterances
each (8 paired targets), strict text and decoded-audio deduplication,
3-10s duration, 16 kHz mono, finite non-silent audio, clipping gate,
pinned Hugging Face dataset revision, PCM hash and attribution.
Targets must not enter the prompt or training. Use common <=7.6 kHz
bandwidth for real vs synthetic Mel analysis; no claim above the 8 kHz
Nyquist limit. Record per-pair CER, speaker similarity to reference AND
real target, duration, constrained DTW, real-target spectral distance,
and blinded listening. Compare inter-model distances separately.
Selection is based on metadata and waveform validity before model scores.
Code and CPU self-tests added; no real dataset download or GPU job in
this change. Await existing native cross-version pilot before requesting
additional GPU use. If successful, validate candidates on real-target
pairs; if not, diagnose its failure first.

## Native cross-version result and paired S2 factor isolation (2026-11-27 KST)
Run 45913817545 completed 16/16 in 625.109s; CT2 CUDA float32 and
CUDA FP32 SV. V2ProPlus CER .06618/SV .77267, native V5dev
.07353/.67239, gain-hard .09559/.69282, V5turbo .14706/.67438.
V5dev gain-hard reduced active 4-8kHz ratio in all four pairs
(mean .010738 -> .003382); this is not a noise removal measurement.
Mean DTW distance to V2ProPlus increased from 9.4799 to 9.6695 dB.
Reject production gain-hard adoption: CER worsened, inter-model
distance did not improve, and S1 prompt normalization confounded it.
Native versions have differing synthesis paths; V2ProPlus is a strong
control, not a proven upper bound.

Stage 1: Lee et al., BigVGAN (ICLR 2023), https://arxiv.org/abs/2206.04658,
supports periodic inductive bias and anti-aliased vocoding as plausible
ways to improve waveform synthesis. CPP clinical study,
https://doi.org/10.1044/2020_AJSLP-20-00001, supports periodicity-related
voice quality measures; clinical thresholds must not transfer to TTS.
Stage 2: user reports subtle noise/artifacts during speech, so high-frequency
energy alone is insufficient. Measure active-frame lagged autocorrelation
periodicity and residual proxy after common 16kHz resampling, with confidence
gate and coverage. Not clinical HNR/CPP, SNR, MOS, or a direct perceptual
noise score. Breathy phonation and unvoiced consonants may be normal.
Use real target, true-Mel vocoder and oracle-semantic S2 to diagnose where
quality is lost, without introducing a post-filter or training prematurely.

Pilot v6: verified Zeroth test metadata, two speakers x two different
real target utterances; eight arms per pair (32 files):
raw V5dev; raw+hard; S2 reference gain; S2 gain+hard; real recording;
real-Mel through V5 vocoder; oracle semantic tokens through V5 S2; native V2ProPlus.
First four share the original raw prompt semantic tokens, identical generated
target tokens and CFM seed, and CFG 1.30, 32 steps, FP32. Gain affects
reference spectrogram and CFM reference Mel only, not S1. Hard clamp
[-12,2] applied after generation; no additional CFM draw for clamp variants.
Oracle and real-Mel controls use real target access and must never be
reported as deployable inference. V2 control native, not shared-token.
Real-target DTW log-Mel comparison restricted to <=7600Hz and 24kHz frontend;
record duration/path stretch and per-pair CER/SV and periodicity coverage.
16kHz recordings cannot establish full-band artifact reduction. No training,
no production edits, no blind-listening claim. Test only, 3000s job cap.
This bounded first real-target pilot uses two of the available ten test
speakers; if promising, expand on distinct speakers and seeds before adoption.

## Pre-result decision protocol (2026-11-27 KST)
Evidence: Wagner et al., Speech Synthesis Evaluation — State-of-the-Art
Assessment and Suggestion for a Novel Research Program (2019),
https://people.kth.se/~ghe/pubs/pdf/wagner2019speech.pdf ;
Minixhofer et al., TTSDS2, SSW 2025,
https://www.isca-archive.org/ssw_2025/minixhofer25_ssw.pdf ;
both motivate multidimensional evaluation rather than a single proxy.
Stage 2: assess matched pairs and effects per speaker, not independent
32-sample averages (only 2 speakers, 4 targets). Preregister conservative
expansion gate for raw-hard/gain/gain-hard: aggregate CER no worse, SV
no worse, mean DTW real-target distance reduced and reduced in both
speakers, no target token cap. This is an engineering screen, not
statistical significance or a required definition of perceptual superiority.
If all fail, diagnose the oracle and true-Mel vocoder control before
another change; no forced adoption. Periodicity and coverage should be
interpreted against real target; do not maximize periodicity blindly.
Add CPU-only paired assessment and informal blinded listening export:
randomized clip names, separate answer key, artifact severity distinct
from naturalness and speaker similarity. Not a standardized MOS test.
Actual listening, fresh speakers/seeds, and higher-bandwidth targets
are still required before production adoption. No new GPU request.

## Native wideband evaluation policy (2026-11-27 KST)
Paired v6 45915504105 completed 32/32, 679.803s, verified Zeroth test pairs.
No candidate met expansion gate: raw CER .29457/SV .84857/real Mel 11.4683,
raw-hard .27132/.84835/11.6009; gain .25581/.83916/11.5179;
gain-hard .25581/.83889/11.5218. Actual target .12403/.90676/0;
true-Mel vocoder .10853/.85241/2.2050; oracle S2 .32558/.88092/8.0649;
V2ProPlus .05426/.66593/13.3620. ASR has nonzero real error.
The 16kHz target bandwidth and 2-speaker count limit conclusions.
Prioritize S2 conditioning/CFM but do not exonerate vocoder at wideband.

Evidence: official Edinburgh VCTK v0.92 https://doi.org/10.7488/ds/2645
records 110 English speakers at 96kHz/24bit in hemi-anechoic room,
distributed at 48kHz/16bit with transcripts for 109 speakers.
Candidate for acoustic mechanism validation, not Korean language performance;
use original archive or verify redistribution fidelity and license.
Korean KSS https://www.kaggle.com/bryanpark/korean-single-speaker-speech-dataset
is 44.1kHz single-speaker, noncommercial research, complementary but not strict 48kHz.
MagicHub TTS-KFSC https://magichub.com/datasets/korean-speech-corpus-for-tts/
48kHz/24bit studio, proprietary, one speaker; no automatic use.
AI Hub multispeaker TTS data https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=542
requires approval; verify raw format and cloud-use conditions first.
No claimed freely accessible native 48kHz Korean multispeaker source verified yet.
No upsampled 16/24kHz files accepted as native wideband ground truth.
Native 48kHz band audit added; upper-band energy is a review flag, not
proof of provenance or a standalone quality threshold.
Repository TTS.py mel_fn_v4 uses 32kHz, 100 Mel bins, fmax=None, FFT1280/hop320;
V5dev vocoder output 48kHz. Separate common <=15.5kHz Mel comparison from
native 48kHz upper-band waveform diagnostics; 16-24kHz is not directly
represented in that Mel frontend. V2ProPlus 32kHz output cannot be
ranked at >16kHz as if it were full-band 48kHz. Keep original real audio
unchanged, compare true-Mel-vocoder / oracle / generated paths, and
retain language-specific CER and speaker verification. No additional
GPU job submitted until corpus acquisition and provenance verified.

## Native 48 kHz VCTK paired pilot v7 (2026-11-27 KST)
Source: https://huggingface.co/datasets/thelarryproject/vctk documents
unfiltered VCTK 0.92 original 48kHz mono FLAC, CC BY 4.0,
utterance ID, speaker ID, microphone, normalized and original text,
duration and CTC mismatch. Prefer mic2 (MKH 800); mic1 has known
low-frequency noise; exclude p280/p315 technical/transcript gaps.
Select at most four pinned parquet shards, two distinct speakers,
one 3-10s reference and two real distinct targets each, CTC mismatch
<=0.3, RMS >=-45 dBFS and clipping <=0.1%; metadata and decoded
waveform checks, PCM16 WAV lossless roundtrip verified against decoded
FLAC, source and WAV hashes saved privately. Band energy is descriptive
only; no arbitrary high-band energy cutoff. No resampling of originals.
Selection not based on model performance; VCTK distributed train
partition used only for frozen evaluation, pretraining overlap unknown.

CoreaSpeech+ https://huggingface.co/datasets/Gong1212/CoreaSpeech-plus
requires approval and released sample rate is not verified; not used.
English VCTK is a controlled acoustic experiment, not proof of Korean
language generalization. Keep Korean validation separate.

Same v6 eight arms and fixed target/prompt tokens, FP32, 32 files;
English frontend and English ASR only for this run. Common spectral
comparison at 32kHz with fmax 15.5kHz, independently report native
48kHz six-band relative energies including 16-20 and 20-24kHz;
32kHz V2ProPlus has no full-band score. Compare real vs true Mel-vocoder
to identify bandwidth reconstruction limitations and real vs oracle S2
to characterize the additional acoustic modeling gap. No direct
noise-removal or perceptual claims. Retain per-speaker expansion gate
and blinded listening requirement. No training, no production changes,
one bounded 3600s GPU job after mandatory CPU selftests.

## Generalization-first corpus strategy (2026-11-27 KST)
User prioritizes high-quality diverse data, including generalization to
cases not explicitly represented during training.
Evidence: Voicebox (NeurIPS 2023), https://arxiv.org/abs/2306.15687,
demonstrates zero-shot speech synthesis at large diverse data scale;
NaturalSpeech 2 (ICLR 2024), https://arxiv.org/abs/2304.09116, uses
44k hours and unseen-speaker evaluation; NaturalSpeech 3 (ICML 2024),
https://proceedings.mlr.press/v235/ju24b.html, reports benefits from
scaling both model and data. These are not controlled proof that 48kHz
alone or unlimited data yields out-of-distribution generalization.
SPEAR-TTS (TACL 2024), https://doi.org/10.1162/TACL_A_00618, motivates
separating audio modeling from parallel text/audio supervision.
Stage 2: optimize quality AND distribution coverage, not memorization
of individual artifacts. Preserve native recording quality, accurate
transcripts, independent quality checks and speaker/style diversity.
Train/validation/test must be speaker disjoint, with exact normalized
text and decoded audio deduplication across partitions; near-duplicate
and pretraining overlap still require independent checks. Evaluate
unseen speaker, text, and held-out style/recording conditions separately;
same-speaker new text is not unseen-speaker generalization.
Add deterministic metadata-only split and equal-duration quality-ranked
vs speaker-balanced subsets at a common quality floor. Compare using
same initialization, optimizer steps, seeds and held-out data, report
quality-score distribution and speaker/text coverage. If quality differs,
stratify/match it before interpreting diversity causally. This is a
design tool, not a trained result. At least ten speakers required.
Wideband VCTK v7 remains in progress; no duplicate GPU job.

## Native 48kHz solver pilot v8 (2026-11-27 KST)
V7 run 45917728836 recovered 32/32 WAVs, all worker exits zero, FP32
ASR/SV, but failed the spectral-stage 20-second limit. Verified VCTK
48kHz mic2 data revision 0492fa00cf72afb16e753e95d0ec3439eebb07a6.
Real CER 0, true-Mel vocoder CER 0, oracle S2 .27232, custom-S1
raw .83929, gain-hard .83482, V2ProPlus .70536. Three of four custom-S1
outputs hit 512-token cap, durations 17.64-20.48s vs real 3.02-6.87s.
Not a valid comparison for acoustic improvement. Real vs true-Mel
vocoder mean relative band dB: 16-20kHz -54.97/-68.26,
20-24kHz -54.54/-89.44; no proof these differences are audible artifacts.
Source: Matcha-TTS https://arxiv.org/abs/2309.03199 and F5-TTS
https://aclanthology.org/2025.acl-long.313/ motivate testing integration
and sampling strategies, not proof of midpoint superiority in this model.
V5 CFM uses Euler and classifier-free guidance; research-only explicit
midpoint integrator at 16 steps (32 velocity evaluations) compared to
Euler32 and Euler64. Same initial noise seed, identical conditioning,
fixed prompt and target tokens per comparison, original CFG1.30; oracle
Euler32 CFG.90 additional comparison. Native S1 stopping rule replaces
custom Generator for separate native-token comparison, capped at 512.
Re-use the six existing 48kHz source files from v7, verify original PCM
hashes and failed report's stage/worker status before starting.
Eight arms x four paired targets = 32 output files. Preserve CER/SV and
band diagnostics even if duration/cap fails; explicitly exclude invalid
duration ratios (<.5 or >2), >20s or cap-hit from DTW only. Report
coverage; exclusions are failures, not improvements. Oracle conditions
diagnostic only. No training or production modifications yet.
User authorizes autonomous development without GPU-quota constraint;
avoid accidental duplicated jobs, but do not sacrifice justified experiments
to quota optimization. Bounded 3600s pilot, mandatory CPU preflight.

## Guidance interval research candidate (2026-11-27 KST)
Kynkäänniemi et al., NeurIPS 2024,
https://arxiv.org/abs/2404.07724, found a restricted guidance interval
can improve image diffusion distribution quality. This does not establish
the same result for speech CFM or its audio-only conditioning branch.
Hypothesis: late guidance may exaggerate spectral detail, while early
or middle guidance can affect identity and global structure. Test instead
of assuming. Research-only solver supports Euler/midpoint and explicit
guidance interval in normalized flow time (0 = noise, 1 = output).
Constant guidance is (0,1); candidate middle interval (.25,.75)
keeps strength 1.30 when active, with no added guidance outside.
A 32-step Euler interval run uses 32 conditional +16 negative
estimator evaluations versus 32+32 for constant guidance. This is
not equal total computation or equal integrated guidance; include
a constant-strength .65 comparison (same integrated nominal guidance),
and optionally late/early interval controls in future paired evaluation.
Synthetic tests check masked prompt preservation, velocity evaluations
and exact constant-vector-field expectation. No acoustic claim.
V8 run 45919753036 remains pinned to prior commit; no parallel GPU
request. Adopt only after same-token native 48kHz results, valid duration
and CER/SV and held-out speakers, plus listening.

## Quality curriculum v9 (2026-11-27 KST)
Evidence: Qwen3-TTS (https://arxiv.org/abs/2601.15621) describes broad
pretraining followed by high-quality continual pretraining; its general
stage is not exclusively low quality. Flow-matching noise robustness
(https://arxiv.org/abs/2406.05699) motivates denoising and noisy prompt
augmentation; Stable-TTS (https://arxiv.org/abs/2412.20155) motivates
preserving existing capabilities during adaptation. TITW
(https://arxiv.org/abs/2409.08711) motivates future real-world corpus testing.
V8: oracle Euler32 CFG .9 reduced CER 27.23% to 19.20% and increased SV
.8485 to .8581, but increased Mel distance 4.9241 to 5.1621. Use CFG .9
as common controlled baseline here, not a universal production default.
Research hypothesis: separate informative variation from nuisance corruption,
and compare broad-to-clean continuation against clean-only and restoration
objectives. Frozen V5 backbone and 100-channel bounded residual flow-velocity
adapter, initially zero, trained with 120 AdamW updates, delta penalty and
gradient clipping. Four training speakers, two independent held-out
speakers, pinned VCTK revision, exclude held-out exact normalized text
and waveform hashes. 8 target utterances train, 4 target utterances test.
Synthetic degradation: 10dB white noise, 8kHz bandwidth simulation,
held-out reverberation-like echoes. Native 48kHz clean originals preserved.
Arms: frozen baseline; clean-only 120 updates; degraded targets 60 then
clean 60; degraded inputs/clean targets 60 then clean 60 with 25% robust
replay. Replay differs from direct curriculum and needs separate ablation
if positive. Same initialization/optimizer budget for trained arms.
Evaluate both clean and held-out room corruption; each condition keeps
oracle semantic tokens identical across arms. 32 waveforms, CER/SV,
common Mel distance and native band statistics. Oracle target access
makes this diagnostic, not deployable text-only synthesis. This is a
small adaptation feasibility experiment, not large-scale pretraining
or a guarantee of no collapse. Synthetic noise not naturally noisy
recordings; external corpus and larger speaker-disjoint test later.
S1 must separately verify semantic-token corruption and language
generalization before training on real low-quality data. Production
unchanged. No empirical claims until GPU completion.

## Generalization and quality roadmap (2026-11-27 KST)
Active v9 45921918936 submitted successfully; no concurrent GPU request.
1. Finish quality curriculum: clean-only vs low-to-clean vs restoration-to-clean
   with fixed update budget and held-out speakers. Require complete cases,
   compare both clean and room; reject clean regression or severe artifacts.
2. Test semantic robustness upstream of S1/S2. WavLM (https://arxiv.org/abs/2110.13900)
   uses denoising and masked prediction. StableToken (https://arxiv.org/abs/2509.22220)
   explicitly addresses noise-robust discrete speech tokenization. These support
   paired clean/noisy consistency, not transplanting another vocabulary.
   Implemented tools/s1_semantic_robustness.py: frozen CNHuBERT and V5 codebook,
   zero-initialized bounded 768-channel feature adapter, train paired noise10
   and band8 plus 25% clean replay, 240 AdamW updates; test clean, seen noise
   and band8, unseen room, disjoint speaker/text/hash. Measure UED to clean
   reference, length, frame agreement, vocabulary perplexity, repeated runs.
   Clean baseline UED is zero; require low clean drift and no diversity collapse,
   report all conditions and per utterance. Agreement alone is not proof of
   phonetic correctness or TTS quality; later validate S1/S2 downstream.
   Initial synthetic test and CPU CI added; no GPU submission yet.
3. Audit native S1 English and Korean frontend, EOS probability, token
   distribution, prompt length and duration calibration before fine-tuning.
4. Separate semantic content and prosody/identity pathway in S1-S2; preserve
   native codebook and checkpoint compatibility. Compare frozen teacher and
   bounded trainable residual before expanding architecture.
5. Test interval guidance vs constant .65/.9/1.3 with identical tokens,
   report estimator evaluations. Midpoint rejected on current pilot.
6. Vocoder: true-Mel matched reconstruction, inspect bands separately;
   do not equate 16-24kHz loss with audible noise. Consider fullband Mel only
   with compatible training, no silent substitution into existing checkpoints.
7. Scale with diverse authentic noisy and clean corpora, speaker-disjoint
   train/validation/test and near-duplicate checks. Evaluate language,
   speaker, emotion, recording domain and listening separately.
No production rollout until paired held-out acoustic and intelligibility
gates pass and listening confirms improvement. Small studies are exploratory;
base-model pretraining overlap remains unknown. 

## v9 results and semantic robustness v10 (2026-11-27 KST)
Run 45921918936 completed 32/32 in 683.772s; all workers exit zero;
FP32, 4 train speakers, 2 held-out speakers. Both workers replicated
identical training seed (not independent replications). Clean CER/SV/Mel:
baseline .19196/.85814/5.16210; clean-only .22321/.85525/5.13587;
low-then-clean .19643/.85275/5.10693; robust-then-clean
.17857/.85528/5.33336. Unseen room:
baseline .25893/.76283/6.47880; clean-only .26339/.76909/6.48853;
low-then-clean .27679/.76814/6.46215; robust-then-clean
.25446/.76593/6.63729. None dominates baseline across CER/SV/Mel.
No severe collapse under frozen bounded residual adaptation; not proof
of robustness during full pretraining. Active HF ratio reduced with
robust arm (clean .00138 to .000889, room .00337 to .00220), not a
direct artifact score. Defer production changes.
V10: enlarge VCTK same pinned revision to 6 training and 4 held-out
speakers, 3 utterances each; exclude speaker, normalized exact text
and waveform hash overlap. 2 independent adapter training seeds,
240 updates each, 12 held-out utterances x 4 corruptions x 2 arms x
2 seeds = 192 token comparisons. Preserve native CNHuBERT, quantizer
and 1024 codebook. Train noise10/band8 and clean replay; hold out
synthetic room. Compare unit edit distance to clean tokens, length,
perplexity, repetition, frame agreement. Predeclared exploratory gate
per seed: clean drift <=2%, both seen corruptions improve, unseen room
nonregression, per-utterance perplexity >=80% baseline and repeat
growth <=4. Both seeds required. No waveform generation in this pilot,
no acoustic quality claim; clean reference tokens are not phonetic
ground truth. Next stage downstream paired synthesis and real noisy
recordings only if evidence supports. Existing 48kHz originals retained.

## v10 result and v11 token-aware gate (2026-11-27 KST)
Run 45923560566 completed 192/192 in 319.793s, independent seeds
20261127/20261128. UED by condition: clean baseline 0, adapters
.15768/.10379; noise10 .88024 -> .87758/.87891; band8 .52428 ->
.51630/.51830; room .64205 -> .63473/.64271. Both fail clean drift
<=.02; seed 1 repeat increase 5, seed 2 unseen room regression.
Reject ungated continuous feature MSE adapter.
StableToken (ICLR 2026, https://proceedings.iclr.cc/paper_files/paper/2026/hash/5186171eebc16ab214e4058ec9fcf63a-Abstract-Conference.html)
motivates discrete-token stability under perturbation; WavLM
(https://arxiv.org/abs/2110.13900) motivates denoising supervision.
Our method is different: maintain frozen CNHuBERT, ssl_proj, codebook
and 1024 IDs; train bounded residual at projected codebook input using
native negative squared Euclidean distances as differentiable logits,
clean-teacher token CE, feature loss, clean replay and identity penalty.
Verify argmax exactly equals native quantizer on every extracted input.
Train learned binary utterance quality detector from feature moments
with paired clean/noisy samples; calibrate threshold=max validation
clean score + .02, clipped to 1; no test labels used to route samples.
Compare baseline, ungated token-aware, and gated restoration. Prior
v10 test becomes validation; new four speakers disjoint from prior six
train and four validation, exact text and hash disjoint. Same source
revision, original native 48kHz. Two seeds, 720 updates each, 288
comparisons (12 fresh test utterances x 4 conditions x 3 arms x 2).
Predeclare gated UED <=.02 on clean, improve both seen corruptions,
no unseen room regression, mean perplexity >=80% baseline and max
repetition <= baseline+4, both seeds. These are exploratory gates,
not proof of downstream TTS quality. No production edits. Gating can
fail under new authentic noise or bypass all corrections; report activation
rates explicitly. No real noisy recordings or waveform evaluation yet.

## v11 results and batched v12 (2026-11-27 KST)
V11 run 45924697066 completed 288/288 in 391.317s. Gated UED,
seed1/2: clean .02893/.02830 vs 0; noise10 .84654/.84528 vs .89119;
band8 .74214/.72579 vs .52327; room .65723/.66792 vs .58302.
Both fail. Ungated clean drift .63333/.61887, both gated clean
false activation 1/12. The universal binary gate and CE specialist
do not preserve mild corruption. Reject deployment.
User requested amortize setup across experiments. V12 runs one
Kaggle setup, one model load per GPU/seed, shared SSL extraction cache,
with setup/feature/fit/native S1 timings. Two independently seeded
workers; original 6 train speakers, v11 test becomes validation,
new 4 test speakers exclude all previous train/validation/test speakers,
exact normalized text and waveform hash. 576 comparisons across
baseline, unconditional spectral subtraction, specialist restoration,
margin-protected specialist, routed specialist, routed+margin.
Separate noise10 and band8 experts; smaller bounded residual (.15),
margin objective rather than overconfident CE, clean replay and identity
penalty. Multiclass quality router threshold per expert calibrated on
validation only; disable expert if no beneficial safe threshold.
Unseen room remains untrained. Assess clean drift <=.02, improvements
on both seen corruptions, unseen room nonregression and no large
mean perplexity drop. No production edits.
In same loaded engine: 48 native S1 generation audit cases (4 test
speakers x 2 targets x 3 sampling settings x 2 seeds). Compare native
top-k15/pen1.35, top-k5/pen1.35, top-k15/pen1.10; cap 512, token length
ratio, cap hits, UED to real tokens. No oracle access for generation,
no waveform synthesis. Token alignment is not pronunciation quality.
Avoid combining experiments that depend on an unverified result;
shared inputs and controls only. Current batch no trained S2/vocoder
changes; follow-up gated on results. Native 48kHz synthetic corruption,
small data and unknown base-model pretraining overlap.

## Batched acoustic guidance v13 (2026-11-27 KST)
V12 run 45925702174 in progress, no duplicate GPU job. Prepare v13
without submitting until v12 completed and its mounted data is verified.
Primary prior evidence: Kynkäänniemi et al. NeurIPS 2024
https://arxiv.org/abs/2404.07724 restricted guidance in image diffusion;
speech applicability not proven. F5-TTS ACL 2025
https://aclanthology.org/2025.acl-long.313/ introduces Sway Sampling in
flow-matching TTS; its time grid is tested here without claiming transfer.
Fixed 48kHz VCTK four speakers x two target sentences from v12 holdout.
12 arms (96 waveforms), shared loaded V5dev on each worker:
real, true-Mel vocoder, constant CFG1.30/.90/.65, CFG1.30 middle
[.25,.75), early [0,.5), late [.5,1), sway=-1 with CFG .90, initial
noise 1.0 instead of .875 with CFG .90, and room-corrupted reference
with constant CFG .90 or middle interval. Same clean reference semantic
tokens and real target oracle tokens across arms; room changes acoustic
reference and conditioning, not prompt IDs. No training. Uniform Euler32
for all except sway time grid, same RNG seed per pair. Count actual
conditional/negative estimator calls. Constant .65 provides matched
nominal integrated guidance to 1.30 half interval; not equal computation.
Compare CER, SV, paired <=15.5kHz Mel distances, native band statistics,
duration, paired per-speaker results, and listening separately. Reject
candidates that worsen baseline clean CER/SV/Mel simultaneously; report
tradeoffs rather than select based on spectral proxy alone. Room subset
only compares room090 vs room-mid. Oracle targets not deployable TTS;
no downstream S1 quality claim, no actual noise MOS. Small sample,
synthetic room effect and unknown pretraining overlap. No production edits.
Time grid, initial noise override, worker, runner and explicit workflow
profile implemented. CPU selftests and GPU result pending.

## v12 actual result (2026-11-27 11:10 KST)
Run 45925702174 completed 576 token and 48 native S1 comparisons,
492.701s total, environment setup 257.048s (52.2%); per worker
model load/cache 36.55/36.68s and fit 5.67/5.69s. No passing arm.
Seed1 baseline/specialist/margin UED: clean 0/.20488/.00053,
noise10 .91030/.85881/.90977, band8 .52229/.47558/.52229,
room .60456/.61677/.60510. Seed2 similar: clean 0/.21285/.00053,
noise10 .91030/.85775/.90924, band8 .52229/.47983/.52229,
room .60456/.61996/.60456. Margin almost eliminates clean drift
but also almost all useful corrections. Routed margin seed2 clean
zero; no improvement on band8. The router calibration candidate list
allowed thresholds below .5 (e.g. .0000555), activating on most clean
samples; not a valid reliable quality gate. Future require class
posterior floor and validation risk constraint on actual routed arm.
Native S1 caps across seeds: 8/16 vs topk5 9/16, penalty1.10 11/16.
Median length ratio 2.44/2.74 native; sampling alone not sufficient.
Continue v13 acoustic guidance; v12 verified completed and source
manifest available. All results exploratory, no deployment.

## v14 S1 EOS and duration calibration (2026-11-27 KST)
Motivation: v12 native caps 8/16, median length ratio 2.44/2.74;
top-k and lower repetition penalty did not improve termination.
VoiceStar (https://jasonppy.github.io/assets/pdfs/VoiceStar.pdf) studies
duration control and text-speech alignment in AR codec TTS; not evidence
that a simple EOS prior transfers to V5. TED-TTS (https://arxiv.org/abs/2601.03170)
motivates inference-time global EOS modulation.
Fit regularized log-duration regression using train-only phone counts
and prompt tokens per phone; report calibration on independent validation
and fresh test. No target duration or target tokens used in free generation.
Teacher-forced logits on correct prompt+target path: final EOS probability,
rank, early maximum EOS probability and target token NLL; native free
generation: raw EOS probability, cap hits, duration error and UED.
Compare native and smooth bounded EOS logit bias strengths 2 and 4,
same random seed, no hard stop except common 512 cap.
Two seeds x four fresh speakers x two utterances x three arms = 48
48kHz downstream waveforms. Same V5dev CFG .90, Euler32 and matched
noise seed, FP32 ASR and SV. A duration-only gain cannot justify adoption
if CER or SV worsens. No production edits. No training of existing
S1 weights; only simple duration regression, small English corpus,
and no assumption of Korean generalization. V14 runnable profile
prepared while v13 is active; no duplicate GPU submission.

## v13 measured result and v14 launch (2026-11-27 13:35 KST)
Run 45926805757 completed all 96 waveforms, 1454.703s, setup
232.111s, all workers exit zero, FP32 CT2 ASR and speaker model.
Eight paired utterances, four speakers, real CER .00515 and true-Mel
vocoder .00515; true-Mel Mel distance 2.02452. CFG1.30 CER/SV/Mel
.16295/.84054/5.87462; CFG.90 .15266/.83676/5.66736;
CFG.65 .14065/.75324/5.76285; middle .15266/.59910/6.07098;
early .15266/.54778/6.80830; late .16638/.83531/5.69306;
sway090 .15952/.83763/5.64348; noise100 .15094/.79809/5.85962.
Room090 .14408/.74582/6.49571; room-mid .15437/.51954/7.18234.
Late guidance preserves most speaker similarity, with 24 versus 48
mean negative estimator calls; mean conditional 48, so total
estimator calls 72 vs 96 (25% reduction, not measured wall-clock speed).
Early/middle guidance harms voice identity consistently across speakers.
No single clean arm dominates baseline on all three measures.
Keep CFG .90 as balanced research reference; neither default changes
nor perceptual noise claims. Oracle target semantics, English VCTK
and synthetic room only. Next launch v14 actual text-only generation
with duration calibration plus teacher-forced diagnosis, 48 paired
audio examples. V13 no longer running; no active GitHub research jobs
at inspection. V14 standalone CPU architecture job was skipped;
mandatory pilot CPU selftests must succeed before GPU submission.

## v14 result and v15 causal audit (2026-11-27 KST)
Run 45940943141 completed 48/48 waveforms, 1766.289s, setup 241.175s.
Across seeds: native cap 12/16, bias2 6/16, bias4 1/16. CER
.82863/.82460/.80141, SV .84596/.84485/.83749. Still unacceptable.
Final EOS on teacher-forced concatenated prompt and target rank 79–173,
probability .0001352–.0021824. Eight distinct utterances (same forced
values repeated in both seeds), not sixteen independent measurements.
Teacher-forced labels verified EOS at end. Cannot attribute to
training class imbalance: padding EOS are included in CE.
Audit findings in current branch: dataset.batch_sequences right pads
phones/BERT while forward and forward_old use make_pad_mask_left;
valid EOS and post-end padding both contribute to summed CE;
reported token accuracy ignores EOS; DPO make_reject_y operates on
padded y rather than valid length, and DPO logps include padding.
Training audio pipeline uses 32k decode, gain 1145.14*(.475/peak+.5),
16k resample; inference uses direct 16k + silence based on S2 sample
rate, not 16k. English/Korean BERT may be zero by checkpoint config.
These are source facts, not proof of historical s1v3 training defects.
No production changes. Source-derived CPU diagnostic demonstrates
mask and extra EOS labels and tests rejected EOS contamination.
V15 reuses fixed v14 English and v6 Korean controls (not fresh holdout);
both versions V5dev/V2ProPlus use same s1v3 checksum, compare projection
and codebook hashes; 12 targets x 6 forced arms x 2 versions =144
EOS/NLL/top1 comparisons; 6 speakers x 2 prompt arms x 2 versions=24
native generation comparisons; exact training preprocessing vs raw
and prompt padding; standalone vs concatenation; native cached prefill
vs training Transformer last logits; production vs research phones
and BERT nonzero counts; 168 comparisons one install. No synthesis
or training in this diagnostic. Korean source is 16kHz only; language
and corpus/sample-rate differences are confounded, not language
causal evidence. Measure quantizer agreement rather than infer it
from codebook dimensions. Per-arm JSON progress restored.

## v15 measured results and v16 training repair (2026-11-27 14:50 KST)
Run 45946132571 completed 168/168 cases, 388.909s, setup 249.305s,
both GPU workers exit zero. Source audit reproduced right-pad/left-mask
mismatch, two extra EOS labels for length3 in length5 batch, and rejected
sequence EOS contamination 64/64 seeded trials on that specific fixture.
V5dev and V2ProPlus have different parameter hashes but identical S1
checksum and exact token equality in all 72 comparisons; EOS metrics
and seeded generation identical. Both native prompt pads 9600 samples
at 16kHz (0.6s), both English/Korean BERT tensors zero.
Native prompt EN median EOS .000779505, rank107.5, token top1 .05290,
4/4 cap; KO .411280, rank2.5, top1 .19809, 0/2 cap. Both prefill and
training last logits agree within 2.01e-5 across forced cases. Production
phones match all tested research targets. Raw/train English target UED
.25664–.40972, Korean .05263–.39462. Training preprocessing, no padding
and 300ms variants did not rescue EN EOS. Language, sample rate and
corpus confounded; no claim of a language-only cause. Quantizer
incompatibility rejected for the tested samples, not globally.
V16: frozen control plus five matched fine-tune arms, native right-pad
and all-EOS CE, left-pad phones/BERT only, true-EOS masked CE only,
combined, combined with real EOS weight4. DPO off throughout;
80 AdamW updates/arm at 2e-5, batch4, same seed-specific batch order,
reset identical initial weights, train only last two S1 blocks and
readout, dropout disabled for controlled pilot, normalized CE per
included weighted tokens (not exact stock summed-CE optimization).
Raw semantic extraction shared across arms to isolate objective;
18 EN train utterances from 6 speakers, 8 validation and 8 test pairs,
4 KO guard pairs (no KO training). All reused v14/v6 diagnostic
holdouts, not a fresh confirmation. 240 forced evaluation rows
including 144 free-generation cases across two seeds; test generation
without oracle duration, common 512 cap. No waveform quality claim,
no production edits, no automatic adoption. Save arm deltas privately.
Native arm is a matched fine-tuning control, not full original training.

## v16 results and v17 expanded multilingual alignment (2026-11-27 KST)
Run 45948297170: 240/240, 570.411s, setup249.185s, both workers exit0.
EN test mean of seed median EOS probabilities, summed caps/16, mean UED:
frozen .000779505,12,3.81002; native .01512884,7,3.04415;
text_pad .03394829,2,1.97917; eos_mask .02070914,8,3.21528;
combined .03390899,2,1.97917; combined_weight4 .11541768,0,1.34524.
Top1 frozen .05290, combined_weight4 .04666; earlier EOS maximum
median increased from .00157 to .16931/.22157. Korean guard cap0
throughout, but NLL increased from 3.8194 to 3.85384/3.85552 and
top1 decreased from .19809 to .17875/.18345. No adoption or waveform
quality claims; reduced UED partly reflects corrected duration.
Padding intervention effective in both seeds, EOS-only masking
inconclusive. v17 keeps both correctness fixes and expands trainable
scope to eight last Transformer blocks, text embeddings and readout.
New VCTK native 48k mic2 from same pinned revision, 24 speakers x 5
utterances, deterministically divided into 16 train/4 validation/4 test;
excluded all v16 EN speakers, exact normalized texts and PCM hashes
before selection. Korean Zeroth train split 8 speakers x4, disjoint
from existing test guard. Korean 16k is a replay guard/training domain,
not a wideband quality corpus. 112 original training utterances;
additional same-speaker prompt/target concatenations share these
utterances, prefix excluded from CE. 3 EN +1 KO sampled per batch,
fixed order across arms for each seed. Two seeds, 240 updates/arm,
AdamW2e-5, FP32, no DPO, dropout disabled, same initial model.
Five arms: corrected EOS2/4/8, EOS4+CTC .2/.6. CTC reads true
target semantic hidden frames only (no prompt or padding), phoneme
targets, separate blank, excludes infeasible alignments with coverage
tracked; auxiliary head not used at inference. This is inspired by
CTC-TTS (https://arxiv.org/abs/2602.19574) and Choi et al. auxiliary
CTC text representation alignment
(https://mm.kaist.ac.kr/pubs/pdfs/choi25b.pdf), not their implementation.
Hidden states can attend all input text, so auxiliary CTC may learn
text copying rather than robust acoustic alignment; free-generation
and downstream audio are necessary for confirmation. 432 paired
forced evaluation rows, 240 free generation cases (16 EN test and
4 KO guard per arm per seed, including frozen). 2,400 updates in
total across both workers. One setup, no production edits; independent
parameter snapshots in private output, no automatic adoption.
Fresh EN test relative to v16, unknown pretraining overlap and
near-duplicates; reuse KO guard, speaker split only. Acquisition
may fail if 32 shards do not supply required qualified speakers.
No waveform evaluation in v17; use a subsequent fixed downstream
synthesis comparison if token/content performance warrants it.

## v17 results and v18 paired waveform validation (2026-11-27 KST)
Run 45952006429 completed 432/432 cases in 1209.130s, setup243.681s,
both workers exit0. 112 source training utterances, 26,118,656 trained
parameters, two seeds, fixed speaker-disjoint EN split relative to v16.
EN test average of two seed medians for EOS probability and absolute
log-duration ratio, mean UED, mean teacher-forced top1:
frozen .000491617,1.52881,4.454659,.070800;
eos2 .0508202,.334748,1.064728,.090343;
eos4 .0685045,.427380,1.024078,.089641;
eos8 .0904084,.594249,1.048155,.088058;
eos4_ctc .1227982,1.115865,.969043,.082848;
eos4_ctc_strong .1084710,1.469273,.964978,.082108.
Caps frozen22/32, all trained0/32. CTC lower UED may reward deletion,
and worsens absolute log-length error; not adopted. CTC coverage
915/960 and 906/960 per seed. KO teacher-forced NLL frozen3.81940,
eos2 3.71322, eos4 3.72097, eos8 3.73247, CTC .2 3.77310, CTC .6
3.79313. No claim of audio improvement. EOS2 prioritized but not a
confirmed quality winner; direct waveform comparison required.
V18 recovers saved v17 parameter snapshots (exact expected keys,
FP32 shape and finite checks, SHA-256) on pinned base S1; fixed first
and fourth EN target per speaker, all four KO guard targets, both
original training seeds. Arms real, oracle target semantics, frozen,
eos2, eos4, eos4_ctc (.2). 144 rows, 120 synthetic waveforms, shared
V5dev 32-step CFG .9 and common per-case seed. Verify token length,
edit count and cap parity with v17, not exact token sequence equality.
Language-specific Whisper large-v3 FP32 CER, ERes2Net speaker cosine,
duration ratios/short and long counts, active HF and voiced periodicity.
Store raw float WAV and level-matched PCM16 WAV; randomized clips
with separate private key. Same v17 holdout, not a fresh confirmation;
KO real audio native 16k and synthesized 48k. Spectral proxies not
perceptual noise metrics, no MOS. No new training or production
change in v18. No automatic deployment; prioritize actual intelligibility,
identity and duration before further scaling.

## v18 results and v19 dual track (2026-11-27 KST)
Run 45967266945: 144/144 waveform cases, 1876.747s, setup239.688s,
six workers exit0, regenerated token metric parity passed. EN
mean seed CER/SV: real 0/.86469, oracle .16018/.85064, frozen
1.09281/.84490, eos2 .84431/.68465, eos4 .85629/.59139, CTC .2
.85778/.48868. Short (<.5 real duration) cases eos2 4/16, eos4
7/16, CTC 10/16. All remain unusable in EN. KO CER/SV: real
.12403/.90676, oracle .18992/.86491, frozen .03101/.82990,
eos2 .04651/.82934, eos4 .02713/.83840, CTC .05039/.81977.
KO only four distinct texts; ASR error on real limits interpretation.
KO median voiced-periodicity mean frozen .88180, eos2 .90802,
eos4 .89633. Acoustic proxies not noise proof.
V19: two independent tracks, two v17 seed checkpoints, one installation.
EN eos2 unchanged vs 160 additional AdamW updates at 1e-5 using
corrected CE vs CE+0.3 softplus ranking (margin .25). Same train
batches and padded geometry (3 EN positives +1 KO replay +3 EN
negatives); negative text is nearest phoneme length, distinct phones,
same prompt and true target tokens, ranking excludes prompt/EOS/pad.
Only positives contribute CE. Train scope same 26.1M as v17, no DPO.
96 forced rows per seed and eight free synthesized test cases/arm.
Length-matched contrast is an experimental objective, not proof of
phoneme alignment: near lengths can still differ, and teacher forcing
may mask errors. EN holdout reused, no independent confirmation.
KO uses frozen eos4 per seed, new Zeroth test four speakers x2
targets excluded v17 train and guard speaker/text/hash. Fixed S1
tokens per case, compare native CFG .9, acoustic reference gain
(target -23dBFS, max gain8), late CFG1.3, combined. Late arm changes
guidance schedule and solver path together; no pure CFG causal claim.
Gain affects acoustic reference only, semantic tokens and reference
phones preserved. No S2 training. KO 16k native source not true
wideband; all synthesized 48k. 112 waveforms, FP32 ASR/SV, duration,
voiced periodicity/HF, raw and level audio, blinded listening.
No production changes or deployment; no new claimed improvement
until results. Korean corpus acquisition may fail if too few eligible
speakers, no fallback to leakage. 

## v19 results and v20 controlled multilingual conditioning (2026-11-27 KST)
Run 45973236470: 112/112 waveform cases, 1582.894s, setup230.470s,
all six workers exit0, identical generated token hashes across KO
acoustic arms. EN mean CER/SV base .84431/.68465, continued
.86228/.55215, contrastive .86826/.53308; held-out NLL worsened
from ~4.897 to ~4.945 despite training CE decrease. Do not adopt
extra small-corpus CE or ranking as configured. KO eight texts, two
seeds: base CER .04231, SV .79752, periodicity .91362, activeHF
.00108741; gain .04808/.79283/.91426/.00089648; late
.04038/.81038/.91805/.00100424; gain+late
.04423/.80848/.92211/.00083084. No subjective quality confirmation.
V20: EN doubled original utterances 80->160 from 32 speakers;
32 KO replay utterances. Additional native 48k VCTK 24 speakers
x5, exclude all v17 speakers/text/audio hashes, split 16/4/4,
new EN validation/test relative to v17 only. Frozen pinned
bert-base-multilingual-cased using existing offset-nearest alignment.
Native projection bias preserved, zero-initialized 768D bridge
overrides original weight on this experimental en/ko path; verify
zero input exact parity and gradients. Separate from production,
no claim about Chinese compatibility. Frozen v17 eos2 vs 240
updates corrected CE with zero features, aligned mBERT, and
within-reference/target reversed feature order control. All three
trained arms reset same v17 seed weights and bridge, 3 EN+1 KO
same sampled batch schedule, AdamW2e-5, FP32. Reverse features still
contain contextual information, not a null lexical control. Existing
alignment uses approximate phone position, not verified word/phone
timings. 128 forced rows per seed, eight EN waveforms per arm.
KO: exact v19 four speaker/eight utterance holdout reused to isolate
native vs research Euler implementation, full .9 vs full 1.3, late
.9 vs late 1.3; same generated token sequence, original reference,
32 steps and matched random seed. No gain variation; no S2 training.
Native .9 vs research full .9 is a solver-path control; late/full
within same research solver isolates time schedule at fixed strength.
144 total waveform cases, 256 forced rows, 1440 additional updates
across both workers. FP32 ASR/SV, duration, acoustic proxies, blinded
clips. No MOS, no deployment. KO source 16k, generated 48k.
Acquisition may fail if insufficient new VCTK speakers. Prior v16
and earlier EN overlap not excluded globally; pretraining overlap
unknown. No claim of better quality until results.
