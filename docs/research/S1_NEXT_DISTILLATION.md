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
