# CODEX ACTIVE RESEARCH HANDOFF

> Project: `pymood1/GPT-SoVITS-improve`
>
> Purpose: Codex should be able to read **this file alone**, understand the current research state, choose the next highest-value task, implement it, validate it, update the TODOs/decisions here, and continue without requiring step-by-step instructions.
>
> This is an **active research handoff**, not a frozen specification. Update it whenever evidence changes.

---

## Development experiment harness

The current fast iteration loop intentionally separates implementation from
model-quality validation:

```text
Codex / ChatGPT:
  choose experiment cases and parameters
  update DEV_EXPERIMENT_PRESET.yaml
  implement the required code
  keep outputs isolated and reproducible
  run only cheap structural checks

User:
  open the Kaggle WebUI
  press the temporary one-click experiment button
  let Kaggle preprocess/train automatically
  run inference/listen
  report the perceptual result

Then update the next Git preset from that feedback.
```

Current temporary preset:

```text
Full FT
V+Attention-Out LoRA r8/a16
QKV+Attention-Out LoRA r8/a16

epoch 15
batch 7
DPO ON
SoVITS/S2 fixed and NOT trained
Controller retrained separately for every GPT/S1 case
```

The WebUI one-click experiment is a development harness, not a product feature.
Remove `DEV_EXPERIMENT_PRESET.yaml`, `tools/dev_experiment_harness.py`, and the
temporary WebUI controls after the final training direction is selected.

## GPT / Emotion Controller pairing invariant

Managed V5dev GPT fine-tunes now treat the Emotion Controller as part of the
same logical model bundle.

Flow:

    GPT/S1 training
        -> exported GPT checkpoint
        -> refresh/create emotion sidecar (reuse prediction cache)
        -> Controller training against that exact GPT checkpoint
        -> Controller validation PASS
        -> GPT training is reported complete

Controller outputs use a deterministic path derived from the GPT checkpoint:

    <output-root>/
      GPT_weights_v5dev/<name>.ckpt
      emotion_controllers/paired/GPT_weights_v5dev/
        <name>_emotion_controller.pt

Rules:

- a managed V5dev GPT is not considered ready until its paired Controller passes;
- inference resolves the Controller from the selected GPT, not from a global
  latest-controller fallback;
- legacy dev-experiment manifests remain a read-only compatibility fallback for
  older experiment outputs;
- changing SoVITS clears the active Controller and then revalidates/reloads the
  GPT-paired Controller against the new S2 provenance;
- pretrained/base GPT checkpoints may still run without a Controller;
- the temporary dev one-click harness disables the normal post-GPT Controller
  hook because it snapshots each S1 case first, then trains the Controller
  against that snapshot using the same deterministic pairing path.

This follows the upstream GPT-SoVITS separation of concerns: s1_train.py still
only trains S1; orchestration and artifact pairing remain in webui.py and small
helper functions.

Validation still required after merge:

- repository syntax/unit CI;
- one Kaggle V5dev GPT training run proving automatic sidecar/controller
  creation and PASS promotion;
- inference startup and GPT switching proving exact paired auto-load;
- SoVITS switching proving paired Controller revalidation/reload;
- listening validation remains separate from structural correctness.


## 0. Operating rule for Codex

For every nontrivial task:

1. Read this file first.
2. Identify the highest-value unresolved hypothesis/blocker.
3. Check existing code/logs before adding code.
4. Change **one major experimental variable at a time** whenever possible.
5. Run the smallest meaningful validation first.
6. Separate evidence into:
   - **FACT**: code/log/objective measurement supports it.
   - **OBSERVATION**: human listening/qualitative behavior.
   - **HYPOTHESIS**: plausible explanation not yet proven.
7. Update this file after work: change, test, result, justified conclusion, remaining uncertainty, next best task.
8. If a branch fails its stop criteria, stop extending it merely because code already exists.
9. Prefer simple, evidence-driven changes over architectural expansion.
10. Never report a runtime success unless it was actually run.

Codex should normally proceed on an obvious next step without asking the user for confirmation. Ask only if data/artifacts are missing, a destructive choice has no safe default, or subjective user preference materially changes the direction.

---

# 1. Main research goal

The project extends GPT-SoVITS V5dev with:

- speaker/reference-aware S1 fine-tuning,
- discrete emotion tags such as `[happy]`, `[surprised]`, `[angry]`, `[disgusted]`,
- an Emotion Controller steering S1 hidden states,
- automatic emotion labeling/dataset preparation,
- leakage-safe validation and OOF diagnostics,
- Kaggle-friendly training/inference.

The target is **not** merely “classify pseudo emotion labels.”

The actual target is:

> Fine-tune GPT-SoVITS while preserving useful reference/speaker information, and make emotion tags create audible, controllable style changes without degrading pronunciation, semantic stability, or speaker identity.

---

# 2. Research principles

## 2.1 Evidence hierarchy

Prefer evidence in this order:

1. Controlled runtime A/B generation
2. Held-out/OOF objective evaluation
3. Repository implementation/checkpoints
4. Upstream implementation/docs/issues
5. Papers
6. Architectural intuition

Do not elevate intuition above measurements.

## 2.2 Listening is important but not ground truth

User listening impressions are valuable **OBSERVATIONS**, not automatic proof of objective degradation/improvement.

Example:

- User hears `[angry]` at strength 1.5–2.0 as somewhat more plausible but also more slurred.
- Record this observation.
- Do **not** claim CER increased, semantic content degraded, speaker identity drifted, or hidden-space corruption occurred until measured.

Convert listening impressions into testable hypotheses.

## 2.3 One-variable-at-a-time

Critical variables include:

- S1 mode/checkpoint
- S2 checkpoint
- dataset
- seed
- epochs
- batch size
- DPO
- LoRA rank/alpha
- Controller checkpoint
- reference audio
- inference sampling
- emotion tag/strength

Do not call a comparison “controlled” when several of these changed.

## 2.4 Avoid overengineering

Before adding a mechanism, ask whether the issue is instead:

- data quantity,
- label noise,
- objective mismatch,
- calibration,
- evaluation confound.

Avoid turning the pipeline into an expert-system-like collection of special cases.

---

# 3. Environment and paths

Repository:

```text
pymood1/GPT-SoVITS-improve
branch: main
```

Kaggle runtime repo:

```text
/kaggle/tmp/GPT-SoVITS
```

Dataset:

```text
/kaggle/input/datasets/pymood1/endfield/datas
/kaggle/input/datasets/pymood1/endfield/training.list
```

Prepared list:

```text
/kaggle/tmp/audio_input/training.list
```

Temporary emotion workspace:

```text
/kaggle/tmp/emotion_opt
```

Final outputs:

```text
/kaggle/working/gpt_sovits_output/
    GPT_weights_v5dev/
    SoVITS_weights_v5dev/
    emotion_controllers/
```

Latest S1 used for Controller training:

```text
/kaggle/working/gpt_sovits_output/GPT_weights_v5dev/project1-e15.ckpt
```

Latest Controller candidate:

```text
/kaggle/working/gpt_sovits_output/emotion_controllers/emotion_dataset_01/emotion_dataset_01_emotion_controller.candidate.pt
```

Diagnostics:

```text
/kaggle/working/gpt_sovits_output/emotion_controllers/emotion_dataset_01/emotion_dataset_01_emotion_controller.candidate.diagnostics.json
```

---

# 4. Dataset facts and limitations

Current dataset:

```text
102 samples
1 speaker: Gilberta
language: Korean
```

Controller classes:

```text
happy       38
angry       13
disgusted    7
surprised   35
```

`fearful=1` is excluded by the minimum-sample rule.

Important limitation:

- The source data is not believed to contain genuinely strong angry speech.
- Therefore pseudo-label names should not be treated as verified semantic emotion ground truth.
- An `angry` label may be a relative acoustic/style direction rather than clean human-perceived anger.

---

# 5. Emotion labeling state

Automatic model:

```text
iic/emotion2vec_plus_large
min confidence: 0.35
speaker-relative calibration: speaker_relative_clr_median_v2
```

Raw top-1 distribution in the recent run was strongly biased:

```text
surprised 77
happy     22
angry      2
unknown    1
```

After speaker-relative processing, Controller targets became:

```text
happy       38
angry       13
disgusted    7
surprised   35
```

This is a pseudo-label/control dataset, not a verified emotion benchmark.

---

# 6. S1 reference-retention problem

S1 is `Text2SemanticDecoder`.

Relevant inference behavior:

```text
reference semantic prompt + target text
        -> S1 autoregressive semantic generation
```

Historical problem:

> Full S1 fine-tuning can adapt strongly to the target speaker, but useful reference-dependent behavior appears capable of weakening.

A central suspected cause is an objective mismatch.

Standard S1 fine-tuning mostly learns from the same utterance's text/semantic target, while real inference uses reference utterance A to generate a different target utterance B.

Therefore:

> **HYPOTHESIS H-S1-1:** same-utterance S1 training does not sufficiently teach cross-utterance reference dependence.

This is not yet proven.

---

# 7. Implemented S1 LoRA modes

Available modes:

```text
full
lora_v_o
lora_qkv_o
```

Important implementation detail:

The custom attention uses combined `in_proj_weight`; ordinary PEFT assumptions about separate `q_proj/k_proj/v_proj` do not apply.

Targets:

```text
lora_v_o    : V row block + attention output projection
lora_qkv_o  : combined QKV + attention output projection
```

LoRA export merges adapter deltas into ordinary inference weights.

Relevant files:

```text
GPT_SoVITS/AR/modules/activation.py
GPT_SoVITS/AR/models/t2s_lightning_module.py
GPT_SoVITS/s1_train.py
webui.py
tools/s1_lora_selftest.py
```

---

# 8. Latest V+Out experiment

Observed configuration:

```text
mode=lora_v_o
target=v+out
rank=8
alpha=16
attention_layers=24
trainable=393,216 / 77,999,618
trainable ratio ~= 0.504%
epochs=15
seed=1234
```

So the V+Out adapter really was active.

## 8.1 Previous strong Full FT Controller result

```text
S1 hidden probe:
macro_acc  = 51.06%
macro_auc  = 0.761

Controller:
best_val_loss       = 0.76188
best_step           = 500
hidden_OOF_CE       = 1.22499
hidden_balanced_CE  = 1.28974
macro_acc           = 44.37%
macro_auc           = 0.694

coverage:
happy       100%
angry       100%
disgusted   100%
surprised   100%
```

## 8.2 Latest V+Out r8/a16 Controller result

```text
S1 hidden probe:
macro_acc  = 46.88%
macro_auc  = 0.739

Controller:
best_val_loss       = 0.96255
best_step           = 500
hidden_OOF_CE       = 1.28626
hidden_balanced_CE  = 1.40214
macro_acc           = 40.14%
macro_auc           = 0.669

coverage:
happy        76%
angry        69%
disgusted    71%
surprised    89%
```

Training auto-extended:

```text
500 -> 600 -> 700
```

Then early stopping restored best step 500.

The run passes the current Controller gate, but is weaker than the previous Full FT comparison on most listed probe/Controller metrics.

---

# 9. Interpretation of V+Out

## FACT

V+Out r8/a16:

- trained successfully,
- used only about 0.5% trainable S1 parameters,
- produced a valid Controller,
- retained measurable emotion/style-separable hidden-state signal,
- was weaker than the previous Full FT run on the currently compared metrics.

## NOT PROVEN

Do not claim yet that:

- V+Out inherently damages reference retention,
- V+Out inherently damages pronunciation,
- Full FT is universally superior,
- the weaker Controller metrics directly caused subjective listening differences.

## Current working interpretation

V+Out may be too restrictive for this small single-speaker adaptation task.

Do **not** spend many runs sweeping V+Out rank/alpha without new evidence.

The next adapter experiment should be:

```text
QKV + Attention Out LoRA
rank=8
alpha=16
```

---

# 10. Critical S2 confound

The most recent combined training flow also launched SoVITS training.

That matters because the main experimental question concerns **S1 reference retention**.

For clean S1 comparison:

> Keep the **same S2 checkpoint** for Full FT, V+Out, and QKV+Out inference tests.

Do not conclude S1 quality from generations that also changed S2.

The latest Controller preprocessing path loaded the pretrained V5 S2:

```text
GPT_SoVITS/pretrained_models/gsv-v5-pretrained/s2Gv5dev.pth
```

But actual listening A/B must explicitly lock S2 too.

---

# 11. Human listening observations — not objective facts

These are **OBSERVATIONS ONLY**.

## Overall

Emotion tagging appears to have some audible effect, but no current tag has reached a clear:

> “wow, this is definitely and strongly different.”

The current subjective level is closer to:

> “there are signs the controller does something, but separation is still modest.”

## happy / surprised

User observation:

- They appear relatively easier to distinguish than angry/disgusted.
- This does **not** mean they are strongly or cleanly separated.
- The correct wording is only: **relatively better among the four tags**.

## angry / disgusted

User observation:

- At strength around `1.0`, intended emotion is weak/uncertain.
- Around `1.5–2.0`, the result can sound somewhat more plausible for the intended emotion.
- It is still not strongly convincing.
- At stronger settings, speech can **sound more slurred / less fluent** to the user.

This does **not** prove CER degradation or semantic corruption.

### HYPOTHESIS H-EMO-1

Increasing angry/disgusted strength may improve perceived style while also increasing pronunciation/fluency degradation.

Validation should use:

```text
same text
same reference
same S1
same S2
same Controller
same sampling/seed when possible
strength = 0, 0.5, 1.0, 1.5, 2.0
```

Measure:

- ASR transcription / CER proxy
- speaker similarity
- optional F0/prosody statistics
- listening

If objective content measures remain stable, do not force a “semantic corruption” explanation just because listening sounded less fluent.

---

# 12. Controller state

Current preferred path: `hidden_direction`.

High-level behavior:

```text
S1 hidden representation
    -> learned class direction
    -> steering through hidden/readout space
```

In this path:

```text
semantic_ce=nan
grad_u=0
u_norm=0
```

are not automatically bugs.

The semantic-likelihood path is mainly diagnostic.

Current Controller success means:

> some hidden-state discrimination/generalization exists

not:

> semantic emotions are solved.

---

# 13. Priority 0 — preserve a clean baseline

Before more architecture changes, ensure reproducibility of:

- Full FT S1 checkpoint used by the stronger Controller run
- latest V+Out S1 checkpoint
- one fixed S2 checkpoint
- corresponding Controllers
- fixed reference audio
- fixed inference texts
- inference settings

If missing, create a compact experiment manifest. Do not create a new framework just for manifests.

---

# 14. Priority 1 — controlled S1 A/B evaluation

Build or reuse a small evaluation harness for:

```text
A. Full FT
B. V+Out r8/a16
C. QKV+Out r8/a16
```

Keep fixed:

```text
S2
reference
text
seed / sampling
emotion tag
emotion strength
```

Minimum evaluation:

### Reference/speaker retention

- speaker embedding similarity if available
- alternate-reference sensitivity
- human A/B listening

### Content stability

- ASR transcription
- CER or comparable text error metric

### Emotion behavior

- no tag
- `[happy]`
- `[surprised]`
- `[angry]`
- `[disgusted]`
- optionally one multi-tag sentence

Do not add dozens of metrics before this comparison exists.

---

# 15. Priority 2 — QKV+Out r8/a16 once

Use:

```text
mode=lora_qkv_o
rank=8
alpha=16
```

Match V+Out conditions as closely as possible:

- same data
- same seed
- same epochs
- same DPO setting
- same S2 at inference
- same Controller training/evaluation

## Decision rule

If QKV+Out gives a meaningful reference-retention or Controller-quality tradeoff, investigate further.

If QKV+Out is again clearly weaker than Full FT in both objective and listening comparison:

> Stop treating LoRA as the primary solution to reference retention.

Do not immediately respond with a large rank/alpha sweep.

---

# 16. Next primary direction if LoRA is not enough

## Full FT + cross-utterance reference objective

This is currently the most important structural hypothesis.

### Goal

Teach S1 to use:

```text
reference utterance A
        -> target utterance B
```

for the same speaker.

### Minimal-change design

Do not rewrite GPT-SoVITS.

Prefer:

1. Sample two utterances from the same speaker:
   - reference A
   - target B
2. Build prompt/reference semantics from A.
3. Build target text/semantic supervision from B.
4. Compute target-generation loss on B.
5. Keep pretrained S1 initialization.
6. Compare against ordinary Full FT.

Why: real inference uses a different reference utterance from the target text, while normal fine-tuning is largely same-utterance.

### First version must stay simple

Do not initially add:

- new encoders
- adversarial losses
- large contrastive systems
- memory banks

First test whether cross-utterance supervision alone helps.

---

# 17. Optional retention regularization

Only after a cross-utterance baseline exists, consider a weak:

```text
hidden-retention / KL-like regularizer
```

Purpose: reduce destructive drift from pretrained behavior.

Keep it weak. If it blocks target-speaker adaptation, reduce or remove it.

Do not make this the first fix.

---

# 18. Controller improvement direction

Do not hide S1/reference problems with stronger Controller gains.

Possible causes of weak tags:

- sample imbalance
- pseudo-label noise
- source data lacking strong versions of some emotions
- gain calibration
- direction quality
- interaction with S1 representation

Do **not** immediately hard-code:

```text
angry gain = 1.8
disgusted gain = 1.8
```

because high strength may coincide with worse perceived fluency.

First run per-class strength sweep:

```text
0
0.5
1.0
1.5
2.0
```

Record:

- emotion score if informative
- ASR/CER
- speaker similarity
- optional F0/prosody
- listening

Then determine whether the issue is mainly:

```text
gain calibration
```

or:

```text
poor/noisy direction
```

Only then consider one at a time:

- per-class gain normalization
- direction norm normalization
- projection constraints
- mild orthogonalization against content-sensitive directions
- clipping

---

# 19. Minority-class / pseudo-label caution

Current class counts:

```text
38 / 35 / 13 / 7
```

Do not claim happy/surprised are intrinsically easier emotions.

Conservative interpretation:

> They have more training samples and are relatively easier to perceive in the current system.

For angry/disgusted:

- more real expressive data may help,
- better labels may help,
- but automatic relabeling must not manufacture semantic certainty.

Human-reviewed anchors are more valuable than aggressively relabeling everything.

---

# 20. Stop criteria

## Stop V+Out tuning when

- QKV+Out gives a better tradeoff, or
- V+Out remains worse than Full FT with no reference-retention advantage.

## Stop LoRA as the primary reference-retention solution when

A controlled QKV+Out test also fails to beat or meaningfully complement Full FT.

Then move to:

```text
Full FT + cross-utterance reference training
```

## Stop increasing Controller strength when

higher strength is accompanied by verified:

- CER degradation,
- large speaker-similarity loss,
- obvious instability,
- or consistently worse human fluency.

## Stop adding architecture when

current evidence points to data, labels, objective mismatch, or evaluation confounds instead.

---

# 21. Active TODO board

Codex must maintain and reorder this board.

Status:

```text
[ ] unresolved
[-] in progress
[x] completed
[!] blocked / invalidated
```

## Highest priority

- [ ] Create/verify reproducible experiment manifest: Full FT vs V+Out vs future QKV+Out.
- [ ] Lock one S2 checkpoint for all S1 reference-retention comparisons.
- [ ] Create/reuse a compact fixed inference test set: reference + texts + tags + strengths.
- [ ] Compare Full FT vs current V+Out using the **same S2**.
- [ ] Run QKV+Out `r8/a16` once under matched conditions.
- [ ] Compare Full FT / V+Out / QKV+Out and make an explicit continue/stop decision for LoRA.

## If LoRA is not clearly useful

- [ ] Implement minimal same-speaker cross-utterance S1 training: reference A -> target B.
- [ ] Add a config/UI switch so baseline Full FT remains available.
- [ ] Add minimal synthetic/unit checks for pair sampling and target masking.
- [ ] Run a small controlled test before adding retention regularization.
- [ ] Compare ordinary Full FT vs cross-utterance Full FT with identical S2/inference conditions.

## Controller

- [x] Add CPU static steering-distribution diagnostic and opt-in generation trace (2026-11-22; synthetic checks only).
- [x] Require paired S1 identity and aggregate multiple generation traces by emotion/strength (2026-11-22; synthetic integration checks).
- [ ] Run paired real-checkpoint diagnostic and generation trace on Kaggle; quantify H2/H4 before changing steering.
- [ ] Build 0–2 strength sweep evaluation for all 4 emotions.
- [ ] Record ASR/CER and speaker similarity across strength.
- [ ] Keep listening notes separate from objective metrics.
- [ ] Determine whether angry/disgusted weakness is mainly calibration or direction quality.
- [ ] Only then consider per-class gain normalization/direction constraints.

## Dataset / labels

- [ ] Preserve pseudo-label provenance.
- [ ] Do not declare angry/disgusted labels semantically correct without evidence.
- [ ] Add human-reviewed anchors if available.
- [ ] Decide whether additional expressive samples are needed before architecture expansion.

## Engineering quality

- [ ] Keep persistent Gradio errors visible until user closes them.
- [ ] Avoid full UI redraws that cause flicker.
- [ ] Keep BERT default ON.
- [ ] Keep preprocessing confidence default at 0.35.
- [ ] Keep Controller validation every 100 steps with val-based early stop/auto-extend.
- [ ] Preserve Python 3.14 compatibility work where practical.
- [ ] Keep architecture simple and avoid unnecessary file fragmentation/circular dependencies.

---

# 22. Completed / known work

- [x] V+Out custom S1 LoRA exists.
- [x] QKV+Out custom S1 LoRA exists.
- [x] LoRA merge-to-normal-inference export exists.
- [x] S1 adapter UI controls exist.
- [x] LoRA synthetic self-test exists.
- [x] hidden-direction Controller path exists.
- [x] leakage-safe validation exists.
- [x] Controller OOF evaluation exists.
- [x] val-based early stopping exists.
- [x] 500-step default with auto-extension exists.
- [x] V+Out r8/a16 run completed.
- [x] Controller was retrained against the changed S1 checkpoint.
- [x] latest Controller run returned PASS under the current gate.
- [x] current listening impressions are explicitly recorded as observations, not facts.

---

# 23. Current decision table

| Question | Current status | Evidence level | Next action |
|---|---|---|---|
| Is hidden-direction raw preference context-dependent? | No, for fixed weights/ID/vocab/dtype it is static | FACT (source + synthetic) | Measure real paired readout scores |
| Is strength 1 steering effectively a no-op / budget-limited? | Not measured on real weights/generation | HYPOTHESIS | Run new CPU diagnostic + opt-in trace |
| Does V+Out train correctly? | Yes | FACT | No implementation debugging unless a new error appears |
| Is V+Out Controller stronger than Full FT? | No; current metrics are weaker | FACT | Do not tune it endlessly |
| Does V+Out preserve reference better? | Unknown | UNRESOLVED | Same-S2 controlled A/B |
| Is QKV+Out better? | Unknown | UNRESOLVED | One matched experiment |
| Are happy/surprised strongly separated? | No such conclusion | OBSERVATION | Only relatively better |
| Are angry/disgusted directions bad? | Unknown | HYPOTHESIS | strength sweep + objective checks |
| Does high angry/disgusted strength hurt speech? | User hears possible slurring | OBSERVATION | CER/speaker/listening validation |
| Is pseudo-label `angry` true anger? | Not established | UNKNOWN | Preserve uncertainty |
| Is cross-utterance training promising? | Well-motivated | HYPOTHESIS | Test if LoRA does not clearly win |

---

# 24. Codex decision policy after every experiment

Update this file with answers to:

```text
What did we test?
What variable changed?
What stayed fixed?
What objective metrics changed?
What listening observation changed?
What is now a FACT?
What remains only a HYPOTHESIS?
Did the result justify more work on this branch?
What is the next highest-information experiment?
```

Then reorder TODOs.

---

# 25. Work-start template

Before a nontrivial task:

```text
이번 작업
- 확인할 가설:
- 변경할 변수:
- 고정할 변수:
- 구현:
- 검증:
- 성공/중단 기준:
```

Keep it brief.

---

# 26. Work-end template

After finishing:

```text
완료
- 변경:
- 실행한 검증:
- 결과:
- 새로 확정된 사실:
- 아직 가설인 것:

남은 일
- 1.
- 2.
- 3.

다음 권장 작업
- ...
```

Also update the Active TODO board and Decision table.

---

# 27. Things Codex must NOT do

Do not:

- claim a runtime test passed if it was not run,
- treat subjective listening as objective proof,
- change S1 and S2 simultaneously in a controlled S1 experiment,
- repeatedly increase LoRA rank without evidence,
- aggressively relabel data to make metrics look better,
- assume `angry`/`disgusted` labels are semantic ground truth,
- add large subsystems before testing a minimal hypothesis,
- rewrite the project when a small patch suffices,
- create circular dependencies,
- fragment the repo into unnecessary helper layers,
- hide failed experiments from this handoff,
- keep an approach alive just because it already has code.

---

# 28. Immediate recommended sequence

Unless new evidence invalidates it:

```text
1. Lock common S2 + fixed inference test set
2. Re-evaluate Full FT vs current V+Out under identical inference conditions
3. Run QKV+Out r8/a16 once
4. Compare all three
5. If LoRA still loses without a clear reference-retention win:
      stop LoRA-first research
      -> implement minimal Full FT + cross-utterance reference A -> target B
6. In parallel, build 0–2 Controller strength sweep
7. Use objective + listening evidence to decide whether Controller needs:
      calibration
      data improvement
      or direction-quality constraints
```

This is a recommendation, not a rigid rule. Codex may reorder tasks when new evidence clearly increases expected information value.

---

# 29. Current bottom line

Current evidence supports:

## 2026-11-22 — Steering magnitude diagnostic (measurement only)

### Change and fixed variables

Added `tools/diagnose_emotion_steering.py`: CPU FP32 inspection using the actual
controller loader and S1 readout only (no S1/S2 model initialization). It requires
the bound S1 SHA-256 and refuses a missing binding or mismatch. This was tightened
in the follow-up below; no unverified-pair static report is now produced.
Safe `weights_only=True` loading is retained. Hidden-direction mode
only; legacy context-dependent mode fails clearly instead of measuring dummy
context. All stored emotion IDs are inspected (four for the requested paired
Controller), strengths 0.5/1/1.5/2, absolute raw/clamped statistics, saturation,
delta standard deviation/pairwise log-odds span and Pearson matrices. Constant
vectors have undefined Pearson entries (`null`). JSON records hashes and scope.

Added process-start environment flag `GPT_SOVITS_STEERING_TRACE=1` in
`AR/models/emotion_control.py`. Default OFF; OFF performs no new tensor work,
RNG calls, device synchronization, statistics or log I/O, just Python flag
branches. Literal zero CPU instruction/latency overhead is not claimed or
benchmarked. Sampling/filtering/tilt/update formulas and all defaults unchanged.
ON records JSON lines to stderr AFTER sampling; optional observer errors cannot
fail generation. Logging itself has synchronization/I/O overhead when enabled.
Records carry generation identity, step, stable row IDs across batch shrinking,
mean absolute clamped delta on non-EOS support (zero for budget-disabled rows),
actual normalized log-probability-ratio magnitude, signed cumulative log ratio,
and budget-disabled/eligible row-step counts/fraction. Last record retains final
values for finished rows as well as live rows. These are sampler row-steps,
not a claim about semantic/audio quality or counts of accepted decoded tokens.
`mean_abs_delta` describes the proposed eligible support tilt; the actual ratio
field also captures normalization and numerical fail-open. No new runtime trace
is emitted for paths that bypass the emotion sampler entirely.

### FACT / OBSERVATION / HYPOTHESIS findings

- **H1 — FACT (source + synthetic):** hidden-direction `preference_scores` uses
  hidden values only indirectly through dtype/device; numerical hidden values do
  not enter its formula. It is normalized `hidden_directions()` dot normalized
  readout rows. Fixed dtype/device/weights/emotion/vocab gives a static vector.
  Actual output probability changes still depend on base probabilities, sampler
  support, fixed EOS mass and remaining budget. Legacy mode is context-dependent.
- **H2 — HYPOTHESIS remains:** isotropic independent random unit-vector cosine
  has std 1/sqrt(H) (~0.03125 at H=1024), but learned directions/readout are NOT
  established isotropic/independent. No actual trained-weight distribution was
  measured here. Small raw scores do not prove no-op: common offsets cancel and
  relative supported-token deltas matter. Static results cannot establish audio
  quality or explain the user's slurring observation. High-strength plausibility/
  slurring remains **OBSERVATION**, not a measured causal effect.
- **H3 — FACT (code read only; training unchanged):** main hidden-direction loss
  is CE(hidden_beta * hidden_emotion_logits(mean-pooled frozen hidden)). Default
  semantic_aux_weight=0. The semantic auxiliary branch additionally requires
  `steering_mode == legacy_low_rank`, so even a positive weight does not train
  it in hidden-direction mode. Controller OOF sets semantic_loss=None. There is
  no generation-likelihood gradient in this objective. Classification and
  steering do share V/emotion coefficients and S1-derived input representations;
  'no generation loss' does NOT mean mathematical parameter independence.
  Class count is checkpoint/data-defined, not hardcoded four.
- **Coverage — FACT:** `prepare_controller_sample` mirrors top-k/top-p,
  temperature/repetition penalty and early EOS exclusion. A target token is
  valid when supported with finite positive base probability and >1 non-EOS
  alternative survives. Coverage is valid token fraction. In main training,
  `skipped_coverage` counts unavailable semantic auxiliary/diagnostic coverage;
  there is NO continue that skips hidden CE/backprop solely for low coverage.
  `evaluate_controller` likewise skips semantic evaluation only; valid hidden
  samples remain eligible for Controller training/probes. Do not read this
  counter as dropped hidden-training examples.
- **H4 — FACT mechanism, HYPOTHESIS incidence:** active condition is remaining
  budget >= 2*clamp; default 2 and .35 requires remaining >= .7, i.e. absolute
  signed cumulative log ratio <=1.3. Cumulative state adds the sampled token's
  log(p_steered/p_base), not absolute deltas, and can cancel across steps.
  Mechanism exists; no actual-generation disabled fraction was measured here.

### Executed verification and limitations

- Existing isolated Python 3.12 CPU Torch environment; user environment unchanged.
- New tests initially failed (missing CLI / absent trace), then passed.
- `python -B -m unittest tests.test_emotion_steering_diagnostics
  tests.test_emotion_control_training_vectorization -q`: **9 tests PASS**.
- CLI was exercised with synthetic saved Controller/S1 files, four emotions and
  all four strengths. Synthetic hidden values changed without changing static
  scores. Legacy mode rejected.
- Same-seed samples, full probability tensors, cumulative ratios and RNG states
  were bitwise equal with trace OFF, ON and pre-instrumentation sampler arithmetic.
  Test includes one budget-disabled batch row, EOS, top-k filtering and stable
  trace row IDs after index_select. This is CPU synthetic evidence, not GPU or
  end-to-end TTS equivalence across all hardware/dtypes.
- Three touched Python files parse using Python 3.10 grammar. This is NOT a
  Python 3.10–3.14 import/version matrix.
- No real checkpoint inspection, GPU inference, listening evaluation, training,
  private-data transfer, default changes, commit, push or PR was performed.
- Usage at start: 9% short-window / 54% weekly remaining; after implementation
  5% / 54%, so work limited to this unit's verification and handoff.

### Exact Kaggle commands (run against these modified sources)

In a Kaggle `%%bash` cell, replace the three artifact paths with the intended
paired S1/Controller and the SAME S2 used in your listening baseline. The old
handoff paths are examples, not evidence these historical files still exist.

```bash
set -euo pipefail
cd /kaggle/tmp/GPT-SoVITS
CONTROLLER="/absolute/path/to/paired_emotion_controller.pt"
S1="/absolute/path/to/paired_s1.ckpt"
S2="/absolute/path/to/fixed_s2.pth"
OUT="/kaggle/working/emotion_steering_diagnostic"
mkdir -p "$OUT"
python tools/diagnose_emotion_steering.py \
  --controller "$CONTROLLER" --s1-checkpoint "$S1" \
  --output "$OUT/static.json" | tee "$OUT/static.summary.txt"
env GPT_SOVITS_STEERING_TRACE=1 version=v5dev \
  gpt_path="$S1" sovits_path="$S2" emotion_controller_path="$CONTROLLER" \
  PYTHONPATH="$PWD:$PWD/GPT_SoVITS${PYTHONPATH:+:$PYTHONPATH}" \
  python -u GPT_SoVITS/inference_webui.py ko_KR \
  2> >(tee "$OUT/inference.stderr.log" >&2)
```

The second command starts the ordinary inference UI, not an automatic TTS run.
Confirm successful paired Controller load, then use the SAME reference/text/
sampling conditions for each emotion and strength. Trace records have
`"event": "emotion_steering"`; stderr also contains ordinary non-JSON logs.
For each generation, use its last step record: the final-by-row mapping retains
finished rows; disabled_fraction denominator is eligible row-steps, excluding
singleton non-EOS support, nonfinite inputs and zero strength/clamp. Changing
the environment flag requires restarting the inference process. OFF: omit the
flag or set it to 0; no controller settings need to change.

### Next priority / conditional proposals only

First run real paired static diagnostics and record ON trace at fixed inference
conditions. If supported-token normalized shifts are actually small and budget
cutoff is not the dominant cause, H2 gains support; static random-reference
similarity alone is insufficient. Do not change multiple variables at once.
Only if measurements support the limitation, propose (do not implement):

- Context-dependent steering: distinguish content-appropriate token preferences
  rather than an identical emotion vector for all hidden contexts.
- Generation-likelihood training objective: test the gap between hidden-label
  separability and the preferences actually applied during semantic generation.
- Strength normalization: test whether measured supported-token scale differs
  across emotions, with explicit quality/budget controls against over-steering.

No candidate is selected or implemented without the user's confirmation.

### 2026-11-22 follow-up — strict pairing and multi-generation summary

**Scope:** measurement-only follow-up requested after report review. Fixed:
steering formulas, strength/clamp/budget/training defaults, S1/S2 and all user
settings. No actual pretrained/private weights were loaded for this follow-up.

**FACT — implementation and verified behavior:**

- Static diagnosis now refuses missing S1 binding as well as mismatched SHA-256.
  A folder named `paired` is not the proof: the stored S1 hash must equal the
  actual file used to obtain readout weights. Output cannot overwrite an input
  checkpoint/report/log. Diagnostic function copies the tiny Controller before
  converting it to CPU FP32, preserving the caller's dtype and train/eval state.
- Trace schema 2 includes emotion ID, strength, sampling settings, device/dtypes,
  Controller-file hash and bound S1 hash. Identity hashing runs only when the
  process-start trace flag is ON. Actual S1/S2 pairing remains the existing
  inference loader's responsibility; its checks were not weakened or changed.
- Generation keys use process/session plus a monotonic counter, rather than a
  reusable Python object address. Row IDs follow batch index selection. No texts,
  reference paths, audio or tokens are added to the trace.
- In addition to proposed clamped delta, logs measure normalized probability
  ratios, p_base-weighted absolute log ratios and total variation
  (`0.5 * sum(abs(p_steered - p_base))`). These are probability-distribution
  measurements, not acoustic/semantic quality scores.
- A CPU FP16 top-k case reproduced a logger-only NaN: 1e-12 rounds to zero in
  FP16, so filtered probabilities caused log(0)-log(0) and 0*NaN. Observation
  arithmetic now uses detached FP32 copies. Sampling probabilities, sampling
  dtype, cumulative update and RNG arithmetic remain unchanged.
- A second synthetic boundary case found that a starting FP32 cumulative state
  must be observed AFTER the same dtype/device conversion that the sampler's
  `ensure_batch` performs. At a near-budget boundary, FP16 rounding otherwise
  made the logged cutoff differ from the actual gate. The trace now copies the
  pre-update state in the probability dtype/device before testing that gate.
  This changes the observer only, not the actual budget condition.
- `--trace-log ... --static-report ...` aggregates ALL observed sampler sequences
  by emotion ID and strength, not the entire log's final generation alone. It
  requires trace hashes to match the verified static JSON, rejects changed
  sampling/dtype/device conditions, ignores exact duplicate records and rejects
  conflicting duplicate steps. Missing initial/intermediate steps and malformed
  trace lines produce warnings. Exception fail-open is reported separately from
  budget cutoff. Numerical safety returns inside the tilt are not individually
  classified; compare proposed delta and actual probability changes.
- Summary reports sample counts, pooled disabled/eligible row-steps, distributions
  of final OBSERVED signed/absolute cumulative ratios and sequence mean TV.
  Both overall and active-only delta/TV averages are retained so cutoff does not
  masquerade as weak active steering. No eligible opportunities gives a null
  cutoff fraction, not evidence the budget is harmless.
- A sampler row can be a sentence fragment. Neither row count nor generation
  count proves distinct texts, whole requests or completed inference. There is
  no explicit request-end marker: final means last observed per-row value.

**Executed CPU verification:** isolated existing Python 3.12 / CPU Torch
environment; no environment packages changed. Expanded tests include actual CLI
static -> saved/loaded synthetic Controller -> 4 emotions x 2 strengths x 2
independent sampler sequences -> summary (16 synthetic generations), strict
missing/mismatch pairing rejection, known quantiles/correlations, mixed identity
rejection, multiple-generation aggregation, duplicate/incomplete logs, fresh
process flag unset/0/1, observer I/O failure, input preservation, state/RNG
equality in hidden and legacy modes, zero strength/clamp, singleton support,
EOS handling, CPU FP16 filtered probabilities and a dtype-converted budget
boundary. The complete focused command below passed **64 tests in 16.406s**.
The real script entry point's `--help` also ran in a fresh Python process.
All three touched Python files parsed using Python 3.10 grammar; no complete
Python 3.10–3.14 real-dependency import matrix or GPU equivalence is claimed.
Independent AST comparison to `HEAD` found all **9 preexisting classes/functions
identical after removing the added trace guards**. No existing steering,
sampling, checkpoint or index-selection arithmetic was changed.

```bash
python -B -m unittest \
  tests.test_emotion_steering_diagnostics \
  tests.test_emotion_control_training_vectorization \
  tests.test_emotion_representation_probe \
  tests.test_emotion_pipeline_helpers -q
```

**HYPOTHESIS / interpretation:** H2 and actual-generation H4 incidence remain
unmeasured. Fraction of all-vocab |delta| below 0.05 is now exported as an
EXPLORATORY statistic, not an acceptance/no-op threshold. A constant delta of
0.03 was tested to cancel on normalization despite nonzero proposed delta.
Conversely, small nonconstant shifts can change samples and compound across
autoregressive steps. High positive emotion-vector correlation or saturation
is a diagnostic signal, not proof of identical emotions or an audio-quality
effect. H1/H3 source conclusions and the user's listening OBSERVATIONS remain
as above. No architecture/calibration/training candidate was implemented.

### 사용자 실행용 짝 가중치 탐색 셀

```python
import glob, os, sys
REPO = "/kaggle/tmp/GPT-SoVITS"
sys.path.insert(0, f"{REPO}/GPT_SoVITS")
from AR.models.emotion_control import load_emotion_controller_checkpoint, sha256_file

files = sorted({
    p for root in ("/kaggle/working", "/kaggle/input")
    for p in glob.glob(f"{root}/**/*", recursive=True)
    if p.endswith((".pt", ".ckpt")) and os.path.isfile(p)
})
for p in files:
    print(f"{os.path.getmtime(p):.0f}  {os.path.getsize(p)/1e6:8.1f}MB  {p}")

controllers = {}
for p in (f for f in files if f.endswith(".pt")):
    try:
        _, meta = load_emotion_controller_checkpoint(p, map_location="cpu")
        controllers[p] = meta["extra"].get("s1_sha256")
    except Exception:
        pass
print("\ncontrollers:", {p: (h[:12] if h else None) for p, h in controllers.items()})

want = {h for h in controllers.values() if h}
for p in (f for f in files if f.endswith(".ckpt")):
    h = sha256_file(p)
    if h in want:
        for c, ch in controllers.items():
            if ch == h:
                print(f"\nPAIR  controller={c}\n      s1={p}")
                print(f"python tools/diagnose_emotion_steering.py --controller '{c}' "
                      f"--s1-checkpoint '{p}' --output /kaggle/working/steering_static.json")
if not want:
    print("\n해시가 있는 controller가 없음 → 짝 검증 불가")

```

- PAIR가 출력되면 그 명령으로 정적 진단을 실행한다.
- 해시가 없거나 짝이 없으면 현재 학습기로 Controller를 재학습하고, `.pt`와 사용한 S1 `.ckpt`를 같은 Output에 함께 저장한다. 학습 기본값은 변경하지 않는다.

사용자가 셀 출력과 정적 진단 JSON을 가져오면 H2/H4를 FACT/HYPOTHESIS로 분리해 기록한다. 정적 진단과 trace 실행 전에는 판정하지 않는다. 해석은 포화율, active-only TV, budget 차단률을 함께 놓고 하며, 단일 지표로 원인을 확정하지 않는다.

### 재학습 분기용 최소 명령 — 텍스트만, 미실행

현재 `tools/train_emotion_controller.py`에 경로·S2 출처와 이전 실험의 감정 구성을 명시하고, 나머지 인자는 생략하여 학습 기본값을 유지한다. 아래 `확인 필요` 입력을 실제 값으로 바꾸기 전에는 실행하지 않는다. 전처리·데이터셋·reference map을 새로 만드는 명령은 포함하지 않는다.

주요 기본값: `--version=v5dev`, `--steps=500`, `--max-auto-steps=1000`, `--rank=8`, `--seed=233333`. `--hidden-lr=0.003`, `--hidden-weight-decay=0.01`, `--semantic-aux-weight=0.0`, clamp `0.35`, budget `2.0`도 그대로 둔다. `--device`는 CUDA 사용 가능 시 `cuda`, 아니면 `cpu`이며 명령에서 강제하지 않는다.

비교 대상은 이전 실험과 같은 4감정(`happy/angry/disgusted/surprised`)이다. argparse의 3감정 기본값(`happy,sad,angry`)에 맡기지 않고 `--emotions` 인자로 명시한다. `EMOTION_V5.md`에는 3감정의 `--emotions happy,sad,angry` 예시만 있고, 두 문서에서 이전 4감정 실험의 실제 인자값·ID 순서는 확인하지 못했다. 따라서 아래 `EMOTIONS`는 **확인 필요(사용자 확인)**로 남기며, 확인된 이전 인자값을 그대로 넣는다. 실제 Controller의 `emotion_names`도 확인한다. 3감정 재학습 결과를 기존 4감정 실험과 같은 모델로 취급하지 않는다.

```bash
set -euo pipefail
cd /kaggle/tmp/GPT-SoVITS
S1_SOURCE="/확인 필요/학습에 사용할 S1.ckpt"
FEATURE_DIR="/확인 필요/전처리 build 디렉터리"
METADATA="/확인 필요/metadata.jsonl"
REFERENCE_MAP="/확인 필요/reference_map.json"
BUILD_MANIFEST="/확인 필요/위 build에 대응하는 build.json"
EMOTIONS="happy,angry,disgusted,surprised"  # 사용자 확인 필요; argparse 도움말은 쉼표 구분 비-neutral 감정 ID 순서를 명시(따옴표 요구는 명시 없음). 진단 도구는 checkpoint 메타데이터 순서를 읽으므로 이 순서 표기는 진단 단계에 영향 없음.
PAIR_DIR="$(mktemp -d /kaggle/working/controller_pair_XXXXXX)"
cp -- "$S1_SOURCE" "$PAIR_DIR/s1.ckpt"
python tools/train_emotion_controller.py \
  --s1-checkpoint "$PAIR_DIR/s1.ckpt" \
  --feature-dir "$FEATURE_DIR" \
  --metadata "$METADATA" \
  --reference-map "$REFERENCE_MAP" \
  --feature-build-manifest "$BUILD_MANIFEST" \
  --emotions "$EMOTIONS" \
  --output "$PAIR_DIR/controller.pt"
printf 'PAIR_DIR=%s\n' "$PAIR_DIR"
```

S1은 먼저 새 `/kaggle/working/controller_pair_XXXXXX` 디렉터리로 복사하고 그 복사본을 실제 학습 입력으로 사용하므로, 학습 후 `.pt`와 짝 S1 `.ckpt`가 같은 Output 아래에 남는다. 저장된 `s1_sha256`은 이 입력 파일로 학습기가 계산한다. 실제 파일 생성과 `[RESULT] PASS/FAIL`, 저장된 감정 목록은 학습 후 확인 필요이며, 종료 코드 0만으로 PASS로 판단하지 않는다.

`build.json`이 없다면 `--feature-build-manifest`를 `--semantic-s2-checkpoint "/확인 필요/6-name2semantic.tsv 생성에 실제 사용한 S2.pth"`로 대체한다. S2 경로를 추측하지 않는다. `--reference-prompt-cache`는 기본 미지정으로 유지했으며, 기존 inference-exact cache의 사용 여부와 경로는 확인 필요다.

**Kaggle measurement procedure:** run the static command above FIRST against
these modified sources and the exact paired files. Then launch the existing UI
with tracing ON. Use at least five distinct fixed short texts for each of the
four emotion IDs, strengths 1.0 and 2.0 (40 requests minimum), with the same
reference, S1/S2, Controller and sampling controls. Record the actual controls;
the current legacy inference UI command does not set/reset a per-request seed,
so do not call this a matched same-seed generation comparison. Seed-controlled
experiments may instead use an existing seed-capable inference entry point.
Human listening remains the user's work.

### 실행 표본표 — 5문장 × 4감정 × 2강도 = 40요청

고정 조건: reference 오디오·전사·언어는 **확인 필요**이며 전 요청에 같은 것을 사용한다. S1/S2와 Controller는 **확인 필요**인 실제 검증된 한 조합으로 고정한다. sampling은 `top_k=15`, `top_p=1.0`, `temperature=1.0`, `repetition_penalty=1.35`로 고정한다. 자동 하네스는 요청마다 **같은 seed=233333**으로 재설정한다. 수동 legacy UI는 요청별 seed를 고정하지 않으므로 같은 seed 비교로 해석하지 않는다. 문장 분할 설정도 동일하게 유지한다. trace는 프로세스 시작 시 `GPT_SOVITS_STEERING_TRACE=1`로 켠다.

청취용 baseline은 trace OFF(`GPT_SOVITS_STEERING_TRACE=0`)로 따로 생성한다.

각 셀은 해당 문장·감정·strength로 보낸 요청 1건이며, 실행한 요청만 `☐`를 `☑`로 바꾼다. 한 요청에서 나온 여러 sampler 조각을 별도 문장으로 세지 않는다.

감정 수에 따라 건수 조정: 5문장 × 감정 수 × 2강도이며, 4감정은 40건, 3감정은 30건이다.

| 번호 | 동일하게 사용할 한국어 문장 | happy 1.0 | happy 2.0 | angry 1.0 | angry 2.0 | disgusted 1.0 | disgusted 2.0 | surprised 1.0 | surprised 2.0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 상자는 문 옆에 있다. | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ |
| 2 | 표의 두 번째 줄에는 무엇이 적혀 있나요? | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ |
| 3 | 이 종이는 책상 위에 놓아 주세요. | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ |
| 4 | 복도를 따라가면 창문이 세 개 있고, 끝에는 작은 회의실이 있습니다. | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ |
| 5 | 오전 열 시에 자료를 나누고, 항목별로 번호를 붙인 다음, 날짜와 수량을 차례대로 확인하는 것으로 정리하겠습니다. | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ |

After generating the test cases, stop that UI process and run in a new Kaggle
`%%bash` cell (or summarize from another shell). Keep the raw log private and
share only bounded summaries:

```bash
set -euo pipefail
cd /kaggle/tmp/GPT-SoVITS
OUT="/kaggle/working/emotion_steering_diagnostic"
python tools/diagnose_emotion_steering.py \
  --trace-log "$OUT/inference.stderr.log" \
  --static-report "$OUT/static.json" \
  --output "$OUT/trace.summary.json" | tee "$OUT/trace.summary.txt"
```

Multiple logs may follow `--trace-log`; all must belong to the same verified
pair and runtime settings. Older schema-1 logs must be regenerated with schema 2.
Absent local output directories (`GPT_weights_v5dev`,
`gpt_sovits_output/emotion_controllers`, `output/emotion_opt`) were checked; no
actual paired artifacts were available there. No GPU run, real-weight static
report, listening result, private-data transfer, commit, push or PR is claimed.

**Next priority:** user runs real paired static diagnosis, then several-text
condition sweep and summary. Review H2/H4 jointly, with sample counts and warnings;
only then consider one of the three conditional candidates above. No fixed
0.05/no-op threshold or automatic H2 verdict has been introduced.

### Publication and measurement-input check — 2026-11-22

**FACT:** the four approved files were committed and pushed as
`d73a73ccb1269f3f9154d1b5f194ec029ed8b763` to
`codex/emotion-steering-diagnostics`. Remote `main` remained `fa4172d` at the
post-push verification. Other local settings/history were not committed.
The following input-discovery/protocol note is local follow-up, not a claim of
another published revision.

**FACT — bounded artifact discovery:** the local `emotion_controllers`,
`gpt_sovits_output/emotion_controllers`, `output/emotion_opt` and
`GPT_weights_v5dev` directories are absent. Configured owner `pymood1` was verified
through the normal Kaggle SDK. The known training notebook
`pymood1/notebook3a6135c9c8` has no saved version number and its output file list
is empty. Another previously used GPT notebook also has no saved version/output;
a related runtime probe has two outputs and no model checkpoint candidates.
Owned Dataset searches for emotion/controller found no matching datasets.
These checks do NOT prove that no checkpoints exist in other locations.

The browser also could not display the known private notebook's contents in its
current authenticated context. This is not proof that the notebook is deleted
or its draft session is stopped; no draft filesystem was inspected. No GPU or
new cloud compute session was started for this search.

**Unresolved input:** the response to the artifact-location question supplied
search instructions, not actual Controller/S1 paths or a saved-output source.
No trained pair was located, so the real static diagnostic has NOT run and no
real static/trace JSON is available. H2 and actual-generation H4 remain
HYPOTHESIS. Do not use a synthetic/untrained Controller as substitute evidence,
invent a binding hash, or launch new training as if it were measurement only.
Modification timestamps identify candidates, not verified pairs. A missing hash
means provenance is unverified; it does not alone establish the file's age.

**Updated measurement protocol:** use at least five distinct texts shared across
the 4 emotion x 2 strength conditions (minimum 40 requests), fixing reference,
S1/S2/Controller and sampling controls. Compare static saturation, active-only
TV and pooled budget-disabled fraction side by side. Nonlinear TV growth does
not uniquely imply clipping: normalization, different generated prefixes/base
distributions, support changes and cutoff can produce the same pattern. No
latency conclusions from tracing ON. Real GPU ON/OFF equivalence and listening
remain unmeasured here.

**Next required input/action:** locate the actual trained Controller plus its
hash-matched S1 in a live accessible session, saved private notebook output,
Dataset or explicitly identified local directory. Once available, run CPU static
diagnosis first, then the condition sweep and summary.

### 2026-11-22 — 40요청 자동 trace 하네스 (Kaggle 미실행)

추가 파일은 `tools/run_steering_trace.py` 하나뿐이며, CPU mock 테스트도 이 파일의 `--self-test` 모드에 포함한다. 기존 TTS·Controller·집계기 코드는 수정하지 않았다. 학습·GPU·실제 가중치 실행·청취 평가는 하지 않았다. H2/H4는 계속 미확정이다. 새 하네스와 이번 문서 수정은 로컬에만 있으며 commit/push하지 않았다.

**FACT — 기존 API 호출 근거:**
- `GPT_SoVITS/TTS_infer_pack/TTS.py:1185`의 `load_emotion_controller()`가 실제 S1/S2 provenance를 검증하고 Controller를 연결한다. 하네스도 GPU 모델 초기화 전 S1 해시와 요청 감정 목록을 먼저 확인한다.
- 같은 파일 `1330–1355`에서 `text`, `text_lang`, `ref_audio_path`, `prompt_text`, `prompt_lang`, sampling 네 인자, `seed`, `emotion_id`, `emotion_strength`를 읽는다. `1348`은 요청마다 `set_seed(seed)`를 호출하며, 함수 정의 `277`은 Python/NumPy/Torch 및 사용 가능한 CUDA RNG를 설정한다.
- `1643–1644`, `1737–1738`에서 해당 emotion 인자를 기존 S1 추론으로 전달한다. 하네스는 inline tag 대신 이 정수 ID/strength 경로를 쓰고 ID는 checkpoint `emotion_names`에서 찾는다.
- `1910–1967`의 `audio_postprocess()`는 `(sr, mono int16 ndarray)`를 반환한다. 하네스는 이 출력을 PCM16 WAV로 저장하고 자체 음량·음색 변환을 하지 않는다.
- 초기화는 `728`, `749` 등에서 `configs.save_configs()`를 호출한다. 원본 YAML은 읽기만 하고 `config.configs_path`를 실행 폴더의 `tts_runtime.yaml`로 지정한 뒤 TTS를 생성하여 사용자 설정을 보존한다.

**실행 방식:** 기본 문장은 위 표의 5개와 동일하다. 순서는 문장 → 감정(`happy angry disgusted surprised`) → strength(`1.0 2.0`)이며 총 40요청이다. 모든 요청에 같은 reference·sampling·seed를 전달하고, 기존 `set_seed()`도 요청 직전에 호출한다. TTS.run 자체도 같은 seed를 재설정한다. GPU kernel의 결정성/실제 음성 동일성을 검증했다는 뜻은 아니다.

각 요청 직전 stderr에 `request_marker`(index/sentence_id/emotion/strength)를 먼저 쓰고 이후 stderr를 원문 그대로 `steering_trace.log`에 함께 저장한다. 기존 집계기는 `emotion_steering`만 읽으므로 바꾸지 않는다. manifest는 입력 조건·해시·요청 문장/감정/강도/순서·WAV·trace generation ID를 기록한다. 참조 전사 원문은 manifest에 복제하지 않고 해시만 남긴다.

출력은 `/kaggle/working/steering_audio/run_*/` 아래 `*.wav`, `steering_trace.log`, `steering_manifest.json`, `tts_runtime.yaml`이다. 매번 새 실행 폴더를 만든다. 실패 시 중단하고 부분 manifest를 보존하며, WAV가 나와도 trace가 전혀 없으면 성공으로 기록하지 않는다. seed=-1·0 이하 strength·누락 감정·중복 조건은 거부한다. 모델/reference/설정 자산은 기존 Kaggle TTS 실행 환경에 있어야 한다.

아래는 **사용자가 실행할 명령 예시이며 미실행**이다. 새 파일을 먼저 Kaggle 레포의 `tools/`에 복사해야 한다. 모든 실제 경로와 reference 텍스트는 확인 필요이며 `all_ko`는 한국어 reference일 때만 사용한다. 다른 언어라면 기존 TTS의 정확한 `prompt_lang` 코드를 확인한다.

```bash
cd /kaggle/tmp/GPT-SoVITS
python -u tools/run_steering_trace.py \
  --controller "/확인 필요/paired_controller.pt" \
  --s1-checkpoint "/확인 필요/paired_s1.ckpt" \
  --s2-checkpoint "/확인 필요/fixed_s2.pth" \
  --reference-audio "/확인 필요/reference.wav" \
  --reference-text "확인 필요: reference의 실제 전사" \
  --reference-language all_ko \
  --emotions happy angry disgusted surprised \
  --strengths 1.0 2.0 --seed 233333 --device cuda \
  --output-root /kaggle/working/steering_audio
```

`--sentences-json`으로 문장 5개의 JSON 목록을 지정할 수 있으며 생략하면 표의 문장이다. `--config`를 생략하면 레포의 기존 `GPT_SoVITS/configs/tts_infer.yaml`을 읽는다. 아래 최신 런북의 정렬 변경으로 device/is_half는 WebUI 기본값을 적용하고, 그 외 source config 설정은 보존한다. 정상 완료 후 표시되는 RUN_DIR의 trace를 이전 정적 진단 JSON과 집계한다:

```bash
python tools/diagnose_emotion_steering.py \
  --trace-log /kaggle/working/steering_audio/run_확인필요/steering_trace.log \
  --static-report /kaggle/working/steering_static.json \
  --output /kaggle/working/steering_summary.json
```

CPU mock 검증 명령: `python -B tools/run_steering_trace.py --self-test`. 기존 격리 Python 3.12 환경에서 **CPU mock 테스트 10개 통과**. 실제 TTS/torch를 import하지 않는 mock으로 요청 순서·40개 전개·metadata ID 매핑·marker 순서·고정 입력·seed 재설정·manifest/WAV 생성·부분 실패·trace 누락·출력 보존·설정 저장 경로 분리를 확인했다. Python 3.10 문법 파싱과 `--help` 진입도 확인했다. **Kaggle 미실행**이며 실제 checkpoint/GPU 결과와 구분한다.

#### H3 보조 분석 — 읽기 전용, 실제 비율 미측정
- **FACT:** `train_emotion_controller.py:607–619`는 steerable non-EOS support가 2개 이상이고 target token이 finite·positive 확률로 support 안에 있어야 valid로 표시하며, valid token 비율을 coverage로 계산한다.
- **FACT:** `2548–2572`는 coverage가 `min_support_coverage` 미만일 때 `skipped_low_coverage`를 1 증가시키고 로그에는 `skipped_coverage`로 출력한다. 현재 코드의 로그라면 `160`은 이 분기에 들어간 학습 시도 횟수이며, 서로 다른 발화 160개를 뜻하지 않는다. 이 로그 값 자체와 과거 실행의 코드 버전은 재실측하지 않았다.
- **FACT:** top-k=5에서는 잘린 target의 base 확률이 0이므로 valid가 아니다. 현재 argparse 기본 top-k는 `15`(`1566`)이며 과거 실행이 5였는지는 그 실행 인자/metadata 확인이 필요하다.
- **FACT:** 낮은 coverage여도 hidden CE/backprop를 건너뛰지 않는다. `2207`은 valid hidden 샘플을 유지하며, 기본 hidden-direction/semantic_aux_weight=0에서는 coverage가 생성 손실을 제거하는 원인이 아니다(원래 그 손실을 사용하지 않는다).
- **HYPOTHESIS:** 같은 logits/prefix/top-p에서 top-k를 15→5로 줄이면 support가 좁아져 class별 likelihood/길이 차이에 따라 semantic 보조 손실·G(top-k) 진단의 표본 선택이 편향될 수 있다. 이것을 hidden 학습 클래스 탈락이나 angry/disgusted 약화의 확인된 원인으로 보지 않는다. 실제 감정별 coverage·누락·반복 횟수와 비율은 미측정이다.

#### 후보 설계 메모 — 모두 HYPOTHESIS, 채택 권고·구현 없음

1. **문맥 의존 steering 복원:** 변경 후보는 `GPT_SoVITS/AR/models/emotion_control.py::LowRankEmotionPreference.preference_scores`의 기존 legacy branch와 `tools/train_emotion_controller.py::main/run_controller_oof_cv`의 optimizer 연결이다. 식: `delta[t,k] = clip(s * ((LN(h_t) V) * e_k) U^T W^T, -kappa, kappa)`. 현재 기본 optimizer는 U를 학습하지 않으므로 mode만 바꾸는 실험은 유효하지 않다. 생성 likelihood 목적이 고정된 대조 조건을 먼저 정한 후 head의 문맥 의존성 하나를 비교해야 한다. **반증 측정:** 같은 loss/데이터/seed에서 held-out 정답 감정 대비 타 감정의 평균 token log-likelihood margin이 정적 head보다 개선되지 않으면 이 복원 가설을 기각한다.
2. **생성 likelihood 기반 손실:** 변경 후보는 `tools/train_emotion_controller.py::main/run_controller_oof_cv`의 loss/semantic branch와 `GPT_SoVITS/AR/models/emotion_control.py::vectorized_sequence_relative_scores`의 gradient 연결이다. 식: `L = L_hidden + lambda * CE(beta * [mean_valid log(p_E,j(y_t)/p_0(y_t))]_j, emotion_label)`. 기존 semantic branch의 분류 loss와 token NLL을 혼동하지 않으며, hidden-direction에서도 실제 readout steering으로 gradient가 흐르는지를 검증해야 한다. **반증 측정:** 사전에 고정한 held-out valid support에서 정답/타 감정 log-likelihood margin이 lambda=0 대조보다 개선되지 않으면 목적 연결 가설을 기각한다.
3. **strength 정규화:** 변경 후보는 `GPT_SoVITS/AR/models/emotion_control.py::LowRankEmotionPreference.preference_scores/tilt_probabilities`의 측정된 score scale 적용 지점이다. 식: `delta[k,v] = clip(s * tau * q[k,v] / sqrt(Var_cal(q[k,:]) + eps), -kappa, kappa)`이며 q는 clamp 전 preference score다. calibration 집합과 tau는 평가 결과를 보기 전에 고정하고, clipping/budget 기본값은 별도 변수로 건드리지 않는 설계다. **반증 측정:** 같은 평균 active-only TV로 맞춘 대조에서 held-out 감정별 active-only TV의 분산이 줄지 않으면 이 scale 보정 가설을 기각한다.

위 세 후보는 결과와 무관하게 구현하지 않으며, 사용자 승인 뒤에만 별도 실험으로 진행한다. H2/H4에 대한 결론은 추가하지 않는다.

#### 하네스 소스 점검 및 dry-run — 2026-11-22 (Kaggle 미실행)

- **FACT — 전달 체인:** `tools/run_steering_trace.py:275`의 `emotion_id`/`emotion_strength` → `GPT_SoVITS/TTS_infer_pack/TTS.py:1353–1370`의 파싱·범위 검사·enabled 판정 → `1632–1644`의 `infer_panel` → `1398–1403`에서 선택되는 직렬 `infer_panel_naive_batched` → `GPT_SoVITS/AR/models/t2s_model.py:887–905`의 kwargs 복사와 `infer_panel_naive` → `925–927`의 `_prepare_emotion_runtime`(`368–379`) → `1001–1011`의 `_sample_token` → `407–413`의 `sample_with_emotion_preference(emotion_id=emotion_id, strength=emotion_strength)` → `emotion_control.py:950`의 preference 계산. 명시한 값은 기본값으로 덮이지 않는다. controller 미부착/ID 없음/strength=0은 steering을 끄는 기존 조건이며, 하네스는 로딩 실패·누락 감정·0 strength를 거부한다. emotion-enabled 요청은 `TTS.py:1604–1608`의 별도 acceleration sampler를 우회한다.
- **FACT — 로딩 비교:** WebUI의 `inference_webui.py:431–454` → `_load_validated_controller_checkpoint`(`355–413`)와 하네스의 `run_steering_trace.py:151` → `TTS.py:1185–1242`는 서로 다른 wrapper다. 공통 loader는 `emotion_control.py:786`이며 둘 다 S1/S2 hash·V5 version·hidden dimension을 검증하고 eval 모드로 연결한다. clamp/budget은 공통 loader의 `862–863`에서 checkpoint config를 읽고, 누락 때만 기존 0.35/2.0을 사용한다. 두 wrapper 모두 이를 재지정하지 않는다.
- **FACT — 차이 목록:** WebUI device/is_half는 `inference_webui.py:90`, controller dtype은 `450–453`에서 결정한다. 하네스는 원본 YAML의 device/is_half와 선택적 `--device`를 사용한다(`run_steering_trace.py:136–149`, `TTS.py:416–419,554,1236–1239`); CPU에서는 half를 끈다. 따라서 WebUI와 dtype/device가 같다고 가정하지 말고 실행 폴더 `tts_runtime.yaml` 및 trace의 실제 dtype/device를 확인한다. WebUI는 S2 파일로 버전을 판별하는 반면 하네스는 controller metadata 버전을 요구한 뒤 TTS의 현재 S2 provenance 검사도 통과해야 한다. WebUI는 기존 CUDA graph 객체를 닫고, 하네스는 새 TTS 인스턴스를 만든다. 정밀도·device의 자동 일치나 WebUI와 출력 동일성은 실측하지 않았다.
- **FACT — seed/배치:** 요청 직전과 `TTS.run:1348`에서 같은 seed를 설정하며 `TTS.py:277–295`는 Python/NumPy/Torch/CUDA RNG를 다룬다. 하네스 `249–253`은 batch_size=1, cut1, parallel_infer=False, split_bucket=False, non-streaming을 명시하고 모두 manifest.fixed_inputs에 기록한다. 분할 후 순서를 보존하고 직렬 sampler를 사용한다. 기존 sampler의 exponential RNG(`AR/models/utils.py:143`)와 V5 합성 noise(`module/models_v5.py:34`)는 Torch RNG를 사용한다. 생성 길이/분할 조각 수에 따른 난수 소비 차이, 캐시/커널의 수치 결정성까지 같다는 보장은 아니며, PYTHONHASHSEED를 프로세스 도중 설정해도 현재 프로세스의 hash seed를 다시 정하지 못한다. 실제 GPU 재현성은 미검증이다.
- **FACT — stderr:** marker는 하네스 `271`, trace는 `emotion_control.py:1125`에서 각각 flush=True다. 기존 print는 본문/개행이 별도 write이므로 모든 동시 writer에 대한 OS 원자성을 보장하지 않는다. 하네스는 요청을 순차 실행하며, `TraceTee.write:169–177`은 tqdm의 미종결 진행 표시 뒤 JSON 앞에 개행을 넣어 레코드가 같은 줄에 붙지 않도록 한다(수치·sampling 변경 없음). 외부 프로세스/임의 background thread가 stderr를 동시에 쓰는 구성은 검증하지 않았다. 집계기 `diagnose_emotion_steering.py:310` 및 `146`은 emotion_steering만 선택하므로 marker는 집계하지 않는다. 집계기는 수정하지 않았다.
- **dry-run:** `python -B tools/run_steering_trace.py --dry-run`은 모델/가중치 로딩·출력 파일 생성·trace 환경변수 설정 없이 40건의 순서/문장/감정/strength/seed JSON만 stdout에 출력한다. 선택적 sentences-json은 읽는다. checkpoint를 읽지 않으므로 emotion_id와 짝 검증은 명시적으로 미확인이다. 실제 실행 때 ID를 metadata에서 매핑한다.
- **복구:** manifest는 각 요청 직전/완료 직후 원자적 replace로 갱신하지만 `--resume`은 미구현이다. 중단된 요청의 부분 trace와 재실행 trace를 기존 집계기 수정 없이 안전하게 분리·검증하는 범위를 작게 유지하기 어렵기 때문이다. 임의 append는 미완료 요청의 기존 trace까지 중복 집계할 수 있으므로 하지 말고, 새 run 폴더에서 재실행하여 로그를 분리한다(하네스는 현재 새 로그를 w 모드로 연다).
- **실행한 검증:** 기존 격리 Python 3.12에서 stdlib CPU mock **12개 통과**(`--self-test`): 기존 10개에 dry-run 40건/모델 경로 미호출/환경 보존과 tqdm 뒤 JSON 분리 테스트를 추가했다. 고정된 직렬·분할 옵션의 실제 payload/manifest도 검사한다. 실제 가중치·Kaggle·GPU·학습은 실행하지 않았다. H2/H4는 미확정이며 기본값·OFF 경로·기존 집계기 변경, commit/push는 없다.

### Kaggle 측정 런북 — 최신 절차 (2026-11-22, 모델 로딩 이후 미실행)

이 절차가 앞선 실행 예시보다 우선한다. 하네스는 아직 로컬이며 사용자 push 승인 후 원격 브랜치에서 받아야 한다. 아래 코드는 Kaggle 노트북에서 순서대로 실행하는 Python 셀이다. 학습/GPU 명령을 Codex가 실행한 것이 아니다. 입력이 없거나 PAIR가 없으면 여기서 중단하고 기존 재학습 분기의 명령만 사용자가 검토한다.

**0. 저장소 준비/커밋 확인:** 아래 셀은 대상 디렉터리가 없는 새 세션용이다. 기존 디렉터리가 있으면 덮어쓰거나 삭제하지 말고 현재 checkout/변경 상태부터 확인한다. **새 커밋이 push되기 전 원격 브랜치에는 하네스가 없다.** 승인된 새 커밋과 아래 log가 같은지 확인한 뒤 진행한다.

```bash
%%bash
git clone -b codex/emotion-steering-diagnostics https://github.com/pymood1/GPT-SoVITS-improve.git /kaggle/tmp/GPT-SoVITS
git -C /kaggle/tmp/GPT-SoVITS log -1 --oneline
```

**1. 짝 찾기 셀 — 기존 사용자 제공 셀 그대로:**

```python
import glob, os, sys
REPO = "/kaggle/tmp/GPT-SoVITS"
sys.path.insert(0, f"{REPO}/GPT_SoVITS")
from AR.models.emotion_control import load_emotion_controller_checkpoint, sha256_file

files = sorted({
    p for root in ("/kaggle/working", "/kaggle/input")
    for p in glob.glob(f"{root}/**/*", recursive=True)
    if p.endswith((".pt", ".ckpt")) and os.path.isfile(p)
})
for p in files:
    print(f"{os.path.getmtime(p):.0f}  {os.path.getsize(p)/1e6:8.1f}MB  {p}")

controllers = {}
for p in (f for f in files if f.endswith(".pt")):
    try:
        _, meta = load_emotion_controller_checkpoint(p, map_location="cpu")
        controllers[p] = meta["extra"].get("s1_sha256")
    except Exception:
        pass
print("\ncontrollers:", {p: (h[:12] if h else None) for p, h in controllers.items()})

want = {h for h in controllers.values() if h}
for p in (f for f in files if f.endswith(".ckpt")):
    h = sha256_file(p)
    if h in want:
        for c, ch in controllers.items():
            if ch == h:
                print(f"\nPAIR  controller={c}\n      s1={p}")
                print(f"python tools/diagnose_emotion_steering.py --controller '{c}' "
                      f"--s1-checkpoint '{p}' --output /kaggle/working/steering_static.json")
if not want:
    print("\n해시가 있는 controller가 없음 → 짝 검증 불가")

```

**2. 입력 경로 지정 → CPU 정적 진단:** PAIR의 실제 두 경로를 입력하고 S2/reference도 지정한다. S2의 semantic provenance는 하네스의 기존 TTS loader가 검증한다. 파일명/시각만으로 짝을 정하지 않는다.

```python
import json, subprocess, sys
from pathlib import Path
REPO = "/kaggle/tmp/GPT-SoVITS"
CONTROLLER = "/확인 필요/PAIR의_controller.pt"
S1 = "/확인 필요/PAIR의_s1.ckpt"
S2 = "/확인 필요/학습에_사용한_semantic_S2와_짝인_s2.pth"
REFERENCE = "/확인 필요/reference.wav"
REFERENCE_TEXT = "확인 필요: reference 실제 전사"
STATIC = "/kaggle/working/steering_static.json"
AUDIO_ROOT = "/kaggle/working/steering_audio"
SUMMARY = "/kaggle/working/steering_summary.json"
CONFIG = str(Path(REPO) / "GPT_SoVITS/configs/tts_infer.yaml")
```

**2a. 파일 preflight — 모델 생성/가중치 로딩 전:** 아래는 설정과 소스의 경로만 읽는다. `process_ckpt.get_sovits_version_from_path_fast`는 버전 식별용 해시/헤더를 읽으며 모델을 만들지 않는다. BERT/CNHuBERT 디렉터리의 존재·비어 있지 않음까지만 확인하고 내부 모델 포맷/완전성은 보장하지 않는다.

```python
import ast, os, yaml
os.chdir(REPO)  # TTS와 같은 기준으로 상대 경로를 해석한다.
sys.path.insert(0, str(Path(REPO) / "GPT_SoVITS"))
for label, path in {"controller": CONTROLLER, "S1": S1, "S2": S2,
                    "reference": REFERENCE, "config": CONFIG}.items():
    assert Path(path).is_file(), (label, path)
assert REFERENCE_TEXT.strip() and not REFERENCE_TEXT.startswith("확인 필요"), "reference 실제 전사를 지정하세요"
from process_ckpt import get_sovits_version_from_path_fast
_, VERSION, IS_LORA = get_sovits_version_from_path_fast(S2)
assert VERSION in {"v5dev", "v5turbo"}, VERSION
cfg = yaml.safe_load(Path(CONFIG).read_text())
source = ast.parse((Path(REPO) / "GPT_SoVITS/TTS_infer_pack/TTS.py").read_text())
config_class = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == "TTS_Config")
assignment = next(n for n in config_class.body if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == "default_configs" for t in n.targets))
defaults = ast.literal_eval(assignment.value)
# TTS.py:357–360의 V5 기본값 생성 및 405–408의 YAML top-level override와 동일한 경로 규칙.
v5_defaults = dict(defaults["v4"], version=VERSION,
                   vits_weights_path=f"GPT_SoVITS/pretrained_models/gsv-v5-pretrained/s2G{VERSION}.pth")
fallback = cfg.get(VERSION, v5_defaults)
custom = dict(cfg.get("custom") or cfg.get(VERSION) or {})
for key in ("bert_base_path", "cnhuhbert_base_path"):
    candidate = custom.get(key)
    effective = candidate if candidate and Path(candidate).exists() else fallback[key]
    path = Path(effective)
    print(key, "=>", path.resolve())
    assert path.is_dir() and any(path.iterdir()), (key, str(path))
vocoder = Path(REPO) / "GPT_SoVITS/pretrained_models/gsv-v5-pretrained/vocoder.pth"
print("vocoder =>", vocoder)
assert vocoder.is_file(), str(vocoder)
if IS_LORA:
    semantic_base = Path(fallback["vits_weights_path"])
    print("LoRA semantic base =>", semantic_base.resolve())
    assert semantic_base.is_file(), str(semantic_base)
print("파일 경로 preflight 완료; 모델 초기화/호환성/해시 짝 검증은 별도")
```

경로 근거: `TTS.py:425–441`의 BERT/CNHuBERT 설정·존재 여부 fallback, `611–624`의 실제 loader, `832–835`의 V5 고정 vocoder 파일, `593–609`의 LoRA semantic base 경로다. CNHuBERT는 `feature_extractor/cnhubert.py:27–37`에서 해당 디렉터리를 local_files_only로 읽는다. vocoder 경로는 YAML 항목으로 추측하지 않고 실제 V5 코드의 고정 경로를 사용했다.

**2b. 정적 진단:**

```python
subprocess.run([sys.executable, "tools/diagnose_emotion_steering.py",
                "--controller", CONTROLLER, "--s1-checkpoint", S1,
                "--output", STATIC], cwd=REPO, check=True)
```

이 JSON을 먼저 확인한다. H2/H4 결론은 실행 전 쓰지 않는다. 정적 진단은 CPU FP32이며 추론 FP16과 동일 정밀도라고 간주하지 않는다. 정적/trace의 비교 대상 가중치 해시는 같아야 한다.

**3. 하네스 dry-run → 40건 계획 확인:** 이 셀은 모델을 로딩하지 않는다. 한국어 reference이면 all_ko를 쓴다. 다른 언어 reference는 정확한 기존 prompt_lang 코드로 바꾼다. sampling은 아래 한 목록으로 고정하여 다음 셀도 그대로 사용한다.

```python
HARNESS_ARGS = [
    "--config", CONFIG,
    "--controller", CONTROLLER, "--s1-checkpoint", S1, "--s2-checkpoint", S2,
    "--reference-audio", REFERENCE, "--reference-text", REFERENCE_TEXT,
    "--reference-language", "all_ko",
    "--emotions", "happy", "angry", "disgusted", "surprised",
    "--strengths", "1.0", "2.0", "--seed", "233333",
    "--top-k", "15", "--top-p", "1.0", "--temperature", "1.0",
    "--repetition-penalty", "1.35", "--output-root", AUDIO_ROOT,
]
preview = subprocess.run([sys.executable, "tools/run_steering_trace.py", "--dry-run",
                          *HARNESS_ARGS], cwd=REPO, check=True, capture_output=True, text=True)
plan = json.loads(preview.stdout)
assert len(plan["requests"]) == 40
print(preview.stdout)
```

**3a. 본 실행 전 1–2건 스모크 → trace 식별자/집계 검증:** 아래는 사용자가 실행할 GPU 스모크이며 Codex 미실행이다. `--limit 2`는 전체 계획의 첫 두 요청만 실행한다(같은 문장/감정의 strength 1.0·2.0). 스모크와 본 실행은 별도 run 폴더/로그이며 합쳐서 집계하지 않는다. 스모크의 표본 부족 경고는 예상되지만 schema/identity/runtime 오류는 해결한 뒤 본 실행한다.

```python
smoke = subprocess.run([sys.executable, "-u", "tools/run_steering_trace.py",
                        *HARNESS_ARGS, "--limit", "2"],
                       cwd=REPO, check=True, stdout=subprocess.PIPE, text=True)
print(smoke.stdout)
smoke_lines = [line for line in smoke.stdout.splitlines() if line.startswith("RUN_DIR=")]
assert len(smoke_lines) == 1
SMOKE_DIR = Path(smoke_lines[0].split("=", 1)[1])
SMOKE_TRACE = str(SMOKE_DIR / "steering_trace.log")
events = []
for line in Path(SMOKE_TRACE).read_text().splitlines():
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        continue
    if isinstance(event, dict):
        events.append(event)
assert sum(e.get("event") == "request_marker" for e in events) == 2
assert any(e.get("event") == "emotion_steering" for e in events)
smoke_manifest = json.loads((SMOKE_DIR / "steering_manifest.json").read_text())
assert smoke_manifest["status"] == "complete" and len(smoke_manifest["requests"]) == 2
assert all(r["status"] == "complete" and r["trace_steps"] > 0 for r in smoke_manifest["requests"])
subprocess.run([sys.executable, "tools/diagnose_emotion_steering.py",
                "--trace-log", SMOKE_TRACE, "--static-report", STATIC,
                "--output", str(SMOKE_DIR / "steering_smoke_summary.json")], cwd=REPO, check=True)
```

**4. 하네스 실행 → 로그/manifest/WAV 보존:** T4에서는 기본 CUDA/FP16. `is_half=False`를 환경에 설정했다면 FP32이므로 청취 설정과 같게 유지한다. trace ON은 하네스가 자식 프로세스에서 설정하므로 1번 셀의 import 상태에 영향받지 않는다. 아래는 **GPU 실행이며 Codex 미실행**이다.

```python
completed = subprocess.run([sys.executable, "-u", "tools/run_steering_trace.py", *HARNESS_ARGS],
                           cwd=REPO, check=True, stdout=subprocess.PIPE, text=True)
print(completed.stdout)
run_lines = [line for line in completed.stdout.splitlines() if line.startswith("RUN_DIR=")]
assert len(run_lines) == 1
RUN_DIR = Path(run_lines[0].split("=", 1)[1])
TRACE = str(RUN_DIR / "steering_trace.log")
MANIFEST = str(RUN_DIR / "steering_manifest.json")
manifest = json.loads(Path(MANIFEST).read_text())
assert manifest["status"] == "complete"
assert len(manifest["requests"]) == 40
print("audio folder:", RUN_DIR, "trace:", TRACE, "manifest:", MANIFEST)
print("actual runtime:", manifest["actual_runtime"], "trace runtime:", manifest["trace_runtime"])
```

stderr는 노트북에서 바로 보이며 실행 폴더의 TRACE에도 저장한다. stdout은 이 셀의 완료 후 출력된다. 각 요청 WAV는 RUN_DIR에, 입력 계획/완료 상태/실제 runtime은 MANIFEST에 저장된다. 실패하면 check=True가 중단하므로 다음 집계를 진행하지 않는다. 청취 baseline은 trace OFF로 별도 생성한다.

**5. 같은 RUN_DIR의 trace + 2번 STATIC → 집계:**

```python
subprocess.run([sys.executable, "tools/diagnose_emotion_steering.py",
                "--trace-log", TRACE, "--static-report", STATIC,
                "--output", SUMMARY], cwd=REPO, check=True)
print("static:", STATIC, "summary:", SUMMARY)
```

**정렬 근거/한계:** `inference_webui.py:90,153–156`의 기본은 CUDA 가능 시 cuda/is_half=True, CPU에서는 half=False다. 하네스의 `webui_runtime_defaults`가 같은 기본과 보통의 is_half=True/False(1/0) 환경 설정을 사용하며 `_CUDA_VISIBLE_DEVICES`도 import 전에 반영한다. Python 표현식을 eval하는 특수 환경값은 안전상 지원하지 않고 오류를 낸다. 원본 YAML은 수정하지 않는다. 실제 S1 readout의 dtype/device, TTS config의 is_half, batch_size/cut을 manifest.actual_runtime에 기록하고, 관측 trace.runtime의 probability_dtype/score_dtype/device_type/sampling을 매 요청 검증하여 다르면 성공 처리 전에 중단한다. 전체 device 문자열(cuda:0 등)과 device_type(cuda)은 의미에 맞게 비교한다. trace에 is_half/batch/cut 필드는 없으므로 manifest에 별도로 남긴다.

**미정렬/미검증:** 고정 seed/직렬/batch=1/cut1은 측정 통제 조건이며 사용자가 실제 WebUI에서 바꾼 seed·cut·기타 UI 설정을 자동으로 알아내지는 않는다. 환경변수로 바꾼 청취 설정도 사용자가 일치시켜야 한다. 실제 T4 하드웨어·driver·커널 결정성과 WebUI/하네스 음성 동일성은 미검증이다. 정적 FP32와 추론 FP16의 수치 차이는 분리해서 해석한다. 포화율·active-only TV·budget 차단률을 함께 보고 단일 지표로 원인을 확정하지 않는다.

**시간/로그 용량:** 실제 40요청의 실행 시간 및 stderr 크기는 현재 모른다. 실제 생성 step 수·분할 수·스텝별 JSON 바이트 수·T4 동기화 지연을 측정하지 않았으므로 숫자 범위를 만들지 않는다. 대략적인 로그 크기는 모든 step JSON의 UTF-8 바이트 합 + marker/진행 로그이며 시간은 모델 준비 + 40요청 생성/동기화/저장 시간이다. 매 step GPU→CPU 동기화 때문에 trace OFF 성능으로 추정하면 부정확하다.

**push 준비만:** 기존 `codex/emotion-steering-diagnostics`에 새 commit을 만들 계획이며 amend/force/main 변경은 없다. 승인 후에만 `git add -- tools/run_steering_trace.py CODEX_ACTIVE_RESEARCH_HANDOFF.md`로 stage한다. 테스트는 하네스 안에 있으므로 별도 테스트 파일은 없다. 예상 `git diff --cached --stat` 파일 목록은 이 두 경로뿐이다(하네스 신규, handoff 수정). 실제 cached 목록을 확인해 다른 파일이 섞이면 중단한다. AGENTS.md·tts_infer.yaml·미추적 Sigma/역사 파일은 제외한다. 현재는 stage/commit/push하지 않았다.

승인 후 새 commit 메시지에는 **하네스 추가, Kaggle 미실행, 기본 동작 변경 없음**을 명시한다. 인용된 승인 예문은 실제 승인이 아니다.

**흔한 실패 — 메시지는 현재 소스 그대로, marker/trace 행은 관측 점검:**

| 메시지/관측 | 원인 / 해야 할 일 |
| --- | --- |
| `Controller has no bound S1 SHA-256; pairing cannot be verified` | 짝 해시 없음 / 검증 가능한 현재 controller를 확보한다(`diagnose_emotion_steering.py:366`). |
| `Controller/S1 SHA-256 pairing mismatch` | 다른 S1 / 해시가 맞는 S1을 지정한다(`diagnose_emotion_steering.py:369`). |
| `Trace runtime differs from loaded model/settings:` | 하네스가 로딩 상태와 다른 runtime을 관측 / dtype/device/sampling을 확인한다(`run_steering_trace.py::check_trace_runtime`). |
| `Runtime dtype/device/sampling settings differ; summarize separately` | 서로 다른 실행 설정이 섞임 / 로그를 설정별로 분리한다(`diagnose_emotion_steering.py:210`). |
| `Trace schema 2 required; regenerate with the current logger` | 구형 trace / 현재 logger로 다시 생성한다(`diagnose_emotion_steering.py:149`). |
| `No usable emotion steering trace records` | 사용 가능한 trace 없음 / 스모크의 trace ON·로그 경로·완료 상태를 확인한다(`diagnose_emotion_steering.py:208`). |
| marker와 trace 개수가 다름(오류 메시지 아님) | marker는 요청당 1개, trace는 step당 1개이므로 정상일 수 있음 / marker=완료 요청 수와 각 manifest 요청의 trace_steps>0을 확인한다. |

**최종 로컬 점검:** `--limit` 추가 후 CPU mock **15개 통과**(추가 테스트 1개: 2건 실행/manifest·dry-run·0/음수 거부). 독립 프로세스 `--dry-run` 기본/`--limit 1`/`--limit 2`는 각각 **40/1/2건**이었다. 실제 `--help`의 --limit, 런북 Python 셀 **8개**의 문법, TTS_Config 기본 경로 dictionary의 AST 추출, Python 3.10 문법 파싱을 확인했다. preflight는 경로 검사 셀만 작성했으며 실제 Kaggle 자산으로 실행하지 않았다. 스모크/본 실행/실제 trace 집계도 미실행이다. staged diff는 비어 있고 변경 대상은 계속 하네스(테스트 내장)·handoff 2파일뿐이다. 승인 전 stage/commit/push하지 않았고 H2/H4는 미확정이다.

**이번 CPU 확인 결과:** 기존 격리 Python 3.12에서 하네스 mock **14개 통과**(T4/CPU 기본값, 환경변수 반영, 실제 runtime manifest, trace runtime 불일치 거부 포함). 두 스크립트의 실제 `--help`를 실행해 위 명령의 모든 CLI 옵션을 확인했다. 런북 Python 셀 5개 문법 파싱, 런북의 HARNESS_ARGS 그대로 사용한 독립 프로세스 dry-run의 **40건/문장 5개/strength 1.0·2.0/seed 233333**, 하네스 Python 3.10 문법 파싱도 통과했다. 이 검증은 입력 placeholder로 모델 없이 진행했으며 짝 찾기·실제 정적 진단·모델 초기화·40건 생성·실제 trace 집계는 미실행이다. 현재 branch는 codex/emotion-steering-diagnostics이며 staged diff는 비어 있었다. 원래 TTS/WebUI/Controller/학습/집계기/기존 진단 테스트 파일은 HEAD 대비 변경 없음도 확인했다.
