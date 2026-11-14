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
        -> automatic emotion sidecar if missing
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
