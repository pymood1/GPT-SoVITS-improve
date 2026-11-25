# GPT-SoVITS Grokking Probe

This branch adds an opt-in delayed-generalization experiment path for both
GPT/S1 and SoVITS/S2. Normal training is unchanged unless the grokking switch is
enabled.

## Why this exists

A grokking test needs three things the normal training path did not guarantee:

1. a truly held-out split with no duplicate leakage;
2. continued optimization after the training set is already fit;
3. metrics that can show a delayed validation improvement.

The S1 path additionally offers AdamW weight decay and Grokfast-style EMA
slow-gradient amplification. The S2 path keeps the discriminator unchanged and
applies Grokfast only to the generator.

## Enable

Global:

```bash
export GPT_SOVITS_GROKKING=1
```

S1 only:

```bash
export GPT_SOVITS_GROKKING_S1=1
export GPT_SOVITS_GROKKING_S2=0
```

S2 only:

```bash
export GPT_SOVITS_GROKKING_S1=0
export GPT_SOVITS_GROKKING_S2=1
```

The existing WebUI/training launch can then be used normally. The feature is
read from the training process environment.

## Conservative defaults

- validation fraction: 10%
- minimum validation samples: 8
- Grokfast EMA alpha: 0.98
- Grokfast lambda: 0.5
- S1 experimental optimizer: AdamW
- S1 learning rate: 1e-4
- S1 weight decay: 0.01
- S1 DPO during grokking: off by default (explicit opt-in available)
- S1 post-fit window: 20 epochs
- S1 precision in grokking mode: FP32
- S1 LR in grokking mode: fixed at the configured 1e-4 by default
- S2 generator weight decay: 0.01
- S2 post-fit observation window: 8 epochs
- S2 train-mel plateau patience: 3 epochs
- S2 significant train-mel improvement threshold: 1%
- S2 held-out validation cap: 32 batches

All are overridable with environment variables defined in
\`tools/grokking.py\`.

## S1 outputs

- \`<output_dir>/grokking_s1_metrics.jsonl\`
- \`<half_weights_dir>/<exp_name>-grok-best.ckpt\`

The JSONL records train top-3, held-out loss/top-3, parameter L2 norm,
memorization epoch, post-fit age and whether validation improved after the
memorization threshold.

S1 training allows a hard ceiling of:

```text
base_epochs + 4 * post_fit_epochs
```

but stops once the configured full post-fit window has elapsed after the first
memorization-threshold crossing. This prevents ordinary early stopping from
cutting off a delayed transition.

## S2 outputs

- \`<exp_dir>/grokking_s2_metrics.jsonl\`
- compact \`<name>_grok_best\` weight through the normal \`savee\` path

S2 validation uses held-out generator mel L1 reconstruction. It deliberately
does not use discriminator loss as the generalization signal, because GAN game
balance is not a clean held-out quality metric. Validation RNG is forked and
reseeded to the same value every epoch so stochastic latent/slice choices do not
masquerade as generalization.

S2 also tracks the distributed mean training mel L1. A fit point is declared
after the configured plateau patience passes without a significant relative
training-mel improvement. Training then preserves a full post-fit observation
window. The hard ceiling is currently base_epochs + 4 * post_fit_epochs; normal
runs usually stop earlier once the post-fit window completes.

## Interpretation

A useful S1 grokking candidate has this ordering:

```text
train top-3 saturates first
        ->
validation remains poor for a delay
        ->
validation loss later falls / validation top-3 rises
        ->
parameter norm typically contracts or reorganizes
```

For S2, call the result "grokking-like delayed generalization" unless the
held-out metric shows a clear delayed transition after an independently defined
training-fit point. Acoustic GAN training does not have the same clean
memorization boundary as semantic-token prediction.

## Safety / baseline isolation

With all grokking environment switches unset, the normal data sharing,
optimizer, precision, epoch count and validation-disabled behavior remain
unchanged.

For a rigorous probe, do not use a user-finetuned pretrained checkpoint that
has already seen the utterances selected for the new held-out split. Generic
upstream pretrained weights are acceptable. S1 Lightning resume checkpoints
are isolated by split seed/fraction/minimum-validation settings, and S2
grokking mode intentionally ignores existing experiment checkpoints.

The S1 grokking path uses the deterministic CE objective by default even when
the normal training config enables DPO. Set
`GPT_SOVITS_GROKKING_S1_KEEP_DPO=1` only for a separate DPO ablation.

The S1 grokking path intentionally does not use this fork's legacy
WarmupCosineLRSchedule because that implementation hard-locks the learning rate
to 0.002. The research path uses a standard PyTorch constant-LR scheduler so
eta and weight decay are controlled variables.


## S1 measurement hardening (2026-11-25)

The S1 validation loader intentionally uses one utterance per batch.
The decoder returns summed CE (including the real EOS) and a Python float
top-3 accuracy (excluding EOS). The probe now records token-normalized CE
and weights epoch aggregation by the corresponding token count; train
top-3 is also token-weighted in probe mode. The baseline objective and
non-probe logs remain unchanged. Validation explicitly rejects batch sizes
other than one because the legacy CE includes padding EOS in larger batches.

Do not compare the new val_loss numerically with logs from the original
summed-per-utterance probe. Use a fresh experiment and generic upstream
pretrained weights, not the user's previous fine-tuned checkpoints.

The delayed_improvement flag is a conservative candidate screen:
- a finite, valid held-out measurement at the first training fit point;
- at least two epochs after that point, with training top-3 still above
  the memorization threshold;
- held-out CE at least 5% below the fit-point CE and top-3 at least
  5 percentage points higher;
- CE better than the pre-fit best (if one existed).

These thresholds are heuristic, not a statistical test of grokking.
Inspect the complete train/validation curves, repeat with other split
seeds, and evaluate fresh-reference synthesis before drawing conclusions.

### Initial Kaggle ablation

Use the same generic pretrained S1, data split, seed, batch size,
AdamW LR, and epoch budget for these arms:
1. Grokfast OFF, weight decay 0.01.
2. Grokfast ON, weight decay 0.01.
3. Grokfast OFF, weight decay 0 (control).
4. Optional Grokfast ON, weight decay 0.

Run S1 only first (S2 fixed); disable DPO for this research comparison.
Keep the existing normal-training defaults (including DPO) outside the
probe. Use independent fresh experiment names/output paths to prevent
automatic checkpoint reuse and JSONL mixing. A single successful
held-out curve is not evidence of improved generated speech.


## 2026-11-25: V5 CFM training path and Kaggle bridge

The WebUI routes v5dev/v5turbo (and v3/v4) to
`GPT_SoVITS/s2_train_v3_lora.py`, not the older GAN trainer.
The opt-in S2 probe now also runs in this LoRA CFM path:

- canonical utterance-name split and matching bucket order across DDP ranks;
- fresh generic upstream pretrained initialization, no automatic resume;
- fixed AdamW learning rate and configurable weight decay;
- Grokfast on trainable parameters after AMP unscale; skip EMA updates
  when any current gradient is nonfinite;
- rank-zero held-out CFM loss (batch size 1, two fixed torch RNG draws per
  utterance, evaluation in FP32), with other ranks synchronized;
- train-CFM plateau + bounded post-fit window, heuristic candidate flag;
- best compact LoRA export and `grokking_s2_cfm_metrics.jsonl`;
  no ordinary full checkpoint saves in this research path.

The CFM loss is a flow-matching velocity loss, not mel reconstruction error,
and does not establish speech naturalness or speaker similarity. The held-out
objective is teacher/reference conditioned; evaluate separately on new text
and independent reference audio. The probe supports filename/identity
deduplication, not acoustic-near-duplicate detection. CFM evaluation uses a
limited held-out subset if max_validation_batches is smaller than the split.
In training the CFM probe preserves the selected FP32/AMP precision.
The plateau and 5% improvement thresholds are engineering heuristics.

S1 now refuses old research metrics/checkpoints instead of auto-resuming an
experiment with possibly different optimizer, split or Grokfast state.
Choose a new experiment name for every arm. S2 CFM also refuses existing
probe output. Normal mode continues using the original training logic.

### Kaggle bridge

The one-time connectivity and private synthetic GPU smoke workflows were
retired after successful validation. Historical evidence is preserved in
Git history and Actions: run 45565061672 recovered the result of the
private kernel from run 45521697065 (attempt 2), confirming two Tesla T4
GPUs, dataset mount, and all four FP32/AMP16 optimization tests.

The active workflow is
`.github/workflows/kaggle-real-model-research.yml`. An explicit change to
`tools/kaggle_remote_request.json` requests a private, commit-pinned model
experiment or result-only recovery. It uses the GitHub Actions secret
`KAGGLE_API_TOKEN` (preferred) or `KAGGLE_KEY` (legacy) and never prints
credentials. Only allowlisted numeric metrics and status are returned to
GitHub; audio, transcripts, weights and raw logs remain private in Kaggle.
Do not put credentials in source, notebook cells, logs, comments or chat.

The active workflow separates queue and execution time and reserves
40 minutes in addition to the training budget for setup, preprocessing,
and finalization. A retry can specify the previous kernel so the new
submission is blocked unless the previous kernel is confirmed unsuccessful
or cancellation is acknowledged. A skipped job is not proof of GPU success.
The first real-model automation run was cancelled; its replacement
run 45570042738 is the active validation as of this change. Ordinary
regression CI and the manual notebook remain supported.


## 2026-11-25: First real Kaggle S1 ablation and next probe

Run `20261125T013234_805522Z` (commit `2507958`): both single-T4 FP32
arms loaded the generic S1 checkpoint with all keys matched and completed
17 epochs. 244 training and 27 held-out utterances; one filtered sample.
Grokfast OFF/ON (alpha .98, lambda .5, AdamW LR 1e-4, WD .01):

| Metric | OFF | ON |
|---|---:|---:|
| best held-out CE | 3.108679 | 3.228725 |
| final held-out CE | 4.815629 | 5.089155 |
| final train top-3 | .987593 | .989849 |
| final held-out top-3 | .346782 | .336310 |
| memorization epoch | 13 | 13 |
| delayed candidate | none | none |

The observed window shows overfitting, not delayed generalization.
There was no epoch-zero measurement; both stopped four epochs after
the fit point, so the result does not rule out a later transition.
The split is utterance-level within three long recordings, not an
independent recording or independent speaker test.

Next runner profile `s1_weight_decay_long_v2`: same generic checkpoint,
seed, batch size, BERT and DPO settings; Grokfast OFF for both arms;
WD .01 versus 1.0; fixed 120 epochs (90-minute safety limit).
This is an exploratory comparison, not a validated training recipe.
A full held-out sanity pass records the untouched pretrained epoch-zero
baseline separately without advancing the phase tracker. If Lightning
does not expose the sanity metrics, the baseline is missing/null, not
a fabricated zero. Fixed epoch budgets are opt-in via
`GPT_SOVITS_GROKKING_S1_FIXED_EPOCHS`; no change to ordinary training.
Compare both curves at the same epoch; use the baseline as an external
reference as the candidate tracker still compares only trained epochs.
No inference quality or S2 GPU result has yet been established.


## 2026-11-25: Fixed-budget S1 result and next S2 probe

Run `20261125T015120_681594Z` (commit `cd7a5688`) completed both
120-epoch FP32 arms, each with 244 training and 27 validation utterances.
Both pretrained epoch-zero measurements were exactly CE 3.5062289,
top-3 .3378503. At WD .01 vs 1.0 respectively:
- Best CE: 3.1085644 vs 3.1081882 (epoch 1).
- Peak top-3: .4197721 vs .4200801 (epoch 2).
- Final CE: 6.7882800 vs 5.9673133.
- Final top-3: .3119803 vs .3024330.
- Final train top-3: .9991707 vs .9992039.
- First memorization: epochs 13 vs 14; no delayed candidate.
- Parameter L2 (epoch 1 to 120): 2732.02 to 2723.04 vs
  2723.92 to 1907.83. Stronger decay reduced CE deterioration
  but did not rescue top-3 or induce delayed generalization.

These are single-split teacher-forced metrics, not listening scores;
the experiment cannot exclude grokking under other training regimes.
Do not deploy a late checkpoint on the basis of training accuracy.

The next default runner profile is `s2_v5dev_cfm_grokfast_v1`:
two independent single-T4 V5dev CFM runs from the same generic base,
Grokfast OFF/ON, WD .01, LR 1e-4, FP32, batch 1, LoRA rank 16,
gradient checkpointing enabled, 12 fixed epochs, 90-minute training
time cap. Both use identical held-out split seed and fixed validation
noise draws. The epoch-zero CFM baseline is recorded separately.
This is an initial feasibility and short-horizon comparison, not
a sufficient test to rule out long-horizon S2 grokking. No S1 training.
The previous S1 profile remains available through
`GPT_SOVITS_RESEARCH_PROFILE=s1` (default s2). Each S2 arm links
immutable preprocessed text/HuBERT/audio files into a separate writable
experiment directory and exports its best compact weights only to tmp.
The single current-run `/kaggle/working/grokking_result.zip` contains
diagnostics and metrics, not raw audio or checkpoints.


## 2026-11-25: First real V5dev CFM result

Run `20261125T022754_360839Z` at commit `3468d1d`:
both independent FP32 T4 arms completed all 12 epochs with exit 0,
245 train / 27 held-out utterances, and all pretrained keys matched.
Epoch-zero held-out CFM = 0.2561326109700733 in both arms.

| Metric | Grokfast OFF | Grokfast ON |
|---|---:|---:|
| Final online train CFM | .0816180214 | .0808058541 |
| Final held-out CFM | .0560889461 | .0559793851 |
| Best held-out CFM | .0560889461 (12) | .0559755704 (11) |
| Fit epoch (old online-loss heuristic) | 6 | 6 |
| Delayed candidate | none | none |

Grokfast ON had lower held-out CFM at every recorded epoch but the
final relative advantage was only 0.195%. Both arms improved mostly
at the first epoch. This is conventional gradual improvement, not
evidence of grokking or verified synthesis quality. The online train
loss varies with random draws, model train mode, and evolving weights;
it is not directly comparable with the fixed eval-mode held-out loss.

Next default profile `s2_v5dev_cfm_grokfast_long_v2` keeps the prior
optimizer, split, rank, precision and batch size; increases the fixed
budget to 30 epochs (90-minute cap remains). In research mode only,
an additional fixed, unique train subset (up to the held-out count)
is evaluated with the same eval-mode two-draw protocol. The plateau
heuristic now uses this train-probe CFM and resets a stale fit point
when a significant train-probe improvement occurs. Epoch-zero and
each epoch include per-utterance mean CFM values for paired analysis,
plus aggregate values and measured epoch duration. The old held-out
draw protocol is preserved. Candidate flags between old and new
protocols are not directly interchangeable. No audio synthesis quality
claim is made. The runner displays active epoch, batch, stage, last
validation loss and best-effort GPU utilization/memory via nvidia-smi.
