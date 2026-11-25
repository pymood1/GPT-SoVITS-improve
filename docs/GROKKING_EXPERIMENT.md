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

The credential-free workflow `.github/workflows/kaggle-connectivity-smoke.yml`
verified public DNS/HTTP, Kaggle CLI 2.2.5, and the T4 x2 accelerator option.

A dedicated private GPU smoke workflow is available at
`.github/workflows/kaggle-private-gpu-smoke.yml`. It only submits a GPU job
on an explicit head commit modifying that workflow (or an explicit rerun
of such a commit), on this repository's research branch. It checks for a
dedicated GitHub Actions secret named `KAGGLE_API_TOKEN` (preferred) or
`KAGGLE_KEY` (legacy, username fixed to the repository owner's Kaggle ID).
It does not print credentials. Do not place tokens in source, notebook cells,
logs, issue comments, or chat. The private kernel uses synthetic data only,
tests FP32 and AMP on each T4, and checks that the existing dataset directory
is mounted. No source audio or learned weights are returned to GitHub.

A missing credential deliberately blocks submission. A successful workflow
with skipped GPU steps is **not** GPU proof. Check the submission and
metrics-retrieval steps individually. A completed GPU smoke is a prerequisite
for actual pretrained model training, not a substitute for it.
