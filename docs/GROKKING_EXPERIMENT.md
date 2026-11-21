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
- S1 post-fit window: 20 epochs
- S1 precision in grokking mode: FP32
- S1 LR in grokking mode: fixed at the configured 1e-4 by default
- S2 generator weight decay: 0.01
- S2 post-fit observation window: 8 epochs
- S2 train-mel plateau patience: 3 epochs
- S2 significant train-mel improvement threshold: 1%
- S2 held-out validation cap: 32 batches

All are overridable with environment variables defined in
\`GPT_SoVITS/tools/grokking.py\`.

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

The S1 grokking path intentionally does not use this fork's legacy
WarmupCosineLRSchedule because that implementation hard-locks the learning rate
to 0.002. The research path uses a standard PyTorch constant-LR scheduler so
eta and weight decay are controlled variables.
