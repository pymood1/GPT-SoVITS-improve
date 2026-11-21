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

\`\`\`bash
export GPT_SOVITS_GROKKING=1
\`\`\`

S1 only:

\`\`\`bash
export GPT_SOVITS_GROKKING_S1=1
export GPT_SOVITS_GROKKING_S2=0
\`\`\`

S2 only:

\`\`\`bash
export GPT_SOVITS_GROKKING_S1=0
export GPT_SOVITS_GROKKING_S2=1
\`\`\`

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
- S2 generator weight decay: 0.01
- S2 extra observation budget: 8 epochs
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

\`\`\`text
base_epochs + 2 * post_fit_epochs
\`\`\`

but stops once the configured full post-fit window has elapsed after the first
memorization-threshold crossing. This prevents ordinary early stopping from
cutting off a delayed transition.

## S2 outputs

- \`<exp_dir>/grokking_s2_metrics.jsonl\`
- compact \`<name>_grok_best\` weight through the normal \`savee\` path

S2 validation uses held-out generator mel L1 reconstruction. It deliberately
does not use discriminator loss as the generalization signal, because GAN game
balance is not a clean held-out quality metric.

## Interpretation

A useful S1 grokking candidate has this ordering:

\`\`\`text
train top-3 saturates first
        ->
validation remains poor for a delay
        ->
validation loss later falls / validation top-3 rises
        ->
parameter norm typically contracts or reorganizes
\`\`\`

For S2, call the result "grokking-like delayed generalization" unless the
held-out metric shows a clear delayed transition after an independently defined
training-fit point. Acoustic GAN training does not have the same clean
memorization boundary as semantic-token prediction.

## Safety / baseline isolation

With all grokking environment switches unset, the normal data sharing,
optimizer, epoch count and validation-disabled behavior remain unchanged.
