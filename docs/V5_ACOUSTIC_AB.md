# V5 acoustic-only noise isolation

Baseline: fork main `a43dfd1df262f773e2c0d77f011ba6dc38200934`.
Official source pin: `RVC-Boss/GPT-SoVITS` main
`6d51ffceafbf240145619af4c92408a6e54b7924` (checked 2026-11-15).

This is a diagnostic, not a new inference architecture or a noise fix.
The earlier V5 routing correction is already present. S1 training, DPO and
emotion-controller behavior are unchanged.

## Established source facts

- `module/models_v5.py`, `module/models.py`, and `module/mel_processing.py`
  are byte-identical to the pinned upstream versions (normalized LF checkout).
- Both acoustic paths use 32 kHz reference mel, FFT/window 1280, hop 320,
  100 mel bins, fmin 0, fmax None, center False; normalization bounds -12..2.
- V5dev defaults now use 32 steps / CFG 1.30 and the same V5 helper. Shared
  runtime noise defaults are no longer forcibly disabled in the general WebUI.
- Source equality is not audio-quality evidence. Runtime checkpoint contents,
  input dtype, generated mel and the actual symptom require the user's runtime.

## Enable in Kaggle

Use the diagnostic branch/PR source, then set this variable **before starting
the inference WebUI process** (changing notebook variables cannot modify an
already running WebUI's environment):

```python
import os
os.environ["GPT_SOVITS_V5_ACOUSTIC_AB_DIR"] = "/kaggle/working/v5_acoustic_ab"
```

If the main WebUI was already started without the variable, launch a separate
normal inference WebUI directly from a notebook cell instead. Stop the previous
inference server first to avoid its port conflict; leave training/runtime intact:

```python
import os
import subprocess
env = dict(os.environ, language="ko_KR", version="v5dev",
           GPT_SOVITS_V5_ACOUSTIC_AB_DIR="/kaggle/working/v5_acoustic_ab")
process = subprocess.Popen(
    ["python", "GPT_SoVITS/inference_webui.py", "ko_KR"],
    cwd="/kaggle/tmp/GPT-SoVITS", env=env,
)
```

Select base V5dev SoVITS, the same GPT, a clean 3–10 s reference with its exact
text, and a short target sentence with no emotion tag. Press synthesize once.
Do not run training or another synthesis/model switch concurrently with this
in-process diagnostic. It fetches only four public pinned source files; all
private tensors and WAVs stay in the runtime. No upload occurs. Internet must be
enabled for that public source read. Unset the variable and restart the inference
process when done. Normal behavior is untouched when the variable is absent.

## What executes

The hook runs on the first text fragment after S1 sampling. It clones the exact
`pred_semantic`, prompt semantic and phones; no S1 is sampled inside the A/B.
With the same loaded eval-mode S2/vocoder, dtype, speed, steps and CFG:

1. A: execute the actual V5 acoustic branch extracted from current WebUI source.
2. A-repeat: same branch, same CPU/CUDA seed (1234), to measure kernel/state drift.
3. B: execute the pinned upstream TTS `_get_ref_spec()` and
   `using_vocoder_synthesis()` methods plus its own resampling/mel/normalization
   definitions. Do not substitute the obsolete upstream general WebUI.

Upstream shared model/mel implementations must match the installed source AST;
otherwise execution fails closed instead of claiming an official comparison.
S1/frontend packages are not loaded again. The engine object is an acoustic-only
facade, sharing the already loaded model/vocoder, not a second full TTS instance.

Source extraction/instrumentation records tensor snapshots without changing
equations. It compares reference spectrum, final reference audio, `fea_ref`,
`ge`, normalized `mel2`, `fea_todo`, returned target `ge`, each CFM input/length/
prompt/output, helper output, denormalized mel, vocoder input and raw waveform.
Shape/dtype/device/nonfinite count/min/max/mean/std/RMS and exact equality/max
absolute difference are reported. The initial noise is reconstructed from the
same RNG state in a nested RNG-preserving scope; it is not a hook inside randn.
Actual per-call steps/CFG/temperature are also recorded. Runtime RNG states and
temporary method wrappers are restored, including on failure.

The full case uses every token (budget: at most 1000 tokens). If more than 100
tokens, an additional clearly labeled 100-token prefix isolates short/single-
chunk behavior. This is a semantic prefix, not a separately generated short
sentence; begin with a genuinely short target sentence for the primary test.
The output prefix is measured independently; no long-text result is substituted
for short-text evidence. With one short case: 3 acoustic executions plus one
vocoder-only reference reconstruction. With the additional full case: 6 executions.

## Artifacts and interpretation

Each synthesis creates a unique directory and prints its location:

- `report.json`: source hashes, selected checkpoint SHA-256, loaded vocoder/S2
  state-to-file checks after dtype casting, runtime flags and stage comparisons.
  Missing checkpoint keys are explicit; common-key equality alone is not full
  checkpoint equivalence. No private transcript is included in the report.
- `private_inputs.pt`: exact runtime-local semantic/prompt/phones/reference path
  for later replay. Contains private derived data; do not publish it.
- `*_webui_A.wav`, `*_webui_A_repeat.wav`, `*_official_B.wav`: raw FLOAT32,
  no individual normalization/clipping. Peaks above 1 remain in these files.
- `*_listen.wav`: browser-compatible PCM16 copies; A, A-repeat, B use one common
  gain, recorded in the report. Prefer these for listening to avoid player
  clipping masquerading as model noise.
- `reference_vocoder_roundtrip_listen.wav`: actual reference mel -> norm ->
  denorm -> same vocoder, completely bypassing CFM and S1. Its own gain is recorded;
  use it to assess artifacts, not as a loudness-matched A/B candidate.

Play generated files inside the same private Kaggle session:

```python
from pathlib import Path
from IPython.display import Audio, display
folder = sorted(Path("/kaggle/working/v5_acoustic_ab").iterdir())[-1]
for path in sorted(folder.glob("*_listen.wav")):
    print(path.name)
    display(Audio(filename=str(path)))
```

Interpret in order:

1. A-repeat differs: establish this drift floor before interpreting A/B.
2. A/B first differs at reference/mel/features: inspect that input boundary.
3. Inputs/noise match but CFM output differs: isolate CFM runtime/kernel/state.
4. Normalized mel matches but vocoder input differs: inspect denorm/dtype.
5. Vocoder inputs match but waveform differs: isolate vocoder/runtime state.
6. A/B audio matches exactly and both sound noisy: route difference is excluded
   for that sample; inspect the common upstream path, asset identity, reference
   and precision. It does not implicate S1 training.
7. Reference roundtrip also noisy: prioritize vocoder/mel/assets/precision.
   Clean roundtrip but noisy A/B narrows toward generated acoustic features/CFM;
   this still needs a controlled intervention before declaring causality.

Do not label any first numerical difference as the first audible noise without
listening and a single-factor follow-up. No fp16/bf16/AMP sweep or runtime flag
change is silently applied. This first comparison measures the actual dtype/
autocast state; any precision intervention must reuse `private_inputs.pt` with
the same checkpoint/reference and change only that precision setting.

## Local evidence

CPU synthetic fixtures execute both real source paths and the real V5 helper
with a small fake S2/CFM/vocoder. Short and multi-chunk A/A-repeat/B agree; a
deliberate +0.25 normalized-mel perturbation is first detected at `mel2[1]`.
RNG/method restoration and NaN-safe reporting are covered. These tests validate
the diagnostic, not real-model synthesis or sound quality. Real Kaggle A/B and
the first audible-noise stage remain unverified until the runtime test above.
