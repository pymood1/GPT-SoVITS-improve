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

In the existing installed Kaggle GPU session, run this ONE cell:

```python
import urllib.request
url = "https://raw.githubusercontent.com/pymood1/GPT-SoVITS-improve/codex/v5-acoustic-ab/tools/kaggle_v5_acoustic_ab.py"
scope = {"__name__": "v5_ab_bootstrap"}
exec(compile(urllib.request.urlopen(url).read(), url, "exec"), scope)
v5_ab_process = scope["launch"]()
```

The cell clones/updates a separate diagnostic checkout, links existing assets,
and starts its own inference server on an available port. It does not reset the
original source, install packages, restart the runtime or kill existing jobs.
It checks for GPU and required installed pretrained assets and reuses an already
running diagnostic server when the cell is repeated. It assumes the normal setup
has already completed; missing assets are reported explicitly.

Open the printed public WebUI link. Base GPT `s1v3.ckpt` and base V5dev SoVITS
`s2Gv5dev.pth` are preselected. A Korean reference and its exact transcript are
selected from the existing training list when available; otherwise upload a
3–10 second reference and enter its exact transcript in the normal fields.
No emotion tag or trained GPT/SoVITS is needed for this first test.

The target is prefilled: **안녕하세요. 오늘은 날씨가 좋아서 잠깐 산책을 다녀오려고 합니다.**
Press synthesize ONCE and wait for all sequences, including FP32, to finish.
Players appear automatically in the notebook cell in listening order. This is
8 acoustic passes plus reference-vocoder probes, not one ordinary inference;
model construction/FP32 on a T4 can take several minutes.
Do not run training or another synthesis/model switch concurrently with this
in-process diagnostic. It fetches only four public pinned source files; all
private tensors and WAVs stay in the runtime. No upload occurs. Internet must be
enabled for that public source read. Unset the variable and restart the inference
process when done. Normal behavior is untouched when the diagnostic variable is absent.

## What executes

The hook runs on the first text fragment after S1 sampling. It clones the exact
`pred_semantic`, prompt semantic and phones; no S1 is sampled inside the A/B.
Each sequence starts with a fresh checkpoint-backed eval-mode S2/vocoder pair;
the forward/reverse sequences deliberately share that pair internally:

1. Forward: A -> A-repeat -> B, same pair and seed 1234 for each invocation.
2. Reverse: B -> A, starting with another fresh pair.
3. Cold A and cold B: each starts with its own fresh pair.
4. FP32 B: another fresh pair loaded from original checkpoint tensors, then
   all floating weights/buffers cast to FP32. CUDA/CPU autocast and TF32 disabled
   only inside this scope, restored afterwards. If files themselves store FP16,
   this does not recover lost weight precision; it tests FP32 arithmetic.

A executes the current actual WebUI branch; B executes upstream TTS `_get_ref_spec()` and
   `using_vocoder_synthesis()` methods plus its own resampling/mel/normalization
   definitions. Neither path substitutes the obsolete upstream general WebUI.

Upstream shared model/mel implementations must match the installed source AST;
otherwise execution fails closed instead of claiming an official comparison.
S1/frontend packages are not loaded again. The engine object is an acoustic-only
facade, not a second full TTS instance. Current-precision fresh models use exact
snapshots of the loaded weights/buffers, preserving parameter dtypes. FP32 uses
the checkpoint weights rather than upcasting rounded live FP16 weights. CPU
baselines never share mutable storage with generated instances. The live pair
is temporarily offloaded for memory, then returned to its original device.

Source extraction/instrumentation records tensor snapshots without changing
equations. It compares reference spectrum, final reference audio, `fea_ref`,
`ge`, normalized `mel2`, `fea_todo`, returned target `ge`, each CFM input/length/
prompt/output, helper output, denormalized mel, vocoder input and raw waveform.
Shape/dtype/device/nonfinite count/min/max/mean/std/RMS and exact equality/max
absolute difference are reported. The initial noise is reconstructed from the
same RNG state in a nested RNG-preserving scope; it is not a hook inside randn.
FP32 instead receives the EXACT measured scaled initial noise from cold B, via
an instance-local override of only the initial CFM `x` assignment. A seed alone
would be confounded by different FP16/FP32 randn values. The rest of the CFM
equations is unchanged, and numerical noise equality is reported explicitly.
Actual per-call steps/CFG/temperature are also recorded. Runtime RNG states and
temporary method wrappers are restored, including on failure.

The full case uses every token (budget: at most 1000 tokens); there is no automatic
prefix case that could warm caches before the full case. Begin with the short
sentence above; run a separate longer sentence only after evaluating short speech.
Parameter dtype/device/count histograms for S2, CFM and vocoder, AMP/autocast
state, cache inventories before/after, buffer mutations and parameter-version
changes are in the suite report. Reverse/cold controls test output sensitivity
to history; cache inventories alone do not establish absence of state effects.

## Artifacts and interpretation

Each synthesis creates a unique directory and prints its location:

- `report.json`: source hashes, selected checkpoint SHA-256, loaded vocoder/S2
  state-to-file checks after dtype casting, runtime flags and stage comparisons.
  The top report links the per-sequence reports and contains their runtime/dtype
  summaries. It separates the first metadata difference from the first value difference.
  Missing checkpoint keys are explicit; common-key equality alone is not full
  checkpoint equivalence. No private transcript is included in the report.
- `private_inputs.pt`: exact runtime-local semantic/prompt/phones/reference path
  for later replay. Contains private derived data; do not publish it.
- `*_webui_A.wav`, `*_webui_A_repeat.wav`, `*_official_B.wav`: raw FLOAT32,
  no individual normalization/clipping. Peaks above 1 remain in these files.
- `*_listen.wav`: browser-compatible PCM16 copies; all routes/precisions and the two listening reference probes use one common
  gain, recorded in the report. Prefer these for listening to avoid player
  clipping masquerading as model noise.
- `reference_vocoder_roundtrip_listen.wav`: actual reference mel -> norm ->
  denorm -> vocoder, completely bypassing CFM and S1. Current and FP32 versions
  are provided in the cold_B/fp32 subdirectories.

Play generated files inside the same private Kaggle session:

```python
from pathlib import Path
from IPython.display import Audio, display
from tools.kaggle_v5_acoustic_ab import show_report
folder = sorted(p.parent for p in Path("/kaggle/working/v5_acoustic_ab").glob("*/report.json"))[-1]
show_report(folder)
```

Interpret in order:

1. Listen to reference reconstruction -> cold A -> cold B -> FP32 B -> FP32
   reference reconstruction. Use the displayed reverse/repeat players if needed.
2. A-repeat or reverse-order outputs differ but cold A/B agree: execution history
   is a confound; do not attribute the warm difference to route code.
3. Cold A/B first differs at reference/mel/features: inspect that input boundary.
4. Inputs/noise match but CFM output differs: isolate CFM runtime/kernel/state.
5. Normalized mel matches but vocoder input differs: inspect denorm/dtype.
6. Vocoder inputs match but waveform differs: isolate vocoder/runtime state.
7. A/B audio matches exactly and both sound noisy: route difference is excluded
   for that sample; inspect the common upstream path, asset identity, reference
   and precision. It does not implicate S1 training.
8. Reference roundtrip also noisy: prioritize vocoder/mel/assets/precision.
   Clean roundtrip but noisy A/B narrows toward generated acoustic features/CFM;
   this still needs a controlled intervention before declaring causality.

9. Current A/B noisy but matched-noise FP32 B clean: precision sensitivity is
   supported for this sample. If reference reconstruction also improves in FP32,
   prioritize vocoder precision; if only generated speech improves, inspect the
   acoustic/CFM value divergence. More targeted intervention is still needed to
   name a specific operation. If FP32 stays noisy, precision alone is not a fix.

Do not label a first numerical difference as the first audible noise without
listening. No training architecture, Controller or DPO change is involved.

## Local evidence

CPU synthetic fixtures execute both real source paths and the real V5 helper
with a small fake S2/CFM/vocoder. Short and multi-chunk A/A-repeat/B agree; a
deliberate +0.25 normalized-mel perturbation is first detected at `mel2[1]`.
RNG/method restoration and NaN-safe reporting are covered. These tests validate
the diagnostic, not real-model synthesis or sound quality. Real Kaggle A/B and
the first audible-noise stage remain unverified until the runtime test above.
