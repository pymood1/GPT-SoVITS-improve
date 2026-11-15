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

## Run all in Google Colab (no WebUI interaction)

Open [`notebooks/GPT_SoVITS_V5_Acoustic_AB_Colab.ipynb`](../notebooks/GPT_SoVITS_V5_Acoustic_AB_Colab.ipynb)
in Google Colab, select a GPU runtime, then choose **Runtime -> Run all**. The
notebook has one code cell. It automatically:

1. clones/fast-forwards the PR branch in a separate `/content` checkout;
2. installs the repository runtime and downloads only the pretrained assets
   needed for `s1v3.ckpt`, `s2Gv5dev.pth`, the V5 vocoder, CNHuBERT and BERT;
3. prefers an existing 3–10 s Korean reference + exact transcript when a training
   list is already present; otherwise downloads the pinned PyTorch VOiCES English
   fallback WAV, verifies its SHA-256, and installs/verifies only missing NLTK
   English-frontend resources before phone conversion;
4. runs the normal semantic path once with the fixed Korean sentence, without
   starting a Gradio server;
5. waits for all A/A-repeat/B/reverse/cold/FP32 controls and displays the
   reference plus eight listening players in order; and
6. creates `/content/v5_acoustic_ab_listening.zip` containing the listening
   WAVs and `report.json` (the private semantic tensor file is excluded).

When no local Korean reference is available, the public fallback reference
transcript is exactly `I had that curiosity beside me at this moment.`.
Before that fallback is used, the launcher verifies the exact GPT-SoVITS English
frontend path, including NLTK POS tagging and phone conversion. Existing NLTK
resources are reused. The target is
**안녕하세요. 오늘은 날씨가 좋아서 잠깐 산책을 다녀오려고 합니다.**
The first run downloads several pretrained assets and then performs eight
acoustic passes plus two reference-vocoder probes, so the single cell may run
for several minutes. Keep the tab connected until the players appear. No
training data, Drive mount, uploaded reference, model selection or WebUI click
is required.

## Primary path: existing Kaggle session + existing data

Kaggle is now the preferred diagnostic path. The diagnostic intentionally reuses
the user's existing GPT-SoVITS installation, pretrained assets, Korean training
metadata and audio. It does **not** download or invent a public reference when
valid existing data is unavailable.

Run one cell:

```python
from pathlib import Path
import importlib.util
import subprocess

repo = Path("/kaggle/working/GPT-SoVITS-improve-diagnostic")
branch = "codex/v5-acoustic-ab"
remote = "https://github.com/pymood1/GPT-SoVITS-improve.git"

if not (repo / ".git").is_dir():
    subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", branch, remote, str(repo)],
        check=True,
    )
else:
    subprocess.run(["git", "fetch", "origin", branch], cwd=repo, check=True)
    subprocess.run(["git", "checkout", branch], cwd=repo, check=True)
    subprocess.run(["git", "reset", "--hard", "origin/" + branch], cwd=repo, check=True)

spec = importlib.util.spec_from_file_location(
    "kaggle_v5_acoustic_ab",
    repo / "tools/kaggle_v5_acoustic_ab.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
result_folder = module.run_kaggle_all(repo)
print("완료 폴더:", result_folder)
```

The runner:

1. fetches the current PR branch through Git;
2. reuses `/kaggle/tmp/GPT-SoVITS` rather than reinstalling the runtime;
3. symlinks the already-downloaded base GPT, V5dev SoVITS, V5 vocoder, CNHuBERT
   and BERT into the dedicated diagnostic checkout;
4. searches existing Kaggle-local `.list` metadata under the normal temp,
   working and user dataset roots;
5. selects a real Korean 3–10 s reference and preserves its exact transcript;
6. displays that original reference before synthesis;
7. runs the semantic path and the full fixed-semantic A/B/order/FP32 suite
   headlessly, with no Gradio interaction; and
8. displays the listening controls and writes
   `/kaggle/working/v5_acoustic_ab/latest_listening.zip`.

If no valid Korean reference + transcript is found, the runner stops and prints
which metadata files it inspected. This is deliberate: for this bug investigation
the user's existing data is a better control than an unrelated public English
sample.

The fixed target remains:
**안녕하세요. 오늘은 날씨가 좋아서 잠깐 산책을 다녀오려고 합니다.**

## Secondary path: standalone Colab

The Colab runner remains only as a portable fallback for a clean environment.
It may use the pinned public reference when no local Korean dataset exists, so it
is not the preferred path for reproducing the user's current training/inference
symptom.

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
