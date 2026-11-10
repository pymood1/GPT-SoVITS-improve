# Codex Development Handoff

Updated: 2026-11-08

This document is the central development handoff for the GPT-SoVITS-improve fork.
Read this file before making architectural changes. Then read EMOTION_V5.md and
inspect the referenced source files.

The main branch is the active branch. Do not assume the old emotion-v5 branch
workflow is still authoritative.

---

## 1. Project goal

The current project has two connected goals.

1. Keep GPT-SoVITS V5dev usable in Colab with a fast but complete setup.
2. Add a small, conservative emotion-control path without rewriting GPT-SoVITS.

The emotion-control design is intentionally minimal-change:

~~~text
text / phonemes
    |
existing GPT-SoVITS S1
    |
small emotion preference controller
    |
slightly adjusted semantic-token sampling probabilities
    |
existing V5 SoVITS / CFM / vocoder
~~~

The controller is not supposed to replace S1, S2, CFM, or the vocoder.

Important philosophy:

- preserve the ordinary GPT-SoVITS path;
- fail open to ordinary GPT-SoVITS when emotion control is unavailable;
- keep the large backbone frozen during controller training;
- avoid adding a large emotion encoder or another LLM unless experiments prove it is needed;
- avoid architecture fragmentation;
- prefer one clear pipeline over many helper layers;
- do not remove required packages or pretrained assets merely to make Colab setup faster.

---

## 2. User / engineering preferences

When continuing this repository:

- prefer simple architecture over many small modules;
- avoid circular dependencies;
- keep code readable and PEP8-ish;
- do not compress code into machine-looking one-liners;
- imports should normally be one per line;
- use blank lines between logical sections;
- long argument lists should be multiline;
- do not insert literal escaped newline text such as backslash-n into notebook source;
- do not silently reduce features to optimize setup time;
- preserve the full dependency/model scope;
- optimize overlap, caching, downloads, extraction, and reuse instead;
- favor minimal invasive changes to upstream GPT-SoVITS;
- use existing upstream paths when possible rather than duplicating model logic.

The user prefers Korean UI and Korean explanations.

---

## 3. Current Colab entry point

Primary notebook:

Colab-Inference-Fast.ipynb

Current flow:

~~~text
Google Drive mount
    |
git fetch/reset or shallow clone
    |
tools/colab_fast_setup.py
    |
main webui.py
    |
V5dev selected
    |
1B fine-tuning selected by default
~~~

The notebook explicitly sets:

~~~text
language = ko_KR
version = v5dev
is_share = True
~~~

Current main WebUI tab flow:

~~~text
1A - preprocessing
     - ordinary GPT-SoVITS dataset formatting
     - optional emotion-pair metadata preparation
1B - training
     - SoVITS / GPT fine-tuning
     - optional V5dev emotion-controller training
1C - inference
~~~

This keeps the upstream GPT-SoVITS workflow visible instead of exposing the
emotion controller as a separate top-level stage. Emotion control remains an
optional extension inside the normal preprocessing/training/inference flow.

Fast-v2 was deleted. Only Colab-Inference-Fast.ipynb should be maintained.

---

## 4. Colab runtime assumptions

The fast setup was developed against a Colab runtime equivalent to:

- Python 3.12.x
- PyTorch 2.11.0 + CUDA 12.8 package
- torchaudio 2.11.0 + CUDA 12.8 package
- Tesla T4
- modern NVIDIA driver

The notebook currently recommends Colab Runtime Version 2026.07 + GPU because a
later default runtime used Python 3.13 and caused compatibility problems with
the dependency set, especially NumPy constraints.

Do not eagerly import torch or torchaudio in the notebook preflight. On a fresh
Colab runtime their first import was observed to take roughly 45-60 seconds and
made the first cell look frozen.

The fast preflight therefore uses:

- importlib.metadata for package versions;
- nvidia-smi for GPU information.

---

## 5. Fast setup architecture

Primary file:

tools/colab_fast_setup.py

Key rules:

- no Conda environment;
- reuse the active Colab Python runtime;
- use uv for Python package installation;
- retain the full upstream requirements;
- OpenCC remains installed;
- only the force-source directive for OpenCC is relaxed so a wheel may be used;
- NumPy is pinned to 1.26.4 for ABI stability;
- torch and torchaudio are reused from Colab;
- torchcodec is installed explicitly;
- faster-whisper from extra-req.txt is retained;
- UVR5 weight download remains optional and currently disabled by default.

The setup overlaps Python dependency installation with asset downloads.

Main download cache:

/content/gpt_sovits_downloads

The setup reuses complete cached downloads.

Important assets:

- pretrained_models.zip
- G2PWModel.zip
- nltk_data.zip
- OpenJTalk dictionary archive
- V5dev S2
- V5 vocoder

The legacy pretrained_models.zip does not contain all V5 assets.
V5dev S2 and the V5 vocoder are downloaded separately from the official
GPT-SoVITS Hugging Face model repository.

Expected V5 paths:

~~~text
GPT_SoVITS/pretrained_models/s1v3.ckpt
GPT_SoVITS/pretrained_models/gsv-v5-pretrained/s2Gv5dev.pth
GPT_SoVITS/pretrained_models/gsv-v5-pretrained/vocoder.pth
~~~

NLTK must be available under:

~~~text
runtime/nltk_data
~~~

because tools/portable_runtime.py points NLTK_DATA there.

The setup syntax-checks these files before completing:

~~~text
webui.py
tools/emotion_training_pipeline.py
GPT_SoVITS/inference_webui.py
GPT_SoVITS/prepare_datasets/3-get-semantic.py
~~~

Historical performance from an earlier successful setup run, before the most
recent emotion-WebUI integration:

- clone: about 3.7 s
- fast preflight: about 0.3 s
- dependencies + parallel downloads: about 137 s
- parallel extraction: about 86 s
- core setup before final verification: about 224 s

These numbers are historical, not a current benchmark guarantee.

Potential setup optimization that has not yet been implemented:
start extracting large archives as soon as each download finishes while uv is
still installing packages. The pretrained ZIP download completed much earlier
than uv in the earlier benchmark, so extraction could potentially be hidden
under package installation. Keep this optimization simple and beware disk I/O
contention.

---

## 6. Previously fixed Colab problems

Do not reintroduce these.

### 6.1 Slow startup caused by torch import

Old behavior:
notebook preflight imported torch/torchaudio and looked frozen.

Fix:
metadata + nvidia-smi only.

### 6.2 Literal backslash-n corruption

Notebook and setup-script edits previously produced literal backslash-n text
inside source and caused SyntaxError.

Fix:
keep real line breaks in notebook cell source and py_compile setup code before
execution.

### 6.3 NumPy ABI mismatch

Observed error class:

~~~text
ValueError: numpy.dtype size changed
~~~

Affected packages included FunASR, TorchCodec, Gradio, and Lightning.

Cause:
NumPy 2 had already been imported into the notebook kernel while the installed
environment was downgraded to NumPy 1.26 in the same process.

Current mitigation:

- do not import NumPy in notebook setup cells;
- pin NumPy 1.26.4;
- verify dependencies in a fresh Python subprocess;
- if the fresh-process ABI check still fails, reinstall NumPy/pandas/SciPy/
  scikit-learn together.

### 6.4 Missing V5 files

The legacy 4+ GB pretrained bundle did not contain:

~~~text
gsv-v5-pretrained/s2Gv5dev.pth
gsv-v5-pretrained/vocoder.pth
~~~

They are now downloaded separately.

### 6.5 Repository retry behavior

Do not delete and reclone the repository on every retry.

Current notebook behavior when .git exists:

~~~text
git fetch --depth 1 origin main
git reset --hard origin/main
~~~

This preserves untracked extracted model assets.

### 6.6 inference_webui import failure

Observed:

~~~text
ModuleNotFoundError: No module named 'tools'
~~~

Cause:
running GPT_SoVITS/inference_webui.py by path sets sys.path[0] to GPT_SoVITS
before portable_runtime can repair the path.

Fix:
launch environments include both repository root and GPT_SoVITS in PYTHONPATH.

### 6.7 Hidden inference errors

The old main WebUI launched inference with fire-and-forget Popen and did not
show the child process output or public link.

Current change_tts_inference behavior:

- uses unbuffered Python;
- captures stdout/stderr;
- streams child logs;
- detects gradio.live URL;
- renders a clickable public inference link in the main WebUI;
- shows exit code and log tail on failure.

This path still needs continued real Colab end-to-end testing after the latest
emotion pipeline changes.

---

## 7. Korean UI

Korean locale:

tools/i18n/locale/ko_KR.json

The main WebUI and inference WebUI were audited for missing i18n keys and the
known missing keys were added.

Several bad translations were corrected, including examples such as:

- Cantonese labels;
- speech speed versus sound speed;
- vocal / accompaniment output folder wording;
- semantic token wording;
- training epoch wording;
- dataset labeling wording.

Important:
a zero missing-key count does not guarantee every sentence is natural Korean.
Continue fixing context-specific awkward wording when it appears in the actual UI.

---

## 8. V5 training support in the main WebUI

config.py already has V5dev and V5turbo pretrained model paths and weight roots.

The main training version selector now exposes V5dev.

Current V5dev paths:

~~~text
S1:
GPT_SoVITS/pretrained_models/s1v3.ckpt

S2:
GPT_SoVITS/pretrained_models/gsv-v5-pretrained/s2Gv5dev.pth
~~~

V5 training internally already existed before the UI was exposed:

- s2_train_v3_lora.py supports v5dev/v5turbo;
- SynthesizerTrnV3 switches to CFMV5 for V5 variants;
- process_ckpt.py defines V5 LoRA checkpoint headers;
- V5 SoVITS fine-tuning is a CFM LoRA path.

Current model-version checkpoint headers include:

~~~text
07 = v5dev-lora
08 = v5turbo-lora
~~~

The semantic preprocessing path was updated to use
get_sovits_version_from_path_fast instead of treating every large S2 file as V3.

---

## 9. Emotion-control architecture

Primary model code:

GPT_SoVITS/AR/models/emotion_control.py

Current concept:
LowRankEmotionPreference

The controller receives an emotion ID and emotion strength and applies a bounded
preference adjustment to S1 token sampling.

It is deliberately small and the S1 backbone remains frozen during controller
training.

The ordinary GPT-SoVITS distribution remains the baseline. Emotion control is
an intervention, not a replacement model.

Important behavior:

- no controller loaded -> original behavior;
- no emotion ID -> original behavior;
- emotion strength 0 -> original behavior;
- runtime safety condition fails -> original behavior;
- acceleration backend currently does not implement the controller sampler;
- emotion-enabled requests therefore use the ordinary inference sampler.

Do not silently move emotion control into the accelerated path until equivalent
sampling semantics are implemented and tested.

---

## 10. Emotion dataset philosophy

Emotion control is not currently trained as a generic emotion classifier.

The intended high-quality training unit is a controlled emotion pair/group:

~~~text
same speaker
same text
different intended emotion
~~~

Example:

~~~text
prompt_0001
  neutral take
  happy take
  sad take
~~~

The canonical dataset builder still does not invent pair metadata by itself.
Automatic discovery is implemented as a separate preprocessing helper so the
canonical ingest remains deterministic.

Primary files:

tools/auto_emotion_sidecar.py
tools/emotion_dataset_builder.py

The 1A WebUI can now generate a sidecar JSONL from the user's existing
GPT-SoVITS .list + audio data. The first implementation is deliberately narrow:

- utterance emotion is predicted with FunASR emotion2vec+ base by default;
- only neutral / happy / sad / angry are accepted for controller training;
- predictions below the confidence threshold remain unpaired;
- automatic groups require the same speaker, language, and exact normalized text;
- pair_source is recorded as auto_exact_text;
- a 3-10 second reference is reserved outside usable pair samples;
- prediction results are cached so interrupted or repeated runs can reuse them.

Manual sidecars remain supported and can replace or edit automatic output.

Important fields:

~~~text
wav
split
source_id
session_id
speaker_uid
emotion_intended
pair_group_id
pair_source
~~~

pair_group_id must refer to one speaker, one language, and exact canonical text.

pair_source must be explicit when pair_group_id is present.

Supported pair sources include recording_manifest, manual, and
auto_exact_text. Fuzzy/semantic text matching is still intentionally not
implemented.

The same audio bytes with conflicting metadata are rejected.

Val/test samples require source_id.

Source, text, and pair groups cannot cross splits.

Canonical file names are content-derived to prevent basename collisions.

---

## 11. Reference sample policy

Emotion training uses one fixed reference per paired speaker.

Primary helper:

tools/emotion_training_pipeline.py

Current WebUI automatic selection policy:

- train split only;
- sample must not be inside an emotion pair_group_id;
- 3 to 10 seconds long;
- neutral/unlabeled is preferred;
- among candidates, duration near 6 seconds is preferred;
- only speakers that actually participate in a multi-emotion pair group require
  a reference.

This separation is important. The reference should represent speaker identity
without becoming another member of the emotion pair being optimized.

---

## 12. Emotion preprocessing

Primary wrapper:

tools/run_emotion_v5_preprocess.py

The wrapper uses existing official GPT-SoVITS preprocessing scripts rather than
reimplementing feature extraction.

It fingerprints relevant inputs and writes isolated build directories:

~~~text
emotion_features/
  build_<fingerprint>/
    2-name2text.txt
    3-bert/
    4-cnhubert/
    5-wav32k/
    6-name2semantic.tsv
    build.json
~~~

A partial build without a valid build.json should not be trusted.

The build provenance includes the model/data identities needed to avoid mixing
features from incompatible S2/token spaces.

---

## 13. Reference prompt export

Primary file:

tools/export_emotion_reference_prompts.py

This exists because training should use the same prompt semantics and text
feature path used by inference.

It creates reference_prompts.pt containing inference-style prompt semantic and
text features.

It validates:

- reference belongs to the correct speaker;
- reference is train split;
- V5 configuration is selected;
- prompt semantic is non-empty;
- prompt text and phones are non-empty;
- S2 identity is saved in the cache.

The emotion WebUI creates a temporary V5dev TTS config pointing to the exact S1
and base V5dev S2 used for controller training.

---

## 14. Emotion controller training

Primary file:

tools/train_emotion_controller.py

Current high-level training sampling:

~~~text
pair group
  -> choose target emotion
  -> positive take from target emotion
  -> negative take from a different emotion
~~~

No Cartesian-product explosion is used.

No external SER reward is currently used.

Controller training uses the ordinary GPT sampler support. A target token only
contributes to the preference objective when it survives the original sampler
support.

Important default sampling-related values:

- top-k: 15
- top-p: 1.0
- temperature: 1.0
- repetition penalty: 1.35

Important controller defaults:

- rank: 8
- max log-odds shift: 0.35
- prefix log-ratio budget: 2.0
- steps: 1000
- learning rate: 1e-3
- preference beta: 4.0
- margin: 0.02
- minimum support coverage: 0.60

The checkpoint stores the ordered emotion names. Do not hardcode emotion IDs in
new code.

The checkpoint also stores S1/S2 provenance hashes.

---

## 15. Current 1B emotion-training WebUI pipeline

Implemented in:

webui.py
tools/emotion_training_pipeline.py

The button runs:

~~~text
1/5 canonical emotion dataset
2/5 fixed reference selection
3/5 V5dev preprocessing
4/5 reference prompt export
5/5 emotion controller training
~~~

Default controller output:

~~~text
/content/drive/MyDrive/GPT-SoVITS/emotion_controller.pt
~~~

The UI has an exact GPT/S1 checkpoint field.

This is required because S1 provenance must match the model used during
inference.

If the user fine-tunes GPT in 1B, the final GPT checkpoint must be selected in
the optional emotion-training section of 1B when training the controller.

Current controller training uses V5dev base S2 for semantic preprocessing.

---

## 16. V5 SoVITS LoRA and S2 provenance

This distinction is important.

V5 SoVITS fine-tuning currently applies LoRA to the CFM path while preserving
the base semantic token space.

Therefore, when inference loads a V5 SoVITS LoRA and an emotion controller, the
controller's S2 semantic provenance is validated against the corresponding base
V5 S2, not against the LoRA file bytes themselves.

Current inference logic:

~~~text
ordinary V5 base S2:
  compare controller S2 hash with selected S2

V5 LoRA:
  compare controller S2 hash with pretrained_sovits_name[model_version]
~~~

S1 is different:
if the user fine-tunes GPT/S1, the controller must be trained against that exact
S1 checkpoint because the checkpoint hash is enforced.

---

## 17. Emotion inference integration

Primary files:

GPT_SoVITS/inference_webui.py
GPT_SoVITS/TTS_infer_pack/TTS.py

The inference WebUI supports:

- controller path input;
- load controller;
- unload controller;
- target emotion selector;
- emotion strength slider.

The controller is rejected if:

- current model is not V5dev/V5turbo;
- controller S1 hash is missing;
- current S1 hash does not match;
- controller S2 hash is missing;
- semantic S2 provenance does not match;
- hidden dimension does not match the active S1.

When optional emotion training in 1B completes, webui.py stores the controller
path and passes it to the child inference process through emotion_controller_path.

inference_webui.py reads that path and attempts to load the controller
automatically after S1/S2 are loaded.

The controller input component was changed from browser-only file upload to a
path textbox so Colab Drive paths can be used directly.

If a controller is available, the main WebUI forces the ordinary inference
script rather than the accelerated inference script.

---

## 18. Future emotion tag interface

Desired future user-facing syntax:

~~~text
[neutral] 안녕하세요.
[happy] 오늘 정말 좋은 일이 있었어요.
[sad:0.5] 그런데 조금 아쉽기도 해요.
~~~

Recommended architecture:

~~~text
raw text
  |
emotion tag parser
  |
segments:
  text + emotion name + strength
  |
map emotion name using controller checkpoint emotion_names
  |
existing GPT-SoVITS inference per segment
  |
short boundary smoothing / crossfade if needed
~~~

Important design decision:
do not initially implement token-level emotion switching inside a single S1
generation. That would require text -> phoneme -> semantic-token alignment and
adds complexity before segment-level control is validated.

First version should use tag boundaries as TTS segment boundaries.

Do not feed strings such as [happy] into BERT/phoneme preprocessing. Parse and
remove tags before text preprocessing.

Possible later syntax:

~~~text
[happy:0.8]
[sad:0.3]
~~~

Emotion names must come from checkpoint metadata rather than a hardcoded ID
table.

---

## 19. Architecture research notes that should not be lost

These are experimental design notes, not all implemented features.

### 19.1 P + M versus global G

A key research question is where emotion information should enter:

- prompt plus modulation path;
- versus a more global conditioning path.

The current preference-controller route is intentionally smaller than either
large redesign.

If revisiting this, measure whether increasing emotion control changes emotion
independently of speaker identity.

### 19.2 Avoid P x M cross mixing unless proven safe

Do not casually introduce prompt-by-emotion cross mixing at frame level.
It can break temporal/frame alignment.

### 19.3 Negative/null branch must be validated

A zero-conditioning or negative branch may be off-manifold.

If CFG-style emotion steering is added later, validate the null branch using
held-out loss / velocity behavior before trusting it.

### 19.4 Minimal controlled experiment idea

Useful experiment family:

~~~text
P/M route x global G route
2 x 2
+
null control N versus N-prime
~~~

Run for both neutral and emotion semantic prediction where applicable.

### 19.5 Metrics previously considered

Speaker preservation:

- ECAPA cosine similarity;
- use separate enrollment audio rather than the same reference when possible.

Emotion movement:

- emotion2vec or equivalent embedding;
- compare neutral-to-emotion direction.

Expressiveness:

- F0 semitone standard deviation.

Quality gates:

- CER / ASR consistency;
- MOS predictor or similar quality proxy.

Do not optimize emotion score alone if speaker identity or intelligibility falls.

### 19.6 Interpolation / CFG order

If later experimenting with continuous emotion movement:

1. condition/embedding interpolation;
2. velocity/model-output convex combination if appropriate;
3. only then CFG, after null-branch validation.

Mel interpolation should only be considered for aligned parallel utterances.

### 19.7 Neutral <-> emotion direction

A useful dataset/research direction is to learn the transformation between
neutral and emotional versions of otherwise matched utterances.

Do not assume that simply reversing an emotional example produces a clean
neutral target. Neutrality needs its own validation.

The preferred direction is controlled paired data rather than trying to infer
neutral/emotion decomposition from arbitrary fine-tuning data without checks.

---

## 20. What is intentionally not implemented yet

Do not add these merely because they sound sophisticated.

- large external emotion encoder in the main inference loop;
- full decoder/backbone fine-tuning for controller training;
- token-level tag switching inside one semantic sequence;
- fuzzy or semantic pair inference beyond exact normalized transcript matching;
- LLM transcript rewriting;
- separate S2/CFM emotion adapter;
- emotion steering inside the acceleration backend;
- complicated P x M cross-attention;
- aggressive CFG before null-branch validation.

Add any of these only when a specific measured limitation justifies them.

---

## 21. Known area needing real runtime validation

The latest WebUI emotion integration has been statically inspected and syntax
checking was added to the Colab setup, but the full post-integration pipeline
still needs a clean end-to-end Colab run.

Priority validation order:

1. fresh Colab notebook opens;
2. Google Drive mounts;
3. fast setup succeeds;
4. webui.py launches in Korean;
5. V5dev is selected;
6. 1B is initially visible;
7. 1A preprocessing and the optional 1B emotion-training section render without Gradio component errors;
8. automatic emotion sidecar generation runs on a small user dataset;
9. exact-text pair groups and the reserved reference make sense in the generated report;
10. canonical emotion dataset creation accepts auto_exact_text provenance;
11. reference auto-selection succeeds;
12. V5 preprocessing produces a valid build directory;
13. reference prompt cache is generated;
14. tiny controller run, for example 3-20 accepted steps;
15. controller checkpoint is written;
16. 1C launches inference WebUI;
17. Gradio public link appears in the parent WebUI;
18. controller auto-load succeeds;
19. emotion selector shows checkpoint emotion_names;
20. strength 0 matches ordinary inference behavior;
21. nonzero strength changes emotion direction without severe speaker drift;
22. V5 LoRA + controller provenance path is tested separately.

Do not begin a long 1000-step training run before the tiny smoke run passes.

---

## 22. Recommended next development tasks

Priority A: runtime correctness

- run the clean Colab validation checklist above;
- fix the first real traceback rather than adding speculative abstractions;
- add a tiny smoke-test mode to the WebUI if repeated manual setup becomes slow;
- verify stopping the emotion pipeline terminates all child processes cleanly;
- verify rerunning the same feature build safely reuses the correct fingerprinted
  build rather than a stale unrelated build.

Priority B: dataset usability

The first automatic sidecar builder is now implemented in 1A. It creates
conservative exact-text pairs from user data and keeps explicit provenance.

Next improvements should be driven by real runs:

- show a compact preview of predicted emotion / confidence / pair group;
- allow users to exclude or edit questionable rows before training;
- measure whether the default confidence threshold is appropriate;
- add better handling for datasets with no repeated exact transcript;
- only then consider fuzzy or semantic text matching.

Do not hide automatic provenance. Auto-generated pairs must remain visibly
distinguishable from manual or recording-manifest pairs.

Priority C: training results

After the pipeline is stable:

- train on a small neutral/happy/sad controlled dataset;
- inspect accepted-step rate and low-support skips;
- compare strength 0 versus no-controller output;
- evaluate speaker similarity, emotion movement, pitch expressiveness, and CER;
- verify emotion ordering from checkpoint metadata.

Priority D: emotion tags

Only after controller quality is demonstrated:

- add tag parser;
- map names through checkpoint metadata;
- segment text at tag boundaries;
- synthesize segments;
- add small crossfade/boundary smoothing;
- keep tag strings out of text preprocessing.

Priority E: performance

Only after correctness:

- overlap archive extraction with dependency installation;
- consider caching emotion preprocessing builds on Drive;
- optimize repeated reference prompt generation;
- only then consider accelerated emotion sampling.

---

## 23. Files Codex should inspect first

Start here:

~~~text
CODEX_HANDOFF.md
EMOTION_V5.md
Colab-Inference-Fast.ipynb
tools/colab_fast_setup.py
webui.py
config.py
tools/emotion_training_pipeline.py
tools/auto_emotion_sidecar.py
tools/emotion_dataset_builder.py
tools/run_emotion_v5_preprocess.py
tools/export_emotion_reference_prompts.py
tools/train_emotion_controller.py
GPT_SoVITS/AR/models/emotion_control.py
GPT_SoVITS/TTS_infer_pack/TTS.py
GPT_SoVITS/inference_webui.py
GPT_SoVITS/process_ckpt.py
GPT_SoVITS/s2_train_v3_lora.py
GPT_SoVITS/module/models.py
GPT_SoVITS/module/models_v5.py
tools/i18n/locale/ko_KR.json
~~~

Useful notebooks also present:

~~~text
notebooks/GPT_SoVITS_V5_Emotion_Colab.ipynb
notebooks/GPT_SoVITS_V5_Emotion_Kaggle.ipynb
~~~

Do not assume those notebooks are newer than Colab-Inference-Fast.ipynb.
Check current code before reusing them.

---

## 24. Development workflow for Codex

Recommended workflow:

1. read this handoff;
2. inspect current main instead of relying on old conversation assumptions;
3. run py_compile on touched Python files;
4. use the smallest possible smoke test first;
5. preserve existing upstream GPT-SoVITS behavior;
6. make one coherent architectural change at a time;
7. keep logs visible;
8. when a runtime error appears, capture the actual traceback before redesigning;
9. update CODEX_HANDOFF.md or EMOTION_V5.md when architecture changes;
10. avoid large refactors unless a measured problem requires them.

If a change affects Colab, test both:

- first-run clean runtime;
- rerun in the same runtime with existing downloads/models.

---

## 25. Ready-to-use Codex prompt

Use this as the starting instruction in Codex:

~~~text
Work on pymood1/GPT-SoVITS-improve main.

First read CODEX_HANDOFF.md and EMOTION_V5.md, then inspect the current source
files referenced there. Do not redesign the project from scratch.

The current priority is end-to-end validation and stabilization of:
Colab fast setup -> V5dev main WebUI -> 1A preprocessing -> 1B training
(optional emotion controller) -> 1C inference -> automatic controller loading.

Preserve ordinary GPT-SoVITS behavior and the existing full dependency/model
scope. Prefer minimal fixes over new abstractions. Do not remove packages or
models to improve setup speed. Keep Korean UI working.

Before long training, run syntax checks and a very small emotion-controller
smoke test. Use real tracebacks to drive fixes. If you change architecture or
important assumptions, update CODEX_HANDOFF.md.
~~~

---

## 26. Final caution

This fork contains both upstream GPT-SoVITS code and experimental emotion-control
work. A change that makes the emotion experiment cleaner but breaks ordinary
GPT-SoVITS is not acceptable.

The correct bias is:

~~~text
ordinary GPT-SoVITS remains stable
+
emotion control remains a small optional layer
+
runtime evidence decides whether architecture becomes more complex
~~~
