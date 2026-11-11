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
     - optional audio -> .list transcription
     - ordinary GPT-SoVITS dataset formatting
     - optional automatic emotion-pair metadata preparation
1B - training
     - SoVITS / GPT fine-tuning
     - optional V5dev emotion-controller training
1C - inference
~~~

The 1A transcription shortcut reuses the repository's existing ASR scripts.
Faster Whisper is the default multilingual path. The UI also exposes the real
Fun-ASR MLT-Nano checkpoint for multilingual transcription, SenseVoice for the
five GPT-SoVITS languages, and the classic Chinese backend. ASR and emotion
classification are separate tasks: emotion2vec+ is used only for emotion
labels/pair discovery.

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

## 8A. English/Korean multilingual BERT text features

Upstream GPT-SoVITS uses Chinese RoBERTa contextual features for Chinese but
normally feeds zero 1024-D BERT features for non-Chinese text. This fork adds an
opt-in English/Korean path for newly fine-tuned S1 checkpoints.

Primary files:

- `GPT_SoVITS/text/multilingual_bert.py`
- `GPT_SoVITS/prepare_datasets/1-get-text.py`
- `GPT_SoVITS/TTS_infer_pack/TextPreprocessor.py`
- `GPT_SoVITS/TTS_infer_pack/TTS.py`
- `GPT_SoVITS/inference_webui.py`

Current design:

- default encoder: `google-bert/bert-base-multilingual-cased`;
- languages: English and Korean only;
- 768-D multilingual BERT features are zero-padded into the existing 1024-D
  GPT-SoVITS BERT feature slot, so S1 tensor shapes do not change;
- fast-tokenizer character offsets are mapped monotonically to phone positions;
- Chinese continues to use the original Chinese RoBERTa path;
- Japanese/Cantonese and other languages keep the legacy zero-vector path.

This is deliberately **not** enabled for legacy S1 checkpoints. Existing S1
models were trained with zero non-Chinese BERT features, so injecting a new
encoder at inference time would be out-of-distribution. The 1A WebUI checkbox
must be enabled before training a new S1. The feature contract is stored in
`text_feature_config.json` and in the S1 checkpoint under `text_features`.
TTS/inference and the emotion-controller pipeline read that contract
automatically and reject mismatched preprocessing artifacts.

The alignment is currently approximate monotonic token-to-phone alignment, not
forced phoneme alignment. Measure an A/B improvement before adding a larger
alignment subsystem.

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

Controller supervision is currently per-sample, not pair-only.

Primary files:

tools/auto_emotion_sidecar.py
tools/emotion_dataset_builder.py

Automatic metadata generation uses FunASR emotion2vec+ large by default. It
predicts utterance emotion scores, calibrates them relative to each speaker's
median CLR profile, and writes conservative `emotion_intended` labels.
Ambiguous rows become neutral/default. Prediction failures remain unlabeled and
do not affect the speaker baseline.

Exact-text pair groups are not required by controller training. Automatic rows
use `pair_group_id=null` / `pair_source=none`. Manual or recording-manifest
pair metadata remains supported as optional provenance and still obeys the
canonical builder's speaker/language/text/split consistency checks.

Important fields include `wav`, `split`, `speaker_uid`,
`emotion_intended`, `pair_group_id` and `pair_source`. The same audio
bytes with conflicting metadata are rejected, val/test data requires source
provenance, and canonical filenames are content-derived to prevent basename
collisions.

---

## 11. Reference sample policy

Emotion training uses one fixed 3-10 second train reference for each speaker
that actually has non-neutral controller targets.

Primary helper:

tools/emotion_training_pipeline.py

Selection policy:

- train + TTS-usable only;
- samples in a `pair_group_id` are not references;
- neutral/unlabeled references are preferred;
- otherwise an emotional sample may be used only when removing it does not
  consume the sole sample of that emotion;
- duration nearest 6 seconds is preferred within the same priority.

The fixed reference is excluded from controller training samples. This keeps
speaker identity conditioning separate from the preference supervision.

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

The frozen S1 backbone produces teacher-forced hidden states/logits. For each
training utterance the tiny controller scores every configured non-neutral
emotion direction using the same top-k/top-p/temperature/repetition-penalty
support used by ordinary inference. Exact-text positive/negative pairs are not
required.

Sampling is balanced by emotion class: choose an emotion uniformly, then choose
one of that emotion's usable utterances. The fixed reference is excluded from
supervision. Missing Chinese BERT features are fatal; non-Chinese rows use the
expected zero BERT feature.

Important defaults:

- top-k: 15
- top-p: 1.0
- temperature: 1.0
- repetition penalty: 1.35
- rank: 8
- max log-odds shift: 0.35
- prefix log-ratio budget: 2.0
- steps: 1000
- learning rate: 1e-3
- preference beta: 4.0
- minimum support coverage: 0.60

There is no separate margin hyperparameter. The checkpoint stores ordered
emotion names plus exact S1 and semantic-S2 provenance hashes. Do not hardcode
emotion IDs.

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
4/5 inference-exact reference prompt export
5/5 balanced emotion controller training
~~~

Default blank paths resolve under:

~~~text
output/emotion_opt/<dataset-id>/
  emotion_sidecar.jsonl
  dataset/
  features/
  <dataset-id>_emotion_controller.pt
~~~

The selected single GPU is honored by preprocessing, reference-prompt export and
controller training. CPU fallback remains supported when CUDA is unavailable.

The UI has an exact GPT/S1 checkpoint field. If S1 was fine-tuned, the
controller must be trained against that exact checkpoint. Semantic preprocessing
uses the V5dev base S2 token space.

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
- inline emotion tags inside the synthesis text.

The user-facing global emotion selector and strength slider are intentionally
hidden. Emotion control is expressed in the text itself so one request can
change emotion between segments.

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

## 18. Emotion tag interface

Implemented user-facing syntax:

~~~text
[neutral] 안녕하세요.
[happy] 오늘 정말 좋은 일이 있었어요.
[sad:0.5] 그런데 조금 아쉽기도 해요.
~~~

Current architecture:

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
concatenate segment audio
~~~

The first implementation deliberately uses tag boundaries as TTS segment
boundaries. It does not attempt token-level emotion switching inside one S1
generation because that would require text -> phoneme -> semantic-token
alignment and substantially more complexity.

Tag strings such as [happy] are removed before BERT/phoneme preprocessing.

Strength can be specified per tag:

~~~text
[happy:0.8]
[sad:0.3]
~~~

If strength is omitted, 1.0 is used. The accepted range is 0 through 2.

Text before the first emotion tag uses ordinary GPT-SoVITS inference. If no
emotion tags exist at all, inference is exactly the ordinary non-emotion path.

Emotion names come from checkpoint metadata rather than a hardcoded ID table.

Boundary crossfade/smoothing is not implemented yet. The current path keeps the
ordinary sentence pause behavior between tag segments.

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

Static review and lightweight CI now cover output completeness, reference
selection safeguards, Python syntax/import-facing compatibility and several
frontend/dependency paths. The remaining high-value validation is a clean GPU
end-to-end run.

Priority order:

1. fresh Colab or Kaggle GPU environment;
2. setup/dependencies complete;
3. emotion2vec+ large actually loads and classifies real audio;
4. automatic sidecar produces at least two usable non-neutral labels;
5. canonical dataset/reference selection succeeds;
6. V5dev preprocessing produces a complete per-item build;
7. reference prompt cache is generated;
8. run a tiny 3-20 accepted-step controller training;
9. verify checkpoint S1/S2 hashes;
10. load it in 1C and compare no-controller / strength 0 / active controller;
11. test inline emotion tags and speaker similarity;
12. separately test V5 LoRA + controller provenance.

One additional numerical check is still desirable: compare teacher-forcing
hidden/logit values from `model.h` against the cached
`t2s_transformer` inference path on the same synthetic prefix. Do not start a
long 1000-step run before the tiny smoke passes.

---

## 22. Recommended next development tasks

Priority A: GPU runtime correctness

- run the tiny end-to-end controller smoke above;
- verify emotion2vec+ large model initialization on the actual Python/runtime;
- compare `model.h` and cached `t2s_transformer` numerically;
- fix the first observed runtime mismatch rather than adding new abstraction.

Priority B: data quality

- preview predicted emotion, confidence and speaker-relative score;
- allow questionable auto labels to be excluded/edited before training;
- measure class counts and accepted-step rate;
- tune confidence thresholds only from actual validation results.

Priority C: training evaluation

- compare strength 0 with no-controller output;
- evaluate speaker similarity, emotion movement, F0 expressiveness and CER;
- check minority emotions separately even though sampling is class-balanced.

Priority D: performance

Only after correctness: optimize repeated reference prompt generation, consider
persistent preprocessing cache, and investigate accelerated emotion sampling.

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
tools/asr/config.py
tools/asr/funasr_asr.py
tools/asr/fasterwhisper_asr.py
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

## Path and subprocess reliability

The WebUI path/subprocess flow is intentionally strict:

- user-pasted paths are cleaned before resolution;
- new subprocess calls use argv lists instead of shell-quoted command strings where practical;
- a zero exit code is not enough when an output artifact is expected; the expected file/build must exist and belong to the current run;
- cancellation is distinct from failure so Stop does not later become a false failure/success message;
- partial canonical/V5 preprocessing outputs are removed or rebuilt instead of becoming permanent retry blockers;
- emotion preprocessing must use the exact build directory reported by its child process rather than guessing the newest directory under a shared root.

When extending 1A/1B/1C, preserve these rules rather than adding another unchecked `Popen(...).wait()` path.

## Gradio public-link invariant

Every user-facing Gradio WebUI in this fork must create a public share link.

- main WebUI: `share=True`;
- normal and fast TTS inference WebUI: `share=True`;
- labeling WebUI: `share=True`;
- UVR5 WebUI: `share=True`;
- Colab/Kaggle launcher-generated inference WebUI: `share=True`.

Do not make remote accessibility depend on a missing `is_share` environment
variable. Parent launchers should also pass `is_share=True` to child WebUIs
for compatibility, and link detection should prefer the public
`.gradio.live` URL rather than returning localhost.

