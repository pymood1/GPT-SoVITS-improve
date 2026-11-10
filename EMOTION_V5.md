# GPT-SoVITS V5 Emotion Control (WIP)

This branch keeps the ordinary GPT-SoVITS V5 path intact and adds a small
S1 probability-preference controller. It is intentionally conservative and
fail-open.

## 1. Current branch and Colab entry point

The active development branch is `main` in:

```text
https://github.com/pymood1/GPT-SoVITS-improve
```

For Colab, use:

```text
Colab-Inference-Fast.ipynb
```

That notebook mounts Google Drive, synchronizes the latest `main`, runs
`tools/colab_fast_setup.py`, and launches the Korean V5dev main WebUI.

For development context and current architectural decisions, read
`CODEX_HANDOFF.md` before changing the emotion pipeline.

The older `emotion-v5` branch checkout flow is historical and should not be
treated as the current source of truth.

## 2. Transcription and emotion analysis are separate

When the user only has audio, 1A can first create the ordinary GPT-SoVITS
`.list` file with the existing ASR backends.

The default UI path is Faster Whisper because it provides broad multilingual
transcription. The alternative FunASR option uses the separate
`FunAudioLLM/Fun-ASR-MLT-Nano-2512` multilingual checkpoint rather than
claiming 31-language coverage for the standard Nano checkpoint. SenseVoice and
the classic Chinese backend remain available.

This ASR step only creates transcript/language metadata. It does not assign the
emotion labels used by the controller.

## 3. Build emotion metadata from an existing user dataset

The 1A WebUI can generate a first-pass emotion sidecar directly from the same
GPT-SoVITS .list and audio folder used by ordinary preprocessing.

Equivalent CLI:

```bash
python tools/auto_emotion_sidecar.py \
  --list /content/data/train.list \
  --wav-root /content/data/wav \
  --output /content/data/emotion_sidecar.auto.jsonl \
  --model iic/emotion2vec_plus_base \
  --min-confidence 0.45
```

Current automatic rule:

```text
same speaker
+ same language
+ exact normalized transcript
+ different confident predicted emotion
= automatic pair candidate
```

The default classifier is emotion2vec+ base through FunASR. Only
neutral/happy/sad/angry are used for controller training by default. Other
classes or predictions below the threshold remain unpaired.

The tool also reserves a 3-10 second reference sample per usable speaker outside
the final pair samples and writes `pair_source=auto_exact_text`. Predictions
are cached next to the sidecar so reruns do not need to classify every unchanged
audio file again.

This is intentionally a first implementation. It does not do fuzzy transcript
matching or pretend uncertain predictions are ground truth. The generated
sidecar remains editable and can be replaced by a manual sidecar.

## 4. Canonicalize an existing wav + .list dataset

```bash
python tools/emotion_dataset_builder.py \
  --list /content/data/train.list \
  --wav-root /content/data/wav \
  --dataset-id speaker_dataset_01 \
  --output /content/data_ready
```

Legacy samples default to `train`. The builder does not invent emotion pairs.

For controlled emotion recordings, provide an explicit sidecar JSONL. Example:

```json
{"wav":"neutral_001.wav","source_id":"session_01","session_id":"session_01","split":"train","emotion_intended":"neutral","pair_group_id":"prompt_0001","pair_source":"recording_manifest"}
{"wav":"happy_001.wav","source_id":"session_01","session_id":"session_01","split":"train","emotion_intended":"happy","pair_group_id":"prompt_0001","pair_source":"recording_manifest"}
{"wav":"sad_001.wav","source_id":"session_01","session_id":"session_01","split":"train","emotion_intended":"sad","pair_group_id":"prompt_0001","pair_source":"recording_manifest"}
```

Then add:

```bash
--sidecar /content/data/emotion_sidecar.jsonl
```

Rules enforced by the builder:

- canonical filenames are content-derived to prevent basename feature collisions;
- transcript punctuation is preserved;
- unsupported language labels fail;
- the same audio with conflicting metadata fails;
- `val`/`test` require explicit `source_id`;
- source/text/pair groups may not cross splits;
- `pair_group_id` must be explicit and must contain one speaker/language/exact text.

## 5. Run the official V5 preprocessing in an isolated build

Use only `/content/data_ready/train.list` for training features.

The wrapper below executes the official GPT-SoVITS 1A/1B/1C scripts unchanged,
but fingerprints the dataset, S2, BERT, CNHuBERT, S2 config and current git
commit before choosing the output directory:

```bash
python tools/run_emotion_v5_preprocess.py \
  --train-list /content/data_ready/train.list \
  --output-root /content/emotion_features \
  --bert-dir GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large \
  --cnhubert-dir GPT_SoVITS/pretrained_models/chinese-hubert-base \
  --pretrained-s2g GPT_SoVITS/pretrained_models/gsv-v5-pretrained/s2Gv5dev.pth \
  --version v5dev
```

It creates an isolated directory such as:

```text
/content/emotion_features/build_0123456789abcdef/
├── 2-name2text.txt
├── 3-bert/
├── 4-cnhubert/
├── 5-wav32k/
├── 6-name2semantic.tsv
└── build.json
```

If any fingerprinted input changes, a different build ID is used. A partial
directory without a valid `build.json` is never reused.

## 6. Choose one fixed train reference per speaker

Create `reference_map.json`:

```json
{
  "speaker_dataset_01:speaker_name": "a_0123456789abcdef.wav"
}
```

The reference must:

- belong to that speaker;
- be in the train split;
- not belong to an emotion `pair_group_id`.

A normal legacy/base-training utterance is a good choice.

## 7. Export inference-exact reference prompts

GPT-SoVITS TTS appends 0.3 seconds of silence when extracting the reference
semantic prefix and also applies prompt-text punctuation preprocessing.
Cache those exact features once:

```bash
python tools/export_emotion_reference_prompts.py \
  --tts-config GPT_SoVITS/configs/tts_infer.yaml \
  --metadata /content/data_ready/metadata.jsonl \
  --reference-map /content/data_ready/reference_map.json \
  --output /content/data_ready/reference_prompts.pt
```

The TTS config must point to `v5dev` or `v5turbo` and the intended V5
pretrained/fine-tuned models.

## 8. Train only the emotion controller

```bash
python tools/train_emotion_controller.py \
  --s1-checkpoint GPT_SoVITS/pretrained_models/s1v3.ckpt \
  --feature-dir /content/emotion_features/build_0123456789abcdef \
  --feature-build-manifest /content/emotion_features/build_0123456789abcdef/build.json \
  --metadata /content/data_ready/metadata.jsonl \
  --reference-map /content/data_ready/reference_map.json \
  --reference-prompt-cache /content/data_ready/reference_prompts.pt \
  --output /content/emotion_controller.pt \
  --emotions neutral,happy,sad,angry \
  --steps 1000
```

Only the small controller is optimized. The S1 backbone is frozen.

The controller checkpoint stores the SHA-256 identity of the S1 checkpoint used
for training. `TTS.load_emotion_controller()` refuses to attach it to another
S1 checkpoint, even when the tensor dimensions happen to match.

When an inference-exact reference prompt cache is used, the trainer checks the
S2 SHA-256 stored in the preprocessing `build.json` against the S2 SHA-256
stored in the reference prompt cache. This prevents target/reference semantic
token spaces from being mixed silently. `--semantic-s2-checkpoint` remains a
fallback only when a build manifest is unavailable.

Training samples are selected as:

```text
pair group
  -> target emotion
  -> positive take of that emotion
  -> negative take of another emotion
```

No Cartesian-product pair expansion and no SER reward are used.

The trainer uses the same default inference sampling settings:

- top-k: 15
- top-p: 1.0
- temperature: 1.0
- repetition penalty: 1.35

A target token contributes to the preference loss only if it survives the
original GPT-SoVITS sampler support.

## 9. Load the controller in TTS

Python example:

```python
from TTS_infer_pack.TTS import TTS, TTS_Config

tts = TTS(TTS_Config("GPT_SoVITS/configs/tts_infer.yaml"))
meta = tts.load_emotion_controller("/content/emotion_controller.pt")
print(meta["emotion_names"])

inputs = {
    # ordinary GPT-SoVITS TTS fields...
    "emotion_id": 1,
    "emotion_strength": 1.0,
}

for sr, audio in tts.run(inputs):
    pass
```

With the default training order:

```text
0 = neutral
1 = happy
2 = sad
3 = angry
```

The checkpoint stores the actual emotion-name order; use that metadata instead
of assuming IDs for other checkpoints.

`emotion_strength` is an intervention strength, not a perceptual percentage.

## 10. Behavior when emotion control is off

If:

- no controller is loaded;
- `emotion_id` is omitted;
- `emotion_strength == 0`;
- or the controller fails a runtime safety condition;

the S1 path falls back to the original GPT-SoVITS distribution.

Normal requests can still use the existing acceleration backend. Emotion-enabled
requests currently fall back to the ordinary `infer_panel` path because the
acceleration backend owns a separate sampler.

## 11. WebUI flow

In the main WebUI:

```text
1A - preprocessing
     - ordinary GPT-SoVITS dataset formatting
     - optional emotion-pair metadata
1B - training
     - SoVITS / GPT fine-tuning
     - optional emotion controller
1C - inference
```

The optional emotion controls are integrated into the normal GPT-SoVITS stages
rather than exposed as a separate top-level tab. The controller path still runs
the existing scripts in order instead of reimplementing their training logic.

The emotion pipeline reuses the same .list and audio-directory inputs shown in
1A. The sidecar can be generated automatically with the exact-text rule above
or supplied manually. Each usable pair group still explicitly records the
emotion and pair provenance, for example:

```json
{"wav":"neutral_001.wav","split":"train","emotion_intended":"neutral","pair_group_id":"prompt_0001","pair_source":"recording_manifest"}
{"wav":"happy_001.wav","split":"train","emotion_intended":"happy","pair_group_id":"prompt_0001","pair_source":"recording_manifest"}
```

Keep at least one 3-10 second train sample per paired speaker outside all
`pair_group_id` groups. The WebUI selects that sample as the fixed reference.

When optional emotion training in 1B finishes, the controller path is passed to
the 1C inference WebUI and loaded automatically when its S1/S2 provenance matches.

## 12. Not implemented on purpose

The current branch includes a WebUI pipeline for:

- canonical emotion-dataset preparation;
- automatic fixed-reference selection for paired speakers;
- V5dev preprocessing;
- inference-exact reference-prompt export;
- emotion-controller training;
- controller auto-handoff to the normal inference WebUI.

For V5dev SoVITS LoRA fine-tunes, controller S2 provenance is checked
against the V5dev base S2 because the current fine-tune path applies LoRA to
the CFM while preserving the semantic token space. GPT/S1 is different:
the controller must be trained against the exact S1 checkpoint used for
inference.

The current branch still does **not** include:

- emotion steering inside the acceleration backend;
- fuzzy or semantic same-text matching beyond exact normalized transcripts;
- LLM transcript rewriting;
- an S2/CFM emotion adapter.

Those should only be added if experiments show a concrete limitation that
cannot be solved safely in the current S1-only design.
