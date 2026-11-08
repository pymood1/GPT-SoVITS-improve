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

## 2. Canonicalize an existing wav + .list dataset

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

## 3. Run the official V5 preprocessing in an isolated build

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

## 4. Choose one fixed train reference per speaker

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

## 5. Export inference-exact reference prompts

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

## 6. Train only the emotion controller

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

## 7. Load the controller in TTS

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

## 8. Behavior when emotion control is off

If:

- no controller is loaded;
- `emotion_id` is omitted;
- `emotion_strength == 0`;
- or the controller fails a runtime safety condition;

the S1 path falls back to the original GPT-SoVITS distribution.

Normal requests can still use the existing acceleration backend. Emotion-enabled
requests currently fall back to the ordinary `infer_panel` path because the
acceleration backend owns a separate sampler.

## 9. WebUI flow

In the main WebUI:

```text
1A - dataset formatting
1B - GPT-SoVITS fine-tuning
1C - emotion controller training
1D - inference
```

The emotion tab runs the existing scripts in order instead of reimplementing
their training logic. A sidecar JSONL is required when building a fresh
emotion dataset. Each usable pair group must explicitly identify the intended
emotion and pair provenance, for example:

```json
{"wav":"neutral_001.wav","split":"train","emotion_intended":"neutral","pair_group_id":"prompt_0001","pair_source":"recording_manifest"}
{"wav":"happy_001.wav","split":"train","emotion_intended":"happy","pair_group_id":"prompt_0001","pair_source":"recording_manifest"}
```

Keep at least one 3-10 second train sample per paired speaker outside all
`pair_group_id` groups. The WebUI selects that sample as the fixed reference.

When training finishes, the controller path is passed to the normal inference
WebUI and loaded automatically when its S1/S2 provenance matches.

## 10. Not implemented on purpose

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
- automatic emotion relabeling;
- fuzzy same-text emotion pairing;
- LLM transcript rewriting;
- an S2/CFM emotion adapter.

Those should only be added if experiments show a concrete limitation that
cannot be solved safely in the current S1-only design.
