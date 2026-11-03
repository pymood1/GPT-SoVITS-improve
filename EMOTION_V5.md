# GPT-SoVITS V5 Emotion Control (WIP)

This branch keeps the ordinary GPT-SoVITS V5 path intact and adds a small
S1 probability-preference controller. It is intentionally conservative and
fail-open.

## 1. Colab checkout

For a fresh Colab runtime:

```bash
git clone https://github.com/RVC-Boss/GPT-SoVITS.git /content/GPT-SoVITS
cd /content/GPT-SoVITS

git remote add emotion-patch https://github.com/pymood1/GPT-SoVITS-improve.git
git fetch emotion-patch emotion-v5
git checkout -B emotion-v5 emotion-patch/emotion-v5

python tools/check_emotion_patch.py
```

This keeps the official repository as `origin` and the experimental fork as
`emotion-patch`.

Use the official GPT-SoVITS installation/model-download procedure for
dependencies and pretrained weights.

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

## 3. Run the official V5 preprocessing

Use only `/content/data_ready/train.list` for training features.

Run the official GPT-SoVITS 1A/1B/1C preprocessing with the V5 pretrained
SoVITS model. Do not reuse an old v2ProPlus/V5 experiment feature directory
after changing the list, audio, preprocessing code, or S2 model.

**Current safe rule:** use a fresh GPT-SoVITS preprocessing experiment/output
folder for each changed dataset/model build.

The controller trainer expects:

```text
<feature_dir>/
├── 2-name2text.txt
├── 3-bert/
└── 6-name2semantic.tsv
```

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
  --feature-dir /content/<fresh_v5_feature_dir> \
  --metadata /content/data_ready/metadata.jsonl \
  --reference-map /content/data_ready/reference_map.json \
  --reference-prompt-cache /content/data_ready/reference_prompts.pt \
  --output /content/emotion_controller.pt \
  --emotions neutral,happy,sad,angry \
  --steps 1000
```

Only the small controller is optimized. The S1 backbone is frozen.

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

## 9. Not implemented on purpose

The current branch does **not** include:

- WebUI emotion controls;
- emotion steering inside the acceleration backend;
- automatic emotion relabeling;
- fuzzy same-text emotion pairing;
- LLM transcript rewriting;
- an S2/CFM emotion adapter.

Those should only be added if experiments show a concrete limitation that
cannot be solved safely in the current S1-only design.
