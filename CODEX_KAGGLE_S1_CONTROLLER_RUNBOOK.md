# Codex Kaggle / Chrome 실행 Runbook — S1 + Emotion Controller

작성일: 2026-11-13  
대상: Codex가 Chrome/Kaggle을 직접 조작하여 학습·검증·수정 루프를 이어갈 때 사용

## 0. 이 문서의 목적

이 문서는 "무엇을 연구할지"보다 **Codex가 실제로 어떤 순서로 실행하고, 무엇을 보고, 다음 행동을 어떻게 결정할지**를 정한다.

핵심 범위는 다음 두 가지다.

- GPT/S1 v3의 reference sensitivity 보존
- S1 hidden/readout geometry 위에서 동작하는 Emotion Controller의 안정성 및 일반화

**S2/SoVITS는 현재 우선 조사 대상이 아니다.** S1/Controller 실험에서 S2는 가능한 한 고정한다.

세부 연구 가설과 실험 배경은 먼저 다음 문서를 읽는다.

- `S1_REFERENCE_RETENTION_EXPERIMENTS.md`
- `RESEARCH_METHODOLOGY_HANDOFF.md` — 연구/디버깅/실험 설계 방법론과 판단 원칙
- `PROJECT_RESEARCH_HISTORY.md` — 실패/진단/수정/결정의 시간순 역사와 폐기된 접근의 이유
- `EMOTION_DATASET_LABELING_RESEARCH.md` — 감정 pseudo-label/SER/불확실성/화자 상대화 연구와 v3 실험 계획
- `ADVERSARIAL_EMOTION_REFINER_DESIGN.md` — GAN/critic 아이디어를 구현 가능한 안정적 label refiner로 정리한 설계

현재 S1 비교 도구:

- `tools/compare_s1_controller_space.py`

현재 Kaggle notebook:

- `GPT_SoVITS_Improve_Kaggle_Fast_Minimal.ipynb`

---

## 1. 현재 상태 요약

### Emotion Controller

현재 Controller는 `hidden_direction` 방식이다.

중요 특성:

- S1 backbone은 Controller 학습 중 frozen.
- fixed reference A + target B의 cross-utterance 조건으로 S1 hidden을 추출.
- target B hidden에서 감정 direction을 학습.
- inference에서는 감정별 hidden direction을 S1 semantic readout에 projection.
- neutral/default는 base S1 그대로 사용.
- multi-tag switching이 실제 청감에서 정상 작동하는 것이 확인됨.
- tag마다 서로 다른 발화 느낌이 분명하게 나타남.
- surprised는 실제 "놀람"과 비교적 잘 정렬됨.
- angry는 데이터 품질/분포 문제로 사전적 angry와 완전히 정렬되지는 않지만 tag direction 자체는 분리됨.

최근 성공 Controller의 기준점:

- hidden OOF macro accuracy: 약 44.37%
- hidden OOF macro AUC: 약 0.694
- hidden balanced CE: 약 1.28974
- 4-class chance CE: ln(4) ~= 1.38629
- best held-out step: 500
- early-stop: 700
- 결과: PASS

따라서 Controller architecture를 먼저 다시 뜯지 않는다.

### S1

현재 사용 방식:

- pretrained: `s1v3.ckpt`
- 일반적인 fine-tuning: full fine-tuning
- UI 기본 epoch: 15
- 현재 fork에서는 DPO 기본 ON

upstream issue에서 `s1v3 + 15 epoch full fine-tuning` 이후 reference/timbre/emotion behavior가 오히려 불안정해지고 pretrained S1 사용 시 개선됐다는 사례가 있음.

현재 의심:

1. full fine-tuning이 reference-sensitive attention/hidden geometry를 덮어씀.
2. S1 fine-tuning objective가 실제 inference의 cross-utterance reference condition과 불일치.
3. validation이 없어 15 epoch 동안 retention이 무너져도 감지하지 못함.
4. DPO가 문제를 추가로 증폭할 가능성은 있으나 아직 미확인.

---

## 2. Codex의 작업 원칙

### 반드시 유지

- S2는 고정.
- 한 번에 변수 하나 또는 명확히 정의된 실험군만 변경.
- pretrained baseline을 항상 유지.
- S1이 바뀌면 Controller도 해당 S1로 다시 학습해야 함.
- Controller checkpoint의 S1 SHA-256 mismatch를 무시하지 않음.
- reference / target text / seed / sampling setting을 비교 실험에서 고정.
- 학습 결과가 "성공적으로 끝남"과 "품질이 개선됨"을 구분.
- training loss 하나만 보고 승격하지 않음.
- repo 코드 수정 전에 현재 파일 내용을 다시 읽고 변경.
- 동일 파일을 서로 다른 가정으로 동시에 수정하지 않음.

### 하지 말 것

- S1과 Controller를 처음부터 joint training하지 않음.
- S2까지 같이 바꿔 원인 분리를 흐리지 않음.
- LoRA가 무조건 좋다고 가정하지 않음.
- reference retention이 실제로 개선됐는지 확인하지 않고 기본값 변경하지 않음.
- 작은 데이터에서 full FT training accuracy만 보고 성공 판정하지 않음.
- Kaggle `/kaggle/working`에 대용량 임시 cache를 쌓지 않음.
- 이미 열린 Kaggle notebook이 GitHub의 최신 notebook cell과 자동 동기화된다고 가정하지 않음.

---

## 3. Kaggle 환경 기준

의도된 source dataset:

```text
/kaggle/input/datasets/pymood1/endfield/datas
/kaggle/input/datasets/pymood1/endfield/training.list
```

runtime / temporary:

```text
/kaggle/tmp/GPT-SoVITS
/kaggle/tmp/huggingface
/kaggle/tmp/audio_input
/kaggle/tmp/emotion_opt
```

persistent final output:

```text
/kaggle/working/gpt_sovits_output/
├─ GPT_weights_v5dev/
├─ SoVITS_weights_v5dev/
└─ emotion_controllers/
```

launch env의 핵심:

```text
HF_HOME=/kaggle/tmp/huggingface
TMPDIR=/kaggle/tmp
GPT_SOVITS_EMOTION_WORK_ROOT=/kaggle/tmp/emotion_opt
GPT_SOVITS_OUTPUT_ROOT=/kaggle/working/gpt_sovits_output
GPT_SOVITS_LOW_DISK_SAVE=1
```

repo 위치:

```text
/kaggle/tmp/GPT-SoVITS
```

다음 경로를 다시 `/kaggle/working/GPT-SoVITS`로 되돌리지 말 것.

---

## 4. Chrome/Kaggle 조작 시작 절차

Codex가 Chrome을 통해 Kaggle notebook을 조작할 수 있을 때 다음 순서로 시작한다.

### Step 1 — notebook 상태 확인

1. 현재 열린 notebook이 `GPT_SoVITS_Improve_Kaggle_Fast_Minimal.ipynb` 기반인지 확인.
2. repository clone/reset cell이 최신 `main`을 받는지 확인.
3. 이미 실행 중인 stale runtime이라면 repo HEAD를 확인.
4. GitHub notebook 자체가 변경된 경우, 열린 Kaggle notebook의 cell이 자동 갱신되지 않는다는 점에 주의.

### Step 2 — GPU / disk 확인

최소 확인:

```bash
nvidia-smi

df -h /kaggle/working /kaggle/tmp /tmp

du -sh /kaggle/working/* 2>/dev/null | sort -h | tail

du -sh /kaggle/tmp/* 2>/dev/null | sort -h | tail
```

두 T4가 있다면:

- 독립적인 GPT/S1과 SoVITS 작업은 각각 GPU 1개 사용 가능.
- 이번 연구에서는 S2를 고정하므로 S1/Controller 실험에 GPU를 집중.
- 작은 Controller 하나를 2-GPU model-parallel로 만들지 않음.

### Step 3 — dataset / pretrained 존재 확인

- source dataset
- `training.list`
- pretrained `s1v3.ckpt`
- current fine-tuned S1 candidate
- current fixed S2
- canonical metadata
- reference map
- reference prompt cache

중 하나라도 없으면 경로를 추측하지 말고 notebook/log에서 실제 생성 위치를 확인한다.

---

## 5. 가장 먼저 실행할 진단

새로운 S1 architecture 구현보다 먼저 현재 pretrained와 full-FT 15ep를 비교한다.

사용:

`tools/compare_s1_controller_space.py`

개념적 실행:

```bash
python tools/compare_s1_controller_space.py \
  --s1 pretrained=/PATH/TO/s1v3.ckpt \
  --s1 full15=/PATH/TO/current_15ep.ckpt \
  --feature-dir /PATH/TO/features \
  --metadata /PATH/TO/canonical/metadata.jsonl \
  --reference-map /PATH/TO/reference_map.json \
  --reference-prompt-cache /PATH/TO/reference_prompts.pt \
  --emotions happy,angry,disgusted,surprised \
  --version v5dev \
  --output /kaggle/working/gpt_sovits_output/s1_space_comparison.json
```

실제 경로는 Kaggle runtime에서 확인한 값으로 대체한다.

### 우선 볼 지표

S1 hidden:

- OOF macro accuracy
- OOF macro AUC
- OOF CE

pretrained 대비 output drift:

- hidden cosine
- semantic probability JS divergence
- top-k overlap

parameter drift:

- audio embedding
- Q
- K
- V
- attention output
- FFN
- LayerNorm
- semantic readout

### 해석

다음 패턴이면 full FT forgetting 가설이 강해짐.

```text
pretrained reference behavior > full15
+
full15 hidden cosine 저하
+
Q/K 또는 audio embedding/readout drift 큼
```

반대로 full15가 reference-sensitive metric과 청감 모두 좋으면 무리하게 LoRA로 갈 필요 없음.

---

## 6. Phase A — Full fine-tuning 원인 분리

Controller OFF 상태에서 다음 S1을 만든다.

```text
A0: pretrained s1v3
A1: full FT 5ep, DPO OFF
A2: full FT 15ep, DPO OFF
A3: full FT 15ep, DPO ON
```

S2는 모두 동일한 checkpoint.

비교 조건:

- 같은 reference
- 같은 target text
- 같은 seed
- 같은 sampling parameter

### 판정

```text
A0 > A1 > A2
```

형태로 reference sensitivity가 떨어지면 update 누적에 따른 forgetting이 유력.

```text
A2 > A3
```

이면 DPO가 추가 악화 요인일 가능성.

A0/A1/A2/A3 차이가 작으면 full FT 자체를 주범으로 확정하지 말 것.

---

## 7. Phase B — LoRA / PEFT

Phase A에서 full FT 문제가 재현될 때만 구현.

### B1 — 1순위: V + attention output LoRA

동결:

```text
Q
K
text embedding
audio semantic embedding
BERT projection
LayerNorm
FFN
ar_predict_layer
```

학습:

```text
V projection LoRA
attention out_proj LoRA
```

초기값:

```text
rank=8
alpha=16
dropout=0.05
```

현재 S1 attention은 Q/K/V가 별도 Linear가 아니라 combined `in_proj_weight`이므로 V row block에만 custom low-rank delta를 적용해야 함.

24-layer, hidden 512, r=8 기준 trainable 약:

```text
393,216 parameters
≈ 0.51% of S1
```

### B2 — 비교군: QKV + output LoRA

```text
QKV combined LoRA
+
attention out_proj LoRA
```

r=8 기준 약 0.76%.

### B3 — 필요할 때만: FFN-only LoRA

attention 전체를 frozen으로 두고 FFN만 LoRA.

B1/B2 결과가 애매할 때만 실행.

### 승격 조건

B1/B2가 full15보다:

- reference sensitivity 우수
- hidden emotion separability 유지/향상
- pronunciation/CER 비열화
- speaker stability 비열화
- Controller OOF 비열화

이면 PEFT 후보로 승격.

---

## 8. Phase C — S1 objective 자체 개선

LoRA만으로 부족할 경우 다음을 추가.

### Cross-utterance reference-conditioned S1 training

same speaker에서:

```text
reference A
target B
A != B
```

구성:

```text
input text:
reference text A + target text B

input semantic:
reference semantic A + target semantic B
```

loss:

```text
reference A semantic:
mask

target B semantic:
CE / optional DPO
```

첫 실험은:

```text
50% 기존 same-utterance objective
50% cross-utterance reference-conditioned objective
```

로 시작.

현재 Controller의 `build_single_sample_batch()` 구조를 참고/재사용하여 S1과 Controller의 conditioning 방식을 맞추는 것을 우선 고려.

---

## 9. Phase C2 — Hidden retention

frozen pretrained S1 teacher와 adapted S1을 같은 reference-conditioned batch에 통과.

target B 구간에서:

```text
h_base  = pretrained S1 hidden
h_tuned = adapted S1 hidden
```

약한 cosine retention loss를 추가.

목적:

- hidden coordinate system 붕괴 방지
- Controller가 학습할 emotion direction의 좌표계 안정화
- pretrained reference response 보존

강한 weight를 사용하지 말 것.

semantic CE보다 retention이 지배하면 adaptation이 막힐 수 있음.

처음에는 KL distillation까지 넣지 않는다.

hidden retention으로 부족할 때만 pretrained-logit KL을 추가 실험.

---

## 10. S1 확정 후 Controller 재학습

S1 candidate 하나를 평가할 때 기존 Controller를 재사용하지 않는다.

순서:

```text
S1 candidate 결정
        ↓
S1 freeze
        ↓
hidden representation probe
        ↓
Emotion Controller 새 학습
        ↓
held-out validation
        ↓
grouped OOF
        ↓
inference
```

Controller 기준:

- OOF macro accuracy
- OOF macro AUC
- balanced CE
- tag direction separation
- single-tag 청감
- multi-tag switching
- alternate-reference robustness

현재 Controller 성공 baseline을 넘는다고 무조건 승격하지 않는다.

S1 reference retention이 나빠졌는데 Controller accuracy만 올라가면 **좋은 결과가 아니다.**

---

## 11. 실제 청감 테스트 프로토콜

같은 target sentence를 사용.

single-tag:

```text
[neutral] ...
[happy] ...
[angry] ...
[disgusted] ...
[surprised] ...
```

strength test:

```text
[happy:0.5]
[happy:1.0]
[happy:1.5]
```

multi-tag:

```text
[neutral] ...
[happy] ...
[surprised] ...
[angry] ...
[disgusted] ...
[neutral] ...
```

확인:

- tag 전환 지점이 실제로 들리는가
- neutral로 돌아오는가
- emotion difference가 단순 pitch-only 변화인가
- 발음이 망가지지 않는가
- speaker identity가 유지되는가
- reference를 바꿔도 tag effect가 유지되는가

---

## 12. Reference robustness test

Controller training은 fixed reference를 쓰고 있으므로 최종 후보에서는 반드시 alternate-reference evaluation을 한다.

같은:

- S1
- Controller
- target text
- emotion tag
- seed

에서 reference만 변경.

좋은 결과:

```text
reference의 말투/특징 변화
+
emotion tag 방향 유지
+
speaker identity 유지
```

나쁜 결과:

- reference를 바꾸면 tag effect가 사라짐.
- reference style을 Controller가 완전히 덮음.
- Controller가 reference를 무시하게 만듦.

이 문제가 실제로 확인되기 전에는 Controller training을 multi-reference 구조로 복잡하게 만들지 않음.

---

## 13. 자동 의사결정 규칙

Codex는 한 실험을 끝낸 뒤 다음처럼 판단.

### Case 1

```text
pretrained S1 청감 > full15
그리고
full15 hidden/reference drift 큼
```

→ LoRA Phase B 진행.

### Case 2

```text
B1(V+O LoRA) > B2(QKV+O LoRA)
```

→ Q/K freeze 유지.

### Case 3

```text
B1/B2가 pretrained retention은 좋지만 speaker adaptation 부족
```

→ FFN LoRA 또는 cross-utterance objective를 순차 추가.

### Case 4

```text
LoRA에서도 reference sensitivity 약함
```

→ reference-conditioned S1 objective 추가.

### Case 5

```text
reference-conditioned objective 후 hidden geometry가 너무 이동
```

→ weak hidden retention 추가.

### Case 6

```text
S1은 좋아졌는데 Controller OOF/청감이 악화
```

→ Controller를 해당 S1에서 새로 학습했는지 먼저 확인.
→ 기존 Controller checkpoint 재사용 여부 확인.
→ hidden probe가 감정 정보를 유지하는지 확인.
→ 그 뒤에만 Controller 구조 변경 검토.

---

## 14. 실패/중단 조건

다음 상황에서는 자동으로 더 많은 epoch를 돌리지 않는다.

- validation/reference metric 지속 악화
- NaN/Inf
- disk full
- S1 checkpoint write failure
- Controller OOF near chance
- reference prompt cache mismatch
- S1 SHA mismatch
- feature build provenance mismatch
- 발음/CER가 분명히 악화
- 반복/무음/참조문장 복창 현상 증가

문제가 발생하면 마지막 정상 checkpoint와 로그를 보존한 뒤 원인부터 좁힌다.

---

## 15. Kaggle disk 원칙

`/kaggle/working`은 final compact artifacts 위주.

임시:

```text
repo
HF cache
pretrained cache
emotion features
canonical working data
full temporary checkpoints
```

→ `/kaggle/tmp`

최종:

```text
compact GPT
compact SoVITS
controller
experiment JSON/report
```

→ `/kaggle/working/gpt_sovits_output`

현재 low-disk mode를 유지.

---

## 16. WebUI 관련 유지 조건

- 한국어 기본.
- Finetune 화면 우선.
- BERT 기본 ON.
- preprocessing confidence 기본 0.35.
- DPO UI 기본 ON은 현재 baseline 비교가 끝날 때까지 유지하되 실험에서 ON/OFF를 명시적으로 분리.
- Gradio Error는 X를 누르기 전까지 유지.
- 2초 polling으로 전체 UI를 redraw해서 깜빡이지 않도록 함.
- progress gauge는 과한 animation 금지.
- `fast-v2` 재도입 금지.
- `fast` 유지.
- V5dev 기준.
- preprocessing a/b/c/d 흐름 유지.
- GPT→SoVITS 자동 학습 흐름을 깨지 않음.

---

## 17. Controller reference 사용 UX

Inference WebUI에는:

`Controller 학습 Reference 사용`

버튼이 있음.

동작:

- Controller checkpoint에 embedded reference metadata가 있으면 우선 사용.
- 구형 Controller면 reference prompt cache / metadata에서 복구.
- reference audio
- prompt text
- prompt language

를 자동 입력.

새 Controller는 가능한 한 training reference identity를 checkpoint metadata에 남긴다.

audio 자체를 checkpoint에 embed하지는 않음.

---

## 18. Codex가 repo를 수정할 때의 규칙

작업 시작 전:

1. 관련 파일 최신 내용 확인.
2. 기존 experiment document 확인.
3. 이미 구현된 기능과 중복 여부 확인.
4. 변경 범위를 최소화.

수정 후:

1. syntax/import 확인 가능하면 실행.
2. 실제 Kaggle runtime으로 확인하지 않은 것은 "runtime verified"라고 기록하지 않음.
3. 기존 checkpoint compatibility 유지.
4. legacy Controller loader 깨지지 않게 함.
5. 결과/의사결정을 문서에 기록.

실험에서 의미 있는 결론이 나온 경우:

- `S1_REFERENCE_RETENTION_EXPERIMENTS.md`에 결과 추가.
- 필요하면 별도 `experiments/*.json` 또는 간결한 markdown result를 추가.
- 단순 로그 전체를 repo에 무분별하게 commit하지 않음.

---

## 감정 라벨링/Adversarial Critic 현재 구현 상태

관련 파일:

- `tools/auto_emotion_sidecar.py` — v2 label 유지 + uncertainty/OOD/MAD diagnostics
- `tools/build_emotion_review_set.py` — 정보량 높은 20~30개 검수 후보 자동 선택
- `tools/emotion_label_refiner.py` — emotion2vec embedding 기반 grouped OOF Critic
- `tools/evaluate_emotion_review.py` — 소량 human review로 raw/v2/critic/v3 비교
- `tools/emotion_labeling_selftest.py` — 실제 audio 없이 synthetic smoke test

중요:
`emotion_label_refiner.py`에는 Critic과 Bounded Refiner G가 모두 구현되어 있다.
하지만 **기본값은 `--mode critic`**이고, G는 Critic OOF gate를 통과해야만 실행된다.

Kaggle에서 감정 연구를 시작할 때 가장 먼저:

~~~bash
cd /kaggle/tmp/GPT-SoVITS
python tools/emotion_labeling_selftest.py
~~~

모두 PASS하면 최신 sidecar를 생성한 뒤 먼저 Critic-only를 실행한다.

Critic / Refiner 초기 gate:

~~~text
OOF macro accuracy >= 35%
AND
OOF macro AUC >= 0.60
~~~

실행 규칙:

~~~text
Critic OOF gate 실패
→ Refiner 실행 차단
→ seed quality / OOD / embedding geometry 조사

Critic gate 통과
→ --mode refiner 실행 가능

Refiner OOF gate 실패
→ final Refiner output 차단
→ G를 더 복잡하게 만들지 말고 teacher/seed/critic evidence 재검토
~~~

Refiner mode의 state policy:

~~~text
high-confidence seed
+ OOF Critic 동의
+ OOF Refiner 동의
+ margin gate
→ accepted_core

seed 기준 미달 v2 label
+ final Critic/Refiner 동의
→ refiner_confirmed_candidate

ambiguous/neutral
+ low OOD
+ Critic/Refiner strong consensus
→ rescued_candidate

불일치
→ conflict/ambiguous
~~~

`refiner_confirmed_candidate`, `rescued_candidate`는 자동 Controller target으로 사용하지 않는다.

Controller용 파생 sidecar는 명시적으로 `--training-sidecar`를 줄 때만 생성한다.
첫 비교는 반드시 `--training-policy core`로 시작한다.

Refiner 실행 예:

~~~bash
python tools/emotion_label_refiner.py \
  --mode refiner \
  --list /kaggle/input/datasets/pymood1/endfield/training.list \
  --wav-root /kaggle/input/datasets/pymood1/endfield/datas \
  --sidecar /kaggle/tmp/emotion_opt/emotion_dataset_01/emotion_sidecar.jsonl \
  --prediction-cache /kaggle/tmp/emotion_opt/emotion_dataset_01/emotion_sidecar.jsonl.predictions.jsonl \
  --embedding-cache /kaggle/tmp/emotion_opt/emotion_dataset_01/emotion_embeddings.pt \
  --output /kaggle/tmp/emotion_opt/emotion_dataset_01/emotion_sidecar_v3_refiner.jsonl \
  --artifact /kaggle/tmp/emotion_opt/emotion_dataset_01/emotion_refiner.pt \
  --training-sidecar /kaggle/tmp/emotion_opt/emotion_dataset_01/emotion_sidecar_v3_core_training.jsonl \
  --training-policy core \
  --device cuda:0 \
  --emotions happy,angry,disgusted,surprised
~~~

상세 설계/현재 commit은 `ADVERSARIAL_EMOTION_REFINER_DESIGN.md`를 따른다.

---

## 19. 현재 추천 실행 순서

첫 Chrome/Kaggle 자동 실행에서 해야 할 일:

```text
1. 최신 main pull
2. GPU/disk/path 확인
3. current pretrained/full15 checkpoint 확인
4. compare_s1_controller_space.py 실행
5. JSON 결과 확보
6. pretrained vs full15 reference 청감
7. 결과를 repo 문서에 기록
8. full FT degradation 확인 시 A1/A2/A3 ablation
9. 그 결과가 확인된 뒤에만 V+O LoRA 구현
10. LoRA candidate 학습
11. 같은 comparison tool 재실행
12. 최종 S1 후보 freeze
13. Controller 재학습
14. OOF + single/multi-tag + alternate-reference 테스트
```

Codex는 8 이후의 다음 단계가 이미 충분히 명확하면 사용자에게 매번 확인을 요구하지 않고, **비파괴적인 범위에서 실험을 계속 진행**해도 된다.

다만 다음은 임의로 하지 않는다.

- dataset 삭제
- 기존 final checkpoint overwrite
- Git history rewrite
- 대량 artifact commit
- S2 architecture 변경
- 기존 working baseline 제거

---

## 20. 성공 정의

최종 목표는 단순히 S1 loss나 Controller accuracy가 아니다.

좋은 최종 조합은:

```text
S1
- pretrained reference sensitivity 유지
- target speaker 적응
- pronunciation 안정
- 반복/무음 없음
- emotion-relevant hidden information 유지

Controller
- grouped OOF generalization
- tag별 확실한 차이
- 감정 의미 정렬 개선
- multi-tag transition 정상
- alternate reference에서도 tag 효과 유지

System
- S2 그대로 사용 가능
- Kaggle T4에서 현실적인 학습 시간/VRAM
- compact checkpoint
- 기존 inference pipeline 최소 변경
```

을 동시에 만족해야 한다.

---

## 21. 한 줄 우선순위

**S2를 건드리지 말고, pretrained S1의 reference-routing / hidden / readout geometry를 최대한 보존하는 S1 adaptation을 찾은 뒤, 그 S1을 freeze하고 Emotion Controller를 다시 학습한다.**
