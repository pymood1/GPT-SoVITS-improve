# S1 v3 fine-tuning: reference retention 연구 및 실험 계획

작성일: 2026-11-13

## 결론 먼저

현재 가장 가능성이 높은 문제는 **S1 v3의 full fine-tuning 자체 + fine-tuning objective와 실제 zero-shot inference의 reference-conditioning 방식이 완전히 같지 않은 점**이다.

지금 S1 fine-tuning은 작은 single-speaker dataset에서 약 77.6M parameter의 S1 Transformer 전체를 업데이트한다. 반면 inference에서는 reference audio에서 얻은 semantic token prefix를 `prompts`로 넣고 target text와 함께 Transformer가 다음 semantic token을 생성한다.

따라서 full fine-tuning이 잘못되면 단순히 "화자 적응"만 일어나는 것이 아니라, pretrained S1이 가지고 있던 **reference prompt를 읽는 방법 자체**가 바뀔 수 있다.

우선순위:

1. pretrained S1 v3와 현재 15-epoch full FT를 정확히 비교해 reference sensitivity 감소를 재현한다.
2. epoch 수와 DPO 영향을 분리한다.
3. 재현되면 S1은 full FT 대신 **frozen backbone + LoRA**를 1순위 대안으로 실험한다.
4. LoRA만으로 reference retention이 충분하지 않으면 **cross-utterance reference-conditioned training**을 추가한다.
5. 최종적으로 reference retention metric을 validation에 넣고 best checkpoint를 선택한다.

아직 LoRA를 기본 학습 방식으로 바꾸지는 않는다. 먼저 아래 ablation으로 원인을 분리하고, 결과가 맞으면 적용한다.

---

## 1. upstream에서 이미 관찰된 현상

### GPT-SoVITS Issue #2843

- V5 dev에서 동일 reference audio를 사용했는데 fine-tuned GPT 사용 시 음색과 감정이 불안정했다고 보고.
- GPT는 `s1v3.ckpt`에서 시작해 15 epochs 학습.
- 작성자가 이후 문제를 "GPT base를 fine-tuning해서 발생"한 것으로 정리했고, **원본 pretrained GPT base를 사용하자 뚜렷한 음색 불안정이 사라졌다고 보고**.
- 우리 현재 조건의 `s1v3 + 15 epoch`와 매우 유사하다.

https://github.com/RVC-Boss/GPT-SoVITS/issues/2843

### GPT-SoVITS Issue #961

- fine-tuned GPT에서 반복, reference audio 내용을 읽는 현상 등의 문제가 보고됨.
- 작성자는 official GPT base + fine-tuned VITS 조합에서는 해당 현상이 줄었다고 기록.
- 이 역시 S1 fine-tuning이 pretrained behavior를 바꿀 수 있다는 독립적인 사례다.

https://github.com/RVC-Boss/GPT-SoVITS/issues/961

### upstream V3/V4 문서

upstream은 V3 계열에 대해 다음 특성을 명시한다.

- base model을 fine-tuning하지 않고 바로 사용할 때도 timbre similarity가 크게 개선됨.
- V3/V4는 V2보다 reference audio에 더 충실하도록 설계됨.
- GPT가 더 안정적이고 richer emotion을 생성하기 쉬워졌다고 설명.

즉 V3의 장점 중 일부가 **pretrained S1/S2의 zero-shot reference behavior**에 이미 들어 있다. 작은 dataset으로 full fine-tuning할 때 이 장점을 보존하는지 따로 검증해야 한다.

References:
- https://github.com/RVC-Boss/GPT-SoVITS
- https://github.com/RVC-Boss/GPT-SoVITS/wiki/GPT%E2%80%90SoVITS%E2%80%90features-%28%E5%90%84%E7%89%88%E6%9C%AC%E7%89%B9%E6%80%A7%29

---

## 2. 현재 S1 v3 구조에서 왜 문제가 생길 수 있는가

현재 fork의 V5도 GPT/S1 학습에는 `s1longer-v2.yaml` 계열 구조를 사용한다.

주요 구조:

- phoneme/text embedding
- BERT projection: 1024 -> 512
- semantic/audio token embedding
- hidden dim 512
- 24 Transformer layers
- 16 attention heads
- FFN 2048
- semantic vocabulary 1025
- final semantic-token readout

config 기준 trainable parameter는 대략 **77.6M**이다.

### inference의 reference 사용 방식

`Text2SemanticDecoder.infer_panel_*`에서는 reference audio에서 추출한 semantic tokens가 `prompts`로 전달된다.

~~~text
target text -> text/BERT embedding ----+
                                       |
reference audio -> semantic prompt ----+--> 24-layer Transformer
                                                |
                                                v
                                      next semantic token logits
~~~

코드 핵심:

~~~python
y = prompts
y_emb = self.ar_audio_embedding(y)
xy_pos = torch.concat([x, y_pos], dim=1)
...
logits = self.ar_predict_layer(xy_dec[:, -1])
~~~

따라서 reference의 말투/리듬/감정 정보가 결과에 반영되려면 아래가 모두 pretrained behavior를 유지해야 한다.

- semantic/audio embedding
- Q/K/V projection
- attention output projection
- FFN
- LayerNorm
- final semantic readout

### fine-tuning의 학습 방식

현재 `Text2SemanticDataset`은 각 utterance마다 다음만 제공한다.

~~~text
phoneme_ids(text)
semantic_ids(same utterance)
BERT feature
~~~

그리고 S1 training은 해당 utterance의 semantic sequence를 autoregressive teacher forcing한다.

즉 fine-tuning objective에는 다음 항목이 **직접 존재하지 않는다**.

> "reference utterance A를 넣었을 때 target utterance B의 생성 스타일이 A를 따라야 한다."

따라서 single-speaker / narrow-style dataset에서는 loss를 낮추는 쉬운 방향이:

~~~text
reference를 세밀하게 읽기
~~~

보다

~~~text
이 dataset/화자는 보통 이런 semantic trajectory를 가진다
~~~

가 될 수 있다.

이 상태에서 Transformer 전체를 업데이트하면 pretrained model이 갖고 있던 reference sensitivity가 약해지는 것은 구조적으로 충분히 가능하다.

---

## 3. 현재 training recipe에서 추가로 위험한 부분

### 3.1 Full fine-tuning

`Text2SemanticLightningModule.configure_optimizers()`는 기본적으로:

~~~python
model_parameters = self.model.parameters()
~~~

를 그대로 optimizer에 넘긴다.

즉 embeddings / all 24 transformer blocks / LayerNorm / readout 모두 업데이트된다.

reference-conditioning이 attention을 통해 이루어지므로, full FT는 바로 그 reference-reading 경로까지 수정한다.

### 3.2 validation이 없음

현재 trainer:

~~~python
limit_val_batches=0
~~~

이고 `validation_step()`도 사실상 비활성화되어 있다.

따라서 15 epochs 중:

- training loss는 좋아지고
- training top-3 accuracy도 좋아지지만
- reference sensitivity가 5 epoch 이후 계속 떨어지는

상황이어도 감지할 수 없다.

현재 15 epochs 기본값은 reference retention 관점에서 best checkpoint라는 근거가 없다.

### 3.3 small / narrow dataset

few-shot adaptation에서 77.6M 전체 parameter를 한 화자, 제한된 감정/말투 분포로 학습한다.

이는 pretrained distribution에서의 표현 공간을 dataset mode 쪽으로 이동시키기 쉽다.

### 3.4 DPO

현재 fork에서는 S1 DPO가 기본 ON이다.

DPO는 reject semantic sequence보다 실제 target semantic sequence를 더 선호하게 만든다. 이것이 reference sensitivity 저하의 직접 원인이라는 근거는 아직 없다.

다만 "dataset의 canonical semantic trajectory를 더 강하게 선호"시키는 방향이므로 full FT와 함께 reference variation을 줄일 가능성은 별도로 검증한다.

**DPO는 원인으로 단정하지 않고 ablation 대상으로만 둔다.**

---

## 4. 핵심 가설

### H1 — full FT forgetting

full fine-tuning이 pretrained S1의 prompt/reference-sensitive representation을 덮어쓴다.

예측:

~~~text
pretrained S1
    > 5ep full FT
    > 15ep full FT
~~~

순으로 reference 변화에 대한 output variation이 줄어든다.

### H2 — objective mismatch

full FT 자체뿐 아니라 현재 fine-tuning objective가 cross-utterance reference conditioning을 직접 학습하지 않는 것이 핵심 원인이다.

예측:

- LoRA로 forgetting은 줄여도 reference sensitivity가 완전히 회복되지 않을 수 있음.
- same-speaker cross-utterance prompt-conditioned objective를 추가하면 개선 가능.

### H3 — DPO amplification

DPO가 full FT의 dataset-mode bias를 강화할 수 있다.

예측:

~~~text
full FT 15ep + DPO
~~~

가

~~~text
full FT 15ep + CE only
~~~

보다 reference sensitivity가 더 낮을 가능성이 있다.

---

## 5. 1차 실험 — 원인 분리

먼저 Emotion Controller를 끄고 S1 자체의 reference retention만 비교한다.

S2/SoVITS, target text, seed, sampling option은 모두 고정한다.

### Run A — pretrained baseline

~~~text
S1: original s1v3.ckpt
S2: current fixed S2
Controller: OFF
~~~

이 결과를 reference sensitivity 100% 기준으로 둔다.

### Run B — Full FT 5 epochs

~~~text
S1: full fine-tune
epochs: 5
DPO: OFF
~~~

### Run C — Full FT 15 epochs

~~~text
S1: full fine-tune
epochs: 15
DPO: OFF
~~~

### Run D — 현재 방식

~~~text
S1: full fine-tune
epochs: 15
DPO: ON
~~~

현재 실제 사용 모델을 그대로 비교한다.

### 판단

A > B > C가 명확하면 epoch/update 누적에 따른 forgetting 가능성이 매우 높다.

C > D라면 DPO가 추가 악화 요인일 가능성이 있다.

A와 B/C/D 차이가 작다면 문제는 S1 full FT보다 다른 경로(S2/reference extraction 등)를 우선 조사한다.

---

## 6. reference sensitivity benchmark

단순 청감만 사용하지 않는다.

### Reference set

같은 speaker에서 스타일이 서로 다른 3~5개 reference를 고정한다.

가능하면:

- neutral
- bright/happy-like
- surprised/strong prosody
- low/calm
- 기타 확실히 다른 말투

target text는 동일하게 유지한다.

### S1-only metric

같은 target text에 reference semantic prompt만 바꾸고 **첫 next-token prediction**을 비교한다.

기록:

- hidden cosine distance
- next-token logit Jensen-Shannon divergence
- top-k token overlap

이 metric은 S2의 영향을 받지 않고 S1이 reference prompt를 실제로 읽는 정도를 직접 본다.

예상:

~~~text
reference A vs B의 S1 hidden/logit 차이

pretrained > LoRA >= partial FT > full FT 15ep
~~~

### Audio metric

동일 target text를 reference만 바꿔 합성한다.

기록:

- emotion/style embedding distance
- F0 mean/std/range
- energy statistics
- duration / speech-rate proxy
- ECAPA speaker cosine
- CER
- repetition/omission

중요한 것은 output이 reference에 따라 달라지면서도 speaker와 intelligibility는 유지되는 것이다.

---

## 7. 2차 실험 — LoRA

1차 실험에서 full FT forgetting이 확인되면 S1 LoRA를 구현한다.

### 기본 설계

**pretrained backbone은 완전 freeze.**

1차 LoRA target:

- each Transformer layer `self_attn.in_proj_weight` (combined QKV)
- each Transformer layer `self_attn.out_proj.weight`

초기 설정:

~~~text
rank = 8
alpha = 16
dropout = 0.05
base weights = frozen
audio/text embedding = frozen
BERT projection = frozen
LayerNorm = frozen
ar_predict_layer = frozen
~~~

24 layers 기준 attention-only LoRA r=8의 추가 trainable parameter는 약 **0.59M**, 전체 S1의 약 **0.76%**다.

처음부터 FFN까지 LoRA하지 않는다. attention-only adaptation으로 부족한 것이 확인될 때만:

- `linear1`
- `linear2`

를 추가한다.

attention + FFN LoRA r=8이어도 약 **1.57M (~2%)** 수준이다.

### 왜 attention-only부터 시작하는가

reference prompt 영향은 Transformer attention을 통해 target semantic generation으로 전달된다.

따라서:

- base attention은 보존
- 작은 low-rank delta만 허용
- embedding/readout semantic space는 고정

하는 것이 현재 목표에 가장 직접적이다.

### 이 코드베이스의 구현 주의점

S1의 `MultiheadAttention`은 Q/K/V가 각각 별도 `nn.Linear`가 아니라 기본적으로:

~~~text
self_attn.in_proj_weight
~~~

라는 combined Parameter를 사용한다.

따라서 HuggingFace PEFT에서 흔히 쓰는 `q_proj`, `v_proj` module name만 지정하는 방식은 그대로 적용하기 어렵다.

권장 구현:

1. training 시 `in_proj_weight`에 custom LoRA parametrization을 적용.
2. `out_proj`에는 일반 LoRA linear를 적용.
3. training 종료 시 low-rank delta를 base weight에 **merge**.
4. 기존 S1 checkpoint 형식으로 export.

이렇게 하면 inference 코드와 checkpoint loader를 별도 LoRA-aware 구조로 바꾸지 않아도 된다.

### Run E

~~~text
S1: pretrained + attention LoRA
rank: 8
epochs: 15
DPO: OFF
~~~

Run A/B/C와 동일 benchmark.

성공 조건:

- Full FT 15보다 reference sensitivity가 확실히 높음.
- pretrained보다 speaker adaptation/target voice consistency가 개선되거나 최소 유지.
- CER/repetition 악화 없음.

그 다음에만 r=4 / r=16을 비교한다.

---

## 8. 3차 후보 — partial fine-tuning

LoRA와 비교용으로만 둔다.

예:

~~~text
bottom 20 Transformer layers: frozen
top 4 Transformer layers: train
readout: train or separate ablation
embedding: frozen
~~~

장점:

- 구현이 단순함.
- standard checkpoint 유지.

단점:

- top layer가 prompt/reference integration에 중요하다면 여전히 forgetting 가능.
- LoRA보다 parameter update 범위가 큼.

따라서 LoRA보다 우선순위는 낮다.

---

## 9. LoRA로도 부족하면 — reference-conditioned fine-tuning

이 경우 root cause는 parameter overwrite뿐 아니라 **training/inference condition mismatch**라고 판단한다.

현재 fine-tuning:

~~~text
text B + semantic sequence B
        -> predict B
~~~

개선:

~~~text
reference semantic A (same speaker)
+ target text B
+ target semantic B
        -> predict B
~~~

구현 원칙:

- A와 B는 같은 speaker.
- 가능하면 서로 다른 utterance.
- reference prompt token에는 loss를 걸지 않음.
- target B semantic token에만 CE/DPO loss.
- 일부 batch는 pretrained-compatible/current objective로 섞어 degeneration 방지.

이 방식은 inference에서 실제로 쓰는 "다른 reference utterance + target text" 조건을 training에도 직접 노출한다.

단, LoRA만으로 해결되면 여기까지 복잡하게 만들지 않는다.

---

## 10. retention regularization 후보

LoRA도 catastrophic forgetting을 완전히 제거하는 것은 아니다.

필요할 때만 다음 순서로 추가한다.

### 1순위: pretrained-logit distillation

frozen pretrained S1을 teacher로 두고 reference benchmark/prompt bank에서:

~~~text
KL(
    tuned_S1(logits),
    pretrained_S1(logits)
)
~~~

을 작은 weight로 추가한다.

목표는 fine-tuning dataset에는 적응하되 reference-conditioned output distribution이 pretrained에서 과도하게 멀어지지 않게 하는 것.

### 2순위: L2-SP / weight anchoring

full/partial FT를 계속 써야 한다면 parameter가 pretrained weight에서 너무 멀리 이동하지 않도록 penalty를 건다.

LoRA baseline이 먼저이며, 처음부터 이것들을 모두 넣지 않는다.

---

## 11. validation / checkpoint selection 개선

현재 S1 trainer는 `limit_val_batches=0`이라 reference retention을 보면서 early stop할 수 없다.

향후 S1 training에는 작은 fixed benchmark를 두고 매 epoch 또는 일정 step마다 다음을 기록한다.

~~~text
train CE / top-3 acc
held-out CE
reference hidden distance
reference logit JS divergence
speaker cosine
CER
~~~

best checkpoint를 단순 training epoch 마지막이 아니라 아래 조건으로 고른다.

1. held-out semantic prediction이 악화되지 않음.
2. pretrained 대비 reference sensitivity retention이 임계값 이상.
3. speaker adaptation이 개선됨.

초기 임계값은 실험 데이터가 쌓인 뒤 정한다. 지금 임의의 숫자로 고정하지 않는다.

---

## 12. Emotion Controller와의 관계

S1 실험에서는 Controller를 먼저 OFF한다.

이유:

- Controller checkpoint는 S1 SHA-256에 묶여 있음.
- S1이 달라지면 controller도 다시 학습해야 함.
- reference sensitivity 실험에 controller를 같이 넣으면 원인 분리가 어려움.

S1 방식을 결정한 뒤 각 후보에 controller를 별도로 재학습한다.

최종 비교:

~~~text
pretrained S1 + controller
vs
full-FT S1 + controller
vs
LoRA S1 + controller
~~~

확인:

- controller grouped OOF macro accuracy / AUC
- multi-tag switching
- tag semantic alignment
- reference sensitivity
- speaker stability
- pronunciation/CER

---

## 13. 현재 결정

현재 시점의 추천 우선순위:

~~~text
1. pretrained vs full-FT 5/15 + DPO ablation
2. reference-sensitivity S1-only benchmark
3. 문제가 재현되면 attention-only LoRA r=8
4. LoRA가 성공하면 S1 기본 fine-tuning 후보로 승격
5. LoRA로 부족할 때만 cross-utterance reference-conditioned objective
6. 이후 validation/early-stop에 retention metric 통합
~~~

**지금 당장 full fine-tuning을 삭제하지 않는다.**
실험 결과가 나오기 전까지 full FT는 baseline으로 유지한다.

---

## 14. 관련 일반 연구

GPT-SoVITS 전용 결과는 아니지만 방향 결정에 참고한다.

- LoRA가 frozen backbone을 유지하면서 full FT보다 forgetting을 줄일 수 있다는 controlled evidence:
  https://arxiv.org/abs/2603.27707
- LoRA/PEFT도 forgetting을 완전히 없애지는 않으며 update 수/parameter 수와 forgetting의 trade-off가 있다는 연구:
  https://arxiv.org/abs/2401.05605
- TTS adaptation을 포함해 PEFT + pretrained-knowledge preservation을 직접 다룬 연구:
  https://arxiv.org/abs/2402.12220

따라서 **"LoRA면 무조건 해결"**로 가정하지 않고 reference-retention benchmark로 판단한다.

---

## 다음 구현 항목

아직 구현하지 않은 항목:

- S1 prompt-sensitivity benchmark script
- S1 LoRA training mode
- LoRA merge/export
- held-out/reference-retention validation
- optional cross-utterance reference-conditioned objective

이 문서의 1차 ablation 결과를 먼저 얻은 뒤 순서대로 구현한다.


---

## 15. 2026-11-13 추가 분석 — S1과 Emotion Controller를 같이 보면 더 명확해지는 점

### Controller 학습은 이미 cross-utterance reference-conditioned 구조다

현재 `tools/train_emotion_controller.py`의 controller 학습 데이터는 S1 fine-tuning보다 실제 inference 조건에 더 가깝다.

`build_single_sample_batch()`는:

~~~text
fixed reference A의 phones + target B의 phones
fixed reference A의 semantic + target B의 semantic
fixed reference A의 BERT + target B의 BERT
~~~

를 하나의 S1 teacher-forcing sequence로 만든다.

그리고 `prepare_controller_sample()`은 reference semantic 구간을 제외한 **target B hidden state만** controller 학습에 사용한다.

즉 현재 controller는 이미:

~~~text
reference A
    +
different target B
    ↓
reference-conditioned target hidden
    ↓
emotion classifier/direction
~~~

을 보고 학습한다.

반면 S1 fine-tuning은 기본적으로 한 utterance 안의 text/semantic next-token prediction만 학습한다.

따라서 현재 구조의 비대칭은:

~~~text
S1 fine-tune:
same-utterance autoregressive objective

Controller:
cross-utterance reference-conditioned hidden objective
~~~

다.

이 비대칭 때문에 S1이 full fine-tuning으로 reference sensitivity를 잃은 뒤, Controller가 그 이미 변형된 hidden space에서 감정 방향을 다시 찾아야 하는 구조가 된다.

### hidden_direction controller의 inference는 현재 context-independent다

현재 `LowRankEmotionPreference.preference_scores()`의 `hidden_direction` mode는 inference 시 현재 `hidden_state` 값으로 preference를 바꾸지 않는다.

개념적으로:

~~~text
learned hidden emotion direction
        ↓
normalized S1 readout에 projection
        ↓
emotion별 fixed vocabulary preference
~~~

이다.

따라서 multi-tag가 잘 작동하는 현재 결과는 합리적이다. tag가 바뀌면 서로 다른 fixed semantic-token preference가 선택된다.

이 설계의 장점:

- inference가 안정적임.
- 현재 hidden의 noise에 덜 민감함.
- bounded tilt와 함께 사용하면 base S1 distribution을 완전히 덮지 않음.

단점:

- S1의 hidden/readout geometry가 fine-tuning으로 크게 이동하면 controller를 다시 학습해야 함.
- reference마다 controller direction 자체가 적응하지는 않음.
- emotion 의미 정렬은 controller training hidden geometry와 S1 readout geometry의 품질에 강하게 의존함.

따라서 **S1의 hidden/readout 좌표계를 가능한 안정적으로 유지하는 것**이 controller에도 직접적인 이득이다.

---

## 16. 수정된 LoRA 우선순위 — QKV 전체보다 V-only부터 비교

기존 계획의 "attention QKV + output LoRA"를 단일 정답으로 보지 않는다.

reference retention이 주목적이라면 다음 가설이 더 직접적이다.

### Attention에서 역할 분리

self-attention에서:

~~~text
Q, K -> 어느 token/reference position을 얼마나 볼지 결정
V    -> 선택된 token에서 어떤 정보를 전달할지 결정
O    -> attention 결과를 hidden space로 다시 mixing
~~~

따라서 pretrained reference routing을 최대한 유지하려면 **Q/K를 고정하는 것이 유리할 가능성**이 있다.

### Run E1 — V + output LoRA

첫 번째 PEFT 후보:

~~~text
frozen:
- Q projection
- K projection
- text embedding
- audio semantic embedding
- BERT projection
- LayerNorm
- final ar_predict_layer
- FFN

LoRA:
- V projection chunk
- attention out_proj
~~~

현재 implementation은 Q/K/V가 하나의 `self_attn.in_proj_weight`에 stacked되어 있으므로 custom parametrization에서 V row block에만 low-rank delta를 적용한다.

512 hidden, rank=8, 24 layers 기준:

~~~text
V LoRA:
24 * 8 * (512 + 512)
= 196,608

out_proj LoRA:
24 * 8 * (512 + 512)
= 196,608

total
= 393,216 parameters
≈ 0.51% of 77.6M S1
~~~

이 모드는 attention score/routing의 Q/K를 그대로 보존하면서 content adaptation만 허용한다.

### Run E2 — full QKV + output LoRA

비교군:

~~~text
LoRA:
- combined QKV
- attention out_proj
~~~

rank=8 기준 약 589,824 parameters, 전체의 약 0.76%.

E1보다 reference sensitivity가 낮고 speaker adaptation 차이가 작다면 V-only를 우선한다.

### Run E3 — FFN-only LoRA

추가 비교가 필요할 때만:

~~~text
LoRA:
- linear1
- linear2
attention Q/K/V/O: completely frozen
~~~

rank=8 기준 약 983,040 parameters, 전체의 약 1.27%.

attention routing을 완전히 보존한다는 장점이 있으나, speaker/prosody adaptation capacity가 부족할 수 있으므로 E1/E2 다음으로 둔다.

### 중요한 공통 원칙

모든 PEFT 후보에서 우선 다음은 freeze한다.

~~~text
ar_audio_embedding
ar_text_embedding
bert_proj
LayerNorm
ar_predict_layer
~~~

특히 `ar_audio_embedding`과 `ar_predict_layer`를 고정하면 semantic code의 input/output 좌표계를 보존한다.

이는 S1 reference retention뿐 아니라 hidden-direction Controller가 readout에 direction을 projection하는 현재 구조에도 중요하다.

---

## 17. S1에 reference-conditioned objective를 직접 넣는 것이 LoRA보다 더 근본적인 개선일 수 있다

LoRA는 **parameter overwrite 범위**를 줄인다.

하지만 현재 S1 fine-tuning objective와 inference의 차이는 그대로 남는다.

Controller에서 이미 검증된 batch 구성 코드를 S1 training 쪽에도 재사용하는 것이 가장 자연스럽다.

### 새 S1 reference-conditioned batch

same speaker에서:

~~~text
reference A
target B
A != B
~~~

를 선택한다.

입력:

~~~text
text = reference_text_A + target_text_B
semantic = reference_semantic_A + target_semantic_B
~~~

loss:

~~~text
reference semantic 구간:
loss mask

target B semantic 구간:
CE / optional DPO
~~~

즉 reference A는 condition으로만 사용하고 B의 generation만 학습한다.

이 방식은 Controller가 현재 hidden probe를 만들 때 쓰는 조건과 S1 학습 조건을 일치시킨다.

### 권장 mixture

모든 batch를 갑자기 paired objective로 바꾸지 않는다.

초기 실험:

~~~text
50% current same-utterance S1 objective
50% cross-utterance reference-conditioned objective
~~~

부터 시작한다.

이유:

- 기존 semantic LM 능력 유지
- reference-conditioned behavior 강화
- 새 objective 하나로 과적합되는 것 방지

비율은 결과를 보고 조정한다.

---

## 18. 가장 유망한 추가 안전장치 — pretrained S1 representation retention

PEFT만으로 forgetting이 0이 되는 것은 아니다.

따라서 tuned S1과 frozen pretrained S1을 같은 reference-conditioned batch에 넣고 **controller가 실제로 사용하는 final hidden geometry를 약하게 보존**할 수 있다.

### Hidden retention loss

target 구간에서:

~~~text
h_base = frozen pretrained S1 hidden
h_tuned = adapted S1 hidden

L_hidden =
1 - cosine(
    normalize(h_tuned),
    normalize(h_base)
)
~~~

을 작은 weight로 추가한다.

목표는 hidden을 완전히 동일하게 만드는 것이 아니라, pretrained coordinate system이 심하게 회전/붕괴하는 것을 막는 것이다.

이 loss는 현재 Controller가 S1 hidden representation에서 emotion direction을 학습하기 때문에 특히 직접적인 의미가 있다.

### Logit retention loss

필요한 경우에만:

~~~text
L_KL =
KL(
    p_pretrained || p_tuned
)
~~~

을 작은 weight로 추가한다.

단, KL을 너무 세게 걸면 speaker adaptation 자체가 막힐 수 있으므로 **hidden retention을 먼저 실험**하고 KL은 2차 옵션으로 둔다.

### 더 좋은 장기 형태 — relational reference retention

alternate references A/B를 준비할 수 있으면 절대 hidden 자체보다:

~~~text
delta_base =
h_base(reference A) - h_base(reference B)

delta_tuned =
h_tuned(reference A) - h_tuned(reference B)
~~~

의 방향/크기를 보존하는 것이 더 목적에 맞다.

즉:

> pretrained S1이 reference를 바꿨을 때 얼마나/어느 방향으로 반응했는지를 tuned S1도 유지하게 한다.

이는 speaker adaptation을 허용하면서 reference sensitivity만 보존할 수 있는 더 직접적인 objective다.

하지만 구현 복잡도가 증가하므로 fixed-reference hidden retention 이후에만 진행한다.

---

## 19. S1과 Controller는 joint training하지 않는 것을 기본으로 유지

현재 단계에서는 S1 LoRA와 Emotion Controller를 동시에 업데이트하지 않는다.

권장 순서:

~~~text
1. pretrained S1
      ↓
2. S1 adaptation
   (LoRA / reference-conditioned / retention)
      ↓
3. S1 freeze
      ↓
4. representation probe
      ↓
5. Emotion Controller train
      ↓
6. grouped OOF
      ↓
7. inference listening test
~~~

이유:

- joint training은 controller loss를 줄이기 위해 S1 hidden space 자체가 감정 label에 맞게 움직일 수 있음.
- 그러면 S1 generality/reference sensitivity와 controller 성능을 분리해서 판단하기 어려움.
- 현재 데이터 규모에서는 과적합 위험이 큼.

S1 adaptation 방식이 안정된 뒤에만 joint/alternating training을 별도 연구 대상으로 고려한다.

---

## 20. Controller 쪽 추가 개선은 구조 변경보다 robustness validation이 우선

현재 hidden-direction Controller는 실제 inference에서 tag 차이를 잘 만들고 multi-tag switching도 확인됨.

따라서 지금 controller 구조를 다시 복잡하게 만드는 것은 우선순위가 낮다.

대신 다음 validation을 추가하는 것이 더 가치 있다.

### 20.1 Cross-S1 comparison

각 S1 후보마다 Controller를 새로 학습하고:

~~~text
S1 representation probe
Controller grouped OOF macro accuracy/AUC
Controller direction cosine matrix
multi-tag listening
~~~

을 같이 기록한다.

좋은 S1은 단순 S1 loss가 낮은 모델이 아니라:

~~~text
reference sensitivity 유지
+
emotion hidden separability 유지/향상
+
controller OOF 유지/향상
~~~

을 동시에 만족해야 한다.

### 20.2 Alternate-reference robustness

현재 controller training은 speaker마다 fixed reference 하나를 사용한다.

이는 emotion target 비교에는 좋은 통제 조건이지만 실제 사용에서는 다른 reference가 들어올 수 있다.

따라서 최종 S1 후보가 정해진 뒤:

~~~text
same target text
same emotion tag
different references
~~~

를 넣고 tag 효과가 유지되는지 확인한다.

초기에는 training을 multi-reference로 바꾸지 않는다.

**fixed-reference train + alternate-reference evaluation**을 먼저 한다.

문제가 확인될 때만 reference consistency regularization 또는 multi-reference training을 고려한다.

---

## 21. 수정된 최종 실험 매트릭스

### Phase A — 현재 문제 재현

~~~text
A0: pretrained S1
A1: full FT 5ep, DPO OFF
A2: full FT 15ep, DPO OFF
A3: full FT 15ep, DPO ON
~~~

### Phase B — PEFT

~~~text
B1: V + out LoRA r=8
B2: QKV + out LoRA r=8
B3: FFN-only LoRA r=8 (필요할 때)
~~~

embedding/readout/norm은 고정.

### Phase C — objective

가장 좋은 Phase B model에:

~~~text
C1: + 50% cross-utterance reference-conditioned batches
C2: C1 + weak hidden-retention loss
C3: C2 + weak logit KL (C2로 부족할 때만)
~~~

### Phase D — Controller

각 유망 S1마다 새 Controller를 학습.

기록:

~~~text
S1 reference retention
S1 hidden emotion probe macro acc/AUC
Controller grouped OOF macro acc/AUC
tag direction separation
single-tag listening
multi-tag switching
alternate-reference robustness
speaker similarity
CER/repetition
~~~

### 현재 가장 유망한 후보

실험 전 가설상 우선순위는:

~~~text
V+out LoRA
+ cross-utterance reference-conditioned objective
+ weak hidden retention
→ S1 freeze
→ Controller 재학습
~~~

이다.

다만 실제 기본값 승격은 반드시 A/B 결과를 본 뒤 결정한다.

---

## 22. 관련 연구에서 이번 수정 방향과 맞는 부분

최근 speaker-adaptive TTS 연구에서도 다음 방향이 반복된다.

- PEFT/LoRA를 이용해 작은 adaptation parameter만 업데이트.
- 적은 target data에서 pretrained synthesis ability를 보존하기 위한 prior-preservation을 별도 objective로 사용.
- LoRA만으로 adaptation이 항상 full FT와 동일하지 않으므로 guidance/보존 전략을 추가.

이 프로젝트에서는 이를 그대로 복제하지 않고 S1 구조에 맞춰:

~~~text
PEFT
+
reference-conditioned objective
+
hidden/readout coordinate preservation
~~~

으로 최소화해서 적용한다.


---

## 23. 추가된 비교 도구

`tools/compare_s1_controller_space.py`를 추가했다.

목적:

- Emotion Controller가 실제로 학습하는 것과 동일한 fixed-reference -> target teacher-forcing 조건에서 S1 후보를 비교.
- S1 hidden emotion probe를 동일한 grouped OOF 방식으로 계산.
- 첫 번째 checkpoint를 baseline으로 두고 hidden/probability drift를 측정.
- full fine-tuning이 어떤 parameter group을 가장 많이 바꿨는지 Q/K/V까지 분리해 측정.

주요 출력:

~~~text
hidden emotion probe:
- OOF macro accuracy
- OOF macro AUC
- OOF CE

baseline drift:
- target hidden cosine similarity
- semantic probability JS divergence
- top-k probability overlap

parameter drift:
- text_embedding
- audio_embedding
- bert_projection
- position_alpha
- attention_q
- attention_k
- attention_v
- attention_out
- ffn
- layer_norm
- semantic_readout
~~~

예시:

~~~bash
python tools/compare_s1_controller_space.py \
  --s1 pretrained=/path/to/s1v3.ckpt \
  --s1 full15=/path/to/Gilberta-e15.ckpt \
  --feature-dir /path/to/features \
  --metadata /path/to/canonical/metadata.jsonl \
  --reference-map /path/to/reference_map.json \
  --reference-prompt-cache /path/to/reference_prompts.pt \
  --emotions happy,angry,disgusted,surprised \
  --version v5dev \
  --output /path/to/s1_space_comparison.json
~~~

첫 실행에서는 pretrained + 현재 full-FT 15ep 두 개만 비교한다.

그 결과에서 특히:

~~~text
attention_q / attention_k drift
audio_embedding drift
semantic_readout drift
hidden cosine
hidden emotion probe
~~~

를 먼저 본다.

만약 Q/K, embedding, readout까지 크게 이동하면서 hidden cosine과 reference behavior가 나빠진다면, V-only LoRA + frozen semantic coordinate 전략을 우선한다.
