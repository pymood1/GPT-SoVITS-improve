# Research Methodology & Handoff — GPT-SoVITS Improve

작성일: 2026-11-13

목적: 이 프로젝트를 이어받는 Codex/AI 개발자가 기존 결론뿐 아니라 연구 방식 자체를 재현할 수 있도록, 지금까지 사용한 판단 기준·디버깅 습관·실험 설계·오탐 방지 원칙·아키텍처 철학·실행 절차를 최대한 상세히 기록한다.

관련 문서:

- CODEX_KAGGLE_S1_CONTROLLER_RUNBOOK.md — 실제 Chrome/Kaggle 실행 절차
- S1_REFERENCE_RETENTION_EXPERIMENTS.md — S1 reference retention 연구 노트
- tools/compare_s1_controller_space.py — S1/Controller 공간 비교 도구

---

# 0. 이 문서를 어떻게 읽을 것인가

이 문서는 특정 기능 하나의 설계서가 아니다.

이 프로젝트에서는 다음과 같은 종류의 문제가 반복된다.

- 학습 loss는 좋아지는데 실제 음질은 나빠짐.
- probe는 좋아 보이는데 inference에서 체감이 없음.
- 특정 class accuracy는 높은데 실제 감정 의미는 정렬되지 않음.
- fine-tuning을 했는데 pretrained보다 generalization이 떨어짐.
- UI/Colab/Kaggle 문제처럼 보이지만 실제로는 disk/path/process 문제.
- 모델 architecture 문제처럼 보이지만 실제로는 objective mismatch.
- 데이터 문제처럼 보이지만 실제로는 validation leakage.
- inference 문제처럼 보이지만 실제로는 training reference mismatch.
- controller가 문제처럼 보이지만 실제로는 S1 hidden geometry가 원인.

따라서 이 문서의 목적은:

~~~text
"어느 파일을 고쳐라"
~~~

가 아니라

~~~text
"어떤 증거를 어떤 순서로 모으고,
어떤 가설을 먼저 버리고,
어떤 실험을 최소 비용으로 실행해야 하는가"
~~~

를 남기는 것이다.

Codex는 결론보다 판단 과정의 구조를 우선적으로 계승해야 한다.

---

# 1. 가장 중요한 연구 철학

## 1.1 먼저 현상을 정확히 정의한다

코드를 보기 전에 사용자가 말한 현상을 구체적으로 다시 정의한다.

예:

~~~text
"감정이 잘 안 된다"
~~~

라고 하면 곧바로 emotion model을 고치지 않는다.

다음처럼 분해한다.

~~~text
태그 자체가 무시되는가?
태그마다 다른 결과는 나는가?
다른 결과는 나는데 의미만 틀리는가?
특정 감정만 잘 되는가?
reference를 바꾸면 tag 효과가 사라지는가?
speaker identity가 깨지는가?
발음이 깨지는가?
~~~

실제로 현재 프로젝트에서 happy / angry / disgusted / surprised가 각자 다른 느낌을 내고 multi-tag switching도 동작했다.

그러면 문제는:

~~~text
controller가 아무것도 못 한다
~~~

가 아니라

~~~text
controller가 서로 다른 direction을 만든다.
다만 일부 direction의 semantic alignment가 부족하다.
~~~

로 바뀐다.

이 차이는 architecture를 전면 재설계할지, data/label alignment만 고칠지를 결정한다.

---

## 1.2 증상을 보고 즉시 architecture를 바꾸지 않는다

이 프로젝트에서 가장 피해야 할 습관:

~~~text
문제 발견
→ 새로운 layer 추가
→ 새로운 loss 추가
→ 새로운 classifier 추가
→ 더 복잡한 pipeline
~~~

이다.

먼저 확인:

1. 현재 architecture가 정말 실패했는가?
2. 데이터가 잘못됐는가?
3. objective가 잘못됐는가?
4. validation이 잘못됐는가?
5. metric이 문제를 잘못 보고 있는가?
6. inference와 training 조건이 다른가?
7. 단순 hyperparameter 문제인가?

현재 Emotion Controller의 좋은 사례:

초기 semantic likelihood 기반 controller가 실패했을 때 바로 큰 network를 추가하지 않았다.

먼저 probe:

- S1 hidden: informative
- semantic full residual: weak
- semantic top-k residual: weak
- text-only: somewhat informative

결론:

~~~text
감정 정보가 S1 hidden에는 있다.
하지만 semantic-likelihood mapping에서 정보가 사라진다.
~~~

그래서 architecture 전체가 아니라 bottleneck만 교체했다.

결과적으로 hidden-direction controller가 성공했다.

이 패턴을 다른 문제에도 적용한다.

---

## 1.3 작은 모델/작은 데이터라는 이유만으로 실패라고 단정하지 않는다

모델 규모 하나만 보고 가능성을 판단하지 않는다.

항상 같이 본다.

- 모델 parameter 수
- 실제 학습 token/sequence 노출량
- 데이터 다양성
- 반복 횟수
- class imbalance
- input condition
- hidden dimension
- objective difficulty
- pretrained prior

"작으니까 안 된다"는 진단은 마지막에 한다.

---

## 1.4 한 번에 변수 하나를 바꾼다

가능하면:

~~~text
A = 현재 baseline
B = 변수 하나만 변경
~~~

으로 비교한다.

여러 변경이 필요한 경우에도 최소한 실험군을 명확히 분리한다.

예:

~~~text
A0 pretrained
A1 full FT 5ep, DPO off
A2 full FT 15ep, DPO off
A3 full FT 15ep, DPO on
~~~

이렇게 해야:

- epoch effect
- full FT effect
- DPO effect

를 분리할 수 있다.

LoRA + 새로운 loss + 새로운 dataset + 새로운 sampler를 한 번에 넣으면 결과가 좋아도 이유를 알 수 없다.

---

# 2. 증거의 우선순위

나는 보통 다음 순서로 신뢰한다.

## Tier 1 — 실제 재현된 runtime 결과

가장 강한 증거:

- 실제 Kaggle/Colab 실행 로그
- 실제 inference audio
- 실제 OOF/held-out 결과
- 실제 disk/GPU 상태
- 실제 exception traceback

"코드상 될 것 같다"보다 훨씬 강하다.

## Tier 2 — 현재 repo 코드

실제 코드의 control flow.

README 설명보다 코드가 우선.

특히:

- 실제 default
- 실제 path
- 실제 loss
- 실제 freeze 여부
- 실제 model loading
- 실제 inference conditioning

을 확인한다.

## Tier 3 — upstream issue / PR / official docs

현재 현상이 upstream에서도 보고됐는지 확인.

하지만 issue 하나를 사실로 일반화하지 않는다.

issue는:

~~~text
같은 현상의 독립 사례
~~~

로 사용한다.

## Tier 4 — 논문

architecture 방향이나 일반 원리 검증에 사용.

논문 결과를 현재 codebase에 그대로 적용하지 않는다.

예:

LoRA가 forgetting을 줄이는 연구가 있어도:

~~~text
GPT-SoVITS S1에서 V-only LoRA가 반드시 최선
~~~

이라고 단정하지 않는다.

실험 필요.

## Tier 5 — 직관

직관은 실험 순서를 정하는 데 사용한다.

결론을 확정하는 데 사용하지 않는다.

---

# 3. 코드 읽는 방식

## 3.1 파일 하나를 고립해서 보지 않는다

항상 data flow를 따라간다.

예:

~~~text
dataset
→ preprocessing
→ feature files
→ trainer
→ checkpoint
→ loader
→ inference
→ UI
~~~

Emotion Controller라면:

~~~text
canonical metadata
→ reference map
→ reference prompt cache
→ S1 teacher forcing
→ hidden cache
→ probe
→ controller training
→ checkpoint metadata
→ inference load
→ tag parser
→ sampling tilt
~~~

이 흐름 전체를 따라간다.

---

## 3.2 시작점과 끝점을 먼저 찾는다

예를 들어 "reference가 inference에 어떻게 들어가나?"를 확인할 때:

먼저 inference function에서 prompts가 어디서 들어오는지 찾는다.

그 다음 역으로:

~~~text
reference audio
→ semantic extraction
→ prompts
→ ar_audio_embedding
→ transformer
~~~

을 따라간다.

처음부터 repository 전체를 읽지 않는다.

---

## 3.3 실제 mutable state를 찾는다

문제가 생기기 쉬운 것:

- global model path
- environment variable
- temp path
- cached feature
- checkpoint metadata
- UI state
- subprocess
- shared GPU
- shared file

특히 Colab/Kaggle에서는:

~~~text
repo code는 최신
하지만 이미 떠 있는 WebUI process는 구버전
~~~

인 경우가 흔하다.

따라서:

~~~text
파일이 바뀌었다
≠
실행 중 process가 바뀌었다
~~~

를 항상 구분한다.

---

## 3.4 실제로 학습되는 parameter를 확인한다

architecture 그림만 보고 판단하지 않는다.

반드시 optimizer에 들어가는 parameter를 본다.

예:

~~~text
optimizer = AdamW(self.model.parameters())
~~~

면 사실상 full FT다.

반대로 requires_grad False가 있거나 parameter group이 따로 있으면 실제 update 범위가 다르다.

LoRA 구현 시에도:

~~~text
LoRA module은 추가했는데 base weight도 optimizer에 들어가 있음
~~~

같은 실수를 반드시 체크한다.

---

# 4. 오탐을 줄이는 디버깅 원칙

사용자는 과한 오탐을 원하지 않는다.

따라서 코드 리뷰에서:

~~~text
"이론적으로 문제일 수 있음"
~~~

과

~~~text
"실제로 현재 실행 경로에서 문제"
~~~

를 구분한다.

## 4.1 문제 레벨

### HIGH

실제 runtime에서 실패하거나 논리적으로 결과가 잘못됨.

예:

- path가 존재하지 않음.
- checkpoint 저장 중 disk full.
- shape mismatch.
- S1 SHA mismatch인데 controller 사용.
- validation leakage.
- wrong target slicing.

### MEDIUM

현재 조건에서 문제 가능성이 높고 결과에 영향 가능.

### LOW

code smell / 향후 가능성.

LOW는 함부로 고치지 않는다.

---

## 4.2 이상해 보인다만으로 수정하지 않는다

예:

~~~text
grad_u=0
u_norm=0
semantic_loss=nan
~~~

만 보면 버그 같을 수 있다.

하지만 hidden_direction mode에서는 U를 사용하지 않고 semantic aux가 diagnostic-only이면 정상일 수 있다.

따라서 반드시 현재 mode / branch를 읽는다.

---

# 5. Root Cause 분석 방식

문제를 발견하면 보통 아래 순서로 좁힌다.

## 5.1 가장 바깥쪽 실패부터

예:

~~~text
training failed
~~~

이면 먼저:

- GPU OOM?
- disk full?
- subprocess?
- dependency?
- path?

를 본다.

model architecture부터 보지 않는다.

실제로 과거 SoVITS/GPT checkpoint failure는 모델 문제가 아니라 Kaggle working disk full이었다.

---

## 5.2 성공한 마지막 지점 찾기

로그에서:

~~~text
어디까지 정상적으로 갔는가?
~~~

를 찾는다.

예:

~~~text
SoVITS 77%
checkpoint save start
Errno 28
~~~

이면 학습 loop가 아니라 save path 문제.

---

## 5.3 조건을 최소화한다

Emotion Controller가 안 되면:

~~~text
S1 hidden probe
semantic residual probe
text-only probe
~~~

로 분리.

이렇게 하면 실패 위치를 찾을 수 있다.

---

# 6. 실험 설계 원칙

## 6.1 baseline은 반드시 보존

새 실험 전에 현재 best를 삭제하지 않는다.

최소:

~~~text
baseline checkpoint
baseline metrics
baseline inference audio
~~~

가 있어야 한다.

---

## 6.2 metric 하나로 결정하지 않는다

TTS/Emotion 문제는 최소 세 축으로 본다.

### Task metric

- emotion OOF accuracy/AUC
- hidden probe
- validation CE

### Speech quality

- CER
- repetition
- omission
- silence
- duration
- F0
- energy

### Human perceptual

- 실제 감정이 들리는가?
- speaker가 유지되는가?
- 자연스러운가?
- tag change가 들리는가?

한 축만 좋으면 승격하지 않는다.

---

## 6.3 chance baseline을 항상 둔다

4-class면:

~~~text
accuracy chance = 25%
CE chance = ln(4) ~= 1.38629
~~~

같은 기준을 둔다.

숫자가 좋아 보이는지만 보지 않는다.

---

## 6.4 OOF를 선호

작은 데이터에서 단일 holdout은 매우 흔들린다.

가능하면 grouped OOF.

특히 correlated samples:

- source_id
- text_group_id
- pair_group_id

같은 그룹이 train/val에 동시에 들어가지 않도록 한다.

---

## 6.5 permutation baseline을 사용

probe가 정말 signal을 읽는지 확인하려면 label permutation과 비교한다.

예:

~~~text
실제 macro acc 51%
permutation 27%
~~~

면 단순 small-data noise보다 실제 signal 가능성이 높다.

---

# 7. Emotion Controller에서 배운 것

## 7.1 첫 실패를 controller가 불가능으로 해석하지 않았다

초기 semantic-likelihood 방식은 약했다.

하지만 representation probe에서:

~~~text
S1 hidden:
macro acc ~51%
AUC ~0.761

semantic residual:
~chance

text:
약한 signal
~~~

이었다.

결론:

~~~text
emotion information exists in hidden space.
mapping to semantic likelihood is bottleneck.
~~~

---

## 7.2 과신과 분리를 구분

이전 controller는:

~~~text
AUC는 괜찮음
accuracy도 chance보다 높음
CE는 매우 나쁨
~~~

이었다.

이건 단순히 아무것도 못 배운 것이 아니다.

logit span이 60 이상 폭발.

즉:

~~~text
ranking은 배움
하지만 지나치게 overconfident
~~~

였다.

그래서 bounded cosine classifier로 바꿈.

결과:

~~~text
hidden span ~2-3
balanced CE < random CE
OOF macro acc ~44%
AUC ~0.694
~~~

로 안정화.

이 사례처럼 CE/AUC/accuracy를 같이 본다.

---

## 7.3 hidden_direction의 현재 의미

현재 inference steering은:

~~~text
emotion direction
→ S1 readout projection
→ vocabulary preference
→ bounded probability tilt
~~~

이다.

장점:

- 안정적
- tag마다 명확한 차이
- multi-tag switching 가능
- base distribution support 보존

단점:

- direction 자체는 context-independent
- S1 hidden/readout geometry에 의존
- label 의미 정렬은 data quality에 의존

현재 결과가 실제로 작동하므로 이 구조를 함부로 복잡하게 만들지 않는다.

---

# 8. S1 문제를 보는 방식

## 8.1 S1은 단순 text LM이 아니다

inference에서 S1은:

~~~text
target text
+
reference semantic prompt
→ target semantic
~~~

을 생성한다.

reference sensitivity는 attention + audio embedding + hidden geometry + readout 전체에 의존.

---

## 8.2 현재 fine-tuning objective mismatch

기존 S1 FT는 주로:

~~~text
utterance B text
→ utterance B semantic
~~~

을 teacher forcing.

하지만 실제 inference:

~~~text
reference A semantic
+
target B text
→ target B semantic
~~~

이다.

이 mismatch가 핵심 가설.

---

## 8.3 full FT가 위험한 이유

작은 single-speaker dataset에서:

- embeddings
- 24 Transformer blocks
- Q/K/V
- FFN
- LayerNorm
- readout

전체가 바뀜.

그러면 pretrained reference routing 자체가 사라질 수 있음.

---

# 9. LoRA 연구에서의 원칙

## 9.1 LoRA = 안전으로 생각하지 않는다

LoRA도 pretrained behavior를 바꿀 수 있다.

중요한 것은:

~~~text
어디에 LoRA를 넣는가?
~~~

이다.

---

## 9.2 reference routing 관점으로 Q/K/V 분리

Self-attention에서 개념적으로:

~~~text
Q/K = 어디를 볼지
V = 무엇을 전달할지
O = 전달한 것을 어떻게 섞을지
~~~

따라서 reference routing을 보존하고 싶다면:

~~~text
Q/K frozen
V + O LoRA
~~~

가 유망.

하지만 이것도 실험으로 확인.

---

## 9.3 semantic coordinate를 보존

현재 Controller는 S1 readout과 hidden geometry를 활용.

따라서 초기 PEFT에서는:

- audio embedding
- text embedding
- LayerNorm
- semantic readout

을 freeze하는 것이 논리적.

---

# 10. Retention Loss 설계 원칙

## 10.1 hidden retention 우선

frozen pretrained S1:

~~~text
h_base
~~~

adapted S1:

~~~text
h_tuned
~~~

target 구간에서 cosine geometry를 약하게 보존.

이유:

Controller가 hidden geometry에 직접 의존.

---

## 10.2 logit KL은 2순위

KL을 강하게 걸면 adaptation 자체가 막힐 수 있다.

hidden retention으로 부족할 때만.

---

## 10.3 relational retention은 장기 후보

더 직접적인 reference sensitivity 보존:

~~~text
delta_base = h(ref A) - h(ref B)
delta_tuned = h'(ref A) - h'(ref B)
~~~

를 비슷하게 유지.

이 방식은 absolute hidden보다 reference response 자체를 보존.

하지만 처음부터 넣지 않는다.

---

# 11. 데이터 연구 원칙

## 11.1 class count만 보지 않는다

예:

~~~text
happy 38
surprised 35
~~~

수는 비슷해도 acoustical separability가 다를 수 있다.

surprised는:

- F0 jump
- energy change
- rhythm break

가 명확.

happy는 더 subtle.

따라서 count만으로 성능 차이를 설명하지 않는다.

---

## 11.2 auto-label을 절대 진리로 쓰지 않는다

emotion2vec이 특정 class를 과대 예측할 수 있음.

따라서:

~~~text
teacher label
~~~

이 아니라:

~~~text
weak direction signal
~~~

로 사용하는 방향을 선호.

---

## 11.3 텍스트 confound를 항상 의심

text-only probe가 informative면:

~~~text
emotion label이 acoustic emotion뿐 아니라 transcript pattern과 연결
~~~

되어 있을 수 있다.

그러므로 same target text inference가 중요.

---

# 12. UI/Notebook 연구 원칙

## 12.1 UI bug와 backend bug를 분리

예:

2초마다 UI가 깜빡임.

이건 training logic 문제가 아니라 polling마다 component 전체 update일 가능성.

backend state는 유지하고 UI update 범위를 줄인다.

---

## 12.2 Gradio error는 사용자가 읽을 수 있어야 함

즉시 사라지는 toast는 좋지 않음.

현재 원칙:

~~~text
Gradio Error duration = None
~~~

사용자가 X로 닫을 때까지 유지.

---

## 12.3 notebook은 wrapper

가능하면 logic은 repo Python 코드에 둔다.

notebook cell은:

- install
- clone
- env
- launch

중심.

왜냐하면 notebook cell logic이 많아지면 Colab/Kaggle divergence가 커짐.

---

# 13. Kaggle 운영 원칙

## 13.1 working과 tmp 역할 분리

final artifact만 Kaggle working.

cache / repo / intermediate는 Kaggle tmp.

---

## 13.2 disk full을 model failure로 착각하지 않음

checkpoint save error는:

~~~bash
df -h
du -sh
~~~

먼저 확인.

---

## 13.3 GPU 여러 장이면 무조건 DDP하지 않음

작은 독립 모델:

~~~text
GPU0: GPT
GPU1: SoVITS
~~~

같은 병렬이 더 단순하고 실용적일 수 있음.

이번 S1/Controller 연구에서도 model-parallel보다 독립 experiment 병렬이 우선.

---

# 14. GitHub 수정 원칙

## 14.1 변경 전에 최신 SHA 확인

같은 파일 수정 전에 반드시 fetch.

stale SHA로 덮어쓰지 않음.

---

## 14.2 한 commit에 한 논리적 변화

가능하면:

~~~text
fix storage
fix controller architecture
add diagnostic
update docs
~~~

를 분리.

문제가 생겼을 때 revert/비교 쉬움.

---

## 14.3 실제 실행하지 않은 검증을 주장하지 않음

Static inspection:

~~~text
코드상 구조 확인
~~~

Runtime:

~~~text
실제 Kaggle에서 실행 성공
~~~

을 명확히 구분.

---

# 15. 실험 로그를 읽는 방식

로그 전체를 처음부터 끝까지 같은 비중으로 보지 않는다.

## 15.1 마지막 정상 지점

가장 중요.

## 15.2 첫 WARNING

항상 fatal은 아님.

예:

TensorFlow CUDA warning이 PyTorch CUDA failure를 의미하지 않을 수 있음.

## 15.3 metric trajectory

한 point보다:

~~~text
step100
step200
step300
...
~~~

추세.

## 15.4 validation gap

train이 좋아지고 val이 나빠지면 overfit.

---

# 16. Early stopping 원칙

초기 정해진 1000 step 같은 숫자를 신뢰하지 않는다.

현재 Controller에서는:

- base steps
- validation interval
- auto-extension
- patience
- restore best checkpoint

구조를 사용.

이 방식은 다른 small controller에도 선호.

---

# 17. 성능이 좋아졌을 때도 의심해야 할 것

PASS라고 끝내지 않는다.

확인:

- threshold gaming?
- leakage?
- class imbalance?
- metric calibration?
- 실제 audio?
- seed sensitivity?

현재 Controller 성공 때도:

~~~text
새 PASS rule로 통과했기 때문에 성공
~~~

이 아니라

~~~text
balanced CE 자체도 random보다 좋음
AUC도 의미 있음
listening에서도 tag difference 확인
~~~

을 같이 봄.

---

# 18. 실패했을 때의 행동

실패하면 곧바로 다른 모델로 갈아타지 않는다.

다음 질문:

~~~text
signal 자체가 없는가?
signal은 있는데 readout이 못 쓰는가?
optimization이 실패했는가?
validation만 이상한가?
data label이 틀렸는가?
~~~

---

# 19. Architecture를 단순하게 유지하는 기준

좋은 architecture:

- 데이터 흐름이 설명 가능
- component 간 책임 명확
- debugging 가능
- single source of truth
- checkpoint compatibility
- optional feature가 base path를 깨지 않음

나쁜 architecture:

- 작은 feature마다 새 process
- helper가 과도하게 쪼개짐
- hidden global state 많음
- 같은 정보가 여러 JSON에 중복
- circular dependency
- UI와 backend logic 뒤섞임

---

# 20. 최소 변경의 의미

최소 변경은:

~~~text
코드 줄 수가 가장 적음
~~~

이 아니다.

의미:

~~~text
기존 proven path를 최대한 보존하면서
문제 원인에 직접 닿는 부분만 바꿈
~~~

예:

semantic likelihood bottleneck이 문제면 전체 GPT를 수정하지 않고 controller mapping만 수정.

---

# 21. Controller와 S1의 책임 경계

S1:

- text/reference를 semantic generation space로 결합
- reference sensitivity
- prosody/semantic trajectory prior

Controller:

- S1 hidden/readout에서 emotion-specific direction 선택
- bounded steering

S2:

- semantic/acoustic decoding

문제가 감정 tag마다 다른 결과는 나는데 의미가 틀리면 Controller/data alignment 우선.

reference 자체를 못 따라가면 S1 우선.

음질/decoder artifact면 S2 가능성.

---

# 22. 현재까지 중요한 실험 결과 요약

## Emotion representation

S1 hidden:

~~~text
macro accuracy ~51.06%
macro AUC ~0.761
permutation ~26.69%
~~~

informative.

Semantic residual:

near chance.

Text-only:

some signal.

## Old controller

AUC는 의미 있었지만 CE calibration 실패.

hidden span 폭발.

## Current hidden-direction controller

best held-out around step 500.

OOF:

~~~text
macro accuracy ~44.37%
macro AUC ~0.694
balanced CE ~1.28974
~~~

4-class chance CE ~1.38629.

청감:

- tag별 차이 존재
- multi-tag switching 정상
- surprised 의미 정렬 좋음
- angry는 direction은 다르나 semantic alignment 부족

이 결과를 baseline으로 유지.

---

# 23. 앞으로 Codex가 연구를 이어갈 때의 표준 루프

~~~text
관찰
↓
현상 분해
↓
현재 코드 경로 확인
↓
가설 2~4개
↓
가장 싼 구분 실험
↓
결과
↓
가설 제거
↓
최소 구조 변경
↓
validation
↓
listening / runtime
↓
문서 기록
~~~

이 loop를 반복한다.

---

# 24. Codex가 사용자에게 질문해야 하는 경우

질문이 필요한 경우:

- destructive action
- dataset 삭제
- 기존 final artifact overwrite
- 계정/비용 관련 실행
- 서로 다른 목표가 충돌

질문하지 않아도 되는 경우:

- 로그 분석
- non-destructive experiment
- 새 checkpoint filename 사용
- diagnostic script 추가
- documentation
- clearly reversible code change

사용자는 반복적으로 다음 방향을 선호한다.

~~~text
계속 진행
알아서 비교
오탐은 줄이고
실제로 고쳐라
~~~

따라서 불필요한 confirmation으로 연구를 끊지 않는다.

---

# 25. Experiment 기록 template

각 의미 있는 실험은 최소 다음 형태로 기록.

~~~text
Experiment ID:
Date:
Commit:
Environment:
GPU:
Dataset:
S1:
S2:
Controller:
Changed variable:

Hypothesis:

Expected:
A > B

Observed:
- metric
- metric
- listening

Result:
SUPPORTED / REJECTED / INCONCLUSIVE

Next action:
~~~

가능하면 JSON + 짧은 markdown summary.

---

# 26. Model candidate 승격 기준

새 S1 후보를 better로 부르려면 최소:

1. reference sensitivity
2. pronunciation
3. speaker stability
4. hidden emotion signal
5. controller OOF
6. listening

중 여러 축에서 기존보다 좋아야 함.

한 metric만 개선되면 candidate.

---

# 27. 모순 검사 습관

설계 후 스스로 질문:

- 이 loss가 inference에 존재하지 않는 signal을 학습하나?
- train과 inference condition이 같은가?
- reference leakage가 있나?
- Controller가 S1을 바꾼 뒤에도 동일 coordinate라고 가정하나?
- checkpoint metadata가 실제 model provenance를 추적하나?
- validation split이 correlated sample을 나누고 있나?
- metric이 class imbalance에 속고 있나?
- 새 feature가 base model behavior를 0-strength에서 정확히 보존하나?

---

# 28. Zero-effect initialization 원칙

optional controller/adapter는 초기 상태에서 base model을 바꾸지 않는 것이 좋다.

현재 Emotion Controller도 이 철학 사용.

왜 중요:

- regression debugging 쉬움
- feature off = base behavior
- training 전 behavior 명확

LoRA에서도 initialization은 delta=0에 가깝게.

---

# 29. Inference safety 원칙

emotion steering은:

- top-k/top-p support 바깥 token을 새로 살리지 않음
- EOS mass 보존
- max log-odds shift bounded
- prefix cumulative budget

같은 안전장치를 사용.

목표는:

~~~text
base model을 대체
~~~

가 아니라

~~~text
base model probability 안에서 방향 조정
~~~

이다.

---

# 30. Reference 선정 원칙

Controller training reference는:

- train split
- leakage-sensitive pair group 밖
- speaker 일치
- 실제 inference에서 재사용 가능
- text/semantic/BERT provenance 일치

현재 reference를 checkpoint metadata에 저장하도록 개선.

---

# 31. Portable artifact 원칙

가능하면 final checkpoint가 자기 provenance를 알게 한다.

예:

- S1 SHA
- S2 SHA
- metadata path
- reference cache path
- reference info
- text feature config
- training mode
- validation metrics

그러나 대용량 audio를 무조건 checkpoint에 embed하지 않는다.

---

# 32. Python compatibility 접근

최신 Python 지원 목표가 있어도 무작정 package version을 올리지 않는다.

먼저:

- 실제 incompatibility
- upstream fix
- fork/issue
- import API change

확인.

작동하는 dependency를 불필요하게 흔들지 않는다.

---

# 33. 외부 모델/논문을 볼 때의 기준

새 기술을 발견하면:

1. 우리 bottleneck과 같은 문제인가?
2. input/output geometry가 비슷한가?
3. parameter scale이 비슷한가?
4. training regime이 비슷한가?
5. 코드에 최소 변경으로 적용 가능한가?

이 5개를 본다.

최신 논문이라는 이유만으로 적용하지 않는다.

---

# 34. 자동화 철학

Codex/Chrome automation은:

~~~text
사람 대신 버튼 누르기
~~~

만 하는 것이 아니라

~~~text
결과를 읽고 다음 실험을 선택하는 closed-loop research
~~~

로 사용.

그러나 destructive action은 자동화하지 않는다.

---

# 35. Browser/Kaggle 연구 시 실제 행동

Codex가 notebook을 조작할 때:

1. notebook이 최신 repo를 pull했는지 확인
2. cell output을 읽음
3. 실패 직전 cell과 traceback 저장
4. repo 수정
5. kernel/runtime 상태 고려
6. 필요한 경우 WebUI restart
7. 같은 experiment 재실행
8. 이전 artifact와 이름 충돌 피함

---

# 36. 결과를 repo에 남기는 기준

남긴다:

- architecture decision
- reproducible experiment
- meaningful metric
- root cause
- new tool
- changed default reason

남기지 않는다:

- 일시적인 console spam
- 수백 MB raw log
- 의미 없는 failed attempt
- secret/token/path private data

---

# 37. 현재 S1 연구에서 가장 중요한 질문

앞으로 Codex가 답해야 할 핵심:

1. full FT가 pretrained reference sensitivity를 실제로 감소시키는가?
2. 감소한다면 어느 parameter group drift와 가장 연관되는가?
3. Q/K frozen, V+O LoRA가 이를 줄이는가?
4. cross-utterance S1 objective가 reference sensitivity를 회복하는가?
5. hidden retention이 Controller-friendly geometry를 보존하는가?
6. improved S1에서 Controller OOF와 실제 tag meaning이 같이 좋아지는가?

---

# 38. 좋은 결과와 나쁜 결과의 예

좋은:

~~~text
reference sensitivity ↑
speaker identity =
CER =
Controller OOF ↑
tag meaning ↑
~~~

나쁜:

~~~text
Controller accuracy ↑
reference sensitivity ↓↓↓
~~~

후자는 over-specialized S1일 가능성.

---

# 39. 중간 결과가 애매할 때

애매하면 architecture를 더 추가하지 않는다.

다음 중 하나:

- seed repeat
- sample 확대
- metric 추가
- listening blind compare
- ablation 단순화

를 먼저.

---

# 40. 청감 평가를 무시하지 않는다

TTS에서 숫자가 완벽해도 사람이 듣기에 안 좋으면 실패.

특히:

- emotion naturalness
- speaker identity
- prosody
- weird artifact
- robotic rhythm

은 human listening이 중요.

다만 한 번 듣고 결론내지 않고 같은 조건 여러 문장.

---

# 41. 지금까지 특히 효과적이었던 연구 패턴

## Pattern A — probe before redesign

hidden signal을 probe하고 bottleneck을 찾음.

## Pattern B — bounded geometry

logit 폭발을 cosine/bounded score로 막음.

## Pattern C — grouped OOF

작은 데이터의 holdout noise 감소.

## Pattern D — diagnostic-only weak paths

semantic likelihood가 약하면 training objective에서 빼고 diagnostic로 남김.

## Pattern E — preserve base behavior

optional feature가 base behavior를 덮지 않게 함.

## Pattern F — provenance in checkpoint

S1/S2 SHA와 metadata 저장.

---

# 42. 과설계 경고

새 idea를 구현하기 전에 스스로 묻는다.

~~~text
이걸 안 넣어도 현재 문제를 해결할 수 있는가?
~~~

YES면 일단 넣지 않는다.

예:

LoRA로 해결되면:

- hidden KL
- relational retention
- multi-reference training
- joint training

까지 동시에 넣지 않는다.

---

# 43. Joint training을 늦추는 이유

S1 + Controller joint training은:

Controller loss를 줄이기 위해 S1 hidden 자체가 label에 맞게 움직일 수 있음.

그러면:

- reference generality
- hidden disentanglement
- controller quality

를 분리하기 어려움.

따라서:

~~~text
S1 adapt
→ freeze
→ Controller train
~~~

이 기본.

---

# 44. Controller semantic alignment 개선을 나중에 할 때

tag difference는 이미 있음.

따라서 다음 단계는:

~~~text
direction separation
~~~

이 아니라

~~~text
direction meaning alignment
~~~

.

가능한 방법:

- high-confidence anchor audio
- contrastive prototype
- weak emotion teacher
- relative alignment

하지만 S1 연구가 먼저.

---

# 45. 실험 비용 최적화

가장 싼 실험부터.

예:

~~~text
pretrained vs existing full15 비교
~~~

는 새 학습 없이 가능.

그 다음 5ep.

그 다음 LoRA 구현.

처음부터 가장 비싼 experiment를 하지 않는다.

---

# 46. 실패를 버리지 않고 information으로 사용

실패 실험도:

~~~text
무엇이 아니다
~~~

를 알려준다.

예:

semantic residual failure는:

~~~text
emotion signal 자체 없음
~~~

이 아니라

~~~text
semantic residual path가 bottleneck
~~~

라는 정보를 줌.

---

# 47. Default 변경 기준

UI default는 실험 결과가 충분할 때만 변경.

예:

BERT ON, confidence 0.35 같은 값은 실제 pipeline behavior를 보고 선택.

S1 full FT → LoRA default 변경도 반드시 evidence 필요.

---

# 48. Regression 방지

새 feature 후:

- pretrained inference
- no-controller
- neutral tag
- old checkpoint loader

가 여전히 작동하는지 확인.

특히 backward compatibility 중요.

---

# 49. Code style

선호:

- 단순
- PEP8
- 명시적 이름
- helper 과도 분산 금지
- circular import 회피
- failure message 구체적
- path provenance 명확

---

# 50. 현재 프로젝트의 실전 사고 패턴

새 문제가 생기면 다음 질문을 순서대로 한다.

~~~text
1. 이건 실제 실패인가, 단순 warning인가?
2. 마지막으로 정상 동작한 지점은?
3. data / training / checkpoint / inference / UI 중 어디서 처음 어긋나는가?
4. signal 자체는 존재하는가?
5. train과 inference condition은 같은가?
6. metric이 진짜 목표를 측정하는가?
7. 가장 작은 ablation은 무엇인가?
8. 기존 path를 유지하면서 어디만 바꾸면 되는가?
9. regression을 어떻게 검증할 것인가?
10. 결과를 재현 가능하게 어떻게 기록할 것인가?
~~~

---

# 51. 로그에서 자주 보는 함정

## CUDA warning

TensorFlow가 CUDA driver를 못 찾았다는 warning이 있어도 PyTorch CUDA가 정상일 수 있다.

실제 torch.cuda 상태를 확인.

## package update warning

FunASR update 알림 자체는 실패가 아님.

## cache miss

첫 실행 cache miss는 정상일 수 있음.

## nan metric

해당 branch에서 metric이 정의되지 않는 구조인지 먼저 확인.

---

# 52. 데이터 leakage 검사 방식

같은 source에서 잘린 clip들이 train/val에 분리되면 validation이 과대평가될 수 있다.

따라서 group key:

- source_id
- text_group_id
- pair_group_id

를 connected group처럼 다룬다.

reference와 target이 같은 leakage-sensitive group이면 reference 선택 자체를 거부하는 방향을 선호.

---

# 53. 음성 모델에서 숫자와 체감이 충돌할 때

다음 우선순위:

1. gross failure 여부: 무음/반복/깨짐
2. intelligibility
3. speaker identity
4. target behavior
5. auxiliary metric

즉 AUC가 높아도 발음이 깨지면 좋은 모델이 아니다.

---

# 54. Model provenance의 중요성

Controller와 S1은 강하게 결합.

따라서:

~~~text
controller A
+
다른 S1 B
~~~

를 억지로 사용하면 안 됨.

checkpoint에는 S1 SHA를 저장.

향후 S1 LoRA merge 후에도 최종 merged checkpoint SHA를 기준으로 Controller를 묶는다.

---

# 55. Inference reference 재사용에서 배운 것

임시 canonical wav를 사용자가 직접 다운로드하게 만드는 UX는 불편.

더 좋은 방식:

- controller가 training reference identity를 기억
- inference UI에서 training reference를 자동 불러옴
- 원본 dataset path 우선
- 임시 path fallback

즉 technical correctness뿐 아니라 실제 workflow friction도 줄인다.

---

# 56. UI default를 정할 때

default는 가장 aggressive한 값이 아니라:

~~~text
처음 사용해도 실패 확률이 낮고,
대부분의 데이터에서 합리적인 값
~~~

으로 둔다.

사용자가 조절 가능하게 유지.

---

# 57. Multi-GPU 판단 방식

model 하나가 너무 커서 한 GPU에 안 들어가는 경우가 아니면 무조건 model parallel을 선호하지 않는다.

독립 가능한 stage:

~~~text
GPT / SoVITS
~~~

는 GPU를 분리해 병렬 실행하는 것이 단순.

Controller는 작아서 1 GPU.

---

# 58. 저장 공간 설계 방식

final compact artifact와 resume/full checkpoint를 구분.

Kaggle에서는:

- full checkpoint는 필요할 때만
- compact inference weight 우선
- latest-only pruning
- temporary intermediate는 tmp

저장 실패는 학습 실패와 분리.

---

# 59. Dependency 문제 해결 방식

순서:

1. exact import error
2. current Python version
3. upstream package support
4. existing issue/fork
5. minimal pin/patch
6. fallback compatibility shim

처음부터 requirements 전체를 갈아엎지 않는다.

---

# 60. 앞으로 Codex가 새 idea를 제안할 때 요구되는 형식

좋은 제안:

~~~text
Observed:
현재 X가 Y에서 실패.

Hypothesis:
Z가 bottleneck.

Minimal test:
A와 B만 비교.

Expected:
Z가 맞으면 metric M이 이렇게 변함.

Implementation:
파일/함수 최소 수정.

Risk:
어떤 regression 가능.

Rollback:
어떻게 되돌림.
~~~

이 형식을 유지.

---

# 61. 실험 후 반드시 답해야 하는 질문

모든 실험 후:

~~~text
무엇을 배웠는가?
어떤 가설이 약해졌는가?
어떤 가설이 강해졌는가?
다음 실험이 왜 가장 정보량이 큰가?
~~~

를 기록.

---

# 62. 결과가 좋을 때 바로 merge/default 하지 않는 이유

좋은 첫 결과는:

- seed luck
- sample selection
- reference luck
- overfit

일 수 있음.

최소 한 번은 다른 문장/reference/seed에서 확인.

---

# 63. Reference sensitivity 측정에서 중요한 점

단순 output 차이가 큰 것이 좋은 것은 아님.

좋은 reference sensitivity:

~~~text
reference 차이를 반영
+
speaker/발음 유지
~~~

나쁜 sensitivity:

~~~text
reference 바꾸면 output이 크게 흔들리지만 불안정
~~~

따라서 divergence 하나만 최대화하지 않는다.

---

# 64. S1 hidden probe의 해석

hidden emotion probe가 좋아지면:

~~~text
감정 정보가 hidden에 더 잘 linearly separable
~~~

이라는 뜻.

하지만:

~~~text
실제 감정이 더 자연스러움
~~~

을 자동 의미하지 않는다.

Controller가 읽을 수 있는 representation인지 판단하는 지표.

---

# 65. Controller OOF의 해석

OOF macro accuracy/AUC는:

~~~text
unseen grouped samples에서 emotion direction discrimination
~~~

을 평가.

하지만 human semantic alignment와는 별도.

따라서 listening 필수.

---

# 66. 감정 의미 정렬을 평가할 때

예:

surprised가 놀람처럼 들리고 angry가 다른 느낌이지만 화남 같지 않다면:

~~~text
direction existence: 성공
direction semantic alignment: 일부 실패
~~~

로 평가.

이 구분이 중요.

---

# 67. S1 adaptation과 Controller를 동시에 평가하는 matrix

각 S1 후보마다:

~~~text
reference retention
hidden emotion probe
speaker consistency
CER
Controller OOF
tag separation
semantic alignment
multi-tag
alternate reference
~~~

를 같이 기록.

최종 선택은 Pareto trade-off.

---

# 68. 과거 오류에서 배운 운영 원칙

- path 하나 잘못되면 마지막 cell만 실패할 수 있음.
- stale notebook cell은 repo fix와 별개.
- disk quota는 실제 df로 확인.
- temp path는 session 종료 시 사라짐.
- final artifact path는 지속성이 필요.

---

# 69. 문서화 수준

Codex가 다음 세션에 들어왔을 때 이전 채팅이 없어도:

1. 현재 architecture
2. 왜 그렇게 됐는지
3. 마지막 성공 baseline
4. 다음 실험
5. 하지 말아야 할 것

을 이해할 수 있어야 함.

---

# 70. 최종 연구 원칙 요약

이 프로젝트에서 좋은 연구는:

~~~text
현상을 정확히 분해하고
현재 code path를 실제로 읽고
가장 작은 실험으로 가설을 구분하고
숫자와 청감을 함께 보고
문제를 만든 component만 수정하고
baseline과 provenance를 보존하고
결과를 다시 문서화하는 것
~~~

이다.

---

# 71. Codex에게 주는 최종 지침

이 문서를 읽은 뒤에도 적힌 순서를 기계적으로 따라가지 말 것.

새 runtime evidence가 기존 가설과 충돌하면 runtime evidence를 우선한다.

다만 결론을 바꿀 때는:

1. 무엇이 기존 가설과 충돌했는지
2. 어떤 코드/로그가 증거인지
3. 새 가설이 무엇인지
4. 가장 싼 검증 실험이 무엇인지

를 repo에 기록한다.

목표는 정답을 외우는 것이 아니라 같은 연구 습관을 유지하는 것이다.
