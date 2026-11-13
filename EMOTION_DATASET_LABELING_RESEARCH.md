# Emotion Dataset Labeling Research — SER / Pseudo-Label / Uncertainty / Speaker Calibration

작성일: 2026-11-13

목적: GPT-SoVITS Improve의 감정 데이터셋 자동 분류 단계(`tools/auto_emotion_sidecar.py`)를 최신 Speech Emotion Recognition(SER) 연구와 대조하고, **현재 구조를 크게 망가뜨리지 않으면서 라벨 정밀도와 Controller 학습 품질을 높일 수 있는 개선 방향**을 정리한다.

이 문서는 구현 전 연구/실험 계획이다. 현재 default 동작을 즉시 바꾸는 문서가 아니다.

관련:
- `ADVERSARIAL_EMOTION_REFINER_DESIGN.md`
- `tools/auto_emotion_sidecar.py`
- `tools/emotion_dataset_builder.py`
- `tools/train_emotion_controller.py`
- `RESEARCH_METHODOLOGY_HANDOFF.md`
- `PROJECT_RESEARCH_HISTORY.md`
- `CODEX_KAGGLE_S1_CONTROLLER_RUNBOOK.md`

---

# 0. 결론 먼저

현재 `speaker_relative_clr_median_v2`는 **절대 emotion2vec+ top-1을 그대로 쓰는 것보다 훨씬 낫다.**

현재 설계의 좋은 점:
- 화자별 habitual tone bias를 제거하려고 함.
- raw score가 아니라 CLR(log-ratio) 상대값을 사용.
- median baseline으로 outlier 영향을 줄임.
- 애매한 샘플을 non-neutral로 강제하지 않음.
- prediction cache / provenance가 있음.
- downstream Controller가 class-balanced sampling + grouped OOF를 사용함.

하지만 현재 가장 큰 약점은 다음과 같다.

## HIGH priority

1. **`other` / `unknown` 확률을 label acceptance에서 사실상 버림**
2. **softmax confidence를 calibration 없이 신뢰**
3. **speaker median만 빼고 class별 분산(MAD)을 정규화하지 않음**
4. **ambiguous와 true neutral을 같은 `neutral/default`로 합침**
5. **utterance 전체 평균 분류 하나만 사용하여 시간적 감정 변화/mixed emotion을 놓침**
6. **pseudo-label 정확도를 검증할 independent human-reviewed gold subset이 없음**

## MEDIUM priority

7. emotion2vec embedding 기반 prototype consistency 검사
8. SenseVoice 같은 독립 모델을 second opinion으로 사용
9. F0/energy/duration 같은 speaker-relative prosody를 diagnostic evidence로 사용
10. hard label 외에 uncertainty/distribution 정보를 sidecar에 보존
11. high-confidence → ambiguous 순 curriculum/sample weighting

## LOW / later

12. text sentiment를 weak conflict flag로 사용
13. fully soft-label Controller
14. multi-model learned ensemble
15. joint SER fine-tuning

현재 데이터가 약 102개 수준이므로 **더 큰 classifier를 붙이는 것보다 잘못 라벨된 5~15개를 제거/수정하는 것이 훨씬 높은 ROI**일 가능성이 크다.

---

# 1. 현재 코드의 실제 동작

현재 default model:
- `iic/emotion2vec_plus_large`

현재 emotion2vec+ parser는 다음 9-class를 인식할 수 있다.

- angry
- disgusted
- fearful
- happy
- neutral
- other
- sad
- surprised
- unknown

하지만 `DEFAULT_EMOTIONS`는 controller-target 후보로:

- neutral
- happy
- sad
- angry
- disgusted
- fearful
- surprised

만 사용한다.

이 distinction 자체는 나쁘지 않다.

문제는 **target class에서 제외하는 것**과 **reject/OOD evidence에서도 제외하는 것**이 현재 같은 효과를 낸다는 점이다.

---

# 2. 현재 speaker-relative CLR 방식

현재 `_clr_scores()`:

~~~text
clr_e = log(p_e) - mean(log(p_classes))
~~~

speaker마다:

~~~text
baseline_e = median(clr_e)
~~~

sample:

~~~text
relative_e = clr_e(sample) - baseline_e
~~~

그리고 non-neutral 중 가장 높은 class를 선택.

추가로 relative scores를 softmax하여 `relative_confidence`를 계산.

accept 조건:

~~~text
best overall == best non-neutral
AND best_relative >= 0.25
AND relative_confidence >= 0.35
~~~

아니면 neutral/default.

---

# 3. 현재 방식이 잘한 점

## 3.1 화자 상대화

SER은 speaker-dependent prosody 영향을 크게 받는다.

같은 pitch/energy라도 어떤 화자에게는 평소 말투이고 다른 화자에게는 강한 감정일 수 있다.

현재 코드는 각 speaker의 median profile을 빼므로:

~~~text
"이 음성이 일반적인 happy인가?"
~~~

보다

~~~text
"이 화자의 평소 voice profile보다 happy 방향으로 얼마나 이동했는가?"
~~~

를 본다.

이 방향은 최근 speaker-aware / speaker-relative SER 연구와 잘 맞는다.

관련:
- Speaker-Aware Multi-Task Learning for Speech Emotion Recognition, Interspeech 2025
- ASER: Attribute-Serialized Emotion Recognition, IEEE SPL 2026
- Inference-Only Speaker Adaptation Improves Cross-Lingual SER, 2025

## 3.2 CLR 사용

softmax probability는 compositional data다.

한 class 확률 상승은 다른 class 확률 하락과 연결된다.

log-ratio 변환은 raw probability difference보다 상대적 class composition을 다루기에 논리적이다.

## 3.3 median baseline

mean보다 outlier에 강함.

small dataset에 적합.

## 3.4 강제로 모든 sample을 감정 class에 넣지 않음

pseudo-label pipeline에서 매우 중요.

잘못된 감정 label을 많이 확보하는 것보다 적은 high-precision label이 Controller에는 더 유리할 수 있다.

현재 Controller는 class-balanced sampling을 이미 하므로 sidecar 단계에서 class count를 억지로 맞출 필요가 없다.

---

# 4. HIGH-1 — `other` / `unknown` OOD mass 문제

emotion2vec+ large는 공식적으로 9 class를 출력한다.

현재 parser는 `other`, `unknown`까지 보존한다.

하지만 speaker-relative calibration에서는:

~~~text
supported = allowed_emotions - {other, unknown}
~~~

이 되어 `other`/`unknown`이 빠진다.

그 결과 예를 들어 raw distribution이:

~~~text
unknown 0.45
other 0.25
happy 0.15
surprised 0.08
neutral 0.04
...
~~~

여도 supported class 내부에서 happy가 상대적으로 가장 높으면 happy 후보가 될 수 있다.

즉:

~~~text
"모델이 이 utterance를 known emotion으로 보는가?"
~~~

라는 질문이 사라진다.

## 개선

별도 값 저장:

~~~text
ood_mass = p(other) + p(unknown)
~~~

그리고 최소한:

~~~text
raw_top1 in {other, unknown}
OR ood_mass > threshold
~~~

면 non-neutral acceptance를 막는다.

중요:
- `neutral`은 실제 emotion class.
- `other/unknown`은 uncertainty / open-set reject.

둘을 같은 default로 취급하면 안 된다.

추천 metadata:

~~~json
{
  "emotion_label_state": "accepted|neutral|ambiguous|ood|failed"
}
~~~

Backward compatibility를 위해 `emotion_intended`는 기존대로 유지 가능하지만:
- accepted → emotion
- true neutral → neutral
- ambiguous/ood/failed → null

형태가 더 의미적으로 정확하다.

---

# 5. HIGH-2 — confidence calibration 문제

현재 두 confidence가 있다.

1. raw emotion2vec+ softmax max
2. relative scores를 다시 softmax한 `emotion_relative_confidence`

둘 다 "정답일 확률"로 보장되지 않는다.

SER에서는 confidence calibration 문제가 잘 알려져 있다.

관련:
- The Importance of Calibration: Rethinking Confidence and Performance of Speech Multi-label Emotion Classifiers, Interspeech 2023
- On Calibration of Speech Classification Models, Interspeech 2024
- Are you sure? Analysing Uncertainty Quantification Approaches for Real-world SER, Interspeech 2024

특히 emotion은 본질적으로 ambiguous하고:
- OOD
- noise
- silence
- mixed emotion
- poor recording
때문에 high softmax가 항상 reliable하지 않다.

## 당장 가능한 개선

각 sample에서 추가 계산:

### normalized entropy

~~~text
H = -sum p_i log p_i / log(K)
~~~

0에 가까움 = sharp  
1에 가까움 = ambiguous

### top1-top2 margin

~~~text
margin = p1 - p2
~~~

### OOD mass

~~~text
p(other) + p(unknown)
~~~

### relative class gap

~~~text
best_relative - second_best_relative
~~~

acceptance를 한 개 confidence threshold가 아니라:

~~~text
relative lift
+
relative gap
+
low OOD
+
reasonable entropy
~~~

의 conjunction으로 바꾸는 것이 더 안전하다.

정확한 threshold는 먼저 human-reviewed subset으로 calibration해야 한다.

---

# 6. HIGH-3 — median만 있고 MAD scale normalization이 없음

현재:

~~~text
relative_e = clr_e - median_e
~~~

만 사용한다.

문제:
각 emotion dimension의 자연 변동폭이 다를 수 있다.

예:

~~~text
happy CLR:
speaker 평소 표준 변동 ±0.8

disgusted CLR:
speaker 평소 표준 변동 ±0.15
~~~

둘 다 `+0.25`를 같은 정도의 의미 있는 deviation으로 볼 수 없다.

## 개선: robust z-score

speaker/class별:

~~~text
median_e
MAD_e = median(|clr_e - median_e|)
scale_e = 1.4826 * MAD_e
~~~

sample:

~~~text
z_e = (clr_e - median_e) / max(scale_e, floor)
~~~

이렇게 하면:

~~~text
"0.25 log-ratio 상승"
~~~

보다

~~~text
"이 speaker의 평소 class 변동폭 대비 몇 sigma 수준의 상승인가"
~~~

에 가까워진다.

### small N 문제

speaker sample이 적으면 MAD 불안정.

fallback:

~~~text
N 충분:
median + MAD

N 부족:
현재 median-only
~~~

또는 shrinkage:

~~~text
scale = blend(speaker MAD, global MAD)
~~~

가능.

현재 single-speaker 102 samples에서는 MAD 적용 조건이 좋다.

---

# 7. HIGH-4 — ambiguous를 neutral로 만드는 문제

현재 code:

~~~text
clear non-neutral → non-neutral label
otherwise → neutral
~~~

이 방식은 Controller target 관점에서는 큰 문제처럼 보이지 않을 수 있다.

Controller는 neutral을 직접 emotion direction class로 학습하지 않고 base로 사용하기 때문이다.

하지만 dataset semantics에서는 문제가 있다.

### true neutral

~~~text
실제로 평범하고 감정 변화가 적은 음성
~~~

### ambiguous

~~~text
happy인지 surprised인지 모델도 확신 못 함
~~~

### OOD

~~~text
7개 emotion으로 설명하기 어려움
~~~

는 서로 다르다.

이 셋을 모두 neutral로 만들면:
- neutral count가 부풀 수 있음.
- reference 선정/분석이 왜곡됨.
- 미래 neutral prototype을 오염.
- SER calibration을 평가하기 어려움.

## 개선

label state를 분리.

~~~text
accepted_non_neutral
neutral
ambiguous
ood
failed
~~~

Controller training:
- accepted_non_neutral만 direction target
- neutral은 base diagnostic/reference 후보
- ambiguous/ood/failed는 direction target 제외

이게 더 깨끗하다.

---

# 8. HIGH-5 — utterance mean pooling 한계

FunASR emotion2vec 구현을 확인하면 classifier path는:

~~~python
x = x.mean(dim=1)
x = self.proj(x)
x = softmax(x)
~~~

이다.

즉 전체 utterance frame을 평균한 뒤 classifying한다.

`granularity="frame"`은 embedding output granularity에는 영향을 주지만 기본 classifier score는 결국 utterance mean projection이다.

따라서:

~~~text
처음 2초 neutral
중간 happy
끝 surprised
~~~

같은 audio가 하나의 평균 class로 압축된다.

2026 SER 연구에서도 utterance-level hard labels가 temporal dynamics를 숨긴다는 문제가 계속 다뤄지고 있다.

관련:
- Conflict-Aware Pseudo-Labeling via Acoustic Signals for Multi-Task SER, Interspeech 2026
- P-SED: weakly supervised Speech Emotion Diarization, Interspeech 2026
- segment/frame-level SER work

## 최소 변경 개선

audio를 별도 chunk로 잘라 같은 emotion2vec+ classifier를 여러 번 호출.

예:

~~~text
full utterance
chunk 1
chunk 2
chunk 3
...
~~~

기록:

~~~text
chunk label distribution
chunk agreement ratio
chunk entropy mean/max
transition count
~~~

acceptance:
- full score strong
- chunk consensus strong

일 때 high-confidence.

conflict가 많으면:
- ambiguous
- 또는 controller target 제외.

### 주의

짧은 chunk는 emotion cue가 부족할 수 있다.

따라서 고정 0.5초 같은 너무 짧은 window는 피한다.

우선 실험:
- clip length에 따라 2~4 chunks
- overlap optional
- 최소 chunk 길이 1.5~2.5s 정도 후보

정확값은 dataset에서 테스트.

---

# 9. HIGH-6 — independent gold subset 부재

현재 Controller OOF가 좋더라도 OOF target 자체가 auto pseudo-label이면:

~~~text
"pseudo-label을 잘 예측한다"
~~~

는 것은 보여도:

~~~text
"사람이 듣는 실제 emotion과 label이 맞다"
~~~

는 것을 독립적으로 증명하지 못한다.

현재 dataset은 약 102 samples.

이 규모에서는 전체를 수동 라벨링할 필요도 없다.

## 가장 ROI 높은 방법

20~30개 review set.

선정:
- 각 class top confident 2~3
- 각 class lowest-margin 2~3
- rare class 전부 또는 최대한 많이
- OOD/unknown high mass
- chunk-conflict samples
- neutral candidate 일부

사람이 직접:
- perceived emotion
- confidence
- optional secondary emotion
을 기록.

그 subset으로:
- class precision
- coverage
- rejection quality
- confusion
- threshold grid
를 조절.

이것이 현재 pipeline에서 가장 강한 improvement일 수 있다.

---

# 10. Prototype consistency

emotion2vec는 embedding extractor로도 사용할 수 있다.

분류 head 하나의 softmax만 믿지 않고 representation space를 같이 볼 수 있다.

## 방법

1. high-confidence / human-reviewed anchor만 class별 수집
2. emotion2vec embedding 추출
3. class prototype = robust centroid
4. ambiguous sample embedding과 prototype cosine 비교

label accept:

~~~text
classifier label == prototype nearest label
~~~

또는:

~~~text
classifier strong
AND prototype margin sufficient
~~~

### 장점

- classifier head calibration 오류를 일부 보완.
- 같은 model representation이지만 독립 evidence 하나 추가.
- speaker-relative prototype도 가능.

### 위험

pseudo-label만으로 prototype을 만들고 반복 refinement하면 self-confirmation loop 가능.

따라서:
- first version은 human-reviewed/high-confidence anchor만 사용
- iteration 1회
- 자동 무한 refinement 금지

관련:
- Iterative Prototype Refinement for Ambiguous SER, Interspeech 2024

---

# 11. Independent second model

optional secondary model로 SenseVoiceSmall을 고려할 가치가 있다.

SenseVoice:
- Korean 지원
- emotion tags 지원
- emotion2vec와 architecture/training path가 다름

따라서 second opinion으로 의미가 있다.

## 사용법

emotion2vec+를 primary로 유지.

~~~text
emotion2vec == SenseVoice
→ confidence boost / agreement flag

emotion2vec != SenseVoice
→ disagreement flag
→ human review 우선
→ automatic label weight 낮춤
~~~

## 하지 말 것

~~~text
두 모델 majority vote = ground truth
~~~

로 만들지 않는다.

두 모델 모두 bias가 있고 domain mismatch가 있다.

또 emotion2vec seed/base/large끼리는 lineage가 강하게 공유되므로 independent ensemble 효과는 SenseVoice보다 약할 수 있다.

---

# 12. emotion2vec+ large가 항상 최선인가?

official model card:
- seed: 201h
- base: 4,788h
- large: 42,526h
- large ~300M

large는 가장 큰 pseudo-labeled corpus를 사용한다.

하지만 "우리 한국어 single-speaker domain에서 large가 무조건 최고"는 보장되지 않는다.

외부 cross-corpus benchmark에서도 seed/base/large 차이가 매우 작게 나오는 사례가 있다.

따라서 비용이 허용되면 작은 ablation:

~~~text
seed
base
large
~~~

에 대해 human-reviewed subset precision을 비교 가능.

단:
- external third-party benchmark 하나로 model default를 바꾸지 않음.
- 우리 data에서 검증해야 함.

현재 default large는 유지하는 것이 타당.

---

# 13. Prosody diagnostics

최근 연구는 speaker-relative prosodic attribute를 감정 evidence로 명시적으로 사용하는 방향도 보인다.

우리 pipeline에서도 cheap diagnostics 가능.

speaker-relative:
- F0 median
- F0 std/range
- voiced ratio
- RMS/energy
- duration
- speech rate proxy
- pause ratio

각 값을 robust z-score로 변환.

## 용도

label 결정의 primary rule로 사용하지 않는다.

대신 consistency flag:

예:

~~~text
surprised predicted
+
F0/energy variation도 speaker baseline 대비 높음
→ supporting evidence
~~~

또는:

~~~text
angry high confidence
+
모든 prosody metric 완전 baseline
→ review priority
~~~

이렇게 사용.

ASER 2026의 speaker-relative prosodic coding은 이 방향을 뒷받침한다.

---

# 14. Text / transcript를 어떻게 쓸 것인가

현재 S1 probe에서 text-only signal이 일부 존재했다.

하지만 최신 cross-modal 연구는 speech emotion과 text emotion의 상관이 제한적이며 특히 acoustic emotion을 text만으로 대체하면 안 된다는 점을 강조한다.

EACL 2026:
- text-speech emotion correlation은 valence에서 상대적으로 강하지만 전체적으로 제한적.
- emotional TTS에서는 text-only prediction 불충분.

따라서:

~~~text
audio SER = primary
text sentiment = weak conflict diagnostic
~~~

가 맞다.

예:

~~~text
audio says happy
text strongly negative
→ acoustic_semantic_conflict = true
~~~

하지만 text로 audio label을 overwrite하지 않는다.

---

# 15. Hard label 대신 distribution 보존

현재 sidecar에는 raw scores가 prediction cache에는 있지만 final sidecar에는 relative score 중심.

미래를 위해 다음을 sidecar에 보존하는 것이 좋다.

~~~json
{
  "emotion_raw_scores": {},
  "emotion_entropy": 0.0,
  "emotion_top1_margin": 0.0,
  "emotion_ood_mass": 0.0,
  "emotion_relative_z_scores": {},
  "emotion_relative_z": 0.0,
  "emotion_relative_gap": 0.0,
  "emotion_label_state": "accepted"
}
~~~

현재 Controller는 hard label만 사용해도 된다.

나중에:
- sample weighting
- curriculum
- soft supervision
을 실험할 때 재분류 없이 활용 가능.

2026 annotation uncertainty 연구는 hard consensus label보다 distribution을 보존하는 것이 human uncertainty를 더 잘 표현할 수 있음을 보여준다.

---

# 16. Curriculum / sample weighting

현재 Controller는 class balancing은 잘하지만 sample quality weighting은 없다.

future option:

~~~text
Tier A:
human reviewed / high-confidence consensus

Tier B:
strong auto-label

Tier C:
ambiguous but usable

Tier D:
reject
~~~

training:
- Tier A/B 먼저
- C는 나중 또는 낮은 weight
- D 제외

CHUCKLE 2026은 human agreement 기반 easy→hard curriculum이 SER 학습에 도움이 될 수 있음을 보여준다.

우리 dataset은 작으므로 복잡한 curriculum보다 먼저:
- A/B만 train
- C 추가 ablation
으로 충분.

---

# 17. Class imbalance 처리

현재 실제 label count는 불균형.

하지만 sidecar 단계에서:

~~~text
rare class를 채우기 위해 threshold를 낮추는 것
~~~

은 추천하지 않는다.

그것은 noisy label을 만든다.

현재 Controller는 least-trained class sampling으로 class imbalance를 상당 부분 처리한다.

따라서 sidecar 목표:

~~~text
coverage 최대화
~~~

보다

~~~text
per-class precision 최대화
~~~

가 우선.

rare class가 3개 미만이면:
- 그 class를 이번 Controller에서 제외
- 또는 human review/추가 데이터 확보

가 더 좋다.

---

# 18. 현재 fearful=1 문제의 해석

fearful이 단 1 sample이면:

- Controller class로 학습할 수 없음.
- threshold를 낮춰 fearful를 억지로 늘리는 것은 위험.

대신:
- fearful raw score가 높은 sample review
- prototype/secondary model agreement
- 실제 데이터에서 fear expression 자체가 적은지 확인

후 충분한 genuine examples가 생기면 class 추가.

---

# 19. surprised 과다 문제

과거 emotion2vec 결과에서 surprised가 과다하게 나왔던 적이 있음.

현재 speaker-relative calibration은 이를 줄이기 위한 좋은 접근이었다.

v3에서는 더 확인:

~~~text
surprised raw prior
speaker-relative z
prototype similarity
chunk consistency
human reviewed precision
~~~

를 분리 기록.

surprised가 많다고 자동으로 bias라고 단정하지 않는다.

실제로 청감에서 surprised direction은 현재 가장 잘 의미 정렬된 class 중 하나.

---

# 20. v3 labeling algorithm 제안

새 default로 바로 바꾸지 않고 experimental mode로 구현.

## Stage 0 — raw inference

emotion2vec+ 9-class full distribution 저장.

~~~text
angry
disgusted
fearful
happy
neutral
other
sad
surprised
unknown
~~~

embedding도 optional cache.

## Stage 1 — raw uncertainty

계산:
- top1
- top2
- top1 margin
- normalized entropy
- OOD mass (`other + unknown`)
- duration

## Stage 2 — speaker-relative robust calibration

primary emotion 7 classes에서:
- CLR
- speaker median
- MAD / shrinkage scale
- relative robust z

계산:
- best z
- second z
- z gap

## Stage 3 — label-state gating

~~~text
prediction failed
→ failed

OOD mass high / raw top1 unknown-other
→ ood

known classes but uncertain
→ ambiguous

clear neutral
→ neutral

strong non-neutral
→ accepted
~~~

## Stage 4 — temporal consistency optional

full utterance + chunks.

conflict:
- mixed label
- high chunk entropy
- low consensus

이면 ambiguous 또는 confidence penalty.

## Stage 5 — prototype optional

classifier class와 embedding prototype이 일치하는지 확인.

## Stage 6 — second model optional

SenseVoice agreement flag.

---

# 21. v3에서 중요한 distinction

## label

~~~text
무슨 감정으로 학습할 것인가
~~~

## quality

~~~text
이 label을 얼마나 믿는가
~~~

## state

~~~text
accepted / neutral / ambiguous / OOD / failed
~~~

이 세 개를 분리한다.

현재 v2는 이들이 일부 섞여 있다.

---

# 22. 추천 metadata schema

~~~json
{
  "emotion_intended": "surprised",
  "emotion_label_state": "accepted",
  "emotion_label_quality": "high",
  "emotion_predicted": "surprised",
  "emotion_predicted_confidence": 0.71,
  "emotion_raw_scores": {
    "surprised": 0.71,
    "happy": 0.12,
    "unknown": 0.05
  },
  "emotion_entropy": 0.42,
  "emotion_top1_margin": 0.59,
  "emotion_ood_mass": 0.07,
  "emotion_relative_z_scores": {
    "surprised": 2.1,
    "happy": 0.4
  },
  "emotion_relative_z": 2.1,
  "emotion_relative_gap": 1.7,
  "emotion_temporal_consensus": 0.8,
  "emotion_temporal_conflict": false,
  "emotion_prototype_label": "surprised",
  "emotion_prototype_margin": 0.18,
  "emotion_secondary_model": "surprised",
  "emotion_model_agreement": true,
  "auto_emotion_calibration": "speaker_relative_robust_v3"
}
~~~

모든 field를 처음부터 의무화할 필요는 없음.

---

# 23. Human review workflow

현재 dataset 규모에서는 반드시 고려할 가치가 있다.

## 자동 selection

review set 약 20~30개.

우선:
1. rare classes
2. each class top-confidence
3. each class boundary
4. high OOD
5. high entropy
6. chunk disagreement
7. prototype disagreement
8. model disagreement

## 사람 입력

최소:
- primary perceived emotion
- confidence (high/medium/low)

optional:
- secondary emotion
- comment

## 저장

`emotion_review.jsonl`

raw auto label은 덮어쓰지 않고 manual override 별도 field.

---

# 24. Threshold calibration 실험

정확한 threshold를 지금 이론으로 정하지 않는다.

review set으로 grid search.

후보 예시일 뿐:

~~~text
OOD max:
0.15 / 0.25 / 0.35

entropy max:
0.55 / 0.65 / 0.75

raw margin min:
0.10 / 0.20 / 0.30

relative robust z min:
0.75 / 1.0 / 1.5 / 2.0

relative z gap min:
0.25 / 0.5 / 0.75
~~~

목표:

~~~text
precision 먼저
그 다음 coverage
~~~

dataset이 작고 Controller가 class-balanced이기 때문.

---

# 25. Evaluation metric

auto-labeler 자체를 평가할 때:

- per-class precision
- macro precision
- coverage
- abstention rate
- OOD rejection rate
- confusion matrix
- class counts after gating
- reviewed subset ECE
- Brier score (enough labels if possible)
- entropy vs correctness
- margin vs correctness

Controller metric과 분리.

---

# 26. v2 vs v3 실험

## V2

current:
`speaker_relative_clr_median_v2`

## V3-A

V2 +
- OOD mass
- entropy
- raw margin
- ambiguous != neutral
- no new MAD yet

## V3-B

V3-A +
- median/MAD robust z

## V3-C

V3-B +
- chunk consensus

## V3-D

V3-C +
- prototype agreement

## V3-E

V3-D +
- SenseVoice disagreement diagnostic

한 번에 모두 넣지 않는다.

---

# 27. 승격 규칙

v3 variant가 default가 되려면:

1. reviewed subset per-class precision이 v2보다 좋아짐
2. rare class가 noise로 채워지지 않음
3. useful label coverage가 지나치게 붕괴하지 않음
4. Controller hidden OOF가 유지/향상
5. 실제 tag semantic alignment가 유지/향상
6. preprocessing 시간/복잡성이 수용 가능

---

# 28. 가장 먼저 구현할 가치가 있는 것

## 1위 — diagnostics only

behavior를 안 바꾸고 sidecar/report에:
- full raw scores
- entropy
- raw top1/top2 margin
- OOD mass

추가.

이건 risk가 거의 없음.

## 2위 — human review selector/evaluator

`tools/build_emotion_review_set.py`  
`tools/evaluate_emotion_sidecar.py`

같은 도구.

102개에서 가장 큰 정보량.

## 3위 — experimental MAD v3

flag로 실행:
`--calibration speaker_relative_robust_v3`

default는 v2 유지.

## 4위 — ambiguous/ood state 분리

downstream compatibility 확인 후.

## 5위 — chunk consensus

추가 compute가 있으므로 그 다음.

## 6위 — prototype / SenseVoice

diagnostic evidence가 부족할 때.

---

# 29. 구현 시 backward compatibility

기존 fields 유지:

- emotion_intended
- emotion_predicted
- emotion_predicted_confidence
- emotion_relative_scores
- emotion_relative_score
- emotion_relative_confidence
- emotion_is_default

새 fields는 additive.

기존 Controller가 모르는 field는 무시하게 유지.

---

# 30. Prediction cache 주의

raw emotion2vec distribution은 이미 cache에 저장.

따라서 uncertainty diagnostics만 추가할 경우 모델 inference를 다시 할 필요 없이 old cache에서 계산 가능.

cache format을 바꾸지 않는다면 `PREDICTION_CACHE_VERSION`을 불필요하게 증가시키지 않는다.

embedding/prototype까지 cache하려면 version/change 필요.

---

# 31. Audio quality / silence

SER uncertainty 논문에서:
- corrupted signals
- silence
- OOD
가 잘못된 prediction 원인이 될 수 있음.

따라서 future cheap checks:
- duration
- RMS
- voiced ratio
- clipping
- silence ratio

quality gate.

하지만 너무 예민한 heuristic으로 valid emotional speech를 버리지 않도록 diagnostic부터.

---

# 32. Recording/domain shift

emotion2vec+가 큰 데이터에서 학습됐어도 우리 target voice / recording setup은 다른 domain.

Cross-corpus SER 연구는 domain shift가 큰 문제임을 계속 보고한다.

speaker-relative calibration은 speaker shift 일부를 줄이지만:
- microphone
- compression
- room reverb
- acting style
까지 해결하지 못함.

human anchor/prototype는 target-domain calibration 역할도 함.

---

# 33. Multilingual/Korean 측면

emotion2vec 연구는 multilingual generalization을 강조하고, SenseVoiceSmall도 Korean을 지원.

하지만:

~~~text
한국어 지원
≠
우리 speaker/domain에서 calibrated
~~~

이므로 model card의 multilingual claim만으로 label을 신뢰하지 않는다.

target-domain review가 필요.

---

# 34. 최신 연구에서 우리에게 직접 유용한 메시지

## emotion2vec 2024

universal representation과 frame/utterance learning은 강한 basis.

→ emotion2vec를 버릴 이유 없음.

## Calibration 2023/2024

SER softmax confidence는 overconfidence 가능.

→ current `0.35` 하나로 reliability를 표현하면 부족.

## UQ 2024

ambiguity/OOD/silence를 reject할 수 있어야 함.

→ unknown/other/entropy 중요.

## Prototype refinement 2024

ambiguous sample은 prototype similarity로 refine 가능.

→ optional v3-D.

## Speaker-aware 2025/2026

speaker characteristics와 relative prosody가 중요.

→ current speaker-relative baseline은 좋은 방향.  
→ median+MAD 강화가 자연스러움.

## EACL 2026 text-speech correlation

text emotion은 speech emotion 대체 불가.

→ text는 weak conflict flag.

## Annotation uncertainty 2026

emotion은 hard label 하나로 압축하기 어려움.

→ score distribution/entropy 저장.

## Conflict-aware pseudo labeling 2026

utterance label은 temporal dynamics를 숨김.

→ chunk consensus.

## CHUCKLE 2026

human agreement가 sample difficulty의 좋은 proxy.

→ 20~30개 human review가 매우 가치 있음.

---

# 35. 관련 주요 자료

emotion2vec:
- https://aclanthology.org/2024.findings-acl.931/
- https://huggingface.co/emotion2vec/emotion2vec_plus_large
- https://github.com/modelscope/FunASR/blob/main/funasr/models/emotion2vec/model.py

Calibration / uncertainty:
- https://www.isca-archive.org/interspeech_2023/chou23_interspeech.html
- https://www.isca-archive.org/interspeech_2024/hao24_interspeech.html
- https://www.isca-archive.org/interspeech_2024/schrufer24_interspeech.html

Ambiguity / prototype:
- https://www.isca-archive.org/interspeech_2024/sun24e_interspeech.html
- https://arxiv.org/abs/2606.27536

Speaker:
- https://www.isca-archive.org/interspeech_2025/shi25d_interspeech.html
- https://signalprocessingsociety.org/publications-resources/ieee-signal-processing-letters/2026/09/aser-attribute-serialized-emotion-recognition-generative-multi-task-learning
- https://aclanthology.org/2025.poleval-main.12/

Cross-modal:
- https://aclanthology.org/2026.findings-eacl.136/

2026 Interspeech:
- https://www.isca-archive.org/interspeech_2026

SenseVoice:
- https://github.com/QwenAudio/SenseVoice

SER datasets / benchmarking:
- https://emo-box.github.io/
- https://lab-msp.com/MSP/MSP-Podcast.html

---

# 36. 최종 권고

현재 구조를 폐기하지 않는다.

다음 순서가 가장 정보 효율적이다.

~~~text
현재 v2 유지
↓
raw uncertainty diagnostics 추가
↓
20~30개 human review set
↓
v2 precision/coverage 실제 측정
↓
MAD robust z experimental
↓
ambiguous/neutral/OOD 분리
↓
chunk consistency
↓
prototype consistency
↓
optional SenseVoice disagreement
↓
좋은 variant만 Controller 재학습 비교
~~~

핵심 목표는:

~~~text
"감정 라벨 수를 많이 만들기"
~~~

가 아니라:

~~~text
"Controller가 믿고 학습할 수 있는 감정 anchor를 만들기"
~~~

이다.
