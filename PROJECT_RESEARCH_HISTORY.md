# Project Research History & Decision Log

작성일: 2026-11-13

목적: GPT-SoVITS Improve 프로젝트에서 지금까지 어떤 문제를 발견했고, 어떤 가설을 세웠고, 무엇을 버렸고, 왜 현재 구조를 선택했는지를 시간 순서와 의사결정 중심으로 기록한다.

이 문서는 "현재 코드가 무엇인가"보다 "왜 현재 코드가 그렇게 되었는가"를 설명한다.

관련:

- RESEARCH_METHODOLOGY_HANDOFF.md
- CODEX_KAGGLE_S1_CONTROLLER_RUNBOOK.md
- S1_REFERENCE_RETENTION_EXPERIMENTS.md

---

# 1. 기본 프로젝트 방향

초기부터 유지한 원칙:

- 기존 GPT-SoVITS pipeline을 최대한 보존.
- 과도한 architecture 재설계 금지.
- 감정 control은 discrete tag 중심.
- slider보다 [happy], [surprised] 같은 tag UX 선호.
- neutral/default는 가능하면 base behavior.
- Controller는 base GPT-SoVITS를 완전히 대체하지 않고 제한된 steering 역할.
- 작은 dataset에서 validation 없이 training loss만 믿지 않음.
- canonical preprocessing → feature → training → inference가 하나의 흐름으로 이어져야 함.

---

# 2. Emotion Controller 초기 문제

초기 아이디어는 S1 semantic likelihood를 감정별로 조절하는 low-rank preference controller였다.

목표:

~~~text
S1 base semantic distribution
+
emotion-specific low-rank adjustment
→ emotion-conditioned semantic generation
~~~

초기 기대:

- S1이 이미 emotion 정보를 포함하고 있다면 작은 controller로 충분할 수 있음.
- backbone frozen이면 speaker/발음 안정성이 좋을 것.
- 작은 parameter로 빠르게 학습 가능.

---

# 3. 첫 중요한 진단 — 감정 정보가 어디 있는가

controller가 기대만큼 학습되지 않았을 때 바로 architecture를 늘리지 않고 representation probe를 수행.

사용한 주요 관찰:

~~~text
S1 hidden probe:
OOF macro accuracy ≈ 51.06%
macro AUC ≈ 0.761
permutation ≈ 26.69%

semantic full residual:
macro accuracy ≈ chance
AUC ≈ 0.518

semantic top-k residual:
macro accuracy ≈ chance
AUC ≈ 0.493

text-only:
macro accuracy ≈ 37.83%
AUC ≈ 0.605
~~~

핵심 결론:

~~~text
감정 signal은 S1 hidden에는 존재.
semantic likelihood/residual 쪽으로 내려가면서 signal이 약해짐.
~~~

따라서 "emotion signal 자체가 없음" 가설은 약해지고 "mapping bottleneck" 가설이 강해짐.

---

# 4. Old controller calibration 실패

이전 controller에서 다음 현상 관찰:

- AUC는 의미 있음.
- macro accuracy도 완전 chance는 아님.
- CE는 매우 나쁨.
- hidden logit span이 크게 폭발.

대표 결과:

~~~text
hidden balanced CE ≈ 4.74662
macro accuracy ≈ 43.97%
macro AUC ≈ 0.719
~~~

4-class random CE는 약 1.386.

즉:

~~~text
ranking/discrimination은 일부 배움
하지만 calibration이 완전히 망가짐
~~~

으로 판단.

이 때문에 "모델이 아무것도 못 배웠다"로 결론내리지 않음.

---

# 5. Hidden-primary 방향 전환

이후 controller를 S1 hidden signal 중심으로 바꿈.

관련 commit:

- 64de8dd0f03ccb7864f822ff4a421eea852e9906
  - hidden emotion logits 추가

- 63000993465a89116ca9ac5d6deddf04a9b51cc9
  - hidden-primary training 정리

초기 hidden-primary는:

~~~text
hidden CE
+
작은 semantic auxiliary
~~~

형태.

하지만 semantic path가 계속 weak했기 때문에 hidden을 primary signal로 둠.

---

# 6. Validation / OOF 강화

작은 controller에서 단일 validation split만 신뢰하지 않기 위해:

- held-out validation
- grouped split
- 5-fold 수준 OOF
- macro accuracy
- macro AUC
- balanced CE

를 함께 사용.

group leakage 방지:

- source_id
- text_group_id
- pair_group_id

를 연결된 group으로 취급.

---

# 7. Hidden-direction architecture 도입

가장 큰 전환.

commit:

- e0d402766a5877953bf0c6c44f0f7662cedec72c
  - Steer directly from learned hidden emotion directions

핵심:

~~~text
controller hidden direction
→ normalized S1 semantic readout
→ token preference
~~~

기존 weak semantic likelihood mapping을 우회.

동시에 hidden classifier는 bounded cosine 형태로 변경.

목표:

- logit explosion 방지
- emotion directions 간 geometry 안정화
- calibration 개선
- semantic residual bottleneck 우회

---

# 8. Hidden-direction training rule

commit:

- d45ad4fd58f69abab67b4dc92948edbd396b507c

주요 default:

~~~text
hidden_lr = 3e-3
hidden_weight_decay = 1e-2
semantic_aux_weight = 0.0
hidden_beta = 2.0
~~~

hidden_direction에서는 semantic CE를 training objective에서 사실상 제거.

semantic path는 diagnostic으로 남김.

이 결정 이유:

~~~text
약한 signal을 억지로 objective에 넣어
좋은 hidden direction까지 망가뜨리지 않기 위해.
~~~

---

# 9. PASS rule 변경

초기에는 CE 위주 gate가 너무 엄격하고 calibration에 민감.

새 rule은 discrimination 중심:

4-class에서 대략:

~~~text
macro accuracy >= 35%
macro AUC >= 0.60
~~~

를 generalization_detected 기준으로 사용.

단, 실제 성공 판단은 이 rule만으로 하지 않음.

나중에 successful run에서 balanced CE 자체도 random보다 좋아져 old/new metric이 함께 개선됨.

---

# 10. 성공한 hidden-direction run

대표 결과:

~~~text
train samples ≈ 74
val samples ≈ 19

step 100:
val loss ≈ 0.90622
acc ≈ 62.35%

step 200:
val loss ≈ 0.85708
acc ≈ 62.35%

step 300:
val loss ≈ 0.80327
acc ≈ 74.26%

step 400:
val loss ≈ 0.78987
acc ≈ 77.83%

step 500:
val loss ≈ 0.76188
acc ≈ 80.95%
best

step 600:
val ≈ 0.85772
patience 1

step 700:
val ≈ 0.91708
patience 2

early stop
restore step 500
~~~

OOF:

~~~text
hidden OOF CE ≈ 1.22499
hidden balanced CE ≈ 1.28974
macro accuracy ≈ 44.37%
macro AUC ≈ 0.694
verdict = generalization_detected
~~~

중요:

4-class random CE ≈ 1.38629.

따라서 새 PASS gate만 통과한 것이 아니라 CE도 random보다 좋아짐.

---

# 11. 이 성공에서 배운 것

이전:

~~~text
hidden span > 60
~~~

새 구조:

~~~text
hidden span ~1.2 → 3.5
~~~

즉 calibration 안정화가 핵심 효과.

또한:

~~~text
semantic_ce=nan
grad_u=0
u_norm=0
~~~

가 hidden_direction에서는 버그가 아니라 정상적인 unused path임을 확인.

---

# 12. 진단 메시지 개선

commit:

- f5c3dd32d565116a20d125f11c79767f455dd567

목적:

semantic diagnostic이 weak하더라도 hidden-direction 모드에서는 그것을 bottleneck failure처럼 표시하지 않게 함.

즉 log 자체가 architecture mode를 정확히 설명하도록 수정.

---

# 13. Canonical dataset reuse UX 문제

같은 Kaggle session에서 canonical dataset이 이미 존재하면 재학습 시 충돌.

commit:

- d1936e02e1b5393beac6630e1beeef8e3257c708

기존 canonical dataset 재사용 default를 True로 변경.

---

# 14. Sidecar path UX 문제

UI에서 sidecar path가 비어 있으면 실제 default path가 있는데도 error 가능.

commit:

- b9341f21063d08e462fdf4a2c5c5d28c3f3199d5

pipeline에서 default sidecar 자동 resolve 및 존재 확인.

---

# 15. Reference reuse UX 개선

사용자가 Controller training reference를 inference에서도 그대로 쓰고 싶어함.

문제:

- canonical temp reference는 Kaggle tmp에 있음.
- 사용자가 직접 다운로드/업로드하는 UX 불편.
- session 종료 시 tmp 삭제.

개선:

### commit 1b0696853f255434df95e8b7edb5129c971b5598

새 Controller checkpoint에 inference reference metadata 저장.

저장 내용:

- speaker_uid
- canonical_wav
- audio_path
- original_path
- text
- language

원본 dataset audio path를 우선.

### commit 7de995588a3ae1c0f53f4e83baeec9f5240f3432

inference helper 추가.

우선순위:

1. checkpoint embedded inference_references
2. legacy checkpoint면 reference_prompt_cache + metadata에서 복구
3. original path
4. canonical temp fallback

### commit 0418bdf8813fd8481eef137eed65c75ec2f6c96b

Inference UI 버튼:

~~~text
Controller 학습 Reference 사용
~~~

추가.

reference audio + prompt text + language 자동 입력.

---

# 16. Multi-tag inference 확인

사용자가 실제 inference에서:

~~~text
[neutral]
[happy]
[surprised]
[angry]
[disgusted]
[neutral]
~~~

같은 multi-tag sequence를 테스트.

결과:

~~~text
다중 태그 정상 동작.
태그마다 발화 느낌이 서로 다름.
~~~

이것은 중요한 functional validation.

---

# 17. 감정 semantic alignment 관찰

실제 청감:

- surprised: 놀란 느낌이 잘 구현되는 편
- happy: surprised보다 덜 명확
- angry: 실제 angry 데이터 부족으로 분노 느낌은 약함
- 그러나 angry 포함 태그마다 서로 다른 스타일 direction 존재

결론:

~~~text
tag direction separation은 성공
semantic meaning alignment는 class별 차이
~~~

따라서 controller 구조 재설계보다 data/alignment 문제 가능성.

---

# 18. S1 reference sensitivity 문제 발견

upstream issue 조사에서:

- GPT/S1 full fine-tuning 후 reference/timbre/emotion 특징이 불안정
- pretrained s1v3 사용 시 개선

사례 확인.

특히 upstream issue #2843 조건이:

~~~text
s1v3
15 epochs
fine-tuned GPT
~~~

로 현재 baseline과 매우 유사.

---

# 19. S1 구조를 다시 읽은 결과

inference:

~~~text
target text
+
reference semantic prompt
→ Transformer
→ next semantic token
~~~

reference semantic은 prompts로 들어가고 audio embedding + attention을 통해 target generation에 영향.

반면 standard S1 fine-tuning dataset은 각 utterance의:

~~~text
text
semantic sequence
~~~

를 teacher forcing.

즉 실제 cross-utterance reference-conditioning objective가 직접 없음.

---

# 20. S1 full fine-tuning 문제 가설

현재 가장 중요한 가설:

1. full FT가 pretrained reference-routing을 덮어씀.
2. training objective와 actual inference condition mismatch.
3. 15 epoch가 너무 길 수 있음.
4. validation이 없어 degradation을 감지 못함.
5. DPO가 dataset-mode bias를 강화할 가능성.

아직 각각 확정 아님.

---

# 21. S1 research document 추가

commit:

- 536ee6de55247b93269ed3b15cc6db523737877f
  - initial S1 reference retention experiment doc

후속:

- 5824bc442de804fa7227e53bdcc1a0dcc9a5b44b
  - S1 + Controller 관계 및 LoRA 전략 개선

- b2b958e757e0d4788e6dfc3469ff95ad2f93ddef
  - comparison tool 사용법 문서화

---

# 22. LoRA 방향의 변화

초기:

~~~text
QKV + attention out LoRA
~~~

후속 분석:

reference routing 보존 목적이라면 Q/K까지 바꾸는 것이 불필요할 수 있음.

새 1순위:

~~~text
Q frozen
K frozen
V LoRA
attention out LoRA
~~~

이유:

~~~text
Q/K = attention routing
V/O = information content/mixing
~~~

pretrained reference routing을 최대한 보존하면서 adaptation capacity 제공.

---

# 23. Semantic coordinate freeze 아이디어

초기 LoRA 후보에서 freeze:

- audio semantic embedding
- text embedding
- BERT projection
- LayerNorm
- semantic readout

이유:

현재 Controller hidden_direction이 S1 hidden/readout geometry를 직접 활용.

S1 adaptation 중 coordinate가 과도하게 회전하면 Controller가 다시 불안정해질 수 있음.

---

# 24. Cross-utterance S1 objective 아이디어

Controller training은 이미:

~~~text
reference A + target B
~~~

조건.

따라서 S1 training도 이 구조에 가까워지는 것이 자연스럽다고 판단.

후보:

~~~text
50% current same-utterance
50% reference A → target B
~~~

reference A semantic 구간에는 loss mask.

target B semantic에만 prediction loss.

---

# 25. Hidden retention 아이디어

frozen pretrained S1과 adapted S1을 같은 reference-conditioned batch에 통과.

target hidden:

~~~text
h_base
h_tuned
~~~

cosine geometry를 약하게 보존.

목적:

- pretrained hidden coordinate 보존
- Controller-friendly representation 유지
- reference sensitivity 유지

logit KL은 그 다음 후보.

---

# 26. Joint training을 현재 보류한 이유

S1과 Controller를 동시에 train하면:

~~~text
Controller loss를 낮추기 위해
S1 hidden 자체가 emotion label에 맞게 변형
~~~

될 수 있음.

그러면:

- S1 generality
- reference sensitivity
- Controller quality

를 분리하기 어려움.

현재 기본:

~~~text
S1 adapt
→ freeze
→ probe
→ Controller train
~~~

---

# 27. S1 comparison tool 추가

commit:

- bcaa0f7857f5df743d956a0c0bf4c43c41882d10

tool:

- tools/compare_s1_controller_space.py

목표:

동일한 controller-like reference-conditioned batch에서 여러 S1 비교.

측정:

- hidden emotion probe
- baseline hidden cosine drift
- semantic probability JS
- top-k overlap
- parameter drift

parameter drift 분해:

- text embedding
- audio embedding
- BERT projection
- position alpha
- Q
- K
- V
- attention out
- FFN
- LayerNorm
- semantic readout

---

# 28. Kaggle storage 문제

과거 Kaggle output이 약 19.5GiB까지 차서 checkpoint save 실패.

증상:

- SoVITS full checkpoint 실패
- GPT checkpoint Errno 28

해결은 model architecture가 아니라 storage policy.

관련 commits:

- 60b9d3a988034622af674e804dbbf5896d1e4a13
  - SoVITS low-disk mode

- 879386b052863bcbd52f41ba26feae220ccdbd05
  - GPT low-disk mode

- 0bc3c870cafe4c38c8999a9c15f5ed6306486200
  - output root support

- 16f05034d447c6da30d417fa52d76b9a966c559a
  - emotion intermediates/output split

- 945bded04866c3246e387fe1f1cea59f559e16f4
  - Kaggle tmp/working path refactor

---

# 29. Kaggle path 원칙 확립

현재:

~~~text
/kaggle/tmp/GPT-SoVITS
/kaggle/tmp/huggingface
/kaggle/tmp/audio_input
/kaggle/tmp/emotion_opt
~~~

temporary.

~~~text
/kaggle/working/gpt_sovits_output
~~~

final.

이후 stale notebook에서 repo path를 /kaggle/working/GPT-SoVITS로 사용하는 문제가 있었음.

정상 repo path는 /kaggle/tmp/GPT-SoVITS.

---

# 30. UI error persistence

사용자가 Gradio error가 잠깐 뜨고 사라지는 문제 지적.

원칙:

~~~text
Error는 사용자가 X를 누르기 전까지 유지
~~~

wrapper에서 duration=None 사용.

---

# 31. UI polling flicker

2초 polling마다 UI 전체가 반짝이는 문제.

교훈:

- backend polling과 component redraw 분리
- state 변화가 없으면 unnecessary update 피함
- progress는 약간의 animation만
- 과도한 motion 금지

---

# 32. Default setting 조정 역사

현재 중요 default:

- BERT ON
- preprocessing confidence 0.35
- DPO ON
- emotion base steps 500
- validation interval 100
- validation-based early stop
- reasonable auto-extension
- Korean UI
- Finetune first
- V5dev
- fast 유지
- fast-v2 제거

이 default들도 절대 불변은 아니며 evidence에 따라 변경 가능.

---

# 33. 자동 데이터 pipeline 방향

목표:

~~~text
사용자 audio
→ preprocessing
→ emotion pair/label candidate
→ canonical sidecar
→ feature
→ GPT/SoVITS/Controller training
~~~

가능한 한 사용자가 중간 파일을 수동 편집하지 않도록.

다만 완전 무설정 automation보다 이해 가능한 단계형 UI 선호.

---

# 34. Emotion class 축소

초기 emotion2vec+ 9 class:

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

실제 데이터에서 surprised 과다 등의 문제.

unknown/other 제외.

낮은 sample class 제외.

현재 controller non-neutral core는:

~~~text
happy
angry
disgusted
surprised
~~~

neutral은 base.

fearful은 sample 부족으로 제외.

---

# 35. 감정 teacher 사용 철학

emotion2vec classifier output을 절대 정답으로 사용하지 않음.

future alignment에서는:

- anchor
- contrastive
- relative direction

같은 weak teacher 방식 선호.

---

# 36. Current architecture를 유지해야 하는 이유

현재 Controller는 실제 청감에서:

- tag difference
- multi-tag switch

를 만들고 있음.

따라서 S1 연구 중에는 Controller architecture를 다시 바꾸지 않는 것이 좋음.

변수 통제 목적.

---

# 37. 지금 가장 중요한 unresolved question

~~~text
pretrained s1v3가 현재 full15 S1보다
reference sensitivity를 실제로 더 잘 유지하는가?
~~~

이 질문을 먼저 runtime으로 확인해야 함.

그 뒤에만 LoRA.

---

# 38. 현재 next experiment

가장 먼저:

~~~text
pretrained
vs
current full15
~~~

tools/compare_s1_controller_space.py 실행.

그 다음 청감.

그 결과가 full FT degradation을 지지하면:

~~~text
full5 DPO off
full15 DPO off
full15 DPO on
~~~

ablation.

---

# 39. 이후 후보 순서

~~~text
B1 V+O LoRA
B2 QKV+O LoRA
B3 FFN-only 필요 시
C1 cross-reference objective
C2 weak hidden retention
C3 KL 필요 시
~~~

한 번에 모두 넣지 않음.

---

# 40. Codex handoff 문서 추가

commit:

- 4473e27a59f690dc826e275b5fafa29340c1e1e7
  - CODEX_KAGGLE_S1_CONTROLLER_RUNBOOK.md

목적:

Chrome/Kaggle 자동 실행에서:

~~~text
실험 선택
→ 실행
→ 로그 읽기
→ 다음 실험 결정
~~~

을 이어가도록.

---

# 41. Research methodology 문서 추가

commit:

- a8eb09a33b87e8ba6059de5d351e515f2ab3dda2
  - RESEARCH_METHODOLOGY_HANDOFF.md

목적:

결론뿐 아니라:

- research habits
- evidence hierarchy
- debugging style
- experiment design
- overengineering prevention
- decision rules

를 넘김.

---

# 42. 이 문서를 읽는 Codex가 하지 말아야 할 것

과거에 약했다고 확인된 path를 이유 없이 primary objective로 복귀시키지 말 것.

특히:

~~~text
semantic likelihood primary controller
~~~

를 다시 기본으로 만들지 말 것.

다시 검토하려면 새로운 evidence가 필요.

---

# 43. 현재 best mental model

전체 system:

~~~text
Reference/Text
      ↓
     S1
(reference-sensitive semantic generator)
      ↓
Emotion Controller
(bounded semantic steering)
      ↓
     S2
(acoustic decoding)
~~~

현재 연구 focus:

~~~text
S1 reference sensitivity 보존
+
Controller-friendly hidden geometry
~~~

S2는 fixed.

---

# 44. 현재 success definition

최종 목표:

~~~text
S1:
reference 잘 따라감
speaker 유지
발음 안정
emotion hidden signal 유지

Controller:
OOF generalization
tag difference
meaning alignment
multi-tag
reference robustness

System:
Kaggle 현실적
compact artifact
단순 pipeline
~~~

---

# 45. 마지막 주의

이 문서는 현재까지의 최선의 설명이다.

새 runtime evidence가 나오면 과거 결론보다 새 evidence가 우선.

하지만 과거 접근을 뒤집을 때는 반드시:

~~~text
무슨 새 증거가 나왔는가?
왜 기존 해석이 틀렸는가?
다음 최소 실험은 무엇인가?
~~~

를 기록한다.
