# Adversarial Emotion Label Refiner — Implementation Design

작성일: 2026-11-13

목적: GAN의 G/D 경쟁 아이디어를 감정 데이터셋 자동 분류에 적용하되, **작은 데이터에서 collapse·self-confirmation·label permutation이 발생하지 않도록 수정한 구현 설계**를 정의한다.

이 문서는 바로 구현 가능한 수준의 구조/입력/손실/학습 순서/CLI/checkpoint/output schema를 정의한다.

관련:
- `EMOTION_DATASET_LABELING_RESEARCH.md`
- `tools/auto_emotion_sidecar.py`
- `tools/emotion_dataset_builder.py`
- `tools/train_emotion_controller.py`

---

# 0. 최종 결론

원래 아이디어:

~~~text
G:
감정 label 생성

D:
그 label이 맞는지 판별

G/D 경쟁
~~~

를 그대로 구현하지 않는다.

그 방식은 현재 데이터에서 위험하다.

이유:
- G와 D가 같은 noisy pseudo-label을 기반으로 학습하면 오류를 서로 강화할 수 있음.
- 실제 angry sample이 없는데 teacher가 특정 style을 angry라고 붙이면 G/D가 그 잘못된 convention을 완벽하게 학습할 수 있음.
- label permutation을 막을 semantic grounding이 없음.
- 102개 수준에서 free-form neural G/D는 capacity가 지나치게 큼.

따라서 다음처럼 수정한다.

~~~text
Frozen emotion2vec+ Teacher
        │
        ├── 9-class probabilities
        └── 1024-d emotion embedding
                    │
        ┌───────────┴────────────┐
        ↓                        ↓
Bounded Refiner G          Match Critic C
(logit residual)        (embedding ↔ label)
        │                        │
        └────────── consensus ───┘
                    │
             refined label state
                    │
       accepted / ambiguous / OOD
~~~

그리고 진짜 adversarial(min-max) component는 **optional nuisance discriminator**로 분리한다.

~~~text
emotion latent
    │
    └── D_nuisance
        speaker/session/content를 맞히려고 함

Refiner/encoder:
D_nuisance가 못 맞히게 함
~~~

즉:

- `C_match` = cooperative compatibility critic
- `D_nuisance` = true adversary

이 구분이 중요하다.

---

# 1. 왜 direct GAN을 쓰지 않는가

일반 GAN:

~~~text
G(z) -> fake sample
D(x) -> real/fake
~~~

에서는 real data distribution이 외부 기준 역할을 한다.

현재 감정 label refinement에는:

~~~text
"진짜 angry label"
~~~

이라는 충분한 real distribution이 없다.

따라서:

~~~text
G가 label 생성
D가 맞다/틀리다 판단
~~~

을 같은 pseudo-label dataset에서 학습하면 D 자체가 ground truth를 모른다.

이 문제를 해결하기 위해 emotion2vec+ teacher를 **semantic anchor**로 유지한다.

Teacher가 완벽해서가 아니라:

~~~text
class name의 의미를 고정
~~~

하는 역할이다.

Refiner는 teacher에서 너무 멀리 갈 수 없다.

---

# 2. 세 component의 역할

## 2.1 Frozen Teacher T

현재:
`iic/emotion2vec_plus_large`

출력:

~~~text
p_T(y | x) : 9-class probability
z_T(x)     : 1024-d emotion embedding
~~~

emotion2vec+ large config의 embed_dim은 1024.

FunASR는 `extract_embedding=True`일 때:
- labels
- scores
- feats
를 함께 반환 가능.

Teacher는 학습하지 않는다.

역할:
- semantic class grounding
- strong prior
- OOD/unknown evidence
- fixed representation source

---

# 3. Refiner G — 자유로운 classifier가 아니라 bounded residual

G가 처음부터 새로운 7-class classifier가 되면 안 된다.

대신:

~~~text
teacher logits
+
작은 bounded correction
~~~

만 허용한다.

## 3.1 입력

G에는 1024-d embedding을 직접 주지 않는 것을 첫 버전으로 추천.

입력:

~~~text
9 teacher log probabilities
7 speaker-relative robust scores
1 entropy
1 raw top1-top2 margin
1 OOD mass
1 relative best-second gap
1 duration-normalized feature (optional)
~~~

약 19~21 dimensions.

이유:
- 102개 dataset에서 1024-d neural classifier는 overfit 위험.
- C_match가 embedding geometry를 담당하므로 evidence source를 분리.
- G와 C가 같은 input을 그대로 보는 것보다 independent evidence가 됨.

## 3.2 architecture

첫 버전:

~~~text
input_dim ~20
Linear(input_dim, 32)
SiLU
Dropout(0.10)
Linear(32, 16)
SiLU
Linear(16, K)
tanh
~~~

K = 현재 trainable emotion classes.

Parameter 수는 매우 작게 유지.

## 3.3 output

~~~text
delta = delta_max * tanh(raw_delta)

refined_logits =
teacher_known_logits
+ delta
~~~

초기:
`delta_max = 1.0`

G가 teacher logit을 무제한 뒤집지 못하게 한다.

`delta_max`는 ablation:
- 0.5
- 1.0
- 1.5

기본은 1.0.

## 3.4 initialization

마지막 layer weight/bias = 0.

즉 training 전:

~~~text
G output == teacher distribution
~~~

zero-effect initialization.

---

# 4. Match Critic C — label compatibility

C는:

~~~text
"이 audio emotion embedding과 이 emotion label이 잘 맞는가?"
~~~

를 score한다.

C는 real/fake binary classifier보다 **energy / ranking critic**으로 구현한다.

## 4.1 입력

emotion2vec embedding:

~~~text
z_T ∈ R^1024
~~~

하지만 직접 1024-d MLP를 쓰지 않는다.

### fixed PCA

dataset 전체 embedding에 label 없이 PCA.

~~~text
1024 -> d
d = min(32, N-1)
~~~

default:
`d=32`

PCA는 unsupervised라 train/val label leakage가 없음.

cross-fit strict mode에서는 PCA 자체도 train fold에서 fit할 수 있지만,
초기 diagnostic에서는 전체 embedding PCA를 허용할 수 있다.

최종 evaluation에서는 train-fold PCA를 권장.

## 4.2 Critic latent

~~~text
u = normalize(PCA(z))
~~~

optionally:

~~~text
u' = normalize(u + B(Au))
~~~

rank-4 residual projection.

처음 버전에서는 identity/prototype critic부터 시작 가능.

## 4.3 class prototypes

high-confidence seed만 사용.

~~~text
c_k =
normalize(mean(u_i for seed label k))
~~~

trainable version에서는 prototype을 Parameter로 만들되 seed centroid로 initialize.

## 4.4 compatibility score

~~~text
C(x, k) =
cosine(u', c_k) / temperature
~~~

default:
`temperature = 0.10`

score 자체는 probability가 아님.

---

# 5. Critic loss

positive:

~~~text
(audio_i, seed_label_i)
~~~

negative:

1. teacher top-2 competing class
2. closest wrong prototype
3. random wrong class

hard negative를 우선.

## 5.1 margin ranking

~~~text
L_rank =
relu(
    margin
    - C(x, y_pos)
    + C(x, y_neg)
)
~~~

default:
`margin = 0.20`

## 5.2 balanced sampling

각 emotion class가 같은 빈도로 critic update에 들어오게 한다.

rare class의 threshold를 낮춰 sample 수를 늘리지 않는다.

class sample < 3이면 critic prototype class에서 제외.

---

# 6. Refiner G loss

G는 critic을 "속여서 아무 label이나 인정받는" 방식으로 학습하지 않는다.

세 loss의 합으로 간다.

## 6.1 Seed supervised loss

high-confidence seed에서:

~~~text
L_seed =
weighted CE(q_G, y_seed)
~~~

teacher confidence / label quality에 따라 sample weight 가능.

초기에는:
- high seed = 1.0
- medium seed = 0.5
- ambiguous = 0.0

## 6.2 Teacher retention

모든 non-OOD sample:

~~~text
L_teacher =
KL(
  stopgrad(p_T_known)
  ||
  q_G
)
~~~

G가 teacher 의미에서 너무 멀리 움직이지 않게 함.

## 6.3 Critic consensus

critic distribution:

~~~text
p_C =
softmax(C(x, k) / T_C)
~~~

G와 critic이 완전히 다른 label distribution을 만들지 않게:

~~~text
L_match =
JS(q_G, stopgrad(p_C))
~~~

또는 symmetric KL.

첫 버전은 JS 추천.

## 6.4 total

~~~text
L_G =
1.0 * L_seed
+ 0.35 * L_teacher
+ 0.50 * L_match
~~~

이 숫자는 initial default이며 반드시 ablation 대상.

중요:
- teacher retention을 0으로 시작하지 않음.
- critic loss가 teacher보다 압도하지 않게 함.

---

# 7. temporal consistency loss — v0.2

chunk inference가 준비되면:

~~~text
q_full
q_chunk1
q_chunk2
...
~~~

평균 chunk distribution:

~~~text
q_chunk =
mean(q_chunk_i)
~~~

loss:

~~~text
L_temporal =
JS(q_full, q_chunk)
~~~

단:
chunk끼리 감정 transition이 강하면 consistency를 강제하지 않고
`temporal_conflict=true`로 abstain.

즉:

~~~text
consensus sample → consistency loss
mixed sample     → reject/ambiguous
~~~

---

# 8. true adversary D_nuisance — 나중에만

G/C가 효과가 확인된 후에만 추가.

## 8.1 목적

emotion latent에서:
- speaker
- session/recording condition
- text/content cluster

같은 nuisance를 제거.

현재 single-speaker dataset이면 speaker discriminator는 의미 없음.

따라서 현재 first candidate:
- session discriminator: explicit session_id가 있을 때
- source discriminator: source_id가 충분히 여러 group일 때

text nuisance는 더 조심해야 하므로 later.

## 8.2 Gradient Reversal Layer

~~~text
latent
  ↓
GRL(lambda)
  ↓
D_nuisance
  ↓
nuisance CE
~~~

D는 nuisance를 잘 맞히려고 하고,
latent encoder는 못 맞히게 함.

이 부분이 실제 adversarial/min-max.

## 8.3 현재 버전에서는 OFF

default:
`nuisance_adversary = false`

이유:
현재 핵심 문제는 noisy emotion labels이며,
nuisance removal을 동시에 넣으면 원인 분리가 어려움.

---

# 9. 핵심 안전장치 — cross-fitting

102개 dataset에서 가장 중요한 부분.

G/C를 전체 dataset에 fit한 뒤 같은 dataset을 refine하면
self-confirmation / memorization을 구분하기 어렵다.

따라서 **OOF cross-fitting**을 기본으로 한다.

## 9.1 fold

기존 project의:
`stratified_group_kfold_indices()`
개념을 재사용.

group:
- source_id
- text_group_id
- pair_group_id

연결된 group은 같은 fold.

## 9.2 각 fold

~~~text
train folds
  ↓
seed selection
  ↓
PCA fit
  ↓
critic fit
  ↓
refiner fit
  ↓
held-out fold refine
~~~

모든 sample은 자기 자신을 학습하지 않은 refiner/critic으로 OOF prediction을 받는다.

이 OOF output을 실제 sidecar v3 candidate 평가에 사용.

## 9.3 final model

v3가 검증된 후에만
전체 high-confidence seed로 final refiner checkpoint를 만들 수 있음.

---

# 10. Seed selection

G/C의 성능은 seed 품질에 매우 의존.

기존 v3 diagnostics를 사용.

## 10.1 high-confidence seed

초기 conservative rule 예:

~~~text
label_state == accepted
OOD mass < 0.25
normalized entropy < 0.70
raw top1 margin > 0.10
relative robust z > 1.0
relative z gap > 0.40
~~~

이 값들은 provisional.

human-reviewed set이 생기면 calibration.

## 10.2 class minimum

`min_seed_per_class = 3`

3 미만 class:
- G/C train class에서 제외
- raw evidence만 보존

---

# 11. output label policy — 자동 재라벨링을 너무 빨리 허용하지 않는다

첫 버전에서는 가장 중요한 안전 규칙.

## 11.1 accepted_core

조건:

~~~text
teacher v3 accepted label
==
G top1
==
C top1
~~~

→ highest confidence controller target.

## 11.2 teacher_conflict

teacher는 accepted인데:

~~~text
G/C가 teacher와 disagree
~~~

→ 자동으로 label 변경하지 않음.

state:
`conflict`

→ controller target에서 제외 / human review priority.

## 11.3 rescued_candidate

teacher는 ambiguous인데:

~~~text
G top1 == C top1
AND strong G margin
AND strong critic margin
AND low OOD
~~~

이면:

`rescued_candidate`

첫 버전에서는 **controller에 자동 투입하지 않음**.

diagnostic/review candidate로 저장.

v3가 실제 precision 개선을 증명한 후에만 medium-quality training sample로 승격.

## 11.4 OOD

teacher OOD이면 G/C가 강해도:
default reject.

semantic anchor 바깥 sample을 neural refiner가 억지로 class에 넣지 않음.

---

# 12. 왜 이 policy가 중요한가

이렇게 하면 G/C가 처음부터 pseudo-label을 마음대로 갈아엎지 않는다.

v0의 역할:

~~~text
teacher label을 검증하고
잘못된/애매한 sample을 찾는 것
~~~

v1 이후:

~~~text
검증된 성능을 바탕으로
ambiguous sample 일부를 rescue
~~~

순서.

---

# 13. checkpoint schema

추천:

`emotion_label_refiner.pt`

~~~python
{
    "format_version": 1,
    "model": {
        "refiner_state_dict": ...,
        "critic_state_dict": ...,
        "pca_mean": ...,
        "pca_components": ...,
        "class_prototypes": ...
    },
    "config": {
        "emotions": [...],
        "teacher_model": "...",
        "embedding_dim": 1024,
        "pca_dim": 32,
        "delta_max": 1.0,
        "critic_temperature": 0.10,
        "ranking_margin": 0.20,
        "loss_weights": {...}
    },
    "provenance": {
        "sidecar_sha256": "...",
        "prediction_cache_sha256": "...",
        "teacher_model": "...",
        "dataset_id": "..."
    },
    "metrics": {
        "oof_seed_accuracy": ...,
        "oof_macro_accuracy": ...,
        "oof_macro_auc": ...,
        "agreement_rate": ...,
        "conflict_rate": ...,
        "rescue_candidates": ...
    }
}
~~~

---

# 14. sidecar additive schema

기존 field를 유지.

추가:

~~~json
{
  "emotion_refiner_label": "surprised",
  "emotion_refiner_confidence": 0.73,
  "emotion_refiner_margin": 0.28,

  "emotion_critic_label": "surprised",
  "emotion_critic_score": 3.2,
  "emotion_critic_margin": 0.42,

  "emotion_teacher_refiner_agreement": true,
  "emotion_refiner_critic_agreement": true,

  "emotion_label_state_v3": "accepted_core",
  "emotion_label_quality_v3": "high",

  "emotion_refiner_oof": true,
  "auto_emotion_refiner": "critic_guided_residual_v1"
}
~~~

기존 Controller는 새 field를 무시.

새 labeling mode만 `emotion_label_state_v3`를 읽게 한다.

---

# 15. prediction cache / embedding cache

현재 prediction cache는 raw scores를 이미 저장.

embedding은 별도 저장 추천.

`emotion_embeddings.pt`

~~~python
{
    cache_key: torch.float16[1024]
}
~~~

102 samples 기준 매우 작음.

emotion2vec large:
`extract_embedding=True`로 scores + feats를 같이 받을 수 있으므로
같은 inference에서 둘을 저장 가능.

기존 cache에 embedding이 없으면:
- 필요한 sample만 re-infer
- raw score cache는 그대로 재사용 가능

---

# 16. 구현 파일 구조

파편화를 피하기 위해 새 script 하나만 추가.

`tools/emotion_label_refiner.py`

subcommands:

~~~text
extract
crossfit
train
apply
report
~~~

또는 first implementation에서는:
`crossfit` 하나가 extract/cache까지 orchestration.

추천 CLI:

~~~bash
python tools/emotion_label_refiner.py crossfit \
  --list training.list \
  --sidecar emotion_sidecar.jsonl \
  --prediction-cache emotion_sidecar.jsonl.predictions.jsonl \
  --embedding-cache emotion_embeddings.pt \
  --model iic/emotion2vec_plus_large \
  --output emotion_sidecar_v3.jsonl \
  --report emotion_refiner_report.json \
  --device cuda:0
~~~

---

# 17. Python classes

한 파일 안에서:

~~~python
class ResidualEmotionRefiner(nn.Module):
    ...

class PrototypeMatchCritic(nn.Module):
    ...

@dataclass
class RefinerConfig:
    ...

def build_refiner_features(...):
    ...

def fit_pca(...):
    ...

def select_seed_samples(...):
    ...

def train_fold(...):
    ...

def crossfit_refiner(...):
    ...

def apply_label_policy(...):
    ...
~~~

별도 helper file을 만들지 않는다.

---

# 18. ResidualEmotionRefiner pseudo-code

~~~python
class ResidualEmotionRefiner(nn.Module):
    def __init__(self, input_dim, class_count, delta_max=1.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(32, 16),
            nn.SiLU(),
            nn.Linear(16, class_count),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.delta_max = float(delta_max)

    def forward(self, features, teacher_logits):
        delta = self.delta_max * torch.tanh(self.net(features))
        return teacher_logits + delta
~~~

---

# 19. PrototypeMatchCritic pseudo-code

~~~python
class PrototypeMatchCritic(nn.Module):
    def __init__(self, pca_dim, class_count, rank=4, temperature=0.10):
        super().__init__()
        self.down = nn.Linear(pca_dim, rank, bias=False)
        self.up = nn.Linear(rank, pca_dim, bias=False)
        self.prototypes = nn.Parameter(
            torch.empty(class_count, pca_dim)
        )
        self.temperature = float(temperature)

        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        h = x + self.up(self.down(x))
        h = F.normalize(h, dim=-1)
        proto = F.normalize(self.prototypes, dim=-1)
        return h @ proto.T / self.temperature
~~~

prototype Parameter는 seed centroid로 initialize.

---

# 20. 학습 schedule

작은 dataset 기준.

## Critic warmup

~~~text
100 steps
lr = 3e-3
weight_decay = 1e-2
~~~

## Refiner + critic alternating

~~~text
up to 400 steps
validation every 50
patience 3
restore best
~~~

각 iteration:

~~~text
1 critic step
1 refiner step
~~~

하지만 critic/refiner loss를 동시에 backward하지 않는다.

## optimizer

Refiner:
`AdamW(lr=2e-3, weight_decay=1e-2)`

Critic:
`AdamW(lr=3e-3, weight_decay=1e-2)`

gradient clip:
`1.0`

mixed precision 불필요할 정도로 작은 모델.

---

# 21. validation

fold train 내부에서도 seed split을 하나 더 크게 만들 필요는 없음.

OOF fold 자체가 primary validation.

training early-stop은 train-fold 내부 deterministic holdout:
- grouped
- min per class 보존

또는 작은 데이터면:
- fixed step
- weight norm / loss sanity
- OOF result로 최종 판단

과도한 nested CV는 비용 대비 의미가 작을 수 있음.

---

# 22. 성공 metric

## Labeler quality

- teacher-G-C triple agreement
- conflict rate
- rescue candidate rate
- OOD reject rate
- class counts
- entropy distribution

## OOF

seed pseudo-label 기준:
- macro accuracy
- macro AUC
- CE

하지만 이것만 성공 기준 아님.

## Human-reviewed subset이 생기면

가장 중요:
- macro precision
- per-class precision
- coverage
- abstention quality

## Downstream

Controller:
- hidden OOF macro accuracy/AUC
- tag semantic alignment
- multi-tag
- speaker/CER

---

# 23. collapse detector

다음이면 FAIL:

~~~text
한 class > 70% output
mean entropy 급락 + accuracy 개선 없음
teacher agreement 급락
critic score span 폭발
prototype cosine collapse
rescued_candidate가 core보다 많음
OOD sample을 대량 known class로 rescue
~~~

---

# 24. critic calibration

C score는 probability가 아니다.

따라서:

~~~text
critic_score
critic_margin
~~~

만 저장.

softmax를 만들 수는 있지만
`critic_confidence = correctness probability`라고 부르지 않는다.

---

# 25. human 없이 얼마나 갈 수 있는가

이 구조는 사람 검수 없이도:

- teacher label 검증
- conflict filtering
- embedding consistency
- OOD reject
- ambiguity detection

을 개선할 수 있다.

왜냐하면 semantic class name은 emotion2vec teacher가 제공하기 때문.

즉 label permutation 문제는 teacher anchor가 막는다.

하지만:

~~~text
teacher 자체가 target-domain에서 systematic하게 잘못된 class 의미를 가진 경우
~~~

는 완전히 해결할 수 없다.

따라서 사람 review는 optional calibration/gold evaluation으로 남는다.

목표는:
`human-free perfection`이 아니라
`human-minimal robust refinement`.

---

# 26. direct GAN과 비교

## 버린 형태

~~~text
G free classifier
vs
D true/fake emotion pair
~~~

문제:
- real emotion labels 부족
- self-confirmation
- instability
- semantic permutation

## 채택 형태

~~~text
Frozen Teacher
+
Bounded G
+
Embedding Critic
+
Abstention
+
OOF cross-fit
~~~

장점:
- semantic grounding
- zero-effect init
- 작은 parameter
- collapse 방지
- 기존 sidecar 호환
- 사람이 없어도 동작
- 나중에 nuisance adversary 추가 가능

---

# 27. 구현 단계

## Phase A — prerequisite

`auto_emotion_sidecar.py`에:
- raw scores sidecar 저장
- entropy
- margin
- OOD mass
- robust v3 stats
추가.

behavior는 v2 그대로.

## Phase B — embedding cache

emotion2vec:
`extract_embedding=True`

`emotion_embeddings.pt` 생성.

## Phase C — critic-only diagnostic

Refiner 없이:
- prototype critic
- OOF agreement
- conflict detection

먼저 확인.

이 단계가 중요하다.

Critic 자체가 teacher label과 아무 관계를 못 배우면 G를 붙이지 않는다.

## Phase D — bounded Refiner

Critic이 informative일 때만 G 추가.

## Phase E — rescued candidates

human-reviewed precision이 확인된 뒤에만 ambiguous rescue를 Controller training에 사용.

## Phase F — nuisance adversary

필요할 때만.

---

# 28. 첫 구현에서 의도적으로 빼는 것

- waveform generator
- waveform GAN discriminator
- text sentiment model
- SenseVoice ensemble
- chunk consistency
- nuisance adversary
- joint Controller training
- iterative relabel loop
- soft-label Controller

첫 구현은:
`teacher + embedding critic + bounded refiner + cross-fit`
만.

---

# 29. 첫 실험 matrix

~~~text
A0 current v2
A1 v3 uncertainty only
A2 v3 + critic-only filter
A3 v3 + critic + bounded refiner
~~~

Controller는 각 accepted label set으로 새로 train.

비교:
- usable class counts
- conflict
- Controller OOF
- listening semantic alignment

---

# 30. 지금 기준 최종 recommendation

가장 구현 가치가 높은 형태:

~~~text
emotion2vec+ large (frozen)
        ↓
9-class scores + 1024-d embedding
        ↓
speaker-relative robust calibration
        ↓
OOD / ambiguity gate
        ↓
Prototype Match Critic
        ↓
Bounded Residual Refiner
        ↓
OOF cross-fit
        ↓
accepted_core / conflict / rescued_candidate
        ↓
Controller에는 accepted_core 우선
~~~

이 구조는 GAN 아이디어의 핵심인
`critic을 통해 generator/refiner가 자기 판단을 검증받는다`
는 개념을 유지하지만,
감정 라벨링 문제에 맞게 안정성과 semantic grounding을 추가한 버전이다.


---

# 31. 2026-11-13 구현 진행 상태

아래 항목은 더 이상 설계만 있는 상태가 아니다.

## 구현 완료 — Phase A diagnostics

파일:
`tools/auto_emotion_sidecar.py`

commit:
`06d00d25d50fcf89aef4b761e1a0adf954b363f1`

추가된 additive diagnostics:
- `emotion_raw_scores`
- `emotion_entropy`
- `emotion_top1_margin`
- `emotion_ood_mass`
- `emotion_raw_top1`
- `emotion_raw_top2`
- `emotion_relative_gap`

중요:
기존 `speaker_relative_clr_median_v2`의 accepted label 결정식은 변경하지 않았다.

즉 이 commit은 behavior-changing labeler가 아니라 diagnostics-only다.

## 구현 완료 — robust MAD diagnostics

파일:
`tools/auto_emotion_sidecar.py`

commit:
`1a2f2efb95e98c53518fb4e0f7684848440f58c8`

추가:
- speaker/class별 median absolute deviation 기반 robust scale
- `emotion_relative_z_scores`
- `emotion_relative_z`
- `emotion_relative_z_gap`
- `emotion_relative_z_top1`

현재 robust z 값도 diagnostics-only다.
v2 accepted label decision에는 사용하지 않는다.

scale floor:
`ROBUST_SCALE_FLOOR = 0.10`

이 값은 v3 threshold calibration 전까지 safety floor이며 최종 calibrated threshold가 아니다.

## 구현 완료 — review set selector

파일:
`tools/build_emotion_review_set.py`

commit:
`a18f9c4d3fe81711cabed18c63a0d96b1752ce11`

목적:
전체 dataset을 사람이 듣게 하지 않고 정보량 높은 약 20~30개만 선택.

선정 근거:
- class anchor
- class boundary
- rare class
- high OOD
- raw top1 vs speaker-relative label disagreement
- neutral anchor
- suspicious neutral boundary

이 도구는 label을 변경하지 않는다.

예:

~~~bash
python tools/build_emotion_review_set.py \
  --sidecar /kaggle/tmp/emotion_opt/emotion_dataset_01/emotion_sidecar.jsonl \
  --output /kaggle/tmp/emotion_opt/emotion_dataset_01/emotion_review.jsonl \
  --limit 30
~~~

출력:
- `emotion_review.jsonl`
- `emotion_review.jsonl.report.json`

review JSONL에는 사람이 채울 수 있도록:
- `human_emotion`
- `human_confidence`
- `human_secondary_emotion`
- `human_notes`
field가 포함된다.

## 구현 완료 — Phase C critic-only OOF

파일:
`tools/emotion_label_refiner.py`

initial commit:
`0c699d052ec6e60d0a978f7d31f6295d9eadca9a`

open-set seed fix:
`6a177746e701e3afd5f119a730d00a134cde9a32`

현재 구현은 의도적으로 **Critic-only**다.

Bounded Refiner G는 아직 구현하지 않았다.

이유:
Critic이 OOF에서 informative하지 않으면 G를 붙일 근거가 없기 때문.

현재 Critic:
- emotion2vec+ embedding cache
- fold train seed로 PCA
- small rank residual metric head
- class prototype initialization
- balanced CE
- hardest-negative margin ranking
- leakage-aware grouped OOF

seed에서 raw top1이:
- `other`
- `unknown`

이면 무조건 제외한다.

Critic은 기존 `emotion_intended`를 절대 덮어쓰지 않는다.

추가 output:
- `emotion_critic_label`
- `emotion_critic_max_probability`
- `emotion_critic_margin`
- `emotion_critic_oof`
- `emotion_label_state_v3`
- `emotion_label_quality_v3`
- `emotion_v3_candidate`
- `auto_emotion_refiner`

현재 state:
- `accepted_core` — high-confidence seed + OOF Critic agreement
- `critic_confirmed_candidate` — seed 기준을 못 넘긴 기존 v2 label에 final Critic이 동의
- `conflict`
- `neutral_candidate`
- `ambiguous`
- `ood`
- `failed`

`critic_confirmed_candidate`는 자동 Controller target으로 승격하지 않는다.

Critic probability는 calibrated correctness probability가 아니다.
따라서 field 이름도 confidence가 아니라 `max_probability`로 둔다.

## 구현 완료 — canonical metadata propagation

파일:
`tools/emotion_dataset_builder.py`

commit:
`14ea6d729b1cb1014ea02c9abb7bbb8de7763d16`

sidecar의 새 diagnostics / critic metadata가 canonical `metadata.jsonl`에서 사라지지 않도록 전달한다.

보존되는 주요 추가 field:
- raw score / entropy / margin / OOD
- relative gap / robust z
- v3 label state
- critic label/margin/OOF
- refiner provenance

---

# 32. Codex/Kaggle에서 다음으로 실행할 실제 명령

먼저 최신 sidecar를 다시 생성한다.
기존 v2 label logic은 그대로이지만 diagnostics field가 추가된다.

WebUI의 "음성에서 상대 감정 라벨 생성" 버튼을 다시 실행해도 된다.

CLI 예:

~~~bash
cd /kaggle/tmp/GPT-SoVITS

python tools/auto_emotion_sidecar.py \
  --list /kaggle/input/datasets/pymood1/endfield/training.list \
  --wav-root /kaggle/input/datasets/pymood1/endfield/datas \
  --output /kaggle/tmp/emotion_opt/emotion_dataset_01/emotion_sidecar.jsonl \
  --model iic/emotion2vec_plus_large \
  --device cuda:0 \
  --min-confidence 0.35
~~~

기존 prediction cache가 같은 output 옆에 있다면 raw SER inference는 재사용 가능하다.

그 다음 review selector:

~~~bash
python tools/build_emotion_review_set.py \
  --sidecar /kaggle/tmp/emotion_opt/emotion_dataset_01/emotion_sidecar.jsonl \
  --output /kaggle/tmp/emotion_opt/emotion_dataset_01/emotion_review.jsonl \
  --limit 30
~~~

그 다음 Critic:

~~~bash
python tools/emotion_label_refiner.py \
  --list /kaggle/input/datasets/pymood1/endfield/training.list \
  --wav-root /kaggle/input/datasets/pymood1/endfield/datas \
  --sidecar /kaggle/tmp/emotion_opt/emotion_dataset_01/emotion_sidecar.jsonl \
  --prediction-cache /kaggle/tmp/emotion_opt/emotion_dataset_01/emotion_sidecar.jsonl.predictions.jsonl \
  --embedding-cache /kaggle/tmp/emotion_opt/emotion_dataset_01/emotion_embeddings.pt \
  --output /kaggle/tmp/emotion_opt/emotion_dataset_01/emotion_sidecar_v3_critic.jsonl \
  --artifact /kaggle/tmp/emotion_opt/emotion_dataset_01/emotion_critic.pt \
  --model iic/emotion2vec_plus_large \
  --device cuda:0 \
  --emotions happy,angry,disgusted,surprised
~~~

emotion2vec+ official behavior상 `extract_embedding=True`이면 classification scores와 embedding feature를 함께 반환한다.
Critic script는 embedding만 cache한다.

---

# 33. Critic 통과 기준

첫 runtime에서 **OOF 수치를 먼저 본다**.

4 active classes라면 chance accuracy 약 25%.

Refiner 구현 진행 조건의 초기 gate:

~~~text
OOF macro accuracy >= 35%
AND
OOF macro AUC >= 0.60
~~~

이는 현재 Controller generalization gate와 비슷한 보수적 시작점이다.

추가 확인:
- 한 class로 collapse하지 않았는가
- `conflict`가 전체 accepted seed 대부분을 차지하지 않는가
- angry pseudo-label만 유독 critic에서 붕괴하지 않는가
- disgusted처럼 작은 class의 fold가 유지되는가

특히 current dataset에서 실제 genuine angry가 없다는 사용자 관찰이 있으므로:

~~~text
angry critic performance가 낮음
~~~

은 Critic 실패라고 단정하지 않는다.

오히려:
`angry pseudo-label cluster가 embedding space에서 일관적인가?`
를 보여주는 진단이 될 수 있다.

---

# 34. Bounded Refiner G 구현 조건

다음 중 하나면 G 구현을 보류:

- Critic OOF near chance
- macro AUC < 0.60
- seed class가 2개 이하로 축소
- OOD seed가 많이 섞임
- class prototype collapse

Critic이 informative일 때만:
`ResidualEmotionRefiner`
를 구현한다.

첫 G는:
- teacher logits에 bounded residual만 추가
- zero-effect init
- seed CE
- teacher KL retention
- critic JS consensus

를 사용.

Critic이 실패하면 G를 더 복잡하게 만들지 말고:
- seed selection
- embedding quality
- pseudo-label quality
를 먼저 조사한다.

---

# 35. Runtime verification 상태

현재까지:
- 새 Python 코드의 구조/문법은 작성 단계에서 정적 검토함.
- GitHub repo에 commit 완료.
- 실제 Kaggle emotion2vec embedding extraction / OOF critic runtime은 **아직 실행하지 않음**.

따라서 Codex는 첫 실행 후 결과를 이 문서와
`PROJECT_RESEARCH_HISTORY.md`
에 기록할 것.


## 2026-11-13 추가 안전 수정

commit:
`910cf2e47e0f2abf9ce7ad001e547da41ac56853`

초기 Critic 구현을 다시 읽는 과정에서 다음 경로를 발견했다.

~~~text
v2에서는 accepted
하지만 high-confidence seed 기준은 실패
+
final Critic은 v2 label에 동의
→ accepted_core
~~~

이 경로는 너무 공격적이었다.

수정 후:

~~~text
high-confidence seed + OOF agreement
→ accepted_core

non-seed v2 label + final Critic agreement
→ critic_confirmed_candidate
~~~

즉 high-quality core와 "검토할 가치가 있는 후보"를 분리한다.

---

# 36. 추가 구현 — human review evaluator

파일:
`tools/evaluate_emotion_review.py`

commit:
`b59f931a1c3b48aa9fe560987166280eed693a83`

사람이 review JSONL에서 일부 sample에만 `human_emotion`을 채워도 다음을 동일 subset에서 비교한다.

- raw emotion2vec
- speaker-relative v2
- Critic
- conservative v3 selective

주요 metric:
- coverage
- selective accuracy
- macro precision
- macro recall
- macro F1
- per-class confusion

v3 selective는 abstain이 가능하므로 accuracy 단독이 아니라 coverage와 함께 본다.

사람 label도 절대 ground truth로 가정하지 않는다.
기본적으로 human confidence가 low인 row는 평가에서 제외한다.

예:

~~~bash
python tools/evaluate_emotion_review.py \
  --review /kaggle/tmp/emotion_opt/emotion_dataset_01/emotion_review.jsonl \
  --sidecar /kaggle/tmp/emotion_opt/emotion_dataset_01/emotion_sidecar_v3_critic.jsonl \
  --output /kaggle/tmp/emotion_opt/emotion_dataset_01/emotion_review_eval.json
~~~

---

# 37. 추가 구현 — synthetic smoke test

파일:
`tools/emotion_labeling_selftest.py`

commit:
`0a6263a2fc12690914880d0c6a3755f88c5bab6d`

실제 audio/FunASR model 없이 다음을 검사한다.

- uncertainty diagnostics
- review set selector
- separable synthetic embedding에서 grouped OOF Critic
- review evaluator

Kaggle에서 가장 먼저 실행:

~~~bash
cd /kaggle/tmp/GPT-SoVITS
python tools/emotion_labeling_selftest.py
~~~

기대:
모든 항목 `[PASS]`.

개발 환경에서 Critic core를 exact source 기준으로 synthetic separable data에 별도 실행했을 때:
- OOF accuracy = 1.0
- OOF macro accuracy = 1.0
- OOF macro AUC = 1.0

review selector/evaluator의 core function도 synthetic input에서 기대 결과를 확인했다.

단:
현재 환경에서는 GitHub DNS가 차단되어 **repo 전체를 clone한 상태의 실제 self-test 파일 실행은 못 했다**.
따라서 Kaggle에서 위 self-test를 첫 runtime gate로 실행한다.
