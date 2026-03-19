# v21_expert 정리

## 1. 모델 개요

v21은 **단일 global 모델의 한계**를 보완하기 위해,  
**global blend(V17-V19) + transfer expert + frozen expert** 구조로 설계한 모델이다.

기존 실험에서 확인한 핵심 문제는 다음과 같았다.

- 전체 OOF AUC는 이미 0.7408 수준에서 포화
- 하지만 subgroup 기준으로 보면
  - **transfer AUC ≈ 0.675**
  - **frozen transfer AUC ≈ 0.620**
  로 상대적으로 낮음
- 즉, 전체 성능보다 **이식군/동결이식군의 구분 성능 개선**이 필요했음

이를 해결하기 위해 v21에서는:

1. **global base**로 기존 최고 수준의 블렌드 결과를 사용하고  
2. **transfer-only expert**와 **frozen-transfer-only expert**를 별도로 학습한 뒤  
3. subgroup별로 다른 가중치를 적용해 최종 예측값을 생성했다.

---

## 2. 전처리

### 2-1. 결측치 처리
- **문자형(object) 컬럼**: `Unknown`으로 통일
- **수치형 컬럼**: 수치 변환 후 결측은 0으로 처리

#### 이유
- CatBoost는 범주형을 직접 처리할 수 있으므로, 결측도 하나의 상태로 유지하는 것이 유리함
- 의료 데이터에서는 `미상`, `알 수 없음`, `없음`이 실제로 다른 의미를 가질 수 있어 문자열로 남기는 것이 중요함

---

### 2-2. 범주형 처리
- CatBoost에 **원본 범주형 컬럼 20개**를 직접 입력

#### 사용된 categorical columns
- 시술 시기 코드
- 시술 당시 나이
- 시술 유형
- 특정 시술 유형
- 배란 유도 유형
- 배아 생성 주요 이유
- 총 시술 횟수
- 클리닉 내 총 시술 횟수
- IVF 시술 횟수
- DI 시술 횟수
- 총 임신 횟수
- IVF 임신 횟수
- DI 임신 횟수
- 총 출산 횟수
- IVF 출산 횟수
- DI 출산 횟수
- 난자 출처
- 정자 출처
- 난자 기증자 나이
- 정자 기증자 나이

#### 이유
- CatBoost는 high-cardinality categorical feature를 잘 처리함
- 의료 데이터 특성상 범주 자체가 의미를 갖는 경우가 많아, 무리하게 one-hot이나 label encoding을 하지 않고 원본을 유지함

---

### 2-3. 숫자형 변환

#### (1) 횟수형 문자열 → 숫자형
다음 컬럼에 대해 `_num` 파생 컬럼 생성
- 총 시술 횟수
- 클리닉 내 총 시술 횟수
- IVF 시술 횟수
- DI 시술 횟수
- 총 임신 횟수
- IVF 임신 횟수
- DI 임신 횟수
- 총 출산 횟수
- IVF 출산 횟수
- DI 출산 횟수

예:
- `"3회"` → `3`
- `"0회"` → `0`

#### 이유
- 시술 이력 기반 비율/실패 횟수/상호작용 피처 계산에 사용하기 위해 숫자화가 필요했음

---

#### (2) 나이 범주 → 숫자형
다음 컬럼에 대해 `_num` 파생 컬럼 생성
- 시술 당시 나이
- 난자 기증자 나이
- 정자 기증자 나이

예:
- `"만18-34세"` → `(18 + 34) / 2 = 26`
- `"만35-37세"` → `36`

#### 이유
- 연령은 IVF 성공률에 매우 큰 영향을 미치므로, 범주형 그대로만 쓰는 것보다 숫자형으로 변환해 연속적인 효과를 반영하고자 함

---

### 2-4. boolean/flag 정규화
아래와 같은 다양한 표현을 **0/1로 통일**
- `Y/N`
- `예/아니오`
- `있음/없음`
- `True/False`
- `"1"/"0"`

예:
- `배란 자극 여부`
- `단일 배아 이식 여부`
- `PGS 시술 여부`
- `동결 배아 사용 여부`
- `불임 원인 - 정자 운동성`
- `불임 원인 - 자궁내막증`
등

#### 이유
- 동일 의미의 값이 문자열 표현 차이로 분리되는 것을 방지
- 불임 원인 score, donor flag, frozen flag 계산의 안정성 확보

---

### 2-5. 상수 컬럼 제거
v21에서 제거된 상수 컬럼:
- `불임 원인 - 여성 요인`
- `난자 채취 경과일`

#### 이유
- train/test 모두에서 변화가 거의 없는 컬럼은 모델에 정보 기여가 없고, 노이즈만 추가할 수 있기 때문

---

## 3. 피처 엔지니어링

v21의 입력 피처는 총 **164개**이며, 다음 세 축으로 구성되었다.

1. **기존 팀원 의료 파생 피처 29개**
2. **v17 frozen 전용 피처 8개**
3. **pruned domain 피처 21개**

---

### 3-1. 팀원 의료 파생 피처 (29개)

이 그룹은 **시술 과정의 효율성과 과거 이력**을 수치화하기 위해 생성되었다.

#### A. 배아/난자 처리 효율 관련

##### 1) 배아생성효율

\[
\text{배아생성효율} = \frac{\text{총 생성 배아 수}}{\text{수집된 신선 난자 수}}
\]

- 난자 채취 대비 실제 배아 생성 효율
- 난자 수가 많아도 배아로 전환되지 않으면 임신 가능성이 낮아질 수 있음

##### 2) ICSI수정효율

\[
\text{ICSI수정효율} = \frac{\text{미세주입에서 생성된 배아 수}}{\text{미세주입된 난자 수}}
\]

- ICSI 수행 시 수정 성공 효율 반영

##### 3) 배아이식비율

\[
\text{배아이식비율} = \frac{\text{이식된 배아 수}}{\text{총 생성 배아 수}}
\]

##### 4) 배아저장비율

\[
\text{배아저장비율} = \frac{\text{저장된 배아 수}}{\text{총 생성 배아 수}}
\]

##### 5) 난자활용률

\[
\text{난자활용률} = \frac{\text{미세주입된 난자 수}}{\text{수집된 신선 난자 수}}
\]

##### 6) 난자대비이식배아수

\[
\text{난자대비이식배아수} = \frac{\text{이식된 배아 수}}{\text{수집된 신선 난자 수}}
\]

#### B. 과거 시술 이력 기반 피처

##### 7) 전체임신률

\[
\text{전체임신률} = \frac{\text{총 임신 횟수\_num}}{\text{총 시술 횟수\_num}}
\]

##### 8) IVF임신률

\[
\text{IVF임신률} = \frac{\text{IVF 임신 횟수\_num}}{\text{IVF 시술 횟수\_num}}
\]

##### 9) DI임신률

\[
\text{DI임신률} = \frac{\text{DI 임신 횟수\_num}}{\text{DI 시술 횟수\_num}}
\]

##### 10) 임신유지율

\[
\text{임신유지율} = \frac{\text{총 출산 횟수\_num}}{\text{총 임신 횟수\_num}}
\]

##### 11) IVF임신유지율

\[
\text{IVF임신유지율} = \frac{\text{IVF 출산 횟수\_num}}{\text{IVF 임신 횟수\_num}}
\]

##### 12) 총실패횟수

\[
\text{총실패횟수} = \max(\text{총 시술 횟수\_num} - \text{총 임신 횟수\_num}, 0)
\]

##### 13) IVF실패횟수

\[
\text{IVF실패횟수} = \max(\text{IVF 시술 횟수\_num} - \text{IVF 임신 횟수\_num}, 0)
\]

##### 14) 반복IVF실패_여부

\[
\text{반복IVF실패\_여부} = \mathbb{1}(\text{IVF실패횟수} \ge 3)
\]

##### 15) 클리닉집중도

\[
\text{클리닉집중도} = \frac{\text{클리닉 내 총 시술 횟수\_num}}{\text{총 시술 횟수\_num}}
\]

##### 16) IVF시술비율

\[
\text{IVF시술비율} = \frac{\text{IVF 시술 횟수\_num}}{\text{총 시술 횟수\_num}}
\]

##### 17) 임신경험있음

\[
\mathbb{1}(\text{총 임신 횟수\_num} > 0)
\]

##### 18) 출산경험있음

\[
\mathbb{1}(\text{총 출산 횟수\_num} > 0)
\]

#### C. 나이 기반 피처

##### 19) 나이_제곱

\[
\text{나이\_제곱} = (\text{시술 당시 나이\_num})^2
\]

##### 20) 나이_임상구간
- 0: <35
- 1: 35–39
- 2: 40–44
- 3: 45+

##### 21) 고령_여부

\[
\mathbb{1}(\text{나이} \ge 35)
\]

##### 22) 초고령_여부

\[
\mathbb{1}(\text{나이} \ge 40)
\]

##### 23) 극고령_여부

\[
\mathbb{1}(\text{나이} \ge 42)
\]

#### D. 나이 × 이력 상호작용

##### 24) 나이X총시술

\[
\text{나이X총시술} = \text{나이} \times \text{총 시술 횟수\_num}
\]

##### 25) 나이XIVF실패

\[
\text{나이XIVF실패} = \text{나이} \times \text{IVF실패횟수}
\]

##### 26) 나이XIVF임신률

\[
\text{나이XIVF임신률} = \text{나이} \times \text{IVF임신률}
\]

##### 27) 초고령X반복실패

\[
\text{초고령X반복실패} = \text{초고령\_여부} \times \text{반복IVF실패\_여부}
\]

##### 28) 복합위험도점수

\[
\text{복합위험도점수} =
\text{고령\_여부} + \text{초고령\_여부} + \text{반복IVF실패\_여부} + (1-\text{임신경험있음})
\]

##### 29) 이식배아수_구간
- 0: 이식 없음
- 1: 1개
- 2: 2개
- 3: 3개 이상

---

### 3-2. v17 공통 피처 + frozen 전용 피처

이 그룹은 **이식 단계와 동결/해동 전략**을 반영하기 위한 피처다.

#### A. 공통 시술/이식 피처

##### 실제이식여부

\[
\mathbb{1}(\text{이식된 배아 수} > 0)
\]

##### total_embryo_ratio

\[
\text{total\_embryo\_ratio} =
\frac{\text{이식된 배아 수} + \text{저장된 배아 수} + \text{해동된 배아 수}}
{\text{총 생성 배아 수} + 1}
\]

##### 수정_성공률

\[
\text{수정\_성공률} = \frac{\text{총 생성 배아 수}}{\text{혼합된 난자 수}}
\]

##### 배아_이용률

\[
\text{배아\_이용률} = \frac{\text{이식된 배아 수}}{\text{총 생성 배아 수}}
\]

##### 배아_잉여율

\[
\text{배아\_잉여율} = \frac{\text{저장된 배아 수}}{\text{총 생성 배아 수}}
\]

##### ivf_transfer_ratio

\[
\text{ivf\_transfer\_ratio} =
\frac{\text{이식된 배아 수}}{\text{총 생성 배아 수}}
\quad (\text{IVF 행에서만})
\]

##### ivf_storage_ratio

\[
\text{ivf\_storage\_ratio} =
\frac{\text{저장된 배아 수}}{\text{총 생성 배아 수}}
\quad (\text{IVF 행에서만})
\]

##### embryo_day_interaction

\[
\text{embryo\_day\_interaction} = \text{이식된 배아 수} \times \text{배아 이식 경과일}
\]

##### transfer_intensity

\[
\text{transfer\_intensity} =
\frac{\text{이식된 배아 수}}{\max(\text{시술 당시 나이\_num}, 1)}
\]

##### age_transfer_interaction

\[
\text{age\_transfer\_interaction} = \text{나이} \times \text{이식된 배아 수}
\]

##### age_x_single_transfer

\[
\text{age\_x\_single\_transfer} =
\text{나이} \times \mathbb{1}(\text{이식된 배아 수} = 1)
\]

##### transfer_day_optimal

\[
\mathbb{1}(\text{배아 이식 경과일} \in \{3,5\})
\]

##### day5plus / blastocyst_signal

\[
\mathbb{1}(\text{배아 이식 경과일} \ge 5)
\]

##### single_x_day5plus

\[
\mathbb{1}(\text{이식된 배아 수}=1 \land \text{이식 경과일}\ge5)
\]

##### multi_x_day5plus

\[
\mathbb{1}(\text{이식된 배아 수}\ge2 \land \text{이식 경과일}\ge5)
\]

##### fresh_transfer_ratio

\[
\text{fresh\_transfer\_ratio} =
\frac{\text{이식된 배아 수} \times \text{신선 배아 사용 여부}}
{\text{총 생성 배아 수}+1}
\]

##### micro_transfer_quality

\[
\text{micro\_transfer\_quality} =
\frac{\text{미세주입 배아 이식 수}}{\text{미세주입에서 생성된 배아 수}+1}
\]

#### B. frozen 전용 피처

##### is_frozen_transfer

\[
\mathbb{1}(\text{동결 배아 사용 여부}>0 \land \text{이식된 배아 수}>0)
\]

##### thaw_to_transfer_ratio

\[
\text{thaw\_to\_transfer\_ratio} =
\frac{\text{이식된 배아 수}}{\text{해동된 배아 수}+0.001}
\times \mathbb{1}(\text{해동된 배아 수}>0)
\]

##### frozen_x_age

\[
\text{frozen\_x\_age} = \text{동결 배아 사용 여부} \times \text{나이}
\]

##### frozen_x_clinic_exp

\[
\text{frozen\_x\_clinic\_exp} =
\text{동결 배아 사용 여부} \times \text{클리닉 내 총 시술 횟수\_num}
\]

##### frozen_x_stored

\[
\text{frozen\_x\_stored} =
\text{동결 배아 사용 여부} \times \text{저장된 배아 수}
\]

##### frozen_day_interaction

\[
\text{frozen\_day\_interaction} =
\text{동결 배아 사용 여부} \times \text{배아 이식 경과일}
\]

##### frozen_single_embryo

\[
\mathbb{1}(\text{동결 배아 사용 여부}>0 \land \text{이식된 배아 수}=1)
\]

##### frozen_x_day5plus

\[
\mathbb{1}(\text{동결 배아 사용 여부}>0 \land \text{배아 이식 경과일}\ge5)
\]

---

### 3-3. pruned domain 피처 (21개)

이 그룹은 **도메인 가설을 반영하되, 실제 성능에 기여한 소수 피처만 남긴 버전**이다.

#### A. 불임 원인 관련

##### male_factor_score

\[
\text{male\_factor\_score} =
\sum \text{(남성 관련 불임 원인 flag)}
\]

포함 예:
- 남성 주 불임 원인
- 남성 부 불임 원인
- 불임 원인 - 남성 요인
- 정자 농도/운동성/형태/면역학적 요인

##### female_factor_score

\[
\text{female\_factor\_score} =
\sum \text{(여성 관련 불임 원인 flag)}
\]

##### severe_sperm_factor_flag

\[
\mathbb{1}(\text{sperm\_issue\_score} \ge 2)
\]

##### unexplained_only_flag

\[
\mathbb{1}(\text{불명확 불임 원인}>0 \land \text{male\_factor\_score}=0 \land \text{female\_factor\_score}=0)
\]

#### B. donor / 연령 관련

##### donor_any_flag

\[
\mathbb{1}(\text{난자 기증} \lor \text{정자 기증} \lor \text{배아 기증})
\]

##### effective_oocyte_age

\[
\text{effective\_oocyte\_age} =
\begin{cases}
\text{난자 기증자 나이\_num}, & \text{if donor oocyte}\\
\text{시술 당시 나이\_num}, & \text{otherwise}
\end{cases}
\]

##### advanced_oocyte_age_flag

\[
\mathbb{1}(\text{effective\_oocyte\_age} \ge 38)
\]

##### advanced_age_autologous_flag

\[
\mathbb{1}(\text{시술 당시 나이}\ge 40 \land \text{본인 난자 사용})
\]

##### advanced_age_donor_oocyte_flag

\[
\mathbb{1}(\text{시술 당시 나이}\ge 40 \land \text{기증 난자 사용})
\]

#### C. 난자/배아 생산성과 가용성

##### retrieved_oocytes_total

\[
\text{retrieved\_oocytes\_total} =
\text{수집된 신선 난자 수} + \text{해동 난자 수}
\]

##### insemination_rate_from_retrieved

\[
\text{insemination\_rate\_from\_retrieved} =
\frac{\text{혼합된 난자 수}}{\text{retrieved\_oocytes\_total}}
\]

##### fertilization_rate_all

\[
\text{fertilization\_rate\_all} =
\frac{\text{총 생성 배아 수}}{\text{혼합된 난자 수}}
\]

##### embryo_yield_per_retrieved

\[
\text{embryo\_yield\_per\_retrieved} =
\frac{\text{총 생성 배아 수}}{\text{retrieved\_oocytes\_total}}
\]

##### transferable_embryos

\[
\text{transferable\_embryos} =
\text{이식된 배아 수} + \text{저장된 배아 수}
\]

##### transferable_rate

\[
\text{transferable\_rate} =
\frac{\text{transferable\_embryos}}{\text{총 생성 배아 수}}
\]

#### D. 냉동/해동/이식 전략

##### freeze_to_transfer_ratio_domain

\[
\text{freeze\_to\_transfer\_ratio\_domain} =
\frac{\text{저장된 배아 수}}{\text{이식된 배아 수}+1}
\]

##### thaw_survival_proxy

\[
\text{thaw\_survival\_proxy} =
\frac{\text{이식된 배아 수}}{\text{해동된 배아 수}}
\]

##### same_or_early_transfer_like

\[
\mathbb{1}(\text{배아 이식 경과일}>0 \land \text{배아 이식 경과일}\le3)
\]

##### extended_culture_flag_domain

\[
\mathbb{1}(\text{배아 이식 경과일}\ge5)
\]

#### E. 실패 이력 × 배아 가용성 상호작용

##### repeat_failure_x_transferable

\[
\text{repeat\_failure\_x\_transferable} =
\text{총실패횟수} \times \text{transferable\_embryos}
\]

##### ivf_failure_x_effective_oocyte_age

\[
\text{ivf\_failure\_x\_effective\_oocyte\_age} =
\text{IVF실패횟수} \times \text{effective\_oocyte\_age}
\]

---

## 4. 모델링 설정

### 4-1. 모델 구조
v21은 단일 모델이 아니라 **3단 구조**로 설계되었다.

#### (1) Global base
기존 최고 수준이었던 외부 blend를 그대로 사용


\[
\text{pred}_{global} = 0.65 \times \text{pred}_{v17} + 0.35 \times \text{pred}_{v19}
\]

- global base OOF AUC: **0.740839**
- global base LogLoss: **0.493248**
- global base AP: **0.452159**

#### (2) Transfer expert
- 학습 대상:


\[
\mathbb{1}(\text{이식된 배아 수} > 0)
\]

- sample 수: **213,516건**
- 양성률: **30.62%**

##### 목적
- 이미 이식이 이루어진 케이스에서 성공/실패를 더 정교하게 분리

#### (3) Frozen expert
- 학습 대상:


\[
\mathbb{1}(\text{이식된 배아 수}>0 \land \text{동결 배아 사용 여부}>0)
\]

- sample 수: **37,281건**
- 양성률: **24.69%**

##### 목적
- frozen transfer subgroup에서만 나타나는 해동/보존/배아 품질 신호를 별도 학습

---

### 4-2. CatBoost 설정

세 모델 모두 CatBoost 기반으로 학습했다.

#### 공통 파라미터
- iterations: `8000`
- learning_rate: `0.00837420773913813`
- depth: `8`
- l2_leaf_reg: `9.013710971500913`
- min_data_in_leaf: `47`
- random_strength: `1.9940254840418206`
- bagging_temperature: `0.7125808252032847`
- border_count: `64`
- eval_metric: `AUC`
- loss_function: `Logloss`
- od_type: `Iter`
- od_wait: `400`
- class_weights: `{0:1.0, 1:1.3}`

#### 학습 방식
- **5 seeds**
  - `[42, 2026, 2604, 123, 777]`
- **5-fold StratifiedKFold**
- **GPU 학습**
- 각 seed마다 다른 fold split 사용 후 평균 앙상블

---

### 4-3. 최종 expert blend 방식
OOF 기준으로 subgroup별 최적 가중치를 탐색했다.

탐색 결과:
- non-frozen transfer 영역:
  - `a_transfer_nonfrozen = 0.30`
- frozen transfer 영역:
  - `b_transfer_frozen = 0.00`
  - `c_frozen_frozen = 0.35`

즉 최종 예측식은 다음과 같다.

#### (1) non-transfer

\[
\text{pred}_{final} = \text{pred}_{global}
\]

#### (2) non-frozen transfer

\[
\text{pred}_{final} =
0.70 \times \text{pred}_{global}
+
0.30 \times \text{pred}_{transfer}
\]

#### (3) frozen transfer

\[
\text{pred}_{final} =
0.65 \times \text{pred}_{global}
+
0.35 \times \text{pred}_{frozen}
\]

#### 해석
- frozen subgroup에서는 transfer expert보다 **frozen expert가 더 유효**
- 따라서 frozen group에서는 transfer expert 가중치가 0으로 선택됨

---

## 5. 성능 기록

### 5-1. v21_expert 자체 성능

#### Global base
- OOF AUC: **0.740839**
- OOF LogLoss: **0.493248**
- OOF AP: **0.452159**

#### Final expert blend
- OOF AUC: **0.740895**
- OOF LogLoss: **0.493275**
- OOF AP: **0.452196**

#### 개선량
- AUC: **+0.000056**
- LogLoss: **+0.000027** (소폭 악화)
- AP: **+0.000037**

즉,
- 전체 ranking 성능(AUC)은 소폭 개선
- positive retrieval 측면(AP)도 소폭 개선
- 다만 probability calibration 측면(LogLoss)은 아주 조금 악화

---

### 5-2. subgroup 성능 변화

| 그룹 | global | final | delta |
|---|---:|---:|---:|
| transfer | 0.6750 | 0.6750 | +0.00007 |
| non-transfer | 0.9401 | 0.9401 | +0.00000 |
| frozen transfer | 0.6200 | 0.6207 | **+0.00069** |
| fresh transfer | 0.6778 | 0.6778 | +0.00004 |
| donor egg | 0.6797 | 0.6801 | +0.00034 |

#### 해석
- 전체 개선의 중심은 **frozen transfer subgroup**
- global 모델이 이미 강했던 non-transfer 영역은 거의 변화 없음
- expert 구조는 특히 **동결 이식군에서 미세 개선**을 만들었다

---

### 5-3. expert 자체 subgroup 성능
- transfer expert subgroup OOF AUC: **0.674740**
- frozen expert subgroup OOF AUC: **0.619906**

#### 해석
expert 단독 성능이 global보다 훨씬 높지는 않았지만,  
**subgroup에 한해 global 예측과 다른 패턴을 학습했기 때문에 blend에서 gain을 제공**했다.

---

### 5-4. 최종 팀 앙상블 성능
- **v21 + 팀원 모델 앙상블 Public AUC: 0.7423956072**

#### 해석
- v21 자체 improvement는 미세했지만,
- 팀원 XGB/LGB/CatBoost 계열과 결합했을 때 **최종 최고 Public AUC**를 달성함
- 즉 v21의 강점은 단독 폭발력이 아니라 **앙상블용 diversity 제공**에 있었음

---

## 6. 피처 중요도 및 해석

### 6-1. Transfer expert 중요도 상위 피처

상위 중요 피처:
1. `effective_oocyte_age`
2. `advanced_oocyte_age_flag`
3. `시술 시기 코드`
4. `advanced_age_autologous_flag`
5. `transfer_intensity`
6. `transfer_day_cat`
7. `배아저장비율`
8. `freeze_to_transfer_ratio_domain`
9. `same_or_early_transfer_like`
10. `embryo_day_interaction`

#### 해석

##### (1) effective_oocyte_age
- 환자 나이보다도 **실제 사용된 난자의 생물학적 연령**이 중요함을 의미
- 특히 donor egg가 포함된 경우, 단순 환자 연령보다 더 직접적인 성공 요인으로 작용

##### (2) advanced_oocyte_age_flag / advanced_age_autologous_flag
- 고령 여부 자체보다도 **고령 + 자가난자 사용 상황**이 더 중요한 위험 신호
- IVF 도메인 지식과도 일치하는 결과

##### (3) transfer_intensity / embryo_day_interaction / transfer_day_cat
- 이식된 배아 수와 이식 시점(day3/day5 등)이 성공률에 큰 영향을 준다는 점을 반영
- 즉 transfer 단계에서는 **배아 수량 + 배양일수 전략**이 핵심

##### (4) 배아저장비율 / freeze_to_transfer_ratio_domain / transferable_rate
- 배아를 얼마나 남길 수 있었는지는 단순 양적 정보가 아니라 **배아 질과 가용성의 proxy** 역할을 함

---

### 6-2. Frozen expert 중요도 상위 피처

상위 중요 피처:
1. `시술 시기 코드`
2. `effective_oocyte_age`
3. `thaw_to_transfer_ratio`
4. `transfer_intensity`
5. `복합위험도점수`
6. `transferable_embryos`
7. `ivf_failure_x_effective_oocyte_age`
8. `thaw_survival_proxy`
9. `advanced_oocyte_age_flag`
10. `배아 이식 경과일`

#### 해석

##### (1) thaw_to_transfer_ratio / thaw_survival_proxy
- frozen subgroup에서는 **해동된 배아가 실제 이식 가능한 상태로 이어졌는지**가 핵심
- frozen expert가 별도 의미를 가지는 가장 대표적인 근거

##### (2) transferable_embryos / freeze_to_transfer_ratio_domain
- 동결 가능한 배아를 많이 확보한 사례는 배아 품질과 시술 안정성이 높을 가능성이 있음

##### (3) ivf_failure_x_effective_oocyte_age
- 단순 반복 실패보다 **반복 실패 × 난자 연령** 조합이 더 중요한 위험 신호
- frozen subgroup에서는 누적 실패 이력의 영향이 더 두드러짐

##### (4) 시술 시기 코드
- 시술 시기별 프로토콜, 병원 운영, 실험실 환경 차이가 내재된 proxy일 수 있음
- 예상 외로 높은 중요도를 보여준 피처

---

### 6-3. 중요도 0 피처 해석

#### transfer expert 중요도 0
- `난자 해동 경과일`
- `is_ivf`
- `시술 유형`
- `실제이식여부`
- `불임 원인 - 정자 면역학적 요인`
- `is_di`

#### frozen expert 중요도 0
- `동결 배아 사용 여부`
- `is_ivf`
- `불임 원인 - 정자 면역학적 요인`
- `난자 혼합 경과일`
- `실제이식여부`
- `난자 해동 경과일`
- `is_frozen_transfer`
- `시술 유형`
- `is_di`

#### 해석
- expert 모델은 이미 subgroup 자체를 조건으로 잘라 학습했기 때문에  
  해당 subgroup 안에서는 이 피처들이 **상수 또는 거의 상수**가 되어 정보가 사라짐
- 이는 expert 구조가 제대로 작동하고 있음을 보여주는 결과이기도 함

---

## 7. 인사이트

### 7-1. 전체 모델은 이미 포화 상태였다
v17~v20 구간에서 전체 OOF AUC는 0.7408 수준에서 거의 변하지 않았다.  
즉 단순히 피처를 더 넣는 방식만으로는 큰 개선을 만들기 어려웠다.

---

### 7-2. 개선해야 할 핵심 구간은 transfer / frozen transfer였다
subgroup 분석 결과,
- non-transfer는 이미 매우 높은 AUC를 보였고,
- 실제 병목은 transfer, 특히 frozen transfer였다.

따라서 v21에서는 전체 모델 개선보다 **subgroup별 expert 모델**을 도입하는 것이 합리적이었다.

---

### 7-3. 도메인 피처는 global보다 expert에서 더 잘 작동했다
v19/v20 실험에서 추가한 도메인 피처는 global 모델에서는 큰 폭의 성능 향상을 만들지 못했지만,  
v21의 transfer/frozen expert에서는 다음과 같은 피처가 상위 중요도로 나타났다.

- `effective_oocyte_age`
- `advanced_age_autologous_flag`
- `transferable_embryos`
- `freeze_to_transfer_ratio_domain`
- `thaw_survival_proxy`

즉 도메인 피처는 전체 모델의 대체재가 아니라,  
**특정 subgroup에서 패턴을 보완하는 보조 모델의 핵심 입력**으로 더 적합했다.

---

### 7-4. frozen subgroup은 transfer expert보다 frozen expert가 더 적합했다
최적 blend 결과:
- frozen group에서 transfer expert weight = 0
- frozen expert weight = 0.35

이는 frozen subgroup에서 필요한 신호가 일반 transfer 패턴과 다르며,  
**해동/보존/동결 배아 관련 피처가 별도 모델에서 더 잘 작동한다**는 점을 보여준다.

---

### 7-5. v21의 가장 큰 가치는 “단독 최고점”이 아니라 “최종 앙상블 기여”였다
v21 단독 개선폭은 크지 않았지만,
최종적으로 **팀원 모델과 앙상블했을 때 Public AUC 0.7423956072로 최고 성능**을 달성했다.

즉 v21은
- 단독 점수보다
- **다른 모델과 다른 신호를 제공하는 다양성(diversity)** 측면에서 더 큰 의미가 있었다.

---

## 8. 발표용 한 줄 요약

### 짧은 요약
> v21은 global 모델의 포화 구간을 넘기 위해 transfer/frozen subgroup expert를 도입한 모델이며,  
> 특히 frozen transfer에서 미세한 개선을 만들고 최종 팀 앙상블 최고 성능에 기여했다.

### 조금 더 자세한 요약
> v21은 0.65×v17 + 0.35×v19 global base 위에 transfer expert와 frozen expert를 얹은 구조로,  
> `effective_oocyte_age`, `transferable_embryos`, `thaw_survival_proxy` 등 도메인 피처를 subgroup에 특화해 활용했다.  
> 단독 OOF AUC는 0.740895로 소폭 개선되었고, 최종적으로 팀원 모델과의 앙상블에서 Public AUC 0.7423956072를 기록했다.