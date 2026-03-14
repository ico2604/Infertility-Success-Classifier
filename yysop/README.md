# 🧬 AI for Infertility Treatment  
### Pregnancy Success Prediction using Machine Learning

난임은 전 세계적으로 증가하는 중요한 의료 문제로, 많은 환자들이 장기간의 치료 과정에서 **신체적·정신적 부담과 높은 비용**을 경험합니다.

난임 치료 과정에서는 여러 차례의 시술이 필요할 수 있으며, 환자에게 큰 경제적·심리적 부담이 발생합니다. 따라서 **최소한의 시술로 임신 성공 가능성을 높이는 것**은 매우 중요한 의료 목표입니다.

본 프로젝트는 **난임 환자 데이터를 기반으로 임신 성공 여부를 예측하는 AI 모델을 개발**하고, 임신 성공에 영향을 미치는 **핵심 의료 특성을 탐색하는 것**을 목표로 합니다.

AI 기반 예측 모델은 의료진이 치료 전략을 결정할 때 **데이터 기반 의사결정 지원 시스템**으로 활용될 수 있습니다.

---

# 🏆 Hackathon Challenge

본 프로젝트는 **난임 환자 데이터를 활용하여 임신 성공 여부를 예측하는 AI 모델 개발**을 목표로 하는 해커톤 과제입니다.

### 목표

- 난임 환자 데이터 분석
- 임신 성공 여부 예측 모델 개발
- 임신 성공에 영향을 미치는 핵심 특성 탐색

---

# 📊 Project Overview

| 항목 | 내용 |
|-----|-----|
| Task | 난임 시술 후 임신 성공 여부 예측 |
| Domain | Infertility Treatment (ART) |
| Model | XGBoost |
| Validation | Stratified K-Fold |
| Evaluation Metric | ROC-AUC / PR-AUC |

---

# 🧠 Modeling Strategy

### Model

**XGBoost**

### 선정 이유

- Tabular 의료 데이터에서 높은 성능
- Feature Interaction 학습 능력
- 안정적인 Generalization

---

# 🔎 Validation Strategy

### 방법

Stratified K-Fold (5 folds)  
OOF Prediction

### 이유

- 데이터 Class imbalance 존재
- Fold 간 클래스 비율 유지
- 안정적인 모델 평가

---

# ⚙️ Feature Engineering

의료 도메인 기반 Feature Engineering 적용  
총 **30+ 파생 변수 생성**

---

## 🧪 Embryo Process Efficiency

난자 → 배아 → 이식 과정 효율 반영

- 배아생성효율
- ICSI수정효율
- 배아이식비율
- 배아저장비율
- 난자활용률

---

## 📊 Treatment History

과거 시술 이력 기반 성공 확률 반영

- 전체임신률
- IVF임신률
- DI임신률
- 임신유지율

---

## ⚠️ Failure Pattern

반복 실패 패턴 반영

- 총실패횟수
- IVF실패횟수
- 반복IVF실패 여부

특히 **IVF 실패 ≥ 3** 은 중요한 임상 위험 요인

---

## 👩‍⚕️ Age Risk Features

난임 치료에서 가장 중요한 변수

- 나이
- 나이²
- 고령 여부
- 초고령 여부
- 극고령 여부

---

## 🔗 Interaction Features

- 나이 × 시술횟수
- 나이 × IVF실패
- 나이 × IVF임신률

---

# 📅 Experiments

---

# Day1 — Baseline Model

### Model Parameters


n_estimators = 3000
learning_rate = 0.02

max_depth = 5
min_child_weight = 5
gamma = 0.1

subsample = 0.8
colsample_bytree = 0.7

reg_alpha = 0.1
reg_lambda = 1.5


### Result

| Metric | Score |
|------|------|
| ROC-AUC | **0.73997** |
| PR-AUC | **0.4505** |

Baseline 모델 성능

**ROC-AUC ≈ 0.74**

---

# Day2 — Model Tuning

### Experiment Framework


baseline
depth_up
lr_down
reg_relax
sampling
best_combo


모든 실험 결과 자동 기록


xgb_v2_results_log.csv


---

### Regularization Relaxation


min_child_weight : 5 → 3
gamma : 0.1 → 0
reg_alpha : 0.1 → 0
reg_lambda : 1.5 → 1


---

### Result

| Metric | Score |
|------|------|
| ROC-AUC | **0.74011** |
| PR-AUC | **0.45054** |

Baseline 대비 **소폭 성능 개선**

---

# 📈 Model Performance

| Version | ROC-AUC |
|------|------|
| Baseline | 0.73997 |
| Tuned | **0.74011** |

✔ Fold variance 매우 낮음  
✔ 안정적인 validation 성능

---

# 📊 Model Visualization

## ROC Curve

![ROC Curve](yysop/outputs/xgb_v2_reg_relax_roc_curve.png)

---

## Feature Importance

![Feature Importance](yysop/outputs/xgb_v1_feature_importance.png)

---

# 🚀 Future Work

### Model Tuning


max_depth = 6
learning_rate = 0.01
n_estimators = 5000


### Feature Interaction

- 나이 × 배아생성효율
- 나이 × IVF임신률
- 실패횟수 × 나이

### Model Comparison

- LightGBM
- CatBoost

---

# 🎯 Goal


ROC-AUC 0.74 → 0.76+


Feature Engineering + Model Tuning을 통한 성능 개선

---

# 📂 Repository Structure


src/
├ eda.py
├ xgb_kfold_v1.py
└ xgb_kfold_v2.py

yysop/
├ data/
└ outputs/

README.md
requirements.txt


---

# 🧑‍⚕️ Impact

본 프로젝트는 난임 치료 데이터 분석을 통해

- 임신 성공 확률 예측
- 환자 맞춤형 치료 전략 지원
- 의료 의사결정 보조

에 활용될 수 있는 **의료 AI 시스템 개발 가능성**을 탐색합니다.