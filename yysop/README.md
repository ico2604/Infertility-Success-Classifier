# 🧬 AI for Infertility Treatment
## Pregnancy Success Prediction using Machine Learning

난임 환자 데이터를 기반으로 **임신 성공 여부를 예측하는 AI 모델**을 개발하는 프로젝트입니다.

난임은 전 세계적으로 증가하는 중요한 의료 문제로, 많은 환자들이 치료 과정에서 **신체적·정신적 부담과 높은 비용**을 경험합니다.  
따라서 **최소한의 시술로 임신 성공 가능성을 높이는 것**은 매우 중요한 의료 목표입니다.

본 프로젝트는 난임 시술 데이터를 활용하여 **임신 성공 여부를 예측**하고  
임신 성공에 영향을 미치는 **핵심 특성(feature)** 을 탐색하는 것을 목표로 합니다.

AI 기반 예측 모델은 의료진의 **데이터 기반 의사결정 지원**과 환자 맞춤형 치료 전략 수립에 기여할 수 있습니다.

---

# 🏆 Hackathon Objective

- 난임 환자 데이터 분석  
- 임신 성공 여부 예측 모델 개발  
- 임신 성공에 영향을 미치는 핵심 특성 탐색  

---

# 📊 Project Overview

| 항목 | 내용 |
|------|------|
| Task | 난임 시술 후 임신 성공 여부 예측 |
| Domain | Infertility Treatment (ART) |
| Model | XGBoost |
| Validation | Stratified K-Fold (5 folds) |
| Evaluation Metric | ROC-AUC / PR-AUC |

---

# 🧠 Model

### 모델
XGBoost

## 📊 Model Performance

| Metric | Score |
|------|------|
| ROC-AUC | **0.74011** |
| PR-AUC | **0.45054** |

### 선정 이유

- Tabular 의료 데이터에서 높은 성능  
- Feature interaction 학습에 강점  
- 안정적인 generalization 성능  

---

# 🔎 Validation Strategy

### 방법

Stratified K-Fold (5 folds)  
OOF Prediction

### 선정 이유

- 데이터에 class imbalance 존재  
- Fold 간 class 비율 유지  
- 안정적인 validation 성능 확보  

---

# ⚙️ Feature Engineering

난임 치료 도메인 지식을 기반으로 **30+ 파생 변수 생성**

## 🧪 Embryo Process Efficiency

난자 → 배아 → 이식 과정의 효율 반영

- 배아생성효율  
- ICSI수정효율  
- 배아이식비율  
- 배아저장비율  
- 난자활용률  

## 📊 Treatment History

과거 시술 이력 기반 성공 확률 반영

- 전체임신률  
- IVF임신률  
- DI임신률  
- 임신유지율  

## ⚠️ Failure Pattern

반복 실패 패턴 반영

- 총실패횟수  
- IVF실패횟수  
- 반복IVF실패 여부  

특히 **IVF 실패 ≥ 3** 은 중요한 임상 위험 요인입니다.

## 👩‍⚕️ Age Risk Features

난임 치료에서 중요한 변수 반영

- 나이  
- 나이²  
- 고령 여부  
- 초고령 여부  
- 극고령 여부  

## 🔗 Interaction Features

- 나이 × 시술횟수  
- 나이 × IVF실패  
- 나이 × IVF임신률  

---

# 📅 Experiments

## Day1 — Baseline Model (v1)

### Model Parameters

```python
n_estimators = 3000
learning_rate = 0.02

max_depth = 5
min_child_weight = 5
gamma = 0.1

subsample = 0.8
colsample_bytree = 0.7

reg_alpha = 0.1
reg_lambda = 1.5
```

### Result

| Metric | Score |
|------|------|
| ROC-AUC | **0.73997** |
| PR-AUC | **0.4505** |

Baseline 성능: **ROC-AUC ≈ 0.74**

## Day2 — Model Tuning (v2)

### Experiment Framework
- baseline
- depth_up
- lr_down
- reg_relax
- sampling
- best_combo

### 실험 로그 자동 기록
xgb_v2_results_log.csv

### Regularization Relaxation
min_child_weight : 5 → 3
gamma            : 0.1 → 0
reg_alpha        : 0.1 → 0
reg_lambda       : 1.5 → 1

### Result

| Metric | Score |
|------|------|
| ROC-AUC | **0.74011** |
| PR-AUC | **0.45054** |

Baseline 대비 소폭 성능 개선

# 📈 Current Performance

| Version | ROC-AUC |
|--------|--------|
| v1 Baseline | 0.73997 |
| v2 Tuned | **0.74011** |

- Fold variance 매우 낮음
- 안정적인 validation 성능 확인

## 📊 Model Visualization

### ROC Curve Comparison

| Baseline (v1) | Tuned (v2) |
|------|------|
| ![](yysop/outputs/xgb_v1_roc_curve.png) | ![](yysop/outputs/xgb_v2_reg_relax_roc_curve.png) |

### PR Curve

![](yysop/outputs/xgb_v1_pr_curve.png)

### Feature Importance

![](yysop/outputs/xgb_v1_feature_importance.png)

Feature Importance

# 🚀 Future Work

### Model Tuning
- max_depth = 6
- learning_rate = 0.01
- n_estimators = 5000

### Feature Interaction Expansion
- 나이 × 배아생성효율
- 나이 × IVF임신률
- 실패횟수 × 나이

### Model Comparison
- LightGBM
- CatBoost

# 🎯 Goal
ROC-AUC 0.74 → 0.76+

Feature Engineering + Model Tuning을 통한 성능 개선

# 📂 Repository Structure
yysop/
├── data/
├── outputs/
│   ├── xgb_v1_roc_curve.png
│   ├── xgb_v1_pr_curve.png
│   ├── xgb_v1_feature_importance.png
│   └── xgb_v2_reg_relax_roc_curve.png
├── src/
│   ├── eda.py
│   ├── xgb_kfold_v1.py
│   └── xgb_kfold_v2.py
└── README.md


# 🧑‍⚕️ Expected Impact

본 프로젝트는 난임 치료 데이터 분석을 통해

임신 성공 확률 예측

환자 맞춤형 치료 전략 지원

의료진의 데이터 기반 의사결정 보조

에 활용될 수 있는 의료 AI 시스템 개발 가능성을 탐색합니다.