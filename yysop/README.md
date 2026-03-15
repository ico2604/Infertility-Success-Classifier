# 🧬 AI for Infertility Treatment
## Pregnancy Success Prediction using Machine Learning
 
난임 환자 데이터를 기반으로 **임신 성공 여부를 예측하는 AI 모델**을 개발하는 프로젝트입니다.
 
난임은 전 세계적으로 증가하는 중요한 의료 문제로, 많은 환자들이 치료 과정에서 **신체적·정신적 부담과 높은 비용**을 경험합니다.  
따라서 **최소한의 시술로 임신 성공 가능성을 높이는 것**은 매우 중요한 의료 목표입니다.
 
본 프로젝트는 난임 시술 데이터를 활용하여 **임신 성공 여부를 예측**하고  
임신 성공에 영향을 미치는 **핵심 특성(feature)** 을 탐색하는 것을 목표로 합니다.
 
AI 기반 예측 모델은 의료진의 **데이터 기반 의사결정 지원**과 환자 맞춤형 치료 전략 수립에 기여할 수 있습니다.
 
---
 
## 🏆 Hackathon Objective
 
- 난임 환자 데이터 분석
- 임신 성공 여부 예측 모델 개발
- 임신 성공에 영향을 미치는 핵심 특성 탐색
 
---
 
## 📊 Project Overview
 
| 항목 | 내용 |
|------|------|
| Task | 난임 시술 후 임신 성공 여부 예측 |
| Domain | Infertility Treatment (ART) |
| Model | XGBoost → Ensemble |
| Validation | Stratified K-Fold (5 folds) |
| Evaluation Metric | ROC-AUC / PR-AUC |
 
---
 
## 🧠 Model
 
### 모델 구성
 
| Model | Version |
|-------|---------|
| XGBoost | reg_relax tuned |
| CatBoost | baseline v2 |
| LightGBM | baseline v1 |
| Rank Ensemble | weighted |
 
### 선정 이유
 
- **XGBoost**: tabular 의료 데이터에서 높은 baseline 성능
- **CatBoost**: 범주형/분포 차이에 강하고 ensemble diversity 기여
- **LightGBM**: leaf-wise split 기반의 다른 예측 패턴 제공
- **Rank Ensemble**: probability calibration 차이를 줄이며 안정적인 결합 가능
 
### 📊 Model Performance
 
| Metric | Score |
|--------|-------|
| Best Single Model (XGBoost v2) ROC-AUC | **0.74011** |
| Best Ensemble OOF ROC-AUC | **0.740448** |
| Public Leaderboard | **0.74218** |
 
---
 
## 🔎 Validation Strategy
 
### 방법
 
- Stratified K-Fold (5 folds)
- OOF Prediction
 
### 선정 이유
 
- 데이터에 class imbalance 존재
- Fold 간 class 비율 유지
- 안정적인 validation 성능 확보
- OOF 기반으로 ensemble 실험의 일반화 성능 검증 가능
 
---
 
## ⚙️ Feature Engineering
 
난임 치료 도메인 지식을 기반으로 **30+ 파생 변수 생성**
 
### 🧪 Embryo Process Efficiency
 
난자 → 배아 → 이식 과정의 효율 반영
 
| Feature | 설명 |
|---------|------|
| 배아생성효율 | 수집 난자 대비 생성 배아 수 |
| ICSI수정효율 | 미세주입 난자 대비 배아 생성 수 |
| 배아이식비율 | 생성 배아 중 이식된 비율 |
| 배아저장비율 | 생성 배아 중 저장된 비율 |
| 난자활용률 | 수집 난자 중 ICSI에 활용된 비율 |
 
### 📊 Treatment History
 
과거 시술 이력 기반 성공 확률 반영
 
| Feature | 설명 |
|---------|------|
| 전체임신률 | 총 시술 대비 임신 성공 비율 |
| IVF임신률 | IVF 시술 대비 임신 성공 비율 |
| DI임신률 | DI 시술 대비 임신 성공 비율 |
| 임신유지율 | 임신 대비 출산 성공 비율 |
 
### ⚠️ Failure Pattern
 
반복 실패 패턴 반영
 
| Feature | 설명 |
|---------|------|
| 총실패횟수 | 총 시술 - 총 임신 횟수 |
| IVF실패횟수 | IVF 시술 - IVF 임신 횟수 |
| 반복IVF실패 여부 | IVF 실패 ≥ 3회 여부 (binary) |
 
> 특히 **IVF 실패 ≥ 3** 은 중요한 임상 위험 요인입니다.
 
### 👩‍⚕️ Age Risk Features
 
| Feature | 설명 |
|---------|------|
| 나이 | 시술 당시 나이 (수치형 변환) |
| 나이² | 비선형 나이 효과 포착 |
| 고령 여부 | 35세 이상 binary |
| 초고령 여부 | 40세 이상 binary |
| 극고령 여부 | 42세 이상 binary |
 
### 🔗 Interaction Features
 
| Feature | 설명 |
|---------|------|
| 나이 × 시술횟수 | 고령 + 반복 시술 복합 지표 |
| 나이 × IVF실패 | 고령 + 반복 실패 복합 지표 |
| 나이 × IVF임신률 | 고령이지만 성공 경험이 있는 경우 |
 
---
 
## 📅 Experiments
 
### Day1 — Baseline Model (v1)
 
#### Model Parameters
 
```python
n_estimators     = 3000
learning_rate    = 0.02
max_depth        = 5
min_child_weight = 5
gamma            = 0.1
subsample        = 0.8
colsample_bytree = 0.7
reg_alpha        = 0.1
reg_lambda       = 1.5
```
 
#### Result
 
| Metric | Score |
|--------|-------|
| ROC-AUC | 0.73997 |
| PR-AUC | 0.4505 |
 
> Baseline 성능: ROC-AUC ≈ 0.74
 
---
 
### Day2 — Model Tuning (v2)
 
#### Experiment Framework
 
실험 구성은 아래 6가지 설정을 비교하였으며, 결과는 자동으로 로그에 기록되었다.
 
```
baseline / depth_up / lr_down / reg_relax / sampling / best_combo
→ xgb_v2_results_log.csv
```
 
#### Regularization Relaxation
 
| Parameter | Baseline | v2 |
|-----------|----------|----|
| min_child_weight | 5 | 3 |
| gamma | 0.1 | 0 |
| reg_alpha | 0.1 | 0 |
| reg_lambda | 1.5 | 1 |
 
#### Result
 
| Metric | Score |
|--------|-------|
| ROC-AUC | 0.74011 |
| PR-AUC | 0.45054 |
 
#### 해석
 
Regularization을 완화했을 때 성능이 소폭 상승하였다.  
이는 현재 데이터에서 과도한 규제가 오히려 feature interaction 학습을 제한했을 가능성을 시사한다.  
 
> 따라서 Day2에서는 **XGBoost v2 (reg_relax)** 를 최적 단일 모델로 선정하였다.
 
---
 
### Day3 — Single Model Saturation and Ensemble Optimization
 
Day3에서는 Day1~Day2에서 진행한 XGBoost 중심 반복 실험 결과를 재정리하고,  
단일 모델 성능이 일정 수준에서 포화(saturation)되는 패턴을 확인한 뒤  
성능 개선 전략을 **single model tuning → ensemble optimization** 방향으로 확장하였다.
 
**핵심 목표**
 
- XGBoost 실험 버전별 성능 흐름 정리
- feature engineering 확장의 실제 효과 재해석
- 왜 XGBoost v2(reg_relax)를 최종 단일 모델로 선택했는지 설명
- 왜 ensemble로 방향을 전환했는지 근거 제시
- Stacking, Probability Ensemble, Rank Ensemble을 비교하고 다음 단계 전략 결정
 
#### 1. Why XGBoost First?
 
프로젝트 초기에는 XGBoost를 중심 모델(base model)로 설정하였다.
 
**선정 이유**
 
- 의료 데이터와 같은 tabular dataset에서 강한 baseline 성능
- 비선형 관계와 feature interaction 학습에 강점
- 비교적 해석 가능한 feature importance 제공
- 반복 실험과 tuning 속도가 빠르며 baseline 구축 효율이 높음
 
> 즉, Day1~Day2 구간에서는 XGBoost를 기준 모델로 삼아  
> 데이터가 도달할 수 있는 성능 상한선을 먼저 확인하는 전략을 사용하였다.
 
#### 2. XGBoost Experiment History
 
| Version | Experiment | Description | OOF ROC-AUC |
|---------|------------|-------------|-------------|
| v1 | baseline | 기본 파라미터 기반 초기 모델 | 0.73997 |
| v2 | reg_relax | Regularization 완화 튜닝 | **0.74011** |
| v3 | reg_relax_feat_plus | 파생변수 확장 | 0.73998 |
| v4 | reg_relax_feat_lite | 파생변수 일부 정리 | 0.74003 |
| v5 | optuna_branch_v5_dedup | Optuna + branch feature + dedup | 0.74004 |
 
**해석**
 
- 초기 baseline 대비 tuning은 유효
- 그러나 추가 feature engineering 확장 효과는 제한적
- 복잡한 파생변수 및 branch feature 추가가 반드시 성능 향상으로 이어지지는 않음
- **Day3 시점에서 가장 강한 XGBoost 단일 모델은 v2(reg_relax)**
 
#### 3. What Was Tested in XGBoost?
 
##### 3.1 Baseline (v1)
 
| Metric | Score |
|--------|-------|
| ROC-AUC | 0.73997 |
| PR-AUC | 0.45050 |
 
> 기본 모델만으로도 ROC-AUC 약 0.74 수준의 강한 baseline을 확보하였다.
 
##### 3.2 Regularization Relaxation (v2)
 
| Metric | Score |
|--------|-------|
| ROC-AUC | 0.74011 |
| PR-AUC | 0.45054 |
 
> Regularization 완화 시 성능 소폭 상승 → 현재 데이터에서 과도한 규제가 interaction 학습을 제한했을 가능성
 
##### 3.3 Feature Expansion (v3)
 
| Metric | Score |
|--------|-------|
| ROC-AUC | 0.73998 |
 
> Feature를 더 추가했지만 성능은 오히려 소폭 하락  
> 기존 feature set이 이미 충분히 강력하고, 추가 feature가 noise에 가까울 수 있음
 
##### 3.4 Lite Feature Set (v4)
 
| Metric | Score |
|--------|-------|
| ROC-AUC | 0.74003 |
 
> feature를 일부 정리했을 때 성능이 약간 회복되었으나 v2를 넘지는 못함  
> **feature를 많이 넣는 것보다 핵심 feature를 유지하는 것이 더 중요**
 
##### 3.5 Optuna + Branch Feature (v5)
 
| Metric | Score |
|--------|-------|
| ROC-AUC | 0.74004 |
| PR-AUC | 0.45062 |
 
> Optuna tuning과 branch feature 확장에도 불구하고 v2를 넘지 못함  
> **→ single model 성능이 포화 상태에 근접했다고 판단**
 
#### 4. Why Was XGBoost v2 Selected?
 
| 선택 이유 |
|-----------|
| 가장 높은 OOF ROC-AUC |
| 실험 구조가 비교적 단순하고 해석 가능성이 높음 |
| 과도한 feature expansion 없이 안정적인 성능 |
| ensemble의 base model로 사용하기 적합 |
| 복잡한 branch feature 없이도 가장 높은 성능 달성 |
 
#### 5. Feature Engineering Analysis
 
##### Feature Engineering Result Summary
 
| Feature Set | ROC-AUC |
|-------------|---------|
| baseline + 주요 파생변수 | 0.74011 |
| 추가 feature 확장 | 0.73998 |
| 일부 feature 정리 | 0.74003 |
| branch + optuna | 0.74004 |
 
**해석**
 
- Feature engineering은 초기 단계에서는 분명 성능 개선에 기여
- 그러나 Day3 시점에서는 추가 확장의 성능 향상이 거의 없음
- 일부 실험에서는 오히려 noise가 증가
 
> **Feature engineering은 이미 1차 포화 구간에 도달했고,**  
> **추가 성능 향상은 ensemble에서 찾는 것이 더 효율적이다.**
 
#### 6. Why Move to Ensemble?
 
| Model | ROC-AUC |
|-------|---------|
| XGBoost (v2 reg_relax) | 0.74011 |
| CatBoost v2 | 0.74005 |
| LightGBM v1 | 0.73987 |
 
**해석**
 
- 세 모델의 성능이 거의 유사함
- 각 모델은 비슷한 수준의 예측력을 가짐
- 그러나 학습 방식이 다르기 때문에 오차 패턴(error pattern)이 다를 가능성이 있음
 
> 단일 모델을 더 미세하게 튜닝하기보다  
> 서로 다른 모델을 결합하는 것이 성능 향상에 더 유리하다고 판단
 
**주목할 점**
 
- CatBoost 단일 성능은 XGBoost보다 약간 낮았음
- 그러나 ensemble weight에서는 CatBoost 비중이 가장 높게 나옴
- **→ CatBoost가 단일 성능 이상으로 ensemble diversity에 기여하고 있음을 의미**
 
#### 7. Ensemble Experiment Flow
 
##### 7.1 Stacking (First Trial)
 
LogisticRegression을 meta model로 적용하여 base model OOF prediction을 결합하였다.
 
**Result**: Stacking은 weighted / rank ensemble을 넘지 못하였다.
 
> 현재 base model 조합에서는 복잡한 meta model보다  
> 단순하고 안정적인 ensemble 방식이 더 적합한 것으로 판단
> 
> **Weighted / Rank Ensemble > Stacking**
 
##### 7.2 Probability Ensemble
 
```
final_pred = w1 × xgb_prob + w2 × cat_prob + w3 × lgb_prob
```
 
| Ensemble | ROC-AUC |
|----------|---------|
| Equal Ensemble | 0.740436 |
| Weighted Ensemble | 0.740444 |
 
**Best weights**
 
```
XGB : 0.33
CAT : 0.40
LGB : 0.27
```
 
##### 7.3 Rank Ensemble
 
```
1. 각 모델 예측값 → percentile rank 변환
2. rank를 가중 평균
```
 
**Why Needed?**
 
Tree 기반 모델은 probability scale이 서로 다를 수 있다.  
Rank ensemble은 상대적 ordering을 중심으로 결합하기 때문에 calibration 차이에 더 안정적이다.
 
| Ensemble | ROC-AUC |
|----------|---------|
| Rank Ensemble | **0.740448** |
 
**Best weights**
 
```
XGB : 0.33
CAT : 0.41
LGB : 0.26
```
 
##### Ensemble Step-by-Step Summary
 
| Step | Method | ROC-AUC |
|------|--------|---------|
| 1 | Stacking (LR meta model) | < baseline |
| 2 | Equal Probability Ensemble | 0.740436 |
| 3 | Weighted Probability Ensemble | 0.740444 |
| 4 | **Rank Ensemble** | **0.740448** |
 
#### 8. Final Day3 Model
 
> **Best Day3 Model: Rank Ensemble**
 
| 구성 | 내용 |
|------|------|
| Base Models | XGB reg_relax / CatBoost v2 / LightGBM v1 |
| Weights | 0.33 / 0.41 / 0.26 |
| Method | Percentile Rank Weighted Average |
 
| Metric | Score |
|--------|-------|
| OOF ROC-AUC | 0.740448 |
| Public LB | **0.74218** |
 
> OOF보다 Public LB가 높게 나타났으며  
> 이는 모델이 **과적합 없이 안정적으로 일반화**되고 있음을 시사한다.
 
#### 9. Why CatBoost Tuning Was Chosen as Next Step
 
**현재 상태**
 
```
XGB 0.74011  ← Optuna tuned
CAT 0.74005  ← baseline (여유 있음)
LGB 0.73987  ← baseline (여유 있음)
```
 
- Hybrid Ensemble / Fine Search → **현재 ceiling 안에서의 미세 최적화**
- CatBoost tuning → **ensemble ceiling 자체를 올릴 수 있는 작업**
 
> 따라서 Day4에서는 CatBoost Optuna tuning을 우선 진행하기로 결정하였다.
 
---
 
### Day4 — Planned Experiments
 
#### 11.1 CatBoost Optuna Tuning
 
**목적**
 
- CatBoost 단일 성능 자체를 끌어올리기
- ensemble diversity를 유지하면서 전체 baseline 향상
 
**예상 효과**
 
- CatBoost single model 성능 개선
- ensemble weight 재구성 가능성
- 최종 앙상블 ceiling 상승
 
#### 11.2 Ensemble Rebuild
 
튜닝된 CatBoost를 기반으로 Probability Ensemble / Rank Ensemble 재탐색
 
#### 11.3 Hybrid Ensemble
 
```
final_pred = a × probability_ensemble + (1 - a) × rank_ensemble
```
 
| 구성 요소 | 반영 정보 |
|-----------|-----------|
| Probability Ensemble | 절대적 확률 크기, calibration 정보, 모델 confidence |
| Rank Ensemble | 상대적 순위, ordering 정보, calibration 차이 완화 |
 
> Hybrid Ensemble은 확률 기반 정보와 순위 기반 정보를 동시에 활용하는 방식이다.
 
`a` 탐색 범위: `0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8`
 
#### 11.4 Fine Search Around Best Weights
 
```
XGB : 0.30 ~ 0.36
CAT : 0.38 ~ 0.44
step = 0.005
```
 
---
 
## 📈 Current Performance
 
| Version | ROC-AUC |
|---------|---------|
| v1 Baseline | 0.73997 |
| v2 Tuned | 0.74011 |
| Best Probability Ensemble | 0.740444 |
| **Best Rank Ensemble** | **0.740448** |
| **Public LB** | **0.74218** |
 
- Fold variance 매우 낮음 → 안정적인 validation 성능 확인
- OOF 대비 Public LB +0.00173 상승 → 안정적인 일반화 확인
 
---
 
## 📊 Model Visualization
 
### ROC Curve
 
| Baseline (v1) | Tuned (v2) |
|:---:|:---:|
| <img src="outputs/xgb_v1_roc_curve.png" width="400"> | <img src="outputs/xgb_v2_baseline_roc_curve.png" width="400"> |
 
### PR Curve
 
<p align="center">
  <img src="outputs/xgb_v1_pr_curve.png" width="500">
</p>
 
### Feature Importance
 
<p align="center">
  <img src="outputs/xgb_v1_feature_importance.png" width="700">
</p>
 
---
 
## 🔑 Key Insights
 
| # | Insight |
|---|---------|
| 1 | **Feature Engineering Saturation**: feature 추가 시 성능 향상이 제한적. single model ≈ 0.740 |
| 2 | **Ensemble Effect**: 앙상블 적용 시 안정적인 성능 향상. 0.74011 → 0.74044 |
| 3 | **Rank Ensemble Advantage**: Tree 기반 모델 조합에서 rank ensemble이 probability ensemble보다 소폭 우수 |
| 4 | **CatBoost Diversity**: 단일 성능 < XGB이지만 앙상블 weight는 가장 높음. error pattern diversity 기여 |
 
---
 
## 🚀 Future Work
 
### Day4 Priority
 
1. CatBoost Optuna Tuning
2. Ensemble Rebuild
3. Hybrid Ensemble (`a × prob + (1-a) × rank`)
4. Fine Weight Search
 
### Longer-Term Plan
 
```
Day5: Ensemble rebuild with tuned CatBoost
Day6: Hybrid Ensemble + Fine Search
Day7: LightGBM Optuna (optional)
```
 
### 🎯 Goal
 
```
ROC-AUC 0.740 → 0.743+
```
 
- 더 높은 일반화 성능 확보
- 안정적인 ensemble 기반 최종 모델 구축
 
---
 
## 📂 Repository Structure
 
```
yysop/
├── data/
├── outputs/
│   ├── xgb_v1_roc_curve.png
│   ├── xgb_v1_pr_curve.png
│   ├── xgb_v1_feature_importance.png
│   └── xgb_v2_reg_relax_roc_curve.png
│
├── src/
│   ├── eda.py
│   ├── xgb_kfold_v1.py
│   ├── xgb_kfold_v2.py
│   │
│   ├── catboost_kfold_v2.py
│   ├── lightgbm_kfold_v1.py
│   │
│   ├── ensemble_v2.py
│   └── ensemble_v4_rank_ensemble.py
│
└── README.md
```
 
---
 
## 🧑‍⚕️ Expected Impact
 
본 프로젝트는 난임 치료 데이터 분석을 통해 다음 분야에 기여할 수 있습니다.
 
- **임신 성공 확률 예측** — 시술 전 성공 가능성 추정
- **환자 맞춤형 치료 전략 지원** — 데이터 기반 개인화 치료 계획
- **의료진의 데이터 기반 의사결정 보조** — 임상 판단 지원 시스템
 
> AI 기반 예측 모델이 난임 치료의 성공률을 높이고  
> 환자의 신체적·정신적·경제적 부담을 줄이는 데 기여하는 것을 목표로 합니다.