# v17 - 팀원 XGB 피처 + frozen 피처 + class_weights + CB 5-seed
시각: 20260317_171824
============================================================

## [1] 데이터 로드
- train: (256351, 69), test: (90067, 68)
- 타겟: 0=190123, 1=66228, 양성비율=25.8%

## [2] 전처리 + 피처 엔지니어링
  전처리 중...
- 피처 수: 145, 카테고리: 20
- 카테고리: ['시술 시기 코드', '시술 당시 나이', '시술 유형', '특정 시술 유형', '배란 유도 유형', '배아 생성 주요 이유', '총 시술 횟수', '클리닉 내 총 시술 횟수', 'IVF 시술 횟수', 'DI 시술 횟수', '총 임신 횟수', 'IVF 임신 횟수', 'DI 임신 횟수', '총 출산 횟수', 'IVF 출산 횟수', 'DI 출산 횟수', '난자 출처', '정자 출처', '난자 기증자 나이', '정자 기증자 나이']
- v17 frozen 신규 피처 (8개): ['is_frozen_transfer', 'thaw_to_transfer_ratio', 'frozen_x_age', 'frozen_x_clinic_exp', 'frozen_x_stored', 'frozen_day_interaction', 'frozen_single_embryo', 'frozen_x_day5plus']
- 팀원 XGB 피처 (29개): ['배아생성효율', 'ICSI수정효율', '배아이식비율', '배아저장비율', '난자활용률', '난자대비이식배아수', '이식배아수_구간', '전체임신률', 'IVF임신률', 'DI임신률', '임신유지율', 'IVF임신유지율', '총실패횟수', 'IVF실패횟수', '반복IVF실패_여부', '클리닉집중도', 'IVF시술비율', '임신경험있음', '출산경험있음', '나이_제곱', '나이_임상구간', '고령_여부', '초고령_여부', '극고령_여부', '나이X총시술', '나이XIVF실패', '나이XIVF임신률', '초고령X반복실패', '복합위험도점수']
- 피처 엔지니어링 소요: 0.1분

## [3] subgroup 통계
- 이식: 213516건, 양성률=30.6%
- 비이식: 42835건, 양성률=1.96%
- frozen transfer: 37281건, 양성률=24.7%

## [4] CatBoost 5-seed 앙상블
### 파라미터:
  iterations: 8000
  learning_rate: 0.00837420773913813
  depth: 8
  l2_leaf_reg: 9.013710971500913
  min_data_in_leaf: 47
  random_strength: 1.9940254840418206
  bagging_temperature: 0.7125808252032847
  border_count: 64
  eval_metric: AUC
  loss_function: Logloss
  od_type: Iter
  od_wait: 400
  allow_writing_files: False
  class_weights: {0: 1.0, 1: 1.3}
### Seeds: [42, 2026, 2604, 123, 777]
### Task type: GPU

  --- Seed 42 (1/5) ---
    Fold 1: AUC=0.7386, iter=1602, 소요=1.9분 [1/25]
    Fold 2: AUC=0.7436, iter=2954, 소요=3.2분 [2/25]
    Fold 3: AUC=0.7406, iter=2124, 소요=2.5분 [3/25]
    Fold 4: AUC=0.7388, iter=1850, 소요=2.3분 [4/25]
    Fold 5: AUC=0.7410, iter=1832, 소요=2.3분 [5/25]
  Seed 42 OOF AUC: 0.7405

  --- Seed 2026 (2/5) ---
    Fold 1: AUC=0.7374, iter=2696, 소요=3.2분 [6/25]
    Fold 2: AUC=0.7401, iter=2164, 소요=2.6분 [7/25]
    Fold 3: AUC=0.7380, iter=1804, 소요=2.3분 [8/25]
    Fold 4: AUC=0.7473, iter=2320, 소요=2.8분 [9/25]
    Fold 5: AUC=0.7399, iter=1838, 소요=2.4분 [10/25]
  Seed 2026 OOF AUC: 0.7405

  --- Seed 2604 (3/5) ---
    Fold 1: AUC=0.7406, iter=2378, 소요=2.9분 [11/25]
    Fold 2: AUC=0.7385, iter=1812, 소요=2.3분 [12/25]
    Fold 3: AUC=0.7419, iter=2019, 소요=2.3분 [13/25]
    Fold 4: AUC=0.7385, iter=2500, 소요=2.8분 [14/25]
    Fold 5: AUC=0.7425, iter=1933, 소요=2.4분 [15/25]
  Seed 2604 OOF AUC: 0.7404

  --- Seed 123 (4/5) ---
    Fold 1: AUC=0.7384, iter=1907, 소요=2.3분 [16/25]
    Fold 2: AUC=0.7409, iter=1892, 소요=2.3분 [17/25]
    Fold 3: AUC=0.7400, iter=1900, 소요=2.3분 [18/25]
    Fold 4: AUC=0.7411, iter=2947, 소요=3.3분 [19/25]
    Fold 5: AUC=0.7424, iter=1777, 소요=2.2분 [20/25]
  Seed 123 OOF AUC: 0.7405

  --- Seed 777 (5/5) ---
    Fold 1: AUC=0.7385, iter=2175, 소요=2.6분 [21/25]
    Fold 2: AUC=0.7421, iter=2064, 소요=2.5분 [22/25]
    Fold 3: AUC=0.7380, iter=2218, 소요=2.6분 [23/25]
    Fold 4: AUC=0.7416, iter=1355, 소요=1.8분 [24/25]
    Fold 5: AUC=0.7430, iter=1763, 소요=2.2분 [25/25]
  Seed 777 OOF AUC: 0.7406

  === CatBoost 5-seed OOF AUC: 0.740816 ===
  개별 seed: ['0.7405', '0.7405', '0.7404', '0.7405', '0.7406']
  학습 소요: 62.1분

## [5] 그룹별 AUC 분석
  transfer: AUC=0.6749 (n=213516, pos=30.6%)
  non-transfer: AUC=0.9402 (n=42835, pos=2.0%)
  frozen transfer: AUC=0.6201 (n=37281, pos=24.7%)
  fresh transfer: AUC=0.6777 (n=176235, pos=31.9%)
  DI: AUC=0.6843 (n=6291, pos=12.9%)
  donor egg: AUC=0.6794 (n=15769, pos=31.5%)
  age_만18-34세: AUC=0.7040 (n=102476, pos=32.3%)
  age_만35-37세: AUC=0.7027 (n=57780, pos=27.8%)
  age_만38-39세: AUC=0.7105 (n=39247, pos=21.7%)
  age_만40-42세: AUC=0.7380 (n=37348, pos=15.9%)
  age_만43-44세: AUC=0.8304 (n=12253, pos=11.8%)
  age_만45-50세: AUC=0.8334 (n=6918, pos=16.8%)

## [6] 종합 평가지표
  OOF AUC:      0.740816
  OOF Log Loss: 0.493261
  OOF AP:       0.452018

### Threshold별 분류 지표
      Th      Acc     Prec      Rec       F1     Spec
    0.20   0.5128   0.3403   0.9437   0.5002   0.3627
    0.25   0.5611   0.3590   0.8898   0.5116   0.4466
    0.30   0.6118   0.3813   0.8075   0.5180   0.5436
    0.35   0.6585   0.4061   0.6955   0.5128   0.6456
    0.40   0.6980   0.4349   0.5648   0.4914   0.7444
    0.45   0.7256   0.4656   0.4204   0.4418   0.8319
    0.50   0.7397   0.4935   0.2912   0.3663   0.8959
  최적 F1: 0.5180 (threshold=0.3)

## [7] 제출 파일
- 파일: C:\Users\ico26\oz\Infertility-Success-Classifier\suho\result\sample_submission_v17_20260317_171824.csv
- 확률: mean=0.3030, std=0.1804, min=0.000584, max=0.8055

## [8] CatBoost 피처 중요도 (상위 30)
  1. transfer_intensity: 15.02
  2. age_transfer_interaction: 11.05
  3. 실제이식여부: 8.56
  4. 이식된 배아 수: 7.92
  5. 이식배아수_구간 ★team: 6.24
  6. egg_sperm_combo: 3.22
  7. 배아_잉여율: 2.83
  8. 수집된 신선 난자 수: 2.83
  9. 난자 출처 [cat]: 2.00
  10. transfer_day_cat: 1.93
  11. 배아저장비율 ★team: 1.83
  12. ivf_storage_ratio: 1.77
  13. 저장된 배아 수: 1.76
  14. 시술 당시 나이_num: 1.45
  15. age_x_single_transfer: 1.42
  16. 난자대비이식배아수 ★team: 1.33
  17. ivf_transfer_ratio: 1.31
  18. 복합위험도점수 ★team: 1.15
  19. 나이_제곱 ★team: 1.11
  20. 시술 시기 코드 [cat]: 1.08
  21. 배아 이식 경과일: 1.07
  22. total_embryo_ratio: 1.06
  23. 배아 생성 주요 이유 [cat]: 0.99
  24. 배아_이용률: 0.91
  25. embryo_day_interaction: 0.90
  26. 배란 유도 유형 [cat]: 0.89
  27. 나이_임상구간 ★team: 0.88
  28. 시술 당시 나이 [cat]: 0.78
  29. 총 생성 배아 수: 0.72
  30. age_x_day5plus: 0.68

### v17 frozen 피처 중요도
  is_frozen_transfer: 0.12 (#70)
  thaw_to_transfer_ratio: 0.49 (#35)
  frozen_x_age: 0.05 (#106)
  frozen_x_clinic_exp: 0.09 (#85)
  frozen_x_stored: 0.06 (#98)
  frozen_day_interaction: 0.10 (#81)
  frozen_single_embryo: 0.00 (#133)
  frozen_x_day5plus: 0.00 (#137)

### 중요도 0 피처 (5개): ['불임 원인 - 여성 요인', '불임 원인 - 정자 면역학적 요인', '난자 채취 경과일', '난자 해동 경과일', '불임 원인 - 정자 형태']

## [9] 버전 비교
| 버전 | 모델 | OOF AUC | Test AUC | 비고 |
|------|------|---------|----------|------|
| v1 | CB원본 | 0.7403 | - | 베이스라인 |
| v9 | XGB+CB 3seed | 0.7405 | 0.74166 | +IVF/DI |
| v12 | CB+XGB+Tab | 0.7406 | - | 3모델 |
| v15 | CB 5seed+Optuna | 0.7407 | 0.74169 | EDA+HP |
| v16 | Multi-model | 0.7407 | - | subgroup |
| **v17** | **CB 5seed** | **0.7408** | **TBD** | **팀원피처+frozen+CW** |

============================================================
## 최종 요약
============================================================
- v17 핵심 변경:
  1. 팀원 XGB 의료 파생변수 29개 통합
  2. frozen transfer 전용 피처 8개 추가
  3. class_weights {0:1.0, 1:1.3} 적용
  4. 카테고리 20개 복원 (횟수형 원본 유지)
  5. CatBoost 5-seed 앙상블 (seed별 다른 fold)
- CB 5-seed OOF AUC: 0.740816
  seed별: ['0.7405', '0.7405', '0.7404', '0.7405', '0.7406']
- OOF LogLoss: 0.493261
- OOF AP: 0.452018
- 피처 수: 145 (cat=20)
- 데이터 누수: 없음
- 총 소요: 62.3분
- 로그: C:\Users\ico26\oz\Infertility-Success-Classifier\suho\result\log_v17.md
============================================================