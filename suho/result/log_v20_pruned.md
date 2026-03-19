# v20_pruned - v17 + 선별 도메인 피처
시각: 20260318_153226
============================================================

## [1] 데이터 로드
- train: (256351, 69), test: (90067, 68)
- 타겟: 0=190123, 1=66228, 양성비율=25.8%

## [2] 전처리 + 피처 엔지니어링
  전처리 중...
- 피처 수: 164, 카테고리: 20
- 카테고리(20개): ['시술 시기 코드', '시술 당시 나이', '시술 유형', '특정 시술 유형', '배란 유도 유형', '배아 생성 주요 이유', '총 시술 횟수', '클리닉 내 총 시술 횟수', 'IVF 시술 횟수', 'DI 시술 횟수', '총 임신 횟수', 'IVF 임신 횟수', 'DI 임신 횟수', '총 출산 횟수', 'IVF 출산 횟수', 'DI 출산 횟수', '난자 출처', '정자 출처', '난자 기증자 나이', '정자 기증자 나이']
- v17 frozen 피처 (8개): ['is_frozen_transfer', 'thaw_to_transfer_ratio', 'frozen_x_age', 'frozen_x_clinic_exp', 'frozen_x_stored', 'frozen_day_interaction', 'frozen_single_embryo', 'frozen_x_day5plus']
- 팀원 XGB 피처 (29개): ['배아생성효율', 'ICSI수정효율', '배아이식비율', '배아저장비율', '난자활용률', '난자대비이식배아수', '이식배아수_구간', '전체임신률', 'IVF임신률', 'DI임신률', '임신유지율', 'IVF임신유지율', '총실패횟수', 'IVF실패횟수', '반복IVF실패_여부', '클리닉집중도', 'IVF시술비율', '임신경험있음', '출산경험있음', '나이_제곱', '나이_임상구간', '고령_여부', '초고령_여부', '극고령_여부', '나이X총시술', '나이XIVF실패', '나이XIVF임신률', '초고령X반복실패', '복합위험도점수']
- v20 pruned domain 피처 (21개): ['male_factor_score', 'female_factor_score', 'severe_sperm_factor_flag', 'unexplained_only_flag', 'donor_any_flag', 'effective_oocyte_age', 'advanced_oocyte_age_flag', 'advanced_age_autologous_flag', 'advanced_age_donor_oocyte_flag', 'retrieved_oocytes_total', 'insemination_rate_from_retrieved', 'fertilization_rate_all', 'embryo_yield_per_retrieved', 'transferable_embryos', 'transferable_rate', 'freeze_to_transfer_ratio_domain', 'thaw_survival_proxy', 'same_or_early_transfer_like', 'extended_culture_flag_domain', 'repeat_failure_x_transferable', 'ivf_failure_x_effective_oocyte_age']
- 제거된 상수 컬럼 (2개): ['불임 원인 - 여성 요인', '난자 채취 경과일']
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
    Fold 1: AUC=0.7389, iter=1477, 소요=1.9분 [1/25]
    Fold 2: AUC=0.7435, iter=2502, 소요=2.9분 [2/25]
    Fold 3: AUC=0.7408, iter=2180, 소요=2.8분 [3/25]
    Fold 4: AUC=0.7389, iter=1494, 소요=1.9분 [4/25]
    Fold 5: AUC=0.7408, iter=1938, 소요=2.5분 [5/25]
  Seed 42 OOF AUC: 0.7406

  --- Seed 2026 (2/5) ---
    Fold 1: AUC=0.7374, iter=2248, 소요=2.6분 [6/25]
    Fold 2: AUC=0.7402, iter=1877, 소요=2.1분 [7/25]
    Fold 3: AUC=0.7382, iter=1670, 소요=2.0분 [8/25]
    Fold 4: AUC=0.7471, iter=2133, 소요=2.5분 [9/25]
    Fold 5: AUC=0.7397, iter=2015, 소요=2.4분 [10/25]
  Seed 2026 OOF AUC: 0.7405

  --- Seed 2604 (3/5) ---
    Fold 1: AUC=0.7402, iter=1887, 소요=2.3분 [11/25]
    Fold 2: AUC=0.7385, iter=2010, 소요=2.4분 [12/25]
    Fold 3: AUC=0.7420, iter=1564, 소요=2.0분 [13/25]
    Fold 4: AUC=0.7384, iter=2500, 소요=3.0분 [14/25]
    Fold 5: AUC=0.7425, iter=1887, 소요=2.3분 [15/25]
  Seed 2604 OOF AUC: 0.7403

  --- Seed 123 (4/5) ---
    Fold 1: AUC=0.7385, iter=2818, 소요=3.1분 [16/25]
    Fold 2: AUC=0.7410, iter=1509, 소요=2.1분 [17/25]
    Fold 3: AUC=0.7397, iter=1691, 소요=2.2분 [18/25]
    Fold 4: AUC=0.7412, iter=2741, 소요=3.2분 [19/25]
    Fold 5: AUC=0.7423, iter=1898, 소요=2.3분 [20/25]
  Seed 123 OOF AUC: 0.7405

  --- Seed 777 (5/5) ---
    Fold 1: AUC=0.7383, iter=1924, 소요=2.3분 [21/25]
    Fold 2: AUC=0.7421, iter=2156, 소요=2.5분 [22/25]
    Fold 3: AUC=0.7382, iter=2067, 소요=2.4분 [23/25]
    Fold 4: AUC=0.7417, iter=1898, 소요=2.4분 [24/25]
    Fold 5: AUC=0.7430, iter=1721, 소요=2.1분 [25/25]
  Seed 777 OOF AUC: 0.7406

  === CatBoost 5-seed OOF AUC: 0.740810 ===
  개별 seed: ['0.7406', '0.7405', '0.7403', '0.7405', '0.7406']
  학습 소요: 60.2분

## [5] 그룹별 AUC 분석
  transfer: AUC=0.6749 (n=213516, pos=30.6%)
  non-transfer: AUC=0.9408 (n=42835, pos=2.0%)
  frozen transfer: AUC=0.6200 (n=37281, pos=24.7%)
  fresh transfer: AUC=0.6777 (n=176235, pos=31.9%)
  DI: AUC=0.6850 (n=6291, pos=12.9%)
  donor egg: AUC=0.6798 (n=15769, pos=31.5%)
  age_만18-34세: AUC=0.7040 (n=102476, pos=32.3%)
  age_만35-37세: AUC=0.7028 (n=57780, pos=27.8%)
  age_만38-39세: AUC=0.7104 (n=39247, pos=21.7%)
  age_만40-42세: AUC=0.7379 (n=37348, pos=15.9%)
  age_만43-44세: AUC=0.8304 (n=12253, pos=11.8%)
  age_만45-50세: AUC=0.8325 (n=6918, pos=16.8%)

## [6] 종합 평가지표
  OOF AUC:      0.740810
  OOF LogLoss:  0.493269
  OOF AP:       0.452123

### Threshold별 분류 지표
      Th      Acc     Prec      Rec       F1     Spec
    0.20   0.5129   0.3403   0.9432   0.5001   0.3630
    0.25   0.5608   0.3589   0.8898   0.5114   0.4462
    0.30   0.6116   0.3811   0.8071   0.5178   0.5435
    0.35   0.6586   0.4062   0.6955   0.5128   0.6458
    0.40   0.6979   0.4348   0.5650   0.4914   0.7442
    0.45   0.7255   0.4654   0.4212   0.4422   0.8315
    0.50   0.7396   0.4932   0.2902   0.3654   0.8961
  최적 F1: 0.5178 (threshold=0.3)

## [7] 제출 파일
- 파일: C:\Users\ico26\oz\Infertility-Success-Classifier\suho\result\sample_submission_v20_pruned_20260318_153226.csv
- 확률: mean=0.3030, std=0.1804, min=0.000490, max=0.8019

## [8] CatBoost 평균 피처 중요도 (상위 40)
  1. transfer_intensity: 11.2502
  2. 이식된 배아 수: 9.9804
  3. age_transfer_interaction: 9.6797
  4. 실제이식여부: 7.9152
  5. 이식배아수_구간 ★team: 6.9482
  6. egg_sperm_combo: 2.1230
  7. effective_oocyte_age ★domain: 2.0497
  8. advanced_age_autologous_flag ★domain: 1.9524
  9. transfer_day_cat: 1.8862
  10. 배아저장비율 ★team: 1.6409
  11. transferable_embryos ★domain: 1.4675
  12. freeze_to_transfer_ratio_domain ★domain: 1.4630
  13. ivf_storage_ratio: 1.4381
  14. 배아 이식 경과일: 1.3981
  15. donor_any_flag ★domain: 1.2977
  16. age_x_single_transfer: 1.2519
  17. embryo_day_interaction: 1.2073
  18. retrieved_oocytes_total ★domain: 1.1751
  19. ivf_transfer_ratio: 1.1425
  20. 시술 시기 코드 [cat]: 1.1290
  21. advanced_oocyte_age_flag ★domain: 1.1211
  22. 배아_잉여율: 1.1170
  23. total_embryo_ratio: 1.0526
  24. 배아 생성 주요 이유 [cat]: 1.0164
  25. 배란 유도 유형 [cat]: 1.0026
  26. 배아이식비율 ★team: 0.9727
  27. 수집된 신선 난자 수: 0.9539
  28. 저장된 배아 수: 0.8427
  29. 난자 출처 [cat]: 0.7935
  30. insemination_rate_from_retrieved ★domain: 0.7882
  31. 총 생성 배아 수: 0.7528
  32. 난자대비이식배아수 ★team: 0.7044
  33. 복합위험도점수 ★team: 0.6654
  34. 정자 출처 [cat]: 0.6079
  35. fresh_transfer_ratio: 0.6005
  36. 배아_이용률: 0.5880
  37. 신선 배아 사용 여부: 0.5202
  38. transferable_rate ★domain: 0.4950
  39. age_x_day5plus: 0.4930
  40. fertilization_rate_all ★domain: 0.4519

### v20 pruned domain 피처 중요도
  effective_oocyte_age: 2.0497 (#7)
  advanced_age_autologous_flag: 1.9524 (#8)
  transferable_embryos: 1.4675 (#11)
  freeze_to_transfer_ratio_domain: 1.4630 (#12)
  donor_any_flag: 1.2977 (#15)
  retrieved_oocytes_total: 1.1751 (#18)
  advanced_oocyte_age_flag: 1.1211 (#21)
  insemination_rate_from_retrieved: 0.7882 (#30)
  transferable_rate: 0.4950 (#38)
  fertilization_rate_all: 0.4519 (#40)
  same_or_early_transfer_like: 0.4335 (#43)
  ivf_failure_x_effective_oocyte_age: 0.3813 (#47)
  repeat_failure_x_transferable: 0.3690 (#48)
  thaw_survival_proxy: 0.3157 (#55)
  embryo_yield_per_retrieved: 0.3113 (#56)
  extended_culture_flag_domain: 0.3094 (#57)
  female_factor_score: 0.1586 (#79)
  male_factor_score: 0.1330 (#85)
  advanced_age_donor_oocyte_flag: 0.0330 (#128)
  unexplained_only_flag: 0.0244 (#135)
  severe_sperm_factor_flag: 0.0000 (#162)

### v17 frozen 피처 중요도
  is_frozen_transfer: 0.1241 (#91)
  thaw_to_transfer_ratio: 0.2577 (#66)
  frozen_x_age: 0.0565 (#118)
  frozen_x_clinic_exp: 0.0880 (#102)
  frozen_x_stored: 0.0386 (#125)
  frozen_day_interaction: 0.0766 (#108)
  frozen_single_embryo: 0.0108 (#148)
  frozen_x_day5plus: 0.0003 (#159)

### 중요도 0 피처 (1개): ['난자 해동 경과일']

## [9] 버전 비교
| 버전 | 모델 | OOF AUC | Test AUC | 비고 |
|------|------|---------|----------|------|
| v1 | CB원본 | 0.7403 | - | 베이스라인 |
| v9 | XGB+CB 3seed | 0.7405 | 0.74166 | +IVF/DI |
| v15 | CB 5seed+Optuna | 0.7407 | 0.74169 | EDA+HP |
| v17 | CB 5seed | 0.7408 | TBD | 팀원피처+frozen+CW |
| v19 | CB 5seed | 0.7408 | TBD | +domain 81개 |
| **v20_pruned** | **CB 5seed** | **0.7408** | **TBD** | **+domain pruned 21개** |

============================================================
## 최종 요약
============================================================
- v20-pruned 핵심 변경:
  1. 팀원 XGB 의료 파생변수 29개 유지
  2. frozen transfer 전용 피처 8개 유지
  3. 도메인 피처를 81개 → 21개로 선별
  4. broad expansion 대신 prune-only 비교
  5. categorical 자동 탐지 유지
  6. CatBoost importance를 25개 모델 평균으로 집계
- CB 5-seed OOF AUC: 0.740810
  seed별: ['0.7406', '0.7405', '0.7403', '0.7405', '0.7406']
- OOF LogLoss: 0.493269
- OOF AP: 0.452123
- 피처 수: 164 (cat=20)
- 상수 제거: 2개
- 데이터 누수: 현재 제공 컬럼 기준 직접적 누수 피처는 미사용
- 총 소요: 60.4분
- 로그: C:\Users\ico26\oz\Infertility-Success-Classifier\suho\result\log_v20_pruned.md
============================================================