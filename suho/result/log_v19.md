# v19 - v17 + 도메인 피처 확장
시각: 20260318_110820
============================================================

## [1] 데이터 로드
- train: (256351, 69), test: (90067, 68)
- 타겟: 0=190123, 1=66228, 양성비율=25.8%

## [2] 전처리 + 피처 엔지니어링
  전처리 중...
- 피처 수: 226, 카테고리: 20
- 카테고리(20개): ['시술 시기 코드', '시술 당시 나이', '시술 유형', '특정 시술 유형', '배란 유도 유형', '배아 생성 주요 이유', '총 시술 횟수', '클리닉 내 총 시술 횟수', 'IVF 시술 횟수', 'DI 시술 횟수', '총 임신 횟수', 'IVF 임신 횟수', 'DI 임신 횟수', '총 출산 횟수', 'IVF 출산 횟수', 'DI 출산 횟수', '난자 출처', '정자 출처', '난자 기증자 나이', '정자 기증자 나이']
- v17 frozen 피처 (8개): ['is_frozen_transfer', 'thaw_to_transfer_ratio', 'frozen_x_age', 'frozen_x_clinic_exp', 'frozen_x_stored', 'frozen_day_interaction', 'frozen_single_embryo', 'frozen_x_day5plus']
- 팀원 XGB 피처 (29개): ['배아생성효율', 'ICSI수정효율', '배아이식비율', '배아저장비율', '난자활용률', '난자대비이식배아수', '이식배아수_구간', '전체임신률', 'IVF임신률', 'DI임신률', '임신유지율', 'IVF임신유지율', '총실패횟수', 'IVF실패횟수', '반복IVF실패_여부', '클리닉집중도', 'IVF시술비율', '임신경험있음', '출산경험있음', '나이_제곱', '나이_임상구간', '고령_여부', '초고령_여부', '극고령_여부', '나이X총시술', '나이XIVF실패', '나이XIVF임신률', '초고령X반복실패', '복합위험도점수']
- v19 domain 피처 (81개): ['male_factor_score', 'female_factor_score', 'couple_factor_score', 'sperm_issue_score', 'tubal_factor_flag', 'ovulatory_factor_flag', 'cervical_factor_flag', 'endometriosis_flag', 'multi_cause_flag', 'unexplained_only_flag', 'male_female_cause_gap', 'severe_sperm_factor_flag', 'donor_oocyte_flag', 'autologous_oocyte_flag', 'donor_sperm_flag', 'partner_sperm_flag', 'donor_any_flag', 'donor_both_gametes_flag', 'surrogacy_flag_num', 'effective_oocyte_age', 'advanced_oocyte_age_flag', 'very_advanced_oocyte_age_flag', 'recipient_minus_oocyte_age', 'young_donor_egg_flag', 'older_sperm_donor_flag', 'advanced_age_autologous_flag', 'advanced_age_donor_oocyte_flag', 'pgt_any_flag', 'genetic_test_both_flag', 'fresh_only_flag', 'frozen_only_flag', 'fresh_frozen_mix_flag', 'single_transfer_plan_flag', 'stimulated_cycle_flag', 'pgt_with_donor_flag', 'donor_any_x_frozen', 'retrieved_oocytes_total', 'fresh_oocyte_share', 'cryo_oocyte_share', 'insemination_rate_from_retrieved'] ...
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
    Fold 1: AUC=0.7388, iter=1447, 소요=1.8분 [1/25]
    Fold 2: AUC=0.7433, iter=2297, 소요=2.6분 [2/25]
    Fold 3: AUC=0.7408, iter=1879, 소요=2.2분 [3/25]
    Fold 4: AUC=0.7389, iter=1542, 소요=1.9분 [4/25]
    Fold 5: AUC=0.7410, iter=1951, 소요=2.3분 [5/25]
  Seed 42 OOF AUC: 0.7405

  --- Seed 2026 (2/5) ---
    Fold 1: AUC=0.7373, iter=1937, 소요=2.3분 [6/25]
    Fold 2: AUC=0.7401, iter=1674, 소요=2.0분 [7/25]
    Fold 3: AUC=0.7381, iter=1665, 소요=2.0분 [8/25]
    Fold 4: AUC=0.7473, iter=2242, 소요=2.5분 [9/25]
    Fold 5: AUC=0.7397, iter=2147, 소요=2.5분 [10/25]
  Seed 2026 OOF AUC: 0.7405

  --- Seed 2604 (3/5) ---
    Fold 1: AUC=0.7402, iter=1626, 소요=1.9분 [11/25]
    Fold 2: AUC=0.7385, iter=1665, 소요=2.0분 [12/25]
    Fold 3: AUC=0.7420, iter=1514, 소요=1.9분 [13/25]
    Fold 4: AUC=0.7386, iter=1936, 소요=2.3분 [14/25]
    Fold 5: AUC=0.7424, iter=1694, 소요=2.0분 [15/25]
  Seed 2604 OOF AUC: 0.7403

  --- Seed 123 (4/5) ---
    Fold 1: AUC=0.7385, iter=2213, 소요=2.6분 [16/25]
    Fold 2: AUC=0.7410, iter=1384, 소요=1.8분 [17/25]
    Fold 3: AUC=0.7398, iter=1647, 소요=2.1분 [18/25]
    Fold 4: AUC=0.7410, iter=1864, 소요=2.2분 [19/25]
    Fold 5: AUC=0.7421, iter=1928, 소요=2.3분 [20/25]
  Seed 123 OOF AUC: 0.7405

  --- Seed 777 (5/5) ---
    Fold 1: AUC=0.7383, iter=2037, 소요=2.4분 [21/25]
    Fold 2: AUC=0.7421, iter=1571, 소요=2.1분 [22/25]
    Fold 3: AUC=0.7382, iter=2426, 소요=2.7분 [23/25]
    Fold 4: AUC=0.7415, iter=1506, 소요=1.9분 [24/25]
    Fold 5: AUC=0.7429, iter=1836, 소요=2.2분 [25/25]
  Seed 777 OOF AUC: 0.7406

  === CatBoost 5-seed OOF AUC: 0.740760 ===
  개별 seed: ['0.7405', '0.7405', '0.7403', '0.7405', '0.7406']
  학습 소요: 54.3분

## [5] 그룹별 AUC 분석
  transfer: AUC=0.6749 (n=213516, pos=30.6%)
  non-transfer: AUC=0.9401 (n=42835, pos=2.0%)
  frozen transfer: AUC=0.6194 (n=37281, pos=24.7%)
  fresh transfer: AUC=0.6777 (n=176235, pos=31.9%)
  DI: AUC=0.6844 (n=6291, pos=12.9%)
  donor egg: AUC=0.6797 (n=15769, pos=31.5%)
  age_만18-34세: AUC=0.7039 (n=102476, pos=32.3%)
  age_만35-37세: AUC=0.7027 (n=57780, pos=27.8%)
  age_만38-39세: AUC=0.7103 (n=39247, pos=21.7%)
  age_만40-42세: AUC=0.7380 (n=37348, pos=15.9%)
  age_만43-44세: AUC=0.8305 (n=12253, pos=11.8%)
  age_만45-50세: AUC=0.8319 (n=6918, pos=16.8%)

## [6] 종합 평가지표
  OOF AUC:      0.740760
  OOF LogLoss:  0.493299
  OOF AP:       0.452231

### Threshold별 분류 지표
      Th      Acc     Prec      Rec       F1     Spec
    0.20   0.5125   0.3401   0.9429   0.4998   0.3626
    0.25   0.5613   0.3590   0.8890   0.5115   0.4471
    0.30   0.6114   0.3810   0.8074   0.5177   0.5431
    0.35   0.6582   0.4058   0.6961   0.5127   0.6450
    0.40   0.6970   0.4338   0.5662   0.4913   0.7426
    0.45   0.7256   0.4657   0.4215   0.4425   0.8316
    0.50   0.7398   0.4941   0.2911   0.3664   0.8962
  최적 F1: 0.5177 (threshold=0.3)

## [7] 제출 파일
- 파일: C:\Users\ico26\oz\Infertility-Success-Classifier\suho\result\sample_submission_v19_20260318_110820.csv
- 확률: mean=0.3030, std=0.1803, min=0.000464, max=0.7988

## [8] CatBoost 평균 피처 중요도 (상위 40)
  1. oocyte_age_transfer_intensity ★domain: 11.9629
  2. transfer_intensity: 7.2725
  3. age_transfer_interaction: 6.7662
  4. 실제이식여부: 5.6872
  5. transfer_attempt_signal ★domain: 5.1216
  6. 이식된 배아 수: 4.4660
  7. 이식배아수_구간 ★team: 4.4402
  8. effective_oocyte_age ★domain: 2.0190
  9. 배아저장비율 ★team: 1.9801
  10. 배아_잉여율: 1.9454
  11. ivf_storage_ratio: 1.6720
  12. transfer_day_cat: 1.4082
  13. oocyte_age_embryo_creation_pressure ★domain: 1.3961
  14. egg_sperm_combo: 1.3101
  15. lab_total_elapsed_proxy ★domain: 1.2548
  16. ivf_transfer_ratio: 1.1831
  17. advanced_age_autologous_flag ★domain: 1.1302
  18. advanced_oocyte_age_flag ★domain: 1.1259
  19. 시술 시기 코드 [cat]: 1.0560
  20. 배아 생성 주요 이유 [cat]: 0.9950
  21. retrieved_oocytes_total ★domain: 0.9500
  22. 배아 이식 경과일: 0.9063
  23. age_x_single_transfer: 0.8922
  24. total_embryo_ratio: 0.8857
  25. embryo_day_interaction: 0.8263
  26. donor_any_flag ★domain: 0.8167
  27. transferable_embryos ★domain: 0.8040
  28. source_profile_code ★domain: 0.7962
  29. 수집된 신선 난자 수: 0.7914
  30. 난자 출처 [cat]: 0.7386
  31. fresh_transfer_ratio: 0.6845
  32. 복합위험도점수 ★team: 0.6807
  33. 배아_이용률: 0.6617
  34. 난자대비이식배아수 ★team: 0.6589
  35. freeze_to_transfer_ratio_domain ★domain: 0.6534
  36. 배아이식비율 ★team: 0.6522
  37. 배란 유도 유형 [cat]: 0.6497
  38. same_or_early_transfer_like ★domain: 0.5643
  39. donor_sperm_flag ★domain: 0.5631
  40. very_advanced_oocyte_age_flag ★domain: 0.5592

### v19 domain 피처 중요도 (상위 30)
  oocyte_age_transfer_intensity: 11.9629 (#1)
  transfer_attempt_signal: 5.1216 (#5)
  effective_oocyte_age: 2.0190 (#8)
  oocyte_age_embryo_creation_pressure: 1.3961 (#13)
  lab_total_elapsed_proxy: 1.2548 (#15)
  advanced_age_autologous_flag: 1.1302 (#17)
  advanced_oocyte_age_flag: 1.1259 (#18)
  retrieved_oocytes_total: 0.9500 (#21)
  donor_any_flag: 0.8167 (#26)
  transferable_embryos: 0.8040 (#27)
  source_profile_code: 0.7962 (#28)
  freeze_to_transfer_ratio_domain: 0.6534 (#35)
  same_or_early_transfer_like: 0.5643 (#38)
  donor_sperm_flag: 0.5631 (#39)
  very_advanced_oocyte_age_flag: 0.5592 (#40)
  insemination_rate_from_retrieved: 0.5493 (#41)
  partner_sperm_flag: 0.5455 (#42)
  fresh_oocyte_share: 0.5005 (#44)
  repeat_failure_x_transferable: 0.4165 (#48)
  ivf_failure_x_effective_oocyte_age: 0.3617 (#50)
  fertilization_rate_all: 0.3540 (#51)
  thaw_survival_proxy: 0.3226 (#55)
  autologous_oocyte_flag: 0.2867 (#59)
  transferable_rate: 0.2840 (#61)
  extended_culture_flag_domain: 0.2652 (#65)
  fresh_only_flag: 0.2582 (#67)
  advanced_oocyte_age_x_transferable: 0.2433 (#69)
  embryo_yield_per_retrieved: 0.2388 (#70)
  embryo_wastage_rate: 0.2073 (#77)
  icsi_share_domain: 0.1936 (#80)

### v17 frozen 피처 중요도
  is_frozen_transfer: 0.1374 (#96)
  thaw_to_transfer_ratio: 0.3394 (#53)
  frozen_x_age: 0.0489 (#152)
  frozen_x_clinic_exp: 0.0749 (#133)
  frozen_x_stored: 0.0549 (#146)
  frozen_day_interaction: 0.0836 (#123)
  frozen_single_embryo: 0.0104 (#189)
  frozen_x_day5plus: 0.0001 (#215)

### 중요도 0 피처 (4개): ['난자 채취 경과일', '난자 해동 경과일', '불임 원인 - 여성 요인', '불임 원인 - 정자 면역학적 요인']

## [9] 버전 비교
| 버전 | 모델 | OOF AUC | Test AUC | 비고 |
|------|------|---------|----------|------|
| v1 | CB원본 | 0.7403 | - | 베이스라인 |
| v9 | XGB+CB 3seed | 0.7405 | 0.74166 | +IVF/DI |
| v15 | CB 5seed+Optuna | 0.7407 | 0.74169 | EDA+HP |
| v17 | CB 5seed | 0.7408 | TBD | 팀원피처+frozen+CW |
| **v19** | **CB 5seed** | **0.7408** | **TBD** | **+domain features** |

============================================================
## 최종 요약
============================================================
- v19 핵심 변경:
  1. 팀원 XGB 의료 파생변수 29개 유지
  2. frozen transfer 전용 피처 8개 유지
  3. 도메인 의료 피처 81개 추가
  4. boolean/flag 컬럼 robust 정규화
  5. categorical 자동 탐지
  6. CatBoost importance를 25개 모델 평균으로 집계
- CB 5-seed OOF AUC: 0.740760
  seed별: ['0.7405', '0.7405', '0.7403', '0.7405', '0.7406']
- OOF LogLoss: 0.493299
- OOF AP: 0.452231
- 피처 수: 226 (cat=20)
- 데이터 누수: 현재 제공 컬럼 기준 직접적 누수 피처는 미사용
- 총 소요: 54.5분
- 로그: C:\Users\ico26\oz\Infertility-Success-Classifier\suho\result\log_v19.md
============================================================