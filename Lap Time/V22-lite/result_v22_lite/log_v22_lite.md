# v22_lite - global blend + transfer expert + frozen expert
시각: 20260319_220239
============================================================

## [1] 데이터 로드
- train: (256351, 69), test: (90067, 68)
- 타겟: 0=190123, 1=66228, 양성비율=25.8%

## [2] 전처리 + 피처 엔지니어링
  전처리 중...
- 피처 수: 165, 카테고리: 20
- 카테고리(20개): ['시술 시기 코드', '시술 당시 나이', '시술 유형', '특정 시술 유형', '배란 유도 유형', '배아 생성 주요 이유', '총 시술 횟수', '클리닉 내 총 시술 횟수', 'IVF 시술 횟수', 'DI 시술 횟수', '총 임신 횟수', 'IVF 임신 횟수', 'DI 임신 횟수', '총 출산 횟수', 'IVF 출산 횟수', 'DI 출산 횟수', '난자 출처', '정자 출처', '난자 기증자 나이', '정자 기증자 나이']
- 팀원 피처: 29개
- v17 frozen 피처: 8개
- pruned domain 피처: 21개
- 제거된 상수 컬럼 (2개): ['불임 원인 - 여성 요인', '난자 채취 경과일']
- FE 소요: 0.1분

## [3] subgroup 통계
- transfer: 213516건, pos=30.62%
- non-transfer: 42835건, pos=1.96%
- frozen transfer: 37281건, pos=24.69%
- Task type: CPU

## [4] global base 준비
- 외부 global blend 사용
  - v17: weight=0.65, oof=C:\Users\mal03\Downloads\v22_lite_notebook\npy\oof_v17_final.npy, test=C:\Users\mal03\Downloads\v22_lite_notebook\npy\test_v17_final.npy
  - v19: weight=0.35, oof=C:\Users\mal03\Downloads\v22_lite_notebook\npy\oof_v19_final.npy, test=C:\Users\mal03\Downloads\v22_lite_notebook\npy\test_v19_final.npy
- external global OOF AUC: 0.740815
- external global LogLoss: 0.493255
- external global AP: 0.452008

## [4.5] expert-specific feature pruning
- transfer expert: subgroup 상수 제거 4개: ['시술 유형', '실제이식여부', 'is_ivf', 'is_di']
- transfer expert 피처 수: 161 (원본: 165)
- frozen expert: subgroup 상수 제거 9개: ['시술 유형', '불임 원인 - 정자 면역학적 요인', '동결 배아 사용 여부', '난자 해동 경과일', '난자 혼합 경과일', '실제이식여부', 'is_ivf', 'is_di', 'is_frozen_transfer']
- frozen expert: tail 추가 제거 2개: ['frozen_single_embryo', 'frozen_x_day5plus']
- frozen expert: 총 제거 11개: ['is_frozen_transfer', '불임 원인 - 정자 면역학적 요인', '실제이식여부', 'is_ivf', 'is_di', '난자 해동 경과일', '시술 유형', 'frozen_x_day5plus', '동결 배아 사용 여부', 'frozen_single_embryo', '난자 혼합 경과일']
- frozen expert 피처 수: 154 (원본: 165)

## [5] transfer expert 학습

### [transfer] subgroup 학습
- n=213516, pos=30.62%

  --- transfer Seed 42 (1/2) ---
    Fold 1: AUC=0.6724, iter=2463, 소요=12.3분 [1/10]
    Fold 2: AUC=0.6817, iter=2460, 소요=13.4분 [2/10]
    Fold 3: AUC=0.6728, iter=2201, 소요=12.2분 [3/10]
    Fold 4: AUC=0.6718, iter=3189, 소요=16.2분 [4/10]
    Fold 5: AUC=0.6740, iter=2451, 소요=12.8분 [5/10]
  transfer Seed 42 OOF AUC: 0.6745

  --- transfer Seed 2026 (2/2) ---
    Fold 1: AUC=0.6765, iter=2640, 소요=13.4분 [6/10]
    Fold 2: AUC=0.6720, iter=2242, 소요=11.4분 [7/10]
    Fold 3: AUC=0.6719, iter=2119, 소요=10.8분 [8/10]
    Fold 4: AUC=0.6767, iter=2431, 소요=12.4분 [9/10]
    Fold 5: AUC=0.6744, iter=2231, 소요=11.4분 [10/10]
  transfer Seed 2026 OOF AUC: 0.6743

  === [transfer] 최종 subgroup OOF AUC: 0.674566 ===
  seed별: ['0.6745', '0.6743']
- transfer expert 학습 소요: 126.3분

## [6] frozen expert 학습

### [frozen] subgroup 학습
- n=37281, pos=24.69%

  --- frozen Seed 42 (1/2) ---
    Fold 1: AUC=0.6103, iter=2332, 소요=5.1분 [1/10]
    Fold 2: AUC=0.6062, iter=1690, 소요=4.8분 [2/10]
    Fold 3: AUC=0.6218, iter=1648, 소요=4.5분 [3/10]
    Fold 4: AUC=0.6238, iter=1757, 소요=4.9분 [4/10]
    Fold 5: AUC=0.6297, iter=1646, 소요=4.0분 [5/10]
  frozen Seed 42 OOF AUC: 0.6181

  --- frozen Seed 2026 (2/2) ---
    Fold 1: AUC=0.6236, iter=2025, 소요=5.2분 [6/10]
    Fold 2: AUC=0.6215, iter=1488, 소요=4.0분 [7/10]
    Fold 3: AUC=0.6161, iter=2684, 소요=7.1분 [8/10]
    Fold 4: AUC=0.6200, iter=1522, 소요=3.9분 [9/10]
    Fold 5: AUC=0.6221, iter=1918, 소요=4.9분 [10/10]
  frozen Seed 2026 OOF AUC: 0.6206

  === [frozen] 최종 subgroup OOF AUC: 0.620152 ===
  seed별: ['0.6181', '0.6206']
- frozen expert 학습 소요: 48.4분

## [7] expert blend weight 탐색
  [coarse] best a=0.35, b=0.00, c=0.50, AUC=0.740927
  [fine]   best a=0.33, b=0.00, c=0.49, AUC=0.740927
- best weights: {'a_transfer_nonfrozen': 0.33000000000000007, 'b_transfer_frozen': 0.0, 'c_frozen_frozen': 0.49000000000000005}
- blended OOF AUC: 0.740927
- blended OOF LogLoss: 0.493233
- blended OOF AP: 0.452101
- blend 탐색 소요: 8.6분

## [8] 그룹별 AUC 분석
  transfer: global=0.6749 -> final=0.6751 (delta=+0.00014, n=213516, pos=30.6%)
  non-transfer: global=0.9401 -> final=0.9401 (delta=+0.00000, n=42835, pos=2.0%)
  frozen transfer: global=0.6202 -> final=0.6210 (delta=+0.00085, n=37281, pos=24.7%)
  fresh transfer: global=0.6777 -> final=0.6778 (delta=+0.00008, n=176235, pos=31.9%)
  DI: global=0.6843 -> final=0.6843 (delta=+0.00000, n=6291, pos=12.9%)
  donor egg: global=0.6794 -> final=0.6801 (delta=+0.00067, n=15769, pos=31.5%)
  age_만18-34세: global=0.7040 -> final=0.7040 (delta=+0.00005, n=102476, pos=32.3%)
  age_만35-37세: global=0.7027 -> final=0.7031 (delta=+0.00034, n=57780, pos=27.8%)
  age_만38-39세: global=0.7105 -> final=0.7106 (delta=+0.00012, n=39247, pos=21.7%)
  age_만40-42세: global=0.7380 -> final=0.7381 (delta=+0.00009, n=37348, pos=15.9%)
  age_만43-44세: global=0.8304 -> final=0.8303 (delta=-0.00012, n=12253, pos=11.8%)
  age_만45-50세: global=0.8334 -> final=0.8332 (delta=-0.00018, n=6918, pos=16.8%)

- transfer expert 자체 subgroup AUC: 0.674566
- frozen expert 자체 subgroup AUC: 0.620152

## [9] 종합 평가지표
### global base
  OOF AUC:      0.740815
  OOF LogLoss:  0.493255
  OOF AP:       0.452008

### final expert blend
  OOF AUC:      0.740927
  OOF LogLoss:  0.493233
  OOF AP:       0.452101

### improvement
  AUC delta:      +0.000112
  LogLoss delta:  -0.000022
  AP delta:       +0.000093

### Threshold별 분류 지표 (final)
      Th      Acc     Prec      Rec       F1     Spec
    0.20   0.5123   0.3400   0.9438   0.5000   0.3619
    0.25   0.5604   0.3586   0.8897   0.5112   0.4457
    0.30   0.6120   0.3815   0.8084   0.5184   0.5435
    0.35   0.6584   0.4061   0.6967   0.5131   0.6451
    0.40   0.6973   0.4341   0.5655   0.4912   0.7432
    0.45   0.7257   0.4659   0.4213   0.4424   0.8318
    0.50   0.7398   0.4938   0.2905   0.3658   0.8962
  최적 F1: 0.5184 (threshold=0.3)

## [10] 제출 파일
- 파일: C:\Users\mal03\Downloads\v22_lite_notebook\result_v22_lite\sample_submission_v22_lite_20260319_220239.csv
- 확률: mean=0.3031, std=0.1801, min=0.000585, max=0.7980

## [11] expert 피처 중요도

### transfer_expert 중요도 상위 35 (피처 161개)
  1. effective_oocyte_age ★domain: 5.7416
  2. advanced_oocyte_age_flag ★domain: 5.1291
  3. advanced_age_autologous_flag ★domain: 3.4849
  4. transfer_intensity: 2.9648
  5. 시술 당시 나이 [cat]: 2.7668
  6. transfer_day_cat: 2.3024
  7. 배아 이식 경과일: 2.2706
  8. embryo_day_interaction: 2.1867
  9. 저장된 배아 수: 1.9108
  10. 시술 당시 나이_num: 1.8342
  11. freeze_to_transfer_ratio_domain ★domain: 1.7957
  12. 나이_제곱 ★team: 1.6651
  13. age_transfer_interaction: 1.5307
  14. 복합위험도점수 ★team: 1.5300
  15. 이식배아수_구간 ★team: 1.5156
  16. ivf_storage_ratio: 1.5125
  17. 배아_잉여율: 1.4970
  18. transferable_rate ★domain: 1.4001
  19. same_or_early_transfer_like ★domain: 1.3877
  20. 배아저장비율 ★team: 1.3285
  21. age_x_single_transfer: 1.2860
  22. age_x_day5plus: 1.2111
  23. 나이_임상구간 ★team: 1.1730
  24. micro_transfer_quality: 1.1339
  25. 배란 유도 유형 [cat]: 1.1244
  26. 배아이식비율 ★team: 1.1014
  27. 배아_이용률: 1.0946
  28. transferable_embryos ★domain: 1.0927
  29. 총 생성 배아 수: 1.0849
  30. 클리닉 내 총 시술 횟수 [cat]: 1.0795
  31. total_embryo_ratio: 1.0335
  32. embryo_transfer_days_x_age: 1.0248
  33. 난자대비이식배아수 ★team: 1.0177
  34. ivf_transfer_ratio: 1.0113
  35. 시술 시기 코드 [cat]: 0.9583
### transfer_expert 중요도 0 피처 (0개): []
- 저장: C:\Users\mal03\Downloads\v22_lite_notebook\result_v22_lite\feature_importance_v22_lite_transfer_expert.csv

### frozen_expert 중요도 상위 35 (피처 154개)
  1. 시술 시기 코드 [cat]: 9.3681
  2. effective_oocyte_age ★domain: 3.1290
  3. 특정 시술 유형 [cat]: 2.9361
  4. thaw_to_transfer_ratio ★v17: 2.7574
  5. 클리닉 내 총 시술 횟수 [cat]: 2.5940
  6. 시술 당시 나이 [cat]: 2.4138
  7. thaw_survival_proxy ★domain: 2.1137
  8. transfer_intensity: 2.0934
  9. advanced_oocyte_age_flag ★domain: 1.7655
  10. 단일 배아 이식 여부: 1.7441
  11. 정자 출처 [cat]: 1.6610
  12. IVF 시술 횟수 [cat]: 1.5728
  13. ivf_failure_x_effective_oocyte_age ★domain: 1.5496
  14. age_transfer_interaction: 1.5419
  15. 난자 기증자 나이 [cat]: 1.4962
  16. 복합위험도점수 ★team: 1.4070
  17. IVF 출산 횟수 [cat]: 1.3994
  18. 클리닉집중도 ★team: 1.3837
  19. 정자 기증자 나이 [cat]: 1.3708
  20. 이식배아수_구간 ★team: 1.3410
  21. same_or_early_transfer_like ★domain: 1.3349
  22. 총 시술 횟수 [cat]: 1.3268
  23. transferable_embryos ★domain: 1.2885
  24. transfer_day_cat: 1.2821
  25. embryo_transfer_days_x_age: 1.2662
  26. embryo_day_interaction: 1.2529
  27. 배아 이식 경과일: 1.2464
  28. frozen_day_interaction ★v17: 1.2368
  29. total_embryo_ratio: 1.2331
  30. 총 출산 횟수 [cat]: 1.2330
  31. 시술 당시 나이_num: 1.2231
  32. 총 임신 횟수 [cat]: 1.1941
  33. DI 시술 횟수 [cat]: 1.1774
  34. female_factor_score ★domain: 1.1670
  35. 나이_제곱 ★team: 1.1653
### frozen_expert 중요도 0 피처 (0개): []
- 저장: C:\Users\mal03\Downloads\v22_lite_notebook\result_v22_lite\feature_importance_v22_lite_frozen_expert.csv

## [12] 버전 비교
| 버전 | 모델 | OOF AUC | Test AUC | 비고 |
|------|------|---------|----------|------|
| v1 | CB원본 | 0.7403 | - | 베이스라인 |
| v9 | XGB+CB 3seed | 0.7405 | 0.74166 | +IVF/DI |
| v15 | CB 5seed+Optuna | 0.7407 | 0.74169 | EDA+HP |
| v17 | CB 5seed | 0.7408 | TBD | 팀원피처+frozen+CW |
| v19 | CB 5seed | 0.7408 | TBD | +domain 81개 |
| v20_pruned | CB 5seed | 0.7408 | TBD | +domain pruned 21개 |
| v21_expert | expert blend | 0.740946 | 0.7424 | global+transfer+frozen |
| **v22_lite** | **expert blend refined** | **0.740927** | **TBD** | **+prune+core feats+fine blend** |

============================================================
## 최종 요약
============================================================
- v22_expert 핵심 변경 (vs v21):
  1. global base source: external_blend
  2. external global blend: {'v17': 0.65, 'v19': 0.35}
  3. transfer expert: subgroup 상수 피처 자동 제거
     - 제거 4개: ['시술 유형', '실제이식여부', 'is_ivf', 'is_di']
  4. frozen expert: subgroup 상수 + frozen tail 피처 제거
     - 제거 11개: ['is_frozen_transfer', '불임 원인 - 정자 면역학적 요인', '실제이식여부', 'is_ivf', 'is_di', '난자 해동 경과일', '시술 유형', 'frozen_x_day5plus', '동결 배아 사용 여부', 'frozen_single_embryo', '난자 혼합 경과일']
  5. 신규 피처: created_minus_transferred, embryo_transfer_days_x_age
  6. blend weight fine search (coarse 0.05 → fine 0.01)
  7. best weights: {'a_transfer_nonfrozen': 0.33000000000000007, 'b_transfer_frozen': 0.0, 'c_frozen_frozen': 0.49000000000000005}
- global base OOF AUC: 0.740815
- final expert blend OOF AUC: 0.740927
- AUC 개선: +0.000112
- final OOF LogLoss: 0.493233
- final OOF AP: 0.452101
- transfer expert seed별: ['0.6745', '0.6743']
- frozen expert seed별: ['0.6181', '0.6206']
- 전체 피처 수: 165 (cat=20)
- transfer expert 피처: 161개
- frozen expert 피처: 154개
- 전역 상수 제거: 2개
- 총 소요: 183.5분
- 로그: C:\Users\mal03\Downloads\v22_lite_notebook\result_v22_lite\log_v22_lite.md
============================================================