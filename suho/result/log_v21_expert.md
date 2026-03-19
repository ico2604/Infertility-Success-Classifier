# v21_expert - global blend + transfer expert + frozen expert
시각: 20260318_165829
============================================================

## [1] 데이터 로드
- train: (256351, 69), test: (90067, 68)
- 타겟: 0=190123, 1=66228, 양성비율=25.8%

## [2] 전처리 + 피처 엔지니어링
  전처리 중...
- 피처 수: 164, 카테고리: 20
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
- Task type: GPU

## [4] global base 준비
- 외부 global blend 사용
  - v17: weight=0.65, oof=C:\Users\ico26\oz\Infertility-Success-Classifier\suho\result\oof_v17_final.npy, test=C:\Users\ico26\oz\Infertility-Success-Classifier\suho\result\test_v17_final.npy
  - v19: weight=0.35, oof=C:\Users\ico26\oz\Infertility-Success-Classifier\suho\result\oof_v19_final.npy, test=C:\Users\ico26\oz\Infertility-Success-Classifier\suho\result\test_v19_final.npy
- external global OOF AUC: 0.740839
- external global LogLoss: 0.493248
- external global AP: 0.452159

## [5] transfer expert 학습

### [transfer] subgroup 학습
- n=213516, pos=30.62%

  --- transfer Seed 42 (1/5) ---
    Fold 1: AUC=0.6723, iter=1632, 소요=2.0분 [1/25]
    Fold 2: AUC=0.6818, iter=1433, 소요=1.7분 [2/25]
    Fold 3: AUC=0.6725, iter=1528, 소요=1.9분 [3/25]
    Fold 4: AUC=0.6712, iter=1924, 소요=2.2분 [4/25]
    Fold 5: AUC=0.6739, iter=1277, 소요=1.7분 [5/25]
  transfer Seed 42 OOF AUC: 0.6743

  --- transfer Seed 2026 (2/5) ---
    Fold 1: AUC=0.6766, iter=2079, 소요=2.4분 [6/25]
    Fold 2: AUC=0.6722, iter=1512, 소요=1.8분 [7/25]
    Fold 3: AUC=0.6720, iter=1610, 소요=1.8분 [8/25]
    Fold 4: AUC=0.6772, iter=1822, 소요=2.0분 [9/25]
    Fold 5: AUC=0.6744, iter=1792, 소요=2.2분 [10/25]
  transfer Seed 2026 OOF AUC: 0.6744

  --- transfer Seed 2604 (3/5) ---
    Fold 1: AUC=0.6759, iter=1271, 소요=1.7분 [11/25]
    Fold 2: AUC=0.6733, iter=1763, 소요=2.2분 [12/25]
    Fold 3: AUC=0.6703, iter=1912, 소요=2.2분 [13/25]
    Fold 4: AUC=0.6777, iter=2085, 소요=2.5분 [14/25]
    Fold 5: AUC=0.6746, iter=1799, 소요=2.2분 [15/25]
  transfer Seed 2604 OOF AUC: 0.6743

  --- transfer Seed 123 (4/5) ---
    Fold 1: AUC=0.6736, iter=1296, 소요=1.6분 [16/25]
    Fold 2: AUC=0.6788, iter=1923, 소요=2.2분 [17/25]
    Fold 3: AUC=0.6743, iter=1521, 소요=1.8분 [18/25]
    Fold 4: AUC=0.6766, iter=1761, 소요=2.1분 [19/25]
    Fold 5: AUC=0.6689, iter=1793, 소요=2.1분 [20/25]
  transfer Seed 123 OOF AUC: 0.6744

  --- transfer Seed 777 (5/5) ---
    Fold 1: AUC=0.6799, iter=1504, 소요=1.8분 [21/25]
    Fold 2: AUC=0.6714, iter=2164, 소요=2.5분 [22/25]
    Fold 3: AUC=0.6732, iter=1688, 소요=2.0분 [23/25]
    Fold 4: AUC=0.6760, iter=1725, 소요=2.0분 [24/25]
    Fold 5: AUC=0.6717, iter=1769, 소요=2.1분 [25/25]
  transfer Seed 777 OOF AUC: 0.6744

  === [transfer] 최종 subgroup OOF AUC: 0.674740 ===
  seed별: ['0.6743', '0.6744', '0.6743', '0.6744', '0.6744']
- transfer expert 학습 소요: 50.8분

## [6] frozen expert 학습

### [frozen] subgroup 학습
- n=37281, pos=24.69%

  --- frozen Seed 42 (1/5) ---
    Fold 1: AUC=0.6073, iter=493, 소요=1.7분 [1/25]
    Fold 2: AUC=0.6062, iter=806, 소요=2.3분 [2/25]
    Fold 3: AUC=0.6219, iter=1185, 소요=2.9분 [3/25]
    Fold 4: AUC=0.6233, iter=1111, 소요=2.9분 [4/25]
    Fold 5: AUC=0.6304, iter=1282, 소요=3.2분 [5/25]
  frozen Seed 42 OOF AUC: 0.6179

  --- frozen Seed 2026 (2/5) ---
    Fold 1: AUC=0.6231, iter=1289, 소요=3.2분 [6/25]
    Fold 2: AUC=0.6217, iter=2296, 소요=5.2분 [7/25]
    Fold 3: AUC=0.6130, iter=1120, 소요=2.9분 [8/25]
    Fold 4: AUC=0.6192, iter=560, 소요=1.8분 [9/25]
    Fold 5: AUC=0.6213, iter=1101, 소요=2.9분 [10/25]
  frozen Seed 2026 OOF AUC: 0.6195

  --- frozen Seed 2604 (3/5) ---
    Fold 1: AUC=0.6259, iter=2237, 소요=5.1분 [11/25]
    Fold 2: AUC=0.6135, iter=383, 소요=1.5분 [12/25]
    Fold 3: AUC=0.6191, iter=907, 소요=2.5분 [13/25]
    Fold 4: AUC=0.6064, iter=411, 소요=1.6분 [14/25]
    Fold 5: AUC=0.6254, iter=1271, 소요=3.2분 [15/25]
  frozen Seed 2604 OOF AUC: 0.6178

  --- frozen Seed 123 (4/5) ---
    Fold 1: AUC=0.6162, iter=594, 소요=2.2분 [16/25]
    Fold 2: AUC=0.6206, iter=1973, 소요=4.5분 [17/25]
    Fold 3: AUC=0.6178, iter=1154, 소요=2.9분 [18/25]
    Fold 4: AUC=0.6173, iter=1099, 소요=2.7분 [19/25]
    Fold 5: AUC=0.6217, iter=590, 소요=1.8분 [20/25]
  frozen Seed 123 OOF AUC: 0.6184

  --- frozen Seed 777 (5/5) ---
    Fold 1: AUC=0.6237, iter=838, 소요=2.2분 [21/25]
    Fold 2: AUC=0.6138, iter=817, 소요=2.2분 [22/25]
    Fold 3: AUC=0.6114, iter=952, 소요=2.4분 [23/25]
    Fold 4: AUC=0.6207, iter=1756, 소요=3.7분 [24/25]
    Fold 5: AUC=0.6261, iter=1756, 소요=3.9분 [25/25]
  frozen Seed 777 OOF AUC: 0.6192

  === [frozen] 최종 subgroup OOF AUC: 0.619906 ===
  seed별: ['0.6179', '0.6195', '0.6178', '0.6184', '0.6192']
- frozen expert 학습 소요: 71.6분

## [7] expert blend weight 탐색
- best weights: {'a_transfer_nonfrozen': 0.30000000000000004, 'b_transfer_frozen': 0.0, 'c_frozen_frozen': 0.35000000000000003}
- blended OOF AUC: 0.740895
- blended OOF LogLoss: 0.493275
- blended OOF AP: 0.452196
- blend 탐색 소요: 4.9분

## [8] 그룹별 AUC 분석
  transfer: global=0.6750 -> final=0.6750 (delta=+0.00007, n=213516, pos=30.6%)
  non-transfer: global=0.9401 -> final=0.9401 (delta=+0.00000, n=42835, pos=2.0%)
  frozen transfer: global=0.6200 -> final=0.6207 (delta=+0.00069, n=37281, pos=24.7%)
  fresh transfer: global=0.6778 -> final=0.6778 (delta=+0.00004, n=176235, pos=31.9%)
  DI: global=0.6845 -> final=0.6845 (delta=+0.00000, n=6291, pos=12.9%)
  donor egg: global=0.6797 -> final=0.6801 (delta=+0.00034, n=15769, pos=31.5%)
  age_만18-34세: global=0.7040 -> final=0.7040 (delta=+0.00002, n=102476, pos=32.3%)
  age_만35-37세: global=0.7028 -> final=0.7030 (delta=+0.00020, n=57780, pos=27.8%)
  age_만38-39세: global=0.7105 -> final=0.7106 (delta=+0.00011, n=39247, pos=21.7%)
  age_만40-42세: global=0.7381 -> final=0.7381 (delta=+0.00003, n=37348, pos=15.9%)
  age_만43-44세: global=0.8306 -> final=0.8305 (delta=-0.00006, n=12253, pos=11.8%)
  age_만45-50세: global=0.8329 -> final=0.8329 (delta=-0.00005, n=6918, pos=16.8%)

- transfer expert 자체 subgroup AUC: 0.674740
- frozen expert 자체 subgroup AUC: 0.619906

## [9] 종합 평가지표
### global base
  OOF AUC:      0.740839
  OOF LogLoss:  0.493248
  OOF AP:       0.452159

### final expert blend
  OOF AUC:      0.740895
  OOF LogLoss:  0.493275
  OOF AP:       0.452196

### improvement
  AUC delta:      +0.000056
  LogLoss delta:  +0.000027
  AP delta:       +0.000037

### Threshold별 분류 지표 (final)
      Th      Acc     Prec      Rec       F1     Spec
    0.20   0.5112   0.3397   0.9452   0.4998   0.3600
    0.25   0.5597   0.3585   0.8918   0.5114   0.4440
    0.30   0.6113   0.3812   0.8091   0.5182   0.5424
    0.35   0.6586   0.4062   0.6957   0.5129   0.6457
    0.40   0.6981   0.4350   0.5641   0.4912   0.7448
    0.45   0.7259   0.4662   0.4196   0.4417   0.8327
    0.50   0.7399   0.4942   0.2901   0.3656   0.8966
  최적 F1: 0.5182 (threshold=0.3)

## [10] 제출 파일
- 파일: C:\Users\ico26\oz\Infertility-Success-Classifier\suho\result\sample_submission_v21_expert_20260318_165829.csv
- 확률: mean=0.3032, std=0.1799, min=0.000542, max=0.8009

## [11] expert 피처 중요도

### transfer_expert 중요도 상위 35
  1. effective_oocyte_age ★domain: 6.1788
  2. advanced_oocyte_age_flag ★domain: 3.8096
  3. 시술 시기 코드 [cat]: 3.5657
  4. advanced_age_autologous_flag ★domain: 3.3685
  5. transfer_intensity: 3.3071
  6. transfer_day_cat: 2.3853
  7. 배아저장비율 ★team: 2.3556
  8. freeze_to_transfer_ratio_domain ★domain: 2.0516
  9. same_or_early_transfer_like ★domain: 1.8608
  10. embryo_day_interaction: 1.7246
  11. 복합위험도점수 ★team: 1.6960
  12. 배아 이식 경과일: 1.6591
  13. age_x_single_transfer: 1.5985
  14. age_transfer_interaction: 1.5934
  15. ivf_storage_ratio: 1.5551
  16. transferable_rate ★domain: 1.4175
  17. micro_transfer_quality: 1.3928
  18. 시술 당시 나이_num: 1.3747
  19. 이식된 배아 수: 1.3130
  20. 저장된 배아 수: 1.3073
  21. ivf_failure_x_effective_oocyte_age ★domain: 1.2602
  22. 총 생성 배아 수: 1.2107
  23. 배아이식비율 ★team: 1.1762
  24. transferable_embryos ★domain: 1.1626
  25. age_x_day5plus: 1.1047
  26. 극고령_여부 ★team: 1.0955
  27. total_embryo_ratio: 1.0938
  28. 배아_잉여율: 1.0927
  29. 나이_임상구간 ★team: 1.0892
  30. 나이_제곱 ★team: 1.0787
  31. 수정_성공률: 1.0440
  32. 시술 당시 나이 [cat]: 1.0428
  33. 배아_이용률: 1.0262
  34. fertilization_rate_all ★domain: 1.0199
  35. ivf_transfer_ratio: 0.9442
### transfer_expert 중요도 0 피처 (6개): ['난자 해동 경과일', 'is_ivf', '시술 유형', '실제이식여부', '불임 원인 - 정자 면역학적 요인', 'is_di']
- 저장: C:\Users\ico26\oz\Infertility-Success-Classifier\suho\result\feature_importance_v21_expert_transfer_expert.csv

### frozen_expert 중요도 상위 35
  1. 시술 시기 코드 [cat]: 11.9622
  2. effective_oocyte_age ★domain: 6.5650
  3. thaw_to_transfer_ratio ★v17: 3.0927
  4. transfer_intensity: 2.9214
  5. 복합위험도점수 ★team: 2.7456
  6. transferable_embryos ★domain: 2.6512
  7. ivf_failure_x_effective_oocyte_age ★domain: 2.5752
  8. thaw_survival_proxy ★domain: 2.4186
  9. advanced_oocyte_age_flag ★domain: 2.3641
  10. 배아 이식 경과일: 2.0400
  11. advanced_age_autologous_flag ★domain: 2.0275
  12. same_or_early_transfer_like ★domain: 1.8855
  13. transfer_day_cat: 1.8828
  14. 특정 시술 유형 [cat]: 1.7571
  15. frozen_day_interaction ★v17: 1.6565
  16. embryo_day_interaction: 1.6087
  17. 단일 배아 이식 여부: 1.5543
  18. total_embryo_ratio: 1.3854
  19. freeze_to_transfer_ratio_domain ★domain: 1.3778
  20. 클리닉집중도 ★team: 1.3493
  21. age_transfer_interaction: 1.2832
  22. 나이XIVF실패 ★team: 1.2812
  23. 저장된 배아 수: 1.1091
  24. PGS 시술 여부: 1.0773
  25. egg_sperm_combo: 1.0221
  26. 해동된 배아 수: 0.9842
  27. age_x_single_transfer: 0.9826
  28. 정자 기증자 나이_num: 0.9615
  29. 나이XIVF임신률 ★team: 0.9612
  30. donor_any_flag ★domain: 0.9606
  31. 이식된 배아 수: 0.9502
  32. 정자 출처 [cat]: 0.9364
  33. 착상 전 유전 검사 사용 여부: 0.9340
  34. female_factor_score ★domain: 0.9095
  35. 시술 당시 나이_num: 0.8878
### frozen_expert 중요도 0 피처 (9개): ['동결 배아 사용 여부', 'is_ivf', '불임 원인 - 정자 면역학적 요인', '난자 혼합 경과일', '실제이식여부', '난자 해동 경과일', 'is_frozen_transfer', '시술 유형', 'is_di']
- 저장: C:\Users\ico26\oz\Infertility-Success-Classifier\suho\result\feature_importance_v21_expert_frozen_expert.csv

## [12] 버전 비교
| 버전 | 모델 | OOF AUC | Test AUC | 비고 |
|------|------|---------|----------|------|
| v1 | CB원본 | 0.7403 | - | 베이스라인 |
| v9 | XGB+CB 3seed | 0.7405 | 0.74166 | +IVF/DI |
| v15 | CB 5seed+Optuna | 0.7407 | 0.74169 | EDA+HP |
| v17 | CB 5seed | 0.7408 | TBD | 팀원피처+frozen+CW |
| v19 | CB 5seed | 0.7408 | TBD | +domain 81개 |
| v20_pruned | CB 5seed | 0.7408 | TBD | +domain pruned 21개 |
| **v21_expert** | **expert blend** | **0.740895** | **TBD** | **global+transfer+frozen** |

============================================================
## 최종 요약
============================================================
- v21_expert 핵심 변경:
  1. global base source: external_blend
  2. external global blend: {'v17': 0.65, 'v19': 0.35}
  3. transfer-only expert 모델 추가
  4. frozen-transfer-only expert 모델 추가
  5. OOF 기반 gating/blending weight 자동 탐색
  6. best weights: {'a_transfer_nonfrozen': 0.30000000000000004, 'b_transfer_frozen': 0.0, 'c_frozen_frozen': 0.35000000000000003}
- global base OOF AUC: 0.740839
- final expert blend OOF AUC: 0.740895
- AUC 개선: +0.000056
- final OOF LogLoss: 0.493275
- final OOF AP: 0.452196
- transfer expert seed별: ['0.6743', '0.6744', '0.6743', '0.6744', '0.6744']
- frozen expert seed별: ['0.6179', '0.6195', '0.6178', '0.6184', '0.6192']
- 피처 수: 164 (cat=20)
- 상수 제거: 2개
- 총 소요: 127.5분
- 로그: C:\Users\ico26\oz\Infertility-Success-Classifier\suho\result\log_v21_expert.md
============================================================