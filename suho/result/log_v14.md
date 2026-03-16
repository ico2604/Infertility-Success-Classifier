# v14 - 이식 그룹 전용 피처 강화 + CatBoost 3-seed
시각: 20260316_164944
============================================================

## [1] 데이터 로드
- train: (256351, 69), test: (90067, 68)
- 타겟: 0=190123, 1=66228, 양성비율=25.8%

## [2] CatBoost 전처리 + 이식 그룹 전용 피처
  전처리 중...
- CB 피처: 87개, 카테고리: 20개
- 카테고리: ['시술 시기 코드', '시술 당시 나이', '시술 유형', '특정 시술 유형', '배란 유도 유형', '배아 생성 주요 이유', '총 시술 횟수', '클리닉 내 총 시술 횟수', 'IVF 시술 횟수', 'DI 시술 횟수', '총 임신 횟수', 'IVF 임신 횟수', 'DI 임신 횟수', '총 출산 횟수', 'IVF 출산 횟수', 'DI 출산 횟수', '난자 출처', '정자 출처', '난자 기증자 나이', '정자 기증자 나이']
- v14 신규 피처 (10개): ['transfer_day_optimal', 'transfer_day_cat', 'embryo_day_interaction', 'fresh_transfer_ratio', 'micro_transfer_quality', 'single_good_embryo', 'frozen_embryo_signal', 'embryo_surplus_after_transfer', 'transfer_intensity', 'age_transfer_interaction']

### v14 신규 피처 통계 (train)
  transfer_day_optimal: mean=0.6981, std=0.4591, nonzero=178961 (69.8%)
  transfer_day_cat: mean=1.8998, std=1.0779, nonzero=213516 (83.3%)
  embryo_day_interaction: mean=4.2704, std=3.4805, nonzero=187876 (73.3%)
  fresh_transfer_ratio: mean=0.2063, std=0.2061, nonzero=176438 (68.8%)
  micro_transfer_quality: mean=0.4319, std=0.4941, nonzero=111286 (43.4%)
  single_good_embryo: mean=0.1977, std=0.3983, nonzero=50689 (19.8%)
  frozen_embryo_signal: mean=0.2785, std=0.9300, nonzero=37281 (14.5%)
  embryo_surplus_after_transfer: mean=2.4362, std=3.2147, nonzero=141421 (55.2%)
  transfer_intensity: mean=0.3976, std=0.5913, nonzero=213516 (83.3%)
  age_transfer_interaction: mean=46.7103, std=27.6945, nonzero=213516 (83.3%)

## [3] 이식/비이식 그룹 확인
- 이식 그룹: 213516건 (83.3%), 양성률=30.6%
- 비이식 그룹: 42835건 (16.7%), 양성률=1.96%

## [4] CatBoost 3-seed 앙상블
### 파라미터: iterations=5000, lr=0.01, depth=8, l2=3
### Seeds: [42, 2026, 2604]

  --- Seed 42 (1/3) ---
    Fold 1: AUC=0.7382, iter=2026, 소요=2.3분 [1/15]
    Fold 2: AUC=0.7432, iter=2291, 소요=2.5분 [2/15]
    Fold 3: AUC=0.7408, iter=1821, 소요=2.1분 [3/15]
    Fold 4: AUC=0.7386, iter=1304, 소요=1.5분 [4/15]
    Fold 5: AUC=0.7409, iter=1514, 소요=1.8분 [5/15]
  Seed 42 OOF AUC: 0.7403

  --- Seed 2026 (2/3) ---
    Fold 1: AUC=0.7370, iter=2171, 소요=2.4분 [6/15]
    Fold 2: AUC=0.7404, iter=1510, 소요=1.8분 [7/15]
    Fold 3: AUC=0.7380, iter=1166, 소요=1.4분 [8/15]
    Fold 4: AUC=0.7473, iter=2244, 소요=2.5분 [9/15]
    Fold 5: AUC=0.7393, iter=1524, 소요=1.8분 [10/15]
  Seed 2026 OOF AUC: 0.7404

  --- Seed 2604 (3/3) ---
    Fold 1: AUC=0.7401, iter=1549, 소요=1.8분 [11/15]
    Fold 2: AUC=0.7382, iter=1575, 소요=1.8분 [12/15]
    Fold 3: AUC=0.7417, iter=1450, 소요=1.6분 [13/15]
    Fold 4: AUC=0.7387, iter=2358, 소요=2.5분 [14/15]
    Fold 5: AUC=0.7425, iter=1914, 소요=2.1분 [15/15]
  Seed 2604 OOF AUC: 0.7402

  === CatBoost 3-seed OOF AUC: 0.7406 ===
  개별 seed: ['0.7403', '0.7404', '0.7402']

## [5] 그룹별 AUC 분석
- 이식 그룹 (213516건): AUC=0.674646
- 비이식 그룹 (42835건): AUC=0.940425
- 전체: AUC=0.740596
- v13 이식 그룹 AUC 대비: v13=0.6745, v14=0.6746 (Δ=+0.0001)

## [6] 종합 평가지표

### 확률 기반 지표
  OOF AUC:      0.740596
  OOF Log Loss: 0.487662
  OOF AP:       0.451617

### Threshold별 분류 지표
   Threshold    Accuracy   Precision      Recall          F1   Specificity
  ----------  ----------  ----------  ----------  ----------  ------------
        0.20      0.5549      0.3565      0.8980      0.5104        0.4354
        0.25      0.6138      0.3823      0.8034      0.5181        0.5477
        0.30      0.6651      0.4101      0.6761      0.5105        0.6613
        0.35      0.7054      0.4417      0.5317      0.4826        0.7659
        0.40      0.7307      0.4735      0.3801      0.4217        0.8528
        0.45      0.7424      0.5030      0.2509      0.3348        0.9136
        0.50      0.7465      0.5428      0.1178      0.1936        0.9654

  최적 F1: 0.5181 (threshold=0.25)

## [7] 제출 파일 생성
- 파일: C:\Users\ico26\oz\Infertility-Success-Classifier\suho\result\sample_submission_v14_20260316_164944.csv
- 확률: mean=0.2583, std=0.1595, min=0.000181, max=0.7768
- 예시:
          ID   probability
  TEST_00000      0.001835
  TEST_00001      0.014629
  TEST_00002      0.140805
  TEST_00003      0.111519
  TEST_00004      0.497715

## [8] CatBoost 피처 중요도 (상위 30)
  1. 시술 당시 나이 [cat]: 7.80
  2. transfer_intensity ★v14: 7.05
  3. 배아_잉여율: 5.43
  4. transfer_day_cat ★v14: 5.27
  5. age_transfer_interaction ★v14: 4.84
  6. 배아_이용률: 4.68
  7. 배아 이식 경과일: 4.40
  8. ivf_transfer_ratio: 4.34
  9. total_embryo_ratio: 4.34
  10. ivf_storage_ratio: 3.99
  11. 수집된 신선 난자 수: 3.77
  12. 난자 출처 [cat]: 3.55
  13. embryo_day_interaction ★v14: 3.31
  14. 실제이식여부: 3.15
  15. 이식된 배아 수: 2.49
  16. 총 생성 배아 수: 2.42
  17. embryo_surplus_after_transfer ★v14: 2.32
  18. transfer_day_optimal ★v14: 2.06
  19. 시술 시기 코드 [cat]: 1.84
  20. 정자 출처 [cat]: 1.82
  21. fresh_transfer_ratio ★v14: 1.80
  22. 저장된 배아 수: 1.40
  23. 배란 유도 유형 [cat]: 1.19
  24. 클리닉 내 총 시술 횟수 [cat]: 1.10
  25. 난자_배아_전환율: 1.03
  26. 배아 생성 주요 이유 [cat]: 1.02
  27. IVF 시술 횟수 [cat]: 0.95
  28. 수정_성공률: 0.85
  29. 미세주입된 난자 수: 0.82
  30. 배란 자극 여부: 0.68

### v14 신규 피처 중요도
  transfer_intensity: 7.05 (전체 2위)
  transfer_day_cat: 5.27 (전체 4위)
  age_transfer_interaction: 4.84 (전체 5위)
  embryo_day_interaction: 3.31 (전체 13위)
  embryo_surplus_after_transfer: 2.32 (전체 17위)
  transfer_day_optimal: 2.06 (전체 18위)
  fresh_transfer_ratio: 1.80 (전체 21위)
  frozen_embryo_signal: 0.30 (전체 45위)
  micro_transfer_quality: 0.28 (전체 47위)
  single_good_embryo: 0.05 (전체 66위)

## [9] 버전 비교
| 버전 | 모델 | OOF AUC | 이식그룹AUC | 비고 |
|------|------|---------|------------|------|
| v1 | CB원본 | 0.7403 | - | 베이스라인 (67피처) |
| v4 | XGB | 0.7392 | - | 컬럼복원 |
| v7 | XGB+CB | 0.7402 | - | 블렌딩 |
| v8 | XGB+CB | 0.7401 | - | +파생변수 |
| v9 | XGB+CB 3seed | 0.7405 | - | +IVF/DI분기 |
| v11 | CB+XGB cross | 0.7405 | - | 크로스블렌딩 |
| v12 | CB+XGB+TabNet | 0.7406 | - | 3모델블렌딩 |
| v13 | CB 3seed | 0.7405 | 0.6745 | 클리핑 실패 |
| **v14** | **CB 3seed** | **0.7406** | **0.6746** | **이식전용피처** |

============================================================
## 최종 요약
============================================================
- v14 핵심: 이식 그룹 전용 피처 10개 추가
  ['transfer_day_optimal', 'transfer_day_cat', 'embryo_day_interaction', 'fresh_transfer_ratio', 'micro_transfer_quality', 'single_good_embryo', 'frozen_embryo_signal', 'embryo_surplus_after_transfer', 'transfer_intensity', 'age_transfer_interaction']
- 모델: CatBoost 3-seed 앙상블
- 파라미터: iterations=5000, lr=0.01, depth=8, l2=3
- CB 3-seed OOF AUC: 0.740596
  seed별: ['0.7403', '0.7404', '0.7402']
- 이식 그룹 AUC: 0.674646 (v13: 0.6745, Δ=+0.0001)
- 비이식 그룹 AUC: 0.940425 (v13: 0.909)
- OOF LogLoss: 0.487662
- OOF AP: 0.451617
- 최적 F1: 0.5181 (th=0.25)
- 클리핑: 없음 (v13에서 하락 확인, 폐기)
- 데이터 누수: 없음
  - 모든 파생변수: 행 단위 연산 (타겟/test 정보 미사용)
  - CatBoost: K-Fold OOF 학습, test 예측만 수행
- 피처 수: 87개 (v13: 77개 → v14: 87개, +10개)
- 제출: C:\Users\ico26\oz\Infertility-Success-Classifier\suho\result\sample_submission_v14_20260316_164944.csv
- 소요: 29.9분
- 로그: C:\Users\ico26\oz\Infertility-Success-Classifier\suho\result\log_v14.md
============================================================