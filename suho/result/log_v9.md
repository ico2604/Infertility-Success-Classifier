# v9 - v8 기반 + IVF/DI 분기 피처 + CatBoost 3-seed 앙상블
시각: 20260315_231425
============================================================

## [1] 데이터 로드
- train: (256351, 69), test: (90067, 68)
- 타겟: 0=190123, 1=66228, 양성비율=25.8%

## [2] XGBoost 전처리 (v8 + IVF/DI 분기)
  [train] step1: (256351, 63), obj=20
  [train] step2: obj=10
  [train] step3: obj=7
  [train] step4: obj=4
  [train] step5: obj=4
  [train] step6: 컬럼=95, obj=4
  [test] step1: (90067, 63), obj=20
  [test] step2: obj=10
  [test] step3: obj=7
  [test] step4: obj=4
  [test] step5: obj=4
  [test] step6: 컬럼=95, obj=4

## [3] Target Encoding (K-Fold)
  - 특정 시술 유형 (고유값: 24) → 덮어쓰기
  - 배아 생성 주요 이유 (고유값: 14) → 덮어쓰기
  - 시술 시기 코드 (고유값: 7) → 덮어쓰기
  - 배란 유도 유형 (고유값: 4) → 덮어쓰기
  - 시술 당시 나이 (고유값: 7) → 시술 당시 나이_te

## [4] XGBoost 피처 정리
- 전체 피처 수: 89
- v9 신규 피처 (15개): ['is_ivf', 'is_di', 'ivf_total_treatment_load', 'di_total_treatment_load', 'ivf_pregnancy_load', 'di_pregnancy_load', 'ivf_transfer_ratio', 'di_transfer_ratio', 'ivf_storage_ratio', 'di_storage_ratio', 'ivf_embryo_age_signal', 'di_embryo_age_signal', 'treatment_history_combo', 'treatment_source_combo', 'treatment_age_history_combo']

## [5] CatBoost 데이터 (네이티브 카테고리 + v8 파생 + IVF/DI 분기)
- CB 피처: 77개, 카테고리: 20개
- v9 신규 피처 (12개): ['is_ivf', 'is_di', 'ivf_total_treatment_load', 'di_total_treatment_load', 'ivf_pregnancy_load', 'di_pregnancy_load', 'ivf_transfer_ratio', 'di_transfer_ratio', 'ivf_storage_ratio', 'di_storage_ratio', 'ivf_embryo_age_signal', 'di_embryo_age_signal']

## [6] CatBoost 3-seed 앙상블 (seed=[42, 2026, 2604])
### 파라미터 : iterations=6000, lr=0.02, depth=6, l2_leaf_reg=10, min_data_in_leaf=20, random_strength=1.2, subsample=0.8, order_count=128, early_stopping_rounds=300

  --- CatBoost Seed 42 (1/3) ---
    Fold 1: AUC=0.7381, iter=1126, 소요=0.9분
    Fold 2: AUC=0.7428, iter=2259, 소요=1.6분
    Fold 3: AUC=0.7401, iter=1736, 소요=1.3분
    Fold 4: AUC=0.7383, iter=1057, 소요=0.9분
    Fold 5: AUC=0.7410, iter=1336, 소요=1.1분
  Seed 42 OOF AUC: 0.7400

  --- CatBoost Seed 2026 (2/3) ---
    Fold 1: AUC=0.7370, iter=3182, 소요=2.2분
    Fold 2: AUC=0.7405, iter=1734, 소요=1.3분
    Fold 3: AUC=0.7375, iter=1390, 소요=1.1분
    Fold 4: AUC=0.7469, iter=1738, 소요=1.3분
    Fold 5: AUC=0.7393, iter=1644, 소요=1.3분
  Seed 2026 OOF AUC: 0.7402

  --- CatBoost Seed 2604 (3/3) ---
    Fold 1: AUC=0.7400, iter=1153, 소요=1.0분
    Fold 2: AUC=0.7382, iter=1562, 소요=1.2분
    Fold 3: AUC=0.7416, iter=1520, 소요=1.2분
    Fold 4: AUC=0.7382, iter=1857, 소요=1.4분
    Fold 5: AUC=0.7423, iter=1906, 소요=1.4분
  Seed 2604 OOF AUC: 0.7400

  === CatBoost 3-seed 앙상블 OOF AUC: 0.7404 ===
  개별 seed: ['0.7400', '0.7402', '0.7400']

## [7] XGBoost 5-Fold (v8 파라미터, depth=7)

  [XGB] Fold 1/5 학습 중...
  [XGB] Fold 1: AUC=0.7373, iter=1069, 소요=0.1분

  [XGB] Fold 2/5 학습 중...
  [XGB] Fold 2: AUC=0.7423, iter=1074, 소요=0.1분

  [XGB] Fold 3/5 학습 중...
  [XGB] Fold 3: AUC=0.7394, iter=664, 소요=0.1분

  [XGB] Fold 4/5 학습 중...
  [XGB] Fold 4: AUC=0.7380, iter=797, 소요=0.1분

  [XGB] Fold 5/5 학습 중...
  [XGB] Fold 5: AUC=0.7395, iter=694, 소요=0.1분

  === XGBoost OOF AUC: 0.7393 ===

## [8] 블렌딩 가중치 탐색
  1차: XGB=0.20, CB=0.80, AUC=0.7405
  미세조정: XGB=0.23, CB=0.77, AUC=0.7405
  → 블렌딩(0.7405) > CB 3-seed(0.7404) → 블렌딩

## [9] 제출 파일
- 파일: C:\Users\ico26\oz\Infertility-Success-Classifier\suho\result\sample_submission_v9_20260315_231425.csv
- 확률: mean=0.2587, std=0.1592, min=0.0004, max=0.7478
- 예시:
        ID  probability
TEST_00000     0.001576
TEST_00001     0.012629
TEST_00002     0.153711
TEST_00003     0.110003
TEST_00004     0.502249

## [10] 피처 중요도

### XGBoost 상위 30
  1. 실제이식여부: 0.3587
  2. total_embryo_ratio: 0.1032
  3. 시술 유형: 0.0785
  4. is_ivf ★v9: 0.0521
  5. 저장된 배아 수: 0.0459
  6. is_di ★v9: 0.0438
  7. 배아 이식 경과일: 0.0220
  8. 신선 배아 사용 여부: 0.0200
  9. 배아_잉여율: 0.0197
  10. 동결 배아 사용 여부: 0.0146
  11. 시술 당시 나이: 0.0140
  12. 이식된 배아 수: 0.0132
  13. treatment_age_history_combo ★v9: 0.0126
  14. purpose_zero: 0.0100
  15. 총 생성 배아 수: 0.0089
  16. is_donor_egg: 0.0088
  17. 난자 기증자 나이: 0.0085
  18. ivf_storage_ratio ★v9: 0.0082
  19. ivf_transfer_ratio ★v9: 0.0067
  20. 배아_이용률: 0.0065
  21. treatment_source_combo ★v9: 0.0064
  22. ivf_embryo_age_signal ★v9: 0.0054
  23. age_peak: 0.0042
  24. 정자 출처: 0.0040
  25. 이식유형_복수신선: 0.0039
  26. 난자 출처: 0.0039
  27. ivf_pregnancy_load ★v9: 0.0036
  28. 이식유형_단일동결: 0.0034
  29. 총 출산 횟수: 0.0033
  30. 단일 배아 이식 여부: 0.0032

### CatBoost 상위 30 (마지막 모델)
  1. 실제이식여부: 35.47
  2. total_embryo_ratio: 19.26
  3. 수집된 신선 난자 수: 4.21
  4. 시술 당시 나이: 3.85
  5. 이식된 배아 수: 3.81
  6. ivf_storage_ratio ★v9: 3.56
  7. 배아 이식 경과일: 3.13
  8. ivf_embryo_age_signal ★v9: 3.10
  9. ivf_transfer_ratio ★v9: 2.34
  10. 난자 출처: 2.01
  11. 저장된 배아 수: 1.98
  12. 정자 출처: 1.87
  13. ivf_total_treatment_load ★v9: 1.66
  14. 총 생성 배아 수: 1.13
  15. 배아_이용률: 1.12
  16. 배아_잉여율: 1.04
  17. 수정_성공률: 0.82
  18. ivf_pregnancy_load ★v9: 0.73
  19. age_peak: 0.68
  20. di_total_treatment_load ★v9: 0.61
  21. 시술 시기 코드: 0.57
  22. 정자 기증자 나이: 0.45
  23. 난자_배아_전환율: 0.38
  24. 클리닉 내 총 시술 횟수: 0.37
  25. IVF 시술 횟수: 0.37
  26. 난자 기증자 나이: 0.36
  27. 배란 유도 유형: 0.34
  28. 해동된 배아 수: 0.32
  29. 배아 생성 주요 이유: 0.32
  30. 미세주입된 난자 수: 0.27

## [11] 버전 비교
| 버전 | 모델 | OOF AUC | 비고 |
|------|------|---------|------|
| v1 | CB원본 | 0.7403 | 베이스라인 |
| v4 | XGB | 0.7392 | 컬럼복원 |
| v7 | XGB+CB | 0.7402 | 블렌딩 |
| v8 | XGB+CB | 0.7401 | +파생변수 |
| v9-CB | CB 3seed | 0.7404 | +IVF/DI분기 |
| v9-XGB | XGB | 0.7393 | +IVF/DI분기 |
| v9 | blend | 0.7405 | 최종 |

============================================================
## 최종 요약
============================================================
- v9 핵심: IVF/DI 분기 피처 12개 추가 (팀원 인사이트)
- CatBoost 3-seed 앙상블: 0.7404
  - seed별: ['0.7400', '0.7402', '0.7400']
  - 피처: 77개 (cat=20)
- XGBoost: 0.7393 (depth=7, 피처=89)
- 블렌딩: XGB=0.23, CB=0.77
- **최종 OOF AUC: 0.7405** (v8 대비 +0.0004)
- 데이터 누수: 없음 (행 단위 연산만 사용)
- 소요: 19.8분
- 로그: C:\Users\ico26\oz\Infertility-Success-Classifier\suho\result\log_v9.md
============================================================
