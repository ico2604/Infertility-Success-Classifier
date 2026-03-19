# v16 - Subgroup-aware Multi-model Blend
시각: 20260317_162233
============================================================

## [1] 데이터 로드
- train: (256351, 69), test: (90067, 68)
- 타겟: 0=190123, 1=66228, 양성비율=25.8%

## [2] 전처리 + 피처 엔지니어링
  피처 수: 140, 카테고리: 10
  카테고리: ['시술 시기 코드', '시술 당시 나이', '시술 유형', '특정 시술 유형', '배란 유도 유형', '배아 생성 주요 이유', '난자 출처', '정자 출처', '난자 기증자 나이', '정자 기증자 나이']
  피처 엔지니어링 소요: 0.0분

## [3] subgroup 통계
  이식 (train): 213516건 (83.3%), 양성률=30.6%
  비이식 (train): 42835건, 양성률=1.96%
  이식 (test): 75086건, 비이식: 14981건

## [4] CV 설정
  [경고] strat_key 최소 클래스=1 < 5, y로 fallback
  5-fold StratifiedKFold 생성 완료
  Task type: GPU

## [5] LGB/XGB 피처 준비
  LGB 피처 수: 154
  소요: 0.1분

============================================================
## [6] Model A: Full CatBoost
============================================================
  파라미터: {
  "iterations": 8000,
  "learning_rate": 0.00837420773913813,
  "depth": 8,
  "l2_leaf_reg": 9.013710971500913,
  "min_data_in_leaf": 47,
  "random_strength": 1.9940254840418206,
  "bagging_temperature": 0.7125808252032847,
  "border_count": 64,
  "eval_metric": "AUC",
  "loss_function": "Logloss",
  "od_type": "Iter",
  "od_wait": 400,
  "allow_writing_files": false
}

  --- Full_CB Seed 42 (1/2) ---
    Fold 1: AUC=0.7385, iter=1439, 소요=1.2분 [1/10]
    Fold 2: AUC=0.7429, iter=1960, 소요=1.6분 [2/10]
    Fold 3: AUC=0.7407, iter=1961, 소요=1.6분 [3/10]
    Fold 4: AUC=0.7386, iter=1550, 소요=1.4분 [4/10]
    Fold 5: AUC=0.7413, iter=2041, 소요=1.8분 [5/10]
  Seed 42 OOF AUC: 0.7404

  --- Full_CB Seed 2026 (2/2) ---
    Fold 1: AUC=0.7370, iter=2484, 소요=2.0분 [6/10]
    Fold 2: AUC=0.7404, iter=1894, 소요=1.6분 [7/10]
    Fold 3: AUC=0.7381, iter=1526, 소요=1.3분 [8/10]
    Fold 4: AUC=0.7472, iter=2108, 소요=1.7분 [9/10]
    Fold 5: AUC=0.7395, iter=1708, 소요=1.6분 [10/10]
  Seed 2026 OOF AUC: 0.7404
  === Full_CB 완료: 15.9분, seed AUCs: ['0.7404', '0.7404'] ===
  저장: oof_v16_Full_CB.npy, test_v16_Full_CB.npy

  Full CB OOF AUC: 0.740592

============================================================
## [7] Model B: Transfer-only CatBoost
============================================================

  --- Transfer_CB Seed 42 (1/1) ---
    Fold 1: AUC=0.6724, iter=2131, 소요=1.8분 [1/5]
    Fold 2: AUC=0.6771, iter=2610, 소요=2.0분 [2/5]
    Fold 3: AUC=0.6757, iter=2399, 소요=1.9분 [3/5]
    Fold 4: AUC=0.6709, iter=1888, 소요=1.5분 [4/5]
    Fold 5: AUC=0.6759, iter=1926, 소요=1.6분 [5/5]
  Seed 42 OOF AUC: 0.6744
  === Transfer_CB 완료: 8.7분, seed AUCs: ['0.6744'] ===
  저장: oof_v16_Transfer_CB.npy, test_v16_Transfer_CB.npy

  Transfer CB (이식그룹) OOF AUC: 0.674360

============================================================
## [8] Model C: Non-transfer CatBoost
============================================================

  --- NonTransfer_CB Seed 42 (1/1) ---
    Fold 1: AUC=0.9324, iter=30, 소요=0.3분 [1/5]
    Fold 2: AUC=0.9533, iter=944, 소요=1.3분 [2/5]
    Fold 3: AUC=0.9480, iter=33, 소요=0.4분 [3/5]
    Fold 4: AUC=0.9322, iter=1142, 소요=1.6분 [4/5]
    Fold 5: AUC=0.9439, iter=149, 소요=0.5분 [5/5]
  Seed 42 OOF AUC: 0.8851
  === NonTransfer_CB 완료: 4.1분, seed AUCs: ['0.8851'] ===
  저장: oof_v16_NonTransfer_CB.npy, test_v16_NonTransfer_CB.npy

  NonTransfer CB (비이식그룹) OOF AUC: 0.885142

============================================================
## [9] Model D: Transfer-only LightGBM/XGBoost
============================================================

  === Model D: Transfer-only LightGBM ===

  --- LightGBM Seed 42 (1/1) ---
    Fold 1: AUC=0.6718, 소요=0.1분 [1/5]
    Fold 2: AUC=0.6767, 소요=0.1분 [2/5]
    Fold 3: AUC=0.6755, 소요=0.1분 [3/5]
    Fold 4: AUC=0.6701, 소요=0.1분 [4/5]
    Fold 5: AUC=0.6752, 소요=0.1분 [5/5]
  Seed 42 Transfer AUC: 0.6738
  === LightGBM 완료: 0.3분, seed AUCs: ['0.6738'] ===
  저장: oof_v16_Transfer_LGB.npy, test_v16_Transfer_LGB.npy

  Transfer LGB/XGB (이식그룹) OOF AUC: 0.673826

## [10] OOF 예측 상관분석
  페어                                  상관계수
  ------------------------------  --------
  full_cb ↔ transfer_cb             0.9962
  full_cb ↔ transfer_lgb            0.9895
  full_cb ↔ nontransfer_cb          0.6404
  transfer_cb ↔ transfer_lgb        0.9904

## [11] 블렌딩 가중치 탐색

  탐색 소요: 11.2분

  방식                      AUC                 Transfer W         NonTransfer W
  ---------------  ----------  -------------------------  --------------------
  prob_blend         0.740694          (0.65, 0.1, 0.25)          (0.98, 0.02)
  rank_blend         0.740687            (0.7, 0.1, 0.2)            (1.0, 0.0)
  z_blend            0.740688            (0.7, 0.1, 0.2)            (1.0, 0.0)
  full_cb_only       0.740592                    (1,0,0)                 (1,0)

  ★ 최적: prob_blend (AUC=0.740694)

## [12] 최종 성능

### 최종 블렌딩 평가지표
  OOF AUC:      0.740694
  OOF Log Loss: 0.487772
  OOF AP:       0.451472

  #### 그룹별 AUC
    transfer: AUC=0.6748 (n=213516, pos=30.6%)
    non-transfer: AUC=0.9405 (n=42835, pos=2.0%)
    IVF & transfer: AUC=0.6748 (n=213516, pos=30.6%)
    DI: AUC=0.6842 (n=6291, pos=12.9%)
    fresh transfer: AUC=0.6777 (n=176438, pos=31.9%)
    frozen transfer: AUC=0.6193 (n=37281, pos=24.7%)
    donor egg: AUC=0.6799 (n=15769, pos=31.5%)
    age_만18-34세: AUC=0.7037 (n=102476, pos=32.3%)
    age_만35-37세: AUC=0.7026 (n=57780, pos=27.8%)
    age_만38-39세: AUC=0.7100 (n=39247, pos=21.7%)
    age_만40-42세: AUC=0.7384 (n=37348, pos=15.9%)
    age_만43-44세: AUC=0.8305 (n=12253, pos=11.8%)
    age_만45-50세: AUC=0.8331 (n=6918, pos=16.8%)

  #### Threshold별 분류 지표
      Th      Acc     Prec      Rec       F1     Spec
     ---      ---      ---      ---      ---      ---
    0.20   0.5530   0.3556   0.8993   0.5097   0.4324
    0.25   0.6124   0.3815   0.8051   0.5177   0.5453
    0.30   0.6649   0.4102   0.6783   0.5112   0.6603
    0.35   0.7060   0.4423   0.5295   0.4820   0.7674
    0.40   0.7306   0.4733   0.3789   0.4209   0.8531
    0.45   0.7426   0.5037   0.2499   0.3341   0.9142
    0.50   0.7463   0.5422   0.1161   0.1912   0.9659
  최적 F1: 0.5177 (threshold=0.25)

### 개별 모델 비교

### Full CB 단독 평가지표
  OOF AUC:      0.740592
  OOF Log Loss: 0.487700
  OOF AP:       0.451490

  #### 그룹별 AUC
    transfer: AUC=0.6747 (n=213516, pos=30.6%)
    non-transfer: AUC=0.9409 (n=42835, pos=2.0%)
    IVF & transfer: AUC=0.6747 (n=213516, pos=30.6%)
    DI: AUC=0.6836 (n=6291, pos=12.9%)
    fresh transfer: AUC=0.6776 (n=176438, pos=31.9%)
    frozen transfer: AUC=0.6183 (n=37281, pos=24.7%)
    donor egg: AUC=0.6795 (n=15769, pos=31.5%)
    age_만18-34세: AUC=0.7037 (n=102476, pos=32.3%)
    age_만35-37세: AUC=0.7023 (n=57780, pos=27.8%)
    age_만38-39세: AUC=0.7100 (n=39247, pos=21.7%)
    age_만40-42세: AUC=0.7382 (n=37348, pos=15.9%)
    age_만43-44세: AUC=0.8303 (n=12253, pos=11.8%)
    age_만45-50세: AUC=0.8333 (n=6918, pos=16.8%)

  #### Threshold별 분류 지표
      Th      Acc     Prec      Rec       F1     Spec
     ---      ---      ---      ---      ---      ---
    0.20   0.5539   0.3560   0.8977   0.5098   0.4342
    0.25   0.6127   0.3816   0.8041   0.5176   0.5461
    0.30   0.6650   0.4101   0.6773   0.5109   0.6607
    0.35   0.7058   0.4421   0.5306   0.4823   0.7668
    0.40   0.7305   0.4731   0.3790   0.4208   0.8529
    0.45   0.7425   0.5032   0.2521   0.3359   0.9133
    0.50   0.7465   0.5433   0.1183   0.1943   0.9654
  최적 F1: 0.5176 (threshold=0.25)

## [13] 피처 중요도

  ### Full CatBoost 상위 30
      1. 이식된 배아 수: 13.55
      2. 실제이식여부: 11.69
      3. age_transfer_interaction [transfer]: 9.89
      4. transfer_intensity [transfer]: 7.38
      5. transferred_embryos_bucket: 6.75
      6. egg_sperm_combo: 3.81
      7. 시술 당시 나이_num: 2.53
      8. 수집된 신선 난자 수: 2.48
      9. age_sq: 2.41
     10. log_stored_embryos: 2.04
     11. ivf_storage_ratio: 1.76
     12. 시술 당시 나이: 1.67
     13. 난자 출처: 1.58
     14. transfer_day_cat [transfer]: 1.50
     15. fresh_transfer_ratio [transfer]: 1.48
     16. 배아 이식 경과일: 1.46
     17. embryo_day_interaction [transfer]: 1.33
     18. stored_over_generated: 1.18
     19. 시술 시기 코드: 1.11
     20. 저장된 배아 수: 1.07
     21. total_embryo_ratio: 1.06
     22. 정자 출처: 0.90
     23. ivf_transfer_ratio: 0.85
     24. 배란 유도 유형: 0.84
     25. transfer_over_generated: 0.81
     26. 수정_성공률: 0.80
     27. 난자_배아_전환율: 0.80
     28. embryo_surplus_after_transfer: 0.73
     29. 배란 자극 여부: 0.69
     30. ivf_failure_count [failure]: 0.68

  #### Full CatBoost - 신규 피처 카테고리별
    [subtype]
      subtype_has_icsi: 0.31 (#50)
      subtype_has_ivf: 0.18 (#66)
      subtype_has_iui: 0.00 (#130)
      subtype_has_ah: 0.00 (#126)
      subtype_has_blastocyst: 0.00 (#120)
      subtype_has_unknown: 0.08 (#86)
      subtype_token_count: 0.01 (#113)
      subtype_is_duplicate: 0.00 (#134)
      subtype_is_mixed_proc: 0.00 (#127)
      subtype_blastocyst_x_single: 0.00 (#123)
      subtype_blastocyst_x_day5plus: 0.01 (#116)
    [purpose]
      purpose_current: 0.14 (#69)
      purpose_embryo_storage: 0.11 (#76)
      purpose_egg_storage: 0.00 (#135)
      purpose_donation: 0.10 (#80)
      purpose_research: 0.00 (#133)
      purpose_token_count: 0.11 (#75)
      purpose_current_and_storage: 0.00 (#129)
      purpose_zero: 0.09 (#83)
    [transfer_interaction]
      transfer_day_optimal: 0.17 (#67)
      transfer_day_cat: 1.50 (#14)
      embryo_day_interaction: 1.33 (#17)
      fresh_transfer_ratio: 1.48 (#15)
      micro_transfer_quality: 0.20 (#63)
      single_good_embryo: 0.01 (#106)
      frozen_embryo_signal: 0.14 (#70)
      transfer_intensity: 7.38 (#4)
      age_transfer_interaction: 9.89 (#3)
      day5plus: 0.57 (#33)
      single_x_day5plus: 0.01 (#114)
      multi_x_day5plus: 0.05 (#92)
      fresh_x_day5plus: 0.33 (#45)
      frozen_x_day5plus: 0.00 (#128)
      age_x_day5plus: 0.09 (#84)
      age_x_single_transfer: 0.58 (#32)
      icsi_x_day5plus: 0.02 (#100)
      blastocyst_signal: 0.25 (#56)
      donor_egg_x_advanced_age: 0.10 (#77)
    [failure]
      total_failure_count: 0.26 (#54)
      ivf_failure_count: 0.68 (#30)
      pregnancy_to_birth_gap: 0.05 (#93)
      ivf_preg_to_birth_gap: 0.01 (#115)
      repeated_failed_transfer: 0.12 (#74)
      first_transfer_cycle: 0.15 (#68)

  ### Transfer CatBoost 상위 30
      1. 시술 당시 나이_num: 8.57
      2. age_sq: 7.38
      3. transfer_intensity [transfer]: 3.96
      4. 시술 시기 코드: 3.94
      5. 시술 당시 나이: 3.17
      6. log_stored_embryos: 2.91
      7. age_x_single_transfer [transfer]: 2.58
      8. ivf_storage_ratio: 2.39
      9. 배아 이식 경과일: 2.37
     10. 수집된 신선 난자 수: 2.30
     11. 난자_배아_전환율: 2.29
     12. egg_sperm_combo: 2.02
     13. ivf_failure_count [failure]: 1.96
     14. transfer_day_cat [transfer]: 1.93
     15. age_transfer_interaction [transfer]: 1.86
     16. 수정_성공률: 1.81
     17. embryo_day_interaction [transfer]: 1.59
     18. 난자 기증자 나이: 1.46
     19. total_embryo_ratio: 1.42
     20. 클리닉 내 총 시술 횟수_num: 1.38
     21. 저장된 배아 수: 1.34
     22. 난자 출처: 1.27
     23. 불임 원인 - 남성 요인: 1.13
     24. 총 임신 횟수_num: 1.07
     25. 총 출산 횟수_num: 1.05
     26. fresh_transfer_ratio [transfer]: 1.05
     27. 불임 원인 - 난관 질환: 1.04
     28. micro_transfer_quality [transfer]: 1.02
     29. 미세주입된 난자 수: 1.01
     30. transfer_over_generated: 0.99

  #### Transfer CatBoost - 신규 피처 카테고리별
    [subtype]
      subtype_has_icsi: 0.24 (#74)
      subtype_has_ivf: 0.22 (#77)
      subtype_has_iui: 0.00 (#124)
      subtype_has_ah: 0.00 (#123)
      subtype_has_blastocyst: 0.02 (#114)
      subtype_has_unknown: 0.14 (#87)
      subtype_token_count: 0.04 (#105)
      subtype_is_duplicate: 0.05 (#103)
      subtype_is_mixed_proc: 0.01 (#116)
      subtype_blastocyst_x_single: 0.01 (#115)
      subtype_blastocyst_x_day5plus: 0.02 (#111)
    [purpose]
      purpose_current: 0.00 (#131)
      purpose_embryo_storage: 0.00 (#130)
      purpose_egg_storage: 0.00 (#129)
      purpose_donation: 0.18 (#82)
      purpose_research: 0.00 (#128)
      purpose_token_count: 0.15 (#86)
      purpose_current_and_storage: 0.00 (#122)
      purpose_zero: 0.00 (#127)
    [transfer_interaction]
      transfer_day_optimal: 0.48 (#63)
      transfer_day_cat: 1.93 (#14)
      embryo_day_interaction: 1.59 (#17)
      fresh_transfer_ratio: 1.05 (#26)
      micro_transfer_quality: 1.02 (#28)
      single_good_embryo: 0.07 (#98)
      frozen_embryo_signal: 0.09 (#93)
      transfer_intensity: 3.96 (#3)
      age_transfer_interaction: 1.86 (#15)
      day5plus: 0.75 (#43)
      single_x_day5plus: 0.05 (#102)
      multi_x_day5plus: 0.21 (#80)
      fresh_x_day5plus: 0.75 (#44)
      frozen_x_day5plus: 0.00 (#120)
      age_x_day5plus: 0.91 (#35)
      age_x_single_transfer: 2.58 (#7)
      icsi_x_day5plus: 0.08 (#96)
      blastocyst_signal: 0.93 (#34)
      donor_egg_x_advanced_age: 0.32 (#71)
    [failure]
      total_failure_count: 0.85 (#38)
      ivf_failure_count: 1.96 (#13)
      pregnancy_to_birth_gap: 0.13 (#88)
      ivf_preg_to_birth_gap: 0.07 (#97)
      repeated_failed_transfer: 0.81 (#40)
      first_transfer_cycle: 0.23 (#76)

  ### NonTransfer CatBoost 상위 20
      1. is_di: 24.40
      2. is_ivf: 19.20
      3. purpose_token_count [purpose]: 13.88
      4. 시술 유형: 8.48
      5. total_embryo_ratio: 4.55
      6. subtype_has_iui [subtype]: 2.88
      7. age_sq: 2.88
      8. 시술 당시 나이_num: 2.69
      9. subtype_has_ivf [subtype]: 2.55
     10. surplus_bucket: 1.66
     11. 신선 배아 사용 여부: 1.24
     12. 불임 원인 - 남성 요인: 1.06
     13. egg_sperm_combo: 1.06
     14. 총 출산 횟수_num: 0.96
     15. purpose_current [purpose]: 0.85
     16. purpose_egg_storage [purpose]: 0.81
     17. DI 출산 횟수_num: 0.69
     18. 동결 배아 사용 여부: 0.51
     19. ivf_failure_count [failure]: 0.44
     20. 총 임신 횟수_num: 0.42

  #### NonTransfer CatBoost - 신규 피처 카테고리별
    [subtype]
      subtype_has_icsi: 0.30 (#27)
      subtype_has_ivf: 2.55 (#9)
      subtype_has_iui: 2.88 (#6)
      subtype_has_ah: 0.03 (#81)
      subtype_has_blastocyst: 0.06 (#65)
      subtype_has_unknown: 0.01 (#86)
      subtype_token_count: 0.10 (#51)
      subtype_is_duplicate: 0.21 (#38)
      subtype_is_mixed_proc: 0.11 (#50)
      subtype_blastocyst_x_single: 0.00 (#118)
      subtype_blastocyst_x_day5plus: 0.00 (#119)
    [purpose]
      purpose_current: 0.85 (#15)
      purpose_embryo_storage: 0.11 (#49)
      purpose_egg_storage: 0.81 (#16)
      purpose_donation: 0.07 (#61)
      purpose_research: 0.00 (#100)
      purpose_token_count: 13.88 (#3)
      purpose_current_and_storage: 0.01 (#87)
      purpose_zero: 0.05 (#71)
    [transfer_interaction]
      transfer_day_optimal: 0.01 (#93)
      transfer_day_cat: 0.00 (#101)
      embryo_day_interaction: 0.00 (#130)
      fresh_transfer_ratio: 0.00 (#123)
      micro_transfer_quality: 0.00 (#131)
      single_good_embryo: 0.00 (#133)
      frozen_embryo_signal: 0.14 (#43)
      transfer_intensity: 0.00 (#134)
      age_transfer_interaction: 0.00 (#135)
      day5plus: 0.01 (#91)
      single_x_day5plus: 0.00 (#136)
      multi_x_day5plus: 0.00 (#137)
      fresh_x_day5plus: 0.00 (#95)
      frozen_x_day5plus: 0.00 (#138)
      age_x_day5plus: 0.00 (#99)
      age_x_single_transfer: 0.00 (#121)
      icsi_x_day5plus: 0.01 (#89)
      blastocyst_signal: 0.09 (#54)
      donor_egg_x_advanced_age: 0.05 (#70)
    [failure]
      total_failure_count: 0.03 (#78)
      ivf_failure_count: 0.44 (#19)
      pregnancy_to_birth_gap: 0.12 (#45)
      ivf_preg_to_birth_gap: 0.04 (#74)
      repeated_failed_transfer: 0.00 (#126)
      first_transfer_cycle: 0.00 (#124)

  ### Transfer LGB/XGB 상위 30
      1. 저장된 배아 수: 57931.17
      2. 시술 당시 나이_num: 43966.89
      3. log_stored_embryos: 27445.11
      4. 배아 이식 경과일: 18634.95
      5. transfer_intensity [transfer]: 15959.84
      6. 배아_이용률: 13134.72
      7. age_sq: 11949.29
      8. 총 생성 배아 수: 11870.31
      9. 배아_잉여율: 8467.49
     10. embryo_surplus_after_transfer: 8281.99
     11. embryo_day_interaction [transfer]: 8174.40
     12. 난자_배아_전환율: 6324.87
     13. total_embryo_ratio: 6161.22
     14. 시술 당시 나이: 5743.76
     15. transfer_day_cat [transfer]: 5402.81
     16. ivf_failure_count [failure]: 5103.71
     17. 난자 기증자 나이: 5020.42
     18. 수정_성공률: 4742.01
     19. 시술 당시 나이_te: 4615.22
     20. egg_sperm_combo: 4381.29
     21. 시술 시기 코드_te: 4281.87
     22. 수집된 신선 난자 수: 4273.63
     23. 시술 시기 코드_count: 4040.44
     24. log_total_embryos: 3480.99
     25. ivf_storage_ratio: 3436.49
     26. micro_transfer_quality [transfer]: 2955.92
     27. 클리닉 내 총 시술 횟수_num: 2910.18
     28. fresh_transfer_ratio [transfer]: 2809.47
     29. 특정 시술 유형_te: 2765.81
     30. 이식된 배아 수: 2612.13

  #### Transfer LGB/XGB - 신규 피처 카테고리별
    [subtype]
      subtype_has_icsi: 22.08 (#122)
      subtype_has_ivf: 159.46 (#97)
      subtype_has_iui: 0.00 (#149)
      subtype_has_ah: 5.18 (#134)
      subtype_has_blastocyst: 13.68 (#128)
      subtype_has_unknown: 195.41 (#93)
      subtype_token_count: 0.00 (#147)
      subtype_is_duplicate: 42.09 (#114)
      subtype_is_mixed_proc: 28.35 (#118)
      subtype_blastocyst_x_single: 7.44 (#130)
      subtype_blastocyst_x_day5plus: 5.97 (#133)
    [purpose]
      purpose_current: 0.00 (#146)
      purpose_embryo_storage: 0.00 (#150)
      purpose_egg_storage: 0.00 (#151)
      purpose_donation: 173.27 (#96)
      purpose_research: 0.00 (#141)
      purpose_token_count: 133.72 (#101)
      purpose_current_and_storage: 0.00 (#153)
      purpose_zero: 0.00 (#152)
    [transfer_interaction]
      transfer_day_optimal: 529.26 (#75)
      transfer_day_cat: 5402.81 (#15)
      embryo_day_interaction: 8174.40 (#11)
      fresh_transfer_ratio: 2809.47 (#28)
      micro_transfer_quality: 2955.92 (#26)
      single_good_embryo: 142.17 (#100)
      frozen_embryo_signal: 14.71 (#127)
      transfer_intensity: 15959.84 (#5)
      age_transfer_interaction: 1422.18 (#53)
      day5plus: 105.08 (#105)
      single_x_day5plus: 6.89 (#131)
      multi_x_day5plus: 1223.91 (#60)
      fresh_x_day5plus: 2502.66 (#35)
      frozen_x_day5plus: 6.40 (#132)
      age_x_day5plus: 681.33 (#68)
      age_x_single_transfer: 1197.73 (#62)
      icsi_x_day5plus: 71.03 (#110)
      blastocyst_signal: 1408.73 (#54)
      donor_egg_x_advanced_age: 500.01 (#77)
    [failure]
      total_failure_count: 1705.76 (#44)
      ivf_failure_count: 5103.71 (#16)
      pregnancy_to_birth_gap: 265.64 (#88)
      ivf_preg_to_birth_gap: 179.93 (#94)
      repeated_failed_transfer: 1717.17 (#43)
      first_transfer_cycle: 66.85 (#112)

## [14] 파일 저장
  제출: C:\Users\ico26\oz\Infertility-Success-Classifier\suho\result\sample_submission_v16_20260317_162233.csv
  확률: mean=0.2587, std=0.1588, min=0.001442, max=0.7503
  npy 저장: oof/test final, y_train, transfer_masks
  모든 파일 저장 완료

## [15] 버전 비교
| 버전 | 모델 | OOF AUC | Test AUC | 비고 |
|------|------|---------|----------|------|
| v1 | CB원본 | 0.7403 | - | 베이스라인 |
| v7 | XGB+CB | 0.7402 | - | 블렌딩 |
| v9 | XGB+CB 3seed | 0.7405 | 0.74166 | +IVF/DI |
| v12 | CB+XGB+Tab | 0.7406 | - | 3모델 |
| v14 | CB 3seed | 0.7406 | - | 이식피처 |
| v15 | CB 5seed+Optuna | 0.7407 | 0.74169 | EDA+HP |
| **v16** | **Multi-model** | **0.7407** | **TBD** | **subgroup blend** |

============================================================
## 최종 요약
============================================================
- Full CB OOF AUC: 0.740592
- 최적 블렌딩: prob_blend
- 최종 OOF AUC: 0.740694
- 총 소요: 40.4분
- 데이터 누수: 없음
- 로그: C:\Users\ico26\oz\Infertility-Success-Classifier\suho\result\log_v16.md
============================================================