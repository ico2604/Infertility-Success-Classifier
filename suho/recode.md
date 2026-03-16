# 모델 실험 기록: {실험명}

## 1. 개요

| 항목 | 내용 |
|------|------|
| 실험명 | {실험명} |
| 작성자 | {이름} |
| 작성일 | {YYYY-MM-DD} |
| 목적 | {목적} |
| 데이터셋 | {데이터셋명} |
| Train / Test | {256351 / 90067} |

## 2. 모델 설정

| 파라미터 | 값 |
|----------|-----|
| 모델 | catboost |
| learning_rate | 0.05 |
| depth | 7 |
| l2_leaf_reg | 3 |
| min_data_in_leaf | 20 |
| eval_metric | AUC |
| random_seed | 42 |
| early_stopping_rounds | 100 |
| cat_features | 42 |
| auto_class_weights | 77 |

## 3. 결과

| 지표 | OOF |
|------|-------|
| Loss | 0.5858 |
| Accuracy | 0.7399 |
| F1-Score | 0.5173 |

## 4. 버전 이력

| 버전 | 날짜 | 변경사항 | Test Acc | Train ACC |
|------|------|----------|----------|------|
| v9 | 20260315 | CB, XGB Model IVF/DI분기 blend | 0.74166 | 0.7405 |

## 5. 메모

