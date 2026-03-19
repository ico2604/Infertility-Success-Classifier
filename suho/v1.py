#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v8.py - 심플 CatBoost + Seed 앙상블
       평가지표: ROC-AUC (확률 제출)
       원본 피처 그대로, 파생변수 없음
"""

import os, sys, warnings, numpy as np, pandas as pd
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

VERSION = "v8"
SEED = 42
N_FOLDS = 5
TARGET = "임신 성공 여부"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
RESULT_DIR = os.path.join(BASE_DIR, "result")
os.makedirs(RESULT_DIR, exist_ok=True)
NOW = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_PATH = os.path.join(RESULT_DIR, f"log_{VERSION}_{NOW}.txt")


class Logger:
    def __init__(self, fp):
        self.terminal = sys.stdout
        self.log = open(fp, "w", encoding="utf-8")

    def write(self, msg):
        self.terminal.write(msg)
        clean = msg.replace("\r", "")
        if clean.strip():
            self.log.write(clean)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


sys.stdout = Logger(LOG_PATH)
print("=" * 60)
print(f"{VERSION} - 심플 CatBoost + Seed 앙상블 (ROC-AUC)")
print(f"시각: {NOW}")
print("=" * 60)

# ============================================================
# 1. 데이터 로드
# ============================================================
print("\n[1] 데이터 로드")
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
sub = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

y = train[TARGET].values
test_ids = test["ID"].values

print(f"  train: {train.shape}, test: {test.shape}")
print(f"  타겟: 0={np.sum(y==0)}, 1={np.sum(y==1)}, 양성비율={y.mean()*100:.1f}%")
print(f"  sub 컬럼: {list(sub.columns)}, sub 예시: {sub.iloc[:3].values.tolist()}")

# ============================================================
# 2. 전처리 - 최소한만
# ============================================================
print("\n[2] 전처리")

train.drop(columns=["ID", TARGET], inplace=True)
test.drop(columns=["ID"], inplace=True)

# train과 test 컬럼 일치 확인
common_cols = list(train.columns)
assert list(train.columns) == list(test.columns), "컬럼 불일치!"

# 카테고리 / 수치 구분
cat_cols = []
num_cols = []
for col in common_cols:
    if train[col].dtype == "object":
        cat_cols.append(col)
    else:
        num_cols.append(col)

# 카테고리: 결측 채우고 LabelEncoding
for col in cat_cols:
    train[col] = train[col].fillna("_missing_")
    test[col] = test[col].fillna("_missing_")
    le = LabelEncoder()
    le.fit(pd.concat([train[col], test[col]]))
    train[col] = le.transform(train[col])
    test[col] = le.transform(test[col])

# 수치: NaN은 CatBoost가 자체 처리하므로 그대로 둠
# (CatBoost는 NaN을 내부적으로 처리 가능)

print(f"  카테고리: {len(cat_cols)}개, 수치: {len(num_cols)}개")
print(f"  총 피처: {len(common_cols)}개")

X_train = train.values.astype(np.float32)
X_test = test.values.astype(np.float32)

# ============================================================
# 3. 3-Seed 앙상블 CatBoost (각 5-Fold)
# ============================================================
SEEDS = [42, 123, 777]
all_oof = []
all_test = []

for seed_idx, seed in enumerate(SEEDS):
    print(f"\n[3-{seed_idx+1}] Seed={seed} 학습")

    oof_pred = np.zeros(len(y))
    test_pred = np.zeros(len(X_test))
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

    for fold, (tr_i, va_i) in enumerate(skf.split(X_train, y)):
        print(f"\n  Seed={seed}, Fold {fold+1}/{N_FOLDS}")

        model = CatBoostClassifier(
            iterations=5000,
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=3,
            min_data_in_leaf=20,
            subsample=0.8,
            colsample_bylevel=0.8,
            eval_metric="AUC",
            random_seed=seed + fold,
            early_stopping_rounds=200,
            verbose=500,
            task_type="CPU",
        )

        model.fit(
            X_train[tr_i],
            y[tr_i],
            eval_set=(X_train[va_i], y[va_i]),
            use_best_model=True,
        )

        pred = model.predict_proba(X_train[va_i])[:, 1]
        oof_pred[va_i] = pred
        test_pred += model.predict_proba(X_test)[:, 1] / N_FOLDS

        auc = roc_auc_score(y[va_i], pred)
        print(f"    AUC: {auc:.4f}, best_iter: {model.best_iteration_}")

    oof_auc = roc_auc_score(y, oof_pred)
    print(f"\n  Seed={seed} OOF AUC: {oof_auc:.4f}")
    all_oof.append(oof_pred)
    all_test.append(test_pred)

# 시드 평균
oof_final = np.mean(all_oof, axis=0)
test_final = np.mean(all_test, axis=0)
oof_auc_final = roc_auc_score(y, oof_final)

print(f"\n{'='*60}")
print(f"  Seed 앙상블 OOF AUC: {oof_auc_final:.4f}")
print(f"{'='*60}")

# ============================================================
# 4. 제출 파일 (확률 그대로 - ROC-AUC 평가)
# ============================================================
print(f"\n[4] 제출 파일 생성")

submission = pd.DataFrame({sub.columns[0]: test_ids, sub.columns[1]: test_final})

main_path = os.path.join(RESULT_DIR, f"sample_submission_{VERSION}_{NOW}.csv")
submission.to_csv(main_path, index=False)

print(f"  파일: {main_path}")
print(f"  예측 확률 통계:")
print(f"    mean={test_final.mean():.4f}, std={test_final.std():.4f}")
print(f"    min={test_final.min():.4f}, max={test_final.max():.4f}")
print(f"    median={np.median(test_final):.4f}")
print(f"  예시 (상위 5개):")
print(submission.head().to_string(index=False))

# ============================================================
# 5. 피처 중요도
# ============================================================
print(f"\n[5] 피처 중요도 (마지막 모델)")
fi = model.get_feature_importance()
fi_df = pd.DataFrame({"feature": common_cols, "importance": fi})
fi_df = fi_df.sort_values("importance", ascending=False)
for i, (_, r) in enumerate(fi_df.head(20).iterrows()):
    print(f"  {i+1}. {r['feature']}: {r['importance']:.2f}")

# ============================================================
# 6. 요약
# ============================================================
print(f"\n{'='*60}")
print(f"최종 요약")
print(f"{'='*60}")
for i, seed in enumerate(SEEDS):
    auc_s = roc_auc_score(y, all_oof[i])
    print(f"  Seed {seed} OOF AUC: {auc_s:.4f}")
print(f"  3-Seed 앙상블 OOF AUC: {oof_auc_final:.4f}")
print(f"  피처 수: {len(common_cols)} (원본 그대로)")
print(f"  모델: CatBoost (iter=5000, lr=0.03, depth=6)")
print(f"  제출: 확률 그대로 (ROC-AUC 평가)")
print(f"  로그: {LOG_PATH}")
print(f"{'='*60}")
