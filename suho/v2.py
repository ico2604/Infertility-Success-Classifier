#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v2.py - CatBoost cat_features 활용 + 하이퍼파라미터 탐색 + 10-Seed 앙상블
       평가지표: ROC-AUC (확률 제출)
"""

import os, sys, warnings, numpy as np, pandas as pd
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from catboost import CatBoostClassifier, Pool

warnings.filterwarnings("ignore")

VERSION = "v2"
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
print(f"{VERSION} - CatBoost cat_features + HP탐색 + 10-Seed 앙상블")
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

# ============================================================
# 2. 전처리 - CatBoost 네이티브 카테고리 활용
# ============================================================
print("\n[2] 전처리 (CatBoost 네이티브 카테고리)")

train.drop(columns=["ID", TARGET], inplace=True)
test.drop(columns=["ID"], inplace=True)

common_cols = list(train.columns)

# 카테고리 / 수치 구분
cat_cols = []
num_cols = []
for col in common_cols:
    if train[col].dtype == "object":
        cat_cols.append(col)
    else:
        num_cols.append(col)

# 카테고리: LabelEncoding 하지 않음! 문자열 그대로 유지
# CatBoost가 자체 ordered target encoding 수행
# 단, NaN은 문자열로 변환
for col in cat_cols:
    train[col] = train[col].fillna("_missing_").astype(str)
    test[col] = test[col].fillna("_missing_").astype(str)

# 수치: NaN 그대로 (CatBoost 내부 처리)
cat_indices = [common_cols.index(c) for c in cat_cols]

print(f"  카테고리: {len(cat_cols)}개 → CatBoost 네이티브 처리")
print(f"  수치: {len(num_cols)}개 → NaN 그대로")
print(f"  총 피처: {len(common_cols)}개")
print(f"  cat_features 인덱스: {cat_indices}")

X_train = train[common_cols]
X_test = test[common_cols]

# ============================================================
# 3. 하이퍼파라미터 탐색 (3-Fold, 4개 설정 비교)
# ============================================================
print("\n[3] 하이퍼파라미터 탐색 (3-Fold quick search)")

param_candidates = [
    {
        "depth": 6,
        "learning_rate": 0.03,
        "l2_leaf_reg": 3,
        "min_data_in_leaf": 20,
        "name": "기본(v1)",
    },
    {
        "depth": 8,
        "learning_rate": 0.01,
        "l2_leaf_reg": 3,
        "min_data_in_leaf": 20,
        "name": "깊고느린",
    },
    {
        "depth": 7,
        "learning_rate": 0.02,
        "l2_leaf_reg": 5,
        "min_data_in_leaf": 50,
        "name": "중간정규화",
    },
    {
        "depth": 8,
        "learning_rate": 0.02,
        "l2_leaf_reg": 1,
        "min_data_in_leaf": 10,
        "name": "깊고약한정규",
    },
]

skf_search = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
search_results = []

for pidx, params in enumerate(param_candidates):
    pname = params.pop("name")
    aucs = []

    print(f"\n  [{pidx+1}/4] {pname}: {params}")

    for fold, (tr_i, va_i) in enumerate(skf_search.split(X_train, y)):
        tr_pool = Pool(X_train.iloc[tr_i], y[tr_i], cat_features=cat_indices)
        va_pool = Pool(X_train.iloc[va_i], y[va_i], cat_features=cat_indices)

        model = CatBoostClassifier(
            iterations=3000,
            **params,
            subsample=0.8,
            colsample_bylevel=0.8,
            eval_metric="AUC",
            random_seed=SEED,
            early_stopping_rounds=150,
            verbose=0,
            task_type="CPU",
        )
        model.fit(tr_pool, eval_set=va_pool, use_best_model=True)
        pred = model.predict_proba(va_pool)[:, 1]
        auc = roc_auc_score(y[va_i], pred)
        aucs.append(auc)

    mean_auc = np.mean(aucs)
    search_results.append({"name": pname, "params": params.copy(), "auc": mean_auc})
    print(f"    → 3-Fold AUC: {[f'{a:.4f}' for a in aucs]}, 평균: {mean_auc:.4f}")

# 최적 파라미터 선택
search_results.sort(key=lambda x: x["auc"], reverse=True)
print(f"\n  === 하이퍼파라미터 탐색 결과 ===")
for r in search_results:
    print(f"    {r['name']}: AUC={r['auc']:.4f}")
best_params = search_results[0]["params"]
best_name = search_results[0]["name"]
print(f"\n  최적: {best_name} → {best_params}")

# ============================================================
# 4. 10-Seed 앙상블 (최적 파라미터, 5-Fold)
# ============================================================
SEEDS = [42, 123, 456, 777, 999, 1234, 2024, 3141, 4567, 7890]
print(f"\n[4] {len(SEEDS)}-Seed 앙상블 (최적 파라미터, {N_FOLDS}-Fold)")
print(f"  Seeds: {SEEDS}")

all_oof = []
all_test = []

for seed_idx, seed in enumerate(SEEDS):
    print(f"\n  --- Seed {seed} ({seed_idx+1}/{len(SEEDS)}) ---")

    oof_pred = np.zeros(len(y))
    test_pred = np.zeros(len(X_test))
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

    fold_aucs = []
    for fold, (tr_i, va_i) in enumerate(skf.split(X_train, y)):
        tr_pool = Pool(X_train.iloc[tr_i], y[tr_i], cat_features=cat_indices)
        va_pool = Pool(X_train.iloc[va_i], y[va_i], cat_features=cat_indices)
        te_pool = Pool(X_test, cat_features=cat_indices)

        model = CatBoostClassifier(
            iterations=5000,
            **best_params,
            subsample=0.8,
            colsample_bylevel=0.8,
            eval_metric="AUC",
            random_seed=seed + fold,
            early_stopping_rounds=200,
            verbose=0,
            task_type="CPU",
        )
        model.fit(tr_pool, eval_set=va_pool, use_best_model=True)

        pred = model.predict_proba(va_pool)[:, 1]
        oof_pred[va_i] = pred
        test_pred += model.predict_proba(te_pool)[:, 1] / N_FOLDS

        auc = roc_auc_score(y[va_i], pred)
        fold_aucs.append(auc)

    oof_auc = roc_auc_score(y, oof_pred)
    fold_str = ", ".join([f"{a:.4f}" for a in fold_aucs])
    print(f"    Folds: [{fold_str}], OOF AUC: {oof_auc:.4f}")

    all_oof.append(oof_pred)
    all_test.append(test_pred)

# ============================================================
# 5. 앙상블 결과
# ============================================================
print(f"\n[5] 앙상블 결과")

# 개별 시드 AUC
print(f"\n  개별 Seed OOF AUC:")
seed_aucs = []
for i, seed in enumerate(SEEDS):
    auc = roc_auc_score(y, all_oof[i])
    seed_aucs.append(auc)
    print(f"    Seed {seed}: {auc:.4f}")

# 누적 앙상블 (시드 추가할수록 어떻게 변하는지)
print(f"\n  누적 앙상블 AUC:")
for k in range(1, len(SEEDS) + 1):
    oof_k = np.mean(all_oof[:k], axis=0)
    auc_k = roc_auc_score(y, oof_k)
    print(f"    {k}개 시드: {auc_k:.4f}")

# 최종 앙상블
oof_final = np.mean(all_oof, axis=0)
test_final = np.mean(all_test, axis=0)
oof_auc_final = roc_auc_score(y, oof_final)

print(f"\n  ★ 최종 {len(SEEDS)}-Seed 앙상블 OOF AUC: {oof_auc_final:.4f}")

# ============================================================
# 6. 확률 Calibration (Platt Scaling)
# ============================================================
print(f"\n[6] 확률 Calibration")

from sklearn.linear_model import LogisticRegression

# OOF 확률로 calibration 모델 학습
cal_model = LogisticRegression(C=1.0, max_iter=1000)
cal_model.fit(oof_final.reshape(-1, 1), y)

oof_calibrated = cal_model.predict_proba(oof_final.reshape(-1, 1))[:, 1]
test_calibrated = cal_model.predict_proba(test_final.reshape(-1, 1))[:, 1]

oof_auc_cal = roc_auc_score(y, oof_calibrated)
print(f"  Calibration 전 OOF AUC: {oof_auc_final:.4f}")
print(f"  Calibration 후 OOF AUC: {oof_auc_cal:.4f}")

# calibration이 AUC를 올렸으면 사용, 아니면 원본
if oof_auc_cal > oof_auc_final:
    final_test = test_calibrated
    final_oof_auc = oof_auc_cal
    cal_used = True
    print(f"  → Calibration 적용 (+{oof_auc_cal - oof_auc_final:.4f})")
else:
    final_test = test_final
    final_oof_auc = oof_auc_final
    cal_used = False
    print(f"  → Calibration 미적용 (효과 없음)")

# ============================================================
# 7. 제출 파일
# ============================================================
print(f"\n[7] 제출 파일 생성")

submission = pd.DataFrame({"ID": test_ids, "probability": final_test})

main_path = os.path.join(RESULT_DIR, f"sample_submission_{VERSION}_{NOW}.csv")
submission.to_csv(main_path, index=False)

print(f"  파일: {main_path}")
print(f"  확률 통계:")
print(f"    mean={final_test.mean():.4f}, std={final_test.std():.4f}")
print(f"    min={final_test.min():.4f}, max={final_test.max():.4f}")
print(f"    median={np.median(final_test):.4f}")
print(f"  예시:")
print(submission.head(10).to_string(index=False))

# ============================================================
# 8. 피처 중요도
# ============================================================
print(f"\n[8] 피처 중요도 (마지막 모델)")
fi = model.get_feature_importance()
fi_df = pd.DataFrame({"feature": common_cols, "importance": fi})
fi_df = fi_df.sort_values("importance", ascending=False)
for i, (_, r) in enumerate(fi_df.head(20).iterrows()):
    tag = " [CAT]" if r["feature"] in cat_cols else ""
    print(f"  {i+1}. {r['feature']}{tag}: {r['importance']:.2f}")

# ============================================================
# 9. 최종 요약
# ============================================================
print(f"\n{'='*60}")
print(f"최종 요약")
print(f"{'='*60}")
print(f"  최적 하이퍼파라미터: {best_name}")
print(f"    {best_params}")
print(f"  cat_features: {len(cat_cols)}개 네이티브 처리")
print(f"  피처 수: {len(common_cols)}개 (원본 그대로)")
print(f"  Seed 앙상블: {len(SEEDS)}개")
print(f"    개별 AUC 범위: {min(seed_aucs):.4f} ~ {max(seed_aucs):.4f}")
print(f"    앙상블 OOF AUC: {oof_auc_final:.4f}")
print(f"  Calibration: {'적용' if cal_used else '미적용'}")
print(f"  최종 OOF AUC: {final_oof_auc:.4f}")
print(f"  v1 대비 변화: {final_oof_auc - 0.7403:+.4f}")
print(f"  제출: 확률 그대로 (ROC-AUC 평가)")
print(f"  로그: {LOG_PATH}")
print(f"{'='*60}")
