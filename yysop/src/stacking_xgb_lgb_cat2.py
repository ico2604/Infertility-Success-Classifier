# =========================================================
# stacking_xgb_lgb_cat2_fixed.py
# ---------------------------------------------------------
# 목적
# 1) XGB / LGB / CAT2(v17) OOF 예측과 test 예측을 불러온다.
# 2) OOF 기반으로 stacking meta feature를 만든다.
# 3) Logistic / Ridge / LightGBM 메타모델을 비교한다.
# 4) OOF AUC를 계산하고 best stacking submission을 저장한다.
# =========================================================

import os
import re
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from lightgbm import LGBMClassifier, early_stopping, log_evaluation


# =========================================================
# 0. 경로 설정
# =========================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")

# ---------------------------------------------------------
# base model OOF / submission
# ---------------------------------------------------------
XGB_OOF_PATH = os.path.join(OUTPUT_DIR, "xgb_v2_reg_relax_oof_predictions.csv")
LGB_OOF_PATH = os.path.join(OUTPUT_DIR, "lightgbm_v1_lightgbm_baseline_v1_oof_predictions.csv")

XGB_SUB_PATH = os.path.join(OUTPUT_DIR, "xgb_v2_reg_relax_submission.csv")
LGB_SUB_PATH = os.path.join(OUTPUT_DIR, "lightgbm_v1_lightgbm_baseline_v1_submission.csv")

# ---------------------------------------------------------
# CAT2(v17)
# ---------------------------------------------------------
CAT2_OOF_NPY_PATH = os.path.join(OUTPUT_DIR, "oof_v17_final.npy")
CAT2_SUB_CSV_PATH = os.path.join(OUTPUT_DIR, "sample_submission_v17_20260317_171824.csv")
CAT2_SUB_NPY_PATH = os.path.join(OUTPUT_DIR, "test_v17_final.npy")

# ---------------------------------------------------------
# output files
# ---------------------------------------------------------
RESULT_LOG_PATH = os.path.join(OUTPUT_DIR, "stacking_results_log.csv")
META_OOF_SAVE_PATH = os.path.join(OUTPUT_DIR, "stacking_meta_oof.csv")
META_TEST_SAVE_PATH = os.path.join(OUTPUT_DIR, "stacking_meta_test.csv")

TARGET_COL = "임신 성공 여부"
ID_COL = "ID"
OOF_PRED_COL = "oof_pred_prob"
SUBMIT_TARGET_COL = "probability"

N_SPLITS = 5
RANDOM_STATE = 42


# =========================================================
# 1. 보조 함수
# =========================================================
def print_section(title: str):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def parse_age_text(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s in ["알 수 없음", "미상", "불명", "unknown", "Unknown"]:
        return np.nan
    nums = re.findall(r"\d+", s)
    if len(nums) >= 2:
        return (float(nums[0]) + float(nums[1])) / 2
    if len(nums) == 1:
        return float(nums[0])
    return np.nan


def load_oof_csv(path, model_name, keep_target=False):
    df = pd.read_csv(path)

    required_cols = [ID_COL, OOF_PRED_COL]
    if keep_target:
        required_cols.append(TARGET_COL)

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"[{model_name}] OOF 파일에 필요한 컬럼이 없습니다: {missing}")

    use_cols = [ID_COL, OOF_PRED_COL]
    if keep_target:
        use_cols = [ID_COL, TARGET_COL, OOF_PRED_COL]

    out = df[use_cols].copy()
    out = out.rename(columns={OOF_PRED_COL: model_name})
    return out


def load_submission_csv(path, model_name):
    df = pd.read_csv(path)

    if ID_COL not in df.columns:
        raise ValueError(f"[{model_name}] submission 파일에 ID 컬럼이 없습니다.")
    pred_col = [c for c in df.columns if c != ID_COL]
    if not pred_col:
        raise ValueError(f"[{model_name}] submission 파일에 예측값 컬럼이 없습니다.")

    out = df[[ID_COL, pred_col[0]]].copy()
    out = out.rename(columns={pred_col[0]: model_name})
    return out


def load_cat2_oof_from_npy(oof_npy_path, train_path):
    train_df = pd.read_csv(train_path, usecols=[ID_COL, TARGET_COL])
    pred = np.load(oof_npy_path)

    if len(train_df) != len(pred):
        raise ValueError(
            f"[cat2] train 행 수({len(train_df)})와 OOF npy 길이({len(pred)})가 다릅니다."
        )

    out = train_df.copy()
    out["cat2"] = pred
    return out


def load_cat2_submission(sub_csv_path, sub_npy_path, test_path):
    if os.path.exists(sub_csv_path):
        df = pd.read_csv(sub_csv_path)
        required = [ID_COL, SUBMIT_TARGET_COL]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"[cat2] csv 제출 파일에 필요한 컬럼이 없습니다: {missing}")
        return df[[ID_COL, SUBMIT_TARGET_COL]].rename(columns={SUBMIT_TARGET_COL: "cat2"})

    test_df = pd.read_csv(test_path, usecols=[ID_COL])
    pred = np.load(sub_npy_path)

    if len(test_df) != len(pred):
        raise ValueError(
            f"[cat2] test 행 수({len(test_df)})와 npy 길이({len(pred)})가 다릅니다."
        )

    out = test_df.copy()
    out["cat2"] = pred
    return out


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def make_meta_features(base_df, raw_df):
    """
    base_df: OOF/sub merged df (ID + xgb/lgb/cat2 [+ target])
    raw_df : train/test raw subset with ID and a few raw columns only
    """
    df = base_df.merge(raw_df, on=ID_COL, how="left").copy()

    # base predictions
    df["xgb_lgb_mean"] = (df["xgb"] + df["lgb"]) / 2.0
    df["xgb_cat2_mean"] = (df["xgb"] + df["cat2"]) / 2.0
    df["lgb_cat2_mean"] = (df["lgb"] + df["cat2"]) / 2.0
    df["base_mean"] = (df["xgb"] + df["lgb"] + df["cat2"]) / 3.0

    df["xgb_cat2_gap"] = np.abs(df["xgb"] - df["cat2"])
    df["lgb_cat2_gap"] = np.abs(df["lgb"] - df["cat2"])
    df["xgb_lgb_gap"] = np.abs(df["xgb"] - df["lgb"])

    # vote-like signals
    df["cat2_is_high"] = (df["cat2"] >= 0.5).astype(int)
    df["xgb_is_high"] = (df["xgb"] >= 0.5).astype(int)
    df["lgb_is_high"] = (df["lgb"] >= 0.5).astype(int)
    df["vote_sum"] = df["cat2_is_high"] + df["xgb_is_high"] + df["lgb_is_high"]

    # raw meta features
    if "이식된 배아 수" in df.columns:
        df["이식된 배아 수"] = pd.to_numeric(df["이식된 배아 수"], errors="coerce").fillna(0)
    else:
        df["이식된 배아 수"] = 0

    if "동결 배아 사용 여부" in df.columns:
        df["동결 배아 사용 여부"] = pd.to_numeric(df["동결 배아 사용 여부"], errors="coerce").fillna(0)
    else:
        df["동결 배아 사용 여부"] = 0

    if "배아 이식 경과일" in df.columns:
        df["배아 이식 경과일"] = pd.to_numeric(df["배아 이식 경과일"], errors="coerce").fillna(0)
    else:
        df["배아 이식 경과일"] = 0

    if "시술 당시 나이" in df.columns:
        df["시술 당시 나이_num"] = df["시술 당시 나이"].apply(parse_age_text).fillna(-1)
    else:
        df["시술 당시 나이_num"] = -1

    df["실제이식여부"] = (df["이식된 배아 수"] > 0).astype(int)
    df["day5plus"] = (df["배아 이식 경과일"] >= 5).astype(int)

    # interactions
    df["cat2_x_transfer"] = df["cat2"] * df["실제이식여부"]
    df["cat2_x_frozen"] = df["cat2"] * df["동결 배아 사용 여부"]
    df["cat2_x_age"] = df["cat2"] * np.clip(df["시술 당시 나이_num"], 0, None)
    df["xgb_x_transfer"] = df["xgb"] * df["실제이식여부"]
    df["lgb_x_transfer"] = df["lgb"] * df["실제이식여부"]

    meta_cols = [
        "xgb", "lgb", "cat2",
        "xgb_lgb_mean", "xgb_cat2_mean", "lgb_cat2_mean", "base_mean",
        "xgb_cat2_gap", "lgb_cat2_gap", "xgb_lgb_gap",
        "vote_sum",
        "실제이식여부", "이식된 배아 수", "동결 배아 사용 여부",
        "배아 이식 경과일", "시술 당시 나이_num", "day5plus",
        "cat2_x_transfer", "cat2_x_frozen", "cat2_x_age",
        "xgb_x_transfer", "lgb_x_transfer",
    ]

    keep_cols = [ID_COL] + meta_cols
    if TARGET_COL in df.columns:
        keep_cols = [ID_COL, TARGET_COL] + meta_cols

    return df[keep_cols].copy(), meta_cols


# =========================================================
# 2. 데이터 로드
# =========================================================
print_section("2. 데이터 로드")

train_raw = pd.read_csv(TRAIN_PATH)
test_raw = pd.read_csv(TEST_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

# base OOF / submission
xgb_oof = load_oof_csv(XGB_OOF_PATH, "xgb", keep_target=True)
lgb_oof = load_oof_csv(LGB_OOF_PATH, "lgb", keep_target=False)
cat2_oof = load_cat2_oof_from_npy(CAT2_OOF_NPY_PATH, TRAIN_PATH)

xgb_sub = load_submission_csv(XGB_SUB_PATH, "xgb")
lgb_sub = load_submission_csv(LGB_SUB_PATH, "lgb")
cat2_sub = load_cat2_submission(CAT2_SUB_CSV_PATH, CAT2_SUB_NPY_PATH, TEST_PATH)

# merge
oof_base = xgb_oof.merge(lgb_oof, on=ID_COL, how="inner")
oof_base = oof_base.merge(cat2_oof, on=[ID_COL, TARGET_COL], how="inner")

test_base = xgb_sub.merge(lgb_sub, on=ID_COL, how="inner")
test_base = test_base.merge(cat2_sub, on=ID_COL, how="inner")

print("OOF base shape :", oof_base.shape)
print("TEST base shape:", test_base.shape)

# ---------------------------------------------------------
# 중요: train_meta_raw에는 TARGET_COL 넣지 않음
# ---------------------------------------------------------
train_meta_raw = train_raw[[c for c in [
    ID_COL, "시술 당시 나이", "이식된 배아 수", "동결 배아 사용 여부", "배아 이식 경과일"
] if c in train_raw.columns]].copy()

test_meta_raw = test_raw[[c for c in [
    ID_COL, "시술 당시 나이", "이식된 배아 수", "동결 배아 사용 여부", "배아 이식 경과일"
] if c in test_raw.columns]].copy()

oof_meta_df, meta_cols = make_meta_features(oof_base, train_meta_raw)
test_meta_df, _ = make_meta_features(test_base, test_meta_raw)

oof_meta_df.to_csv(META_OOF_SAVE_PATH, index=False, encoding="utf-8-sig")
test_meta_df.to_csv(META_TEST_SAVE_PATH, index=False, encoding="utf-8-sig")

print("OOF meta shape :", oof_meta_df.shape)
print("TEST meta shape:", test_meta_df.shape)
print("meta cols      :", len(meta_cols))
print("Has target?    :", TARGET_COL in oof_meta_df.columns)

X = oof_meta_df[meta_cols].copy()
X_test = test_meta_df[meta_cols].copy()
y = oof_meta_df[TARGET_COL].values


# =========================================================
# 3. 기준 성능
# =========================================================
print_section("3. 기준 성능")

for col in ["xgb", "lgb", "cat2", "base_mean"]:
    auc = roc_auc_score(y, oof_meta_df[col].values)
    print(f"{col:>12} AUC: {auc:.6f}")


# =========================================================
# 4. Meta model 1: Logistic Regression
# =========================================================
print_section("4. Logistic Regression Stacking")

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

oof_logit = np.zeros(len(X))
pred_logit = np.zeros(len(X_test))

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
    X_tr = X.iloc[tr_idx].copy()
    X_va = X.iloc[va_idx].copy()
    y_tr = y[tr_idx]
    y_va = y[va_idx]

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_va_sc = scaler.transform(X_va)
    X_te_sc = scaler.transform(X_test)

    model = LogisticRegression(
        C=0.5,
        max_iter=2000,
        random_state=RANDOM_STATE,
        solver="liblinear"
    )
    model.fit(X_tr_sc, y_tr)

    va_pred = model.predict_proba(X_va_sc)[:, 1]
    te_pred = model.predict_proba(X_te_sc)[:, 1]

    oof_logit[va_idx] = va_pred
    pred_logit += te_pred / N_SPLITS

    fold_auc = roc_auc_score(y_va, va_pred)
    print(f"Fold {fold} AUC: {fold_auc:.6f}")

logit_auc = roc_auc_score(y, oof_logit)
print(f"\nLogistic Stacking OOF AUC: {logit_auc:.6f}")


# =========================================================
# 5. Meta model 2: Ridge
# =========================================================
print_section("5. Ridge Stacking")

oof_ridge = np.zeros(len(X))
pred_ridge = np.zeros(len(X_test))

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
    X_tr = X.iloc[tr_idx].copy()
    X_va = X.iloc[va_idx].copy()
    y_tr = y[tr_idx]
    y_va = y[va_idx]

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_va_sc = scaler.transform(X_va)
    X_te_sc = scaler.transform(X_test)

    model = Ridge(alpha=3.0, random_state=RANDOM_STATE)
    model.fit(X_tr_sc, y_tr)

    va_raw = model.predict(X_va_sc)
    te_raw = model.predict(X_te_sc)

    va_pred = sigmoid(va_raw)
    te_pred = sigmoid(te_raw)

    oof_ridge[va_idx] = va_pred
    pred_ridge += te_pred / N_SPLITS

    fold_auc = roc_auc_score(y_va, va_pred)
    print(f"Fold {fold} AUC: {fold_auc:.6f}")

ridge_auc = roc_auc_score(y, oof_ridge)
print(f"\nRidge Stacking OOF AUC: {ridge_auc:.6f}")


# =========================================================
# 6. Meta model 3: Small LightGBM
# =========================================================
print_section("6. Small LightGBM Stacking")

oof_meta_lgb = np.zeros(len(X))
pred_meta_lgb = np.zeros(len(X_test))

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
    X_tr = X.iloc[tr_idx].copy()
    X_va = X.iloc[va_idx].copy()
    y_tr = y[tr_idx]
    y_va = y[va_idx]

    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=7,
        max_depth=3,
        min_child_samples=80,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.5,
        reg_lambda=3.0,
        objective="binary",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="auc",
        callbacks=[
            early_stopping(50, first_metric_only=True),
            log_evaluation(0)
        ]
    )

    va_pred = model.predict_proba(X_va)[:, 1]
    te_pred = model.predict_proba(X_test)[:, 1]

    oof_meta_lgb[va_idx] = va_pred
    pred_meta_lgb += te_pred / N_SPLITS

    fold_auc = roc_auc_score(y_va, va_pred)
    print(f"Fold {fold} AUC: {fold_auc:.6f}")

meta_lgb_auc = roc_auc_score(y, oof_meta_lgb)
print(f"\nSmall LGB Stacking OOF AUC: {meta_lgb_auc:.6f}")


# =========================================================
# 7. 결과 비교
# =========================================================
print_section("7. 결과 비교")

results = pd.DataFrame([
    {"model": "xgb", "oof_auc": roc_auc_score(y, oof_meta_df["xgb"].values)},
    {"model": "lgb", "oof_auc": roc_auc_score(y, oof_meta_df["lgb"].values)},
    {"model": "cat2", "oof_auc": roc_auc_score(y, oof_meta_df["cat2"].values)},
    {"model": "base_mean", "oof_auc": roc_auc_score(y, oof_meta_df["base_mean"].values)},
    {"model": "stack_logistic", "oof_auc": logit_auc},
    {"model": "stack_ridge", "oof_auc": ridge_auc},
    {"model": "stack_small_lgb", "oof_auc": meta_lgb_auc},
]).sort_values("oof_auc", ascending=False).reset_index(drop=True)

print(results.to_string(index=False))

best_model_name = results.iloc[0]["model"]

if best_model_name == "stack_logistic":
    best_oof = oof_logit
    best_pred = pred_logit
elif best_model_name == "stack_ridge":
    best_oof = oof_ridge
    best_pred = pred_ridge
elif best_model_name == "stack_small_lgb":
    best_oof = oof_meta_lgb
    best_pred = pred_meta_lgb
elif best_model_name == "cat2":
    best_oof = oof_meta_df["cat2"].values
    best_pred = test_meta_df["cat2"].values
elif best_model_name == "xgb":
    best_oof = oof_meta_df["xgb"].values
    best_pred = test_meta_df["xgb"].values
elif best_model_name == "lgb":
    best_oof = oof_meta_df["lgb"].values
    best_pred = test_meta_df["lgb"].values
else:
    best_oof = oof_meta_df["base_mean"].values
    best_pred = test_meta_df["base_mean"].values


# =========================================================
# 8. 저장
# =========================================================
print_section("8. 저장")

timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

oof_save_path = os.path.join(OUTPUT_DIR, f"stacking_best_oof_{best_model_name}_{timestamp}.csv")
sub_save_path = os.path.join(OUTPUT_DIR, f"stacking_best_submission_{best_model_name}_{timestamp}.csv")

oof_save_df = pd.DataFrame({
    ID_COL: oof_meta_df[ID_COL].values,
    TARGET_COL: y,
    "stacking_oof_pred": best_oof
})
oof_save_df.to_csv(oof_save_path, index=False, encoding="utf-8-sig")

submission = sample_sub.copy()
submission[SUBMIT_TARGET_COL] = best_pred
submission.to_csv(sub_save_path, index=False, encoding="utf-8-sig")

print(f"Best model      : {best_model_name}")
print(f"OOF save path   : {oof_save_path}")
print(f"SUB save path   : {sub_save_path}")

log_row = {
    "timestamp": timestamp,
    "best_model": best_model_name,
    "xgb_auc": round(float(roc_auc_score(y, oof_meta_df["xgb"].values)), 6),
    "lgb_auc": round(float(roc_auc_score(y, oof_meta_df["lgb"].values)), 6),
    "cat2_auc": round(float(roc_auc_score(y, oof_meta_df["cat2"].values)), 6),
    "base_mean_auc": round(float(roc_auc_score(y, oof_meta_df["base_mean"].values)), 6),
    "stack_logistic_auc": round(float(logit_auc), 6),
    "stack_ridge_auc": round(float(ridge_auc), 6),
    "stack_small_lgb_auc": round(float(meta_lgb_auc), 6),
    "meta_feature_count": len(meta_cols),
}

log_df = pd.DataFrame([log_row])
if os.path.exists(RESULT_LOG_PATH):
    old = pd.read_csv(RESULT_LOG_PATH)
    log_df = pd.concat([old, log_df], ignore_index=True)

log_df.to_csv(RESULT_LOG_PATH, index=False, encoding="utf-8-sig")
print(f"LOG save path   : {RESULT_LOG_PATH}")


# =========================================================
# 9. 완료
# =========================================================
print_section("9. 완료")
print(f"Best meta model : {best_model_name}")
print(results.to_string(index=False))