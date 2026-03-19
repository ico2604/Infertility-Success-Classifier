# =========================================================
# ensemble_v3_stacking.py
# ---------------------------------------------------------
# 목적
# 1) XGB / CAT / LGB OOF 예측 파일을 불러온다.
# 2) stacking용 meta feature를 생성한다.
# 3) LogisticRegression meta model에서 C 여러 개를 자동 비교한다.
# 4) weighted ensemble baseline(0.740440)과 직접 비교한다.
# 5) best C 기준으로 stacking OOF / submission / results log를 저장한다.
#
# 실행 예시
#   python yysop/src/ensemble_v3_stacking.py
# =========================================================

import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")


# =========================================================
# 0. 경로 설정
# =========================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# XGB (Optuna)
XGB_OOF_PATH = os.path.join(OUTPUT_DIR, "xgb_optuna_v1_oof_predictions.csv")
XGB_SUB_PATH = os.path.join(OUTPUT_DIR, "xgb_optuna_v1_submission.csv")

#XGB_OOF_PATH = os.path.join(OUTPUT_DIR, "xgb_v5_optuna_branch_v5_dedup_oof_predictions.csv")
CAT_OOF_PATH = os.path.join(OUTPUT_DIR, "catboost_v2_catboost_baseline_v2_oof_predictions.csv")
LGB_OOF_PATH = os.path.join(OUTPUT_DIR, "lightgbm_v1_lightgbm_baseline_v1_oof_predictions.csv")

#XGB_SUB_PATH = os.path.join(OUTPUT_DIR, "xgb_v5_optuna_branch_v5_dedup_submission.csv")
CAT_SUB_PATH = os.path.join(OUTPUT_DIR, "catboost_v2_catboost_baseline_v2_submission.csv")
LGB_SUB_PATH = os.path.join(OUTPUT_DIR, "lightgbm_v1_lightgbm_baseline_v1_submission.csv")

RESULT_LOG_PATH = os.path.join(OUTPUT_DIR, "ensemble_v3_stacking_results_log.csv")
C_SEARCH_RESULT_PATH = os.path.join(OUTPUT_DIR, "ensemble_v3_stacking_c_search_results.csv")
META_FEATURE_PATH = os.path.join(OUTPUT_DIR, "ensemble_v3_stacking_meta_features.csv")

TARGET_COL = "임신 성공 여부"
ID_COL = "ID"
OOF_PRED_COL = "oof_pred_prob"

SEED = 42
N_FOLDS = 5

# 현재 확정 baseline
BASELINE_AUC = 0.740440
BASELINE_NAME = "best_weighted_ensemble"
BASELINE_WEIGHTS = "xgb=0.31, cat=0.43, lgb=0.26"

# C 자동 비교 후보
C_LIST = [1.0, 0.3, 0.1, 0.03, 0.01]

# meta feature 모드
# "basic"   : xgb_pred, cat_pred, lgb_pred
# "extended": basic + mean/std/max/min/gap/is_max
META_FEATURE_MODE = "extended"

EXPERIMENT_NAME = "ensemble_v3_stacking_lr_multiC"


# =========================================================
# 1. 보조 함수
# =========================================================
def print_section(title: str):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def load_oof_file(path, model_name, keep_target=False):
    df = pd.read_csv(path)

    required_cols = [ID_COL, OOF_PRED_COL]
    if keep_target:
        required_cols.append(TARGET_COL)

    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"[{model_name}] OOF 파일에 필요한 컬럼이 없습니다: {missing_cols}")

    use_cols = [ID_COL, OOF_PRED_COL]
    if keep_target:
        use_cols = [ID_COL, TARGET_COL, OOF_PRED_COL]

    out = df[use_cols].copy()
    out = out.rename(columns={OOF_PRED_COL: f"{model_name}_pred"})
    return out


def load_submission_file(path, model_name):
    df = pd.read_csv(path)

    required_cols = [ID_COL, TARGET_COL]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"[{model_name}] submission 파일에 필요한 컬럼이 없습니다: {missing_cols}")

    out = df[[ID_COL, TARGET_COL]].copy()
    out = out.rename(columns={TARGET_COL: f"{model_name}_pred"})
    return out


def merge_oof_dfs(xgb, cat, lgb):
    merged = xgb.merge(cat, on=ID_COL, how="inner")
    merged = merged.merge(lgb, on=ID_COL, how="inner")

    required_cols = [ID_COL, TARGET_COL, "xgb_pred", "cat_pred", "lgb_pred"]
    missing_cols = [c for c in required_cols if c not in merged.columns]
    if missing_cols:
        raise ValueError(f"OOF 병합 후 필요한 컬럼이 없습니다: {missing_cols}")

    return merged


def merge_submission_dfs(xgb, cat, lgb):
    merged = xgb.merge(cat, on=ID_COL, how="inner")
    merged = merged.merge(lgb, on=ID_COL, how="inner")

    required_cols = [ID_COL, "xgb_pred", "cat_pred", "lgb_pred"]
    missing_cols = [c for c in required_cols if c not in merged.columns]
    if missing_cols:
        raise ValueError(f"submission 병합 후 필요한 컬럼이 없습니다: {missing_cols}")

    return merged


def build_meta_features(df):
    out = df.copy()

    out["pred_mean"] = out[["xgb_pred", "cat_pred", "lgb_pred"]].mean(axis=1)
    out["pred_std"] = out[["xgb_pred", "cat_pred", "lgb_pred"]].std(axis=1)
    out["pred_max"] = out[["xgb_pred", "cat_pred", "lgb_pred"]].max(axis=1)
    out["pred_min"] = out[["xgb_pred", "cat_pred", "lgb_pred"]].min(axis=1)

    out["xgb_cat_gap"] = out["xgb_pred"] - out["cat_pred"]
    out["xgb_lgb_gap"] = out["xgb_pred"] - out["lgb_pred"]
    out["cat_lgb_gap"] = out["cat_pred"] - out["lgb_pred"]

    out["xgb_is_max"] = (
        (out["xgb_pred"] >= out["cat_pred"]) & (out["xgb_pred"] >= out["lgb_pred"])
    ).astype(float)
    out["cat_is_max"] = (
        (out["cat_pred"] >= out["xgb_pred"]) & (out["cat_pred"] >= out["lgb_pred"])
    ).astype(float)
    out["lgb_is_max"] = (
        (out["lgb_pred"] >= out["xgb_pred"]) & (out["lgb_pred"] >= out["cat_pred"])
    ).astype(float)

    return out


def get_meta_columns(df, mode="extended"):
    basic_cols = ["xgb_pred", "cat_pred", "lgb_pred"]

    extended_cols = basic_cols + [
        "pred_mean", "pred_std", "pred_max", "pred_min",
        "xgb_cat_gap", "xgb_lgb_gap", "cat_lgb_gap",
        "xgb_is_max", "cat_is_max", "lgb_is_max"
    ]

    if mode == "basic":
        cols = basic_cols
    elif mode == "extended":
        cols = extended_cols
    else:
        raise ValueError(f"지원하지 않는 META_FEATURE_MODE 입니다: {mode}")

    missing_cols = [c for c in cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"meta feature 컬럼이 없습니다: {missing_cols}")

    return cols


def run_stacking_cv(X_meta, y_meta, X_meta_test, c_value, n_folds=5, seed=42):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    oof_pred = np.zeros(len(X_meta))
    test_pred = np.zeros(len(X_meta_test))
    fold_aucs = []
    fold_models = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_meta, y_meta), start=1):
        X_tr, X_va = X_meta[tr_idx], X_meta[va_idx]
        y_tr, y_va = y_meta[tr_idx], y_meta[va_idx]

        model = LogisticRegression(
            C=c_value,
            max_iter=5000,
            random_state=seed,
            solver="lbfgs"
        )
        model.fit(X_tr, y_tr)

        va_pred = model.predict_proba(X_va)[:, 1]
        te_pred = model.predict_proba(X_meta_test)[:, 1]

        oof_pred[va_idx] = va_pred
        test_pred += te_pred / n_folds

        fold_auc = roc_auc_score(y_va, va_pred)
        fold_aucs.append(fold_auc)
        fold_models.append(model)

    final_auc = roc_auc_score(y_meta, oof_pred)

    return {
        "c_value": c_value,
        "oof_pred": oof_pred,
        "test_pred": test_pred,
        "fold_aucs": fold_aucs,
        "mean_auc": float(np.mean(fold_aucs)),
        "std_auc": float(np.std(fold_aucs)),
        "final_auc": float(final_auc),
        "models": fold_models
    }


# =========================================================
# 2. 데이터 로드
# =========================================================
print_section("2. 데이터 로드")

xgb_oof = load_oof_file(XGB_OOF_PATH, "xgb", keep_target=True)
cat_oof = load_oof_file(CAT_OOF_PATH, "cat", keep_target=False)
lgb_oof = load_oof_file(LGB_OOF_PATH, "lgb", keep_target=False)

xgb_sub = load_submission_file(XGB_SUB_PATH, "xgb")
cat_sub = load_submission_file(CAT_SUB_PATH, "cat")
lgb_sub = load_submission_file(LGB_SUB_PATH, "lgb")

oof_df = merge_oof_dfs(xgb_oof, cat_oof, lgb_oof)
sub_df = merge_submission_dfs(xgb_sub, cat_sub, lgb_sub)

print("OOF merged shape :", oof_df.shape)
print("SUB merged shape :", sub_df.shape)


# =========================================================
# 3. base model / baseline 확인
# =========================================================
print_section("3. base model / baseline 확인")

y = oof_df[TARGET_COL].values

xgb_auc = roc_auc_score(y, oof_df["xgb_pred"].values)
cat_auc = roc_auc_score(y, oof_df["cat_pred"].values)
lgb_auc = roc_auc_score(y, oof_df["lgb_pred"].values)
equal_auc = roc_auc_score(
    y,
    (oof_df["xgb_pred"].values + oof_df["cat_pred"].values + oof_df["lgb_pred"].values) / 3.0
)

print(f"xgb AUC              : {xgb_auc:.6f}")
print(f"cat AUC              : {cat_auc:.6f}")
print(f"lgb AUC              : {lgb_auc:.6f}")
print(f"equal ensemble AUC   : {equal_auc:.6f}")
print(f"{BASELINE_NAME} AUC : {BASELINE_AUC:.6f}")
print(f"baseline weights     : {BASELINE_WEIGHTS}")


# =========================================================
# 4. stacking meta feature 생성
# =========================================================
print_section("4. stacking meta feature 생성")

meta_oof_df = build_meta_features(oof_df)
meta_sub_df = build_meta_features(sub_df)

meta_cols = get_meta_columns(meta_oof_df, mode=META_FEATURE_MODE)
print(f"META_FEATURE_MODE : {META_FEATURE_MODE}")
print(f"meta feature 수     : {len(meta_cols)}")
print(meta_cols)

meta_feature_export_cols = [ID_COL]
if TARGET_COL in meta_oof_df.columns:
    meta_feature_export_cols.append(TARGET_COL)
meta_feature_export_cols += meta_cols

meta_feature_export = meta_oof_df[meta_feature_export_cols].copy()
meta_feature_export.to_csv(META_FEATURE_PATH, index=False, encoding="utf-8-sig")
print(f"[저장] {META_FEATURE_PATH}")


# =========================================================
# 5. C 여러 개 자동 비교
# =========================================================
print_section("5. C 여러 개 자동 비교")

X_meta = meta_oof_df[meta_cols].values
y_meta = meta_oof_df[TARGET_COL].values
X_meta_test = meta_sub_df[meta_cols].values

search_rows = []
all_results = []

best_result = None
best_auc = -1.0

for c_value in C_LIST:
    print(f"\n{'─' * 60}")
    print(f"C = {c_value}")
    print(f"{'─' * 60}")

    result = run_stacking_cv(
        X_meta=X_meta,
        y_meta=y_meta,
        X_meta_test=X_meta_test,
        c_value=c_value,
        n_folds=N_FOLDS,
        seed=SEED
    )

    gain_vs_baseline = result["final_auc"] - BASELINE_AUC
    gain_vs_equal = result["final_auc"] - equal_auc
    gain_vs_best_single = result["final_auc"] - max(xgb_auc, cat_auc, lgb_auc)

    row = {
        "c_value": c_value,
        "meta_feature_mode": META_FEATURE_MODE,
        "n_meta_features": len(meta_cols),
        "stacking_auc": round(float(result["final_auc"]), 6),
        "gain_vs_baseline": round(float(gain_vs_baseline), 6),
        "gain_vs_equal": round(float(gain_vs_equal), 6),
        "gain_vs_best_single": round(float(gain_vs_best_single), 6),
        "fold1_auc": round(float(result["fold_aucs"][0]), 6),
        "fold2_auc": round(float(result["fold_aucs"][1]), 6),
        "fold3_auc": round(float(result["fold_aucs"][2]), 6),
        "fold4_auc": round(float(result["fold_aucs"][3]), 6),
        "fold5_auc": round(float(result["fold_aucs"][4]), 6),
        "mean_auc": round(float(result["mean_auc"]), 6),
        "std_auc": round(float(result["std_auc"]), 6),
    }
    search_rows.append(row)
    all_results.append(result)

    print(f"fold aucs         : {[round(v, 6) for v in result['fold_aucs']]}")
    print(f"stacking oof auc  : {result['final_auc']:.6f}")
    print(f"gain vs baseline  : {gain_vs_baseline:+.6f}")
    print(f"gain vs equal     : {gain_vs_equal:+.6f}")

    if result["final_auc"] > best_auc:
        best_auc = result["final_auc"]
        best_result = result

c_search_df = pd.DataFrame(search_rows).sort_values(
    ["stacking_auc", "mean_auc"], ascending=False
).reset_index(drop=True)
c_search_df.to_csv(C_SEARCH_RESULT_PATH, index=False, encoding="utf-8-sig")

print("\n[C 비교 결과]")
print(c_search_df.to_string(index=False))
print(f"\n[저장] {C_SEARCH_RESULT_PATH}")


# =========================================================
# 6. best C 결과 선택
# =========================================================
print_section("6. best C 결과 선택")

best_c = best_result["c_value"]
best_stack_auc = best_result["final_auc"]
best_mean_auc = best_result["mean_auc"]
best_std_auc = best_result["std_auc"]
best_fold_aucs = best_result["fold_aucs"]
best_stack_oof_pred = best_result["oof_pred"]
best_stack_test_pred = best_result["test_pred"]

gain_vs_baseline = best_stack_auc - BASELINE_AUC
gain_vs_equal = best_stack_auc - equal_auc
best_single_auc = max(xgb_auc, cat_auc, lgb_auc)
gain_vs_best_single = best_stack_auc - best_single_auc

print(f"best C                  : {best_c}")
print(f"best stacking OOF AUC   : {best_stack_auc:.6f}")
print(f"gain vs baseline        : {gain_vs_baseline:+.6f}")
print(f"gain vs equal ensemble  : {gain_vs_equal:+.6f}")
print(f"gain vs best single     : {gain_vs_best_single:+.6f}")

if best_stack_auc > BASELINE_AUC:
    print("✓ baseline 초과")
else:
    print("✗ baseline 미달")

if best_std_auc > 0.01:
    print("⚠ fold 간 편차가 큽니다.")
else:
    print("✓ fold 간 편차가 안정적입니다.")


# =========================================================
# 7. best stacking OOF 저장
# =========================================================
print_section("7. best stacking OOF 저장")

STACKING_OOF_PATH = os.path.join(
    OUTPUT_DIR, f"ensemble_v3_stacking_lr_bestC_{str(best_c).replace('.', 'p')}_oof.csv"
)

stacking_oof_df = pd.DataFrame({
    ID_COL: meta_oof_df[ID_COL].values,
    TARGET_COL: y_meta,
    "stacking_oof_pred": best_stack_oof_pred
})
stacking_oof_df.to_csv(STACKING_OOF_PATH, index=False, encoding="utf-8-sig")
print(f"[저장] {STACKING_OOF_PATH}")


# =========================================================
# 8. best stacking submission 저장
# =========================================================
print_section("8. best stacking submission 저장")

STACKING_SUB_PATH = os.path.join(
    OUTPUT_DIR, f"ensemble_v3_stacking_lr_bestC_{str(best_c).replace('.', 'p')}_submission.csv"
)

stacking_submission = pd.DataFrame({
    ID_COL: meta_sub_df[ID_COL].values,
    TARGET_COL: best_stack_test_pred
})
stacking_submission.to_csv(STACKING_SUB_PATH, index=False, encoding="utf-8-sig")
print(f"[저장] {STACKING_SUB_PATH}")
print(stacking_submission.head(10))


# =========================================================
# 9. 결과 로그 저장
# =========================================================
print_section("9. 결과 로그 저장")

log_row = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "experiment_name": EXPERIMENT_NAME,
    "meta_model": "LogisticRegression",
    "meta_feature_mode": META_FEATURE_MODE,
    "best_c": best_c,
    "c_candidates": ", ".join([str(c) for c in C_LIST]),
    "n_folds": N_FOLDS,
    "n_meta_features": len(meta_cols),
    "xgb_auc": round(float(xgb_auc), 6),
    "cat_auc": round(float(cat_auc), 6),
    "lgb_auc": round(float(lgb_auc), 6),
    "equal_auc": round(float(equal_auc), 6),
    "baseline_name": BASELINE_NAME,
    "baseline_auc": round(float(BASELINE_AUC), 6),
    "baseline_weights": BASELINE_WEIGHTS,
    "stacking_auc": round(float(best_stack_auc), 6),
    "gain_vs_baseline": round(float(gain_vs_baseline), 6),
    "gain_vs_equal": round(float(gain_vs_equal), 6),
    "gain_vs_best_single": round(float(gain_vs_best_single), 6),
    "fold1_auc": round(float(best_fold_aucs[0]), 6),
    "fold2_auc": round(float(best_fold_aucs[1]), 6),
    "fold3_auc": round(float(best_fold_aucs[2]), 6),
    "fold4_auc": round(float(best_fold_aucs[3]), 6),
    "fold5_auc": round(float(best_fold_aucs[4]), 6),
    "mean_auc": round(float(best_mean_auc), 6),
    "std_auc": round(float(best_std_auc), 6),
}

result_log_df = pd.DataFrame([log_row])

if os.path.exists(RESULT_LOG_PATH):
    old = pd.read_csv(RESULT_LOG_PATH)
    result_log_df = pd.concat([old, result_log_df], ignore_index=True)

result_log_df.to_csv(RESULT_LOG_PATH, index=False, encoding="utf-8-sig")
print(f"[저장] {RESULT_LOG_PATH}")
print(pd.DataFrame([log_row]).to_string(index=False))


# =========================================================
# 10. 완료 요약
# =========================================================
print_section("10. 완료")

print(f"실험명                   : {EXPERIMENT_NAME}")
print(f"meta feature mode        : {META_FEATURE_MODE}")
print(f"best C                   : {best_c}")
print(f"baseline AUC             : {BASELINE_AUC:.6f}")
print(f"best stacking OOF AUC    : {best_stack_auc:.6f}")
print(f"gain vs baseline         : {gain_vs_baseline:+.6f}")
print(f"best single AUC          : {best_single_auc:.6f}")
print(f"equal ensemble AUC       : {equal_auc:.6f}")
print(f"mean AUC ± std           : {best_mean_auc:.6f} ± {best_std_auc:.6f}")

print("\n저장 파일:")
print(f"  1. {META_FEATURE_PATH}")
print(f"  2. {C_SEARCH_RESULT_PATH}")
print(f"  3. {STACKING_OOF_PATH}")
print(f"  4. {STACKING_SUB_PATH}")
print(f"  5. {RESULT_LOG_PATH}")