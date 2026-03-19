# =========================================================
# ensemble_v5_stacking_meta_model.py
# ---------------------------------------------------------
# 목적
# 1) XGBoost / CatBoost / LightGBM의 OOF 예측값을 불러온다.
# 2) 단일 모델 성능과 모델 간 예측 상관계수를 비교한다.
# 3) Meta Feature를 생성한다.
# 4) Logistic Regression 기반 Stacking을 수행한다.
# 5) 여러 C 값에 대해 OOF AUC를 비교한다.
# 6) Best Stacking OOF / submission / results log를 저장한다.
# 7) 제출 파일은 sample_submission.csv 형식(ID, probability)에 맞춰 저장한다.
#
# 현재 기준 모델 조합
# - XGB : xgb_optuna_v1
# - CAT : catboost_optuna_fixed_v6_combo
# - LGB : lightgbm_ivf_signal_v2
# =========================================================

import os
import warnings
warnings.filterwarnings("ignore")

from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


# =========================================================
# 0. 경로 / 기본 설정
# =========================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_COL = "임신 성공 여부"   # OOF 평가용
ID_COL = "ID"
OOF_PRED_COL = "oof_pred_prob"

# 대회 제출 형식
SUBMIT_ID_COL = "ID"
SUBMIT_TARGET_COL = "probability"

SEED = 42
N_FOLDS = 5

# ---------------------------------------------------------
# 입력 파일 경로
# ---------------------------------------------------------
XGB_OOF_PATH = os.path.join(OUTPUT_DIR, "xgb_optuna_v1_oof_predictions.csv")
XGB_SUB_PATH = os.path.join(OUTPUT_DIR, "xgb_optuna_v1_submission.csv")

CAT_OOF_PATH = os.path.join(OUTPUT_DIR, "catboost_optuna_fixed_v6_combo_oof_predictions.csv")
CAT_SUB_PATH = os.path.join(OUTPUT_DIR, "catboost_optuna_fixed_v6_combo_submission.csv")

LGB_OOF_PATH = os.path.join(OUTPUT_DIR, "lightgbm_ivf_signal_v2_oof_predictions.csv")
LGB_SUB_PATH = os.path.join(OUTPUT_DIR, "lightgbm_ivf_signal_v2_submission.csv")

# ---------------------------------------------------------
# Meta model 설정
# ---------------------------------------------------------
CANDIDATE_C_VALUES = [1.0, 0.3, 0.1, 0.03, 0.01]
META_MAX_ITER = 2000

EXPERIMENT_NAME = "ensemble_v5_stacking_meta_model"
RESULT_LOG_PATH = os.path.join(OUTPUT_DIR, "ensemble_v5_stacking_results_log.csv")
SEARCH_RESULT_PATH = os.path.join(OUTPUT_DIR, "ensemble_v5_stacking_search_results.csv")


# =========================================================
# 1. 보조 함수
# =========================================================
def print_section(title: str):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def check_file_exists(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"파일이 없습니다: {path}")


def load_and_sort_csv(path: str, id_col: str):
    df = pd.read_csv(path)
    df = df.sort_values(id_col).reset_index(drop=True)
    return df


def save_log_row(log_path: str, row: dict):
    row_df = pd.DataFrame([row])
    if os.path.exists(log_path):
        old = pd.read_csv(log_path)
        out = pd.concat([old, row_df], ignore_index=True)
    else:
        out = row_df
    out.to_csv(log_path, index=False, encoding="utf-8-sig")


def evaluate_auc(y_true, pred, name=None):
    auc = roc_auc_score(y_true, pred)
    if name is not None:
        print(f"{name:<28}: {auc:.6f}")
    return auc


def load_oof_predictions(path, model_name):
    df = load_and_sort_csv(path, ID_COL)
    required_cols = [ID_COL, TARGET_COL, OOF_PRED_COL]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"[{model_name}] OOF 파일에 필요한 컬럼이 없습니다: {missing_cols}")

    out = df[[ID_COL, TARGET_COL, OOF_PRED_COL]].copy()
    out = out.rename(columns={OOF_PRED_COL: f"{model_name}_pred"})
    return out


def load_submission_predictions(path, model_name):
    df = load_and_sort_csv(path, SUBMIT_ID_COL)
    if SUBMIT_ID_COL not in df.columns:
        raise ValueError(f"[{model_name}] submission 파일에 ID 컬럼이 없습니다.")

    pred_cols = [c for c in df.columns if c != SUBMIT_ID_COL]
    if len(pred_cols) == 0:
        raise ValueError(f"[{model_name}] submission 파일에 예측값 컬럼이 없습니다.")

    pred_col = pred_cols[0]
    out = df[[SUBMIT_ID_COL, pred_col]].copy()
    out = out.rename(columns={pred_col: f"{model_name}_pred"})
    return out


def make_meta_features(df: pd.DataFrame):
    meta = pd.DataFrame(index=df.index)

    xgb = df["xgb_pred"].values
    cat = df["cat_pred"].values
    lgb = df["lgb_pred"].values

    meta["xgb_pred"] = xgb
    meta["cat_pred"] = cat
    meta["lgb_pred"] = lgb

    meta["pred_mean"] = (xgb + cat + lgb) / 3.0
    meta["pred_std"] = np.std(np.vstack([xgb, cat, lgb]), axis=0)
    meta["pred_max"] = np.maximum(np.maximum(xgb, cat), lgb)
    meta["pred_min"] = np.minimum(np.minimum(xgb, cat), lgb)

    meta["xgb_cat_gap"] = np.abs(xgb - cat)
    meta["xgb_lgb_gap"] = np.abs(xgb - lgb)
    meta["cat_lgb_gap"] = np.abs(cat - lgb)

    meta["xgb_is_max"] = ((xgb >= cat) & (xgb >= lgb)).astype(int)
    meta["cat_is_max"] = ((cat >= xgb) & (cat >= lgb)).astype(int)
    meta["lgb_is_max"] = ((lgb >= xgb) & (lgb >= cat)).astype(int)

    return meta


# =========================================================
# 2. 입력 파일 확인
# =========================================================
print_section("2. 입력 파일 확인")

for path in [
    XGB_OOF_PATH, XGB_SUB_PATH,
    CAT_OOF_PATH, CAT_SUB_PATH,
    LGB_OOF_PATH, LGB_SUB_PATH
]:
    check_file_exists(path)
    print(f"[OK] {path}")


# =========================================================
# 3. OOF 예측값 로드
# =========================================================
print_section("3. OOF 예측값 로드")

xgb_oof = load_oof_predictions(XGB_OOF_PATH, "xgb")
cat_oof = load_oof_predictions(CAT_OOF_PATH, "cat")
lgb_oof = load_oof_predictions(LGB_OOF_PATH, "lgb")

assert (xgb_oof[ID_COL].values == cat_oof[ID_COL].values).all(), "XGB와 CAT OOF의 ID가 다릅니다."
assert (xgb_oof[ID_COL].values == lgb_oof[ID_COL].values).all(), "XGB와 LGB OOF의 ID가 다릅니다."

assert (xgb_oof[TARGET_COL].values == cat_oof[TARGET_COL].values).all(), "XGB와 CAT OOF의 target이 다릅니다."
assert (xgb_oof[TARGET_COL].values == lgb_oof[TARGET_COL].values).all(), "XGB와 LGB OOF의 target이 다릅니다."

oof_df = xgb_oof.merge(cat_oof[[ID_COL, "cat_pred"]], on=ID_COL, how="inner")
oof_df = oof_df.merge(lgb_oof[[ID_COL, "lgb_pred"]], on=ID_COL, how="inner")

y_true = oof_df[TARGET_COL].values

xgb_pred = oof_df["xgb_pred"].values
cat_pred = oof_df["cat_pred"].values
lgb_pred = oof_df["lgb_pred"].values

print(f"OOF merged shape : {oof_df.shape}")


# =========================================================
# 4. 단일 모델 성능 / 기본 비교
# =========================================================
print_section("4. 단일 모델 성능 / 기본 비교")

xgb_auc = evaluate_auc(y_true, xgb_pred, "XGBoost")
cat_auc = evaluate_auc(y_true, cat_pred, "CatBoost")
lgb_auc = evaluate_auc(y_true, lgb_pred, "LightGBM")

best_single_auc = max(xgb_auc, cat_auc, lgb_auc)

prob_equal_pred = (xgb_pred + cat_pred + lgb_pred) / 3.0
prob_equal_auc = evaluate_auc(y_true, prob_equal_pred, "Prob Equal Ensemble")

# 현재 최고 rank ensemble 기준값
RANK_BASELINE_AUC = 0.740634
print(f"{'Rank Baseline Ensemble':<28}: {RANK_BASELINE_AUC:.6f}")


# =========================================================
# 5. 예측 상관계수 비교
# =========================================================
print_section("5. 예측 상관계수 비교")

corr_xgb_cat = np.corrcoef(xgb_pred, cat_pred)[0, 1]
corr_xgb_lgb = np.corrcoef(xgb_pred, lgb_pred)[0, 1]
corr_cat_lgb = np.corrcoef(cat_pred, lgb_pred)[0, 1]

print(f"XGB-CAT : {corr_xgb_cat:.6f}")
print(f"XGB-LGB : {corr_xgb_lgb:.6f}")
print(f"CAT-LGB : {corr_cat_lgb:.6f}")


# =========================================================
# 6. submission 예측값 로드
# =========================================================
print_section("6. submission 예측값 로드")

xgb_sub = load_submission_predictions(XGB_SUB_PATH, "xgb")
cat_sub = load_submission_predictions(CAT_SUB_PATH, "cat")
lgb_sub = load_submission_predictions(LGB_SUB_PATH, "lgb")

assert (xgb_sub[SUBMIT_ID_COL].values == cat_sub[SUBMIT_ID_COL].values).all(), "XGB와 CAT submission ID가 다릅니다."
assert (xgb_sub[SUBMIT_ID_COL].values == lgb_sub[SUBMIT_ID_COL].values).all(), "XGB와 LGB submission ID가 다릅니다."

sub_df = xgb_sub.merge(cat_sub, on=SUBMIT_ID_COL, how="inner")
sub_df = sub_df.merge(lgb_sub, on=SUBMIT_ID_COL, how="inner")

print(f"submission merged shape : {sub_df.shape}")


# =========================================================
# 7. Meta Feature 생성
# =========================================================
print_section("7. Meta Feature 생성")

X_meta = make_meta_features(oof_df)
X_test_meta = make_meta_features(sub_df)

print(f"Meta train shape : {X_meta.shape}")
print(f"Meta test shape  : {X_test_meta.shape}")
print("Meta columns:")
print(list(X_meta.columns))


# =========================================================
# 8. Stacking CV 탐색
# =========================================================
print_section("8. Stacking CV 탐색")

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

search_rows = []

best_stack_auc = -1.0
best_c = None
best_oof_pred = None
best_test_pred = None
best_fold_aucs = None

for c_value in CANDIDATE_C_VALUES:
    print(f"\n[C = {c_value}]")

    stack_oof = np.zeros(len(X_meta), dtype=float)
    stack_test = np.zeros(len(X_test_meta), dtype=float)
    fold_aucs = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_meta, y_true), start=1):
        X_tr = X_meta.iloc[tr_idx].copy()
        X_va = X_meta.iloc[va_idx].copy()
        y_tr = y_true[tr_idx]
        y_va = y_true[va_idx]

        meta_model = LogisticRegression(
            C=c_value,
            max_iter=META_MAX_ITER,
            random_state=SEED,
            solver="lbfgs"
        )

        meta_model.fit(X_tr, y_tr)

        va_pred = meta_model.predict_proba(X_va)[:, 1]
        te_pred = meta_model.predict_proba(X_test_meta)[:, 1]

        stack_oof[va_idx] = va_pred
        stack_test += te_pred / N_FOLDS

        fold_auc = roc_auc_score(y_va, va_pred)
        fold_aucs.append(fold_auc)

    stack_auc = roc_auc_score(y_true, stack_oof)
    mean_auc = np.mean(fold_aucs)
    std_auc = np.std(fold_aucs)

    print(f"OOF AUC      : {stack_auc:.6f}")
    print(f"Fold Mean AUC: {mean_auc:.6f}")
    print(f"Fold Std AUC : {std_auc:.6f}")

    search_rows.append({
        "C": c_value,
        "oof_auc": round(float(stack_auc), 6),
        "mean_auc": round(float(mean_auc), 6),
        "std_auc": round(float(std_auc), 6),
        "fold1_auc": round(float(fold_aucs[0]), 6),
        "fold2_auc": round(float(fold_aucs[1]), 6),
        "fold3_auc": round(float(fold_aucs[2]), 6),
        "fold4_auc": round(float(fold_aucs[3]), 6),
        "fold5_auc": round(float(fold_aucs[4]), 6),
    })

    if stack_auc > best_stack_auc:
        best_stack_auc = stack_auc
        best_c = c_value
        best_oof_pred = stack_oof.copy()
        best_test_pred = stack_test.copy()
        best_fold_aucs = fold_aucs.copy()

search_df = pd.DataFrame(search_rows).sort_values("oof_auc", ascending=False).reset_index(drop=True)
search_df.to_csv(SEARCH_RESULT_PATH, index=False, encoding="utf-8-sig")

print("\n[Stacking Search Result]")
print(search_df.to_string(index=False))

print(f"\nBest C         : {best_c}")
print(f"Best OOF AUC   : {best_stack_auc:.6f}")
print(f"[저장] {SEARCH_RESULT_PATH}")


# =========================================================
# 9. 결과 요약
# =========================================================
print_section("9. 결과 요약")

gain_vs_best_single = best_stack_auc - best_single_auc
gain_vs_prob_equal = best_stack_auc - prob_equal_auc
gain_vs_rank_baseline = best_stack_auc - RANK_BASELINE_AUC

print(f"Best Single AUC         : {best_single_auc:.6f}")
print(f"Prob Equal Ensemble AUC : {prob_equal_auc:.6f}")
print(f"Rank Baseline AUC       : {RANK_BASELINE_AUC:.6f}")
print(f"Best Stacking AUC       : {best_stack_auc:.6f}")

print(f"\nGain vs Best Single     : {gain_vs_best_single:+.6f}")
print(f"Gain vs Prob Equal      : {gain_vs_prob_equal:+.6f}")
print(f"Gain vs Rank Baseline   : {gain_vs_rank_baseline:+.6f}")

if best_stack_auc > RANK_BASELINE_AUC:
    print("✓ 현재 Rank Baseline 초과")
else:
    print("✗ 현재 Rank Baseline 미달")


# =========================================================
# 10. OOF 저장
# =========================================================
print_section("10. Stacking OOF 저장")

stack_name = f"ensemble_v5_stack_lr_c{str(best_c).replace('.', '_')}"

stack_oof_df = pd.DataFrame({
    ID_COL: oof_df[ID_COL].values,
    TARGET_COL: y_true,
    "xgb_pred_prob": xgb_pred,
    "cat_pred_prob": cat_pred,
    "lgb_pred_prob": lgb_pred,
    "stack_oof_pred_prob": best_oof_pred,
    "stack_oof_pred_label": (best_oof_pred >= 0.5).astype(int),
})
stack_oof_df["correct"] = (stack_oof_df[TARGET_COL] == stack_oof_df["stack_oof_pred_label"]).astype(int)

stack_oof_path = os.path.join(OUTPUT_DIR, f"{stack_name}_oof.csv")
stack_oof_df.to_csv(stack_oof_path, index=False, encoding="utf-8-sig")
print(f"[저장] {stack_oof_path}")


# =========================================================
# 11. submission 저장
# =========================================================
print_section("11. Stacking submission 저장")

stack_submission = pd.DataFrame({
    SUBMIT_ID_COL: sub_df[SUBMIT_ID_COL].values,
    SUBMIT_TARGET_COL: best_test_pred,
})

stack_sub_path = os.path.join(OUTPUT_DIR, f"{stack_name}_submission.csv")
stack_submission.to_csv(stack_sub_path, index=False, encoding="utf-8-sig")
print(f"[저장] {stack_sub_path}")
print(stack_submission.head(5))


# =========================================================
# 12. 로그 저장
# =========================================================
print_section("12. 결과 로그 저장")

log_row = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "experiment_name": EXPERIMENT_NAME,

    "xgb_auc": round(float(xgb_auc), 6),
    "cat_auc": round(float(cat_auc), 6),
    "lgb_auc": round(float(lgb_auc), 6),
    "best_single_auc": round(float(best_single_auc), 6),

    "prob_equal_auc": round(float(prob_equal_auc), 6),
    "rank_baseline_auc": round(float(RANK_BASELINE_AUC), 6),

    "corr_xgb_cat": round(float(corr_xgb_cat), 6),
    "corr_xgb_lgb": round(float(corr_xgb_lgb), 6),
    "corr_cat_lgb": round(float(corr_cat_lgb), 6),

    "best_c": best_c,
    "best_stack_auc": round(float(best_stack_auc), 6),
    "gain_vs_best_single": round(float(gain_vs_best_single), 6),
    "gain_vs_prob_equal": round(float(gain_vs_prob_equal), 6),
    "gain_vs_rank_baseline": round(float(gain_vs_rank_baseline), 6),

    "fold1_auc": round(float(best_fold_aucs[0]), 6),
    "fold2_auc": round(float(best_fold_aucs[1]), 6),
    "fold3_auc": round(float(best_fold_aucs[2]), 6),
    "fold4_auc": round(float(best_fold_aucs[3]), 6),
    "fold5_auc": round(float(best_fold_aucs[4]), 6),
    "mean_auc": round(float(np.mean(best_fold_aucs)), 6),
    "std_auc": round(float(np.std(best_fold_aucs)), 6),

    "xgb_oof_file": os.path.basename(XGB_OOF_PATH),
    "cat_oof_file": os.path.basename(CAT_OOF_PATH),
    "lgb_oof_file": os.path.basename(LGB_OOF_PATH),
    "xgb_sub_file": os.path.basename(XGB_SUB_PATH),
    "cat_sub_file": os.path.basename(CAT_SUB_PATH),
    "lgb_sub_file": os.path.basename(LGB_SUB_PATH),
}

save_log_row(RESULT_LOG_PATH, log_row)
print(f"[저장] {RESULT_LOG_PATH}")

log_preview = pd.read_csv(RESULT_LOG_PATH)
show_cols = [
    "timestamp",
    "best_c",
    "best_stack_auc",
    "rank_baseline_auc",
    "gain_vs_rank_baseline",
    "gain_vs_best_single",
    "mean_auc",
    "std_auc",
]
print(log_preview[show_cols].tail(10).to_string(index=False))


# =========================================================
# 13. 종료 요약
# =========================================================
print_section("13. 완료")

print("단일 모델")
print(f"  XGB : {xgb_auc:.6f}")
print(f"  CAT : {cat_auc:.6f}")
print(f"  LGB : {lgb_auc:.6f}")

print("\n기준 비교")
print(f"  Best Single     : {best_single_auc:.6f}")
print(f"  Prob Equal      : {prob_equal_auc:.6f}")
print(f"  Rank Baseline   : {RANK_BASELINE_AUC:.6f}")
print(f"  Best Stacking   : {best_stack_auc:.6f}")

print("\nBest Meta Setting")
print(f"  LogisticRegression C : {best_c}")

print("\nGain")
print(f"  vs Best Single   : {gain_vs_best_single:+.6f}")
print(f"  vs Prob Equal    : {gain_vs_prob_equal:+.6f}")
print(f"  vs Rank Baseline : {gain_vs_rank_baseline:+.6f}")

print("\n예측 상관계수")
print(f"  XGB-CAT : {corr_xgb_cat:.6f}")
print(f"  XGB-LGB : {corr_xgb_lgb:.6f}")
print(f"  CAT-LGB : {corr_cat_lgb:.6f}")

print("\n저장 파일")
print(f"  1. {SEARCH_RESULT_PATH}")
print(f"  2. {stack_oof_path}")
print(f"  3. {stack_sub_path}")
print(f"  4. {RESULT_LOG_PATH}")