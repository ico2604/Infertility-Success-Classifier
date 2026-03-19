# =========================================================
# ensemble_v7_hybrid_rank_prob_xgb_lgb_cat2.py
# ---------------------------------------------------------
# 목적
# 1) XGB / LGB / CAT2(v17) 예측을 불러온다.
# 2) weighted rank ensemble을 탐색한다.
# 3) weighted probability ensemble도 탐색한다.
# 4) best rank + prob 를 alpha로 섞는 hybrid ensemble을 탐색한다.
# 5) best hybrid OOF / submission / results log를 저장한다.
# =========================================================

import os
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


# =========================================================
# 0. 경로 설정
# =========================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")

# ---------------------------------------------------------
# XGB / LGB
# ---------------------------------------------------------
XGB_OOF_PATH = os.path.join(OUTPUT_DIR, "xgb_v2_reg_relax_oof_predictions.csv")
LGB_OOF_PATH = os.path.join(OUTPUT_DIR, "lightgbm_v1_lightgbm_baseline_v1_oof_predictions.csv")

XGB_SUB_PATH = os.path.join(OUTPUT_DIR, "xgb_v2_reg_relax_submission.csv")
LGB_SUB_PATH = os.path.join(OUTPUT_DIR, "lightgbm_v1_lightgbm_baseline_v1_submission.csv")

# ---------------------------------------------------------
# CAT2(v17)
# ---------------------------------------------------------
CAT2_OOF_NPY_PATH = os.path.join(OUTPUT_DIR, "oof_v17_final.npy")
CAT2_SUB_NPY_PATH = os.path.join(OUTPUT_DIR, "test_v17_final.npy")
CAT2_SUB_CSV_PATH = os.path.join(OUTPUT_DIR, "sample_submission_v17_20260317_171824.csv")

# 중요도 파일
XGB_IMPORTANCE_PATH = os.path.join(OUTPUT_DIR, "xgb_v2_reg_relax_feature_importance.csv")
LGB_IMPORTANCE_PATH = os.path.join(OUTPUT_DIR, "lightgbm_v1_lightgbm_baseline_v1_feature_importance.csv")
CAT2_IMPORTANCE_PATH = os.path.join(OUTPUT_DIR, "feature_importance_v17.csv")

RESULT_LOG_PATH = os.path.join(OUTPUT_DIR, "ensemble_v7_hybrid_results_log.csv")
RANK_SEARCH_RESULT_PATH = os.path.join(OUTPUT_DIR, "ensemble_v7_rank_search_results.csv")
PROB_SEARCH_RESULT_PATH = os.path.join(OUTPUT_DIR, "ensemble_v7_prob_search_results.csv")
HYBRID_SEARCH_RESULT_PATH = os.path.join(OUTPUT_DIR, "ensemble_v7_hybrid_search_results.csv")

TARGET_COL = "임신 성공 여부"
ID_COL = "ID"
OOF_PRED_COL = "oof_pred_prob"

SUBMIT_ID_COL = "ID"
SUBMIT_TARGET_COL = "probability"

BASELINE_AUC = 0.740444
BASELINE_NAME = "prob_weighted_ensemble_3models_old"
BASELINE_WEIGHTS = (0.33, 0.27, 0.40)  # xgb, lgb, cat2

SEARCH_STEP = 0.02
ALPHA_STEP = 0.05
ROUND_DIGITS = 4

EXPERIMENT_NAME = "ensemble_v7_hybrid_rank_prob_xgb_lgb_cat2"


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

    if ID_COL not in df.columns:
        raise ValueError(f"[{model_name}] submission 파일에 ID 컬럼이 없습니다.")

    if df.shape[1] < 2:
        raise ValueError(f"[{model_name}] submission 파일에 예측값 컬럼이 없습니다.")

    pred_col = [c for c in df.columns if c != ID_COL][0]
    out = df[[ID_COL, pred_col]].copy()
    out = out.rename(columns={pred_col: f"{model_name}_pred"})
    return out


def load_sample_submission(path):
    df = pd.read_csv(path)
    required_cols = [SUBMIT_ID_COL, SUBMIT_TARGET_COL]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"sample_submission 컬럼이 올바르지 않습니다: {missing_cols}")
    return df


def load_cat2_oof_from_npy(oof_npy_path, train_path, model_name="cat2"):
    train_df = pd.read_csv(train_path, usecols=[ID_COL, TARGET_COL])
    pred = np.load(oof_npy_path)

    if len(train_df) != len(pred):
        raise ValueError(
            f"[{model_name}] train 행 수({len(train_df)})와 OOF npy 길이({len(pred)})가 다릅니다."
        )

    out = train_df.copy()
    out[f"{model_name}_pred"] = pred
    return out


def load_cat2_submission_from_csv(sub_csv_path, model_name="cat2"):
    df = pd.read_csv(sub_csv_path)

    required_cols = [ID_COL, SUBMIT_TARGET_COL]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"[{model_name}] csv 제출 파일에 필요한 컬럼이 없습니다: {missing_cols}")

    out = df[[ID_COL, SUBMIT_TARGET_COL]].copy()
    out = out.rename(columns={SUBMIT_TARGET_COL: f"{model_name}_pred"})
    return out


def load_cat2_submission_from_npy(sub_npy_path, test_path, model_name="cat2"):
    test_df = pd.read_csv(test_path, usecols=[ID_COL])
    pred = np.load(sub_npy_path)

    if len(test_df) != len(pred):
        raise ValueError(
            f"[{model_name}] test 행 수({len(test_df)})와 submission npy 길이({len(pred)})가 다릅니다."
        )

    out = test_df.copy()
    out[f"{model_name}_pred"] = pred
    return out


def merge_oof_dfs(xgb, lgb, cat2):
    merged = xgb.merge(lgb, on=ID_COL, how="inner")
    merged = merged.merge(cat2, on=[ID_COL, TARGET_COL], how="inner")

    required_cols = [
        ID_COL, TARGET_COL,
        "xgb_pred", "lgb_pred", "cat2_pred"
    ]
    missing_cols = [c for c in required_cols if c not in merged.columns]
    if missing_cols:
        raise ValueError(f"OOF 병합 후 필요한 컬럼이 없습니다: {missing_cols}")

    return merged


def merge_submission_dfs(xgb, lgb, cat2):
    merged = xgb.merge(lgb, on=ID_COL, how="inner")
    merged = merged.merge(cat2, on=ID_COL, how="inner")

    required_cols = [ID_COL, "xgb_pred", "lgb_pred", "cat2_pred"]
    missing_cols = [c for c in required_cols if c not in merged.columns]
    if missing_cols:
        raise ValueError(f"submission 병합 후 필요한 컬럼이 없습니다: {missing_cols}")

    return merged


def to_rank_percentile(arr):
    s = pd.Series(arr)
    return s.rank(method="average", pct=True).values


def generate_weight_candidates_3(step=0.02):
    weights = []
    grid = np.arange(0, 1 + step, step)

    for wx in grid:
        for wl in grid:
            wc = 1.0 - wx - wl
            if wc < 0 or wc > 1:
                continue

            wx_r = round(float(wx), ROUND_DIGITS)
            wl_r = round(float(wl), ROUND_DIGITS)
            wc_r = round(float(wc), ROUND_DIGITS)

            if abs((wx_r + wl_r + wc_r) - 1.0) > 1e-8:
                continue

            weights.append((wx_r, wl_r, wc_r))

    return list(dict.fromkeys(weights))


def generate_alpha_candidates(step=0.05):
    vals = np.arange(0, 1 + step, step)
    vals = [round(float(x), 4) for x in vals]
    return vals


# =========================================================
# 2. 데이터 로드
# =========================================================
print_section("2. 데이터 로드")

sample_sub = load_sample_submission(SAMPLE_SUB_PATH)

xgb_oof = load_oof_file(XGB_OOF_PATH, "xgb", keep_target=True)
lgb_oof = load_oof_file(LGB_OOF_PATH, "lgb", keep_target=False)
cat2_oof = load_cat2_oof_from_npy(CAT2_OOF_NPY_PATH, TRAIN_PATH, model_name="cat2")

xgb_sub = load_submission_file(XGB_SUB_PATH, "xgb")
lgb_sub = load_submission_file(LGB_SUB_PATH, "lgb")

if os.path.exists(CAT2_SUB_CSV_PATH):
    cat2_sub = load_cat2_submission_from_csv(CAT2_SUB_CSV_PATH, model_name="cat2")
else:
    cat2_sub = load_cat2_submission_from_npy(CAT2_SUB_NPY_PATH, TEST_PATH, model_name="cat2")

oof_df = merge_oof_dfs(xgb_oof, lgb_oof, cat2_oof)
sub_df = merge_submission_dfs(xgb_sub, lgb_sub, cat2_sub)

print("OOF merged shape :", oof_df.shape)
print("SUB merged shape :", sub_df.shape)
print("Sample sub shape :", sample_sub.shape)


# =========================================================
# 3. 기본 성능 확인
# =========================================================
print_section("3. 기본 성능 확인")

y = oof_df[TARGET_COL].values

xgb_pred = oof_df["xgb_pred"].values
lgb_pred = oof_df["lgb_pred"].values
cat2_pred = oof_df["cat2_pred"].values

xgb_sub_pred = sub_df["xgb_pred"].values
lgb_sub_pred = sub_df["lgb_pred"].values
cat2_sub_pred = sub_df["cat2_pred"].values

xgb_auc = roc_auc_score(y, xgb_pred)
lgb_auc = roc_auc_score(y, lgb_pred)
cat2_auc = roc_auc_score(y, cat2_pred)

prob_equal_pred_3 = (xgb_pred + lgb_pred + cat2_pred) / 3.0
prob_equal_auc_3 = roc_auc_score(y, prob_equal_pred_3)

baseline_prob_pred = (
    BASELINE_WEIGHTS[0] * xgb_pred +
    BASELINE_WEIGHTS[1] * lgb_pred +
    BASELINE_WEIGHTS[2] * cat2_pred
)
baseline_prob_auc = roc_auc_score(y, baseline_prob_pred)

print(f"xgb AUC                  : {xgb_auc:.6f}")
print(f"lgb AUC                  : {lgb_auc:.6f}")
print(f"cat2(v17) AUC            : {cat2_auc:.6f}")
print(f"prob equal ensemble 3 AUC: {prob_equal_auc_3:.6f}")
print(f"{BASELINE_NAME} AUC      : {baseline_prob_auc:.6f}")


# =========================================================
# 4. rank 변환
# =========================================================
print_section("4. rank 변환")

for col in ["xgb_pred", "lgb_pred", "cat2_pred"]:
    oof_df[col.replace("_pred", "_rank")] = to_rank_percentile(oof_df[col].values)
    sub_df[col.replace("_pred", "_rank")] = to_rank_percentile(sub_df[col].values)

xgb_rank = oof_df["xgb_rank"].values
lgb_rank = oof_df["lgb_rank"].values
cat2_rank = oof_df["cat2_rank"].values

xgb_sub_rank = sub_df["xgb_rank"].values
lgb_sub_rank = sub_df["lgb_rank"].values
cat2_sub_rank = sub_df["cat2_rank"].values

rank_equal_pred_3 = (xgb_rank + lgb_rank + cat2_rank) / 3.0
rank_equal_auc_3 = roc_auc_score(y, rank_equal_pred_3)

print(f"rank equal ensemble 3 AUC: {rank_equal_auc_3:.6f}")


# =========================================================
# 5. weighted rank ensemble grid search
# =========================================================
print_section("5. weighted rank ensemble grid search")

weight_candidates = generate_weight_candidates_3(step=SEARCH_STEP)
print(f"rank weight 조합 수: {len(weight_candidates)}")

rank_rows = []
best_rank_auc = -1.0
best_rank_weights = None
best_rank_oof_pred = None

for i, (wx, wl, wc) in enumerate(weight_candidates, start=1):
    pred = wx * xgb_rank + wl * lgb_rank + wc * cat2_rank
    auc = roc_auc_score(y, pred)

    rank_rows.append({
        "xgb_weight": wx,
        "lgb_weight": wl,
        "cat2_weight": wc,
        "rank_auc": auc
    })

    if auc > best_rank_auc:
        best_rank_auc = auc
        best_rank_weights = (wx, wl, wc)
        best_rank_oof_pred = pred.copy()

    if i % 500 == 0 or i == len(weight_candidates):
        print(f"진행률: {i}/{len(weight_candidates)}")

rank_search_df = pd.DataFrame(rank_rows).sort_values("rank_auc", ascending=False).reset_index(drop=True)
rank_search_df.to_csv(RANK_SEARCH_RESULT_PATH, index=False, encoding="utf-8-sig")

print("\nTop 10 weighted rank ensembles")
print(rank_search_df.head(10).to_string(index=False))

print("\n[Best Weighted Rank Ensemble]")
print(
    f"xgb={best_rank_weights[0]:.4f}, "
    f"lgb={best_rank_weights[1]:.4f}, "
    f"cat2={best_rank_weights[2]:.4f}"
)
print(f"Best Rank OOF AUC: {best_rank_auc:.6f}")


# =========================================================
# 6. weighted probability ensemble grid search
# =========================================================
print_section("6. weighted probability ensemble grid search")

prob_rows = []
best_prob_auc = -1.0
best_prob_weights = None
best_prob_oof_pred = None

for i, (wx, wl, wc) in enumerate(weight_candidates, start=1):
    pred = wx * xgb_pred + wl * lgb_pred + wc * cat2_pred
    auc = roc_auc_score(y, pred)

    prob_rows.append({
        "xgb_weight": wx,
        "lgb_weight": wl,
        "cat2_weight": wc,
        "prob_auc": auc
    })

    if auc > best_prob_auc:
        best_prob_auc = auc
        best_prob_weights = (wx, wl, wc)
        best_prob_oof_pred = pred.copy()

    if i % 500 == 0 or i == len(weight_candidates):
        print(f"진행률: {i}/{len(weight_candidates)}")

prob_search_df = pd.DataFrame(prob_rows).sort_values("prob_auc", ascending=False).reset_index(drop=True)
prob_search_df.to_csv(PROB_SEARCH_RESULT_PATH, index=False, encoding="utf-8-sig")

print("\nTop 10 weighted probability ensembles")
print(prob_search_df.head(10).to_string(index=False))

print("\n[Best Weighted Probability Ensemble]")
print(
    f"xgb={best_prob_weights[0]:.4f}, "
    f"lgb={best_prob_weights[1]:.4f}, "
    f"cat2={best_prob_weights[2]:.4f}"
)
print(f"Best Prob OOF AUC: {best_prob_auc:.6f}")


# =========================================================
# 7. hybrid ensemble search
# =========================================================
print_section("7. hybrid ensemble search")

alpha_candidates = generate_alpha_candidates(step=ALPHA_STEP)

# prob 후보 3개
prob_candidate_dict = {
    "prob_equal": {
        "oof_pred": prob_equal_pred_3,
        "sub_pred": (xgb_sub_pred + lgb_sub_pred + cat2_sub_pred) / 3.0,
        "weights": (1/3, 1/3, 1/3),
        "auc": prob_equal_auc_3,
    },
    "prob_baseline": {
        "oof_pred": baseline_prob_pred,
        "sub_pred": (
            BASELINE_WEIGHTS[0] * xgb_sub_pred +
            BASELINE_WEIGHTS[1] * lgb_sub_pred +
            BASELINE_WEIGHTS[2] * cat2_sub_pred
        ),
        "weights": BASELINE_WEIGHTS,
        "auc": baseline_prob_auc,
    },
    "prob_best": {
        "oof_pred": best_prob_oof_pred,
        "sub_pred": (
            best_prob_weights[0] * xgb_sub_pred +
            best_prob_weights[1] * lgb_sub_pred +
            best_prob_weights[2] * cat2_sub_pred
        ),
        "weights": best_prob_weights,
        "auc": best_prob_auc,
    }
}

best_rank_sub_pred = (
    best_rank_weights[0] * xgb_sub_rank +
    best_rank_weights[1] * lgb_sub_rank +
    best_rank_weights[2] * cat2_sub_rank
)

hybrid_rows = []
best_hybrid_auc = -1.0
best_hybrid_alpha = None
best_hybrid_prob_name = None
best_hybrid_oof_pred = None
best_hybrid_sub_pred = None

for prob_name, info in prob_candidate_dict.items():
    prob_oof_pred = info["oof_pred"]
    prob_sub_pred = info["sub_pred"]

    for alpha in alpha_candidates:
        hybrid_oof_pred = alpha * prob_oof_pred + (1.0 - alpha) * best_rank_oof_pred
        hybrid_auc = roc_auc_score(y, hybrid_oof_pred)

        hybrid_rows.append({
            "prob_name": prob_name,
            "alpha_prob": alpha,
            "alpha_rank": round(1.0 - alpha, 4),
            "hybrid_auc": hybrid_auc,
            "rank_auc": best_rank_auc,
            "prob_auc": info["auc"],
            "prob_weights_xgb": round(float(info["weights"][0]), 4),
            "prob_weights_lgb": round(float(info["weights"][1]), 4),
            "prob_weights_cat2": round(float(info["weights"][2]), 4),
            "rank_weights_xgb": round(float(best_rank_weights[0]), 4),
            "rank_weights_lgb": round(float(best_rank_weights[1]), 4),
            "rank_weights_cat2": round(float(best_rank_weights[2]), 4),
        })

        if hybrid_auc > best_hybrid_auc:
            best_hybrid_auc = hybrid_auc
            best_hybrid_alpha = alpha
            best_hybrid_prob_name = prob_name
            best_hybrid_oof_pred = hybrid_oof_pred.copy()
            best_hybrid_sub_pred = alpha * prob_sub_pred + (1.0 - alpha) * best_rank_sub_pred

hybrid_search_df = pd.DataFrame(hybrid_rows).sort_values("hybrid_auc", ascending=False).reset_index(drop=True)
hybrid_search_df.to_csv(HYBRID_SEARCH_RESULT_PATH, index=False, encoding="utf-8-sig")

print("\nTop 20 hybrid ensembles")
print(hybrid_search_df.head(20).to_string(index=False))

print("\n[Best Hybrid Ensemble]")
print(f"prob source  : {best_hybrid_prob_name}")
print(f"alpha(prob)  : {best_hybrid_alpha:.4f}")
print(f"alpha(rank)  : {1.0 - best_hybrid_alpha:.4f}")
print(f"Best Hybrid OOF AUC: {best_hybrid_auc:.6f}")


# =========================================================
# 8. best hybrid submission 생성
# =========================================================
print_section("8. best hybrid submission 생성")

hybrid_oof_path = os.path.join(
    OUTPUT_DIR,
    f"ensemble_v7_hybrid_best_{best_hybrid_prob_name}_"
    f"prob{int(round(best_hybrid_alpha * 100))}_"
    f"rank{int(round((1.0 - best_hybrid_alpha) * 100))}_oof.csv"
)
hybrid_sub_path = os.path.join(
    OUTPUT_DIR,
    f"ensemble_v7_hybrid_best_{best_hybrid_prob_name}_"
    f"prob{int(round(best_hybrid_alpha * 100))}_"
    f"rank{int(round((1.0 - best_hybrid_alpha) * 100))}_submission.csv"
)

hybrid_oof_df = pd.DataFrame({
    ID_COL: oof_df[ID_COL].values,
    TARGET_COL: y,
    "hybrid_ensemble_oof_pred": best_hybrid_oof_pred
})
hybrid_oof_df.to_csv(hybrid_oof_path, index=False, encoding="utf-8-sig")

final_submission = sample_sub.copy()
final_submission[SUBMIT_TARGET_COL] = best_hybrid_sub_pred
hybrid_submission_df = final_submission[[SUBMIT_ID_COL, SUBMIT_TARGET_COL]].copy()
hybrid_submission_df.to_csv(hybrid_sub_path, index=False, encoding="utf-8-sig")

print(f"[저장] {hybrid_oof_path}")
print(f"[저장] {hybrid_sub_path}")


# =========================================================
# 9. 결과 요약
# =========================================================
print_section("9. 결과 요약 / baseline 비교")

best_single_auc = max(xgb_auc, lgb_auc, cat2_auc)
gain_vs_prob_baseline = best_hybrid_auc - baseline_prob_auc
gain_vs_rank_best = best_hybrid_auc - best_rank_auc
gain_vs_prob_best = best_hybrid_auc - best_prob_auc
gain_vs_best_single = best_hybrid_auc - best_single_auc

print(f"best single auc           : {best_single_auc:.6f}")
print(f"prob equal auc (3)        : {prob_equal_auc_3:.6f}")
print(f"prob baseline auc         : {baseline_prob_auc:.6f}")
print(f"best weighted prob auc    : {best_prob_auc:.6f}")
print(f"best weighted rank auc    : {best_rank_auc:.6f}")
print(f"best hybrid auc           : {best_hybrid_auc:.6f}")

print(f"\ngain vs prob baseline     : {gain_vs_prob_baseline:+.6f}")
print(f"gain vs best rank         : {gain_vs_rank_best:+.6f}")
print(f"gain vs best prob         : {gain_vs_prob_best:+.6f}")
print(f"gain vs best single       : {gain_vs_best_single:+.6f}")


# =========================================================
# 10. OOF 상관계수
# =========================================================
print_section("10. OOF 상관계수")

corr_df = oof_df[["xgb_pred", "lgb_pred", "cat2_pred"]].corr()
print(corr_df)

corr_path = os.path.join(OUTPUT_DIR, "ensemble_v7_oof_correlation.csv")
corr_df.to_csv(corr_path, encoding="utf-8-sig")
print(f"\n저장: {corr_path}")


# =========================================================
# 11. 결과 로그 저장
# =========================================================
print_section("11. 결과 로그 저장")

log_row = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "experiment_name": EXPERIMENT_NAME,
    "baseline_name": BASELINE_NAME,

    "xgb_auc": round(float(xgb_auc), 6),
    "lgb_auc": round(float(lgb_auc), 6),
    "cat2_auc": round(float(cat2_auc), 6),

    "prob_equal_auc_3": round(float(prob_equal_auc_3), 6),
    "prob_baseline_auc": round(float(baseline_prob_auc), 6),
    "best_prob_auc": round(float(best_prob_auc), 6),

    "rank_equal_auc_3": round(float(rank_equal_auc_3), 6),
    "best_rank_auc": round(float(best_rank_auc), 6),

    "best_hybrid_auc": round(float(best_hybrid_auc), 6),
    "best_hybrid_prob_name": best_hybrid_prob_name,
    "best_hybrid_alpha_prob": round(float(best_hybrid_alpha), 4),
    "best_hybrid_alpha_rank": round(float(1.0 - best_hybrid_alpha), 4),

    "best_rank_xgb_weight": round(float(best_rank_weights[0]), 4),
    "best_rank_lgb_weight": round(float(best_rank_weights[1]), 4),
    "best_rank_cat2_weight": round(float(best_rank_weights[2]), 4),

    "best_prob_xgb_weight": round(float(best_prob_weights[0]), 4),
    "best_prob_lgb_weight": round(float(best_prob_weights[1]), 4),
    "best_prob_cat2_weight": round(float(best_prob_weights[2]), 4),

    "gain_vs_prob_baseline": round(float(gain_vs_prob_baseline), 6),
    "gain_vs_best_rank": round(float(gain_vs_rank_best), 6),
    "gain_vs_best_prob": round(float(gain_vs_prob_best), 6),

    "search_step": SEARCH_STEP,
    "alpha_step": ALPHA_STEP,
}

result_log_df = pd.DataFrame([log_row])
if os.path.exists(RESULT_LOG_PATH):
    old = pd.read_csv(RESULT_LOG_PATH)
    result_log_df = pd.concat([old, result_log_df], ignore_index=True)

result_log_df.to_csv(RESULT_LOG_PATH, index=False, encoding="utf-8-sig")
print(f"[저장] {RESULT_LOG_PATH}")
print(pd.DataFrame([log_row]).to_string(index=False))


# =========================================================
# 12. 완료
# =========================================================
print_section("12. 완료")

print(f"실험명                    : {EXPERIMENT_NAME}")
print(f"best hybrid auc           : {best_hybrid_auc:.6f}")
print(f"best hybrid prob source   : {best_hybrid_prob_name}")
print(f"best hybrid alpha(prob)   : {best_hybrid_alpha:.4f}")
print(f"best hybrid alpha(rank)   : {1.0 - best_hybrid_alpha:.4f}")
print(
    f"best rank weights         : "
    f"xgb={best_rank_weights[0]:.4f}, "
    f"lgb={best_rank_weights[1]:.4f}, "
    f"cat2={best_rank_weights[2]:.4f}"
)
print(
    f"best prob weights         : "
    f"xgb={best_prob_weights[0]:.4f}, "
    f"lgb={best_prob_weights[1]:.4f}, "
    f"cat2={best_prob_weights[2]:.4f}"
)

print("\n저장 파일:")
print(f"  1. {RANK_SEARCH_RESULT_PATH}")
print(f"  2. {PROB_SEARCH_RESULT_PATH}")
print(f"  3. {HYBRID_SEARCH_RESULT_PATH}")
print(f"  4. {hybrid_oof_path}")
print(f"  5. {hybrid_sub_path}")
print(f"  6. {RESULT_LOG_PATH}")
print(f"  7. {corr_path}")