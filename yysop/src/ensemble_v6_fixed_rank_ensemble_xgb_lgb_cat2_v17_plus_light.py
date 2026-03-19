# =========================================================
# ensemble_v6_fixed_rank_ensemble_xgb_lgb_cat2_v17_plus_light.py
# ---------------------------------------------------------
# 목적
# 1) XGB / LGB / CAT2(v17_plus_light) 예측을 불러온다.
# 2) CAT2는 npy/csv 형식도 지원한다.
# 3) 고정 가중치 rank ensemble을 평가한다.
# 4) OOF / submission / results log를 저장한다.
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
RESULT_DIR = os.path.join(BASE_DIR, "src", "result")   # v17_plus_light 산출물 위치
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
# CAT2(v17_plus_light)
# ---------------------------------------------------------
CAT2_OOF_NPY_PATH = os.path.join(RESULT_DIR, "oof_v17_plus_light_final.npy")
CAT2_SUB_NPY_PATH = os.path.join(RESULT_DIR, "test_v17_plus_light_final.npy")
CAT2_SUB_CSV_PATH = os.path.join(
    RESULT_DIR,
    "sample_submission_v17_plus_light_20260318_182327.csv"
)

# 중요도 파일
XGB_IMPORTANCE_PATH = os.path.join(OUTPUT_DIR, "xgb_v2_reg_relax_feature_importance.csv")
LGB_IMPORTANCE_PATH = os.path.join(OUTPUT_DIR, "lightgbm_v1_lightgbm_baseline_v1_feature_importance.csv")
CAT2_IMPORTANCE_PATH = os.path.join(RESULT_DIR, "feature_importance_v17_plus_light.csv")

RESULT_LOG_PATH = os.path.join(OUTPUT_DIR, "ensemble_v6_fixed_rank_results_log.csv")
SEARCH_RESULT_PATH = os.path.join(OUTPUT_DIR, "ensemble_v6_fixed_rank_result_v17_plus_light.csv")

TARGET_COL = "임신 성공 여부"
ID_COL = "ID"
OOF_PRED_COL = "oof_pred_prob"

SUBMIT_ID_COL = "ID"
SUBMIT_TARGET_COL = "probability"

BASELINE_AUC = 0.740444
BASELINE_NAME = "prob_weighted_ensemble_3models_old"
BASELINE_WEIGHTS = "xgb=0.33, cat=0.40, lgb=0.27"

EXPERIMENT_NAME = "ensemble_v6_fixed_rank_xgb20_lgb08_cat272_v17_plus_light"

# =========================================================
# 고정 가중치
# 순서: xgb / lgb / cat2
# =========================================================
FIXED_XGB_WEIGHT = 0.20
FIXED_LGB_WEIGHT = 0.08
FIXED_CAT2_WEIGHT = 0.72


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

    required_cols = [ID_COL, TARGET_COL, "xgb_pred", "lgb_pred", "cat2_pred"]
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


# =========================================================
# 2. 데이터 로드
# =========================================================
print_section("2. 데이터 로드")

print("[경로 확인]")
print("CAT2_OOF_NPY_PATH      :", CAT2_OOF_NPY_PATH, os.path.exists(CAT2_OOF_NPY_PATH))
print("CAT2_SUB_NPY_PATH      :", CAT2_SUB_NPY_PATH, os.path.exists(CAT2_SUB_NPY_PATH))
print("CAT2_SUB_CSV_PATH      :", CAT2_SUB_CSV_PATH, os.path.exists(CAT2_SUB_CSV_PATH))
print("CAT2_IMPORTANCE_PATH   :", CAT2_IMPORTANCE_PATH, os.path.exists(CAT2_IMPORTANCE_PATH))

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

xgb_auc = roc_auc_score(y, oof_df["xgb_pred"].values)
lgb_auc = roc_auc_score(y, oof_df["lgb_pred"].values)
cat2_auc = roc_auc_score(y, oof_df["cat2_pred"].values)

prob_equal_pred_3 = (
    oof_df["xgb_pred"].values
    + oof_df["lgb_pred"].values
    + oof_df["cat2_pred"].values
) / 3.0
prob_equal_auc_3 = roc_auc_score(y, prob_equal_pred_3)

print(f"xgb AUC                  : {xgb_auc:.6f}")
print(f"lgb AUC                  : {lgb_auc:.6f}")
print(f"cat2(v17_plus_light) AUC : {cat2_auc:.6f}")
print(f"prob equal ensemble 3 AUC: {prob_equal_auc_3:.6f}")
print(f"{BASELINE_NAME} AUC      : {BASELINE_AUC:.6f}")


# =========================================================
# 4. rank 변환
# =========================================================
print_section("4. rank 변환")

for col in ["xgb_pred", "lgb_pred", "cat2_pred"]:
    oof_df[col.replace("_pred", "_rank")] = to_rank_percentile(oof_df[col].values)
    sub_df[col.replace("_pred", "_rank")] = to_rank_percentile(sub_df[col].values)

rank_equal_pred_3 = (
    oof_df["xgb_rank"].values
    + oof_df["lgb_rank"].values
    + oof_df["cat2_rank"].values
) / 3.0
rank_equal_auc_3 = roc_auc_score(y, rank_equal_pred_3)

print(f"rank equal ensemble 3 AUC: {rank_equal_auc_3:.6f}")


# =========================================================
# 5. fixed weighted rank ensemble
# =========================================================
print_section("5. fixed weighted rank ensemble")

wx = FIXED_XGB_WEIGHT
wl = FIXED_LGB_WEIGHT
w3 = FIXED_CAT2_WEIGHT

if abs((wx + wl + w3) - 1.0) > 1e-8:
    raise ValueError("고정 가중치 합이 1이 아닙니다.")

xgb_rank = oof_df["xgb_rank"].values
lgb_rank = oof_df["lgb_rank"].values
cat2_rank = oof_df["cat2_rank"].values

best_oof_pred = wx * xgb_rank + wl * lgb_rank + w3 * cat2_rank
best_auc = roc_auc_score(y, best_oof_pred)
best_weights = (wx, wl, w3)

fixed_result_df = pd.DataFrame([{
    "xgb_weight": wx,
    "lgb_weight": wl,
    "cat2_weight": w3,
    "rank_auc": best_auc
}])
fixed_result_df.to_csv(SEARCH_RESULT_PATH, index=False, encoding="utf-8-sig")

print("[Fixed Weighted Rank Ensemble]")
print(f"xgb={wx:.4f}, lgb={wl:.4f}, cat2={w3:.4f}")
print(f"Fixed Rank OOF AUC: {best_auc:.6f}")


# =========================================================
# 6. best submission 생성
# =========================================================
print_section("6. best submission 생성")

best_sub_pred = (
    wx * sub_df["xgb_rank"].values
    + wl * sub_df["lgb_rank"].values
    + w3 * sub_df["cat2_rank"].values
)

rank_oof_path = os.path.join(
    OUTPUT_DIR,
    f"ensemble_v6_fixed_rank_xgb{int(round(wx * 100))}_"
    f"lgb{int(round(wl * 100))}_cat2{int(round(w3 * 100))}_v17_plus_light_oof.csv"
)
rank_sub_path = os.path.join(
    OUTPUT_DIR,
    f"ensemble_v6_fixed_rank_xgb{int(round(wx * 100))}_"
    f"lgb{int(round(wl * 100))}_cat2{int(round(w3 * 100))}_v17_plus_light_submission.csv"
)

rank_oof_df = pd.DataFrame({
    ID_COL: oof_df[ID_COL].values,
    TARGET_COL: y,
    "rank_ensemble_oof_pred": best_oof_pred
})
rank_oof_df.to_csv(rank_oof_path, index=False, encoding="utf-8-sig")

final_submission = sample_sub.copy()
final_submission[SUBMIT_TARGET_COL] = best_sub_pred
rank_submission_df = final_submission[[SUBMIT_ID_COL, SUBMIT_TARGET_COL]].copy()
rank_submission_df.to_csv(rank_sub_path, index=False, encoding="utf-8-sig")

print(f"[저장] {rank_oof_path}")
print(f"[저장] {rank_sub_path}")


# =========================================================
# 7. 결과 요약
# =========================================================
print_section("7. 결과 요약 / baseline 비교")

gain_vs_prob_baseline = best_auc - BASELINE_AUC
gain_vs_rank_equal_3 = best_auc - rank_equal_auc_3
gain_vs_prob_equal_3 = best_auc - prob_equal_auc_3
best_single_auc = max(xgb_auc, lgb_auc, cat2_auc)
gain_vs_best_single = best_auc - best_single_auc

print(f"best single auc           : {best_single_auc:.6f}")
print(f"prob equal auc (3)        : {prob_equal_auc_3:.6f}")
print(f"rank equal auc (3)        : {rank_equal_auc_3:.6f}")
print(f"prob weighted baseline    : {BASELINE_AUC:.6f}")
print(f"fixed weighted rank auc   : {best_auc:.6f}")

print(f"\ngain vs prob baseline     : {gain_vs_prob_baseline:+.6f}")
print(f"gain vs rank equal (3)    : {gain_vs_rank_equal_3:+.6f}")
print(f"gain vs prob equal (3)    : {gain_vs_prob_equal_3:+.6f}")
print(f"gain vs best single       : {gain_vs_best_single:+.6f}")


# =========================================================
# 8. OOF 상관계수
# =========================================================
print_section("8. OOF 상관계수")

corr_df = oof_df[["xgb_pred", "lgb_pred", "cat2_pred"]].corr()
print(corr_df)

corr_path = os.path.join(OUTPUT_DIR, "ensemble_v6_oof_correlation_v17_plus_light.csv")
corr_df.to_csv(corr_path, encoding="utf-8-sig")
print(f"\n저장: {corr_path}")


# =========================================================
# 9. 중요도 병합
# =========================================================
print_section("9. Ensemble Feature Importance")

imp_frames = []

if os.path.exists(XGB_IMPORTANCE_PATH):
    xgb_imp = pd.read_csv(XGB_IMPORTANCE_PATH).rename(columns={"importance": "xgb_imp"})
    imp_frames.append(xgb_imp)

if os.path.exists(LGB_IMPORTANCE_PATH):
    lgb_imp = pd.read_csv(LGB_IMPORTANCE_PATH).rename(columns={"importance": "lgb_imp"})
    imp_frames.append(lgb_imp)

if os.path.exists(CAT2_IMPORTANCE_PATH):
    cat2_imp = pd.read_csv(CAT2_IMPORTANCE_PATH).rename(columns={"importance": "cat2_imp"})
    imp_frames.append(cat2_imp)

if len(imp_frames) >= 2:
    imp = imp_frames[0]
    for df_imp in imp_frames[1:]:
        imp = imp.merge(df_imp, on="feature", how="outer")
    imp = imp.fillna(0)

    for col in ["xgb_imp", "lgb_imp", "cat2_imp"]:
        if col not in imp.columns:
            imp[col] = 0

    imp["ensemble_importance"] = (
        wx * imp["xgb_imp"] +
        wl * imp["lgb_imp"] +
        w3 * imp["cat2_imp"]
    )

    imp = imp.sort_values("ensemble_importance", ascending=False)

    print(imp.head(30).to_string(index=False))

    ensemble_imp_path = os.path.join(OUTPUT_DIR, "ensemble_v6_feature_importance_v17_plus_light.csv")
    imp.to_csv(ensemble_imp_path, index=False, encoding="utf-8-sig")
    print(f"\n저장: {ensemble_imp_path}")
else:
    print("중요도 파일이 충분하지 않아 스킵합니다.")


# =========================================================
# 10. 로그 저장
# =========================================================
print_section("10. 결과 로그 저장")

log_row = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "experiment_name": EXPERIMENT_NAME,
    "baseline_name": BASELINE_NAME,
    "baseline_auc": round(float(BASELINE_AUC), 6),
    "xgb_auc": round(float(xgb_auc), 6),
    "lgb_auc": round(float(lgb_auc), 6),
    "cat2_auc": round(float(cat2_auc), 6),
    "prob_equal_auc_3": round(float(prob_equal_auc_3), 6),
    "rank_equal_auc_3": round(float(rank_equal_auc_3), 6),
    "fixed_rank_auc": round(float(best_auc), 6),
    "fixed_xgb_weight": round(float(best_weights[0]), 4),
    "fixed_lgb_weight": round(float(best_weights[1]), 4),
    "fixed_cat2_weight": round(float(best_weights[2]), 4),
    "gain_vs_prob_baseline": round(float(gain_vs_prob_baseline), 6),
}

result_log_df = pd.DataFrame([log_row])
if os.path.exists(RESULT_LOG_PATH):
    old = pd.read_csv(RESULT_LOG_PATH)
    result_log_df = pd.concat([old, result_log_df], ignore_index=True)

result_log_df.to_csv(RESULT_LOG_PATH, index=False, encoding="utf-8-sig")
print(f"[저장] {RESULT_LOG_PATH}")
print(pd.DataFrame([log_row]).to_string(index=False))


# =========================================================
# 11. 완료
# =========================================================
print_section("11. 완료")

print(f"실험명                    : {EXPERIMENT_NAME}")
print(f"fixed rank ensemble auc   : {best_auc:.6f}")
print(
    f"fixed rank weights        : "
    f"xgb={best_weights[0]:.4f}, "
    f"lgb={best_weights[1]:.4f}, "
    f"cat2={best_weights[2]:.4f}"
)
print(f"submit columns            : [{SUBMIT_ID_COL}, {SUBMIT_TARGET_COL}]")

print("\n저장 파일:")
print(f"  1. {SEARCH_RESULT_PATH}")
print(f"  2. {rank_oof_path}")
print(f"  3. {rank_sub_path}")
print(f"  4. {RESULT_LOG_PATH}")
print(f"  5. {corr_path}")