# =========================================================
# ensemble_v8_rank_ensemble_xgb_lgb_v21.py
# ---------------------------------------------------------
# 목적
# 1) XGB / LGB / V21(CatBoost 대체) 예측을 불러온다.
# 2) 3모델 weighted rank ensemble을 평가한다.
# 3) best rank ensemble OOF / submission / results log를 저장한다.
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
# V21 (기존 cat2 대체)
# ---------------------------------------------------------
V21_OOF_NPY_PATH = os.path.join(OUTPUT_DIR, "oof_v21_expert_final.npy")
V21_SUB_NPY_PATH = os.path.join(OUTPUT_DIR, "test_v21_expert_final.npy")

RESULT_LOG_PATH = os.path.join(OUTPUT_DIR, "ensemble_v8_rank_results_log.csv")
SEARCH_RESULT_PATH = os.path.join(OUTPUT_DIR, "ensemble_v8_rank_search_results.csv")

TARGET_COL = "임신 성공 여부"
ID_COL = "ID"
OOF_PRED_COL = "oof_pred_prob"

SUBMIT_ID_COL = "ID"
SUBMIT_TARGET_COL = "probability"

SEARCH_STEP = 0.02
ROUND_DIGITS = 4

EXPERIMENT_NAME = "ensemble_v8_rank_ensemble_xgb_lgb_v21"


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


def load_model_oof_from_npy(oof_npy_path, train_path, model_name="v21"):
    train_df = pd.read_csv(train_path, usecols=[ID_COL, TARGET_COL])
    pred = np.load(oof_npy_path)

    if len(train_df) != len(pred):
        raise ValueError(
            f"[{model_name}] train 행 수({len(train_df)})와 OOF npy 길이({len(pred)})가 다릅니다."
        )

    out = train_df.copy()
    out[f"{model_name}_pred"] = pred
    return out


def load_model_submission_from_npy(sub_npy_path, test_path, model_name="v21"):
    test_df = pd.read_csv(test_path, usecols=[ID_COL])
    pred = np.load(sub_npy_path)

    if len(test_df) != len(pred):
        raise ValueError(
            f"[{model_name}] test 행 수({len(test_df)})와 submission npy 길이({len(pred)})가 다릅니다."
        )

    out = test_df.copy()
    out[f"{model_name}_pred"] = pred
    return out


def merge_oof_dfs(xgb, lgb, v21):
    merged = xgb.merge(lgb, on=ID_COL, how="inner")
    merged = merged.merge(v21, on=[ID_COL, TARGET_COL], how="inner")

    required_cols = [
        ID_COL, TARGET_COL,
        "xgb_pred", "lgb_pred", "v21_pred"
    ]
    missing_cols = [c for c in required_cols if c not in merged.columns]
    if missing_cols:
        raise ValueError(f"OOF 병합 후 필요한 컬럼이 없습니다: {missing_cols}")

    return merged


def merge_submission_dfs(xgb, lgb, v21):
    merged = xgb.merge(lgb, on=ID_COL, how="inner")
    merged = merged.merge(v21, on=ID_COL, how="inner")

    required_cols = [ID_COL, "xgb_pred", "lgb_pred", "v21_pred"]
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
            wv = 1.0 - wx - wl
            if wv < 0 or wv > 1:
                continue

            wx_r = round(float(wx), ROUND_DIGITS)
            wl_r = round(float(wl), ROUND_DIGITS)
            wv_r = round(float(wv), ROUND_DIGITS)

            if abs((wx_r + wl_r + wv_r) - 1.0) > 1e-8:
                continue

            weights.append((wx_r, wl_r, wv_r))

    return list(dict.fromkeys(weights))


# =========================================================
# 2. 데이터 로드
# =========================================================
print_section("2. 데이터 로드")

sample_sub = load_sample_submission(SAMPLE_SUB_PATH)

xgb_oof = load_oof_file(XGB_OOF_PATH, "xgb", keep_target=True)
lgb_oof = load_oof_file(LGB_OOF_PATH, "lgb", keep_target=False)
v21_oof = load_model_oof_from_npy(V21_OOF_NPY_PATH, TRAIN_PATH, model_name="v21")

xgb_sub = load_submission_file(XGB_SUB_PATH, "xgb")
lgb_sub = load_submission_file(LGB_SUB_PATH, "lgb")
v21_sub = load_model_submission_from_npy(V21_SUB_NPY_PATH, TEST_PATH, model_name="v21")

oof_df = merge_oof_dfs(xgb_oof, lgb_oof, v21_oof)
sub_df = merge_submission_dfs(xgb_sub, lgb_sub, v21_sub)

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
v21_auc = roc_auc_score(y, oof_df["v21_pred"].values)

prob_equal_pred_3 = (
    oof_df["xgb_pred"].values
    + oof_df["lgb_pred"].values
    + oof_df["v21_pred"].values
) / 3.0
prob_equal_auc_3 = roc_auc_score(y, prob_equal_pred_3)

print(f"xgb AUC                  : {xgb_auc:.6f}")
print(f"lgb AUC                  : {lgb_auc:.6f}")
print(f"v21 AUC                  : {v21_auc:.6f}")
print(f"prob equal ensemble 3 AUC: {prob_equal_auc_3:.6f}")


# =========================================================
# 4. rank 변환
# =========================================================
print_section("4. rank 변환")

for col in ["xgb_pred", "lgb_pred", "v21_pred"]:
    oof_df[col.replace("_pred", "_rank")] = to_rank_percentile(oof_df[col].values)
    sub_df[col.replace("_pred", "_rank")] = to_rank_percentile(sub_df[col].values)

rank_equal_pred_3 = (
    oof_df["xgb_rank"].values
    + oof_df["lgb_rank"].values
    + oof_df["v21_rank"].values
) / 3.0
rank_equal_auc_3 = roc_auc_score(y, rank_equal_pred_3)

print(f"rank equal ensemble 3 AUC: {rank_equal_auc_3:.6f}")


# =========================================================
# 5. weighted rank ensemble grid search
# =========================================================
print_section("5. weighted rank ensemble grid search")

weight_candidates = generate_weight_candidates_3(step=SEARCH_STEP)
print(f"weight 조합 수: {len(weight_candidates)}")

rows = []
best_auc = -1.0
best_weights = None
best_oof_pred = None

xgb_rank = oof_df["xgb_rank"].values
lgb_rank = oof_df["lgb_rank"].values
v21_rank = oof_df["v21_rank"].values

for i, (wx, wl, wv) in enumerate(weight_candidates, start=1):
    pred = wx * xgb_rank + wl * lgb_rank + wv * v21_rank
    auc = roc_auc_score(y, pred)

    rows.append({
        "xgb_weight": wx,
        "lgb_weight": wl,
        "v21_weight": wv,
        "rank_auc": auc
    })

    if auc > best_auc:
        best_auc = auc
        best_weights = (wx, wl, wv)
        best_oof_pred = pred.copy()

    if i % 500 == 0 or i == len(weight_candidates):
        print(f"진행률: {i}/{len(weight_candidates)}")

search_result_df = pd.DataFrame(rows).sort_values("rank_auc", ascending=False).reset_index(drop=True)
search_result_df.to_csv(SEARCH_RESULT_PATH, index=False, encoding="utf-8-sig")

print("\nTop 10 weighted rank ensembles")
print(search_result_df.head(10).to_string(index=False))

print("\n[Best Weighted Rank Ensemble]")
print(
    f"xgb={best_weights[0]:.4f}, "
    f"lgb={best_weights[1]:.4f}, "
    f"v21={best_weights[2]:.4f}"
)
print(f"Best Rank OOF AUC: {best_auc:.6f}")


# =========================================================
# 6. best submission 생성
# =========================================================
print_section("6. best submission 생성")

wx, wl, wv = best_weights
best_sub_pred = (
    wx * sub_df["xgb_rank"].values
    + wl * sub_df["lgb_rank"].values
    + wv * sub_df["v21_rank"].values
)

rank_oof_path = os.path.join(
    OUTPUT_DIR,
    f"ensemble_v8_rank_best_xgb{int(round(wx * 100))}_"
    f"lgb{int(round(wl * 100))}_v21{int(round(wv * 100))}_oof.csv"
)
rank_sub_path = os.path.join(
    OUTPUT_DIR,
    f"ensemble_v8_rank_best_xgb{int(round(wx * 100))}_"
    f"lgb{int(round(wl * 100))}_v21{int(round(wv * 100))}_submission.csv"
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
print_section("7. 결과 요약")

best_single_auc = max(xgb_auc, lgb_auc, v21_auc)
gain_vs_rank_equal_3 = best_auc - rank_equal_auc_3
gain_vs_prob_equal_3 = best_auc - prob_equal_auc_3
gain_vs_best_single = best_auc - best_single_auc

print(f"best single auc           : {best_single_auc:.6f}")
print(f"prob equal auc (3)        : {prob_equal_auc_3:.6f}")
print(f"rank equal auc (3)        : {rank_equal_auc_3:.6f}")
print(f"best weighted rank auc    : {best_auc:.6f}")

print(f"\ngain vs rank equal (3)    : {gain_vs_rank_equal_3:+.6f}")
print(f"gain vs prob equal (3)    : {gain_vs_prob_equal_3:+.6f}")
print(f"gain vs best single       : {gain_vs_best_single:+.6f}")


# =========================================================
# 8. OOF 상관계수
# =========================================================
print_section("8. OOF 상관계수")

corr_df = oof_df[["xgb_pred", "lgb_pred", "v21_pred"]].corr()
print(corr_df)

corr_path = os.path.join(OUTPUT_DIR, "ensemble_v8_oof_correlation.csv")
corr_df.to_csv(corr_path, encoding="utf-8-sig")
print(f"\n저장: {corr_path}")


# =========================================================
# 9. 로그 저장
# =========================================================
print_section("9. 결과 로그 저장")

log_row = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "experiment_name": EXPERIMENT_NAME,
    "xgb_auc": round(float(xgb_auc), 6),
    "lgb_auc": round(float(lgb_auc), 6),
    "v21_auc": round(float(v21_auc), 6),
    "prob_equal_auc_3": round(float(prob_equal_auc_3), 6),
    "rank_equal_auc_3": round(float(rank_equal_auc_3), 6),
    "best_rank_auc": round(float(best_auc), 6),
    "best_xgb_weight": round(float(best_weights[0]), 4),
    "best_lgb_weight": round(float(best_weights[1]), 4),
    "best_v21_weight": round(float(best_weights[2]), 4),
    "gain_vs_best_single": round(float(gain_vs_best_single), 6),
    "search_step": SEARCH_STEP,
}

result_log_df = pd.DataFrame([log_row])
if os.path.exists(RESULT_LOG_PATH):
    old = pd.read_csv(RESULT_LOG_PATH)
    result_log_df = pd.concat([old, result_log_df], ignore_index=True)

result_log_df.to_csv(RESULT_LOG_PATH, index=False, encoding="utf-8-sig")
print(f"[저장] {RESULT_LOG_PATH}")
print(pd.DataFrame([log_row]).to_string(index=False))


# =========================================================
# 10. 완료
# =========================================================
print_section("10. 완료")

print(f"실험명                    : {EXPERIMENT_NAME}")
print(f"best rank ensemble auc    : {best_auc:.6f}")
print(
    f"best rank weights         : "
    f"xgb={best_weights[0]:.4f}, "
    f"lgb={best_weights[1]:.4f}, "
    f"v21={best_weights[2]:.4f}"
)
print(f"submit columns            : [{SUBMIT_ID_COL}, {SUBMIT_TARGET_COL}]")

print("\n저장 파일:")
print(f"  1. {SEARCH_RESULT_PATH}")
print(f"  2. {rank_oof_path}")
print(f"  3. {rank_sub_path}")
print(f"  4. {RESULT_LOG_PATH}")
print(f"  5. {corr_path}")