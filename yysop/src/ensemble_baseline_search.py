# =========================================================
# ensemble_baseline_search.py
# ---------------------------------------------------------
# 목적
# 1) XGB / CAT / LGB OOF 예측 파일을 불러온다.
# 2) weighted ensemble best 조합을 grid search로 찾는다.
# 3) best weighted ensemble을 baseline으로 기록한다.
# 4) best baseline submission 파일도 저장한다.
#
# 실행 예시
#   python yysop/src/ensemble_baseline_search.py
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
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

XGB_OOF_PATH = os.path.join(OUTPUT_DIR, "xgb_v2_reg_relax_oof_predictions.csv")
CAT_OOF_PATH = os.path.join(OUTPUT_DIR, "catboost_v2_catboost_baseline_v2_oof_predictions.csv")
LGB_OOF_PATH = os.path.join(OUTPUT_DIR, "lightgbm_v1_lightgbm_baseline_v1_oof_predictions.csv")

XGB_SUB_PATH = os.path.join(OUTPUT_DIR, "xgb_v2_reg_relax_submission.csv")
CAT_SUB_PATH = os.path.join(OUTPUT_DIR, "catboost_v2_catboost_baseline_v2_submission.csv")
LGB_SUB_PATH = os.path.join(OUTPUT_DIR, "lightgbm_v1_lightgbm_baseline_v1_submission.csv")

BASELINE_LOG_PATH = os.path.join(OUTPUT_DIR, "ensemble_baseline_log.csv")
DETAIL_RESULT_PATH = os.path.join(OUTPUT_DIR, "ensemble_weight_search_results.csv")

TARGET_COL = "임신 성공 여부"
ID_COL = "ID"
PRED_COL = "oof_pred_prob"

SEARCH_STEP = 0.01
ROUND_DIGITS = 4


# =========================================================
# 1. 보조 함수
# =========================================================
def print_section(title: str):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def load_oof_file(path, model_name, keep_target=False):
    df = pd.read_csv(path)

    required_cols = [ID_COL, PRED_COL]
    if keep_target:
        required_cols.append(TARGET_COL)

    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"[{model_name}] OOF 파일에 필요한 컬럼이 없습니다: {missing_cols}")

    use_cols = [ID_COL, PRED_COL]
    if keep_target:
        use_cols = [ID_COL, TARGET_COL, PRED_COL]

    out = df[use_cols].copy()
    out = out.rename(columns={PRED_COL: f"{model_name}_pred"})
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
        raise ValueError(f"병합 후 필요한 컬럼이 없습니다: {missing_cols}")

    return merged


def merge_submission_dfs(xgb, cat, lgb):
    merged = xgb.merge(cat, on=ID_COL, how="inner")
    merged = merged.merge(lgb, on=ID_COL, how="inner")

    required_cols = [ID_COL, "xgb_pred", "cat_pred", "lgb_pred"]
    missing_cols = [c for c in required_cols if c not in merged.columns]
    if missing_cols:
        raise ValueError(f"submission 병합 후 필요한 컬럼이 없습니다: {missing_cols}")

    return merged


def generate_weight_candidates(step=0.01):
    weights = []
    grid = np.arange(0, 1 + step, step)

    for wx in grid:
        for wc in grid:
            wl = 1.0 - wx - wc
            if wl < 0 or wl > 1:
                continue

            wx_r = round(float(wx), ROUND_DIGITS)
            wc_r = round(float(wc), ROUND_DIGITS)
            wl_r = round(float(wl), ROUND_DIGITS)

            if abs((wx_r + wc_r + wl_r) - 1.0) > 1e-8:
                continue

            weights.append((wx_r, wc_r, wl_r))

    weights = list(dict.fromkeys(weights))
    return weights


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
print("\n[OOF head]")
print(oof_df.head())
print("\n[SUB head]")
print(sub_df.head())


# =========================================================
# 3. 단일 모델 / 기본 앙상블 baseline 확인
# =========================================================
print_section("3. 단일 모델 / 기본 앙상블 baseline 확인")

y = oof_df[TARGET_COL].values
xgb_pred = oof_df["xgb_pred"].values
cat_pred = oof_df["cat_pred"].values
lgb_pred = oof_df["lgb_pred"].values

single_scores = {
    "xgb": roc_auc_score(y, xgb_pred),
    "cat": roc_auc_score(y, cat_pred),
    "lgb": roc_auc_score(y, lgb_pred),
}

equal_pred = (xgb_pred + cat_pred + lgb_pred) / 3.0
equal_auc = roc_auc_score(y, equal_pred)

for k, v in single_scores.items():
    print(f"{k:>5s} AUC : {v:.6f}")
print(f"{'equal':>5s} AUC : {equal_auc:.6f}")


# =========================================================
# 4. weighted ensemble grid search
# =========================================================
print_section("4. weighted ensemble grid search")

weight_candidates = generate_weight_candidates(step=SEARCH_STEP)
print(f"weight 조합 수: {len(weight_candidates)}")

rows = []
best_auc = -1.0
best_weights = None
best_pred = None

for i, (wx, wc, wl) in enumerate(weight_candidates, start=1):
    pred = wx * xgb_pred + wc * cat_pred + wl * lgb_pred
    auc = roc_auc_score(y, pred)

    rows.append({
        "xgb_weight": wx,
        "cat_weight": wc,
        "lgb_weight": wl,
        "oof_auc": auc
    })

    if auc > best_auc:
        best_auc = auc
        best_weights = (wx, wc, wl)
        best_pred = pred.copy()

    if i % 500 == 0 or i == len(weight_candidates):
        print(f"진행률: {i}/{len(weight_candidates)}")

search_result_df = pd.DataFrame(rows).sort_values("oof_auc", ascending=False).reset_index(drop=True)
search_result_df.to_csv(DETAIL_RESULT_PATH, index=False, encoding="utf-8-sig")

print("\nTop 10 weighted ensembles")
print(search_result_df.head(10).to_string(index=False))

print("\n[Best Weighted Ensemble]")
print(f"xgb={best_weights[0]:.4f}, cat={best_weights[1]:.4f}, lgb={best_weights[2]:.4f}")
print(f"Best OOF AUC: {best_auc:.6f}")
print(f"[저장] {DETAIL_RESULT_PATH}")


# =========================================================
# 5. baseline 기록
# =========================================================
print_section("5. baseline 기록")

baseline_row = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "baseline_name": "best_weighted_ensemble",
    "search_step": SEARCH_STEP,
    "xgb_auc": round(float(single_scores["xgb"]), 6),
    "cat_auc": round(float(single_scores["cat"]), 6),
    "lgb_auc": round(float(single_scores["lgb"]), 6),
    "equal_auc": round(float(equal_auc), 6),
    "best_weighted_auc": round(float(best_auc), 6),
    "best_xgb_weight": round(float(best_weights[0]), 4),
    "best_cat_weight": round(float(best_weights[1]), 4),
    "best_lgb_weight": round(float(best_weights[2]), 4),
    "notes": "Stacking comparison baseline"
}

baseline_log_df = pd.DataFrame([baseline_row])

if os.path.exists(BASELINE_LOG_PATH):
    old = pd.read_csv(BASELINE_LOG_PATH)
    baseline_log_df = pd.concat([old, baseline_log_df], ignore_index=True)

baseline_log_df.to_csv(BASELINE_LOG_PATH, index=False, encoding="utf-8-sig")
print(f"[저장] {BASELINE_LOG_PATH}")
print(pd.DataFrame([baseline_row]).to_string(index=False))


# =========================================================
# 6. best baseline OOF 저장
# =========================================================
print_section("6. best baseline OOF 저장")

baseline_oof_df = pd.DataFrame({
    ID_COL: oof_df[ID_COL].values,
    TARGET_COL: y,
    "weighted_oof_pred": best_pred,
})

baseline_oof_path = os.path.join(OUTPUT_DIR, "best_weighted_ensemble_oof.csv")
baseline_oof_df.to_csv(baseline_oof_path, index=False, encoding="utf-8-sig")
print(f"[저장] {baseline_oof_path}")


# =========================================================
# 7. best baseline submission 생성
# =========================================================
print_section("7. best baseline submission 생성")

wx, wc, wl = best_weights
sub_pred = (
    wx * sub_df["xgb_pred"].values
    + wc * sub_df["cat_pred"].values
    + wl * sub_df["lgb_pred"].values
)

baseline_submission = pd.DataFrame({
    ID_COL: sub_df[ID_COL].values,
    TARGET_COL: sub_pred
})

baseline_sub_path = os.path.join(OUTPUT_DIR, "best_weighted_ensemble_submission.csv")
baseline_submission.to_csv(baseline_sub_path, index=False, encoding="utf-8-sig")

print(f"[저장] {baseline_sub_path}")
print(baseline_submission.head(10))


# =========================================================
# 8. 완료 요약
# =========================================================
print_section("8. 완료")

print("Single Best AUC")
for k, v in single_scores.items():
    print(f"  {k:>5s} : {v:.6f}")

print(f"\nEqual Ensemble AUC     : {equal_auc:.6f}")
print(f"Best Weighted AUC      : {best_auc:.6f}")
print(f"Best Weights           : xgb={best_weights[0]:.4f}, cat={best_weights[1]:.4f}, lgb={best_weights[2]:.4f}")

print("\n저장 파일:")
print(f"  1. {DETAIL_RESULT_PATH}")
print(f"  2. {BASELINE_LOG_PATH}")
print(f"  3. {baseline_oof_path}")
print(f"  4. {baseline_sub_path}")