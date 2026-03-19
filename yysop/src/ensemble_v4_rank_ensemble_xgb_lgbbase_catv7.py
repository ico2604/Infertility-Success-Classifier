# =========================================================
# ensemble_v4_rank_ensemble_xgb_lgbbase_catv7.py
# ---------------------------------------------------------
# 목적
# 1) XGB / CAT / LGB OOF 예측 파일을 불러온다.
# 2) 각 모델 예측값을 rank로 변환한다.
# 3) rank ensemble(equal / weighted)을 평가한다.
# 4) 현재 3모델 probability weighted baseline과 비교한다.
# 5) best rank ensemble OOF / submission / results log를 저장한다.
# 6) 제출 파일은 sample_submission.csv 형식(ID, probability)에 맞춰 저장한다.
#
# 모델 조합
# - XGB : xgb_optuna_v1
# - CAT : catboost_optuna_fixed_v7_ivf_combo
# - LGB : lightgbm_baseline_v1
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

# =========================================================
# 사용할 모델 파일 경로
# =========================================================

# XGB (Optuna v1)
XGB_OOF_PATH = os.path.join(OUTPUT_DIR, "xgb_optuna_v1_oof_predictions.csv")
XGB_SUB_PATH = os.path.join(OUTPUT_DIR, "xgb_optuna_v1_submission.csv")

# CAT (Optuna fixed v7 ivf combo)
CAT_OOF_PATH = os.path.join(OUTPUT_DIR, "catboost_optuna_fixed_v7_ivf_combo_oof_predictions.csv")
CAT_SUB_PATH = os.path.join(OUTPUT_DIR, "catboost_optuna_fixed_v7_ivf_combo_submission.csv")

# LGB (baseline v1)
LGB_OOF_PATH = os.path.join(OUTPUT_DIR, "lightgbm_v1_lightgbm_baseline_v1_oof_predictions.csv")
LGB_SUB_PATH = os.path.join(OUTPUT_DIR, "lightgbm_v1_lightgbm_baseline_v1_submission.csv")

RESULT_LOG_PATH = os.path.join(OUTPUT_DIR, "ensemble_v4_rank_xgb_lgbbase_catv7_results_log.csv")
SEARCH_RESULT_PATH = os.path.join(OUTPUT_DIR, "ensemble_v4_rank_xgb_lgbbase_catv7_search_results.csv")

TARGET_COL = "임신 성공 여부"
ID_COL = "ID"
OOF_PRED_COL = "oof_pred_prob"

SUBMIT_ID_COL = "ID"
SUBMIT_TARGET_COL = "probability"

# =========================================================
# baseline 비교용 설정
# ---------------------------------------------------------
# False면 현재 조합(XGB+CATv7+LGB_base)의 probability weighted baseline을
# 간단 탐색해서 비교함
# True면 FIXED_BASELINE_AUC와 비교
# =========================================================
USE_FIXED_BASELINE = False

FIXED_BASELINE_AUC = 0.740631
FIXED_BASELINE_NAME = "rank_xgb_lgbivf_catv6_prev_best"
FIXED_BASELINE_WEIGHTS = "xgb=0.32, cat=0.52, lgb=0.16"

SEARCH_STEP = 0.01
ROUND_DIGITS = 4

EXPERIMENT_NAME = "ensemble_v4_rank_xgb_lgbbase_catv7"


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


def to_rank_percentile(arr):
    s = pd.Series(arr)
    return s.rank(method="average", pct=True).values


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

    return list(dict.fromkeys(weights))


# =========================================================
# 2. 데이터 로드
# =========================================================
print_section("2. 데이터 로드")

sample_sub = load_sample_submission(SAMPLE_SUB_PATH)

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
print("Sample sub shape :", sample_sub.shape)


# =========================================================
# 3. 기본 성능 확인
# =========================================================
print_section("3. 기본 성능 확인")

y = oof_df[TARGET_COL].values

xgb_auc = roc_auc_score(y, oof_df["xgb_pred"].values)
cat_auc = roc_auc_score(y, oof_df["cat_pred"].values)
lgb_auc = roc_auc_score(y, oof_df["lgb_pred"].values)

prob_equal_pred = (
    oof_df["xgb_pred"].values
    + oof_df["cat_pred"].values
    + oof_df["lgb_pred"].values
) / 3.0
prob_equal_auc = roc_auc_score(y, prob_equal_pred)

if USE_FIXED_BASELINE:
    BASELINE_AUC = FIXED_BASELINE_AUC
    BASELINE_NAME = FIXED_BASELINE_NAME
    BASELINE_WEIGHTS = FIXED_BASELINE_WEIGHTS
else:
    prob_baseline_candidates = [
        (0.34, 0.50, 0.16),
        (0.34, 0.51, 0.15),
        (0.33, 0.51, 0.16),
        (0.33, 0.52, 0.15),
        (0.32, 0.52, 0.16),
        (0.35, 0.50, 0.15),
        (0.35, 0.49, 0.16),
        (0.34, 0.49, 0.17),
        (0.32, 0.53, 0.15),
        (0.31, 0.53, 0.16),
    ]

    best_prob_auc = -1.0
    best_prob_weights = None

    for wx, wc, wl in prob_baseline_candidates:
        pred = (
            wx * oof_df["xgb_pred"].values
            + wc * oof_df["cat_pred"].values
            + wl * oof_df["lgb_pred"].values
        )
        auc = roc_auc_score(y, pred)

        if auc > best_prob_auc:
            best_prob_auc = auc
            best_prob_weights = (wx, wc, wl)

    BASELINE_AUC = best_prob_auc
    BASELINE_NAME = "prob_weighted_current_3model"
    BASELINE_WEIGHTS = (
        f"xgb={best_prob_weights[0]:.2f}, "
        f"cat={best_prob_weights[1]:.2f}, "
        f"lgb={best_prob_weights[2]:.2f}"
    )

print(f"xgb AUC                 : {xgb_auc:.6f}")
print(f"cat AUC                 : {cat_auc:.6f}")
print(f"lgb AUC                 : {lgb_auc:.6f}")
print(f"prob equal ensemble AUC : {prob_equal_auc:.6f}")
print(f"{BASELINE_NAME} AUC     : {BASELINE_AUC:.6f}")
print(f"baseline weights        : {BASELINE_WEIGHTS}")


# =========================================================
# 4. rank 변환
# =========================================================
print_section("4. rank 변환")

oof_df["xgb_rank"] = to_rank_percentile(oof_df["xgb_pred"].values)
oof_df["cat_rank"] = to_rank_percentile(oof_df["cat_pred"].values)
oof_df["lgb_rank"] = to_rank_percentile(oof_df["lgb_pred"].values)

sub_df["xgb_rank"] = to_rank_percentile(sub_df["xgb_pred"].values)
sub_df["cat_rank"] = to_rank_percentile(sub_df["cat_pred"].values)
sub_df["lgb_rank"] = to_rank_percentile(sub_df["lgb_pred"].values)

rank_equal_pred = (
    oof_df["xgb_rank"].values
    + oof_df["cat_rank"].values
    + oof_df["lgb_rank"].values
) / 3.0
rank_equal_auc = roc_auc_score(y, rank_equal_pred)

print(f"rank equal ensemble AUC : {rank_equal_auc:.6f}")


# =========================================================
# 5. weighted rank ensemble grid search
# =========================================================
print_section("5. weighted rank ensemble grid search")

weight_candidates = generate_weight_candidates(step=SEARCH_STEP)
print(f"weight 조합 수: {len(weight_candidates)}")

rows = []
best_auc = -1.0
best_weights = None
best_oof_pred = None

xgb_rank = oof_df["xgb_rank"].values
cat_rank = oof_df["cat_rank"].values
lgb_rank = oof_df["lgb_rank"].values

for i, (wx, wc, wl) in enumerate(weight_candidates, start=1):
    pred = wx * xgb_rank + wc * cat_rank + wl * lgb_rank
    auc = roc_auc_score(y, pred)

    rows.append({
        "xgb_weight": wx,
        "cat_weight": wc,
        "lgb_weight": wl,
        "rank_auc": auc
    })

    if auc > best_auc:
        best_auc = auc
        best_weights = (wx, wc, wl)
        best_oof_pred = pred.copy()

    if i % 500 == 0 or i == len(weight_candidates):
        print(f"진행률: {i}/{len(weight_candidates)}")

search_result_df = pd.DataFrame(rows).sort_values(
    "rank_auc", ascending=False
).reset_index(drop=True)
search_result_df.to_csv(SEARCH_RESULT_PATH, index=False, encoding="utf-8-sig")

print("\nTop 10 weighted rank ensembles")
print(search_result_df.head(10).to_string(index=False))

print("\n[Best Weighted Rank Ensemble]")
print(f"xgb={best_weights[0]:.4f}, cat={best_weights[1]:.4f}, lgb={best_weights[2]:.4f}")
print(f"Best Rank OOF AUC: {best_auc:.6f}")
print(f"[저장] {SEARCH_RESULT_PATH}")


# =========================================================
# 6. best submission 생성
# =========================================================
print_section("6. best submission 생성")

wx, wc, wl = best_weights
best_sub_pred = (
    wx * sub_df["xgb_rank"].values
    + wc * sub_df["cat_rank"].values
    + wl * sub_df["lgb_rank"].values
)

rank_oof_path = os.path.join(
    OUTPUT_DIR,
    f"ensemble_v4_rank_xgb_lgbbase_catv7_best_xgb{int(round(wx * 100))}_cat{int(round(wc * 100))}_lgb{int(round(wl * 100))}_oof.csv"
)
rank_sub_path = os.path.join(
    OUTPUT_DIR,
    f"ensemble_v4_rank_xgb_lgbbase_catv7_best_xgb{int(round(wx * 100))}_cat{int(round(wc * 100))}_lgb{int(round(wl * 100))}_submission.csv"
)

rank_oof_df = pd.DataFrame({
    ID_COL: oof_df[ID_COL].values,
    TARGET_COL: y,
    "rank_ensemble_oof_pred": best_oof_pred
})
rank_oof_df.to_csv(rank_oof_path, index=False, encoding="utf-8-sig")

final_submission = sample_sub.copy()

if len(final_submission) != len(best_sub_pred):
    raise ValueError(
        f"sample_submission 행 수({len(final_submission)})와 예측값 길이({len(best_sub_pred)})가 다릅니다."
    )

final_submission[SUBMIT_TARGET_COL] = best_sub_pred
rank_submission_df = final_submission[[SUBMIT_ID_COL, SUBMIT_TARGET_COL]].copy()
rank_submission_df.to_csv(rank_sub_path, index=False, encoding="utf-8-sig")

print(f"[저장] {rank_oof_path}")
print(f"[저장] {rank_sub_path}")
print(rank_submission_df.head(10))
print("\n[제출 파일 컬럼 확인]")
print(rank_submission_df.columns.tolist())


# =========================================================
# 7. 결과 요약 / baseline 비교
# =========================================================
print_section("7. 결과 요약 / baseline 비교")

gain_vs_prob_baseline = best_auc - BASELINE_AUC
gain_vs_rank_equal = best_auc - rank_equal_auc
gain_vs_prob_equal = best_auc - prob_equal_auc
best_single_auc = max(xgb_auc, cat_auc, lgb_auc)
gain_vs_best_single = best_auc - best_single_auc

print(f"best single auc          : {best_single_auc:.6f}")
print(f"prob equal auc           : {prob_equal_auc:.6f}")
print(f"rank equal auc           : {rank_equal_auc:.6f}")
print(f"prob weighted baseline   : {BASELINE_AUC:.6f}")
print(f"best weighted rank auc   : {best_auc:.6f}")

print(f"\ngain vs prob baseline    : {gain_vs_prob_baseline:+.6f}")
print(f"gain vs rank equal       : {gain_vs_rank_equal:+.6f}")
print(f"gain vs prob equal       : {gain_vs_prob_equal:+.6f}")
print(f"gain vs best single      : {gain_vs_best_single:+.6f}")

if best_auc > BASELINE_AUC:
    print("✓ probability weighted baseline 초과")
else:
    print("✗ probability weighted baseline 미달")


# =========================================================
# 8. 결과 로그 저장
# =========================================================
print_section("8. 결과 로그 저장")

log_row = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "experiment_name": EXPERIMENT_NAME,
    "baseline_name": BASELINE_NAME,
    "baseline_auc": round(float(BASELINE_AUC), 6),
    "baseline_weights": BASELINE_WEIGHTS,
    "xgb_auc": round(float(xgb_auc), 6),
    "cat_auc": round(float(cat_auc), 6),
    "lgb_auc": round(float(lgb_auc), 6),
    "best_single_auc": round(float(best_single_auc), 6),
    "prob_equal_auc": round(float(prob_equal_auc), 6),
    "rank_equal_auc": round(float(rank_equal_auc), 6),
    "best_rank_auc": round(float(best_auc), 6),
    "best_xgb_weight": round(float(best_weights[0]), 4),
    "best_cat_weight": round(float(best_weights[1]), 4),
    "best_lgb_weight": round(float(best_weights[2]), 4),
    "gain_vs_prob_baseline": round(float(gain_vs_prob_baseline), 6),
    "gain_vs_rank_equal": round(float(gain_vs_rank_equal), 6),
    "gain_vs_prob_equal": round(float(gain_vs_prob_equal), 6),
    "gain_vs_best_single": round(float(gain_vs_best_single), 6),
    "search_step": SEARCH_STEP,
    "submit_target_col": SUBMIT_TARGET_COL,
    "xgb_oof_path": XGB_OOF_PATH,
    "cat_oof_path": CAT_OOF_PATH,
    "lgb_oof_path": LGB_OOF_PATH,
}

result_log_df = pd.DataFrame([log_row])

if os.path.exists(RESULT_LOG_PATH):
    old = pd.read_csv(RESULT_LOG_PATH)
    result_log_df = pd.concat([old, result_log_df], ignore_index=True)

result_log_df.to_csv(RESULT_LOG_PATH, index=False, encoding="utf-8-sig")
print(f"[저장] {RESULT_LOG_PATH}")
print(pd.DataFrame([log_row]).to_string(index=False))


# =========================================================
# 9. 완료
# =========================================================
print_section("9. 완료")

print(f"실험명                   : {EXPERIMENT_NAME}")
print(f"prob weighted baseline   : {BASELINE_AUC:.6f}")
print(f"best rank ensemble auc   : {best_auc:.6f}")
print(f"best rank weights        : xgb={best_weights[0]:.4f}, cat={best_weights[1]:.4f}, lgb={best_weights[2]:.4f}")
print(f"gain vs baseline         : {gain_vs_prob_baseline:+.6f}")
print(f"submit columns           : [{SUBMIT_ID_COL}, {SUBMIT_TARGET_COL}]")

print("\n저장 파일:")
print(f"  1. {SEARCH_RESULT_PATH}")
print(f"  2. {rank_oof_path}")
print(f"  3. {rank_sub_path}")
print(f"  4. {RESULT_LOG_PATH}")

# 9. 완료
# ====================================================================================================    
# 실험명                   : ensemble_v4_rank_xgb_lgbbase_catv7
# prob weighted baseline   : 0.740596
# best rank ensemble auc   : 0.740598
# best rank weights        : xgb=0.3500, cat=0.5000, lgb=0.1500
# gain vs baseline         : +0.000001
# submit columns           : [ID, probability]