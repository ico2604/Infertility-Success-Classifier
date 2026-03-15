# =========================================================
# Ensemble v2 — XGB + CAT + LGBM
# ---------------------------------------------------------
# 목적
# 1) XGBoost / CatBoost / LightGBM의 OOF 예측값을 불러온다.
# 2) 단일 모델 성능과 모델 간 예측 상관계수를 비교한다.
# 3) 2모델 / 3모델 Weighted Ensemble 성능을 비교한다.
# 4) 3모델 grid search 기반 최적 가중치를 찾는다.
# 5) AUC 비율 기반 reference weight도 함께 계산한다.
# 6) equal / ref / best-grid submission을 모두 저장한다.
# 7) 앙상블 실험 결과를 로그 파일로 누적 저장한다.
#
# 실행 방법
#   python yysop/src/ensemble_v2.py
# =========================================================

import os
import warnings
warnings.filterwarnings("ignore")

from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


# =========================================================
# 0. 경로 / 기본 설정
# =========================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_COL = "임신 성공 여부"
ID_COL = "ID"
OOF_PRED_COL = "oof_pred_prob"

# ---------------------------------------------------------
# 입력 파일 경로
# ---------------------------------------------------------
# XGB_OOF_PATH = os.path.join(OUTPUT_DIR, "xgb_v2_reg_relax_oof_predictions.csv")
# XGB_SUB_PATH = os.path.join(OUTPUT_DIR, "xgb_v2_reg_relax_submission.csv")

# XGB (Optuna)
XGB_OOF_PATH = os.path.join(OUTPUT_DIR, "xgb_optuna_v1_oof_predictions.csv")
XGB_SUB_PATH = os.path.join(OUTPUT_DIR, "xgb_optuna_v1_submission.csv")

CAT_OOF_PATH = os.path.join(OUTPUT_DIR, "catboost_v2_catboost_baseline_v2_oof_predictions.csv")
CAT_SUB_PATH = os.path.join(OUTPUT_DIR, "catboost_v2_catboost_baseline_v2_submission.csv")

LGB_OOF_PATH = os.path.join(OUTPUT_DIR, "lightgbm_v1_lightgbm_baseline_v1_oof_predictions.csv")
LGB_SUB_PATH = os.path.join(OUTPUT_DIR, "lightgbm_v1_lightgbm_baseline_v1_submission.csv")

# ---------------------------------------------------------
# 3모델 grid search 설정
# - 과적합 방지를 위해 너무 촘촘하지 않게 0.05 유지
# ---------------------------------------------------------
GRID_STEP = 0.05

# ---------------------------------------------------------
# 현재 직접 지정해서 보고 싶은 가중치 (선택)
# 합은 1.0이어야 함
# ---------------------------------------------------------
MANUAL_XGB_W = 0.40
MANUAL_CAT_W = 0.40
MANUAL_LGB_W = 0.20

assert abs(MANUAL_XGB_W + MANUAL_CAT_W + MANUAL_LGB_W - 1.0) < 1e-8, "수동 가중치 합이 1.0이어야 합니다."


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


def weighted_sum(weights, *arrays):
    out = np.zeros(len(arrays[0]), dtype=float)
    for w, arr in zip(weights, arrays):
        out += w * arr
    return out


def format_weight_name(prefix: str, wx: float, wc: float, wl: float):
    return f"{prefix}_xgb{int(round(wx * 100)):02d}_cat{int(round(wc * 100)):02d}_lgb{int(round(wl * 100)):02d}"


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
        print(f"{name:<24}: {auc:.6f}")
    return auc


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

xgb_oof = load_and_sort_csv(XGB_OOF_PATH, ID_COL)
cat_oof = load_and_sort_csv(CAT_OOF_PATH, ID_COL)
lgb_oof = load_and_sort_csv(LGB_OOF_PATH, ID_COL)

required_oof_cols = [ID_COL, TARGET_COL, OOF_PRED_COL]
for col in required_oof_cols:
    assert col in xgb_oof.columns, f"XGB OOF 파일에 '{col}' 컬럼이 없습니다."
    assert col in cat_oof.columns, f"CAT OOF 파일에 '{col}' 컬럼이 없습니다."
    assert col in lgb_oof.columns, f"LGB OOF 파일에 '{col}' 컬럼이 없습니다."

assert (xgb_oof[ID_COL].values == cat_oof[ID_COL].values).all(), "XGB와 CAT OOF의 ID가 다릅니다."
assert (xgb_oof[ID_COL].values == lgb_oof[ID_COL].values).all(), "XGB와 LGB OOF의 ID가 다릅니다."

assert (xgb_oof[TARGET_COL].values == cat_oof[TARGET_COL].values).all(), "XGB와 CAT OOF의 타깃이 다릅니다."
assert (xgb_oof[TARGET_COL].values == lgb_oof[TARGET_COL].values).all(), "XGB와 LGB OOF의 타깃이 다릅니다."

y_true = xgb_oof[TARGET_COL].values

xgb_pred = xgb_oof[OOF_PRED_COL].values
cat_pred = cat_oof[OOF_PRED_COL].values
lgb_pred = lgb_oof[OOF_PRED_COL].values

print(f"XGB OOF shape : {xgb_oof.shape}")
print(f"CAT OOF shape : {cat_oof.shape}")
print(f"LGB OOF shape : {lgb_oof.shape}")


# =========================================================
# 4. 단일 모델 성능 비교
# =========================================================
print_section("4. 단일 모델 OOF AUC 비교")

xgb_auc = evaluate_auc(y_true, xgb_pred, "XGBoost")
cat_auc = evaluate_auc(y_true, cat_pred, "CatBoost")
lgb_auc = evaluate_auc(y_true, lgb_pred, "LightGBM")


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
# 6. 2모델 앙상블 비교
# =========================================================
print_section("6. 2모델 앙상블 비교")

pred_xgb_cat_50 = 0.50 * xgb_pred + 0.50 * cat_pred
pred_xgb_lgb_50 = 0.50 * xgb_pred + 0.50 * lgb_pred
pred_cat_lgb_50 = 0.50 * cat_pred + 0.50 * lgb_pred

auc_xgb_cat_50 = evaluate_auc(y_true, pred_xgb_cat_50, "XGB+CAT (0.5/0.5)")
auc_xgb_lgb_50 = evaluate_auc(y_true, pred_xgb_lgb_50, "XGB+LGB (0.5/0.5)")
auc_cat_lgb_50 = evaluate_auc(y_true, pred_cat_lgb_50, "CAT+LGB (0.5/0.5)")


# =========================================================
# 7. 3모델 기준 가중치 준비
# =========================================================
print_section("7. 3모델 기준 가중치 준비")

# 1) equal weight
equal_w = (1/3, 1/3, 1/3)

# 2) AUC 비율 기반 reference weight
total_auc = xgb_auc + cat_auc + lgb_auc
ref_w = (
    xgb_auc / total_auc,
    cat_auc / total_auc,
    lgb_auc / total_auc
)

# 3) manual weight
manual_w = (MANUAL_XGB_W, MANUAL_CAT_W, MANUAL_LGB_W)

print(f"Equal weight   : XGB {equal_w[0]:.4f} / CAT {equal_w[1]:.4f} / LGB {equal_w[2]:.4f}")
print(f"Ref weight     : XGB {ref_w[0]:.4f} / CAT {ref_w[1]:.4f} / LGB {ref_w[2]:.4f}")
print(f"Manual weight  : XGB {manual_w[0]:.4f} / CAT {manual_w[1]:.4f} / LGB {manual_w[2]:.4f}")


# =========================================================
# 8. 3모델 앙상블 성능 비교
# =========================================================
print_section("8. 3모델 앙상블 성능 비교")

pred_equal = weighted_sum(equal_w, xgb_pred, cat_pred, lgb_pred)
pred_ref = weighted_sum(ref_w, xgb_pred, cat_pred, lgb_pred)
pred_manual = weighted_sum(manual_w, xgb_pred, cat_pred, lgb_pred)

equal_auc = evaluate_auc(y_true, pred_equal, "3-model equal")
ref_auc = evaluate_auc(y_true, pred_ref, "3-model ref(AUC)")
manual_auc = evaluate_auc(y_true, pred_manual, "3-model manual")


# =========================================================
# 9. 3모델 grid search
# =========================================================
print_section("9. 3모델 grid search")

print(f"Grid step: {GRID_STEP:.2f}")

best_auc = -1.0
best_w = None

grid_values = np.arange(0.0, 1.0 + GRID_STEP / 2, GRID_STEP)
grid_values = np.round(grid_values, 4)

for wx in grid_values:
    for wc in grid_values:
        wl = 1.0 - wx - wc
        if wl < 0 or wl > 1:
            continue

        wl = round(float(wl), 4)
        pred = wx * xgb_pred + wc * cat_pred + wl * lgb_pred
        auc = roc_auc_score(y_true, pred)

        if auc > best_auc:
            best_auc = auc
            best_w = (float(wx), float(wc), float(wl))

best_xgb_w, best_cat_w, best_lgb_w = best_w

print(f"Best grid AUC  : {best_auc:.6f}")
print(f"Best weights   : XGB {best_xgb_w:.4f} / CAT {best_cat_w:.4f} / LGB {best_lgb_w:.4f}")

grid_vs_ref_gap = best_auc - ref_auc
print(f"\nGrid vs Ref gap: {grid_vs_ref_gap:+.6f}")

if grid_vs_ref_gap >= 0.001:
    print("⚠ Grid 결과가 Ref 대비 너무 많이 높습니다. OOF 과적합 가능성을 의심하세요.")
else:
    print("✓ Grid와 Ref 차이가 크지 않습니다. 비교적 자연스러운 범위입니다.")

pred_best = weighted_sum(best_w, xgb_pred, cat_pred, lgb_pred)
best_confirm_auc = roc_auc_score(y_true, pred_best)

print(f"Best confirm AUC: {best_confirm_auc:.6f}")


# =========================================================
# 10. 앙상블 효과 요약
# =========================================================
print_section("10. 앙상블 효과 요약")

summary_rows = [
    ("XGBoost", xgb_auc),
    ("CatBoost", cat_auc),
    ("LightGBM", lgb_auc),
    ("XGB+CAT (0.5/0.5)", auc_xgb_cat_50),
    ("XGB+LGB (0.5/0.5)", auc_xgb_lgb_50),
    ("CAT+LGB (0.5/0.5)", auc_cat_lgb_50),
    ("3-model equal", equal_auc),
    ("3-model ref", ref_auc),
    ("3-model manual", manual_auc),
    ("3-model best-grid", best_auc),
]

summary_df = pd.DataFrame(summary_rows, columns=["ensemble", "auc"])
summary_df["vs_best_single"] = summary_df["auc"] - max(xgb_auc, cat_auc, lgb_auc)
print(summary_df.to_string(index=False))


# =========================================================
# 11. OOF 저장
# =========================================================
print_section("11. 앙상블 OOF 저장")

equal_name = format_weight_name("ensemble_v2_equal", *equal_w)
ref_name = format_weight_name("ensemble_v2_ref", *ref_w)
manual_name = format_weight_name("ensemble_v2_manual", *manual_w)
best_name = format_weight_name("ensemble_v2_bestgrid", *best_w)

oof_save_specs = [
    (equal_name, pred_equal),
    (ref_name, pred_ref),
    (manual_name, pred_manual),
    (best_name, pred_best),
]

saved_oof_paths = []

for name, pred in oof_save_specs:
    oof_df = pd.DataFrame({
        ID_COL: xgb_oof[ID_COL].values,
        TARGET_COL: y_true,
        "xgb_pred_prob": xgb_pred,
        "cat_pred_prob": cat_pred,
        "lgb_pred_prob": lgb_pred,
        "oof_pred_prob": pred,
        "oof_pred_label": (pred >= 0.5).astype(int),
    })
    oof_df["correct"] = (oof_df[TARGET_COL] == oof_df["oof_pred_label"]).astype(int)

    path = os.path.join(OUTPUT_DIR, f"{name}_oof.csv")
    oof_df.to_csv(path, index=False, encoding="utf-8-sig")
    saved_oof_paths.append(path)
    print(f"[저장] {path}")


# =========================================================
# 12. submission 로드 및 정렬
# =========================================================
print_section("12. submission 로드 및 정렬")

xgb_sub = load_and_sort_csv(XGB_SUB_PATH, ID_COL)
cat_sub = load_and_sort_csv(CAT_SUB_PATH, ID_COL)
lgb_sub = load_and_sort_csv(LGB_SUB_PATH, ID_COL)

required_sub_cols = [ID_COL, TARGET_COL]
for col in required_sub_cols:
    assert col in xgb_sub.columns, f"XGB submission 파일에 '{col}' 컬럼이 없습니다."
    assert col in cat_sub.columns, f"CAT submission 파일에 '{col}' 컬럼이 없습니다."
    assert col in lgb_sub.columns, f"LGB submission 파일에 '{col}' 컬럼이 없습니다."

assert (xgb_sub[ID_COL].values == cat_sub[ID_COL].values).all(), "XGB와 CAT submission의 ID가 다릅니다."
assert (xgb_sub[ID_COL].values == lgb_sub[ID_COL].values).all(), "XGB와 LGB submission의 ID가 다릅니다."

print("submission ID 정렬 및 일치 확인 완료")

xgb_test = xgb_sub[TARGET_COL].values
cat_test = cat_sub[TARGET_COL].values
lgb_test = lgb_sub[TARGET_COL].values


# =========================================================
# 13. submission 생성
# =========================================================
print_section("13. submission 생성")

sub_save_specs = [
    (equal_name, equal_w),
    (ref_name, ref_w),
    (manual_name, manual_w),
    (best_name, best_w),
]

saved_sub_paths = []

for name, weights in sub_save_specs:
    pred_test = weighted_sum(weights, xgb_test, cat_test, lgb_test)

    submission = pd.DataFrame({
        ID_COL: xgb_sub[ID_COL].values,
        TARGET_COL: pred_test,
    })

    path = os.path.join(OUTPUT_DIR, f"{name}_submission.csv")
    submission.to_csv(path, index=False, encoding="utf-8-sig")
    saved_sub_paths.append(path)
    print(f"[저장] {path}")
    print(submission.head(3))


# =========================================================
# 14. 로그 저장
# =========================================================
print_section("14. 앙상블 실험 로그 저장")

log_path = os.path.join(OUTPUT_DIR, "ensemble_v2_results_log.csv")

log_row = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

    "xgb_auc": round(float(xgb_auc), 6),
    "cat_auc": round(float(cat_auc), 6),
    "lgb_auc": round(float(lgb_auc), 6),

    "corr_xgb_cat": round(float(corr_xgb_cat), 6),
    "corr_xgb_lgb": round(float(corr_xgb_lgb), 6),
    "corr_cat_lgb": round(float(corr_cat_lgb), 6),

    "auc_xgb_cat_50": round(float(auc_xgb_cat_50), 6),
    "auc_xgb_lgb_50": round(float(auc_xgb_lgb_50), 6),
    "auc_cat_lgb_50": round(float(auc_cat_lgb_50), 6),

    "equal_auc": round(float(equal_auc), 6),
    "ref_auc": round(float(ref_auc), 6),
    "manual_auc": round(float(manual_auc), 6),
    "best_auc": round(float(best_auc), 6),

    "equal_xgb_w": round(float(equal_w[0]), 6),
    "equal_cat_w": round(float(equal_w[1]), 6),
    "equal_lgb_w": round(float(equal_w[2]), 6),

    "ref_xgb_w": round(float(ref_w[0]), 6),
    "ref_cat_w": round(float(ref_w[1]), 6),
    "ref_lgb_w": round(float(ref_w[2]), 6),

    "manual_xgb_w": round(float(manual_w[0]), 6),
    "manual_cat_w": round(float(manual_w[1]), 6),
    "manual_lgb_w": round(float(manual_w[2]), 6),

    "best_xgb_w": round(float(best_w[0]), 6),
    "best_cat_w": round(float(best_w[1]), 6),
    "best_lgb_w": round(float(best_w[2]), 6),

    "grid_step": GRID_STEP,
    "grid_vs_ref_gap": round(float(grid_vs_ref_gap), 6),

    "best_vs_best_single": round(float(best_auc - max(xgb_auc, cat_auc, lgb_auc)), 6),
    "ref_vs_best_single": round(float(ref_auc - max(xgb_auc, cat_auc, lgb_auc)), 6),
    "equal_vs_best_single": round(float(equal_auc - max(xgb_auc, cat_auc, lgb_auc)), 6),

    "xgb_oof_file": os.path.basename(XGB_OOF_PATH),
    "cat_oof_file": os.path.basename(CAT_OOF_PATH),
    "lgb_oof_file": os.path.basename(LGB_OOF_PATH),

    "xgb_sub_file": os.path.basename(XGB_SUB_PATH),
    "cat_sub_file": os.path.basename(CAT_SUB_PATH),
    "lgb_sub_file": os.path.basename(LGB_SUB_PATH),
}

save_log_row(log_path, log_row)
print(f"[저장] {log_path}")

log_preview = pd.read_csv(log_path)
show_cols = [
    "timestamp",
    "xgb_auc", "cat_auc", "lgb_auc",
    "equal_auc", "ref_auc", "best_auc",
    "best_xgb_w", "best_cat_w", "best_lgb_w",
    "grid_vs_ref_gap"
]
print(log_preview[show_cols].tail(10).to_string(index=False))


# =========================================================
# 15. 종료 요약
# =========================================================
print_section("15. 완료")

print("단일 모델")
print(f"  XGB : {xgb_auc:.6f}")
print(f"  CAT : {cat_auc:.6f}")
print(f"  LGB : {lgb_auc:.6f}")

print("\n2모델")
print(f"  XGB+CAT (0.5/0.5) : {auc_xgb_cat_50:.6f}")
print(f"  XGB+LGB (0.5/0.5) : {auc_xgb_lgb_50:.6f}")
print(f"  CAT+LGB (0.5/0.5) : {auc_cat_lgb_50:.6f}")

print("\n3모델")
print(f"  Equal   : {equal_auc:.6f}")
print(f"  Ref     : {ref_auc:.6f}")
print(f"  Manual  : {manual_auc:.6f}")
print(f"  Best    : {best_auc:.6f}")

print("\n가중치")
print(f"  Equal   : XGB {equal_w[0]:.4f} / CAT {equal_w[1]:.4f} / LGB {equal_w[2]:.4f}")
print(f"  Ref     : XGB {ref_w[0]:.4f} / CAT {ref_w[1]:.4f} / LGB {ref_w[2]:.4f}")
print(f"  Manual  : XGB {manual_w[0]:.4f} / CAT {manual_w[1]:.4f} / LGB {manual_w[2]:.4f}")
print(f"  Best    : XGB {best_w[0]:.4f} / CAT {best_w[1]:.4f} / LGB {best_w[2]:.4f}")

print("\n예측 상관계수")
print(f"  XGB-CAT : {corr_xgb_cat:.6f}")
print(f"  XGB-LGB : {corr_xgb_lgb:.6f}")
print(f"  CAT-LGB : {corr_cat_lgb:.6f}")

print("\n저장된 OOF 파일")
for p in saved_oof_paths:
    print(f"  - {p}")

print("\n저장된 submission 파일")
for p in saved_sub_paths:
    print(f"  - {p}")

print(f"\n로그 파일")
print(f"  - {log_path}")

print("\n제출 추천 순서")
print("  1. ref submission")
print("  2. best-grid submission")
print("  3. equal submission")
print("  4. manual submission")