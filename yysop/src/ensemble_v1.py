# =========================================================



# ---------------------------------------------------------
# 목적
# 1) XGBoost OOF / submission 예측값을 불러온다.
# 2) CatBoost OOF / submission 예측값을 불러온다.
# 3) 두 모델의 성능과 예측 상관을 비교한다.
# 4) 가중 평균(Weighted Average) 앙상블로 최종 예측을 생성한다.
# 5) 현재 설정 가중치와 최적 grid 가중치 기준 제출 파일을 저장한다.
# 6) 앙상블 결과를 로그 파일로 누적 저장한다.
#
# 앙상블 방식
# - Weighted Average:
#   w_xgb * xgb_pred + w_cat * cat_pred
#
# 실행 방법
#   python yysop/src/ensemble_v1.py
#
# 주의
# - XGBoost와 CatBoost OOF / submission 파일이 먼저 생성되어 있어야 함
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
# 사용할 파일 경로
# - XGB는 현재 최고 성능인 reg_relax 기준
# - CatBoost는 baseline v2 기준
# ---------------------------------------------------------
XGB_OOF_PATH = os.path.join(OUTPUT_DIR, "xgb_v2_reg_relax_oof_predictions.csv")
XGB_SUB_PATH = os.path.join(OUTPUT_DIR, "xgb_v2_reg_relax_submission.csv")

CAT_OOF_PATH = os.path.join(OUTPUT_DIR, "catboost_v2_catboost_baseline_v2_oof_predictions.csv")
CAT_SUB_PATH = os.path.join(OUTPUT_DIR, "catboost_v2_catboost_baseline_v2_submission.csv")

# ---------------------------------------------------------
# 현재 설정 가중치
# - 처음엔 0.5 / 0.5부터
# - 이후 best grid 결과 보고 수정 가능
# ---------------------------------------------------------
XGB_WEIGHT = 0.50
CAT_WEIGHT = 0.50

assert abs(XGB_WEIGHT + CAT_WEIGHT - 1.0) < 1e-8, "가중치 합이 1.0이 되어야 합니다."

# 가중치 grid 설정
GRID_START = 0.00
GRID_END = 1.00
GRID_STEP = 0.02

# 결과 이름
CURRENT_ENSEMBLE_NAME = f"xgb{int(round(XGB_WEIGHT * 100)):02d}_cat{int(round(CAT_WEIGHT * 100)):02d}"


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


# =========================================================
# 2. 파일 존재 여부 확인
# =========================================================
print_section("2. 입력 파일 확인")

for path in [XGB_OOF_PATH, XGB_SUB_PATH, CAT_OOF_PATH, CAT_SUB_PATH]:
    check_file_exists(path)
    print(f"[OK] {path}")


# =========================================================
# 3. OOF 예측값 로드
# =========================================================
print_section("3. OOF 예측값 로드")

xgb_oof = load_and_sort_csv(XGB_OOF_PATH, ID_COL)
cat_oof = load_and_sort_csv(CAT_OOF_PATH, ID_COL)

print(f"XGB OOF shape : {xgb_oof.shape}")
print(f"CAT OOF shape : {cat_oof.shape}")

required_oof_cols = [ID_COL, TARGET_COL, OOF_PRED_COL]
for col in required_oof_cols:
    assert col in xgb_oof.columns, f"XGB OOF 파일에 '{col}' 컬럼이 없습니다."
    assert col in cat_oof.columns, f"CAT OOF 파일에 '{col}' 컬럼이 없습니다."

assert (xgb_oof[ID_COL].values == cat_oof[ID_COL].values).all(), \
    "XGB와 CatBoost OOF의 ID 순서가 다릅니다."

assert (xgb_oof[TARGET_COL].values == cat_oof[TARGET_COL].values).all(), \
    "XGB와 CatBoost OOF의 타깃 값이 다릅니다."

y_true = xgb_oof[TARGET_COL].values
xgb_oof_pred = xgb_oof[OOF_PRED_COL].values
cat_oof_pred = cat_oof[OOF_PRED_COL].values


# =========================================================
# 4. 단일 모델 OOF AUC 비교
# =========================================================
print_section("4. 단일 모델 OOF AUC 비교")

xgb_auc = roc_auc_score(y_true, xgb_oof_pred)
cat_auc = roc_auc_score(y_true, cat_oof_pred)

print(f"\n{'모델':>12}  {'OOF AUC':>10}")
print("-" * 26)
print(f"{'XGBoost':>12}  {xgb_auc:>10.6f}")
print(f"{'CatBoost':>12}  {cat_auc:>10.6f}")
print("-" * 26)
print(f"{'차이':>12}  {abs(xgb_auc - cat_auc):>10.6f}")

better_model = "XGBoost" if xgb_auc >= cat_auc else "CatBoost"
print(f"\n현재 단일 모델 기준 우수 모델: {better_model}")

# 참고용 비율
total_auc = xgb_auc + cat_auc
ref_xgb = round(xgb_auc / total_auc, 3)
ref_cat = round(cat_auc / total_auc, 3)
print(f"\n[참고용 비율] XGB: {ref_xgb:.3f} / CatBoost: {ref_cat:.3f}")
print("실제 최적 가중치는 아래 OOF grid 탐색 결과를 우선 참고하세요.")


# =========================================================
# 5. 예측 상관계수 확인
# =========================================================
print_section("5. 예측 상관계수 확인")

pred_corr = np.corrcoef(xgb_oof_pred, cat_oof_pred)[0, 1]
print(f"OOF prediction correlation: {pred_corr:.6f}")

if pred_corr >= 0.99:
    print("해석: 두 모델 예측이 매우 유사합니다. 앙상블 이득이 작을 수 있습니다.")
elif pred_corr >= 0.97:
    print("해석: 예측이 꽤 유사하지만, 소폭의 앙상블 개선은 기대할 수 있습니다.")
else:
    print("해석: 예측 다양성이 비교적 있어 앙상블 이득 가능성이 있습니다.")


# =========================================================
# 6. 앙상블 OOF AUC 계산
# =========================================================
print_section("6. 앙상블 OOF AUC")

print(f"\n{'XGB 가중치':>12}  {'CAT 가중치':>12}  {'앙상블 AUC':>12}")
print("-" * 42)

weight_grid = np.arange(GRID_START, GRID_END + GRID_STEP / 2, GRID_STEP)
weight_grid = np.round(weight_grid, 2)

best_weight_auc = -1.0
best_xgb_w = None
best_cat_w = None

for w in weight_grid:
    ensemble_pred = w * xgb_oof_pred + (1 - w) * cat_oof_pred
    ensemble_auc = roc_auc_score(y_true, ensemble_pred)

    marker = "  ← 현재 설정" if abs(w - XGB_WEIGHT) < 1e-12 else ""
    if ensemble_auc > best_weight_auc:
        best_weight_auc = ensemble_auc
        best_xgb_w = float(w)
        best_cat_w = float(round(1 - w, 2))

    print(f"{w:>12.2f}  {1-w:>12.2f}  {ensemble_auc:>12.6f}{marker}")

print("-" * 42)
print(f"\n최적 가중치 (grid): XGB {best_xgb_w:.2f} / CAT {best_cat_w:.2f}  →  AUC {best_weight_auc:.6f}")

# 현재 설정 가중치 앙상블
oof_ensemble_current = XGB_WEIGHT * xgb_oof_pred + CAT_WEIGHT * cat_oof_pred
oof_ensemble_auc_current = roc_auc_score(y_true, oof_ensemble_current)

# 최적 grid 가중치 앙상블
oof_ensemble_best = best_xgb_w * xgb_oof_pred + best_cat_w * cat_oof_pred
oof_ensemble_auc_best = roc_auc_score(y_true, oof_ensemble_best)

print(f"\n[현재 설정 앙상블 AUC] {oof_ensemble_auc_current:.6f}")
print(f"  단일 XGB 대비: {oof_ensemble_auc_current - xgb_auc:+.6f}")
print(f"  단일 CAT 대비: {oof_ensemble_auc_current - cat_auc:+.6f}")

print(f"\n[최적 grid 앙상블 AUC] {oof_ensemble_auc_best:.6f}")
print(f"  단일 XGB 대비: {oof_ensemble_auc_best - xgb_auc:+.6f}")
print(f"  단일 CAT 대비: {oof_ensemble_auc_best - cat_auc:+.6f}")


# =========================================================
# 7. 앙상블 효과 요약
# =========================================================
print_section("7. 앙상블 효과 요약")

print(f"\n{'':>20}  {'AUC':>10}  {'vs XGB':>10}  {'vs CAT':>10}")
print("-" * 56)
print(f"{'XGBoost (단독)':>20}  {xgb_auc:>10.6f}  {'—':>10}  {xgb_auc - cat_auc:>+10.6f}")
print(f"{'CatBoost (단독)':>20}  {cat_auc:>10.6f}  {cat_auc - xgb_auc:>+10.6f}  {'—':>10}")
print(f"{'앙상블 (현재 설정)':>20}  {oof_ensemble_auc_current:>10.6f}  {oof_ensemble_auc_current - xgb_auc:>+10.6f}  {oof_ensemble_auc_current - cat_auc:>+10.6f}")
print(f"{'앙상블 (최적 grid)':>20}  {oof_ensemble_auc_best:>10.6f}  {oof_ensemble_auc_best - xgb_auc:>+10.6f}  {oof_ensemble_auc_best - cat_auc:>+10.6f}")

if oof_ensemble_auc_best > max(xgb_auc, cat_auc):
    print("\n✓ 앙상블이 단일 모델보다 AUC가 높습니다.")
else:
    print("\n⚠ 앙상블 이득이 거의 없거나 없습니다.")
    print("  → 다른 가중치, CatBoost 추가 튜닝, LightGBM 추가를 검토하세요.")


# =========================================================
# 8. 앙상블 OOF 저장
# =========================================================
print_section("8. 앙상블 OOF 저장")

oof_current_df = pd.DataFrame({
    ID_COL: xgb_oof[ID_COL].values,
    TARGET_COL: y_true,
    "xgb_pred_prob": xgb_oof_pred,
    "cat_pred_prob": cat_oof_pred,
    "oof_pred_prob": oof_ensemble_current,
    "oof_pred_label": (oof_ensemble_current >= 0.5).astype(int),
})
oof_current_df["correct"] = (oof_current_df[TARGET_COL] == oof_current_df["oof_pred_label"]).astype(int)

current_oof_path = os.path.join(OUTPUT_DIR, f"ensemble_v1_{CURRENT_ENSEMBLE_NAME}_oof.csv")
oof_current_df.to_csv(current_oof_path, index=False, encoding="utf-8-sig")
print(f"[저장] 현재 설정 OOF -> {current_oof_path}")

BEST_ENSEMBLE_NAME = f"xgb{int(round(best_xgb_w * 100)):02d}_cat{int(round(best_cat_w * 100)):02d}"

oof_best_df = pd.DataFrame({
    ID_COL: xgb_oof[ID_COL].values,
    TARGET_COL: y_true,
    "xgb_pred_prob": xgb_oof_pred,
    "cat_pred_prob": cat_oof_pred,
    "oof_pred_prob": oof_ensemble_best,
    "oof_pred_label": (oof_ensemble_best >= 0.5).astype(int),
})
oof_best_df["correct"] = (oof_best_df[TARGET_COL] == oof_best_df["oof_pred_label"]).astype(int)

best_oof_path = os.path.join(OUTPUT_DIR, f"ensemble_v1_{BEST_ENSEMBLE_NAME}_bestgrid_oof.csv")
oof_best_df.to_csv(best_oof_path, index=False, encoding="utf-8-sig")
print(f"[저장] 최적 grid OOF -> {best_oof_path}")


# =========================================================
# 9. 제출 파일 생성
# =========================================================
print_section("9. 제출 파일 생성")

xgb_sub = load_and_sort_csv(XGB_SUB_PATH, ID_COL)
cat_sub = load_and_sort_csv(CAT_SUB_PATH, ID_COL)

required_sub_cols = [ID_COL, TARGET_COL]
for col in required_sub_cols:
    assert col in xgb_sub.columns, f"XGB submission 파일에 '{col}' 컬럼이 없습니다."
    assert col in cat_sub.columns, f"CAT submission 파일에 '{col}' 컬럼이 없습니다."

assert (xgb_sub[ID_COL].values == cat_sub[ID_COL].values).all(), \
    "XGB와 CatBoost submission의 ID 순서가 다릅니다."

xgb_test_pred = xgb_sub[TARGET_COL].values
cat_test_pred = cat_sub[TARGET_COL].values

# 현재 설정 가중치
ensemble_test_current = XGB_WEIGHT * xgb_test_pred + CAT_WEIGHT * cat_test_pred
submission_current = pd.DataFrame({
    ID_COL: xgb_sub[ID_COL].values,
    TARGET_COL: ensemble_test_current,
})
current_sub_path = os.path.join(OUTPUT_DIR, f"ensemble_v1_{CURRENT_ENSEMBLE_NAME}_submission.csv")
submission_current.to_csv(current_sub_path, index=False, encoding="utf-8-sig")
print(f"[저장] 현재 설정 submission -> {current_sub_path}")
print(submission_current.head(10))

# 최적 grid 가중치
ensemble_test_best = best_xgb_w * xgb_test_pred + best_cat_w * cat_test_pred
submission_best = pd.DataFrame({
    ID_COL: xgb_sub[ID_COL].values,
    TARGET_COL: ensemble_test_best,
})
best_sub_path = os.path.join(OUTPUT_DIR, f"ensemble_v1_{BEST_ENSEMBLE_NAME}_bestgrid_submission.csv")
submission_best.to_csv(best_sub_path, index=False, encoding="utf-8-sig")
print(f"[저장] 최적 grid submission -> {best_sub_path}")
print(submission_best.head(10))


# =========================================================
# 10. 앙상블 실험 로그 저장
# =========================================================
print_section("10. 앙상블 실험 로그 저장")

log_path = os.path.join(OUTPUT_DIR, "ensemble_v1_results_log.csv")

log_row = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "current_ensemble_name": CURRENT_ENSEMBLE_NAME,
    "best_ensemble_name": BEST_ENSEMBLE_NAME,
    "xgb_weight_current": XGB_WEIGHT,
    "cat_weight_current": CAT_WEIGHT,
    "xgb_oof_auc": round(float(xgb_auc), 6),
    "cat_oof_auc": round(float(cat_auc), 6),
    "pred_corr": round(float(pred_corr), 6),
    "ensemble_oof_auc_current": round(float(oof_ensemble_auc_current), 6),
    "ensemble_oof_auc_bestgrid": round(float(oof_ensemble_auc_best), 6),
    "best_xgb_weight": round(float(best_xgb_w), 2),
    "best_cat_weight": round(float(best_cat_w), 2),
    "gain_current_vs_xgb": round(float(oof_ensemble_auc_current - xgb_auc), 6),
    "gain_current_vs_cat": round(float(oof_ensemble_auc_current - cat_auc), 6),
    "gain_best_vs_xgb": round(float(oof_ensemble_auc_best - xgb_auc), 6),
    "gain_best_vs_cat": round(float(oof_ensemble_auc_best - cat_auc), 6),
    "xgb_oof_file": os.path.basename(XGB_OOF_PATH),
    "cat_oof_file": os.path.basename(CAT_OOF_PATH),
    "xgb_sub_file": os.path.basename(XGB_SUB_PATH),
    "cat_sub_file": os.path.basename(CAT_SUB_PATH),
}

save_log_row(log_path, log_row)
print(f"[저장] {log_path}")

log_preview = pd.read_csv(log_path)
print("\n[앙상블 실험 누적 요약]")
print(log_preview[[
    "timestamp",
    "current_ensemble_name",
    "best_ensemble_name",
    "xgb_oof_auc",
    "cat_oof_auc",
    "pred_corr",
    "ensemble_oof_auc_current",
    "ensemble_oof_auc_bestgrid",
    "gain_best_vs_xgb"
]].to_string(index=False))


# =========================================================
# 11. 종료 요약
# =========================================================
print_section("11. 완료")

print(f"\n{'':>22}  {'AUC':>10}")
print("-" * 36)
print(f"{'XGBoost':>22}  {xgb_auc:>10.6f}")
print(f"{'CatBoost':>22}  {cat_auc:>10.6f}")
print(f"{'앙상블 (현재 설정)':>22}  {oof_ensemble_auc_current:>10.6f}")
print(f"{'앙상블 (최적 grid)':>22}  {oof_ensemble_auc_best:>10.6f}")

print(f"\n현재 설정 가중치 : XGB {XGB_WEIGHT:.2f} / CAT {CAT_WEIGHT:.2f}")
print(f"최적 grid 가중치 : XGB {best_xgb_w:.2f} / CAT {best_cat_w:.2f}")
print(f"예측 상관계수    : {pred_corr:.6f}")

print("\n저장 파일:")
print(f"  1. {current_oof_path}")
print(f"  2. {best_oof_path}")
print(f"  3. {current_sub_path}")
print(f"  4. {best_sub_path}")
print(f"  5. {log_path}")

print("\n다음 액션 추천:")
print("  - best grid 가중치 submission 제출")
print("  - CatBoost depth_up / reg_relax 추가 실험")
print("  - 이후 LightGBM 추가 후 3모델 앙상블 검토")