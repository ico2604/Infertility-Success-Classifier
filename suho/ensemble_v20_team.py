#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ensemble_v20_team.py
================================================================
v20_pruned + 팀원(XGB/CAT/LGB) 결과 앙상블

기능:
  1. v20_pruned OOF/Test 로드
  2. 팀원 XGB/CAT/LGB OOF/Sub 로드
  3. prob blend / rank blend 둘 다 탐색
  4. 모든 부분집합(2~4개) + simplex weight search
  5. OOF AUC 기준 최고 조합 선택
  6. 최종 submission / 결과표 / 로그 저장

주의:
  - TEAM_OUTPUT_DIR 경로는 실제 팀원 결과 폴더로 수정 필요
  - csv 예측 컬럼명은 자동 탐지하지만, 애매하면 직접 지정 권장
"""

import os
import re
import json
import time
import datetime
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, log_loss, average_precision_score

warnings.filterwarnings("ignore")

# ============================================================
# 경로 설정
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
RESULT_DIR = os.path.join(BASE_DIR, "result")
TEAM_OUTPUT_DIR = os.path.join(BASE_DIR, "team_output")
os.makedirs(RESULT_DIR, exist_ok=True)

VERSION = "ensemble_v20_team"
NOW = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_PATH = os.path.join(RESULT_DIR, f"log_{VERSION}_{NOW}.md")

# ============================================================
# 입력 파일
# ============================================================
TARGET_COL = "임신 성공 여부"

# 내 모델(v20)
V20_OOF_PATH = os.path.join(RESULT_DIR, "oof_v20_pruned_final.npy")
V20_TEST_PATH = os.path.join(RESULT_DIR, "test_v20_pruned_final.npy")

# 팀원 결과
XGB_OOF_PATH = os.path.join(TEAM_OUTPUT_DIR, "xgb_v2_reg_relax_oof_predictions.csv")
XGB_SUB_PATH = os.path.join(TEAM_OUTPUT_DIR, "xgb_v2_reg_relax_submission.csv")


LGB_OOF_PATH = os.path.join(
    TEAM_OUTPUT_DIR, "lightgbm_v1_lightgbm_baseline_v1_oof_predictions.csv"
)
LGB_SUB_PATH = os.path.join(
    TEAM_OUTPUT_DIR, "lightgbm_v1_lightgbm_baseline_v1_submission.csv"
)

# ============================================================
# 탐색 설정
# ============================================================
COARSE_STEP = 0.05  # 0.05 simplex grid
FINE_STEP = 0.01  # local refine step
FINE_RADIUS = 0.10  # coarse best 주변 ±0.10 범위 refine
MIN_MODELS_IN_BLEND = 2
MAX_MODELS_IN_BLEND = 4

CLIP_EPS = 1e-7

# ============================================================
# 로그
# ============================================================
log_lines = []


def log(msg=""):
    print(msg)
    log_lines.append(str(msg))


def save_log():
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))


# ============================================================
# 유틸
# ============================================================
ID_CANDIDATES = [
    "ID",
    "id",
    "Id",
    "sample_id",
    "row_id",
    "index",
    "idx",
    "환자ID",
    "환자_id",
    "case_id",
]

TARGET_CANDIDATES = [TARGET_COL, "target", "label", "y", "true", "정답"]

PRED_CANDIDATES = [
    "probability",
    "prediction",
    "pred",
    "preds",
    "score",
    "prob",
    "oof_pred",
    "oof_prediction",
    "oof_prob",
    "y_pred",
    "prediction_proba",
]


def clip01(arr, eps=CLIP_EPS):
    arr = np.asarray(arr, dtype=float).reshape(-1)
    return np.clip(arr, eps, 1 - eps)


def as_rank(arr):
    arr = np.asarray(arr, dtype=float).reshape(-1)
    return rankdata(arr, method="average") / len(arr)


def detect_id_col(df: pd.DataFrame, expected_id_name: str = None):
    if expected_id_name is not None and expected_id_name in df.columns:
        return expected_id_name

    for c in df.columns:
        if c in ID_CANDIDATES:
            return c

    for c in df.columns:
        cl = str(c).lower()
        if cl == "id" or cl.endswith("_id") or "id" in cl:
            return c
    return None


def detect_pred_col(df: pd.DataFrame, path: str, expected_len: int = None):
    cols = list(df.columns)

    # 1) 후보 이름 우선
    for c in PRED_CANDIDATES:
        if c in cols:
            return c

    # 2) 2컬럼이면 id 아닌 쪽 또는 두 번째 컬럼
    if len(cols) == 2:
        id_col = detect_id_col(df)
        if id_col is not None:
            other = [c for c in cols if c != id_col]
            if len(other) == 1:
                return other[0]
        return cols[1]

    # 3) numeric col 중 target/id 제외 후 선택
    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    numeric_cols = [
        c for c in numeric_cols if c not in ID_CANDIDATES and c not in TARGET_CANDIDATES
    ]

    if len(numeric_cols) == 1:
        return numeric_cols[0]

    if len(numeric_cols) >= 2:
        # unique가 많고 0/1 binary target 아닌 쪽을 우선
        cand = []
        for c in numeric_cols:
            nunique = df[c].nunique(dropna=True)
            is_binary_like = nunique <= 2
            cand.append((c, nunique, is_binary_like))
        cand = sorted(
            cand, key=lambda x: (x[2], -x[1])
        )  # binary 아닌 것 우선, unique 큰 것 우선
        return cand[0][0]

    raise ValueError(f"[예측 컬럼 탐지 실패] path={path}, columns={cols}")


def load_csv_prediction(path: str, expected_len: int, expected_ids: pd.Series = None):
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    pred_col = detect_pred_col(df, path=path, expected_len=expected_len)
    id_col = detect_id_col(
        df, expected_id_name=(expected_ids.name if expected_ids is not None else None)
    )

    # id align 가능하면 align
    if expected_ids is not None and id_col is not None:
        key_name = "__expected_id__"
        tmp = pd.DataFrame({key_name: expected_ids.values})
        merged = tmp.merge(
            df[[id_col, pred_col]].copy(),
            left_on=key_name,
            right_on=id_col,
            how="left",
        )
        arr = pd.to_numeric(merged[pred_col], errors="coerce").values
        if np.isnan(arr).any():
            n_nan = np.isnan(arr).sum()
            raise ValueError(
                f"[ID align 후 NaN 발생] {path}, pred_col={pred_col}, id_col={id_col}, nan={n_nan}"
            )
    else:
        arr = pd.to_numeric(df[pred_col], errors="coerce").values
        if len(arr) != expected_len:
            raise ValueError(
                f"[길이 불일치] {path}, pred_col={pred_col}, len={len(arr)}, expected={expected_len}"
            )

    arr = clip01(arr)
    return arr, pred_col, id_col, df.shape


def load_npy_prediction(path: str, expected_len: int):
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    arr = np.load(path)
    arr = np.asarray(arr).reshape(-1)
    if len(arr) != expected_len:
        raise ValueError(
            f"[길이 불일치] {path}, len={len(arr)}, expected={expected_len}"
        )
    arr = clip01(arr)
    return arr


def simplex_grid(n_models: int, step: float):
    """weights 합=1 grid 생성"""
    units = int(round(1 / step))

    def rec(k, remain, prefix):
        if k == n_models - 1:
            yield prefix + [remain]
            return
        for i in range(remain + 1):
            yield from rec(k + 1, remain - i, prefix + [i])

    for ints in rec(0, units, []):
        yield np.array(ints, dtype=float) / units


def local_simplex_grid(
    best_weights: np.ndarray, step: float = 0.01, radius: float = 0.10
):
    """
    coarse best 주변 국소 탐색.
    각 weight는 best_i ± radius 범위 내,
    step 단위, 합=1
    """
    n = len(best_weights)
    units = int(round(1 / step))
    best_int = np.round(best_weights * units).astype(int)
    rad_int = int(round(radius / step))

    lows = np.maximum(0, best_int - rad_int)
    highs = np.minimum(units, best_int + rad_int)

    def rec(k, remain, prefix):
        if k == n - 1:
            val = remain
            if lows[k] <= val <= highs[k]:
                yield prefix + [val]
            return

        low_k = lows[k]
        high_k = highs[k]

        # 뒤쪽 최소/최대 고려해서 pruning
        min_rest = np.sum(lows[k + 1 :])
        max_rest = np.sum(highs[k + 1 :])

        start = max(low_k, remain - max_rest)
        end = min(high_k, remain - min_rest)

        for v in range(start, end + 1):
            yield from rec(k + 1, remain - v, prefix + [v])

    for ints in rec(0, units, []):
        yield np.array(ints, dtype=float) / units


def evaluate_pred(y, pred):
    pred = clip01(pred)
    return {
        "auc": roc_auc_score(y, pred),
        "logloss": log_loss(y, pred),
        "ap": average_precision_score(y, pred),
    }


def weighted_blend(pred_mat: np.ndarray, weights: np.ndarray):
    return pred_mat @ weights


# ============================================================
# 탐색
# ============================================================
def search_best_blend_for_subset(
    y: np.ndarray,
    pred_mat: np.ndarray,
    subset_names: list,
    mode: str,
    coarse_step: float = 0.05,
    fine_step: float = 0.01,
    fine_radius: float = 0.10,
):
    n_models = pred_mat.shape[1]

    # 1) coarse
    best = None
    for w in simplex_grid(n_models, coarse_step):
        pred = weighted_blend(pred_mat, w)
        auc = roc_auc_score(y, pred)
        if (best is None) or (auc > best["auc"]):
            m = evaluate_pred(y, pred)
            best = {
                "mode": mode,
                "subset": subset_names.copy(),
                "weights": w.copy(),
                "auc": m["auc"],
                "logloss": m["logloss"],
                "ap": m["ap"],
                "stage": "coarse",
            }

    # 2) fine local refine
    fine_best = best.copy()
    for w in local_simplex_grid(best["weights"], step=fine_step, radius=fine_radius):
        pred = weighted_blend(pred_mat, w)
        auc = roc_auc_score(y, pred)
        if auc > fine_best["auc"]:
            m = evaluate_pred(y, pred)
            fine_best = {
                "mode": mode,
                "subset": subset_names.copy(),
                "weights": w.copy(),
                "auc": m["auc"],
                "logloss": m["logloss"],
                "ap": m["ap"],
                "stage": "fine",
            }

    return fine_best


def run_ensemble_search(y, oof_dict, test_dict):
    """
    prob blend / rank blend 둘 다 탐색
    부분집합 크기: 2~4
    """
    model_names = list(oof_dict.keys())
    results = []

    # 단일 모델 성능도 기록
    for name in model_names:
        m = evaluate_pred(y, oof_dict[name])
        results.append(
            {
                "mode": "single",
                "subset": name,
                "weights": {name: 1.0},
                "auc": m["auc"],
                "logloss": m["logloss"],
                "ap": m["ap"],
                "stage": "single",
            }
        )

    # prob / rank mode
    for mode in ["prob", "rank"]:
        log(f"\n### [{mode}] blend 탐색 시작")

        if mode == "prob":
            oof_used = {k: clip01(v) for k, v in oof_dict.items()}
            test_used = {k: clip01(v) for k, v in test_dict.items()}
        else:
            oof_used = {k: as_rank(v) for k, v in oof_dict.items()}
            test_used = {k: as_rank(v) for k, v in test_dict.items()}

        for r in range(
            MIN_MODELS_IN_BLEND, min(MAX_MODELS_IN_BLEND, len(model_names)) + 1
        ):
            for subset in combinations(model_names, r):
                subset = list(subset)
                pred_mat = np.column_stack([oof_used[k] for k in subset])

                best_subset = search_best_blend_for_subset(
                    y=y,
                    pred_mat=pred_mat,
                    subset_names=subset,
                    mode=mode,
                    coarse_step=COARSE_STEP,
                    fine_step=FINE_STEP,
                    fine_radius=FINE_RADIUS,
                )

                weight_map = {
                    subset[i]: float(best_subset["weights"][i])
                    for i in range(len(subset))
                }
                results.append(
                    {
                        "mode": mode,
                        "subset": " + ".join(subset),
                        "weights": weight_map,
                        "auc": best_subset["auc"],
                        "logloss": best_subset["logloss"],
                        "ap": best_subset["ap"],
                        "stage": best_subset["stage"],
                    }
                )

                log(
                    f"  subset={subset} | "
                    f"AUC={best_subset['auc']:.6f} | "
                    f"LogLoss={best_subset['logloss']:.6f} | "
                    f"AP={best_subset['ap']:.6f} | "
                    f"weights={weight_map}"
                )

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        ["auc", "ap"], ascending=[False, False]
    ).reset_index(drop=True)

    # best row
    best_row = results_df.iloc[0].to_dict()

    # best submission 생성용
    if best_row["mode"] == "single":
        best_model = best_row["subset"]
        best_test = test_dict[best_model].copy()
        best_oof = oof_dict[best_model].copy()
    else:
        subset = best_row["subset"].split(" + ")
        weights = best_row["weights"]

        if best_row["mode"] == "prob":
            test_used = {k: clip01(v) for k, v in test_dict.items()}
            oof_used = {k: clip01(v) for k, v in oof_dict.items()}
        else:
            test_used = {k: as_rank(v) for k, v in test_dict.items()}
            oof_used = {k: as_rank(v) for k, v in oof_dict.items()}

        best_test = np.zeros(len(next(iter(test_used.values()))), dtype=float)
        best_oof = np.zeros(len(next(iter(oof_used.values()))), dtype=float)

        for k in subset:
            best_test += weights[k] * test_used[k]
            best_oof += weights[k] * oof_used[k]

        best_test = clip01(best_test)
        best_oof = clip01(best_oof)

    return results_df, best_row, best_oof, best_test


# ============================================================
# 메인
# ============================================================
def main():
    t0 = time.time()

    log(f"# {VERSION}")
    log(f"시각: {NOW}")
    log("=" * 70)

    # --------------------------------------------------------
    # [1] 데이터 로드
    # --------------------------------------------------------
    log("\n## [1] 데이터 로드")
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    sample_sub = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

    y = train_df[TARGET_COL].values
    train_ids = train_df.iloc[:, 0]
    test_ids = sample_sub.iloc[:, 0]

    log(f"- train: {train_df.shape}")
    log(f"- sample_submission: {sample_sub.shape}")
    log(f"- target 0={np.sum(y==0)}, 1={np.sum(y==1)}, pos={np.mean(y)*100:.2f}%")

    # --------------------------------------------------------
    # [2] 예측 파일 로드
    # --------------------------------------------------------
    log("\n## [2] 예측 파일 로드")

    oof_dict = {}
    test_dict = {}

    # v20 npy
    v20_oof = load_npy_prediction(V20_OOF_PATH, expected_len=len(train_df))
    v20_test = load_npy_prediction(V20_TEST_PATH, expected_len=len(sample_sub))
    oof_dict["v20"] = v20_oof
    test_dict["v20"] = v20_test
    log(f"- v20 OOF: {V20_OOF_PATH}")
    log(f"- v20 TEST: {V20_TEST_PATH}")
    log(f"  v20 OOF AUC={roc_auc_score(y, v20_oof):.6f}")

    # team csvs
    for model_name, oof_path, sub_path in [
        ("xgb", XGB_OOF_PATH, XGB_SUB_PATH),
        ("lgb", LGB_OOF_PATH, LGB_SUB_PATH),
    ]:
        oof_arr, oof_col, oof_id_col, oof_shape = load_csv_prediction(
            oof_path, expected_len=len(train_df), expected_ids=train_ids
        )
        sub_arr, sub_col, sub_id_col, sub_shape = load_csv_prediction(
            sub_path, expected_len=len(sample_sub), expected_ids=test_ids
        )

        oof_dict[model_name] = oof_arr
        test_dict[model_name] = sub_arr

        log(f"- {model_name.upper()} OOF: {oof_path}")
        log(
            f"  shape={oof_shape}, pred_col={oof_col}, id_col={oof_id_col}, auc={roc_auc_score(y, oof_arr):.6f}"
        )
        log(f"- {model_name.upper()} SUB: {sub_path}")
        log(f"  shape={sub_shape}, pred_col={sub_col}, id_col={sub_id_col}")

    # --------------------------------------------------------
    # [3] OOF 상관 분석
    # --------------------------------------------------------
    log("\n## [3] OOF 상관 분석")
    oof_df = pd.DataFrame({k: v for k, v in oof_dict.items()})
    corr_df = oof_df.corr(method="pearson")
    corr_path = os.path.join(RESULT_DIR, f"corr_{VERSION}_{NOW}.csv")
    corr_df.to_csv(corr_path, encoding="utf-8-sig")

    log("- OOF Pearson correlation")
    for row_name in corr_df.index:
        vals = ", ".join(
            [f"{col}={corr_df.loc[row_name, col]:.4f}" for col in corr_df.columns]
        )
        log(f"  {row_name}: {vals}")
    log(f"- 저장: {corr_path}")

    # --------------------------------------------------------
    # [4] 앙상블 탐색
    # --------------------------------------------------------
    log("\n## [4] 앙상블 탐색 시작")
    t_search = time.time()

    results_df, best_row, best_oof, best_test = run_ensemble_search(
        y=y,
        oof_dict=oof_dict,
        test_dict=test_dict,
    )

    search_time = (time.time() - t_search) / 60
    log(f"\n- 탐색 소요: {search_time:.2f}분")

    # --------------------------------------------------------
    # [5] 결과 저장
    # --------------------------------------------------------
    log("\n## [5] 결과 저장")

    # 결과표 저장
    results_save = results_df.copy()
    results_save["weights"] = results_save["weights"].apply(
        lambda x: json.dumps(x, ensure_ascii=False)
    )
    result_csv_path = os.path.join(RESULT_DIR, f"ensemble_search_{VERSION}_{NOW}.csv")
    results_save.to_csv(result_csv_path, index=False, encoding="utf-8-sig")
    log(f"- 탐색 결과 저장: {result_csv_path}")

    # best info json
    best_info = {
        "best_mode": best_row["mode"],
        "best_subset": best_row["subset"],
        "best_weights": best_row["weights"],
        "best_auc": float(best_row["auc"]),
        "best_logloss": float(best_row["logloss"]),
        "best_ap": float(best_row["ap"]),
        "coarse_step": COARSE_STEP,
        "fine_step": FINE_STEP,
        "fine_radius": FINE_RADIUS,
    }
    best_json_path = os.path.join(RESULT_DIR, f"best_{VERSION}_{NOW}.json")
    with open(best_json_path, "w", encoding="utf-8") as f:
        json.dump(best_info, f, ensure_ascii=False, indent=2)
    log(f"- best config 저장: {best_json_path}")

    # OOF / TEST npy
    oof_path = os.path.join(RESULT_DIR, f"oof_{VERSION}_{NOW}.npy")
    test_path = os.path.join(RESULT_DIR, f"test_{VERSION}_{NOW}.npy")
    np.save(oof_path, best_oof)
    np.save(test_path, best_test)
    log(f"- OOF npy 저장: {oof_path}")
    log(f"- TEST npy 저장: {test_path}")

    # submission 저장
    sub = sample_sub.copy()
    pred_col = sub.columns[-1]
    sub[pred_col] = best_test
    sub_path = os.path.join(RESULT_DIR, f"submission_{VERSION}_{NOW}.csv")
    sub.to_csv(sub_path, index=False)
    log(f"- submission 저장: {sub_path}")

    # --------------------------------------------------------
    # [6] 상위 결과 출력
    # --------------------------------------------------------
    log("\n## [6] 상위 결과 Top 20")
    top20 = results_df.head(20).copy()

    for i, row in top20.iterrows():
        log(
            f"{i+1:>2}. "
            f"mode={row['mode']:<6} | "
            f"subset={row['subset']:<30} | "
            f"AUC={row['auc']:.6f} | "
            f"LogLoss={row['logloss']:.6f} | "
            f"AP={row['ap']:.6f} | "
            f"weights={row['weights']}"
        )

    # --------------------------------------------------------
    # [7] 최종 요약
    # --------------------------------------------------------
    final_metrics = evaluate_pred(y, best_oof)
    total_time = (time.time() - t0) / 60

    log("\n" + "=" * 70)
    log("## 최종 요약")
    log("=" * 70)
    log(f"- best mode:   {best_row['mode']}")
    log(f"- best subset: {best_row['subset']}")
    log(f"- best weights: {best_row['weights']}")
    log(f"- best OOF AUC:     {final_metrics['auc']:.6f}")
    log(f"- best OOF LogLoss: {final_metrics['logloss']:.6f}")
    log(f"- best OOF AP:      {final_metrics['ap']:.6f}")
    log(f"- submission: {sub_path}")
    log(f"- 총 소요: {total_time:.2f}분")
    log("=" * 70)

    save_log()
    print(f"\n완료! 로그: {LOG_PATH}")


if __name__ == "__main__":
    main()
