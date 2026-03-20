#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Reproduce the v22-lite 2-seed 3-model rank ensemble submission.

This script is packaged to run out of the box from this folder:

  python make_rank_ensemble.py

It will read the bundled input artifacts, rebuild the submission, and compare
the rebuilt CSV against the archived leaderboard submission.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


DEFAULT_W_XGB = 0.16
DEFAULT_W_LGB = 0.04
DEFAULT_W_CAT = 0.80

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_XGB_OOF = BASE_DIR / "ensemble_inputs" / "xgb_v2_reg_relax_oof_predictions.csv"
DEFAULT_XGB_SUB = BASE_DIR / "ensemble_inputs" / "xgb_v2_reg_relax_submission.csv"
DEFAULT_LGB_OOF = (
    BASE_DIR / "ensemble_inputs" / "lightgbm_v1_lightgbm_baseline_v1_oof_predictions.csv"
)
DEFAULT_LGB_SUB = (
    BASE_DIR / "ensemble_inputs" / "lightgbm_v1_lightgbm_baseline_v1_submission.csv"
)
DEFAULT_CAT_OOF = BASE_DIR / "result_v22_lite" / "oof_v22_lite_final.npy"
DEFAULT_CAT_TEST = BASE_DIR / "result_v22_lite" / "test_v22_lite_final.npy"
DEFAULT_SAMPLE_SUB = BASE_DIR / "data" / "sample_submission.csv"
DEFAULT_OUTPUT = BASE_DIR / "rank_ensemble_v22_lite_2seed_champ_weight_rebuilt.csv"
DEFAULT_VERIFY = BASE_DIR / "rank_ensemble_v22_lite_2seed_champ_weight.csv"


def to_rank(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average", pct=True).to_numpy()


def find_target_column(df: pd.DataFrame) -> str:
    for candidate in ("임신 성공 여부", "target"):
        if candidate in df.columns:
            return candidate

    excluded = {"ID", "oof_pred_prob", "oof_pred_label", "correct", "probability"}
    candidates = [c for c in df.columns if c not in excluded]
    if len(candidates) == 1:
        return candidates[0]
    if len(df.columns) >= 2:
        return df.columns[1]
    raise ValueError("Could not infer target column.")


def find_oof_prediction_column(df: pd.DataFrame) -> str:
    for candidate in ("oof_pred_prob", "probability", "pred", "prediction"):
        if candidate in df.columns:
            return candidate

    excluded = {"ID", "oof_pred_label", "correct", find_target_column(df)}
    candidates = [c for c in df.columns if c not in excluded]
    if candidates:
        return candidates[-1]
    raise ValueError("Could not infer OOF prediction column.")


def find_submission_prediction_column(df: pd.DataFrame) -> str:
    for candidate in ("probability", "임신 성공 여부", "target"):
        if candidate in df.columns and candidate != "ID":
            return candidate

    candidates = [c for c in df.columns if c != "ID"]
    if candidates:
        return candidates[-1]
    raise ValueError("Could not infer submission prediction column.")


def load_oof_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    target_col = find_target_column(df)
    pred_col = find_oof_prediction_column(df)
    return df[target_col].to_numpy(), df[pred_col].to_numpy()


def load_submission_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    if "ID" not in df.columns:
        raise ValueError(f"ID column not found in {path}")
    pred_col = find_submission_prediction_column(df)
    return df["ID"].to_numpy(), df[pred_col].to_numpy()


def compare_submissions(left_path: Path, right_path: Path) -> None:
    left = pd.read_csv(left_path)
    right = pd.read_csv(right_path)

    if list(left.columns) != list(right.columns):
        raise ValueError(
            f"Column mismatch between {left_path.name} and {right_path.name}: "
            f"{left.columns.tolist()} vs {right.columns.tolist()}"
        )
    if len(left) != len(right):
        raise ValueError(
            f"Row count mismatch between {left_path.name} and {right_path.name}: "
            f"{len(left)} vs {len(right)}"
        )
    if not left["ID"].equals(right["ID"]):
        raise ValueError("ID column mismatch in verification step.")

    diffs = np.abs(left["probability"].to_numpy() - right["probability"].to_numpy())
    if not np.allclose(diffs, 0.0):
        raise ValueError(f"Verification failed: probability mismatch (max diff={diffs.max()})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild the archived v22-lite 2-seed 3-model rank ensemble submission."
    )
    parser.add_argument("--xgb-oof", type=Path, default=DEFAULT_XGB_OOF)
    parser.add_argument("--xgb-sub", type=Path, default=DEFAULT_XGB_SUB)
    parser.add_argument("--lgb-oof", type=Path, default=DEFAULT_LGB_OOF)
    parser.add_argument("--lgb-sub", type=Path, default=DEFAULT_LGB_SUB)
    parser.add_argument("--cat-oof", type=Path, default=DEFAULT_CAT_OOF)
    parser.add_argument("--cat-test", type=Path, default=DEFAULT_CAT_TEST)
    parser.add_argument("--sample-sub", type=Path, default=DEFAULT_SAMPLE_SUB)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--weight-xgb", type=float, default=DEFAULT_W_XGB)
    parser.add_argument("--weight-lgb", type=float, default=DEFAULT_W_LGB)
    parser.add_argument("--weight-cat", type=float, default=DEFAULT_W_CAT)
    parser.add_argument("--verify-against", type=Path, default=DEFAULT_VERIFY)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    total_weight = args.weight_xgb + args.weight_lgb + args.weight_cat
    if not np.isclose(total_weight, 1.0):
        raise ValueError(f"weights must sum to 1.0, got {total_weight}")

    y_xgb, xgb_oof = load_oof_csv(args.xgb_oof)
    y_lgb, lgb_oof = load_oof_csv(args.lgb_oof)
    cat_oof = np.load(args.cat_oof)

    if not np.array_equal(y_xgb, y_lgb):
        raise ValueError("XGB and LGB OOF target arrays do not match.")
    if len(cat_oof) != len(y_xgb):
        raise ValueError(f"CatBoost OOF length mismatch: {len(cat_oof)} vs {len(y_xgb)}")

    xgb_ids, xgb_sub = load_submission_csv(args.xgb_sub)
    lgb_ids, lgb_sub = load_submission_csv(args.lgb_sub)
    cat_test = np.load(args.cat_test)
    sample_sub = pd.read_csv(args.sample_sub)

    if not np.array_equal(xgb_ids, lgb_ids):
        raise ValueError("XGB and LGB submission IDs do not match.")
    if len(cat_test) != len(xgb_sub):
        raise ValueError(f"CatBoost test length mismatch: {len(cat_test)} vs {len(xgb_sub)}")
    if len(sample_sub) != len(xgb_sub):
        raise ValueError(
            f"Sample submission length mismatch: {len(sample_sub)} vs {len(xgb_sub)}"
        )

    oof_pred = (
        args.weight_xgb * to_rank(xgb_oof)
        + args.weight_lgb * to_rank(lgb_oof)
        + args.weight_cat * to_rank(cat_oof)
    )
    oof_auc = roc_auc_score(y_xgb, oof_pred)

    test_pred = (
        args.weight_xgb * to_rank(xgb_sub)
        + args.weight_lgb * to_rank(lgb_sub)
        + args.weight_cat * to_rank(cat_test)
    )

    out_df = sample_sub.copy()
    if "probability" not in out_df.columns:
        raise ValueError("sample submission must contain a 'probability' column.")
    out_df["probability"] = test_pred

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)

    print("=== Rank Ensemble Completed ===")
    print(
        f"weights: xgb={args.weight_xgb:.4f}, "
        f"lgb={args.weight_lgb:.4f}, cat={args.weight_cat:.4f}"
    )
    print(f"OOF AUC: {oof_auc:.6f}")
    print(f"Saved: {args.output}")

    if args.verify_against:
        compare_submissions(args.output, args.verify_against)
        print(f"Verification passed against: {args.verify_against}")


if __name__ == "__main__":
    main()
