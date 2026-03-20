#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Reproduce the v22-lite 2-seed 3-model rank ensemble submission.

Blend:
  - XGB reg_relax: 0.16
  - LGB v1: 0.04
  - CatBoost v22-lite: 0.80

Example:
  python make_rank_ensemble.py ^
    --xgb-oof "C:\\path\\xgb_v2_reg_relax_oof_predictions.csv" ^
    --xgb-sub "C:\\path\\xgb_v2_reg_relax_submission.csv" ^
    --lgb-oof "C:\\path\\lightgbm_v1_lightgbm_baseline_v1_oof_predictions.csv" ^
    --lgb-sub "C:\\path\\lightgbm_v1_lightgbm_baseline_v1_submission.csv" ^
    --cat-oof "C:\\path\\oof_v22_lite_final.npy" ^
    --cat-test "C:\\path\\test_v22_lite_final.npy" ^
    --sample-sub "C:\\path\\sample_submission.csv" ^
    --output "C:\\path\\rank_ensemble_v22_lite_2seed_champ_weight.csv"
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


def to_rank(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average", pct=True).to_numpy()


def load_oof_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)

    if "oof_pred_prob" in df.columns:
        pred_col = "oof_pred_prob"
    else:
        numeric_cols = [c for c in df.columns if c != "ID"]
        pred_candidates = [c for c in numeric_cols if c not in ("임신 성공 여부", "target")]
        if not pred_candidates:
            raise ValueError(f"OOF prediction column not found in {path}")
        pred_col = pred_candidates[-1]

    if "임신 성공 여부" in df.columns:
        y_col = "임신 성공 여부"
    elif "target" in df.columns:
        y_col = "target"
    else:
        raise ValueError(f"Target column not found in {path}")

    return df[y_col].to_numpy(), df[pred_col].to_numpy()


def load_submission_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    if "ID" not in df.columns:
        raise ValueError(f"ID column not found in {path}")

    if "probability" in df.columns:
        pred_col = "probability"
    elif "임신 성공 여부" in df.columns:
        pred_col = "임신 성공 여부"
    else:
        pred_candidates = [c for c in df.columns if c != "ID"]
        if not pred_candidates:
            raise ValueError(f"Submission prediction column not found in {path}")
        pred_col = pred_candidates[-1]

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

    if not np.allclose(left["probability"].to_numpy(), right["probability"].to_numpy()):
        max_diff = np.max(
            np.abs(left["probability"].to_numpy() - right["probability"].to_numpy())
        )
        raise ValueError(f"Verification failed: probability mismatch (max diff={max_diff})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the v22-lite 2-seed 3-model rank ensemble submission."
    )
    parser.add_argument("--xgb-oof", type=Path, required=True)
    parser.add_argument("--xgb-sub", type=Path, required=True)
    parser.add_argument("--lgb-oof", type=Path, required=True)
    parser.add_argument("--lgb-sub", type=Path, required=True)
    parser.add_argument("--cat-oof", type=Path, required=True)
    parser.add_argument("--cat-test", type=Path, required=True)
    parser.add_argument("--sample-sub", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("rank_ensemble_v22_lite_2seed_champ_weight.csv"),
    )
    parser.add_argument("--weight-xgb", type=float, default=DEFAULT_W_XGB)
    parser.add_argument("--weight-lgb", type=float, default=DEFAULT_W_LGB)
    parser.add_argument("--weight-cat", type=float, default=DEFAULT_W_CAT)
    parser.add_argument(
        "--verify-against",
        type=Path,
        default=None,
        help="Optional existing submission CSV to compare against.",
    )
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
        raise ValueError(
            f"CatBoost OOF length mismatch: {len(cat_oof)} vs {len(y_xgb)}"
        )

    xgb_ids, xgb_sub = load_submission_csv(args.xgb_sub)
    lgb_ids, lgb_sub = load_submission_csv(args.lgb_sub)
    cat_test = np.load(args.cat_test)
    sample_sub = pd.read_csv(args.sample_sub)

    if not np.array_equal(xgb_ids, lgb_ids):
        raise ValueError("XGB and LGB submission IDs do not match.")
    if len(cat_test) != len(xgb_sub):
        raise ValueError(
            f"CatBoost test length mismatch: {len(cat_test)} vs {len(xgb_sub)}"
        )
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

    if args.verify_against is not None:
        compare_submissions(args.output, args.verify_against)
        print(f"Verification passed against: {args.verify_against}")


if __name__ == "__main__":
    main()
