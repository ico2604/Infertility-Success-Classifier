from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from train_catboost import CatBoostConfig, default_config, run_training


V3_1_EXPERIMENT_NAME = "catboost_fe_v3_1_ivf_di_split"
V3_1_REFERENCE_SUMMARY = (
    Path(__file__).resolve().parent
    / "outputs"
    / "catboost"
    / "20260314_021753_catboost_fe_v3_1_ivf_di_split"
    / "run_summary.json"
)


def build_v3_1_config(data_dir: Path | None = None) -> CatBoostConfig:
    base_config = default_config(data_dir)
    return replace(
        base_config,
        experiment_name=V3_1_EXPERIMENT_NAME,
        iterations=6000,
        learning_rate=0.02,
        depth=6,
        l2_leaf_reg=10.0,
        random_strength=1.2,
        bagging_temperature=0.8,
        rsm=0.8,
        border_count=128,
        early_stopping_rounds=300,
        task_type="GPU",
        devices="0",
        enable_ivf_di_split=True,
        enable_infertility_bundle=False,
        enable_rare_cleanup=False,
        enable_pruning=False,
        enable_time_delta_signals=False,
        enable_missing_semantics=False,
        enable_cause_history_interactions=False,
        enable_expert_ivf_di=False,
        notes=(
            "V3-1 전용 래퍼: V2 CatBoost 파라미터를 유지하고 "
            "IVF/DI 분기 피처만 활성화한 기준 실험"
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="V3-1 전용 CatBoost 래퍼. 공용 학습기와 공용 V3 feature 코드를 재사용합니다."
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = build_v3_1_config(args.data_dir)
    return run_training(config, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
