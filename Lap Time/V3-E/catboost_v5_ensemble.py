from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from eda_baseline_inspection import SUBMISSION_FILE, TRAIN_FILE, detect_default_data_dir
from experiment_tracker import append_registry, create_experiment_dir, read_registry_frame, write_json
from lightgbm_feature_engineering import ID_COLUMN, TARGET_COLUMN


BRANCH_DIR = Path(__file__).resolve().parent
ROOT_DIR = BRANCH_DIR.parent
DEFAULT_OUTPUT_ROOT = BRANCH_DIR / "outputs" / "ensemble"
DEFAULT_REGISTRY_PATH = BRANCH_DIR / "outputs" / "experiment_registry.csv"
DEFAULT_V3_SUMMARY_PATH = BRANCH_DIR / "outputs" / "catboost" / "v3_campaign_summary.json"
DEFAULT_V4_SUMMARY_PATH = BRANCH_DIR / "outputs" / "catboost" / "v4_campaign_summary.json"
DEFAULT_EXTERNAL_ROOT = ROOT_DIR.parent / "클로드 오즈코딩" / "난임 환자 임신 성공률"


@dataclass(frozen=True)
class CandidateSpec:
    label: str
    source_type: str
    model_name: str
    exp_id: str
    overall_auc: float | None
    oof_path: Path
    submission_path: Path
    oof_ids_path: Path | None = None
    submission_ids_path: Path | None = None
    notes: str = ""


def _best_lightgbm_candidate(registry_path: Path) -> CandidateSpec | None:
    registry = read_registry_frame(registry_path)
    if registry.empty:
        return None
    lightgbm_rows = registry[registry["model_name"] == "lightgbm"].copy()
    if lightgbm_rows.empty:
        return None
    lightgbm_rows["overall_auc"] = pd.to_numeric(lightgbm_rows["overall_auc"], errors="coerce")
    lightgbm_rows = lightgbm_rows.dropna(subset=["overall_auc"])
    if lightgbm_rows.empty:
        return None
    best_row = lightgbm_rows.sort_values("overall_auc", ascending=False).iloc[0]
    output_dir = Path(str(best_row["output_dir"]))
    return CandidateSpec(
        label="codex_lgb_seedavg",
        source_type="internal_csv",
        model_name="lightgbm",
        exp_id=str(best_row["exp_id"]),
        overall_auc=float(best_row["overall_auc"]),
        oof_path=output_dir / "oof_predictions.csv",
        submission_path=output_dir / "submission_lightgbm.csv",
        notes="best internal LightGBM",
    )


def _completed_v3_candidates(v3_summary_path: Path) -> list[CandidateSpec]:
    if not v3_summary_path.exists():
        return []
    summary = json.loads(v3_summary_path.read_text(encoding="utf-8"))
    name_to_label = {
        "catboost_fe_v3_1_ivf_di_split": "codex_v3_1",
        "catboost_fe_v3_2_infertility_bundle": "codex_v3_2",
        "catboost_fe_v3_3_rare_cleanup_prune": "codex_v3_3",
    }
    candidates: list[CandidateSpec] = []
    for item in summary.get("experiments", []):
        if item.get("status_code") != "completed":
            continue
        if item.get("experiment_name") not in name_to_label:
            continue
        output_dir = Path(str(item["output_dir"]))
        candidates.append(
            CandidateSpec(
                label=name_to_label[item["experiment_name"]],
                source_type="internal_csv",
                model_name="catboost",
                exp_id=str(item.get("exp_id") or item["experiment_name"]),
                overall_auc=float(item["overall_auc"]) if item.get("overall_auc") is not None else None,
                oof_path=output_dir / "oof_predictions.csv",
                submission_path=output_dir / "submission_catboost.csv",
                notes=item["experiment_name"],
            )
        )
    return candidates


def _external_step09_candidates(external_root: Path) -> list[CandidateSpec]:
    models_dir = external_root / "models"
    sample_submission_path = external_root / "open" / "sample_submission.csv"
    candidate_map = {
        "claude_v3_lgb": ("oof_v3_lgb.npy", "test_v3_lgb.npy", "lightgbm"),
        "claude_v3_xgb": ("oof_v3_xgb.npy", "test_v3_xgb.npy", "xgboost"),
        "claude_v3_cat": ("oof_v3_cat.npy", "test_v3_cat.npy", "catboost"),
        "claude_v3_ensemble": ("oof_v3_ensemble.npy", "test_v3_ensemble.npy", "ensemble"),
    }
    candidates: list[CandidateSpec] = []
    for label, (oof_name, test_name, model_name) in candidate_map.items():
        oof_path = models_dir / oof_name
        submission_path = models_dir / test_name
        if not oof_path.exists() or not submission_path.exists():
            continue
        candidates.append(
            CandidateSpec(
                label=label,
                source_type="external_npy",
                model_name=model_name,
                exp_id=label,
                overall_auc=None,
                oof_path=oof_path,
                submission_path=submission_path,
                submission_ids_path=sample_submission_path if sample_submission_path.exists() else None,
                notes="external Step09 candidate",
            )
        )
    return candidates


def _load_internal_candidate(
    spec: CandidateSpec,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    oof_frame = pd.read_csv(spec.oof_path)
    submission_path = spec.submission_path
    if not submission_path.exists():
        sibling_candidates = sorted(spec.oof_path.parent.glob("submission*.csv"))
        if not sibling_candidates:
            raise FileNotFoundError(submission_path)
        submission_path = sibling_candidates[0]
    submission_frame = pd.read_csv(submission_path)
    oof_pred_column = [column for column in oof_frame.columns if column not in {ID_COLUMN, TARGET_COLUMN}][0]
    submission_pred_column = [column for column in submission_frame.columns if column != ID_COLUMN][0]
    return (
        oof_frame[[ID_COLUMN, oof_pred_column]].rename(columns={oof_pred_column: spec.label}),
        submission_frame[[ID_COLUMN, submission_pred_column]].rename(columns={submission_pred_column: spec.label}),
    )


def _load_external_candidate(
    spec: CandidateSpec,
    train_ids: pd.Series,
    submission_ids: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    oof_values = np.load(spec.oof_path)
    test_values = np.load(spec.submission_path)
    if len(oof_values) != len(train_ids):
        raise RuntimeError(
            f"external OOF row mismatch: {spec.label} -> {len(oof_values)} vs train_ids {len(train_ids)}"
        )
    if len(test_values) != len(submission_ids):
        raise RuntimeError(
            f"external submission row mismatch: {spec.label} -> {len(test_values)} vs submission_ids {len(submission_ids)}"
        )

    candidate_submission_ids = submission_ids
    if spec.submission_ids_path is not None and spec.submission_ids_path.exists():
        candidate_submission_ids = pd.read_csv(spec.submission_ids_path, encoding="utf-8-sig")[ID_COLUMN]
        if len(candidate_submission_ids) != len(submission_ids):
            raise RuntimeError(f"external submission ID length mismatch for {spec.label}")
        if candidate_submission_ids.tolist() != submission_ids.tolist():
            reorder_frame = pd.DataFrame({ID_COLUMN: candidate_submission_ids, spec.label: test_values})
            submission_frame = pd.DataFrame({ID_COLUMN: submission_ids}).merge(reorder_frame, on=ID_COLUMN, how="left")
            if submission_frame[spec.label].isna().any():
                raise RuntimeError(f"external submission ID reorder failed for {spec.label}")
        else:
            submission_frame = pd.DataFrame({ID_COLUMN: submission_ids, spec.label: test_values})
    else:
        submission_frame = pd.DataFrame({ID_COLUMN: submission_ids, spec.label: test_values})

    return (
        pd.DataFrame({ID_COLUMN: train_ids, spec.label: oof_values}),
        submission_frame,
    )


def _load_candidate_frames(
    spec: CandidateSpec,
    train_ids: pd.Series,
    submission_ids: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if spec.source_type == "internal_csv":
        return _load_internal_candidate(spec)
    if spec.source_type == "external_npy":
        return _load_external_candidate(spec, train_ids, submission_ids)
    raise RuntimeError(f"Unsupported candidate source_type: {spec.source_type}")


def _default_candidates(
    *,
    v3_summary_path: Path,
    registry_path: Path,
    external_root: Path,
) -> list[CandidateSpec]:
    candidates = _completed_v3_candidates(v3_summary_path)
    lightgbm_candidate = _best_lightgbm_candidate(registry_path)
    if lightgbm_candidate is not None:
        candidates.append(lightgbm_candidate)
    candidates.extend(_external_step09_candidates(external_root))
    return candidates


def _merge_candidate_predictions(
    *,
    candidates: list[CandidateSpec],
    train_ids: pd.Series,
    y_true: pd.Series,
    submission_ids: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged_oof = pd.DataFrame({ID_COLUMN: train_ids, TARGET_COLUMN: y_true})
    merged_submission = pd.DataFrame({ID_COLUMN: submission_ids})

    for spec in candidates:
        oof_frame, submission_frame = _load_candidate_frames(spec, train_ids, submission_ids)
        merged_oof = merged_oof.merge(oof_frame, on=ID_COLUMN, how="inner")
        merged_submission = merged_submission.merge(submission_frame, on=ID_COLUMN, how="inner")

    if len(merged_oof) != len(train_ids):
        raise RuntimeError("Merged OOF row count mismatch after candidate alignment")
    if len(merged_submission) != len(submission_ids):
        raise RuntimeError("Merged submission row count mismatch after candidate alignment")
    return merged_oof, merged_submission


def _evaluate_rank_average(merged_oof: pd.DataFrame, labels: list[str]) -> tuple[float, pd.Series]:
    ranked = merged_oof[labels].rank(pct=True)
    prediction = ranked.mean(axis=1)
    auc = float(roc_auc_score(merged_oof[TARGET_COLUMN], prediction))
    return auc, prediction


def _grid_weight_tuples(candidate_count: int, step: float = 0.05) -> list[tuple[float, ...]]:
    units = int(round(1.0 / step))
    weight_tuples: list[tuple[float, ...]] = []
    if candidate_count == 0:
        return weight_tuples
    for values in np.ndindex(*([units + 1] * candidate_count)):
        if sum(values) != units:
            continue
        weight_tuples.append(tuple(value / units for value in values))
    return weight_tuples


def _evaluate_soft_vote(
    merged_oof: pd.DataFrame,
    labels: list[str],
    *,
    step: float = 0.05,
) -> tuple[float, dict[str, float], pd.Series]:
    matrix = merged_oof[labels].to_numpy(dtype=float)
    y_true = merged_oof[TARGET_COLUMN].to_numpy(dtype=int)
    best_auc = float("-inf")
    best_weights: dict[str, float] = {}
    best_prediction = pd.Series(np.zeros(len(merged_oof)), index=merged_oof.index, dtype="float64")
    for weights in _grid_weight_tuples(len(labels), step=step):
        weight_array = np.asarray(weights, dtype=float)
        prediction_array = matrix @ weight_array
        auc = float(roc_auc_score(y_true, prediction_array))
        if auc > best_auc:
            best_auc = auc
            best_weights = {label: weight for label, weight in zip(labels, weights)}
            best_prediction = pd.Series(prediction_array, index=merged_oof.index, dtype="float64")
    return best_auc, best_weights, best_prediction


def _curated_strategy_sets(candidate_labels: list[str]) -> list[dict[str, Any]]:
    available = set(candidate_labels)

    def _present(labels: list[str]) -> list[str]:
        return [label for label in labels if label in available]

    strategies = [
        {
            "name": "our_core",
            "labels": _present(["codex_v3_1", "codex_v3_2", "codex_v3_3", "codex_lgb_seedavg"]),
            "allow_soft": True,
        },
        {
            "name": "external_core",
            "labels": _present(["claude_v3_lgb", "claude_v3_xgb", "claude_v3_cat", "claude_v3_ensemble"]),
            "allow_soft": True,
        },
        {
            "name": "cross_team_doc4",
            "labels": _present(["codex_v3_1", "claude_v3_lgb", "claude_v3_xgb", "claude_v3_cat"]),
            "allow_soft": True,
        },
        {
            "name": "cross_team_doc5",
            "labels": _present(["codex_v3_1", "claude_v3_lgb", "claude_v3_xgb", "claude_v3_cat", "claude_v3_ensemble"]),
            "allow_soft": False,
        },
        {
            "name": "cross_team_doc6",
            "labels": _present(
                ["codex_v3_1", "codex_v3_2", "claude_v3_lgb", "claude_v3_xgb", "claude_v3_cat", "claude_v3_ensemble"]
            ),
            "allow_soft": False,
        },
        {
            "name": "cross_team_doc7",
            "labels": _present(
                [
                    "codex_v3_1",
                    "codex_v3_2",
                    "codex_v3_3",
                    "claude_v3_lgb",
                    "claude_v3_xgb",
                    "claude_v3_cat",
                    "claude_v3_ensemble",
                ]
            ),
            "allow_soft": False,
        },
        {
            "name": "top4_by_single_auc",
            "labels": candidate_labels[:4],
            "allow_soft": True,
        },
        {
            "name": "top6_by_single_auc",
            "labels": candidate_labels[:6],
            "allow_soft": False,
        },
    ]
    return [strategy for strategy in strategies if len(strategy["labels"]) >= 2]


def _submission_from_labels(
    merged_submission: pd.DataFrame,
    labels: list[str],
    *,
    mode: str,
    weights: dict[str, float] | None = None,
) -> pd.Series:
    if mode == "rank_average":
        return merged_submission[labels].rank(pct=True).mean(axis=1)
    weights = weights or {}
    return sum(weights.get(label, 0.0) * merged_submission[label] for label in labels)


def run_v5_ensemble(
    *,
    v4_summary_path: Path = DEFAULT_V4_SUMMARY_PATH,
    v3_summary_path: Path = DEFAULT_V3_SUMMARY_PATH,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    data_dir: Path | None = None,
    external_root: Path = DEFAULT_EXTERNAL_ROOT,
) -> dict[str, Any]:
    resolved_data_dir = data_dir or detect_default_data_dir()
    train_path = resolved_data_dir / TRAIN_FILE
    submission_template_path = resolved_data_dir / SUBMISSION_FILE

    train_frame = pd.read_csv(train_path, encoding="utf-8-sig")
    submission_template = pd.read_csv(submission_template_path, encoding="utf-8-sig")
    train_ids = train_frame[ID_COLUMN]
    y_true = train_frame[TARGET_COLUMN].astype(int)
    submission_ids = submission_template[ID_COLUMN]

    candidates = _default_candidates(
        v3_summary_path=v3_summary_path,
        registry_path=registry_path,
        external_root=external_root,
    )
    if len(candidates) < 2:
        raise RuntimeError("V5 ensemble 생성 실패: 후보 모델이 2개 미만입니다.")

    merged_oof, merged_submission = _merge_candidate_predictions(
        candidates=candidates,
        train_ids=train_ids,
        y_true=y_true,
        submission_ids=submission_ids,
    )

    candidate_stats: list[dict[str, Any]] = []
    for spec in candidates:
        auc = float(roc_auc_score(y_true, merged_oof[spec.label]))
        candidate_stats.append(
            {
                "label": spec.label,
                "model_name": spec.model_name,
                "exp_id": spec.exp_id,
                "source_type": spec.source_type,
                "overall_auc": round(auc, 6),
                "notes": spec.notes,
            }
        )
    candidate_stats.sort(key=lambda item: item["overall_auc"], reverse=True)
    ordered_labels = [item["label"] for item in candidate_stats]

    leaderboard_rows: list[dict[str, Any]] = []
    best_record: dict[str, Any] | None = None

    def register_result(name: str, labels: list[str], mode: str, auc: float, weights: dict[str, float] | None) -> None:
        nonlocal best_record
        record = {
            "strategy_name": name,
            "labels": labels,
            "mode": mode,
            "overall_auc": round(float(auc), 6),
            "weights": weights or {},
        }
        leaderboard_rows.append(record)
        if best_record is None:
            best_record = record
            return
        current_best_auc = float(best_record["overall_auc"])
        if float(record["overall_auc"]) > current_best_auc:
            best_record = record
            return
        if float(record["overall_auc"]) == current_best_auc:
            current_v3_weight = float(record["weights"].get("codex_v3_1", 0.0))
            best_v3_weight = float(best_record["weights"].get("codex_v3_1", 0.0))
            if current_v3_weight > best_v3_weight:
                best_record = record

    for strategy in _curated_strategy_sets(ordered_labels):
        labels = strategy["labels"]
        rank_auc, _ = _evaluate_rank_average(merged_oof, labels)
        register_result(strategy["name"] + "_rank_average", labels, "rank_average", rank_auc, None)
        if strategy.get("allow_soft") and len(labels) <= 4:
            soft_auc, soft_weights, _ = _evaluate_soft_vote(merged_oof, labels, step=0.1)
            register_result(strategy["name"] + "_soft_vote", labels, "soft_vote", soft_auc, soft_weights)

    coarse_soft_top = sorted(
        [row for row in leaderboard_rows if row["mode"] == "soft_vote"],
        key=lambda row: row["overall_auc"],
        reverse=True,
    )[:1]
    for row in coarse_soft_top:
        fine_auc, fine_weights, _ = _evaluate_soft_vote(merged_oof, list(row["labels"]), step=0.05)
        register_result(row["strategy_name"] + "_fine", list(row["labels"]), "soft_vote", fine_auc, fine_weights)

    if best_record is None:
        raise RuntimeError("V5 ensemble 생성 실패: 유효한 전략이 없습니다.")

    leaderboard_frame = pd.DataFrame(leaderboard_rows).sort_values("overall_auc", ascending=False).reset_index(drop=True)
    exp_id, output_dir = create_experiment_dir(output_root, "v5_best_ensemble")
    output_dir.mkdir(parents=True, exist_ok=True)

    best_labels = list(best_record["labels"])
    best_mode = str(best_record["mode"])
    best_weights = dict(best_record["weights"])
    best_submission_prediction = _submission_from_labels(
        merged_submission,
        best_labels,
        mode=best_mode,
        weights=best_weights,
    )

    submission_frame = submission_template.copy()
    submission_frame["probability"] = best_submission_prediction
    submission_frame.to_csv(output_dir / "submission_ensemble.csv", index=False, encoding="utf-8-sig")

    leaderboard_frame.to_csv(output_dir / "strategy_leaderboard.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(candidate_stats).to_csv(output_dir / "candidate_leaderboard.csv", index=False, encoding="utf-8-sig")
    merged_oof[best_labels + [TARGET_COLUMN]].corr(numeric_only=True).to_csv(
        output_dir / "candidate_correlation.csv",
        encoding="utf-8-sig",
    )

    summary = {
        "exp_id": exp_id,
        "model_name": "ensemble",
        "overall_auc": round(float(best_record["overall_auc"]), 6),
        "ensemble_mode": best_mode,
        "candidate_labels": best_labels,
        "weights": best_weights,
        "strategy_name": best_record["strategy_name"],
        "candidate_stats": candidate_stats,
        "top_strategies": leaderboard_frame.head(10).to_dict(orient="records"),
        "output_dir": str(output_dir),
        "source_summary": {
            "v3_summary_path": str(v3_summary_path),
            "v4_summary_path": str(v4_summary_path),
            "external_root": str(external_root),
        },
    }
    write_json(output_dir / "run_summary.json", summary)

    append_registry(
        registry_path,
        {
            "timestamp": exp_id.split("_", 1)[0],
            "exp_id": exp_id,
            "model_name": "ensemble",
            "feature_version": best_mode,
            "train_rows": len(merged_oof),
            "test_rows": len(merged_submission),
            "feature_count": len(best_labels),
            "categorical_count": 0,
            "folds": 0,
            "seeds": "",
            "overall_auc": round(float(best_record["overall_auc"]), 6),
            "seed_auc_mean": round(float(best_record["overall_auc"]), 6),
            "seed_auc_std": 0.0,
            "params": {
                "strategy_name": best_record["strategy_name"],
                "candidate_labels": best_labels,
                "weights": best_weights,
                "top_strategies": leaderboard_frame.head(5).to_dict(orient="records"),
            },
            "feature_flags": {},
            "feature_report": {
                "candidate_sources": {spec.label: spec.source_type for spec in candidates},
                "candidate_exp_ids": {spec.label: spec.exp_id for spec in candidates},
            },
            "notes": "V5 expanded ensemble with external Step09 candidates",
            "output_dir": str(output_dir),
        },
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v4-summary-path", type=Path, default=DEFAULT_V4_SUMMARY_PATH)
    parser.add_argument("--v3-summary-path", type=Path, default=DEFAULT_V3_SUMMARY_PATH)
    parser.add_argument("--registry-path", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--data-dir", type=Path, default=detect_default_data_dir())
    parser.add_argument("--external-root", type=Path, default=DEFAULT_EXTERNAL_ROOT)
    args = parser.parse_args()

    summary = run_v5_ensemble(
        v4_summary_path=args.v4_summary_path,
        v3_summary_path=args.v3_summary_path,
        registry_path=args.registry_path,
        output_root=args.output_root,
        data_dir=args.data_dir,
        external_root=args.external_root,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
