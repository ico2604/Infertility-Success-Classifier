#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v15 - v14 기반 + EDA 기반 피처 강화 + Optuna 하이퍼파라미터 탐색 + CatBoost 5-seed
=================================================================================
핵심 변경:
  1. EDA 기반 새 피처:
     - purpose_current: 배아 생성 이유에 '현재 시술용' 포함 여부 (성공률 27.4% vs 0.1%)
     - infertility_count: 불임 원인 개수 (0개 23.5% → 1개 27.3%)
     - has_infertility: 불임 원인 1개 이상 (성공률 +3.8%p)
     - donor_egg_flag: 기증 난자 사용 (성공률 31.5% vs 25.8%)
     - egg_sperm_source_combo: 난자×정자 출처 조합 (수치 인코딩)
     - embryo_per_egg: 난자당 배아 생성 수 (수정 효율 지표)
     - mixed_egg_ratio: 혼합 난자 비율
     - fresh_egg_dominance: 신선 난자 지배 비율
     - transfer_day_squared: 배아 이식 경과일 제곱 (비선형)
     - first_treatment: 첫 시술 여부 (0회: 29.1% vs 6회+: 20.3%)
  2. Optuna로 CatBoost 하이퍼파라미터 최적화 (3-fold, 20 trials)
  3. 최적 파라미터로 CatBoost 5-seed 앙상블
"""

import os
import sys
import time
import datetime
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    log_loss,
    average_precision_score,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
)
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

# ============================================================
# 경로 설정
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
RESULT_DIR = os.path.join(BASE_DIR, "result")
os.makedirs(RESULT_DIR, exist_ok=True)

VERSION = "v15"
NOW = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_PATH = os.path.join(RESULT_DIR, f"log_{VERSION}.md")

SEEDS = [42, 2026, 2604, 123, 777]
N_FOLDS = 5
OPTUNA_TRIALS = 20
OPTUNA_FOLDS = 3

log_lines = []


def log(msg=""):
    print(msg)
    log_lines.append(msg)


def save_log():
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))


# ============================================================
log(f"# {VERSION} - EDA 기반 피처 강화 + Optuna HP 탐색 + CatBoost 5-seed")
log(f"시각: {NOW}")
log("=" * 60)

# ============================================================
# [1] 데이터 로드
# ============================================================
log("\n## [1] 데이터 로드")
t0 = time.time()

train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
sub = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

target_col = "임신 성공 여부"
y = train[target_col].values

log(f"- train: {train.shape}, test: {test.shape}")
log(f"- 타겟: 0={np.sum(y==0)}, 1={np.sum(y==1)}, 양성비율={np.mean(y)*100:.1f}%")

# ============================================================
# [2] 전처리 + 피처 엔지니어링
# ============================================================
log("\n## [2] CatBoost 전처리 + EDA 기반 피처 엔지니어링")


def preprocess_and_engineer(df, is_train=True):
    """CatBoost 네이티브 카테고리 유지 + v14 피처 + v15 EDA 기반 피처"""
    df = df.copy()

    # ID, 타겟 제거
    drop_cols = [df.columns[0]]
    if is_train and target_col in df.columns:
        drop_cols.append(target_col)
    df = df.drop(columns=drop_cols, errors="ignore")

    # 카테고리 / 수치 분리
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    num_cols = df.select_dtypes(exclude=["object"]).columns.tolist()

    # 카테고리: NaN → "Unknown"
    for c in cat_cols:
        df[c] = df[c].fillna("Unknown").astype(str)

    # 수치: NaN → 0
    for c in num_cols:
        df[c] = df[c].fillna(0)

    # ---- 기존 파생변수 (v8/v9/v14) ----
    embryo_col = "이식된 배아 수"
    total_col = "총 생성 배아 수"
    fresh_egg_col = "수집된 신선 난자 수"
    stored_col = "저장된 배아 수"
    mixed_col = "혼합된 난자 수"
    partner_col = "파트너 정자와 혼합된 난자 수"
    micro_col = "미세주입된 난자 수"
    micro_embryo_col = "미세주입에서 생성된 배아 수"
    micro_transfer_col = "미세주입 배아 이식 수"
    transfer_day_col = "배아 이식 경과일"
    type_col = "시술 유형"
    single_col = "단일 배아 이식 여부"
    fresh_use_col = "신선 배아 사용 여부"
    frozen_use_col = "동결 배아 사용 여부"
    thawed_col = "해동된 배아 수"
    purpose_col = "배아 생성 주요 이유"
    egg_source_col = "난자 출처"
    sperm_source_col = "정자 출처"
    age_col = "시술 당시 나이"
    total_proc_col = "총 시술 횟수"

    # 실제이식여부
    if embryo_col in df.columns:
        df["실제이식여부"] = (df[embryo_col] > 0).astype(int)

    # total_embryo_ratio
    if embryo_col in df.columns and total_col in df.columns:
        df["total_embryo_ratio"] = df[embryo_col] / (df[total_col] + 1)

    # 수정 성공률
    if total_col in df.columns and fresh_egg_col in df.columns:
        df["수정_성공률"] = df[total_col] / (df[fresh_egg_col] + 1)

    # 배아 이용률
    if embryo_col in df.columns and total_col in df.columns:
        df["배아_이용률"] = df[embryo_col] / (df[total_col] + 1)

    # 배아 잉여율
    if stored_col in df.columns and total_col in df.columns:
        df["배아_잉여율"] = df[stored_col] / (df[total_col] + 1)

    # 난자_배아_전환율
    if (
        total_col in df.columns
        and fresh_egg_col in df.columns
        and mixed_col in df.columns
    ):
        df["난자_배아_전환율"] = df[total_col] / (df[fresh_egg_col] + df[mixed_col] + 1)

    # IVF/DI 분기 피처
    if type_col in df.columns:
        df["is_ivf"] = (df[type_col] == "IVF").astype(int)
        df["is_di"] = (df[type_col] == "DI").astype(int)
        ivf_mask = df["is_ivf"] == 1

        if embryo_col in df.columns and total_col in df.columns:
            df["ivf_transfer_ratio"] = 0.0
            df.loc[ivf_mask, "ivf_transfer_ratio"] = df.loc[ivf_mask, embryo_col] / (
                df.loc[ivf_mask, total_col] + 1
            )

        if stored_col in df.columns and total_col in df.columns:
            df["ivf_storage_ratio"] = 0.0
            df.loc[ivf_mask, "ivf_storage_ratio"] = df.loc[ivf_mask, stored_col] / (
                df.loc[ivf_mask, total_col] + 1
            )

    # ---- v14 이식 전용 피처 ----
    if transfer_day_col in df.columns:
        df["transfer_day_optimal"] = df[transfer_day_col].apply(
            lambda x: 1 if 2 <= x <= 5 else 0
        )
        day_map = {0: 0, 1: 1, 2: 2, 3: 2, 4: 3, 5: 3, 6: 4, 7: 4}
        df["transfer_day_cat"] = df[transfer_day_col].map(day_map).fillna(0).astype(int)

    if embryo_col in df.columns and transfer_day_col in df.columns:
        df["embryo_day_interaction"] = df[embryo_col] * df[transfer_day_col]

    if embryo_col in df.columns and fresh_egg_col in df.columns:
        df["fresh_transfer_ratio"] = df[embryo_col] / (df[fresh_egg_col] + 1)

    if micro_transfer_col in df.columns and embryo_col in df.columns:
        df["micro_transfer_quality"] = df[micro_transfer_col] / (df[embryo_col] + 0.001)
        df.loc[df[embryo_col] == 0, "micro_transfer_quality"] = 0

    if single_col in df.columns and total_col in df.columns:
        df["single_good_embryo"] = (
            (df[single_col] == 1) & (df[total_col] >= 3)
        ).astype(int)

    if thawed_col in df.columns and embryo_col in df.columns:
        df["frozen_embryo_signal"] = df[thawed_col] * df[embryo_col]

    if stored_col in df.columns and embryo_col in df.columns:
        df["embryo_surplus_after_transfer"] = df[stored_col]

    if embryo_col in df.columns and total_col in df.columns:
        df["transfer_intensity"] = (
            df[embryo_col] / (df[total_col] + 1) * (df[total_col] > 0).astype(int)
        )

    # # ---- v15 EDA 기반 신규 피처 ----

    # # 1) purpose_current: 배아 생성 이유에 '현재 시술용' 포함 (성공률 27.4% vs 0.1%)
    # if purpose_col in df.columns:
    #     df["purpose_current"] = (
    #         df[purpose_col].str.contains("현재 시술용", na=False).astype(int)
    #     )

    # # 2) 불임 원인 개수 (0개: 23.5%, 1개: 27.3%, 2개: 27.9%)
    # infertility_cols = [c for c in df.columns if c.startswith("불임 원인 -")]
    # if len(infertility_cols) > 0:
    #     df["infertility_count"] = df[infertility_cols].sum(axis=1)
    #     df["has_infertility"] = (df["infertility_count"] > 0).astype(int)

    # # 3) 기증 난자 플래그 (성공률 31.5% vs 25.8%)
    # if egg_source_col in df.columns:
    #     df["donor_egg_flag"] = (df[egg_source_col] == "기증 제공").astype(int)

    # # 4) 난자×정자 출처 조합 (수치 인코딩 – 행 단위, train 통계 미사용)
    # if egg_source_col in df.columns and sperm_source_col in df.columns:
    #     egg_map = {"본인 제공": 0, "기증 제공": 1, "알 수 없음": 2, "Unknown": 2}
    #     sperm_map = {
    #         "배우자 제공": 0,
    #         "기증 제공": 1,
    #         "미할당": 2,
    #         "배우자 및 기증 제공": 3,
    #         "Unknown": 2,
    #     }
    #     df["egg_source_num"] = df[egg_source_col].map(egg_map).fillna(2).astype(int)
    #     df["sperm_source_num"] = (
    #         df[sperm_source_col].map(sperm_map).fillna(2).astype(int)
    #     )
    #     df["egg_sperm_combo"] = df["egg_source_num"] * 10 + df["sperm_source_num"]
    #     df = df.drop(columns=["egg_source_num", "sperm_source_num"])

    # # 5) 난자당 배아 생성 수 (수정 효율)
    # if total_col in df.columns and mixed_col in df.columns:
    #     df["embryo_per_egg"] = df[total_col] / (df[mixed_col] + 1)

    # # 6) 혼합 난자 비율
    # if mixed_col in df.columns and fresh_egg_col in df.columns:
    #     df["mixed_egg_ratio"] = df[mixed_col] / (df[fresh_egg_col] + 1)

    # # 7) 신선 난자 지배 비율 (partner / total mixed)
    # if partner_col in df.columns and mixed_col in df.columns:
    #     df["fresh_egg_dominance"] = df[partner_col] / (df[mixed_col] + 1)

    # # 8) 배아 이식 경과일 제곱 (비선형 효과)
    # if transfer_day_col in df.columns:
    #     df["transfer_day_sq"] = df[transfer_day_col] ** 2

    # # 9) 첫 시술 여부 (0회: 29.1%, 높은 성공률)
    # if total_proc_col in df.columns:
    #     proc_map = {
    #         "0회": 0,
    #         "1회": 1,
    #         "2회": 2,
    #         "3회": 3,
    #         "4회": 4,
    #         "5회": 5,
    #         "6회 이상": 6,
    #     }
    #     proc_num = df[total_proc_col].map(proc_map)
    #     if proc_num.isna().all():
    #         # 이미 숫자형일 수 있음
    #         proc_num = pd.to_numeric(df[total_proc_col], errors="coerce").fillna(0)
    #     else:
    #         proc_num = proc_num.fillna(0)
    #     df["first_treatment"] = (proc_num == 0).astype(int)

    # # 10) 나이×이식 배아 상호작용
    # if age_col in df.columns and embryo_col in df.columns:
    #     age_mid_map = {
    #         "만18-34세": 26,
    #         "만35-37세": 36,
    #         "만38-39세": 38.5,
    #         "만40-42세": 41,
    #         "만43-44세": 43.5,
    #         "만45-50세": 47.5,
    #         "알 수 없음": 35,
    #     }
    #     age_num = df[age_col].map(age_mid_map)
    #     if age_num.isna().all():
    #         age_num = pd.to_numeric(df[age_col], errors="coerce").fillna(35)
    #     else:
    #         age_num = age_num.fillna(35)
    #     df["age_transfer_interaction"] = age_num * df[embryo_col]
    #     df["age_mid"] = age_num

    # # 카테고리 재확인
    # cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    return df, cat_cols


log("  전처리 중...")
X_train_cb, cat_features_names = preprocess_and_engineer(train, is_train=True)
X_test_cb, _ = preprocess_and_engineer(test, is_train=True)

cat_features_idx = [X_train_cb.columns.get_loc(c) for c in cat_features_names]
feature_names_cb = X_train_cb.columns.tolist()

log(f"- CB 피처: {len(feature_names_cb)}개, 카테고리: {len(cat_features_idx)}개")
log(f"- 카테고리: {cat_features_names}")

# # v15 신규 피처 식별
# v15_new = [
#     "purpose_current",
#     "infertility_count",
#     "has_infertility",
#     "donor_egg_flag",
#     "egg_sperm_combo",
#     "embryo_per_egg",
#     "mixed_egg_ratio",
#     "fresh_egg_dominance",
#     "transfer_day_sq",
#     "first_treatment",
#     "age_mid",
# ]
# v15_found = [f for f in v15_new if f in feature_names_cb]
# log(f"- v15 신규 피처 ({len(v15_found)}개): {v15_found}")

# # 신규 피처 통계
# log(f"\n### v15 신규 피처 통계 (train)")
# for f in v15_found:
#     col = X_train_cb[f]
#     nz = (col != 0).sum()
#     log(
#         f"  {f}: mean={col.mean():.4f}, std={col.std():.4f}, nonzero={nz} ({nz/len(col)*100:.1f}%)"
#     )

# 이식/비이식 그룹 확인
embryo_raw = train["이식된 배아 수"].fillna(0).values
train_implant_mask = embryo_raw > 0
log(f"\n### 그룹 확인")
log(
    f"- 이식: {train_implant_mask.sum()}건 ({train_implant_mask.mean()*100:.1f}%), 양성률={y[train_implant_mask].mean()*100:.1f}%"
)
log(
    f"- 비이식: {(~train_implant_mask).sum()}건, 양성률={y[~train_implant_mask].mean()*100:.2f}%"
)

# ============================================================
# [3] Optuna 하이퍼파라미터 탐색
# ============================================================
log(
    f"\n## [3] Optuna 하이퍼파라미터 탐색 ({OPTUNA_TRIALS} trials, {OPTUNA_FOLDS}-fold)"
)

import optuna
from optuna.samplers import TPESampler

optuna.logging.set_verbosity(optuna.logging.WARNING)


def objective(trial):
    params = {
        "iterations": 5000,
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
        "depth": trial.suggest_int("depth", 5, 9),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.5, 10.0, log=True),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 50),
        "random_strength": trial.suggest_float("random_strength", 0.5, 2.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.5),
        "border_count": trial.suggest_categorical("border_count", [64, 128, 254]),
        "eval_metric": "AUC",
        "random_seed": 42,
        "early_stopping_rounds": 200,
        "verbose": 0,
        "task_type": "GPU",
        "devices": "0",
    }

    skf = StratifiedKFold(n_splits=OPTUNA_FOLDS, shuffle=True, random_state=42)
    auc_scores = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train_cb, y)):
        X_tr = X_train_cb.iloc[tr_idx]
        X_va = X_train_cb.iloc[va_idx]
        y_tr = y[tr_idx]
        y_va = y[va_idx]

        model = CatBoostClassifier(**params)
        model.fit(
            X_tr, y_tr, eval_set=(X_va, y_va), cat_features=cat_features_idx, verbose=0
        )

        va_pred = model.predict_proba(X_va)[:, 1]
        auc_scores.append(roc_auc_score(y_va, va_pred))

    return np.mean(auc_scores)


t_optuna = time.time()
study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))

log(f"  Optuna 탐색 시작...")


# 진행률 콜백
class OptunaCallback:
    def __init__(self, n_trials):
        self.n_trials = n_trials
        self.start_time = time.time()

    def __call__(self, study, trial):
        elapsed = (time.time() - self.start_time) / 60
        remaining = elapsed / (trial.number + 1) * (self.n_trials - trial.number - 1)
        log(
            f"  [{trial.number+1}/{self.n_trials}] AUC={trial.value:.6f} "
            f"(best={study.best_value:.6f}) "
            f"소요={elapsed:.1f}분, 남은≈{remaining:.1f}분"
        )
        save_log()  # 중간 저장


callback = OptunaCallback(OPTUNA_TRIALS)
study.optimize(objective, n_trials=OPTUNA_TRIALS, callbacks=[callback])

optuna_time = (time.time() - t_optuna) / 60
best_params = study.best_params
best_optuna_auc = study.best_value

log(f"\n### Optuna 결과 (소요: {optuna_time:.1f}분)")
log(f"  Best AUC ({OPTUNA_FOLDS}-fold): {best_optuna_auc:.6f}")
log(f"  Best params:")
for k, v in best_params.items():
    log(f"    {k}: {v}")

# 상위 5개 trial
log(f"\n### 상위 5개 Trial")
trials_sorted = sorted(
    study.trials, key=lambda t: t.value if t.value is not None else 0, reverse=True
)
for i, trial in enumerate(trials_sorted[:5]):
    log(
        f"  {i+1}. AUC={trial.value:.6f} | "
        f"depth={trial.params['depth']}, lr={trial.params['learning_rate']:.4f}, "
        f"l2={trial.params['l2_leaf_reg']:.2f}, min_leaf={trial.params['min_data_in_leaf']}"
    )

# ============================================================
# [4] CatBoost 5-seed 앙상블 (Optuna 최적 파라미터)
# ============================================================
log(f"\n## [4] CatBoost 5-seed 앙상블 (Optuna 최적 파라미터)")
log(f"### Seeds: {SEEDS}")
log(f"### 파라미터: {best_params}")

oof_cb = np.zeros(len(y))
test_cb = np.zeros(len(test))
seed_cb_aucs = []
cb_importances = None

total_models = len(SEEDS) * N_FOLDS
model_count = 0
t_train = time.time()

for si, seed_val in enumerate(SEEDS):
    log(f"\n  --- Seed {seed_val} ({si+1}/{len(SEEDS)}) ---")
    oof_seed = np.zeros(len(y))
    test_seed = np.zeros(len(test))

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed_val)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train_cb, y)):
        model_count += 1
        t_fold = time.time()

        X_tr = X_train_cb.iloc[tr_idx]
        X_va = X_train_cb.iloc[va_idx]
        y_tr = y[tr_idx]
        y_va = y[va_idx]

        cb_model = CatBoostClassifier(
            iterations=5000,
            learning_rate=best_params["learning_rate"],
            depth=best_params["depth"],
            l2_leaf_reg=best_params["l2_leaf_reg"],
            min_data_in_leaf=best_params["min_data_in_leaf"],
            random_strength=best_params["random_strength"],
            bagging_temperature=best_params["bagging_temperature"],
            border_count=best_params["border_count"],
            eval_metric="AUC",
            random_seed=seed_val + fold,
            early_stopping_rounds=200,
            verbose=0,
            task_type="GPU",
            devices="0",
        )

        cb_model.fit(
            X_tr,
            y_tr,
            eval_set=(X_va, y_va),
            cat_features=cat_features_idx,
            verbose=0,
        )

        va_pred = cb_model.predict_proba(X_va)[:, 1]
        oof_seed[va_idx] = va_pred
        test_seed += cb_model.predict_proba(X_test_cb)[:, 1] / N_FOLDS

        va_auc = roc_auc_score(y_va, va_pred)
        elapsed = (time.time() - t_fold) / 60
        total_elapsed = (time.time() - t_train) / 60

        log(
            f"    Fold {fold+1}: AUC={va_auc:.4f}, iter={cb_model.best_iteration_}, "
            f"소요={elapsed:.1f}분 [{model_count}/{total_models}, 총={total_elapsed:.1f}분]"
        )

        # 마지막 시드 마지막 폴드에서 중요도
        if si == len(SEEDS) - 1 and fold == N_FOLDS - 1:
            cb_importances = cb_model.get_feature_importance()

    seed_auc = roc_auc_score(y, oof_seed)
    seed_cb_aucs.append(seed_auc)
    log(f"  Seed {seed_val} OOF AUC: {seed_auc:.4f}")

    oof_cb += oof_seed / len(SEEDS)
    test_cb += test_seed / len(SEEDS)

train_time = (time.time() - t_train) / 60
cb_auc = roc_auc_score(y, oof_cb)
log(f"\n  === CatBoost 5-seed OOF AUC: {cb_auc:.6f} ===")
log(f"  개별 seed: {[f'{a:.4f}' for a in seed_cb_aucs]}")
log(f"  학습 소요: {train_time:.1f}분")

# ============================================================
# [5] 그룹별 AUC 분석
# ============================================================
log("\n## [5] 그룹별 AUC 분석")

implant_auc = roc_auc_score(y[train_implant_mask], oof_cb[train_implant_mask])
non_implant_mask = ~train_implant_mask
if y[non_implant_mask].sum() > 0:
    non_implant_auc = roc_auc_score(y[non_implant_mask], oof_cb[non_implant_mask])
else:
    non_implant_auc = float("nan")

log(f"- 이식 그룹 ({train_implant_mask.sum()}건): AUC={implant_auc:.6f}")
log(f"- 비이식 그룹 ({non_implant_mask.sum()}건): AUC={non_implant_auc:.6f}")
log(f"- 전체: AUC={cb_auc:.6f}")
log(
    f"- v14 이식 그룹 AUC: 0.6746, v15: {implant_auc:.4f} (Δ={implant_auc-0.6746:+.4f})"
)

# ============================================================
# [6] 종합 평가지표
# ============================================================
log("\n## [6] 종합 평가지표")

oof_ll = log_loss(y, oof_cb)
oof_ap = average_precision_score(y, oof_cb)

log(f"\n### 확률 기반 지표")
log(f"  OOF AUC:      {cb_auc:.6f}")
log(f"  OOF Log Loss: {oof_ll:.6f}")
log(f"  OOF AP:       {oof_ap:.6f}")

log(f"\n### Threshold별 분류 지표")
log(
    f"  {'Threshold':>10}  {'Accuracy':>10}  {'Precision':>10}  {'Recall':>10}  {'F1':>10}  {'Specificity':>12}"
)
log(f"  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*12}")

best_f1 = 0
best_f1_th = 0.25
for th in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
    y_pred_bin = (oof_cb >= th).astype(int)
    acc = accuracy_score(y, y_pred_bin)
    prec = precision_score(y, y_pred_bin, zero_division=0)
    rec = recall_score(y, y_pred_bin, zero_division=0)
    f1 = f1_score(y, y_pred_bin, zero_division=0)
    spec = np.sum((y == 0) & (y_pred_bin == 0)) / np.sum(y == 0)
    log(
        f"  {th:>10.2f}  {acc:>10.4f}  {prec:>10.4f}  {rec:>10.4f}  {f1:>10.4f}  {spec:>12.4f}"
    )
    if f1 > best_f1:
        best_f1 = f1
        best_f1_th = th

log(f"\n  최적 F1: {best_f1:.4f} (threshold={best_f1_th})")

# ============================================================
# [7] 제출 파일 생성
# ============================================================
log("\n## [7] 제출 파일 생성")

sub_main = sub.copy()
sub_main["probability"] = test_cb
main_path = os.path.join(RESULT_DIR, f"sample_submission_{VERSION}_{NOW}.csv")
sub_main.to_csv(main_path, index=False)

log(f"- 파일: {main_path}")
log(
    f"- 확률: mean={test_cb.mean():.4f}, std={test_cb.std():.4f}, "
    f"min={test_cb.min():.6f}, max={test_cb.max():.4f}"
)

log(f"\n- 예시:")
log(f"  {'ID':>10}  {'probability':>12}")
for i in range(5):
    log(f"  {sub_main.iloc[i, 0]:>10}  {test_cb[i]:>12.6f}")

# ============================================================
# [8] 피처 중요도
# ============================================================
log("\n## [8] CatBoost 피처 중요도 (상위 30)")

if cb_importances is not None:
    imp_df = pd.DataFrame(
        {"feature": feature_names_cb, "importance": cb_importances}
    ).sort_values("importance", ascending=False)

    cat_set = set(cat_features_names)

    v15_new_names = [
        "purpose_current",
        "infertility_count",
        "has_infertility",
        "donor_egg_flag",
        "egg_sperm_combo",
        "embryo_per_egg",
        "mixed_egg_ratio",
        "fresh_egg_dominance",
        "transfer_day_sq",
        "first_treatment",
        "age_mid",
    ]
    v15_set = set([f for f in v15_new_names if f in feature_names_cb])

    v14_features = [
        "transfer_day_optimal",
        "transfer_day_cat",
        "embryo_day_interaction",
        "fresh_transfer_ratio",
        "micro_transfer_quality",
        "single_good_embryo",
        "frozen_embryo_signal",
        "embryo_surplus_after_transfer",
        "transfer_intensity",
        "age_transfer_interaction",
    ]

    for rank_i, (_, row) in enumerate(imp_df.head(30).iterrows(), 1):
        marks = ""
        if row["feature"] in cat_set:
            marks += " [cat]"
        if row["feature"] in v15_set:
            marks += " ★v15"
        elif row["feature"] in v14_features:
            marks += " ★v14"
        log(f"  {rank_i}. {row['feature']}{marks}: {row['importance']:.2f}")

    # v15 신규 피처 중요도
    log(f"\n### v15 신규 피처 중요도")
    for f in sorted(v15_set):
        if f in imp_df["feature"].values:
            imp_val = imp_df[imp_df["feature"] == f]["importance"].values[0]
            rank = list(imp_df["feature"].values).index(f) + 1
            log(f"  {f}: {imp_val:.2f} (전체 {rank}위)")


# ============================================================
# [9] 버전 비교
# ============================================================
log("\n## [9] 버전 비교")
log("| 버전 | 모델 | OOF AUC | 이식그룹AUC | 비고 |")
log("|------|------|---------|------------|------|")
log("| v1 | CB원본 | 0.7403 | - | 베이스라인 (67피처) |")
log("| v4 | XGB | 0.7392 | - | 컬럼복원 |")
log("| v7 | XGB+CB | 0.7402 | - | 블렌딩 |")
log("| v8 | XGB+CB | 0.7401 | - | +파생변수 |")
log("| v9 | XGB+CB 3seed | 0.7405 | - | +IVF/DI분기 |")
log("| v11 | CB+XGB cross | 0.7405 | - | 크로스블렌딩 |")
log("| v12 | CB+XGB+TabNet | 0.7406 | - | 3모델블렌딩 |")
log("| v13 | CB 3seed | 0.7405 | 0.6745 | 클리핑실패 |")
log("| v14 | CB 3seed | 0.7406 | 0.6746 | 이식전용피처 |")
log(
    f"| **v15** | **CB 5seed+Optuna** | **{cb_auc:.4f}** | **{implant_auc:.4f}** | **EDA피처+HP탐색** |"
)

# ============================================================
# [10] 최종 요약
# ============================================================
total_time = time.time() - t0
log(f"\n{'='*60}")
log(f"## 최종 요약")
log(f"{'='*60}")
log(f"- v15 핵심 변경:")
log(f"  1. EDA 기반 신규 피처 {len(v15_found)}개 추가")
log(f"  2. Optuna HP 탐색 ({OPTUNA_TRIALS} trials, {OPTUNA_FOLDS}-fold)")
log(f"  3. CatBoost 5-seed 앙상블 (3→5 seed)")
log(f"- Optuna 최적 파라미터:")
for k, v in best_params.items():
    log(f"    {k}: {v}")
log(f"- Optuna best {OPTUNA_FOLDS}-fold AUC: {best_optuna_auc:.6f}")
log(f"- CB 5-seed OOF AUC: {cb_auc:.6f}")
log(f"  seed별: {[f'{a:.4f}' for a in seed_cb_aucs]}")
log(f"- 이식 그룹 AUC: {implant_auc:.6f} (v14: 0.6746)")
log(f"- 비이식 그룹 AUC: {non_implant_auc:.6f}")
log(f"- OOF LogLoss: {oof_ll:.6f}")
log(f"- OOF AP: {oof_ap:.6f}")
log(f"- 최적 F1: {best_f1:.4f} (th={best_f1_th})")
log(f"- 피처 수: {len(feature_names_cb)}개 (cat={len(cat_features_idx)})")
log(f"- 제출: {main_path}")
log(f"- 데이터 누수: 없음")
log(f"  - 모든 파생변수: 행 단위 연산 (타겟/test 통계 미사용)")
log(f"  - 범주형 인코딩: CatBoost 네이티브 (별도 인코딩 없음)")
log(f"  - 결측치: 수치→0, 범주→'Unknown' (행 독립)")
log(f"- Optuna 소요: {optuna_time:.1f}분")
log(f"- 학습 소요: {train_time:.1f}분")
log(f"- 총 소요: {total_time/60:.1f}분")
log(f"- 로그: {LOG_PATH}")
log(f"{'='*60}")

save_log()
print(f"\n완료! 로그: {LOG_PATH}")
