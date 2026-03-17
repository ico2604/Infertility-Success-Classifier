#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v16 - Subgroup-aware Multi-model Blend
=======================================
Model A: Full CatBoost (전체 데이터)
Model B: Transfer-only CatBoost (이식 그룹)
Model C: Non-transfer CatBoost (비이식 그룹)
Model D: Transfer-only LightGBM/XGBoost (이식 그룹)
+ subgroup-aware gating blend
"""

import os
import sys
import time
import datetime
import warnings
import logging
import json
import re
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
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

try:
    import lightgbm as lgb

    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    import xgboost as xgb

    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

    def tqdm(iterable, **kwargs):
        return iterable


try:
    import optuna
    from optuna.samplers import TPESampler

    HAS_OPTUNA = True
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    HAS_OPTUNA = False

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
RESULT_DIR = os.path.join(BASE_DIR, "result")
os.makedirs(RESULT_DIR, exist_ok=True)

VERSION = "v16"
NOW = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_PATH = os.path.join(RESULT_DIR, f"log_{VERSION}.md")

CFG = {
    "N_FOLDS": 5,
    "FULL_CB_SEEDS": [42, 2026],
    "TRANSFER_CB_SEEDS": [42],
    "NONTRANSFER_CB_SEEDS": [42],
    "TRANSFER_LGB_SEEDS": [42],
    "USE_GROUP_WEIGHTS": False,
    "TRANSFER_OPTUNA": False,
    "TRANSFER_OPTUNA_TRIALS": 12,
    "TRANSFER_OPTUNA_FOLDS": 3,
    "GPU_AVAILABLE": True,
}

FULL_CB_PARAMS = {
    "iterations": 8000,
    "learning_rate": 0.00837420773913813,
    "depth": 8,
    "l2_leaf_reg": 9.013710971500913,
    "min_data_in_leaf": 47,
    "random_strength": 1.9940254840418206,
    "bagging_temperature": 0.7125808252032847,
    "border_count": 64,
    "eval_metric": "AUC",
    "loss_function": "Logloss",
    "od_type": "Iter",
    "od_wait": 400,
    "verbose": False,
    "allow_writing_files": False,
}

TRANSFER_CB_PARAMS = {
    "iterations": 9000,
    "learning_rate": 0.00837420773913813,
    "depth": 8,
    "l2_leaf_reg": 9.013710971500913,
    "min_data_in_leaf": 47,
    "random_strength": 1.9940254840418206,
    "bagging_temperature": 0.7125808252032847,
    "border_count": 64,
    "eval_metric": "AUC",
    "loss_function": "Logloss",
    "od_type": "Iter",
    "od_wait": 500,
    "verbose": False,
    "allow_writing_files": False,
}

NONTRANSFER_CB_PARAMS = {
    "iterations": 5000,
    "learning_rate": 0.01,
    "depth": 7,
    "l2_leaf_reg": 5.0,
    "min_data_in_leaf": 30,
    "random_strength": 1.5,
    "bagging_temperature": 0.8,
    "border_count": 128,
    "eval_metric": "AUC",
    "loss_function": "Logloss",
    "od_type": "Iter",
    "od_wait": 300,
    "verbose": False,
    "allow_writing_files": False,
}

LGB_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.02,
    "num_leaves": 63,
    "max_depth": 7,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 30,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "verbose": -1,
    "n_jobs": -1,
}

XGB_PARAMS = {
    "n_estimators": 5000,
    "learning_rate": 0.02,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "min_child_weight": 20,
    "eval_metric": "auc",
    "tree_method": "hist",
    "verbosity": 0,
}

CAT_COLS_CANDIDATES = [
    "시술 시기 코드",
    "시술 당시 나이",
    "시술 유형",
    "특정 시술 유형",
    "배란 유도 유형",
    "배아 생성 주요 이유",
    "총 시술 횟수",
    "클리닉 내 총 시술 횟수",
    "IVF 시술 횟수",
    "DI 시술 횟수",
    "총 임신 횟수",
    "IVF 임신 횟수",
    "DI 임신 횟수",
    "총 출산 횟수",
    "IVF 출산 횟수",
    "DI 출산 횟수",
    "난자 출처",
    "정자 출처",
    "난자 기증자 나이",
    "정자 기증자 나이",
]

COUNT_COLS = [
    "총 시술 횟수",
    "클리닉 내 총 시술 횟수",
    "IVF 시술 횟수",
    "DI 시술 횟수",
    "총 임신 횟수",
    "IVF 임신 횟수",
    "DI 임신 횟수",
    "총 출산 횟수",
    "IVF 출산 횟수",
    "DI 출산 횟수",
]

AGE_MID_MAP = {
    "만18-34세": 26,
    "만35-37세": 36,
    "만38-39세": 38.5,
    "만40-42세": 41,
    "만43-44세": 43.5,
    "만45-50세": 47.5,
    "알 수 없음": -1,
}

COUNT_MAP = {"0회": 0, "1회": 1, "2회": 2, "3회": 3, "4회": 4, "5회": 5, "6회 이상": 6}

# ============================================================
# LOGGING
# ============================================================
log_lines: List[str] = []


def log(msg: str = ""):
    print(msg)
    log_lines.append(msg)


def save_log():
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))


# ============================================================
# UTILITIES
# ============================================================
def safe_div(a, b, fill: float = 0.0):
    result = np.where((b > 0) & np.isfinite(b), a / b, fill)
    return np.where(np.isfinite(result), result, fill)


def col_exists(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns


def safe_col(df: pd.DataFrame, col: str, fill=0):
    if col in df.columns:
        return df[col].fillna(fill).values
    return np.full(len(df), fill)


def detect_gpu() -> str:
    if not CFG["GPU_AVAILABLE"]:
        return "CPU"
    try:
        test_model = CatBoostClassifier(
            iterations=1, task_type="GPU", devices="0", verbose=False
        )
        test_model.fit(
            pd.DataFrame({"a": [0, 1, 0, 1], "b": [1, 0, 1, 0]}),
            [0, 1, 0, 1],
            verbose=False,
        )
        return "GPU"
    except Exception:
        return "CPU"


# ============================================================
# DATA LOADING
# ============================================================
def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, str]:
    log("\n## [1] 데이터 로드")
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    sub = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

    target_col = "임신 성공 여부"
    if target_col not in train.columns:
        for c in train.columns:
            if "임신" in c and "성공" in c:
                target_col = c
                break

    y = train[target_col].values
    log(f"- train: {train.shape}, test: {test.shape}")
    log(f"- 타겟: 0={np.sum(y==0)}, 1={np.sum(y==1)}, 양성비율={np.mean(y)*100:.1f}%")
    return train, test, sub, y, target_col


# ============================================================
# FEATURE ENGINEERING
# ============================================================
def make_base_features(
    df: pd.DataFrame, target_col: str, is_train: bool
) -> pd.DataFrame:
    """공통 전처리 + 피처 엔지니어링"""
    df = df.copy()

    # ID/타겟 제거
    id_col = df.columns[0]
    drop = [id_col]
    if is_train and target_col in df.columns:
        drop.append(target_col)
    df = df.drop(columns=drop, errors="ignore")

    # 카테고리 결측 → Unknown
    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = df[c].fillna("Unknown").astype(str)

    # 수치 결측 → 0
    for c in df.select_dtypes(exclude=["object"]).columns:
        df[c] = df[c].fillna(0)

    # ---- 나이 중간값 ----
    age_col = "시술 당시 나이"
    if col_exists(df, age_col):
        df["시술 당시 나이_num"] = df[age_col].map(AGE_MID_MAP)
        if df["시술 당시 나이_num"].isna().mean() > 0.5:
            df["시술 당시 나이_num"] = pd.to_numeric(
                df[age_col], errors="coerce"
            ).fillna(-1)
        else:
            df["시술 당시 나이_num"] = df["시술 당시 나이_num"].fillna(-1)
        df["age_missing"] = (df["시술 당시 나이_num"] < 0).astype(int)
        age_num = df["시술 당시 나이_num"].clip(lower=0)
        df["age_sq"] = age_num**2
    else:
        df["시술 당시 나이_num"] = -1
        df["age_missing"] = 1
        df["age_sq"] = 0

    # ---- 횟수형 매핑 ----
    for cc in COUNT_COLS:
        if col_exists(df, cc):
            mapped = df[cc].map(COUNT_MAP)
            if mapped.isna().mean() > 0.5:
                mapped = pd.to_numeric(df[cc], errors="coerce").fillna(0)
            else:
                mapped = mapped.fillna(0)
            df[f"{cc}_num"] = mapped.astype(float)

    # 횟수형: _num 생성 후 원본 문자열 드랍 (동일 정보 중복 제거)
    drop_count_originals = [
        cc for cc in COUNT_COLS if col_exists(df, cc) and col_exists(df, f"{cc}_num")
    ]
    df = df.drop(columns=drop_count_originals, errors="ignore")

    # ---- 기본 파생 ----
    embryo_col = "이식된 배아 수"
    total_col = "총 생성 배아 수"
    fresh_egg_col = "수집된 신선 난자 수"
    stored_col = "저장된 배아 수"
    mixed_col = "혼합된 난자 수"
    partner_col = "파트너 정자와 혼합된 난자 수"
    micro_col = "미세주입된 난자 수"
    micro_emb_col = "미세주입에서 생성된 배아 수"
    micro_tr_col = "미세주입 배아 이식 수"
    tr_day_col = "배아 이식 경과일"
    type_col = "시술 유형"
    single_col = "단일 배아 이식 여부"
    fresh_use_col = "신선 배아 사용 여부"
    frozen_use_col = "동결 배아 사용 여부"
    thawed_col = "해동된 배아 수"
    purpose_col = "배아 생성 주요 이유"
    egg_src_col = "난자 출처"
    sperm_src_col = "정자 출처"
    spec_type_col = "특정 시술 유형"

    embryo = safe_col(df, embryo_col, 0)
    total_emb = safe_col(df, total_col, 0)
    fresh_egg = safe_col(df, fresh_egg_col, 0)
    stored = safe_col(df, stored_col, 0)
    mixed = safe_col(df, mixed_col, 0)
    tr_day = safe_col(df, tr_day_col, 0)
    thawed = safe_col(df, thawed_col, 0)
    micro_tr = safe_col(df, micro_tr_col, 0)
    micro_emb = safe_col(df, micro_emb_col, 0)
    single = safe_col(df, single_col, 0)
    fresh_use = safe_col(df, fresh_use_col, 0)
    frozen_use = safe_col(df, frozen_use_col, 0)
    age_num_arr = df["시술 당시 나이_num"].clip(lower=0).values

    df["실제이식여부"] = (embryo > 0).astype(int)
    df["is_ivf"] = (
        (df[type_col] == "IVF").astype(int) if col_exists(df, type_col) else 0
    )
    df["is_di"] = (df[type_col] == "DI").astype(int) if col_exists(df, type_col) else 0

    # failure / gap
    tot_proc = safe_col(df, "총 시술 횟수_num", 0)
    tot_preg = safe_col(df, "총 임신 횟수_num", 0)
    ivf_proc = safe_col(df, "IVF 시술 횟수_num", 0)
    ivf_preg = safe_col(df, "IVF 임신 횟수_num", 0)
    tot_birth = safe_col(df, "총 출산 횟수_num", 0)
    ivf_birth = safe_col(df, "IVF 출산 횟수_num", 0)

    df["total_failure_count"] = np.maximum(tot_proc - tot_preg, 0)
    df["ivf_failure_count"] = np.maximum(ivf_proc - ivf_preg, 0)
    df["pregnancy_to_birth_gap"] = np.maximum(tot_preg - tot_birth, 0)
    df["ivf_preg_to_birth_gap"] = np.maximum(ivf_preg - ivf_birth, 0)
    df["repeated_failed_transfer"] = (
        (tot_proc >= 2) & (tot_preg == 0) & (embryo > 0)
    ).astype(int)
    df["first_transfer_cycle"] = ((tot_proc == 0) & (embryo > 0)).astype(int)

    # 효율/비율
    df["수정_성공률"] = safe_div(total_emb, mixed)
    df["배아_이용률"] = safe_div(embryo, total_emb)
    df["배아_잉여율"] = safe_div(stored, total_emb)
    df["난자_배아_전환율"] = safe_div(total_emb, fresh_egg)
    df["transfer_over_generated"] = safe_div(embryo, total_emb)
    df["stored_over_generated"] = safe_div(stored, total_emb)
    df["total_embryo_ratio"] = safe_div(embryo + stored + thawed, total_emb + 1)
    df["embryo_surplus_after_transfer"] = np.maximum(total_emb - embryo, 0)
    df["many_embryos_single_transfer"] = ((total_emb >= 5) & (embryo == 1)).astype(int)

    # 로그/버킷
    df["log_total_embryos"] = np.log1p(np.maximum(total_emb, 0))
    df["log_mixed_oocytes"] = np.log1p(np.maximum(mixed, 0))
    df["log_stored_embryos"] = np.log1p(np.maximum(stored, 0))

    emb_bins = [-1, 0, 1, 2, 10]
    emb_labels = [0, 1, 2, 3]
    _emb_cut = pd.cut(pd.Series(embryo), bins=emb_bins, labels=emb_labels)
    df["transferred_embryos_bucket"] = (
        pd.to_numeric(_emb_cut, errors="coerce").fillna(0).astype(int)
    )

    surplus = np.maximum(total_emb - embryo, 0)
    sur_bins = [-1, 0, 1, 3, 999]
    sur_labels = [0, 1, 2, 3]
    _sur_cut = pd.cut(pd.Series(surplus), bins=sur_bins, labels=sur_labels)
    df["surplus_bucket"] = (
        pd.to_numeric(_sur_cut, errors="coerce").fillna(0).astype(int)
    )

    # IVF 전용
    ivf_mask = df["is_ivf"].values == 1
    df["ivf_transfer_ratio"] = 0.0
    df.loc[ivf_mask, "ivf_transfer_ratio"] = safe_div(
        embryo[ivf_mask], total_emb[ivf_mask]
    )
    df["ivf_storage_ratio"] = 0.0
    df.loc[ivf_mask, "ivf_storage_ratio"] = safe_div(
        stored[ivf_mask], total_emb[ivf_mask]
    )

    # ---- 이식 전용 피처 ----
    df["transfer_day_optimal"] = np.isin(tr_day, [3, 5]).astype(int)
    day_cat_map = {0: 0, 1: 1, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 5}
    df["transfer_day_cat"] = (
        pd.Series(tr_day).map(day_cat_map).fillna(0).astype(int).values
    )
    df["embryo_day_interaction"] = embryo * tr_day
    df["fresh_transfer_ratio"] = safe_div(
        embryo * fresh_use.astype(float), total_emb + 1
    )
    df["micro_transfer_quality"] = safe_div(micro_tr, micro_emb + 1)
    df["single_good_embryo"] = ((embryo == 1) & (tr_day >= 5)).astype(int)
    df["frozen_embryo_signal"] = (thawed > 0).astype(int) + frozen_use.astype(int)
    df["transfer_intensity"] = safe_div(embryo, age_num_arr + 1)
    df["age_transfer_interaction"] = age_num_arr * embryo

    # day5plus interactions
    day5plus = (tr_day >= 5).astype(int)
    df["day5plus"] = day5plus
    df["transfer_day_is_2_3"] = np.isin(tr_day, [2, 3]).astype(int)
    df["transfer_day_is_5_6"] = np.isin(tr_day, [5, 6]).astype(int)
    df["single_x_day5plus"] = ((embryo == 1) & (tr_day >= 5)).astype(int)
    df["multi_x_day5plus"] = ((embryo >= 2) & (tr_day >= 5)).astype(int)
    df["fresh_x_day5plus"] = ((fresh_use > 0) & (tr_day >= 5)).astype(int)
    df["frozen_x_day5plus"] = ((frozen_use > 0) & (tr_day >= 5)).astype(int)
    df["age_x_day5plus"] = age_num_arr * day5plus
    df["age_x_single_transfer"] = age_num_arr * (embryo == 1).astype(int)

    if col_exists(df, egg_src_col):
        df["donor_egg_x_advanced_age"] = (
            (df[egg_src_col] == "기증 제공") & (age_num_arr >= 40)
        ).astype(int)
    else:
        df["donor_egg_x_advanced_age"] = 0

    # ---- 불임 원인 ----
    inf_cols = [c for c in df.columns if c.startswith("불임 원인 -")]
    if inf_cols:
        inf_vals = df[inf_cols].values
        df["infertility_count"] = inf_vals.sum(axis=1)
        df["has_infertility"] = (df["infertility_count"] > 0).astype(int)

    # ---- donor_egg_flag ----
    if col_exists(df, egg_src_col):
        df["donor_egg_flag"] = (df[egg_src_col] == "기증 제공").astype(int)

    # ---- egg_sperm combo ----
    if col_exists(df, egg_src_col) and col_exists(df, sperm_src_col):
        egg_map = {"본인 제공": 0, "기증 제공": 1, "알 수 없음": 2, "Unknown": 2}
        sperm_map = {
            "배우자 제공": 0,
            "기증 제공": 1,
            "미할당": 2,
            "배우자 및 기증 제공": 3,
            "Unknown": 2,
        }
        e_num = df[egg_src_col].map(egg_map).fillna(2).astype(int)
        s_num = df[sperm_src_col].map(sperm_map).fillna(2).astype(int)
        df["egg_sperm_combo"] = e_num * 10 + s_num

    # ---- purpose 토큰 ----
    if col_exists(df, purpose_col):
        p_str = df[purpose_col].fillna("Unknown").astype(str)
        df["purpose_current"] = p_str.str.contains("현재 시술용", na=False).astype(int)
        df["purpose_embryo_storage"] = p_str.str.contains(
            "배아 저장용", na=False
        ).astype(int)
        df["purpose_egg_storage"] = p_str.str.contains("난자 저장용", na=False).astype(
            int
        )
        df["purpose_donation"] = p_str.str.contains("기증용", na=False).astype(int)
        df["purpose_research"] = p_str.str.contains("연구용", na=False).astype(int)
        df["purpose_token_count"] = (
            df["purpose_current"]
            + df["purpose_embryo_storage"]
            + df["purpose_egg_storage"]
            + df["purpose_donation"]
            + df["purpose_research"]
        )
        df["purpose_current_and_storage"] = (
            (df["purpose_current"] == 1)
            & ((df["purpose_embryo_storage"] == 1) | (df["purpose_egg_storage"] == 1))
        ).astype(int)
        df["purpose_zero"] = ((df["purpose_current"] == 0) & (embryo == 0)).astype(int)

    # ---- 특정 시술 유형 토큰 ----
    if col_exists(df, spec_type_col):
        st_str = df[spec_type_col].fillna("Unknown").astype(str).str.upper()
        tokens = st_str.str.replace("/", " ", regex=False).str.replace(
            ":", " ", regex=False
        )
        df["subtype_has_icsi"] = tokens.str.contains("ICSI", na=False).astype(int)
        df["subtype_has_ivf"] = tokens.str.contains(
            r"\bIVF\b", na=False, regex=True
        ).astype(int)
        df["subtype_has_iui"] = tokens.str.contains("IUI", na=False).astype(int)
        df["subtype_has_ah"] = tokens.str.contains(
            r"\bAH\b", na=False, regex=True
        ).astype(int)
        df["subtype_has_blastocyst"] = tokens.str.contains(
            "BLASTOCYST", na=False
        ).astype(int)
        df["subtype_has_unknown"] = (st_str == "UNKNOWN").astype(int)
        df["subtype_token_count"] = (
            tokens.str.split()
            .apply(lambda x: len(x) if isinstance(x, list) else 0)
            .fillna(0)
            .astype(int)
        )

        raw = df[spec_type_col].fillna("Unknown").astype(str)
        parts = raw.str.split(r"[:/]", regex=True)
        df["subtype_is_duplicate"] = parts.apply(
            lambda x: (
                int(len(x) >= 2 and len(set(s.strip() for s in x)) == 1)
                if isinstance(x, list)
                else 0
            )
        )
        df["subtype_is_mixed_proc"] = (
            (df["subtype_has_icsi"] == 1) & (df["subtype_has_ivf"] == 1)
        ).astype(int)
        df["subtype_blastocyst_x_single"] = df["subtype_has_blastocyst"] * (
            embryo == 1
        ).astype(int)
        df["subtype_blastocyst_x_day5plus"] = df["subtype_has_blastocyst"] * day5plus

        df["icsi_x_day5plus"] = df["subtype_has_icsi"] * day5plus
        df["blastocyst_signal"] = (
            (tr_day >= 5) | (df["subtype_has_blastocyst"] == 1)
        ).astype(int)

    # first_treatment
    df["first_treatment"] = (tot_proc == 0).astype(int)

    return df


def get_cb_cat_features(df: pd.DataFrame) -> Tuple[List[int], List[str]]:
    """CatBoost용 카테고리 인덱스 반환"""
    found = [c for c in CAT_COLS_CANDIDATES if c in df.columns]
    idxs = [df.columns.get_loc(c) for c in found]
    return idxs, found


def make_lgb_features(
    train_cb: pd.DataFrame,
    test_cb: pd.DataFrame,
    y: np.ndarray,
    cat_names: List[str],
    fold_indices: List[Tuple[np.ndarray, np.ndarray]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """LightGBM/XGBoost용 수치 전처리 – OOF TE + label encoding"""
    tr = train_cb.copy()
    te = test_cb.copy()

    te_cols = [
        "특정 시술 유형",
        "배아 생성 주요 이유",
        "시술 시기 코드",
        "배란 유도 유형",
        "난자 출처",
        "정자 출처",
        "시술 당시 나이",
    ]

    # Label encoding for all cat cols
    for c in cat_names:
        if c not in tr.columns:
            continue
        combined_vals = pd.concat([tr[c], te[c]], axis=0).astype(str)
        codes, uniques = pd.factorize(combined_vals)
        n_tr = len(tr)
        tr[c] = codes[:n_tr]
        te[c] = codes[n_tr:]

    # OOF target encoding
    for c in te_cols:
        if c not in train_cb.columns:
            continue
        te_col_name = f"{c}_te"
        tr[te_col_name] = 0.0
        global_mean = y.mean()

        for tr_idx, va_idx in fold_indices:
            cat_vals = train_cb[c].iloc[tr_idx].astype(str)
            y_tr = y[tr_idx]
            te_map = {}
            for val in cat_vals.unique():
                mask = cat_vals == val
                te_map[val] = y_tr[mask.values].mean()
            va_vals = train_cb[c].iloc[va_idx].astype(str)
            tr.iloc[va_idx, tr.columns.get_loc(te_col_name)] = (
                va_vals.map(te_map).fillna(global_mean).values
            )

        # test: full train map
        full_cat = train_cb[c].astype(str)
        full_map = {}
        for val in full_cat.unique():
            mask = full_cat == val
            full_map[val] = y[mask.values].mean()
        te[te_col_name] = (
            test_cb[c].astype(str).map(full_map).fillna(global_mean).values
        )

    # Count encoding
    for c in te_cols:
        if c not in train_cb.columns:
            continue
        cnt_col = f"{c}_count"
        cnt_map = train_cb[c].astype(str).value_counts().to_dict()
        tr[cnt_col] = train_cb[c].astype(str).map(cnt_map).fillna(0).values
        te[cnt_col] = test_cb[c].astype(str).map(cnt_map).fillna(0).values

    # object 컬럼 제거
    obj_cols = tr.select_dtypes(include=["object"]).columns.tolist()
    tr = tr.drop(columns=obj_cols, errors="ignore")
    te = te.drop(columns=obj_cols, errors="ignore")

    # NaN → 0
    tr = tr.fillna(0)
    te = te.fillna(0)

    return tr, te


def build_stratify_key(
    y: np.ndarray, is_transfer: np.ndarray, train_df: pd.DataFrame
) -> np.ndarray:
    """복합 stratify key 생성"""
    type_col = "시술 유형"
    age_col = "시술 당시 나이"
    t_str = (
        train_df[type_col].fillna("UNK").astype(str).values
        if col_exists(train_df, type_col)
        else np.full(len(y), "UNK")
    )
    a_str = (
        train_df[age_col].fillna("UNK").astype(str).values
        if col_exists(train_df, age_col)
        else np.full(len(y), "UNK")
    )

    combined = [
        f"{yi}_{ti}_{si}_{ai}" for yi, ti, si, ai in zip(y, is_transfer, t_str, a_str)
    ]
    codes, _ = pd.factorize(combined)
    return codes


# ============================================================
# TRAINING FUNCTIONS
# ============================================================
def train_catboost_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    X_test: pd.DataFrame,
    cat_idx: List[int],
    params: dict,
    seeds: List[int],
    n_folds: int,
    fold_indices: List[Tuple[np.ndarray, np.ndarray]],
    task_type: str,
    model_name: str,
    sample_weight: Optional[np.ndarray] = None,
    subset_mask_train: Optional[np.ndarray] = None,
    subset_mask_test: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Any]:

    oof_full = np.zeros(len(y))
    test_full = np.zeros(len(X_test))
    seed_aucs = []
    importances = None

    total_models = len(seeds) * n_folds
    model_count = 0
    t_start = time.time()

    for si, seed_val in enumerate(seeds):
        log(f"\n  --- {model_name} Seed {seed_val} ({si+1}/{len(seeds)}) ---")
        oof_seed = np.zeros(len(y))
        test_seed = np.zeros(len(X_test))

        # seed별로 다른 fold split 생성
        if si == 0:
            seed_fold_indices = fold_indices
        else:
            skf_seed = StratifiedKFold(
                n_splits=n_folds, shuffle=True, random_state=seed_val
            )
            seed_fold_indices = list(skf_seed.split(X, y))

        for fold_i, (tr_idx, va_idx) in enumerate(seed_fold_indices):
            model_count += 1
            t_fold = time.time()

            if subset_mask_train is not None:
                tr_idx = tr_idx[subset_mask_train[tr_idx]]
                va_idx = va_idx[subset_mask_train[va_idx]]
                if len(tr_idx) == 0 or len(va_idx) == 0:
                    continue

            X_tr = X.iloc[tr_idx]
            X_va = X.iloc[va_idx]
            y_tr = y[tr_idx]
            y_va = y[va_idx]

            cb_params = params.copy()
            cb_params["random_seed"] = seed_val * 100 + fold_i
            cb_params["task_type"] = task_type
            if task_type == "GPU":
                cb_params["devices"] = "0"

            model = CatBoostClassifier(**cb_params)

            fit_kwargs = {
                "eval_set": (X_va, y_va),
                "cat_features": cat_idx,
                "verbose": False,
            }
            if sample_weight is not None:
                fit_kwargs["sample_weight"] = sample_weight[tr_idx]

            model.fit(X_tr, y_tr, **fit_kwargs)

            va_pred = model.predict_proba(X_va)[:, 1]
            oof_seed[va_idx] = va_pred

            if subset_mask_test is not None:
                test_subset = X_test[subset_mask_test]
                test_seed[subset_mask_test] += (
                    model.predict_proba(test_subset)[:, 1] / n_folds
                )
            else:
                test_seed += model.predict_proba(X_test)[:, 1] / n_folds

            va_auc = roc_auc_score(y_va, va_pred) if len(np.unique(y_va)) > 1 else 0.0
            elapsed = (time.time() - t_fold) / 60
            log(
                f"    Fold {fold_i+1}: AUC={va_auc:.4f}, iter={model.best_iteration_}, "
                f"소요={elapsed:.1f}분 [{model_count}/{total_models}]"
            )

            if si == len(seeds) - 1 and fold_i == n_folds - 1:
                importances = model.get_feature_importance()

        if subset_mask_train is not None:
            valid_mask = subset_mask_train & (oof_seed != 0)
            if valid_mask.sum() > 0 and len(np.unique(y[valid_mask])) > 1:
                seed_auc = roc_auc_score(y[valid_mask], oof_seed[valid_mask])
            else:
                seed_auc = 0.0
        else:
            seed_auc = roc_auc_score(y, oof_seed)

        seed_aucs.append(seed_auc)
        log(f"  Seed {seed_val} OOF AUC: {seed_auc:.4f}")

        oof_full += oof_seed / len(seeds)
        test_full += test_seed / len(seeds)

    train_time = (time.time() - t_start) / 60
    log(
        f"  === {model_name} 완료: {train_time:.1f}분, seed AUCs: {[f'{a:.4f}' for a in seed_aucs]} ==="
    )

    # npy 저장
    np.save(os.path.join(RESULT_DIR, f"oof_{VERSION}_{model_name}.npy"), oof_full)
    np.save(os.path.join(RESULT_DIR, f"test_{VERSION}_{model_name}.npy"), test_full)
    log(f"  저장: oof_{VERSION}_{model_name}.npy, test_{VERSION}_{model_name}.npy")
    save_log()

    return oof_full, test_full, importances


def train_lgb_transfer_cv(
    X_lgb_train: pd.DataFrame,
    y: np.ndarray,
    X_lgb_test: pd.DataFrame,
    seeds: List[int],
    n_folds: int,
    fold_indices: List[Tuple[np.ndarray, np.ndarray]],
    transfer_mask_train: np.ndarray,
    transfer_mask_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:

    oof_full = np.zeros(len(y))
    test_full = np.zeros(len(X_lgb_test))
    importances = None

    use_lgb = HAS_LGB
    engine_name = "LightGBM" if use_lgb else ("XGBoost" if HAS_XGB else None)

    if engine_name is None:
        log("  [경고] LightGBM/XGBoost 모두 미설치 → Model D 스킵")
        return oof_full, test_full, None

    log(f"\n  === Model D: Transfer-only {engine_name} ===")

    total_models = len(seeds) * n_folds
    model_count = 0
    seed_aucs = []
    t_start = time.time()

    for si, seed_val in enumerate(seeds):
        log(f"\n  --- {engine_name} Seed {seed_val} ({si+1}/{len(seeds)}) ---")
        oof_seed = np.zeros(len(y))
        test_seed = np.zeros(len(X_lgb_test))

        # seed별 다른 fold split
        if si == 0:
            seed_fold_indices = fold_indices
        else:
            skf_seed = StratifiedKFold(
                n_splits=n_folds, shuffle=True, random_state=seed_val
            )
            seed_fold_indices = list(skf_seed.split(X_lgb_train, y))

        for fold_i, (tr_idx, va_idx) in enumerate(seed_fold_indices):
            model_count += 1
            t_fold = time.time()

            tr_idx_sub = tr_idx[transfer_mask_train[tr_idx]]
            va_idx_sub = va_idx[transfer_mask_train[va_idx]]
            if len(tr_idx_sub) == 0 or len(va_idx_sub) == 0:
                continue

            X_tr = X_lgb_train.iloc[tr_idx_sub]
            X_va = X_lgb_train.iloc[va_idx_sub]
            y_tr = y[tr_idx_sub]
            y_va = y[va_idx_sub]

            test_sub = X_lgb_test[transfer_mask_test]

            if use_lgb:
                params = LGB_PARAMS.copy()
                params["seed"] = seed_val * 100 + fold_i
                dtrain = lgb.Dataset(X_tr, label=y_tr)
                dval = lgb.Dataset(X_va, label=y_va, reference=dtrain)
                model = lgb.train(
                    params,
                    dtrain,
                    num_boost_round=5000,
                    valid_sets=[dval],
                    callbacks=[
                        lgb.early_stopping(200, verbose=False),
                        lgb.log_evaluation(0),
                    ],
                )
                va_pred = model.predict(X_va)
                oof_seed[va_idx_sub] = va_pred
                test_seed[transfer_mask_test] += model.predict(test_sub) / n_folds

                if si == len(seeds) - 1 and fold_i == n_folds - 1:
                    importances = model.feature_importance(importance_type="gain")
            else:
                params = XGB_PARAMS.copy()
                params["random_state"] = seed_val * 100 + fold_i
                if CFG["GPU_AVAILABLE"]:
                    params["tree_method"] = "gpu_hist"
                    params["gpu_id"] = 0
                model = xgb.XGBClassifier(**params)
                model.fit(
                    X_tr,
                    y_tr,
                    eval_set=[(X_va, y_va)],
                    verbose=False,
                )
                va_pred = model.predict_proba(X_va)[:, 1]
                oof_seed[va_idx_sub] = va_pred
                test_seed[transfer_mask_test] += (
                    model.predict_proba(test_sub)[:, 1] / n_folds
                )

                if si == len(seeds) - 1 and fold_i == n_folds - 1:
                    importances = model.feature_importances_

            va_auc = roc_auc_score(y_va, va_pred) if len(np.unique(y_va)) > 1 else 0.0
            elapsed = (time.time() - t_fold) / 60
            log(
                f"    Fold {fold_i+1}: AUC={va_auc:.4f}, 소요={elapsed:.1f}분 [{model_count}/{total_models}]"
            )

        valid_mask = transfer_mask_train & (oof_seed != 0)
        if valid_mask.sum() > 0 and len(np.unique(y[valid_mask])) > 1:
            seed_auc = roc_auc_score(y[valid_mask], oof_seed[valid_mask])
        else:
            seed_auc = 0.0
        seed_aucs.append(seed_auc)
        log(f"  Seed {seed_val} Transfer AUC: {seed_auc:.4f}")

        oof_full += oof_seed / len(seeds)
        test_full += test_seed / len(seeds)

    train_time = (time.time() - t_start) / 60
    log(
        f"  === {engine_name} 완료: {train_time:.1f}분, seed AUCs: {[f'{a:.4f}' for a in seed_aucs]} ==="
    )

    # npy 저장
    model_tag = "Transfer_LGB" if use_lgb else "Transfer_XGB"
    np.save(os.path.join(RESULT_DIR, f"oof_{VERSION}_{model_tag}.npy"), oof_full)
    np.save(os.path.join(RESULT_DIR, f"test_{VERSION}_{model_tag}.npy"), test_full)
    log(f"  저장: oof_{VERSION}_{model_tag}.npy, test_{VERSION}_{model_tag}.npy")
    save_log()

    return oof_full, test_full, importances


# ============================================================
# BLENDING
# ============================================================
def search_blend_weights(
    y: np.ndarray,
    oof_full_cb: np.ndarray,
    oof_transfer_cb: np.ndarray,
    oof_nontransfer_cb: np.ndarray,
    oof_transfer_lgb: np.ndarray,
    transfer_mask: np.ndarray,
) -> Dict:
    """3가지 블렌딩 방식 탐색"""

    results = {}
    non_mask = ~transfer_mask

    # ---- 1) Probability blend ----
    best_prob = {"auc": 0, "tw": (1, 0, 0), "nw": (1, 0)}

    for a in np.arange(0, 1.05, 0.05):
        for b in np.arange(0, 1.05 - a, 0.05):
            c = round(1.0 - a - b, 2)
            if c < -0.01:
                continue
            c = max(c, 0)

            for d in np.arange(0, 1.05, 0.02):
                e = round(1.0 - d, 2)
                if e < -0.01:
                    continue
                e = max(e, 0)

                pred = np.zeros(len(y))
                pred[transfer_mask] = (
                    a * oof_full_cb[transfer_mask]
                    + b * oof_transfer_cb[transfer_mask]
                    + c * oof_transfer_lgb[transfer_mask]
                )
                pred[non_mask] = (
                    d * oof_full_cb[non_mask] + e * oof_nontransfer_cb[non_mask]
                )

                try:
                    auc = roc_auc_score(y, pred)
                except:
                    continue

                if auc > best_prob["auc"]:
                    best_prob = {
                        "auc": auc,
                        "tw": (round(a, 2), round(b, 2), round(c, 2)),
                        "nw": (round(d, 2), round(e, 2)),
                    }

    results["prob_blend"] = best_prob

    # ---- 2) Rank blend ----
    def to_rank(arr):
        return sp_stats.rankdata(arr) / len(arr)

    r_full = to_rank(oof_full_cb)
    r_tcb = to_rank(oof_transfer_cb)
    r_ncb = to_rank(oof_nontransfer_cb)
    r_tlgb = to_rank(oof_transfer_lgb)

    best_rank = {"auc": 0, "tw": (1, 0, 0), "nw": (1, 0)}

    for a in np.arange(0, 1.05, 0.1):
        for b in np.arange(0, 1.05 - a, 0.1):
            c = round(1.0 - a - b, 2)
            if c < -0.01:
                continue
            c = max(c, 0)

            for d in np.arange(0, 1.05, 0.1):
                e = round(1.0 - d, 2)
                if e < -0.01:
                    continue
                e = max(e, 0)

                pred = np.zeros(len(y))
                pred[transfer_mask] = (
                    a * r_full[transfer_mask]
                    + b * r_tcb[transfer_mask]
                    + c * r_tlgb[transfer_mask]
                )
                pred[non_mask] = d * r_full[non_mask] + e * r_ncb[non_mask]

                try:
                    auc = roc_auc_score(y, pred)
                except:
                    continue

                if auc > best_rank["auc"]:
                    best_rank = {
                        "auc": auc,
                        "tw": (round(a, 2), round(b, 2), round(c, 2)),
                        "nw": (round(d, 2), round(e, 2)),
                    }

    results["rank_blend"] = best_rank

    # ---- 3) Z-score blend ----
    def to_zscore(arr):
        m, s = arr.mean(), arr.std()
        if s < 1e-10:
            return arr - m
        return (arr - m) / s

    z_full = to_zscore(oof_full_cb)
    z_tcb = to_zscore(oof_transfer_cb)
    z_ncb = to_zscore(oof_nontransfer_cb)
    z_tlgb = to_zscore(oof_transfer_lgb)

    best_z = {"auc": 0, "tw": (1, 0, 0), "nw": (1, 0)}

    for a in np.arange(0, 1.05, 0.1):
        for b in np.arange(0, 1.05 - a, 0.1):
            c = round(1.0 - a - b, 2)
            if c < -0.01:
                continue
            c = max(c, 0)

            for d in np.arange(0, 1.05, 0.1):
                e = round(1.0 - d, 2)
                if e < -0.01:
                    continue
                e = max(e, 0)

                pred = np.zeros(len(y))
                pred[transfer_mask] = (
                    a * z_full[transfer_mask]
                    + b * z_tcb[transfer_mask]
                    + c * z_tlgb[transfer_mask]
                )
                pred[non_mask] = d * z_full[non_mask] + e * z_ncb[non_mask]

                try:
                    auc = roc_auc_score(y, pred)
                except:
                    continue

                if auc > best_z["auc"]:
                    best_z = {
                        "auc": auc,
                        "tw": (round(a, 2), round(b, 2), round(c, 2)),
                        "nw": (round(d, 2), round(e, 2)),
                    }

    results["z_blend"] = best_z

    return results


def apply_best_blend(
    method: str,
    weights: Dict,
    full_cb: np.ndarray,
    transfer_cb: np.ndarray,
    nontransfer_cb: np.ndarray,
    transfer_lgb: np.ndarray,
    transfer_mask: np.ndarray,
    oof_full_cb_stats: Optional[Dict] = None,
    oof_transfer_cb_stats: Optional[Dict] = None,
    oof_nontransfer_cb_stats: Optional[Dict] = None,
    oof_transfer_lgb_stats: Optional[Dict] = None,
    oof_full_cb_ref: Optional[np.ndarray] = None,
) -> np.ndarray:
    """최적 블렌딩 적용"""
    non_mask = ~transfer_mask
    tw = weights["tw"]
    nw = weights["nw"]

    if method == "prob_blend":
        pred = np.zeros(len(full_cb))
        pred[transfer_mask] = (
            tw[0] * full_cb[transfer_mask]
            + tw[1] * transfer_cb[transfer_mask]
            + tw[2] * transfer_lgb[transfer_mask]
        )
        pred[non_mask] = nw[0] * full_cb[non_mask] + nw[1] * nontransfer_cb[non_mask]
        return pred

    elif method == "rank_blend":

        def to_rank(arr):
            return sp_stats.rankdata(arr) / len(arr)

        pred = np.zeros(len(full_cb))
        pred[transfer_mask] = (
            tw[0] * to_rank(full_cb)[transfer_mask]
            + tw[1] * to_rank(transfer_cb)[transfer_mask]
            + tw[2] * to_rank(transfer_lgb)[transfer_mask]
        )
        pred[non_mask] = (
            nw[0] * to_rank(full_cb)[non_mask]
            + nw[1] * to_rank(nontransfer_cb)[non_mask]
        )
        return pred

    elif method == "z_blend":

        def z(arr, ref_mean, ref_std):
            if ref_std < 1e-10:
                return arr - ref_mean
            return (arr - ref_mean) / ref_std

        pred = np.zeros(len(full_cb))
        pred[transfer_mask] = (
            tw[0]
            * z(full_cb, oof_full_cb_stats["mean"], oof_full_cb_stats["std"])[
                transfer_mask
            ]
            + tw[1]
            * z(
                transfer_cb, oof_transfer_cb_stats["mean"], oof_transfer_cb_stats["std"]
            )[transfer_mask]
            + tw[2]
            * z(
                transfer_lgb,
                oof_transfer_lgb_stats["mean"],
                oof_transfer_lgb_stats["std"],
            )[transfer_mask]
        )
        pred[non_mask] = (
            nw[0]
            * z(full_cb, oof_full_cb_stats["mean"], oof_full_cb_stats["std"])[non_mask]
            + nw[1]
            * z(
                nontransfer_cb,
                oof_nontransfer_cb_stats["mean"],
                oof_nontransfer_cb_stats["std"],
            )[non_mask]
        )
        return pred

    return full_cb.copy()


# ============================================================
# EVALUATION
# ============================================================
def evaluate_predictions(
    y: np.ndarray,
    pred: np.ndarray,
    transfer_mask: np.ndarray,
    train_raw: pd.DataFrame,
    label: str = "Final",
):
    log(f"\n### {label} 평가지표")

    auc = roc_auc_score(y, pred)
    ll = log_loss(y, np.clip(pred, 1e-7, 1 - 1e-7))
    ap = average_precision_score(y, pred)

    log(f"  OOF AUC:      {auc:.6f}")
    log(f"  OOF Log Loss: {ll:.6f}")
    log(f"  OOF AP:       {ap:.6f}")

    # 그룹별 AUC
    log(f"\n  #### 그룹별 AUC")
    groups = {}
    groups["transfer"] = transfer_mask
    groups["non-transfer"] = ~transfer_mask

    type_col = "시술 유형"
    if col_exists(train_raw, type_col):
        groups["IVF & transfer"] = transfer_mask & (train_raw[type_col] == "IVF").values
        groups["DI"] = (train_raw[type_col] == "DI").values

    fresh_col = "신선 배아 사용 여부"
    frozen_col = "동결 배아 사용 여부"
    if col_exists(train_raw, fresh_col):
        groups["fresh transfer"] = (
            transfer_mask & (train_raw[fresh_col].fillna(0) == 1).values
        )
    if col_exists(train_raw, frozen_col):
        groups["frozen transfer"] = (
            transfer_mask & (train_raw[frozen_col].fillna(0) == 1).values
        )

    egg_col = "난자 출처"
    if col_exists(train_raw, egg_col):
        groups["donor egg"] = (train_raw[egg_col] == "기증 제공").values

    age_col = "시술 당시 나이"
    if col_exists(train_raw, age_col):
        for age_val in [
            "만18-34세",
            "만35-37세",
            "만38-39세",
            "만40-42세",
            "만43-44세",
            "만45-50세",
        ]:
            groups[f"age_{age_val}"] = (train_raw[age_col] == age_val).values

    for gname, gmask in groups.items():
        gmask = gmask & np.isfinite(pred)
        if gmask.sum() > 50 and len(np.unique(y[gmask])) > 1:
            g_auc = roc_auc_score(y[gmask], pred[gmask])
            g_pos = y[gmask].mean()
            log(f"    {gname}: AUC={g_auc:.4f} (n={gmask.sum()}, pos={g_pos*100:.1f}%)")

    # Threshold 표
    log(f"\n  #### Threshold별 분류 지표")
    log(f"  {'Th':>6}  {'Acc':>7}  {'Prec':>7}  {'Rec':>7}  {'F1':>7}  {'Spec':>7}")
    log(f"  {'---':>6}  {'---':>7}  {'---':>7}  {'---':>7}  {'---':>7}  {'---':>7}")

    best_f1, best_th = 0, 0.25
    for th in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        yb = (pred >= th).astype(int)
        acc = accuracy_score(y, yb)
        prec = precision_score(y, yb, zero_division=0)
        rec = recall_score(y, yb, zero_division=0)
        f1 = f1_score(y, yb, zero_division=0)
        spec = np.sum((y == 0) & (yb == 0)) / max(np.sum(y == 0), 1)
        log(
            f"  {th:>6.2f}  {acc:>7.4f}  {prec:>7.4f}  {rec:>7.4f}  {f1:>7.4f}  {spec:>7.4f}"
        )
        if f1 > best_f1:
            best_f1, best_th = f1, th

    log(f"  최적 F1: {best_f1:.4f} (threshold={best_th})")

    return auc


# ============================================================
# MAIN
# ============================================================
def main():
    global log_lines
    t_total = time.time()

    log(f"# {VERSION} - Subgroup-aware Multi-model Blend")
    log(f"시각: {NOW}")
    log("=" * 60)

    # ---- [1] 데이터 ----
    train_raw, test_raw, sub, y, target_col = load_data()

    # ---- [2] 피처 엔지니어링 ----
    log("\n## [2] 전처리 + 피처 엔지니어링")
    t_fe = time.time()

    X_train_cb = make_base_features(train_raw, target_col, is_train=True)
    X_test_cb = make_base_features(test_raw, target_col, is_train=False)

    # 컬럼 동기화
    common_cols = [c for c in X_train_cb.columns if c in X_test_cb.columns]
    X_train_cb = X_train_cb[common_cols]
    X_test_cb = X_test_cb[common_cols]

    cat_idx, cat_names = get_cb_cat_features(X_train_cb)

    log(f"  피처 수: {len(X_train_cb.columns)}, 카테고리: {len(cat_idx)}")
    log(f"  카테고리: {cat_names}")
    log(f"  피처 엔지니어링 소요: {(time.time()-t_fe)/60:.1f}분")

    # ---- [3] subgroup ----
    log("\n## [3] subgroup 통계")
    embryo_raw = (
        train_raw["이식된 배아 수"].fillna(0).values
        if "이식된 배아 수" in train_raw.columns
        else np.zeros(len(y))
    )
    transfer_mask_train = embryo_raw > 0

    embryo_test = (
        test_raw["이식된 배아 수"].fillna(0).values
        if "이식된 배아 수" in test_raw.columns
        else np.zeros(len(test_raw))
    )
    transfer_mask_test = embryo_test > 0

    log(
        f"  이식 (train): {transfer_mask_train.sum()}건 ({transfer_mask_train.mean()*100:.1f}%), "
        f"양성률={y[transfer_mask_train].mean()*100:.1f}%"
    )
    log(
        f"  비이식 (train): {(~transfer_mask_train).sum()}건, "
        f"양성률={y[~transfer_mask_train].mean()*100:.2f}%"
    )
    log(
        f"  이식 (test): {transfer_mask_test.sum()}건, 비이식: {(~transfer_mask_test).sum()}건"
    )

    # ---- [4] CV 설정 ----
    log("\n## [4] CV 설정")
    strat_key = build_stratify_key(y, transfer_mask_train.astype(int), train_raw)

    # stratify key가 너무 세분화되면 fallback
    min_class_count = pd.Series(strat_key).value_counts().min()
    if min_class_count < CFG["N_FOLDS"]:
        log(
            f"  [경고] strat_key 최소 클래스={min_class_count} < {CFG['N_FOLDS']}, y로 fallback"
        )
        strat_key = y

    skf = StratifiedKFold(n_splits=CFG["N_FOLDS"], shuffle=True, random_state=42)
    fold_indices = list(skf.split(X_train_cb, strat_key))
    log(f"  {CFG['N_FOLDS']}-fold StratifiedKFold 생성 완료")

    # GPU 감지
    task_type = detect_gpu()
    log(f"  Task type: {task_type}")

    # ---- sample weights ----
    sample_weight = None
    if CFG["USE_GROUP_WEIGHTS"]:
        log("\n  [INFO] 샘플 가중치 적용")
        sample_weight = np.ones(len(y))
        sample_weight[transfer_mask_train] = 1.25
        sample_weight[~transfer_mask_train] = 0.75
        age_num = (
            X_train_cb["시술 당시 나이_num"].values
            if "시술 당시 나이_num" in X_train_cb.columns
            else np.zeros(len(y))
        )
        age_bonus = transfer_mask_train & (age_num >= 35) & (age_num <= 42)
        sample_weight[age_bonus] *= 1.10
        log(f"  가중치: transfer=1.25, non-transfer=0.75, age35-42 bonus=1.10")

    # ---- LGB 피처 준비 ----
    log("\n## [5] LGB/XGB 피처 준비")
    t_lgb_fe = time.time()
    X_train_lgb, X_test_lgb = make_lgb_features(
        X_train_cb, X_test_cb, y, cat_names, fold_indices
    )
    log(f"  LGB 피처 수: {len(X_train_lgb.columns)}")
    log(f"  소요: {(time.time()-t_lgb_fe)/60:.1f}분")

    # ============================================================
    # MODEL A: Full CatBoost
    # ============================================================
    log("\n" + "=" * 60)
    log("## [6] Model A: Full CatBoost")
    log("=" * 60)
    log(
        f"  파라미터: {json.dumps({k: v for k, v in FULL_CB_PARAMS.items() if k != 'verbose'}, indent=2, default=str)}"
    )

    oof_full_cb, test_full_cb, imp_full_cb = train_catboost_cv(
        X_train_cb,
        y,
        X_test_cb,
        cat_idx,
        FULL_CB_PARAMS,
        CFG["FULL_CB_SEEDS"],
        CFG["N_FOLDS"],
        fold_indices,
        task_type,
        "Full_CB",
        sample_weight=sample_weight,
    )

    full_cb_auc = roc_auc_score(y, oof_full_cb)
    log(f"\n  Full CB OOF AUC: {full_cb_auc:.6f}")

    # ============================================================
    # MODEL B: Transfer-only CatBoost
    # ============================================================
    log("\n" + "=" * 60)
    log("## [7] Model B: Transfer-only CatBoost")
    log("=" * 60)

    oof_transfer_cb, test_transfer_cb, imp_transfer_cb = train_catboost_cv(
        X_train_cb,
        y,
        X_test_cb,
        cat_idx,
        TRANSFER_CB_PARAMS,
        CFG["TRANSFER_CB_SEEDS"],
        CFG["N_FOLDS"],
        fold_indices,
        task_type,
        "Transfer_CB",
        subset_mask_train=transfer_mask_train,
        subset_mask_test=transfer_mask_test,
    )

    tr_valid = transfer_mask_train & (oof_transfer_cb != 0)
    if tr_valid.sum() > 0:
        transfer_cb_auc = roc_auc_score(y[tr_valid], oof_transfer_cb[tr_valid])
        log(f"\n  Transfer CB (이식그룹) OOF AUC: {transfer_cb_auc:.6f}")

    # ============================================================
    # MODEL C: Non-transfer CatBoost
    # ============================================================
    log("\n" + "=" * 60)
    log("## [8] Model C: Non-transfer CatBoost")
    log("=" * 60)

    oof_nontransfer_cb, test_nontransfer_cb, imp_nontransfer_cb = train_catboost_cv(
        X_train_cb,
        y,
        X_test_cb,
        cat_idx,
        NONTRANSFER_CB_PARAMS,
        CFG["NONTRANSFER_CB_SEEDS"],
        CFG["N_FOLDS"],
        fold_indices,
        task_type,
        "NonTransfer_CB",
        subset_mask_train=~transfer_mask_train,
        subset_mask_test=~transfer_mask_test,
    )

    nt_valid = (~transfer_mask_train) & (oof_nontransfer_cb != 0)
    if nt_valid.sum() > 0 and len(np.unique(y[nt_valid])) > 1:
        nt_cb_auc = roc_auc_score(y[nt_valid], oof_nontransfer_cb[nt_valid])
        log(f"\n  NonTransfer CB (비이식그룹) OOF AUC: {nt_cb_auc:.6f}")

    # ============================================================
    # MODEL D: Transfer-only LightGBM/XGBoost
    # ============================================================
    log("\n" + "=" * 60)
    log("## [9] Model D: Transfer-only LightGBM/XGBoost")
    log("=" * 60)

    oof_transfer_lgb, test_transfer_lgb, imp_transfer_lgb = train_lgb_transfer_cv(
        X_train_lgb,
        y,
        X_test_lgb,
        CFG["TRANSFER_LGB_SEEDS"],
        CFG["N_FOLDS"],
        fold_indices,
        transfer_mask_train,
        transfer_mask_test,
    )

    lgb_valid = transfer_mask_train & (oof_transfer_lgb != 0)
    if lgb_valid.sum() > 0:
        lgb_auc = roc_auc_score(y[lgb_valid], oof_transfer_lgb[lgb_valid])
        log(f"\n  Transfer LGB/XGB (이식그룹) OOF AUC: {lgb_auc:.6f}")

    # ============================================================
    # 상관 분석
    # ============================================================
    log("\n## [10] OOF 예측 상관분석")

    def safe_corr(a, b, mask):
        a_m, b_m = a[mask], b[mask]
        if len(a_m) < 10:
            return 0.0
        return np.corrcoef(a_m, b_m)[0, 1]

    pairs = [
        ("full_cb ↔ transfer_cb", oof_full_cb, oof_transfer_cb, transfer_mask_train),
        ("full_cb ↔ transfer_lgb", oof_full_cb, oof_transfer_lgb, transfer_mask_train),
        (
            "full_cb ↔ nontransfer_cb",
            oof_full_cb,
            oof_nontransfer_cb,
            ~transfer_mask_train,
        ),
        (
            "transfer_cb ↔ transfer_lgb",
            oof_transfer_cb,
            oof_transfer_lgb,
            transfer_mask_train,
        ),
    ]

    log(f"  {'페어':<30}  {'상관계수':>8}")
    log(f"  {'-'*30}  {'-'*8}")
    for name, a, b, mask in pairs:
        valid = mask & (a != 0) & (b != 0)
        corr = safe_corr(a, b, valid)
        log(f"  {name:<30}  {corr:>8.4f}")

    # ============================================================
    # 블렌딩
    # ============================================================
    log("\n## [11] 블렌딩 가중치 탐색")
    t_blend = time.time()

    blend_results = search_blend_weights(
        y,
        oof_full_cb,
        oof_transfer_cb,
        oof_nontransfer_cb,
        oof_transfer_lgb,
        transfer_mask_train,
    )

    log(f"\n  탐색 소요: {(time.time()-t_blend)/60:.1f}분")
    log(f"\n  {'방식':<15}  {'AUC':>10}  {'Transfer W':>25}  {'NonTransfer W':>20}")
    log(f"  {'-'*15}  {'-'*10}  {'-'*25}  {'-'*20}")

    best_method = None
    best_auc = 0

    for method, info in blend_results.items():
        log(
            f"  {method:<15}  {info['auc']:>10.6f}  {str(info['tw']):>25}  {str(info['nw']):>20}"
        )
        if info["auc"] > best_auc:
            best_auc = info["auc"]
            best_method = method

    # Full CB 단독과도 비교
    log(f"  {'full_cb_only':<15}  {full_cb_auc:>10.6f}  {'(1,0,0)':>25}  {'(1,0)':>20}")
    if full_cb_auc > best_auc:
        best_auc = full_cb_auc
        best_method = "full_cb_only"

    log(f"\n  ★ 최적: {best_method} (AUC={best_auc:.6f})")

    # 최종 OOF
    if best_method == "full_cb_only":
        oof_final = oof_full_cb.copy()
    else:
        oof_final = apply_best_blend(
            best_method,
            blend_results[best_method],
            oof_full_cb,
            oof_transfer_cb,
            oof_nontransfer_cb,
            oof_transfer_lgb,
            transfer_mask_train,
            oof_full_cb_stats={"mean": oof_full_cb.mean(), "std": oof_full_cb.std()},
            oof_transfer_cb_stats={
                "mean": oof_transfer_cb[transfer_mask_train].mean(),
                "std": oof_transfer_cb[transfer_mask_train].std(),
            },
            oof_nontransfer_cb_stats={
                "mean": oof_nontransfer_cb[~transfer_mask_train].mean(),
                "std": oof_nontransfer_cb[~transfer_mask_train].std(),
            },
            oof_transfer_lgb_stats={
                "mean": oof_transfer_lgb[transfer_mask_train].mean(),
                "std": oof_transfer_lgb[transfer_mask_train].std(),
            },
        )

    # 최종 Test
    if best_method == "full_cb_only":
        test_final = test_full_cb.copy()
    else:
        test_final = apply_best_blend(
            best_method,
            blend_results[best_method],
            test_full_cb,
            test_transfer_cb,
            test_nontransfer_cb,
            test_transfer_lgb,
            transfer_mask_test,
            oof_full_cb_stats={"mean": oof_full_cb.mean(), "std": oof_full_cb.std()},
            oof_transfer_cb_stats={
                "mean": oof_transfer_cb[transfer_mask_train].mean(),
                "std": oof_transfer_cb[transfer_mask_train].std(),
            },
            oof_nontransfer_cb_stats={
                "mean": oof_nontransfer_cb[~transfer_mask_train].mean(),
                "std": oof_nontransfer_cb[~transfer_mask_train].std(),
            },
            oof_transfer_lgb_stats={
                "mean": oof_transfer_lgb[transfer_mask_train].mean(),
                "std": oof_transfer_lgb[transfer_mask_train].std(),
            },
        )

    # ============================================================
    # 평가
    # ============================================================
    log("\n## [12] 최종 성능")
    final_auc = evaluate_predictions(
        y, oof_final, transfer_mask_train, train_raw, "최종 블렌딩"
    )

    log("\n### 개별 모델 비교")
    evaluate_predictions(y, oof_full_cb, transfer_mask_train, train_raw, "Full CB 단독")

    # ============================================================
    # 피처 중요도
    # ============================================================
    log("\n## [13] 피처 중요도")
    feature_names = X_train_cb.columns.tolist()

    subtype_feats = [f for f in feature_names if f.startswith("subtype_")]
    purpose_feats = [f for f in feature_names if f.startswith("purpose_")]
    transfer_int_feats = [
        "transfer_day_optimal",
        "transfer_day_cat",
        "embryo_day_interaction",
        "fresh_transfer_ratio",
        "micro_transfer_quality",
        "single_good_embryo",
        "frozen_embryo_signal",
        "transfer_intensity",
        "age_transfer_interaction",
        "day5plus",
        "single_x_day5plus",
        "multi_x_day5plus",
        "fresh_x_day5plus",
        "frozen_x_day5plus",
        "age_x_day5plus",
        "age_x_single_transfer",
        "icsi_x_day5plus",
        "blastocyst_signal",
        "donor_egg_x_advanced_age",
    ]
    failure_feats = [
        "total_failure_count",
        "ivf_failure_count",
        "pregnancy_to_birth_gap",
        "ivf_preg_to_birth_gap",
        "repeated_failed_transfer",
        "first_transfer_cycle",
    ]

    def print_importance(imp_arr, feat_names, top_n, label):
        if imp_arr is None:
            log(f"\n  ### {label}: 없음")
            return
        imp_df = pd.DataFrame({"feature": feat_names, "importance": imp_arr})
        imp_df = imp_df.sort_values("importance", ascending=False)
        log(f"\n  ### {label} 상위 {top_n}")
        for i, (_, row) in enumerate(imp_df.head(top_n).iterrows(), 1):
            marks = ""
            if row["feature"] in subtype_feats:
                marks = " [subtype]"
            elif row["feature"] in purpose_feats:
                marks = " [purpose]"
            elif row["feature"] in transfer_int_feats:
                marks = " [transfer]"
            elif row["feature"] in failure_feats:
                marks = " [failure]"
            log(f"    {i:>3}. {row['feature']}{marks}: {row['importance']:.2f}")

        # 신규 피처 카테고리별 요약
        log(f"\n  #### {label} - 신규 피처 카테고리별")
        for cat_name, feat_list in [
            ("subtype", subtype_feats),
            ("purpose", purpose_feats),
            ("transfer_interaction", transfer_int_feats),
            ("failure", failure_feats),
        ]:
            found = [f for f in feat_list if f in imp_df["feature"].values]
            if found:
                log(f"    [{cat_name}]")
                for f in found:
                    v = imp_df[imp_df["feature"] == f]["importance"].values[0]
                    r = list(imp_df["feature"].values).index(f) + 1
                    log(f"      {f}: {v:.2f} (#{r})")

    print_importance(imp_full_cb, feature_names, 30, "Full CatBoost")
    print_importance(imp_transfer_cb, feature_names, 30, "Transfer CatBoost")
    print_importance(imp_nontransfer_cb, feature_names, 20, "NonTransfer CatBoost")

    if imp_transfer_lgb is not None:
        lgb_feat_names = X_train_lgb.columns.tolist()
        print_importance(imp_transfer_lgb, lgb_feat_names, 30, "Transfer LGB/XGB")

    # ============================================================
    # 저장
    # ============================================================
    log("\n## [14] 파일 저장")

    # submission
    sub_main = sub.copy()
    sub_main["probability"] = test_final
    sub_path = os.path.join(RESULT_DIR, f"sample_submission_{VERSION}_{NOW}.csv")
    sub_main.to_csv(sub_path, index=False)
    log(f"  제출: {sub_path}")
    log(
        f"  확률: mean={test_final.mean():.4f}, std={test_final.std():.4f}, "
        f"min={test_final.min():.6f}, max={test_final.max():.4f}"
    )

    # npy - 최종 결과
    np.save(os.path.join(RESULT_DIR, f"oof_{VERSION}_final.npy"), oof_final)
    np.save(os.path.join(RESULT_DIR, f"test_{VERSION}_final.npy"), test_final)
    np.save(os.path.join(RESULT_DIR, f"y_train_{VERSION}.npy"), y)
    np.save(
        os.path.join(RESULT_DIR, f"transfer_mask_train_{VERSION}.npy"),
        transfer_mask_train,
    )
    np.save(
        os.path.join(RESULT_DIR, f"transfer_mask_test_{VERSION}.npy"),
        transfer_mask_test,
    )
    log(f"  npy 저장: oof/test final, y_train, transfer_masks")

    # OOF csv files
    train_id = train_raw.iloc[:, 0]
    test_id = test_raw.iloc[:, 0]

    for name, arr in [
        ("full_cb", oof_full_cb),
        ("transfer_cb", oof_transfer_cb),
        ("nontransfer_cb", oof_nontransfer_cb),
        ("transfer_lgb", oof_transfer_lgb),
        ("final", oof_final),
    ]:
        path = os.path.join(RESULT_DIR, f"oof_{VERSION}_{name}.csv")
        pd.DataFrame({"ID": train_id, "prediction": arr}).to_csv(path, index=False)

    for name, arr in [
        ("full_cb", test_full_cb),
        ("transfer_cb", test_transfer_cb),
        ("nontransfer_cb", test_nontransfer_cb),
        ("transfer_lgb", test_transfer_lgb),
        ("final", test_final),
    ]:
        path = os.path.join(RESULT_DIR, f"test_pred_{VERSION}_{name}.csv")
        pd.DataFrame({"ID": test_id, "prediction": arr}).to_csv(path, index=False)

    # Feature importance CSV
    if imp_full_cb is not None:
        pd.DataFrame({"feature": feature_names, "importance": imp_full_cb}).sort_values(
            "importance", ascending=False
        ).to_csv(
            os.path.join(RESULT_DIR, f"feature_importance_{VERSION}_full_cb.csv"),
            index=False,
        )

    log(f"  모든 파일 저장 완료")

    # ============================================================
    # 버전 비교
    # ============================================================
    log("\n## [15] 버전 비교")
    log("| 버전 | 모델 | OOF AUC | Test AUC | 비고 |")
    log("|------|------|---------|----------|------|")
    log("| v1 | CB원본 | 0.7403 | - | 베이스라인 |")
    log("| v7 | XGB+CB | 0.7402 | - | 블렌딩 |")
    log("| v9 | XGB+CB 3seed | 0.7405 | 0.74166 | +IVF/DI |")
    log("| v12 | CB+XGB+Tab | 0.7406 | - | 3모델 |")
    log("| v14 | CB 3seed | 0.7406 | - | 이식피처 |")
    log("| v15 | CB 5seed+Optuna | 0.7407 | 0.74169 | EDA+HP |")
    log(
        f"| **{VERSION}** | **Multi-model** | **{final_auc:.4f}** | **TBD** | **subgroup blend** |"
    )

    # ============================================================
    # 최종 요약
    # ============================================================
    total_time = (time.time() - t_total) / 60
    log(f"\n{'='*60}")
    log(f"## 최종 요약")
    log(f"{'='*60}")
    log(f"- Full CB OOF AUC: {full_cb_auc:.6f}")
    log(f"- 최적 블렌딩: {best_method}")
    log(f"- 최종 OOF AUC: {final_auc:.6f}")
    log(f"- 총 소요: {total_time:.1f}분")
    log(f"- 데이터 누수: 없음")
    log(f"- 로그: {LOG_PATH}")
    log(f"{'='*60}")

    save_log()
    print(f"\n완료! 로그: {LOG_PATH}")


if __name__ == "__main__":
    main()
