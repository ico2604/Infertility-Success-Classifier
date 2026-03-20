#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v22_lite_notebook - v22-lite 독립 검증 (노트북 전용)
================================================================
v22 대비 변경:
  - created_minus_transferred 제거 (ablation)
유지:
  - subgroup-constant pruning
  - frozen tail 제거
  - embryo_transfer_days_x_age
  - blend weight fine search

핵심 아이디어:
  1. 현재 최고 global base는 v17-v19 blend
     -> 가능하면 외부 OOF/Test npy를 불러와 global base로 사용
     -> 없으면 내부 global(v20-pruned feature set) 학습 fallback
  2. transfer-only expert 모델 학습
  3. frozen-transfer-only expert 모델 학습
  4. OOF 기반으로 gating/blending weight 자동 탐색
     - non-frozen transfer: global + transfer expert
     - frozen transfer: global + transfer expert + frozen expert

목표:
  - 단일 global 모델이 못 올리는 transfer/frozen subgroup을 expert로 보완
"""

import os
import re
import json
import time
import datetime
import warnings
import numpy as np
import pandas as pd

from typing import List, Dict, Optional, Tuple
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
# 경로 / 설정
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULT_DIR = os.path.join(BASE_DIR, "result_v22_lite")
os.makedirs(RESULT_DIR, exist_ok=True)

VERSION = "v22_lite"
NOW = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_PATH = os.path.join(RESULT_DIR, f"log_{VERSION}.md")

# 현재 가장 좋은 global base
USE_EXTERNAL_GLOBAL_BLEND = True
EXTERNAL_GLOBAL_BLEND = {
    "v17": 0.65,
    "v19": 0.35,
}
# npy 파일은 패키지 내 npy/ 폴더에 포함
NPY_DIR = os.path.join(BASE_DIR, "npy")
EXTERNAL_OOF_PATHS = {
    "v17": os.path.join(NPY_DIR, "oof_v17_final.npy"),
    "v19": os.path.join(NPY_DIR, "oof_v19_final.npy"),
}
EXTERNAL_TEST_PATHS = {
    "v17": os.path.join(NPY_DIR, "test_v17_final.npy"),
    "v19": os.path.join(NPY_DIR, "test_v19_final.npy"),
}

# expert 학습용 seed
# SEEDS_FULL = [42, 2026, 2604, 123, 777]  # full run용
SEEDS = [42, 2026]  # 2-seed gate (노트북 독립 검증)
N_FOLDS = 5

CB_PARAMS = {
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
    "class_weights": {0: 1.0, 1: 1.3},
}

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

AGE_COLS = ["시술 당시 나이", "난자 기증자 나이", "정자 기증자 나이"]

FLAG_COLS = [
    "배란 자극 여부",
    "단일 배아 이식 여부",
    "착상 전 유전 검사 사용 여부",
    "착상 전 유전 진단 사용 여부",
    "남성 주 불임 원인",
    "남성 부 불임 원인",
    "여성 주 불임 원인",
    "여성 부 불임 원인",
    "부부 주 불임 원인",
    "부부 부 불임 원인",
    "불명확 불임 원인",
    "불임 원인 - 난관 질환",
    "불임 원인 - 남성 요인",
    "불임 원인 - 배란 장애",
    "불임 원인 - 여성 요인",
    "불임 원인 - 자궁경부 문제",
    "불임 원인 - 자궁내막증",
    "불임 원인 - 정자 농도",
    "불임 원인 - 정자 면역학적 요인",
    "불임 원인 - 정자 운동성",
    "불임 원인 - 정자 형태",
    "동결 배아 사용 여부",
    "신선 배아 사용 여부",
    "기증 배아 사용 여부",
    "대리모 여부",
    "PGD 시술 여부",
    "PGS 시술 여부",
]

BINARY_MAP = {
    "Y": 1,
    "N": 0,
    "y": 1,
    "n": 0,
    "Yes": 1,
    "No": 0,
    "YES": 1,
    "NO": 0,
    "예": 1,
    "아니오": 0,
    "있음": 1,
    "없음": 0,
    "True": 1,
    "False": 0,
    "TRUE": 1,
    "FALSE": 0,
    "1": 1,
    "0": 0,
    1: 1,
    0: 0,
    True: 1,
    False: 0,
}

TEAM_FEATS = [
    "배아생성효율",
    "ICSI수정효율",
    "배아이식비율",
    "배아저장비율",
    "난자활용률",
    "난자대비이식배아수",
    "이식배아수_구간",
    "전체임신률",
    "IVF임신률",
    "DI임신률",
    "임신유지율",
    "IVF임신유지율",
    "총실패횟수",
    "IVF실패횟수",
    "반복IVF실패_여부",
    "클리닉집중도",
    "IVF시술비율",
    "임신경험있음",
    "출산경험있음",
    "나이_제곱",
    "나이_임상구간",
    "고령_여부",
    "초고령_여부",
    "극고령_여부",
    "나이X총시술",
    "나이XIVF실패",
    "나이XIVF임신률",
    "초고령X반복실패",
    "복합위험도점수",
]

V17_FROZEN_FEATS = [
    "is_frozen_transfer",
    "thaw_to_transfer_ratio",
    "frozen_x_age",
    "frozen_x_clinic_exp",
    "frozen_x_stored",
    "frozen_day_interaction",
    "frozen_single_embryo",
    "frozen_x_day5plus",
]

PRUNED_DOMAIN_FEATS = [
    "male_factor_score",
    "female_factor_score",
    "severe_sperm_factor_flag",
    "unexplained_only_flag",
    "donor_any_flag",
    "effective_oocyte_age",
    "advanced_oocyte_age_flag",
    "advanced_age_autologous_flag",
    "advanced_age_donor_oocyte_flag",
    "retrieved_oocytes_total",
    "insemination_rate_from_retrieved",
    "fertilization_rate_all",
    "embryo_yield_per_retrieved",
    "transferable_embryos",
    "transferable_rate",
    "freeze_to_transfer_ratio_domain",
    "thaw_survival_proxy",
    "same_or_early_transfer_like",
    "extended_culture_flag_domain",
    "repeat_failure_x_transferable",
    "ivf_failure_x_effective_oocyte_age",
]

log_lines: List[str] = []


def log(msg: str = ""):
    print(msg)
    log_lines.append(msg)


def save_log():
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))


def safe_div_arr(a, b, fill=0.0):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    return np.where((b > 0) & np.isfinite(a) & np.isfinite(b), a / b, fill)


def parse_korean_count(x):
    if pd.isna(x):
        return np.nan
    x = str(x).strip()
    if re.fullmatch(r"\d+", x):
        return float(x)
    m = re.search(r"(\d+)", x)
    return float(m.group(1)) if m else np.nan


def age_to_numeric(x):
    if pd.isna(x):
        return np.nan
    x = str(x).strip()
    if x in ["알 수 없음", "미상", "불명", "unknown", "Unknown"]:
        return np.nan
    nums = re.findall(r"\d+", x)
    if len(nums) >= 2:
        return (float(nums[0]) + float(nums[1])) / 2
    if len(nums) == 1:
        return float(nums[0])
    return np.nan


def normalize_binary_col(df, col):
    if col not in df.columns:
        return

    s = df[col]
    if pd.api.types.is_numeric_dtype(s):
        df[col] = pd.to_numeric(s, errors="coerce").fillna(0).astype(int)
        return

    s_str = s.astype(str).str.strip()
    mapped = s_str.map(BINARY_MAP)
    numeric = pd.to_numeric(s_str, errors="coerce")
    out = np.where(pd.notna(mapped), mapped, numeric)
    df[col] = pd.to_numeric(out, errors="coerce").fillna(0).astype(int)


def get_num(df, col):
    if col not in df.columns:
        return pd.Series(0.0, index=df.index)
    return pd.to_numeric(df[col], errors="coerce").fillna(0.0)


def get_str(df, col):
    if col not in df.columns:
        return pd.Series(["Unknown"] * len(df), index=df.index)
    return df[col].fillna("Unknown").astype(str)


def prune_subgroup_constants(X_sub):
    """subgroup 내에서 값이 완전히 하나뿐인(nunique == 1) 컬럼만 제거.
    준상수(99% 동일)는 건드리지 않음 — 막판에 과하게 잘라낼 위험 방지."""
    drop_cols = []
    for col in X_sub.columns:
        if X_sub[col].nunique(dropna=False) <= 1:
            drop_cols.append(col)
    return drop_cols


def raw_binary_to_array(s):
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").fillna(0).astype(int).values
    s_str = s.astype(str).str.strip()
    mapped = s_str.map(BINARY_MAP)
    numeric = pd.to_numeric(s_str, errors="coerce")
    out = pd.Series(np.where(pd.notna(mapped), mapped, numeric), index=s.index)
    return pd.to_numeric(out, errors="coerce").fillna(0).astype(int).values


def detect_gpu():
    try:
        m = CatBoostClassifier(
            iterations=1, task_type="GPU", devices="0", verbose=False
        )
        m.fit(
            pd.DataFrame({"a": [0, 1, 0, 1], "b": [1, 0, 1, 0]}),
            [0, 1, 0, 1],
            verbose=False,
        )
        return "GPU"
    except Exception:
        return "CPU"


# ============================================================
# pruned domain features
# ============================================================
def add_domain_features_pruned(df: pd.DataFrame):
    df = df.copy()
    added = []

    def add(name, values):
        df[name] = values
        added.append(name)

    zero = pd.Series(0.0, index=df.index)

    age = get_num(df, "시술 당시 나이_num")
    egg_donor_age = get_num(df, "난자 기증자 나이_num")

    embryo = get_num(df, "이식된 배아 수")
    stored = get_num(df, "저장된 배아 수")
    thawed_embryo = get_num(df, "해동된 배아 수")
    total_embryo = get_num(df, "총 생성 배아 수")

    fresh_oocyte = get_num(df, "수집된 신선 난자 수")
    thawed_oocyte = get_num(df, "해동 난자 수")
    mixed_oocyte = get_num(df, "혼합된 난자 수")

    tr_day = get_num(df, "배아 이식 경과일")

    egg_src = get_str(df, "난자 출처")
    sperm_src = get_str(df, "정자 출처")
    donor_embryo_use = get_num(df, "기증 배아 사용 여부")

    total_fail = get_num(df, "총실패횟수") if "총실패횟수" in df.columns else zero
    ivf_fail = get_num(df, "IVF실패횟수") if "IVF실패횟수" in df.columns else zero

    male_cols = [
        "남성 주 불임 원인",
        "남성 부 불임 원인",
        "불임 원인 - 남성 요인",
        "불임 원인 - 정자 농도",
        "불임 원인 - 정자 면역학적 요인",
        "불임 원인 - 정자 운동성",
        "불임 원인 - 정자 형태",
    ]
    female_cols = [
        "여성 주 불임 원인",
        "여성 부 불임 원인",
        "불임 원인 - 여성 요인",
        "불임 원인 - 난관 질환",
        "불임 원인 - 배란 장애",
        "불임 원인 - 자궁경부 문제",
        "불임 원인 - 자궁내막증",
    ]
    sperm_sub_cols = [
        "불임 원인 - 정자 농도",
        "불임 원인 - 정자 면역학적 요인",
        "불임 원인 - 정자 운동성",
        "불임 원인 - 정자 형태",
    ]

    male_factor_score = zero.copy()
    for c in male_cols:
        if c in df.columns:
            male_factor_score += get_num(df, c)

    female_factor_score = zero.copy()
    for c in female_cols:
        if c in df.columns:
            female_factor_score += get_num(df, c)

    sperm_issue_score = zero.copy()
    for c in sperm_sub_cols:
        if c in df.columns:
            sperm_issue_score += get_num(df, c)

    unexplained = get_num(df, "불명확 불임 원인")

    donor_oocyte_flag = egg_src.eq("기증 제공").astype(int)
    autologous_oocyte_flag = egg_src.eq("본인 제공").astype(int)
    donor_sperm_flag = sperm_src.eq("기증 제공").astype(int)
    donor_any_flag = (
        (donor_oocyte_flag > 0) | (donor_sperm_flag > 0) | (donor_embryo_use > 0)
    ).astype(int)

    effective_oocyte_age = np.where(
        (donor_oocyte_flag > 0) & (egg_donor_age > 0),
        egg_donor_age,
        age,
    )
    effective_oocyte_age = pd.Series(effective_oocyte_age, index=df.index).astype(float)

    retrieved_oocytes_total = fresh_oocyte + thawed_oocyte
    transferable_embryos = embryo + stored

    add("male_factor_score", male_factor_score)
    add("female_factor_score", female_factor_score)
    add("severe_sperm_factor_flag", (sperm_issue_score >= 2).astype(int))
    add(
        "unexplained_only_flag",
        (
            (unexplained > 0) & (male_factor_score == 0) & (female_factor_score == 0)
        ).astype(int),
    )
    add("donor_any_flag", donor_any_flag)
    add("effective_oocyte_age", effective_oocyte_age)
    add("advanced_oocyte_age_flag", (effective_oocyte_age >= 38).astype(int))
    add(
        "advanced_age_autologous_flag",
        ((age >= 40) & (autologous_oocyte_flag > 0)).astype(int),
    )
    add(
        "advanced_age_donor_oocyte_flag",
        ((age >= 40) & (donor_oocyte_flag > 0)).astype(int),
    )
    add("retrieved_oocytes_total", retrieved_oocytes_total)
    add(
        "insemination_rate_from_retrieved",
        safe_div_arr(mixed_oocyte, retrieved_oocytes_total),
    )
    add("fertilization_rate_all", safe_div_arr(total_embryo, mixed_oocyte))
    add(
        "embryo_yield_per_retrieved",
        safe_div_arr(total_embryo, retrieved_oocytes_total),
    )
    add("transferable_embryos", transferable_embryos)
    add("transferable_rate", safe_div_arr(transferable_embryos, total_embryo))
    add("freeze_to_transfer_ratio_domain", safe_div_arr(stored, embryo + 1))
    add("thaw_survival_proxy", safe_div_arr(embryo, thawed_embryo))
    add("same_or_early_transfer_like", ((tr_day > 0) & (tr_day <= 3)).astype(int))
    add("extended_culture_flag_domain", (tr_day >= 5).astype(int))
    add("repeat_failure_x_transferable", total_fail * transferable_embryos)
    add("ivf_failure_x_effective_oocyte_age", ivf_fail * effective_oocyte_age)

    return df, added


# ============================================================
# preprocess
# ============================================================
def preprocess_full(df, is_train=True, target_col="임신 성공 여부"):
    df = df.copy()

    id_col = df.columns[0]
    drop = [id_col]
    if is_train and target_col in df.columns:
        drop.append(target_col)
    df = df.drop(columns=drop, errors="ignore")

    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = df[c].fillna("Unknown").astype(str)

    for cc in COUNT_COLS:
        if cc in df.columns:
            df[f"{cc}_num"] = df[cc].apply(parse_korean_count)

    for ac in AGE_COLS:
        if ac in df.columns:
            df[f"{ac}_num"] = df[ac].apply(age_to_numeric)

    for fc in FLAG_COLS:
        normalize_binary_col(df, fc)

    for c in df.select_dtypes(exclude=["object"]).columns:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    def has(*cols):
        return all(c in df.columns for c in cols)

    # ---------------- team features ----------------
    if has("총 생성 배아 수", "수집된 신선 난자 수"):
        df["배아생성효율"] = safe_div_arr(
            df["총 생성 배아 수"].values, df["수집된 신선 난자 수"].values
        )
    if has("미세주입에서 생성된 배아 수", "미세주입된 난자 수"):
        df["ICSI수정효율"] = safe_div_arr(
            df["미세주입에서 생성된 배아 수"].values, df["미세주입된 난자 수"].values
        )
    if has("이식된 배아 수", "총 생성 배아 수"):
        df["배아이식비율"] = safe_div_arr(
            df["이식된 배아 수"].values, df["총 생성 배아 수"].values
        )
    if has("저장된 배아 수", "총 생성 배아 수"):
        df["배아저장비율"] = safe_div_arr(
            df["저장된 배아 수"].values, df["총 생성 배아 수"].values
        )
    if has("미세주입된 난자 수", "수집된 신선 난자 수"):
        df["난자활용률"] = safe_div_arr(
            df["미세주입된 난자 수"].values, df["수집된 신선 난자 수"].values
        )
    if has("이식된 배아 수", "수집된 신선 난자 수"):
        df["난자대비이식배아수"] = safe_div_arr(
            df["이식된 배아 수"].values, df["수집된 신선 난자 수"].values
        )

    if "이식된 배아 수" in df.columns:
        emb = pd.to_numeric(df["이식된 배아 수"], errors="coerce").fillna(0).values
        tmp = pd.cut(pd.Series(emb), bins=[-1, 0, 1, 2, 100], labels=[0, 1, 2, 3])
        df["이식배아수_구간"] = (
            pd.to_numeric(tmp, errors="coerce").fillna(0).astype(int)
        )

    if has("총 임신 횟수_num", "총 시술 횟수_num"):
        df["전체임신률"] = safe_div_arr(
            df["총 임신 횟수_num"].values, df["총 시술 횟수_num"].values
        )
    if has("IVF 임신 횟수_num", "IVF 시술 횟수_num"):
        df["IVF임신률"] = safe_div_arr(
            df["IVF 임신 횟수_num"].values, df["IVF 시술 횟수_num"].values
        )
    if has("DI 임신 횟수_num", "DI 시술 횟수_num"):
        df["DI임신률"] = safe_div_arr(
            df["DI 임신 횟수_num"].values, df["DI 시술 횟수_num"].values
        )
    if has("총 출산 횟수_num", "총 임신 횟수_num"):
        df["임신유지율"] = safe_div_arr(
            df["총 출산 횟수_num"].values, df["총 임신 횟수_num"].values
        )
    if has("IVF 출산 횟수_num", "IVF 임신 횟수_num"):
        df["IVF임신유지율"] = safe_div_arr(
            df["IVF 출산 횟수_num"].values, df["IVF 임신 횟수_num"].values
        )
    if has("총 시술 횟수_num", "총 임신 횟수_num"):
        df["총실패횟수"] = np.maximum(
            df["총 시술 횟수_num"].values - df["총 임신 횟수_num"].values, 0
        )
    if has("IVF 시술 횟수_num", "IVF 임신 횟수_num"):
        df["IVF실패횟수"] = np.maximum(
            df["IVF 시술 횟수_num"].values - df["IVF 임신 횟수_num"].values, 0
        )
    if "IVF실패횟수" in df.columns:
        df["반복IVF실패_여부"] = (df["IVF실패횟수"] >= 3).astype(int)
    if has("클리닉 내 총 시술 횟수_num", "총 시술 횟수_num"):
        df["클리닉집중도"] = safe_div_arr(
            df["클리닉 내 총 시술 횟수_num"].values, df["총 시술 횟수_num"].values
        )
    if has("IVF 시술 횟수_num", "총 시술 횟수_num"):
        df["IVF시술비율"] = safe_div_arr(
            df["IVF 시술 횟수_num"].values, df["총 시술 횟수_num"].values
        )
    if "총 임신 횟수_num" in df.columns:
        df["임신경험있음"] = (df["총 임신 횟수_num"] > 0).astype(int)
    if "총 출산 횟수_num" in df.columns:
        df["출산경험있음"] = (df["총 출산 횟수_num"] > 0).astype(int)

    if "시술 당시 나이_num" in df.columns:
        age = df["시술 당시 나이_num"].values
        df["나이_제곱"] = age**2
        tmp = pd.cut(
            pd.Series(age), bins=[0, 35, 40, 45, 100], labels=[0, 1, 2, 3], right=False
        )
        df["나이_임상구간"] = pd.to_numeric(tmp, errors="coerce").fillna(0).astype(int)
        df["고령_여부"] = (age >= 35).astype(int)
        df["초고령_여부"] = (age >= 40).astype(int)
        df["극고령_여부"] = (age >= 42).astype(int)

    if has("시술 당시 나이_num", "총 시술 횟수_num"):
        df["나이X총시술"] = (
            df["시술 당시 나이_num"].values * df["총 시술 횟수_num"].values
        )
    if has("시술 당시 나이_num", "IVF실패횟수"):
        df["나이XIVF실패"] = df["시술 당시 나이_num"].values * df["IVF실패횟수"].values
    if has("시술 당시 나이_num", "IVF임신률"):
        df["나이XIVF임신률"] = df["시술 당시 나이_num"].values * df["IVF임신률"].values
    if has("초고령_여부", "반복IVF실패_여부"):
        df["초고령X반복실패"] = df["초고령_여부"].values * df["반복IVF실패_여부"].values

    risk = np.zeros(len(df))
    for rc in ["고령_여부", "초고령_여부", "반복IVF실패_여부"]:
        if rc in df.columns:
            risk += df[rc].values
    if "임신경험있음" in df.columns:
        risk += 1 - df["임신경험있음"].values
    df["복합위험도점수"] = risk

    # ---------------- v17 common features ----------------
    embryo = get_num(df, "이식된 배아 수").values
    total_emb = get_num(df, "총 생성 배아 수").values
    stored = get_num(df, "저장된 배아 수").values
    mixed = get_num(df, "혼합된 난자 수").values
    tr_day = get_num(df, "배아 이식 경과일").values
    thawed = get_num(df, "해동된 배아 수").values
    fresh_use = get_num(df, "신선 배아 사용 여부").values
    frozen_use = get_num(df, "동결 배아 사용 여부").values
    micro_tr = get_num(df, "미세주입 배아 이식 수").values
    micro_emb = get_num(df, "미세주입에서 생성된 배아 수").values
    age_num = get_num(df, "시술 당시 나이_num").values
    clinic_num = get_num(df, "클리닉 내 총 시술 횟수_num").values

    df["실제이식여부"] = (embryo > 0).astype(int)

    if "시술 유형" in df.columns:
        df["is_ivf"] = (df["시술 유형"].astype(str) == "IVF").astype(int)
        df["is_di"] = (df["시술 유형"].astype(str) == "DI").astype(int)

    df["total_embryo_ratio"] = safe_div_arr(embryo + stored + thawed, total_emb + 1)
    df["수정_성공률"] = safe_div_arr(total_emb, mixed)
    df["배아_이용률"] = safe_div_arr(embryo, total_emb)
    df["배아_잉여율"] = safe_div_arr(stored, total_emb)

    if "시술 유형" in df.columns:
        ivf_mask = (df["시술 유형"].astype(str) == "IVF").values
        df["ivf_transfer_ratio"] = 0.0
        df["ivf_storage_ratio"] = 0.0
        if ivf_mask.sum() > 0:
            df.loc[ivf_mask, "ivf_transfer_ratio"] = safe_div_arr(
                embryo[ivf_mask], total_emb[ivf_mask]
            )
            df.loc[ivf_mask, "ivf_storage_ratio"] = safe_div_arr(
                stored[ivf_mask], total_emb[ivf_mask]
            )

    day5plus = (tr_day >= 5).astype(int)
    df["transfer_day_optimal"] = np.isin(tr_day, [3, 5]).astype(int)
    day_cat_map = {0: 0, 1: 1, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 5}
    df["transfer_day_cat"] = (
        pd.Series(tr_day).map(day_cat_map).fillna(0).astype(int).values
    )
    df["embryo_day_interaction"] = embryo * tr_day
    df["fresh_transfer_ratio"] = safe_div_arr(
        embryo * fresh_use.astype(float), total_emb + 1
    )
    df["micro_transfer_quality"] = safe_div_arr(micro_tr, micro_emb + 1)
    df["transfer_intensity"] = safe_div_arr(embryo, np.clip(age_num, 1, None))
    df["age_transfer_interaction"] = np.clip(age_num, 0, None) * embryo
    df["age_x_single_transfer"] = np.clip(age_num, 0, None) * (embryo == 1).astype(int)
    df["day5plus"] = day5plus
    df["single_x_day5plus"] = ((embryo == 1) & (tr_day >= 5)).astype(int)
    df["multi_x_day5plus"] = ((embryo >= 2) & (tr_day >= 5)).astype(int)
    df["fresh_x_day5plus"] = ((fresh_use > 0) & (tr_day >= 5)).astype(int)
    df["age_x_day5plus"] = np.clip(age_num, 0, None) * day5plus
    df["blastocyst_signal"] = (tr_day >= 5).astype(int)

    if "난자 출처" in df.columns:
        df["donor_egg_x_advanced_age"] = (
            (df["난자 출처"].astype(str) == "기증 제공") & (age_num >= 40)
        ).astype(int)

    df["is_frozen_transfer"] = ((frozen_use > 0) & (embryo > 0)).astype(int)
    df["thaw_to_transfer_ratio"] = safe_div_arr(embryo, thawed + 0.001) * (
        thawed > 0
    ).astype(float)
    df["frozen_x_age"] = frozen_use.astype(float) * np.clip(age_num, 0, None)
    df["frozen_x_clinic_exp"] = frozen_use.astype(float) * clinic_num
    df["frozen_x_stored"] = frozen_use.astype(float) * stored
    df["frozen_day_interaction"] = frozen_use.astype(float) * tr_day
    df["frozen_single_embryo"] = ((frozen_use > 0) & (embryo == 1)).astype(int)
    df["frozen_x_day5plus"] = ((frozen_use > 0) & (tr_day >= 5)).astype(int)

    inf_cols = [c for c in df.columns if c.startswith("불임 원인 -")]
    if len(inf_cols) > 0:
        df["infertility_count"] = df[inf_cols].sum(axis=1)
        df["has_infertility"] = (df["infertility_count"] > 0).astype(int)

    if "난자 출처" in df.columns and "정자 출처" in df.columns:
        egg_map = {"본인 제공": 0, "기증 제공": 1, "알 수 없음": 2, "Unknown": 2}
        sperm_map = {
            "배우자 제공": 0,
            "기증 제공": 1,
            "미할당": 2,
            "배우자 및 기증 제공": 3,
            "Unknown": 2,
        }
        df["egg_sperm_combo"] = df["난자 출처"].astype(str).map(egg_map).fillna(
            2
        ).astype(int) * 10 + df["정자 출처"].astype(str).map(sperm_map).fillna(
            2
        ).astype(
            int
        )

    if "총 시술 횟수_num" in df.columns:
        df["first_treatment"] = (df["총 시술 횟수_num"] == 0).astype(int)

    # ---------------- v22-lite features ----------------
    # created_minus_transferred 제거 (ablation)

    # embryo_transfer_days_x_age (나이 × 이식 경과일 교호작용) — 유지
    # 기존 age_transfer_interaction = age × embryo수 와 다른 축
    df["embryo_transfer_days_x_age"] = tr_day * age_num

    # ---------------- pruned domain ----------------
    df, added_domain = add_domain_features_pruned(df)

    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    return df, cat_cols, added_domain


# ============================================================
# external global blend
# ============================================================
def load_external_global_blend():
    oof_sum = None
    test_sum = None
    loaded = []

    for ver, w in EXTERNAL_GLOBAL_BLEND.items():
        oof_path = EXTERNAL_OOF_PATHS.get(ver, "")
        test_path = EXTERNAL_TEST_PATHS.get(ver, "")
        if (not os.path.exists(oof_path)) or (not os.path.exists(test_path)):
            print(f"  [WARN] {ver} npy 없음: oof={oof_path}, test={test_path}")
            return None, None, loaded

        oof_arr = np.load(oof_path)
        test_arr = np.load(test_path)

        if oof_sum is None:
            oof_sum = np.zeros_like(oof_arr, dtype=float)
            test_sum = np.zeros_like(test_arr, dtype=float)

        oof_sum += w * oof_arr
        test_sum += w * test_arr
        loaded.append((ver, w, oof_path, test_path))

    return oof_sum, test_sum, loaded


# ============================================================
# training util
# ============================================================
def train_cb_subgroup_ensemble(
    X_train: pd.DataFrame,
    y: np.ndarray,
    X_test: pd.DataFrame,
    cat_idx: List[int],
    seeds: List[int],
    n_folds: int,
    task_type: str,
    subgroup_mask: np.ndarray,
    tag: str,
):
    """
    subgroup_mask=True 인 train 행만 사용해 모델 학습
    반환:
      oof_full: 전체 train 길이, subgroup 위치만 값 존재, 나머지 NaN
      test_pred: 전체 test 길이
      imp_avg: 평균 feature importance
      seed_aucs: subgroup 내부 OOF AUC 목록
    """
    assert len(subgroup_mask) == len(y)

    sub_idx = np.where(subgroup_mask)[0]
    X_sub = X_train.iloc[sub_idx].reset_index(drop=True)
    y_sub = y[sub_idx]

    oof_sub_avg = np.zeros(len(sub_idx), dtype=float)
    test_avg = np.zeros(len(X_test), dtype=float)
    seed_aucs = []

    feature_importance_sum = np.zeros(X_train.shape[1], dtype=float)
    total_models = len(seeds) * n_folds
    model_counter = 0

    log(f"\n### [{tag}] subgroup 학습")
    log(f"- n={len(sub_idx)}, pos={y_sub.mean()*100:.2f}%")

    for si, seed_val in enumerate(seeds):
        log(f"\n  --- {tag} Seed {seed_val} ({si+1}/{len(seeds)}) ---")
        oof_seed_sub = np.zeros(len(sub_idx), dtype=float)
        test_seed = np.zeros(len(X_test), dtype=float)

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed_val)
        fold_indices = list(skf.split(X_sub, y_sub))

        for fold_i, (tr_local, va_local) in enumerate(fold_indices):
            model_counter += 1
            t_fold = time.time()

            X_tr = X_sub.iloc[tr_local]
            X_va = X_sub.iloc[va_local]
            y_tr = y_sub[tr_local]
            y_va = y_sub[va_local]

            params = CB_PARAMS.copy()
            params["random_seed"] = seed_val * 100 + fold_i
            params["task_type"] = task_type
            if task_type == "GPU":
                params["devices"] = "0"

            model = CatBoostClassifier(**params)
            model.fit(
                X_tr,
                y_tr,
                eval_set=(X_va, y_va),
                cat_features=cat_idx,
                verbose=False,
            )

            va_pred = model.predict_proba(X_va)[:, 1]
            te_pred = model.predict_proba(X_test)[:, 1]

            oof_seed_sub[va_local] = va_pred
            test_seed += te_pred / n_folds

            fold_auc = roc_auc_score(y_va, va_pred)
            elapsed = (time.time() - t_fold) / 60
            best_iter = (
                model.best_iteration_ if model.best_iteration_ is not None else -1
            )

            log(
                f"    Fold {fold_i+1}: AUC={fold_auc:.4f}, iter={best_iter}, "
                f"소요={elapsed:.1f}분 [{model_counter}/{total_models}]"
            )

            fold_imp = model.get_feature_importance()
            if fold_imp is not None and len(fold_imp) == X_train.shape[1]:
                feature_importance_sum += fold_imp

        seed_auc = roc_auc_score(y_sub, oof_seed_sub)
        seed_aucs.append(seed_auc)
        oof_sub_avg += oof_seed_sub / len(seeds)
        test_avg += test_seed / len(seeds)
        log(f"  {tag} Seed {seed_val} OOF AUC: {seed_auc:.4f}")

    oof_full = np.full(len(y), np.nan, dtype=float)
    oof_full[sub_idx] = oof_sub_avg
    imp_avg = feature_importance_sum / total_models

    final_auc = roc_auc_score(y_sub, oof_sub_avg)
    log(f"\n  === [{tag}] 최종 subgroup OOF AUC: {final_auc:.6f} ===")
    log(f"  seed별: {[f'{a:.4f}' for a in seed_aucs]}")

    return oof_full, test_avg, imp_avg, seed_aucs, sub_idx


# ============================================================
# blend search
# ============================================================
def _blend_eval(
    y, global_oof, transfer_oof, frozen_oof,
    non_frozen_transfer, frozen_mask,
    a, b, c,
):
    """하나의 (a, b, c) 조합에 대한 예측 + 평가."""
    pred = global_oof.copy()
    pred[non_frozen_transfer] = (
        (1 - a) * global_oof[non_frozen_transfer]
        + a * transfer_oof[non_frozen_transfer]
    )
    pred[frozen_mask] = (
        (1 - b - c) * global_oof[frozen_mask]
        + b * transfer_oof[frozen_mask]
        + c * frozen_oof[frozen_mask]
    )
    auc = roc_auc_score(y, pred)
    ll = log_loss(y, np.clip(pred, 1e-7, 1 - 1e-7))
    ap = average_precision_score(y, pred)
    return auc, ll, ap, pred


def search_expert_blend(
    y: np.ndarray,
    global_oof: np.ndarray,
    transfer_oof: np.ndarray,
    frozen_oof: np.ndarray,
    transfer_mask: np.ndarray,
    frozen_mask: np.ndarray,
):
    """
    2-phase search:
      Phase 1 (coarse): 0.05 grid
      Phase 2 (fine):   coarse best ±0.05, 0.01 grid
    """
    non_frozen_transfer = transfer_mask & ~frozen_mask
    best = None

    def _update_best(a, b, c):
        nonlocal best
        if b + c > 1:
            return
        auc, ll, ap, pred = _blend_eval(
            y, global_oof, transfer_oof, frozen_oof,
            non_frozen_transfer, frozen_mask, a, b, c,
        )
        if (best is None) or (auc > best["auc"]):
            best = {
                "a_transfer_nonfrozen": float(a),
                "b_transfer_frozen": float(b),
                "c_frozen_frozen": float(c),
                "auc": float(auc),
                "logloss": float(ll),
                "ap": float(ap),
                "pred": pred.copy(),
            }

    # Phase 1: coarse search (step 0.05)
    grid_a = np.arange(0.00, 0.61, 0.05)
    grid_b = np.arange(0.00, 0.71, 0.05)
    grid_c = np.arange(0.00, 0.71, 0.05)

    for a in grid_a:
        for b in grid_b:
            for c in grid_c:
                _update_best(a, b, c)

    coarse_best = best.copy()
    log(f"  [coarse] best a={coarse_best['a_transfer_nonfrozen']:.2f}, "
        f"b={coarse_best['b_transfer_frozen']:.2f}, "
        f"c={coarse_best['c_frozen_frozen']:.2f}, "
        f"AUC={coarse_best['auc']:.6f}")

    # Phase 2: fine search (step 0.01) around coarse best ±0.05
    ca = coarse_best["a_transfer_nonfrozen"]
    cb = coarse_best["b_transfer_frozen"]
    cc = coarse_best["c_frozen_frozen"]

    fine_a = np.arange(max(0.0, ca - 0.05), min(1.001, ca + 0.06), 0.01)
    fine_b = np.arange(max(0.0, cb - 0.05), min(1.001, cb + 0.06), 0.01)
    fine_c = np.arange(max(0.0, cc - 0.05), min(1.001, cc + 0.06), 0.01)

    for a in fine_a:
        for b in fine_b:
            for c in fine_c:
                _update_best(a, b, c)

    log(f"  [fine]   best a={best['a_transfer_nonfrozen']:.2f}, "
        f"b={best['b_transfer_frozen']:.2f}, "
        f"c={best['c_frozen_frozen']:.2f}, "
        f"AUC={best['auc']:.6f}")

    return best


def apply_expert_blend_to_test(
    global_test: np.ndarray,
    transfer_test: np.ndarray,
    frozen_test: np.ndarray,
    transfer_mask_test: np.ndarray,
    frozen_mask_test: np.ndarray,
    weights: Dict[str, float],
):
    a = weights["a_transfer_nonfrozen"]
    b = weights["b_transfer_frozen"]
    c = weights["c_frozen_frozen"]

    pred = global_test.copy()

    non_frozen_transfer = transfer_mask_test & ~frozen_mask_test
    pred[non_frozen_transfer] = (1 - a) * global_test[
        non_frozen_transfer
    ] + a * transfer_test[non_frozen_transfer]

    pred[frozen_mask_test] = (
        (1 - b - c) * global_test[frozen_mask_test]
        + b * transfer_test[frozen_mask_test]
        + c * frozen_test[frozen_mask_test]
    )
    return pred


# ============================================================
# main
# ============================================================
log(f"# {VERSION} - global blend + transfer expert + frozen expert")
log(f"시각: {NOW}")
log("=" * 60)

# [1] load
log("\n## [1] 데이터 로드")
t0 = time.time()

train_raw = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test_raw = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
sub = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

target_col = "임신 성공 여부"
y = train_raw[target_col].values

log(f"- train: {train_raw.shape}, test: {test_raw.shape}")
log(f"- 타겟: 0={np.sum(y==0)}, 1={np.sum(y==1)}, 양성비율={np.mean(y)*100:.1f}%")

# [2] preprocess
log("\n## [2] 전처리 + 피처 엔지니어링")
t_fe = time.time()

log("  전처리 중...")
X_train, cat_train, domain_added_train = preprocess_full(
    train_raw, is_train=True, target_col=target_col
)
X_test, cat_test, _ = preprocess_full(test_raw, is_train=False, target_col=target_col)

common_cols = [c for c in X_train.columns if c in X_test.columns]
X_train = X_train[common_cols].copy()
X_test = X_test[common_cols].copy()

constant_cols = [
    c
    for c in X_train.columns
    if X_train[c].nunique(dropna=False) <= 1 and X_test[c].nunique(dropna=False) <= 1
]
if len(constant_cols) > 0:
    X_train = X_train.drop(columns=constant_cols)
    X_test = X_test.drop(columns=constant_cols)

cat_found = [c for c in X_train.columns if c in set(cat_train) and c in set(cat_test)]
cat_idx = [X_train.columns.get_loc(c) for c in cat_found]

team_found = [f for f in TEAM_FEATS if f in X_train.columns]
v17_found = [f for f in V17_FROZEN_FEATS if f in X_train.columns]
domain_found = [f for f in PRUNED_DOMAIN_FEATS if f in X_train.columns]

log(f"- 피처 수: {len(X_train.columns)}, 카테고리: {len(cat_idx)}")
log(f"- 카테고리({len(cat_found)}개): {cat_found}")
log(f"- 팀원 피처: {len(team_found)}개")
log(f"- v17 frozen 피처: {len(v17_found)}개")
log(f"- pruned domain 피처: {len(domain_found)}개")
log(f"- 제거된 상수 컬럼 ({len(constant_cols)}개): {constant_cols}")
log(f"- FE 소요: {(time.time()-t_fe)/60:.1f}분")

pd.DataFrame({"feature": X_train.columns}).to_csv(
    os.path.join(RESULT_DIR, f"feature_list_{VERSION}.csv"),
    index=False,
    encoding="utf-8-sig",
)

# [3] subgroup masks
log("\n## [3] subgroup 통계")
transfer_mask_train = (
    pd.to_numeric(train_raw["이식된 배아 수"], errors="coerce").fillna(0).values > 0
)
transfer_mask_test = (
    pd.to_numeric(test_raw["이식된 배아 수"], errors="coerce").fillna(0).values > 0
)

if "동결 배아 사용 여부" in train_raw.columns:
    frozen_train = raw_binary_to_array(train_raw["동결 배아 사용 여부"]) > 0
else:
    frozen_train = np.zeros(len(train_raw), dtype=bool)

if "동결 배아 사용 여부" in test_raw.columns:
    frozen_test = raw_binary_to_array(test_raw["동결 배아 사용 여부"]) > 0
else:
    frozen_test = np.zeros(len(test_raw), dtype=bool)

frozen_transfer_train = transfer_mask_train & frozen_train
frozen_transfer_test = transfer_mask_test & frozen_test

log(
    f"- transfer: {transfer_mask_train.sum()}건, pos={y[transfer_mask_train].mean()*100:.2f}%"
)
log(
    f"- non-transfer: {(~transfer_mask_train).sum()}건, pos={y[~transfer_mask_train].mean()*100:.2f}%"
)
log(
    f"- frozen transfer: {frozen_transfer_train.sum()}건, pos={y[frozen_transfer_train].mean()*100:.2f}%"
)

task_type = "CPU"  # GPU 감지 생략, CPU 강제 사용
log(f"- Task type: {task_type}")

# [4] global base
log("\n## [4] global base 준비")
global_oof = None
global_test = None
global_source = ""

if USE_EXTERNAL_GLOBAL_BLEND:
    ext_oof, ext_test, loaded = load_external_global_blend()
    if ext_oof is not None:
        global_oof = ext_oof
        global_test = ext_test
        global_source = "external_blend"
        log("- 외부 global blend 사용")
        for ver, w, oof_path, test_path in loaded:
            log(f"  - {ver}: weight={w}, oof={oof_path}, test={test_path}")
        log(f"- external global OOF AUC: {roc_auc_score(y, global_oof):.6f}")
        log(
            f"- external global LogLoss: {log_loss(y, np.clip(global_oof, 1e-7, 1-1e-7)):.6f}"
        )
        log(f"- external global AP: {average_precision_score(y, global_oof):.6f}")

if global_oof is None:
    log("- 외부 blend 파일 미존재 또는 사용 실패 -> 내부 global 모델 fallback")
    # fallback: global도 현재 feature로 직접 학습
    oof_g, test_g, imp_g, seed_aucs_g, _ = train_cb_subgroup_ensemble(
        X_train=X_train,
        y=y,
        X_test=X_test,
        cat_idx=cat_idx,
        seeds=SEEDS,
        n_folds=N_FOLDS,
        task_type=task_type,
        subgroup_mask=np.ones(len(y), dtype=bool),
        tag="global",
    )
    global_oof = oof_g.copy()
    global_test = test_g.copy()
    global_source = "internal_global"

np.save(os.path.join(RESULT_DIR, f"oof_{VERSION}_global_base.npy"), global_oof)
np.save(os.path.join(RESULT_DIR, f"test_{VERSION}_global_base.npy"), global_test)

# [4.5] expert-specific feature pruning (Step 1 + Step 2)
log("\n## [4.5] expert-specific feature pruning")

# --- Transfer expert: subgroup constant pruning ---
transfer_X_sub = X_train.iloc[np.where(transfer_mask_train)[0]]
transfer_drop = prune_subgroup_constants(transfer_X_sub)
log(f"- transfer expert: subgroup 상수 제거 {len(transfer_drop)}개: {transfer_drop}")

X_train_transfer = X_train.drop(columns=transfer_drop)
X_test_transfer = X_test.drop(columns=transfer_drop)
cat_found_transfer = [c for c in cat_found if c in X_train_transfer.columns]
cat_idx_transfer = [X_train_transfer.columns.get_loc(c) for c in cat_found_transfer]
log(f"- transfer expert 피처 수: {len(X_train_transfer.columns)} (원본: {len(X_train.columns)})")

# --- Frozen expert: subgroup constant pruning + frozen tail drop ---
frozen_X_sub = X_train.iloc[np.where(frozen_transfer_train)[0]]
frozen_drop = prune_subgroup_constants(frozen_X_sub)
log(f"- frozen expert: subgroup 상수 제거 {len(frozen_drop)}개: {frozen_drop}")

# Step 2: frozen tail 제거
frozen_tail_drop = ["frozen_single_embryo", "frozen_x_day5plus"]
frozen_extra_drop = [
    c for c in frozen_tail_drop
    if c in X_train.columns and c not in frozen_drop
]
frozen_all_drop = list(set(frozen_drop + frozen_extra_drop))
log(f"- frozen expert: tail 추가 제거 {len(frozen_extra_drop)}개: {frozen_extra_drop}")
log(f"- frozen expert: 총 제거 {len(frozen_all_drop)}개: {frozen_all_drop}")

X_train_frozen = X_train.drop(columns=frozen_all_drop)
X_test_frozen = X_test.drop(columns=frozen_all_drop)
cat_found_frozen = [c for c in cat_found if c in X_train_frozen.columns]
cat_idx_frozen = [X_train_frozen.columns.get_loc(c) for c in cat_found_frozen]
log(f"- frozen expert 피처 수: {len(X_train_frozen.columns)} (원본: {len(X_train.columns)})")

# 피처 이름 저장 (importance 로깅용)
transfer_feature_names = X_train_transfer.columns.tolist()
frozen_feature_names = X_train_frozen.columns.tolist()

pd.DataFrame({"feature": transfer_feature_names}).to_csv(
    os.path.join(RESULT_DIR, f"feature_list_{VERSION}_transfer.csv"),
    index=False, encoding="utf-8-sig",
)
pd.DataFrame({"feature": frozen_feature_names}).to_csv(
    os.path.join(RESULT_DIR, f"feature_list_{VERSION}_frozen.csv"),
    index=False, encoding="utf-8-sig",
)

# [5] transfer expert
log("\n## [5] transfer expert 학습")
t_train_transfer = time.time()
transfer_oof, transfer_test_pred, transfer_imp, transfer_seed_aucs, transfer_idx = (
    train_cb_subgroup_ensemble(
        X_train=X_train_transfer,
        y=y,
        X_test=X_test_transfer,
        cat_idx=cat_idx_transfer,
        seeds=SEEDS,
        n_folds=N_FOLDS,
        task_type=task_type,
        subgroup_mask=transfer_mask_train,
        tag="transfer",
    )
)
log(f"- transfer expert 학습 소요: {(time.time()-t_train_transfer)/60:.1f}분")

# [6] frozen expert
log("\n## [6] frozen expert 학습")
t_train_frozen = time.time()
frozen_oof, frozen_test_pred, frozen_imp, frozen_seed_aucs, frozen_idx = (
    train_cb_subgroup_ensemble(
        X_train=X_train_frozen,
        y=y,
        X_test=X_test_frozen,
        cat_idx=cat_idx_frozen,
        seeds=SEEDS,
        n_folds=N_FOLDS,
        task_type=task_type,
        subgroup_mask=frozen_transfer_train,
        tag="frozen",
    )
)
log(f"- frozen expert 학습 소요: {(time.time()-t_train_frozen)/60:.1f}분")

np.save(
    os.path.join(RESULT_DIR, f"oof_{VERSION}_transfer.npy"),
    np.nan_to_num(transfer_oof, nan=-1),
)
np.save(os.path.join(RESULT_DIR, f"test_{VERSION}_transfer.npy"), transfer_test_pred)
np.save(
    os.path.join(RESULT_DIR, f"oof_{VERSION}_frozen.npy"),
    np.nan_to_num(frozen_oof, nan=-1),
)
np.save(os.path.join(RESULT_DIR, f"test_{VERSION}_frozen.npy"), frozen_test_pred)

# sanity
assert np.all(np.isfinite(global_oof))
assert np.all(np.isfinite(global_test))
assert np.all(np.isfinite(transfer_oof[transfer_mask_train]))
assert np.all(np.isfinite(frozen_oof[frozen_transfer_train]))

# [7] blend search
log("\n## [7] expert blend weight 탐색")
t_blend = time.time()

best_blend = search_expert_blend(
    y=y,
    global_oof=global_oof,
    transfer_oof=transfer_oof,
    frozen_oof=frozen_oof,
    transfer_mask=transfer_mask_train,
    frozen_mask=frozen_transfer_train,
)
final_oof = best_blend["pred"]

best_weights = {
    "a_transfer_nonfrozen": best_blend["a_transfer_nonfrozen"],
    "b_transfer_frozen": best_blend["b_transfer_frozen"],
    "c_frozen_frozen": best_blend["c_frozen_frozen"],
}

log(f"- best weights: {best_weights}")
log(f"- blended OOF AUC: {best_blend['auc']:.6f}")
log(f"- blended OOF LogLoss: {best_blend['logloss']:.6f}")
log(f"- blended OOF AP: {best_blend['ap']:.6f}")
log(f"- blend 탐색 소요: {(time.time()-t_blend)/60:.1f}분")

final_test = apply_expert_blend_to_test(
    global_test=global_test,
    transfer_test=transfer_test_pred,
    frozen_test=frozen_test_pred,
    transfer_mask_test=transfer_mask_test,
    frozen_mask_test=frozen_transfer_test,
    weights=best_weights,
)

np.save(os.path.join(RESULT_DIR, f"oof_{VERSION}_final.npy"), final_oof)
np.save(os.path.join(RESULT_DIR, f"test_{VERSION}_final.npy"), final_test)
np.save(os.path.join(RESULT_DIR, f"y_train_{VERSION}.npy"), y)

with open(
    os.path.join(RESULT_DIR, f"blend_weights_{VERSION}.json"), "w", encoding="utf-8"
) as f:
    json.dump(
        {
            "global_source": global_source,
            "external_global_blend": (
                EXTERNAL_GLOBAL_BLEND if global_source == "external_blend" else None
            ),
            "best_weights": best_weights,
            "metrics": {
                "auc": best_blend["auc"],
                "logloss": best_blend["logloss"],
                "ap": best_blend["ap"],
            },
        },
        f,
        ensure_ascii=False,
        indent=2,
    )

# [8] subgroup AUC
log("\n## [8] 그룹별 AUC 분석")

groups = {
    "transfer": transfer_mask_train,
    "non-transfer": ~transfer_mask_train,
    "frozen transfer": frozen_transfer_train,
    "fresh transfer": transfer_mask_train & ~frozen_train,
}

if "시술 유형" in train_raw.columns:
    groups["DI"] = (train_raw["시술 유형"].astype(str) == "DI").values
if "난자 출처" in train_raw.columns:
    groups["donor egg"] = (train_raw["난자 출처"].astype(str) == "기증 제공").values

age_col = "시술 당시 나이"
if age_col in train_raw.columns:
    for av in [
        "만18-34세",
        "만35-37세",
        "만38-39세",
        "만40-42세",
        "만43-44세",
        "만45-50세",
    ]:
        groups[f"age_{av}"] = (train_raw[age_col].astype(str) == av).values

global_auc_full = roc_auc_score(y, global_oof)

for gname, gmask in groups.items():
    if gmask.sum() > 50 and len(np.unique(y[gmask])) > 1:
        g_auc_global = roc_auc_score(y[gmask], global_oof[gmask])
        g_auc_final = roc_auc_score(y[gmask], final_oof[gmask])
        delta = g_auc_final - g_auc_global
        log(
            f"  {gname}: "
            f"global={g_auc_global:.4f} -> final={g_auc_final:.4f} "
            f"(delta={delta:+.5f}, n={gmask.sum()}, pos={y[gmask].mean()*100:.1f}%)"
        )

# transfer/frozen expert 자체 성능 참고
if np.isfinite(transfer_oof[transfer_mask_train]).all():
    transfer_only_auc = roc_auc_score(
        y[transfer_mask_train], transfer_oof[transfer_mask_train]
    )
    log(f"\n- transfer expert 자체 subgroup AUC: {transfer_only_auc:.6f}")

if (
    np.isfinite(frozen_oof[frozen_transfer_train]).all()
    and frozen_transfer_train.sum() > 50
):
    frozen_only_auc = roc_auc_score(
        y[frozen_transfer_train], frozen_oof[frozen_transfer_train]
    )
    log(f"- frozen expert 자체 subgroup AUC: {frozen_only_auc:.6f}")

# [9] 종합 평가지표
log("\n## [9] 종합 평가지표")

global_ll = log_loss(y, np.clip(global_oof, 1e-7, 1 - 1e-7))
global_ap = average_precision_score(y, global_oof)

final_auc = roc_auc_score(y, final_oof)
final_ll = log_loss(y, np.clip(final_oof, 1e-7, 1 - 1e-7))
final_ap = average_precision_score(y, final_oof)

log("### global base")
log(f"  OOF AUC:      {global_auc_full:.6f}")
log(f"  OOF LogLoss:  {global_ll:.6f}")
log(f"  OOF AP:       {global_ap:.6f}")

log("\n### final expert blend")
log(f"  OOF AUC:      {final_auc:.6f}")
log(f"  OOF LogLoss:  {final_ll:.6f}")
log(f"  OOF AP:       {final_ap:.6f}")

log("\n### improvement")
log(f"  AUC delta:      {final_auc - global_auc_full:+.6f}")
log(f"  LogLoss delta:  {final_ll - global_ll:+.6f}")
log(f"  AP delta:       {final_ap - global_ap:+.6f}")

log("\n### Threshold별 분류 지표 (final)")
log(f"  {'Th':>6}  {'Acc':>7}  {'Prec':>7}  {'Rec':>7}  {'F1':>7}  {'Spec':>7}")
best_f1, best_th = 0, 0.25

for th in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
    yb = (final_oof >= th).astype(int)
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

# [10] 제출 파일
log("\n## [10] 제출 파일")

sub_main = sub.copy()
sub_main["probability"] = final_test
sub_path = os.path.join(RESULT_DIR, f"sample_submission_{VERSION}_{NOW}.csv")
sub_main.to_csv(sub_path, index=False)

log(f"- 파일: {sub_path}")
log(
    f"- 확률: mean={final_test.mean():.4f}, std={final_test.std():.4f}, "
    f"min={final_test.min():.6f}, max={final_test.max():.4f}"
)

# [11] 피처 중요도
log("\n## [11] expert 피처 중요도")

global_feature_names = X_train.columns.tolist()
cat_set = set(cat_found)
team_set = set([f for f in TEAM_FEATS if f in X_train.columns])
frozen_v17_set = set([f for f in V17_FROZEN_FEATS if f in X_train.columns])
domain_set = set([f for f in PRUNED_DOMAIN_FEATS if f in X_train.columns])


def log_importance_block(imp_arr, tag, feat_names, topn=30):
    if imp_arr is None or len(imp_arr) != len(feat_names):
        log(f"\n### {tag} 중요도: imp_arr 길이 불일치 (arr={len(imp_arr) if imp_arr is not None else 'None'}, names={len(feat_names)})")
        return None

    imp_df = pd.DataFrame(
        {"feature": feat_names, "importance": imp_arr}
    ).sort_values("importance", ascending=False)

    log(f"\n### {tag} 중요도 상위 {topn} (피처 {len(feat_names)}개)")
    for rank_i, (_, row) in enumerate(imp_df.head(topn).iterrows(), 1):
        marks = ""
        if row["feature"] in cat_set:
            marks += " [cat]"
        if row["feature"] in frozen_v17_set:
            marks += " ★v17"
        elif row["feature"] in team_set:
            marks += " ★team"
        elif row["feature"] in domain_set:
            marks += " ★domain"

        log(f"  {rank_i}. {row['feature']}{marks}: {row['importance']:.4f}")

    zero_imp = imp_df[imp_df["importance"] == 0]["feature"].tolist()
    log(
        f"### {tag} 중요도 0 피처 ({len(zero_imp)}개): {zero_imp[:25]}{' ...' if len(zero_imp) > 25 else ''}"
    )

    save_path = os.path.join(RESULT_DIR, f"feature_importance_{VERSION}_{tag}.csv")
    imp_df.to_csv(save_path, index=False, encoding="utf-8-sig")
    log(f"- 저장: {save_path}")
    return imp_df


transfer_imp_df = log_importance_block(transfer_imp, "transfer_expert", transfer_feature_names, topn=35)
frozen_imp_df = log_importance_block(frozen_imp, "frozen_expert", frozen_feature_names, topn=35)

# [12] 버전 비교
log("\n## [12] 버전 비교")
log("| 버전 | 모델 | OOF AUC | Test AUC | 비고 |")
log("|------|------|---------|----------|------|")
log("| v1 | CB원본 | 0.7403 | - | 베이스라인 |")
log("| v9 | XGB+CB 3seed | 0.7405 | 0.74166 | +IVF/DI |")
log("| v15 | CB 5seed+Optuna | 0.7407 | 0.74169 | EDA+HP |")
log("| v17 | CB 5seed | 0.7408 | TBD | 팀원피처+frozen+CW |")
log("| v19 | CB 5seed | 0.7408 | TBD | +domain 81개 |")
log("| v20_pruned | CB 5seed | 0.7408 | TBD | +domain pruned 21개 |")
log("| v21_expert | expert blend | 0.740946 | 0.7424 | global+transfer+frozen |")
log(
    f"| **{VERSION}** | **expert blend refined** | **{final_auc:.6f}** | **TBD** | **+prune+core feats+fine blend** |"
)

# [13] 최종 요약
total_time = (time.time() - t0) / 60
log(f"\n{'='*60}")
log("## 최종 요약")
log(f"{'='*60}")
log("- v22_expert 핵심 변경 (vs v21):")
log(f"  1. global base source: {global_source}")
if global_source == "external_blend":
    log(f"  2. external global blend: {EXTERNAL_GLOBAL_BLEND}")
else:
    log("  2. external blend 실패로 internal global fallback 사용")
log("  3. transfer expert: subgroup 상수 피처 자동 제거")
log(f"     - 제거 {len(transfer_drop)}개: {transfer_drop}")
log("  4. frozen expert: subgroup 상수 + frozen tail 피처 제거")
log(f"     - 제거 {len(frozen_all_drop)}개: {frozen_all_drop}")
log("  5. 신규 피처: created_minus_transferred, embryo_transfer_days_x_age")
log("  6. blend weight fine search (coarse 0.05 → fine 0.01)")
log(f"  7. best weights: {best_weights}")
log(f"- global base OOF AUC: {global_auc_full:.6f}")
log(f"- final expert blend OOF AUC: {final_auc:.6f}")
log(f"- AUC 개선: {final_auc - global_auc_full:+.6f}")
log(f"- final OOF LogLoss: {final_ll:.6f}")
log(f"- final OOF AP: {final_ap:.6f}")
log(f"- transfer expert seed별: {[f'{a:.4f}' for a in transfer_seed_aucs]}")
log(f"- frozen expert seed별: {[f'{a:.4f}' for a in frozen_seed_aucs]}")
log(f"- 전체 피처 수: {len(X_train.columns)} (cat={len(cat_idx)})")
log(f"- transfer expert 피처: {len(transfer_feature_names)}개")
log(f"- frozen expert 피처: {len(frozen_feature_names)}개")
log(f"- 전역 상수 제거: {len(constant_cols)}개")
log(f"- 총 소요: {total_time:.1f}분")
log(f"- 로그: {LOG_PATH}")
log(f"{'='*60}")

save_log()
print(f"\n완료! 로그: {LOG_PATH}")
