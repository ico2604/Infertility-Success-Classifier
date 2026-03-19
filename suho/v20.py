#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v20-pruned - v17 + 선별 도메인 피처(prune-only)
================================================================
핵심 변경:
  1. v17의 팀원/XGB 피처 + frozen 피처 유지
  2. v19의 도메인 피처 81개를 21개로 선별(pruned)
  3. boolean/flag 컬럼 robust 정규화 유지
  4. categorical 자동 탐지 유지
  5. CatBoost importance를 25개 모델 평균으로 집계

목표:
  - v19의 "넓은 확장" 대신
  - "효과 있었던 축만 남긴 최소 확장"으로 성능 비교
"""

import os
import re
import time
import datetime
import warnings
import numpy as np
import pandas as pd

from typing import List
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
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
RESULT_DIR = os.path.join(BASE_DIR, "result")
os.makedirs(RESULT_DIR, exist_ok=True)

VERSION = "v20_pruned"
NOW = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_PATH = os.path.join(RESULT_DIR, f"log_{VERSION}.md")

SEEDS = [42, 2026, 2604, 123, 777]
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
# 전처리
# ============================================================
def preprocess_full(df, is_train=True, target_col="임신 성공 여부"):
    df = df.copy()

    id_col = df.columns[0]
    drop = [id_col]
    if is_train and target_col in df.columns:
        drop.append(target_col)
    df = df.drop(columns=drop, errors="ignore")

    # object 결측
    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = df[c].fillna("Unknown").astype(str)

    # 횟수형 _num
    for cc in COUNT_COLS:
        if cc in df.columns:
            df[f"{cc}_num"] = df[cc].apply(parse_korean_count)

    # 나이형 _num
    for ac in AGE_COLS:
        if ac in df.columns:
            df[f"{ac}_num"] = df[ac].apply(age_to_numeric)

    # bool/flag 정규화
    for fc in FLAG_COLS:
        normalize_binary_col(df, fc)

    # 수치 결측
    for c in df.select_dtypes(exclude=["object"]).columns:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    def has(*cols):
        return all(c in df.columns for c in cols)

    # --------------------------------------------------------
    # 팀원/XGB 피처
    # --------------------------------------------------------
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
        bins = [-1, 0, 1, 2, 100]
        labels = [0, 1, 2, 3]
        tmp = pd.cut(pd.Series(emb), bins=bins, labels=labels)
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
        bins_age = [0, 35, 40, 45, 100]
        labels_age = [0, 1, 2, 3]
        tmp = pd.cut(pd.Series(age), bins=bins_age, labels=labels_age, right=False)
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

    # --------------------------------------------------------
    # v17 공통 피처
    # --------------------------------------------------------
    embryo_col = "이식된 배아 수"
    total_col = "총 생성 배아 수"
    stored_col = "저장된 배아 수"
    mixed_col = "혼합된 난자 수"
    tr_day_col = "배아 이식 경과일"
    thawed_col = "해동된 배아 수"
    fresh_use_col = "신선 배아 사용 여부"
    frozen_use_col = "동결 배아 사용 여부"
    micro_tr_col = "미세주입 배아 이식 수"
    micro_emb_col = "미세주입에서 생성된 배아 수"
    type_col = "시술 유형"
    egg_src_col = "난자 출처"

    embryo = get_num(df, embryo_col).values
    total_emb = get_num(df, total_col).values
    stored = get_num(df, stored_col).values
    mixed = get_num(df, mixed_col).values
    tr_day = get_num(df, tr_day_col).values
    thawed = get_num(df, thawed_col).values
    fresh_use = get_num(df, fresh_use_col).values
    frozen_use = get_num(df, frozen_use_col).values
    micro_tr = get_num(df, micro_tr_col).values
    micro_emb = get_num(df, micro_emb_col).values
    age_num = get_num(df, "시술 당시 나이_num").values
    clinic_num = get_num(df, "클리닉 내 총 시술 횟수_num").values

    df["실제이식여부"] = (embryo > 0).astype(int)

    if type_col in df.columns:
        df["is_ivf"] = (df[type_col].astype(str) == "IVF").astype(int)
        df["is_di"] = (df[type_col].astype(str) == "DI").astype(int)

    df["total_embryo_ratio"] = safe_div_arr(embryo + stored + thawed, total_emb + 1)
    df["수정_성공률"] = safe_div_arr(total_emb, mixed)
    df["배아_이용률"] = safe_div_arr(embryo, total_emb)
    df["배아_잉여율"] = safe_div_arr(stored, total_emb)

    if type_col in df.columns:
        ivf_mask = (df[type_col].astype(str) == "IVF").values
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

    if egg_src_col in df.columns:
        df["donor_egg_x_advanced_age"] = (
            (df[egg_src_col].astype(str) == "기증 제공") & (age_num >= 40)
        ).astype(int)

    # v17 frozen transfer 전용
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

    # 불임 원인 개수
    inf_cols = [c for c in df.columns if c.startswith("불임 원인 -")]
    if len(inf_cols) > 0:
        df["infertility_count"] = df[inf_cols].sum(axis=1)
        df["has_infertility"] = (df["infertility_count"] > 0).astype(int)

    # egg_sperm combo
    if egg_src_col in df.columns and "정자 출처" in df.columns:
        egg_map = {"본인 제공": 0, "기증 제공": 1, "알 수 없음": 2, "Unknown": 2}
        sperm_map = {
            "배우자 제공": 0,
            "기증 제공": 1,
            "미할당": 2,
            "배우자 및 기증 제공": 3,
            "Unknown": 2,
        }
        df["egg_sperm_combo"] = df[egg_src_col].astype(str).map(egg_map).fillna(
            2
        ).astype(int) * 10 + df["정자 출처"].astype(str).map(sperm_map).fillna(
            2
        ).astype(
            int
        )

    if "총 시술 횟수_num" in df.columns:
        df["first_treatment"] = (df["총 시술 횟수_num"] == 0).astype(int)

    # --------------------------------------------------------
    # v20 pruned domain features
    # --------------------------------------------------------
    df, added_domain = add_domain_features_pruned(df)

    # categorical 자동 탐지
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    return df, cat_cols, added_domain


# ============================================================
# 메인 실행부
# ============================================================
log(f"# {VERSION} - v17 + 선별 도메인 피처")
log(f"시각: {NOW}")
log("=" * 60)

# ============================================================
# [1] 데이터 로드
# ============================================================
log("\n## [1] 데이터 로드")
t0 = time.time()

train_raw = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test_raw = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
sub = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

target_col = "임신 성공 여부"
y = train_raw[target_col].values

log(f"- train: {train_raw.shape}, test: {test_raw.shape}")
log(f"- 타겟: 0={np.sum(y==0)}, 1={np.sum(y==1)}, 양성비율={np.mean(y)*100:.1f}%")

# ============================================================
# [2] 전처리 + 피처 엔지니어링
# ============================================================
log("\n## [2] 전처리 + 피처 엔지니어링")
t_fe = time.time()

log("  전처리 중...")
X_train, cat_train, domain_added_train = preprocess_full(
    train_raw, is_train=True, target_col=target_col
)
X_test, cat_test, domain_added_test = preprocess_full(
    test_raw, is_train=False, target_col=target_col
)

# 컬럼 동기화
common_cols = [c for c in X_train.columns if c in X_test.columns]
X_train = X_train[common_cols].copy()
X_test = X_test[common_cols].copy()

# 상수 컬럼 제거(비교 안정화용)
constant_cols = [
    c
    for c in X_train.columns
    if X_train[c].nunique(dropna=False) <= 1 and X_test[c].nunique(dropna=False) <= 1
]
if len(constant_cols) > 0:
    X_train = X_train.drop(columns=constant_cols)
    X_test = X_test.drop(columns=constant_cols)

# categorical 자동 동기화
cat_found = [c for c in X_train.columns if c in set(cat_train) and c in set(cat_test)]
cat_idx = [X_train.columns.get_loc(c) for c in cat_found]

team_found = [f for f in TEAM_FEATS if f in X_train.columns]
v17_found = [f for f in V17_FROZEN_FEATS if f in X_train.columns]
domain_found = [f for f in PRUNED_DOMAIN_FEATS if f in X_train.columns]

log(f"- 피처 수: {len(X_train.columns)}, 카테고리: {len(cat_idx)}")
log(f"- 카테고리({len(cat_found)}개): {cat_found}")
log(f"- v17 frozen 피처 ({len(v17_found)}개): {v17_found}")
log(f"- 팀원 XGB 피처 ({len(team_found)}개): {team_found}")
log(f"- v20 pruned domain 피처 ({len(domain_found)}개): {domain_found}")
log(
    f"- 제거된 상수 컬럼 ({len(constant_cols)}개): {constant_cols[:20]}{' ...' if len(constant_cols) > 20 else ''}"
)
log(f"- 피처 엔지니어링 소요: {(time.time()-t_fe)/60:.1f}분")

pd.DataFrame({"feature": X_train.columns}).to_csv(
    os.path.join(RESULT_DIR, f"feature_list_{VERSION}.csv"),
    index=False,
    encoding="utf-8-sig",
)

# ============================================================
# [3] subgroup 통계
# ============================================================
log("\n## [3] subgroup 통계")

embryo_raw = (
    pd.to_numeric(train_raw["이식된 배아 수"], errors="coerce").fillna(0).values
)
transfer_mask_train = embryo_raw > 0

frozen_col = "동결 배아 사용 여부"
if frozen_col in train_raw.columns:
    frozen_train = raw_binary_to_array(train_raw[frozen_col]) > 0
else:
    frozen_train = np.zeros(len(y), dtype=bool)

frozen_transfer_train = transfer_mask_train & frozen_train

log(
    f"- 이식: {transfer_mask_train.sum()}건, 양성률={y[transfer_mask_train].mean()*100:.1f}%"
)
log(
    f"- 비이식: {(~transfer_mask_train).sum()}건, 양성률={y[~transfer_mask_train].mean()*100:.2f}%"
)
log(
    f"- frozen transfer: {frozen_transfer_train.sum()}건, 양성률={y[frozen_transfer_train].mean()*100:.1f}%"
)

# ============================================================
# [4] CatBoost 5-seed 앙상블
# ============================================================
log("\n## [4] CatBoost 5-seed 앙상블")
log("### 파라미터:")
for k, v in CB_PARAMS.items():
    if k != "verbose":
        log(f"  {k}: {v}")
log(f"### Seeds: {SEEDS}")

task_type = detect_gpu()
log(f"### Task type: {task_type}")

oof_cb = np.zeros(len(y))
test_cb = np.zeros(len(X_test))
seed_aucs = []

total_models = len(SEEDS) * N_FOLDS
model_count = 0
t_train = time.time()

feature_importance_sum = np.zeros(len(X_train.columns), dtype=float)

for si, seed_val in enumerate(SEEDS):
    log(f"\n  --- Seed {seed_val} ({si+1}/{len(SEEDS)}) ---")
    oof_seed = np.zeros(len(y))
    test_seed = np.zeros(len(X_test))

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed_val)
    fold_indices = list(skf.split(X_train, y))

    for fold_i, (tr_idx, va_idx) in enumerate(fold_indices):
        model_count += 1
        t_fold = time.time()

        X_tr = X_train.iloc[tr_idx]
        X_va = X_train.iloc[va_idx]
        y_tr = y[tr_idx]
        y_va = y[va_idx]

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

        oof_seed[va_idx] = va_pred
        test_seed += te_pred / N_FOLDS

        va_auc = roc_auc_score(y_va, va_pred)
        best_iter = model.best_iteration_ if model.best_iteration_ is not None else -1
        elapsed = (time.time() - t_fold) / 60

        log(
            f"    Fold {fold_i+1}: AUC={va_auc:.4f}, iter={best_iter}, "
            f"소요={elapsed:.1f}분 [{model_count}/{total_models}]"
        )

        fold_imp = model.get_feature_importance()
        if fold_imp is not None and len(fold_imp) == len(X_train.columns):
            feature_importance_sum += fold_imp

    seed_auc = roc_auc_score(y, oof_seed)
    seed_aucs.append(seed_auc)
    log(f"  Seed {seed_val} OOF AUC: {seed_auc:.4f}")

    oof_cb += oof_seed / len(SEEDS)
    test_cb += test_seed / len(SEEDS)

    np.save(os.path.join(RESULT_DIR, f"oof_{VERSION}_seed{seed_val}.npy"), oof_seed)
    save_log()

train_time = (time.time() - t_train) / 60
cb_auc = roc_auc_score(y, oof_cb)
cb_importances = feature_importance_sum / total_models

log(f"\n  === CatBoost 5-seed OOF AUC: {cb_auc:.6f} ===")
log(f"  개별 seed: {[f'{a:.4f}' for a in seed_aucs]}")
log(f"  학습 소요: {train_time:.1f}분")

np.save(os.path.join(RESULT_DIR, f"oof_{VERSION}_cb.npy"), oof_cb)
np.save(os.path.join(RESULT_DIR, f"test_{VERSION}_cb.npy"), test_cb)

# ============================================================
# [5] 그룹별 AUC 분석
# ============================================================
log("\n## [5] 그룹별 AUC 분석")

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

for gname, gmask in groups.items():
    if gmask.sum() > 50 and len(np.unique(y[gmask])) > 1:
        g_auc = roc_auc_score(y[gmask], oof_cb[gmask])
        log(
            f"  {gname}: AUC={g_auc:.4f} (n={gmask.sum()}, pos={y[gmask].mean()*100:.1f}%)"
        )

# ============================================================
# [6] 종합 평가지표
# ============================================================
log("\n## [6] 종합 평가지표")

oof_ll = log_loss(y, np.clip(oof_cb, 1e-7, 1 - 1e-7))
oof_ap = average_precision_score(y, oof_cb)

log(f"  OOF AUC:      {cb_auc:.6f}")
log(f"  OOF LogLoss:  {oof_ll:.6f}")
log(f"  OOF AP:       {oof_ap:.6f}")

log("\n### Threshold별 분류 지표")
log(f"  {'Th':>6}  {'Acc':>7}  {'Prec':>7}  {'Rec':>7}  {'F1':>7}  {'Spec':>7}")
best_f1, best_th = 0, 0.25

for th in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
    yb = (oof_cb >= th).astype(int)
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

# ============================================================
# [7] 제출 파일
# ============================================================
log("\n## [7] 제출 파일")

sub_main = sub.copy()
sub_main["probability"] = test_cb
sub_path = os.path.join(RESULT_DIR, f"sample_submission_{VERSION}_{NOW}.csv")
sub_main.to_csv(sub_path, index=False)

log(f"- 파일: {sub_path}")
log(
    f"- 확률: mean={test_cb.mean():.4f}, std={test_cb.std():.4f}, "
    f"min={test_cb.min():.6f}, max={test_cb.max():.4f}"
)

np.save(os.path.join(RESULT_DIR, f"oof_{VERSION}_final.npy"), oof_cb)
np.save(os.path.join(RESULT_DIR, f"test_{VERSION}_final.npy"), test_cb)
np.save(os.path.join(RESULT_DIR, f"y_train_{VERSION}.npy"), y)

# ============================================================
# [8] 피처 중요도
# ============================================================
log("\n## [8] CatBoost 평균 피처 중요도 (상위 40)")

if cb_importances is not None and len(cb_importances) == len(X_train.columns):
    feature_names = X_train.columns.tolist()
    imp_df = pd.DataFrame(
        {"feature": feature_names, "importance": cb_importances}
    ).sort_values("importance", ascending=False)

    cat_set = set(cat_found)
    team_set = set(team_found)
    frozen_set = set(v17_found)
    domain_set = set(domain_found)

    for rank_i, (_, row) in enumerate(imp_df.head(40).iterrows(), 1):
        marks = ""
        if row["feature"] in cat_set:
            marks += " [cat]"
        if row["feature"] in frozen_set:
            marks += " ★v17"
        elif row["feature"] in team_set:
            marks += " ★team"
        elif row["feature"] in domain_set:
            marks += " ★domain"

        log(f"  {rank_i}. {row['feature']}{marks}: {row['importance']:.4f}")

    log("\n### v20 pruned domain 피처 중요도")
    domain_imp = imp_df[imp_df["feature"].isin(domain_set)]
    for _, row in domain_imp.iterrows():
        rank = imp_df.index.get_loc(row.name) + 1
        log(f"  {row['feature']}: {row['importance']:.4f} (#{rank})")

    log("\n### v17 frozen 피처 중요도")
    for f in v17_found:
        if f in imp_df["feature"].values:
            rr = imp_df[imp_df["feature"] == f].iloc[0]
            rank = imp_df.index.get_loc(rr.name) + 1
            log(f"  {f}: {rr['importance']:.4f} (#{rank})")

    zero_imp = imp_df[imp_df["importance"] == 0]["feature"].tolist()
    log(
        f"\n### 중요도 0 피처 ({len(zero_imp)}개): {zero_imp[:30]}{' ...' if len(zero_imp) > 30 else ''}"
    )

    imp_df.to_csv(
        os.path.join(RESULT_DIR, f"feature_importance_{VERSION}.csv"),
        index=False,
        encoding="utf-8-sig",
    )

# ============================================================
# [9] 버전 비교
# ============================================================
log("\n## [9] 버전 비교")
log("| 버전 | 모델 | OOF AUC | Test AUC | 비고 |")
log("|------|------|---------|----------|------|")
log("| v1 | CB원본 | 0.7403 | - | 베이스라인 |")
log("| v9 | XGB+CB 3seed | 0.7405 | 0.74166 | +IVF/DI |")
log("| v15 | CB 5seed+Optuna | 0.7407 | 0.74169 | EDA+HP |")
log("| v17 | CB 5seed | 0.7408 | TBD | 팀원피처+frozen+CW |")
log("| v19 | CB 5seed | 0.7408 | TBD | +domain 81개 |")
log(
    f"| **{VERSION}** | **CB 5seed** | **{cb_auc:.4f}** | **TBD** | **+domain pruned 21개** |"
)

# ============================================================
# [10] 최종 요약
# ============================================================
total_time = (time.time() - t0) / 60
log(f"\n{'='*60}")
log("## 최종 요약")
log(f"{'='*60}")
log("- v20-pruned 핵심 변경:")
log(f"  1. 팀원 XGB 의료 파생변수 {len(team_found)}개 유지")
log(f"  2. frozen transfer 전용 피처 {len(v17_found)}개 유지")
log(f"  3. 도메인 피처를 81개 → {len(domain_found)}개로 선별")
log("  4. broad expansion 대신 prune-only 비교")
log("  5. categorical 자동 탐지 유지")
log("  6. CatBoost importance를 25개 모델 평균으로 집계")
log(f"- CB 5-seed OOF AUC: {cb_auc:.6f}")
log(f"  seed별: {[f'{a:.4f}' for a in seed_aucs]}")
log(f"- OOF LogLoss: {oof_ll:.6f}")
log(f"- OOF AP: {oof_ap:.6f}")
log(f"- 피처 수: {len(X_train.columns)} (cat={len(cat_idx)})")
log(f"- 상수 제거: {len(constant_cols)}개")
log("- 데이터 누수: 현재 제공 컬럼 기준 직접적 누수 피처는 미사용")
log(f"- 총 소요: {total_time:.1f}분")
log(f"- 로그: {LOG_PATH}")
log(f"{'='*60}")

save_log()
print(f"\n완료! 로그: {LOG_PATH}")
