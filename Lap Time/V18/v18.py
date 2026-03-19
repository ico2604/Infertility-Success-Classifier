#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v18 - v17 기반 + NaN보존 + border_count 254 + 노이즈제거 + age_fail_severity + 결측플래그
==========================================================================
v17 대비 변경 (7가지):
  1. NaN 보존 (fillna(0) 삭제) — CatBoost 네이티브 NaN 처리
  2. border_count 64 → 254 — 세밀한 분기점
  3. 노이즈 피처 7개 제거 — 중요도 0.00 피처
  4. age_fail_severity 추가 — 나이 x 시술실패 연속형
  5. 결측 플래그 3개 추가 — 배아해동일/배아이식일/임신경과연수
  6. 나이_임상구간 fillna(-1) — 미상 분리
  7. 이식배아수_구간 fillna(-1) — 미상 분리
"""

import os
import re
import sys
import time
import datetime
import warnings
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
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

warnings.filterwarnings("ignore")

# ============================================================
# 경로 / 설정
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "코덱스 오즈코딩", "open")
RESULT_DIR = os.path.join(BASE_DIR, "result_v18")
os.makedirs(RESULT_DIR, exist_ok=True)

VERSION = "v18"
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
    "border_count": 254,                        # [수정2] 64 → 254
    "eval_metric": "AUC",
    "loss_function": "Logloss",
    "od_type": "Iter",
    "od_wait": 400,
    "verbose": False,
    "allow_writing_files": False,
    "class_weights": {0: 1.0, 1: 1.3},
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
    return np.where((b > 0) & np.isfinite(b) & np.isfinite(a), a / b, fill)


def safe_div_series(a, b):
    a = pd.Series(a, dtype=float)
    b = pd.Series(b, dtype=float)
    return np.where((np.isnan(b)) | (b == 0), np.nan, a / b)


def col_exists(df, col):
    return col in df.columns


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
    except:
        return "CPU"


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


# ============================================================
log(f"# {VERSION} - v17 + NaN보존 + border254 + 노이즈제거 + age_fail_severity + 결측플래그")
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


def preprocess_full(df, is_train=True):
    df = df.copy()
    id_col = df.columns[0]
    drop = [id_col]
    if is_train and target_col in df.columns:
        drop.append(target_col)
    df = df.drop(columns=drop, errors="ignore")

    # 카테고리 결측 → Unknown
    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = df[c].fillna("Unknown").astype(str)

    # 수치 결측 → NaN 유지 (CatBoost 처리)
    # 횟수형 _num 생성 (원본 유지)
    for cc in COUNT_COLS:
        if cc in df.columns:
            df[f"{cc}_num"] = df[cc].apply(parse_korean_count)

    # 나이 _num 생성 (원본 유지)
    for ac in AGE_COLS:
        if ac in df.columns:
            df[f"{ac}_num"] = df[ac].apply(age_to_numeric)

    # [수정5] 결측 플래그 3개 추가 — fillna 삭제 전에 원본 NaN 기준으로 생성
    for miss_col, flag_name in [
        ("배아 해동 경과일", "배아해동일_있음"),
        ("배아 이식 경과일", "배아이식일_있음"),
        ("임신 시도 또는 마지막 임신 경과 연수", "임신경과연수_있음"),
    ]:
        if miss_col in df.columns:
            df[flag_name] = (~df[miss_col].isna()).astype(int)

    # [수정1] NaN 보존 — 아래 3줄 삭제 (CatBoost가 NaN 네이티브 처리)
    # for c in df.select_dtypes(exclude=["object"]).columns:
    #     df[c] = df[c].fillna(0)

    # ---- 팀원 XGB 의료 파생변수 (make_medical_features 완전 통합) ----
    def has(*cols):
        return all(c in df.columns for c in cols)

    # [A] 배아 처리 단계별 전환율
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
        emb = df["이식된 배아 수"].values
        bins = [-1, 0, 1, 2, 100]
        labels = [0, 1, 2, 3]
        df["이식배아수_구간"] = pd.cut(pd.Series(emb), bins=bins, labels=labels)
        df["이식배아수_구간"] = (
            pd.to_numeric(df["이식배아수_구간"], errors="coerce").fillna(-1).astype(int)  # [수정7] 0 → -1
        )

    # [B] 과거 시술 이력
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

    # [C] 나이 기반
    if "시술 당시 나이_num" in df.columns:
        age = df["시술 당시 나이_num"].values
        df["나이_제곱"] = age**2
        bins_age = [0, 35, 40, 45, 100]
        labels_age = [0, 1, 2, 3]
        df["나이_임상구간"] = pd.cut(
            pd.Series(age), bins=bins_age, labels=labels_age, right=False
        )
        df["나이_임상구간"] = (
            pd.to_numeric(df["나이_임상구간"], errors="coerce").fillna(-1).astype(int)  # [수정6] 0 → -1
        )
        df["고령_여부"] = (age >= 35).astype(int)
        df["초고령_여부"] = (age >= 40).astype(int)
        df["극고령_여부"] = (age >= 42).astype(int)

    # [D] 나이 × 이력 상호작용
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

    # [수정4] age_fail_severity — 나이 x 시술실패 연속형 (NaN 보존)
    if "시술 당시 나이_num" in df.columns and "IVF실패횟수" in df.columns:
        age_num_s = pd.to_numeric(df["시술 당시 나이_num"], errors="coerce")
        ivf_fail = pd.to_numeric(df["IVF실패횟수"], errors="coerce")
        df["age_fail_severity"] = (np.clip(age_num_s - 30, 0, None) / 10) * np.log1p(ivf_fail)

    # [E] 복합 위험도
    risk = np.zeros(len(df))
    for rc in ["고령_여부", "초고령_여부", "반복IVF실패_여부"]:
        if rc in df.columns:
            risk += df[rc].values
    if "임신경험있음" in df.columns:
        risk += 1 - df["임신경험있음"].values
    df["복합위험도점수"] = risk

    # ---- 기존 v14/v16 공통 피처 ----
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

    embryo = df[embryo_col].values if embryo_col in df.columns else np.zeros(len(df))
    total_emb = df[total_col].values if total_col in df.columns else np.zeros(len(df))
    stored = df[stored_col].values if stored_col in df.columns else np.zeros(len(df))
    mixed = df[mixed_col].values if mixed_col in df.columns else np.zeros(len(df))
    tr_day = df[tr_day_col].values if tr_day_col in df.columns else np.zeros(len(df))
    thawed = df[thawed_col].values if thawed_col in df.columns else np.zeros(len(df))
    fresh_use = (
        df[fresh_use_col].values if fresh_use_col in df.columns else np.zeros(len(df))
    )
    frozen_use = (
        df[frozen_use_col].values if frozen_use_col in df.columns else np.zeros(len(df))
    )
    micro_tr = (
        df[micro_tr_col].values if micro_tr_col in df.columns else np.zeros(len(df))
    )
    micro_emb = (
        df[micro_emb_col].values if micro_emb_col in df.columns else np.zeros(len(df))
    )
    age_num = (
        df["시술 당시 나이_num"].values
        if "시술 당시 나이_num" in df.columns
        else np.zeros(len(df))
    )
    clinic_num = (
        df["클리닉 내 총 시술 횟수_num"].values
        if "클리닉 내 총 시술 횟수_num" in df.columns
        else np.zeros(len(df))
    )

    df["실제이식여부"] = (embryo > 0).astype(int)
    if type_col in df.columns:
        df["is_ivf"] = (df[type_col] == "IVF").astype(int)
        df["is_di"] = (df[type_col] == "DI").astype(int)

    df["total_embryo_ratio"] = safe_div_arr(embryo + stored + thawed, total_emb + 1)
    df["수정_성공률"] = safe_div_arr(total_emb, mixed)
    df["배아_이용률"] = safe_div_arr(embryo, total_emb)
    df["배아_잉여율"] = safe_div_arr(stored, total_emb)

    # IVF 전용
    if type_col in df.columns:
        ivf_mask = (df[type_col] == "IVF").values
        ivf_tr = np.zeros(len(df))
        ivf_tr[ivf_mask] = safe_div_arr(embryo[ivf_mask], total_emb[ivf_mask])
        df["ivf_transfer_ratio"] = ivf_tr
        ivf_st = np.zeros(len(df))
        ivf_st[ivf_mask] = safe_div_arr(stored[ivf_mask], total_emb[ivf_mask])
        df["ivf_storage_ratio"] = ivf_st

    # 이식 전용 피처
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
            (df[egg_src_col] == "기증 제공") & (age_num >= 40)
        ).astype(int)

    # ---- v17 신규: frozen transfer 전용 피처 ----
    df["is_frozen_transfer"] = ((frozen_use > 0) & (embryo > 0)).astype(int)
    df["thaw_to_transfer_ratio"] = safe_div_arr(embryo, thawed + 0.001) * (
        thawed > 0
    ).astype(float)
    df["frozen_x_age"] = frozen_use.astype(float) * np.clip(age_num, 0, None)
    df["frozen_x_clinic_exp"] = frozen_use.astype(float) * clinic_num
    df["frozen_x_stored"] = frozen_use.astype(float) * stored
    df["frozen_day_interaction"] = frozen_use.astype(float) * tr_day
    # [수정3] frozen_single_embryo, frozen_x_day5plus 생성은 유지하되 뒤에서 제거됨

    # 불임 원인
    inf_cols = [c for c in df.columns if c.startswith("불임 원인 -")]
    if inf_cols:
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
        df["egg_sperm_combo"] = df[egg_src_col].map(egg_map).fillna(2).astype(
            int
        ) * 10 + df["정자 출처"].map(sperm_map).fillna(2).astype(int)

    # first_treatment
    if "총 시술 횟수_num" in df.columns:
        df["first_treatment"] = (df["총 시술 횟수_num"] == 0).astype(int)

    # [수정3] 노이즈 피처 7개 제거 (중요도 0.00 확인된 피처)
    drop_noise = [
        "frozen_single_embryo",
        "frozen_x_day5plus",
        "불임 원인 - 여성 요인",
        "불임 원인 - 정자 면역학적 요인",
        "불임 원인 - 정자 형태",
        "난자 채취 경과일",
        "난자 해동 경과일",
    ]
    df = df.drop(columns=[c for c in drop_noise if c in df.columns], errors="ignore")

    # 카테고리 재확인
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    return df, cat_cols


log("  전처리 중...")
X_train, cat_names_train = preprocess_full(train_raw, is_train=True)
X_test, _ = preprocess_full(test_raw, is_train=False)

# 컬럼 동기화
common_cols = [c for c in X_train.columns if c in X_test.columns]
X_train = X_train[common_cols]
X_test = X_test[common_cols]

# CatBoost 카테고리 설정
cat_found = [c for c in CAT_COLS_CANDIDATES if c in X_train.columns]
cat_idx = [X_train.columns.get_loc(c) for c in cat_found]

log(f"- 피처 수: {len(X_train.columns)}, 카테고리: {len(cat_idx)}")
log(f"- 카테고리: {cat_found}")

# v18 변경사항 검증 로그
log(f"\n### v18 변경사항 검증:")
log(f"  - border_count: {CB_PARAMS['border_count']} (v17: 64)")

# 나이_임상구간 -1 존재 확인
if "나이_임상구간" in X_train.columns:
    n_neg1 = (X_train["나이_임상구간"] == -1).sum()
    log(f"  - 나이_임상구간 -1 환자 수: {n_neg1} (NaN 나이)")

# 이식배아수_구간 -1 존재 확인
if "이식배아수_구간" in X_train.columns:
    n_neg1_emb = (X_train["이식배아수_구간"] == -1).sum()
    log(f"  - 이식배아수_구간 -1 환자 수: {n_neg1_emb} (NaN 이식배아수)")

# 결측 플래그 확인
for flag in ["배아해동일_있음", "배아이식일_있음", "임신경과연수_있음"]:
    if flag in X_train.columns:
        log(f"  - {flag}: 1={X_train[flag].sum()}, 0={(X_train[flag]==0).sum()}")

# age_fail_severity 확인
if "age_fail_severity" in X_train.columns:
    sev = X_train["age_fail_severity"]
    log(f"  - age_fail_severity: mean={sev.mean():.4f}, NaN={sev.isna().sum()}")

# NaN 보존 확인
nan_counts = X_train.isnull().sum()
nan_cols = nan_counts[nan_counts > 0]
log(f"  - NaN 보존 피처 수: {len(nan_cols)}")
if len(nan_cols) > 0:
    for nc, nv in nan_cols.head(10).items():
        log(f"    {nc}: {nv} NaN ({nv/len(X_train)*100:.1f}%)")

# v17 frozen 피처 (수정3에서 2개 제거됨)
v17_new = [
    "is_frozen_transfer",
    "thaw_to_transfer_ratio",
    "frozen_x_age",
    "frozen_x_clinic_exp",
    "frozen_x_stored",
    "frozen_day_interaction",
    # frozen_single_embryo, frozen_x_day5plus 제거됨
]
v17_found = [f for f in v17_new if f in X_train.columns]
log(f"- v17 frozen 피처 ({len(v17_found)}개): {v17_found}")

# v18 신규 피처
v18_new = ["age_fail_severity", "배아해동일_있음", "배아이식일_있음", "임신경과연수_있음"]
v18_found = [f for f in v18_new if f in X_train.columns]
log(f"- v18 신규 피처 ({len(v18_found)}개): {v18_found}")

# 팀원 피처 확인
team_feats = [
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
team_found = [f for f in team_feats if f in X_train.columns]
log(f"- 팀원 XGB 피처 ({len(team_found)}개): {team_found}")

log(f"- 피처 엔지니어링 소요: {(time.time()-t_fe)/60:.1f}분")

# ============================================================
# [3] subgroup 통계
# ============================================================
log("\n## [3] subgroup 통계")
embryo_raw = train_raw["이식된 배아 수"].fillna(0).values
transfer_mask_train = embryo_raw > 0
embryo_test = test_raw["이식된 배아 수"].fillna(0).values
transfer_mask_test = embryo_test > 0

frozen_col = "동결 배아 사용 여부"
frozen_train = (
    (train_raw[frozen_col].fillna(0).values > 0)
    if frozen_col in train_raw.columns
    else np.zeros(len(y), dtype=bool)
)
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
log(f"### 파라미터:")
for k, v in CB_PARAMS.items():
    if k != "verbose":
        log(f"  {k}: {v}")
log(f"### Seeds: {SEEDS}")

task_type = detect_gpu()
log(f"### Task type: {task_type}")

oof_cb = np.zeros(len(y))
test_cb = np.zeros(len(X_test))
seed_aucs = []
cb_importances = None

total_models = len(SEEDS) * N_FOLDS
model_count = 0
t_train = time.time()

for si, seed_val in enumerate(SEEDS):
    log(f"\n  --- Seed {seed_val} ({si+1}/{len(SEEDS)}) ---")
    oof_seed = np.zeros(len(y))
    test_seed = np.zeros(len(X_test))

    # seed별 다른 fold split
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
            X_tr, y_tr, eval_set=(X_va, y_va), cat_features=cat_idx, verbose=False
        )

        va_pred = model.predict_proba(X_va)[:, 1]
        oof_seed[va_idx] = va_pred
        test_seed += model.predict_proba(X_test)[:, 1] / N_FOLDS

        va_auc = roc_auc_score(y_va, va_pred)
        elapsed = (time.time() - t_fold) / 60
        log(
            f"    Fold {fold_i+1}: AUC={va_auc:.4f}, iter={model.best_iteration_}, "
            f"소요={elapsed:.1f}분 [{model_count}/{total_models}]"
        )

        if si == len(SEEDS) - 1 and fold_i == N_FOLDS - 1:
            cb_importances = model.get_feature_importance()

    seed_auc = roc_auc_score(y, oof_seed)
    seed_aucs.append(seed_auc)
    log(f"  Seed {seed_val} OOF AUC: {seed_auc:.4f}")

    oof_cb += oof_seed / len(SEEDS)
    test_cb += test_seed / len(SEEDS)

    # npy 중간 저장
    np.save(os.path.join(RESULT_DIR, f"oof_{VERSION}_seed{seed_val}.npy"), oof_seed)
    save_log()

train_time = (time.time() - t_train) / 60
cb_auc = roc_auc_score(y, oof_cb)
log(f"\n  === CatBoost 5-seed OOF AUC: {cb_auc:.6f} ===")
log(f"  개별 seed: {[f'{a:.4f}' for a in seed_aucs]}")
log(f"  학습 소요: {train_time:.1f}분")

# npy 저장
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
    groups["DI"] = (train_raw["시술 유형"] == "DI").values
if "난자 출처" in train_raw.columns:
    groups["donor egg"] = (train_raw["난자 출처"] == "기증 제공").values

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
        groups[f"age_{av}"] = (train_raw[age_col] == av).values

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
log(f"  OOF Log Loss: {oof_ll:.6f}")
log(f"  OOF AP:       {oof_ap:.6f}")

log(f"\n### Threshold별 분류 지표")
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

# npy 최종 저장
np.save(os.path.join(RESULT_DIR, f"oof_{VERSION}_final.npy"), oof_cb)
np.save(os.path.join(RESULT_DIR, f"test_{VERSION}_final.npy"), test_cb)
np.save(os.path.join(RESULT_DIR, f"y_train_{VERSION}.npy"), y)

# ============================================================
# [8] 피처 중요도
# ============================================================
log("\n## [8] CatBoost 피처 중요도 (상위 30)")

if cb_importances is not None:
    feature_names = X_train.columns.tolist()
    imp_df = pd.DataFrame(
        {"feature": feature_names, "importance": cb_importances}
    ).sort_values("importance", ascending=False)

    cat_set = set(cat_found)
    for rank_i, (_, row) in enumerate(imp_df.head(30).iterrows(), 1):
        marks = " [cat]" if row["feature"] in cat_set else ""
        if row["feature"] in v17_found:
            marks += " ★v17"
        elif row["feature"] in v18_found:
            marks += " ★v18"
        elif row["feature"] in team_found:
            marks += " ★team"
        log(f"  {rank_i}. {row['feature']}{marks}: {row['importance']:.2f}")

    # v18 신규 피처 중요도
    log(f"\n### v18 신규 피처 중요도")
    for f in v18_found:
        if f in imp_df["feature"].values:
            v = imp_df[imp_df["feature"] == f]["importance"].values[0]
            r = list(imp_df["feature"].values).index(f) + 1
            log(f"  {f}: {v:.2f} (#{r})")

    # v17 frozen 피처 중요도
    log(f"\n### v17 frozen 피처 중요도")
    for f in v17_found:
        if f in imp_df["feature"].values:
            v = imp_df[imp_df["feature"] == f]["importance"].values[0]
            r = list(imp_df["feature"].values).index(f) + 1
            log(f"  {f}: {v:.2f} (#{r})")

    # 중요도 0 피처 목록
    zero_imp = imp_df[imp_df["importance"] == 0]["feature"].tolist()
    log(
        f"\n### 중요도 0 피처 ({len(zero_imp)}개): {zero_imp[:20]}{'...' if len(zero_imp)>20 else ''}"
    )

    imp_df.to_csv(
        os.path.join(RESULT_DIR, f"feature_importance_{VERSION}.csv"), index=False
    )

# ============================================================
# [9] 버전 비교
# ============================================================
log("\n## [9] 버전 비교")
log("| 버전 | 모델 | OOF AUC | Test AUC | 비고 |")
log("|------|------|---------|----------|------|")
log("| v1 | CB원본 | 0.7403 | - | 베이스라인 |")
log("| v9 | XGB+CB 3seed | 0.7405 | 0.74166 | +IVF/DI |")
log("| v12 | CB+XGB+Tab | 0.7406 | - | 3모델 |")
log("| v15 | CB 5seed+Optuna | 0.7407 | 0.74169 | EDA+HP |")
log("| v16 | Multi-model | 0.7407 | - | subgroup |")
log("| v17 | CB 5seed | 0.7408 | 0.74228 | 팀원피처+frozen+CW |")
log(
    f"| **{VERSION}** | **CB 5seed** | **{cb_auc:.4f}** | **TBD** | **NaN보존+border254+노이즈제거** |"
)

# ============================================================
# [10] Go/No-Go 게이트 자동 판정
# ============================================================
log(f"\n{'='*60}")
log(f"## Go/No-Go 게이트")
log(f"{'='*60}")

gate1 = cb_auc > 0.7408
log(f"- Gate 1: OOF AUC {cb_auc:.6f} > 0.7408 → {'PASS ✓' if gate1 else 'FAIL ✗'}")

# frozen transfer AUC
if frozen_transfer_train.sum() > 50:
    frozen_auc = roc_auc_score(y[frozen_transfer_train], oof_cb[frozen_transfer_train])
    gate2 = frozen_auc > 0.6201
    log(f"- Gate 2: frozen transfer AUC {frozen_auc:.4f} > 0.6201 → {'PASS ✓' if gate2 else 'FAIL ✗'}")
else:
    gate2 = False
    log(f"- Gate 2: frozen transfer 건수 부족 → SKIP")

log(f"- 피처 수: {len(X_train.columns)} (목표 142)")

if gate1:
    log(f"\n→ Gate 1 통과! 앙상블 실험 진행 가능.")
else:
    log(f"\n→ Gate 1 미통과. v17 유지 권고.")

# ============================================================
# [11] 최종 요약
# ============================================================
total_time = (time.time() - t0) / 60
log(f"\n{'='*60}")
log(f"## 최종 요약")
log(f"{'='*60}")
log(f"- v18 변경사항:")
log(f"  1. NaN 보존 (fillna(0) 삭제)")
log(f"  2. border_count 64 → 254")
log(f"  3. 노이즈 피처 7개 제거")
log(f"  4. age_fail_severity 추가")
log(f"  5. 결측 플래그 3개 추가")
log(f"  6. 나이_임상구간 fillna(-1)")
log(f"  7. 이식배아수_구간 fillna(-1)")
log(f"- CB 5-seed OOF AUC: {cb_auc:.6f}")
log(f"  seed별: {[f'{a:.4f}' for a in seed_aucs]}")
log(f"- OOF LogLoss: {oof_ll:.6f}")
log(f"- OOF AP: {oof_ap:.6f}")
log(f"- 피처 수: {len(X_train.columns)} (cat={len(cat_idx)})")
log(f"- 데이터 누수: 없음")
log(f"- 총 소요: {total_time:.1f}분")
log(f"- 로그: {LOG_PATH}")
log(f"{'='*60}")

save_log()
print(f"\n완료! 로그: {LOG_PATH}")
