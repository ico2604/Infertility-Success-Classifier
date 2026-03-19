#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v17_plus_clean_frozen_only - v17 기반 frozen 피처만 정리 + 핵심 신규 피처 추가
==========================================================================
v17 대비 변경사항:
  [제거]
  1. frozen 피처 7개 제거 (frozen_day_interaction만 유지)

  [유지]
  2. is_ivf, is_di 유지
  3. has_infertility 유지
  4. egg_sperm_combo 유지

  [신규 추가 - 정제 6개]
  1. 배아 이식 경과일_missing_flag
  2. age_midpoint
  3. created_minus_transferred
  4. embryo_usage_combo
  5. treatment_source_combo_v3
  6. ivf_transfer_ratio_signal_v3

  [안정성 보강]
  1. cat_features: object 컬럼 전체 자동 포함
  2. frozen_train: pd.to_numeric으로 타입 안전 처리
  3. _ord 컬럼명 참조 오류 수정
  4. 시간 로그 추가
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

VERSION = "v17_plus_clean_frozen_only"
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
    "egg_sperm_combo",
    "embryo_usage_combo",
    "treatment_source_combo_v3",
]

AGE_MIDPOINT_MAP = {
    "만18-34세": 26.0,
    "만35-37세": 36.0,
    "만38-39세": 38.5,
    "만40-42세": 41.0,
    "만43-44세": 43.5,
    "만45-50세": 47.5,
    "알 수 없음": np.nan,
    "Unknown": np.nan,
}

COUNT_MAPPING = {
    "0회": 0, "1회": 1, "2회": 2,
    "3회": 3, "4회": 4, "5회": 5,
    "6회 이상": 6,
}

COUNT_COLS = [
    "총 시술 횟수", "클리닉 내 총 시술 횟수",
    "IVF 시술 횟수", "DI 시술 횟수",
    "총 임신 횟수", "IVF 임신 횟수", "DI 임신 횟수",
    "총 출산 횟수", "IVF 출산 횟수", "DI 출산 횟수",
]
AGE_COLS = ["시술 당시 나이", "난자 기증자 나이", "정자 기증자 나이"]

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


def safe_to_numeric_arr(x, default=0.0):
    return pd.to_numeric(pd.Series(x), errors="coerce").fillna(default).values


def detect_gpu():
    try:
        m = CatBoostClassifier(iterations=1, task_type="GPU", devices="0", verbose=False)
        m.fit(
            pd.DataFrame({"a": [0, 1, 0, 1], "b": [1, 0, 1, 0]}),
            [0, 1, 0, 1],
            verbose=False,
        )
        return "GPU"
    except Exception:
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
# 실행 시작
# ============================================================
t0 = time.time()
run_start = datetime.datetime.now()

log("=" * 70)
log(f"실행 시작: {run_start.strftime('%Y-%m-%d %H:%M:%S')}")
log("=" * 70)
log(f"# {VERSION} - frozen 정리 + 핵심 신규 피처")
log(f"시각: {NOW}")
log("=" * 60)

# ============================================================
# [1] 데이터 로드
# ============================================================
log("\n## [1] 데이터 로드")
t_load = time.time()

train_raw = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test_raw = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
sub = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

load_time_sec = time.time() - t_load

target_col = "임신 성공 여부"
y = train_raw[target_col].values

log(f"- train: {train_raw.shape}, test: {test_raw.shape}")
log(f"- 타겟: 0={np.sum(y==0)}, 1={np.sum(y==1)}, 양성비율={np.mean(y)*100:.1f}%")
log(f"- 데이터 로드 시간: {load_time_sec:.2f}초")

# ============================================================
# [2] 전처리 + 피처 엔지니어링
# ============================================================
log("\n## [2] 전처리 + 피처 엔지니어링")
t_fe = time.time()


def preprocess_full(df, is_train=True):
    df = df.copy()
    id_col = df.columns[0]
    drop_cols = [id_col]
    if is_train and target_col in df.columns:
        drop_cols.append(target_col)
    df = df.drop(columns=drop_cols, errors="ignore")

    # 신규 결측 플래그
    if "배아 이식 경과일" in df.columns:
        df["배아 이식 경과일_missing_flag"] = df["배아 이식 경과일"].isna().astype(float)

    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = df[c].fillna("Unknown").astype(str)

    for cc in COUNT_COLS:
        if cc in df.columns:
            df[f"{cc}_num"] = df[cc].apply(parse_korean_count)

    for ac in AGE_COLS:
        if ac in df.columns:
            df[f"{ac}_num"] = df[ac].apply(age_to_numeric)

    if "시술 당시 나이" in df.columns:
        df["age_midpoint"] = df["시술 당시 나이"].map(AGE_MIDPOINT_MAP)

    for cc in COUNT_COLS:
        ord_name = cc.replace(" ", "_") + "_ord"
        if cc in df.columns:
            df[ord_name] = df[cc].map(COUNT_MAPPING).fillna(0).astype(float)

    for c in df.select_dtypes(exclude=["object"]).columns:
        df[c] = df[c].fillna(0)

    def has(*cols):
        return all(c in df.columns for c in cols)

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
    sperm_src_col = "정자 출처"

    embryo = df[embryo_col].values if embryo_col in df.columns else np.zeros(len(df))
    total_emb = df[total_col].values if total_col in df.columns else np.zeros(len(df))
    stored = df[stored_col].values if stored_col in df.columns else np.zeros(len(df))
    mixed = df[mixed_col].values if mixed_col in df.columns else np.zeros(len(df))
    tr_day = df[tr_day_col].values if tr_day_col in df.columns else np.zeros(len(df))
    thawed = df[thawed_col].values if thawed_col in df.columns else np.zeros(len(df))
    fresh_use = safe_to_numeric_arr(df[fresh_use_col]) if fresh_use_col in df.columns else np.zeros(len(df))
    frozen_use = safe_to_numeric_arr(df[frozen_use_col]) if frozen_use_col in df.columns else np.zeros(len(df))
    micro_tr = df[micro_tr_col].values if micro_tr_col in df.columns else np.zeros(len(df))
    micro_emb = df[micro_emb_col].values if micro_emb_col in df.columns else np.zeros(len(df))
    age_num = df["시술 당시 나이_num"].values if "시술 당시 나이_num" in df.columns else np.zeros(len(df))

    # ============================================================
    # 기존 strong 피처
    # ============================================================
    if has("총 생성 배아 수", "수집된 신선 난자 수"):
        df["배아생성효율"] = safe_div_arr(df["총 생성 배아 수"].values, df["수집된 신선 난자 수"].values)
    if has("미세주입에서 생성된 배아 수", "미세주입된 난자 수"):
        df["ICSI수정효율"] = safe_div_arr(df["미세주입에서 생성된 배아 수"].values, df["미세주입된 난자 수"].values)
    if has("이식된 배아 수", "총 생성 배아 수"):
        df["배아이식비율"] = safe_div_arr(df["이식된 배아 수"].values, df["총 생성 배아 수"].values)
    if has("저장된 배아 수", "총 생성 배아 수"):
        df["배아저장비율"] = safe_div_arr(df["저장된 배아 수"].values, df["총 생성 배아 수"].values)
    if has("미세주입된 난자 수", "수집된 신선 난자 수"):
        df["난자활용률"] = safe_div_arr(df["미세주입된 난자 수"].values, df["수집된 신선 난자 수"].values)
    if has("이식된 배아 수", "수집된 신선 난자 수"):
        df["난자대비이식배아수"] = safe_div_arr(df["이식된 배아 수"].values, df["수집된 신선 난자 수"].values)

    if "이식된 배아 수" in df.columns:
        df["이식배아수_구간"] = pd.cut(
            pd.Series(embryo), bins=[-1, 0, 1, 2, 100], labels=[0, 1, 2, 3]
        )
        df["이식배아수_구간"] = pd.to_numeric(df["이식배아수_구간"], errors="coerce").fillna(0).astype(int)

    if has("총 임신 횟수_num", "총 시술 횟수_num"):
        df["전체임신률"] = safe_div_arr(df["총 임신 횟수_num"].values, df["총 시술 횟수_num"].values)
    if has("IVF 임신 횟수_num", "IVF 시술 횟수_num"):
        df["IVF임신률"] = safe_div_arr(df["IVF 임신 횟수_num"].values, df["IVF 시술 횟수_num"].values)
    if has("DI 임신 횟수_num", "DI 시술 횟수_num"):
        df["DI임신률"] = safe_div_arr(df["DI 임신 횟수_num"].values, df["DI 시술 횟수_num"].values)
    if has("총 출산 횟수_num", "총 임신 횟수_num"):
        df["임신유지율"] = safe_div_arr(df["총 출산 횟수_num"].values, df["총 임신 횟수_num"].values)
    if has("IVF 출산 횟수_num", "IVF 임신 횟수_num"):
        df["IVF임신유지율"] = safe_div_arr(df["IVF 출산 횟수_num"].values, df["IVF 임신 횟수_num"].values)
    if has("총 시술 횟수_num", "총 임신 횟수_num"):
        df["총실패횟수"] = np.maximum(df["총 시술 횟수_num"].values - df["총 임신 횟수_num"].values, 0)
    if has("IVF 시술 횟수_num", "IVF 임신 횟수_num"):
        df["IVF실패횟수"] = np.maximum(df["IVF 시술 횟수_num"].values - df["IVF 임신 횟수_num"].values, 0)
    if "IVF실패횟수" in df.columns:
        df["반복IVF실패_여부"] = (df["IVF실패횟수"] >= 3).astype(int)
    if has("클리닉 내 총 시술 횟수_num", "총 시술 횟수_num"):
        df["클리닉집중도"] = safe_div_arr(df["클리닉 내 총 시술 횟수_num"].values, df["총 시술 횟수_num"].values)
    if has("IVF 시술 횟수_num", "총 시술 횟수_num"):
        df["IVF시술비율"] = safe_div_arr(df["IVF 시술 횟수_num"].values, df["총 시술 횟수_num"].values)
    if "총 임신 횟수_num" in df.columns:
        df["임신경험있음"] = (df["총 임신 횟수_num"] > 0).astype(int)
    if "총 출산 횟수_num" in df.columns:
        df["출산경험있음"] = (df["총 출산 횟수_num"] > 0).astype(int)

    if "시술 당시 나이_num" in df.columns:
        df["나이_제곱"] = age_num ** 2
        df["나이_임상구간"] = pd.cut(
            pd.Series(age_num), bins=[0, 35, 40, 45, 100], labels=[0, 1, 2, 3], right=False
        )
        df["나이_임상구간"] = pd.to_numeric(df["나이_임상구간"], errors="coerce").fillna(0).astype(int)
        df["고령_여부"] = (age_num >= 35).astype(int)
        df["초고령_여부"] = (age_num >= 40).astype(int)
        df["극고령_여부"] = (age_num >= 42).astype(int)

    if has("시술 당시 나이_num", "총 시술 횟수_num"):
        df["나이X총시술"] = df["시술 당시 나이_num"].values * df["총 시술 횟수_num"].values
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

    df["실제이식여부"] = (embryo > 0).astype(int)
    df["total_embryo_ratio"] = safe_div_arr(embryo + stored + thawed, total_emb + 1)
    df["수정_성공률"] = safe_div_arr(total_emb, mixed)
    df["배아_이용률"] = safe_div_arr(embryo, total_emb)
    df["배아_잉여율"] = safe_div_arr(stored, total_emb)

    if type_col in df.columns:
        treatment_upper = df[type_col].fillna("").astype(str).str.upper()
        df["is_ivf"] = (treatment_upper == "IVF").astype(int)
        df["is_di"] = (treatment_upper == "DI").astype(int)

        ivf_mask = (df[type_col] == "IVF").values
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
    df["transfer_day_cat"] = pd.Series(tr_day).map(day_cat_map).fillna(0).astype(int).values
    df["embryo_day_interaction"] = embryo * tr_day
    df["fresh_transfer_ratio"] = safe_div_arr(embryo * fresh_use, total_emb + 1)
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

    # frozen 관련은 이것만 유지
    df["frozen_day_interaction"] = frozen_use * tr_day

    inf_cols = [c for c in df.columns if c.startswith("불임 원인 -")]
    if inf_cols:
        inf_block = df[inf_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
        df["infertility_count"] = inf_block.sum(axis=1)
        df["has_infertility"] = (df["infertility_count"] > 0).astype(int)

    if "총 시술 횟수_num" in df.columns:
        df["first_treatment"] = (df["총 시술 횟수_num"] == 0).astype(int)

    if egg_src_col in df.columns and sperm_src_col in df.columns:
        df["egg_sperm_combo"] = (
            df[egg_src_col].fillna("Unknown").astype(str) + "__" +
            df[sperm_src_col].fillna("Unknown").astype(str)
        )

    # ============================================================
    # 신규 정제 6개
    # ============================================================
    if has("총 생성 배아 수", "이식된 배아 수"):
        df["created_minus_transferred"] = (
            df["총 생성 배아 수"].values - df["이식된 배아 수"].values
        ).clip(min=0)

    for col in ["동결 배아 사용 여부", "신선 배아 사용 여부", "기증 배아 사용 여부"]:
        if col not in df.columns:
            df[col] = "0"
    df["embryo_usage_combo"] = (
        df["동결 배아 사용 여부"].fillna(0).astype(str) + "__" +
        df["신선 배아 사용 여부"].fillna(0).astype(str) + "__" +
        df["기증 배아 사용 여부"].fillna(0).astype(str)
    )

    if type_col in df.columns:
        treatment_upper = df[type_col].fillna("").astype(str).str.upper()
        is_ivf = (treatment_upper == "IVF").astype(float)

        if "배아이식비율" in df.columns:
            df["ivf_transfer_ratio_signal_v3"] = is_ivf * df["배아이식비율"].fillna(0)

        df["treatment_source_combo_v3"] = (
            treatment_upper + "__" +
            df.get("난자 출처", pd.Series("Unknown", index=df.index)).fillna("Unknown").astype(str) + "__" +
            df.get("정자 출처", pd.Series("Unknown", index=df.index)).fillna("Unknown").astype(str)
        )

    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    return df, cat_cols


log("  전처리 중...")
X_train, cat_names_train = preprocess_full(train_raw, is_train=True)
X_test, _ = preprocess_full(test_raw, is_train=False)

fe_time_sec = time.time() - t_fe

common_cols = [c for c in X_train.columns if c in X_test.columns]
X_train = X_train[common_cols]
X_test = X_test[common_cols]

cat_found = [c for c in CAT_COLS_CANDIDATES if c in X_train.columns]
all_object_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
cat_found = sorted(set(cat_found + all_object_cols))
cat_idx = [X_train.columns.get_loc(c) for c in cat_found]

log(f"- 피처 수: {len(X_train.columns)}, 범주형: {len(cat_idx)}")
log(f"- 범주형 컬럼: {cat_found}")

missing_cat = sorted(set(cat_names_train) - set(cat_found))
if missing_cat:
    log(f"- ⚠️ cat_found 누락 object 컬럼: {missing_cat}")
else:
    log(f"- ✓ cat_found 누락 없음 (preprocess object {len(cat_names_train)}개 전부 포함)")

new_features = [
    "배아 이식 경과일_missing_flag",
    "age_midpoint",
    "created_minus_transferred",
    "embryo_usage_combo",
    "treatment_source_combo_v3",
    "ivf_transfer_ratio_signal_v3",
]
new_found = [f for f in new_features if f in X_train.columns]
log(f"- {VERSION} 신규 피처 ({len(new_found)}개): {new_found}")
log(f"- 피처 엔지니어링 시간: {fe_time_sec/60:.2f}분 ({fe_time_sec:.1f}초)")

# ============================================================
# [3] subgroup 통계
# ============================================================
log("\n## [3] subgroup 통계")
embryo_raw = train_raw["이식된 배아 수"].fillna(0).values
transfer_mask_train = embryo_raw > 0

frozen_col = "동결 배아 사용 여부"
frozen_train = (
    pd.to_numeric(train_raw[frozen_col], errors="coerce").fillna(0).values > 0
    if frozen_col in train_raw.columns
    else np.zeros(len(y), dtype=bool)
)
frozen_transfer_train = transfer_mask_train & frozen_train

log(f"- 이식: {transfer_mask_train.sum()}건, 양성률={y[transfer_mask_train].mean()*100:.1f}%")
log(f"- 비이식: {(~transfer_mask_train).sum()}건, 양성률={y[~transfer_mask_train].mean()*100:.2f}%")
log(f"- frozen transfer: {frozen_transfer_train.sum()}건, 양성률={y[frozen_transfer_train].mean()*100:.1f}%")

# ============================================================
# [4] CatBoost seed ensemble
# ============================================================
log("\n## [4] CatBoost seed 앙상블")
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
cb_importances = None

total_models = len(SEEDS) * N_FOLDS
model_count = 0
t_train = time.time()

for si, seed_val in enumerate(SEEDS):
    t_seed = time.time()
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
            X_tr, y_tr,
            eval_set=(X_va, y_va),
            cat_features=cat_idx,
            verbose=False
        )

        va_pred = model.predict_proba(X_va)[:, 1]
        oof_seed[va_idx] = va_pred
        test_seed += model.predict_proba(X_test)[:, 1] / N_FOLDS

        va_auc = roc_auc_score(y_va, va_pred)
        fold_time_sec = time.time() - t_fold

        log(
            f"    Fold {fold_i+1}: "
            f"AUC={va_auc:.4f}, "
            f"iter={model.best_iteration_}, "
            f"소요={fold_time_sec/60:.2f}분 ({fold_time_sec:.1f}초) "
            f"[{model_count}/{total_models}]"
        )

        if si == len(SEEDS) - 1 and fold_i == N_FOLDS - 1:
            cb_importances = model.get_feature_importance()

    seed_auc = roc_auc_score(y, oof_seed)
    seed_aucs.append(seed_auc)
    seed_time_sec = time.time() - t_seed

    log(f"  Seed {seed_val} OOF AUC: {seed_auc:.4f}")
    log(f"  Seed {seed_val} 총 소요: {seed_time_sec/60:.2f}분 ({seed_time_sec:.1f}초)")

    oof_cb += oof_seed / len(SEEDS)
    test_cb += test_seed / len(SEEDS)

    np.save(os.path.join(RESULT_DIR, f"oof_{VERSION}_seed{seed_val}.npy"), oof_seed)
    save_log()

train_time_sec = time.time() - t_train
cb_auc = roc_auc_score(y, oof_cb)

log(f"\n  === CatBoost seed ensemble OOF AUC: {cb_auc:.6f} ===")
log(f"  개별 seed: {[f'{a:.4f}' for a in seed_aucs]}")
log(f"  전체 학습 시간: {train_time_sec/60:.2f}분 ({train_time_sec:.1f}초)")

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
    groups["IVF"] = (train_raw["시술 유형"] == "IVF").values
    groups["DI"] = (train_raw["시술 유형"] == "DI").values
if "난자 출처" in train_raw.columns:
    groups["donor egg"] = (train_raw["난자 출처"] == "기증 제공").values

for gname, gmask in groups.items():
    if gmask.sum() > 50 and len(np.unique(y[gmask])) > 1:
        g_auc = roc_auc_score(y[gmask], oof_cb[gmask])
        log(f"  {gname}: AUC={g_auc:.4f} (n={gmask.sum()}, pos={y[gmask].mean()*100:.1f}%)")

# ============================================================
# [6] 종합 평가지표
# ============================================================
log("\n## [6] 종합 평가지표")

oof_ll = log_loss(y, np.clip(oof_cb, 1e-7, 1 - 1e-7))
oof_ap = average_precision_score(y, oof_cb)

log(f"  OOF AUC:      {cb_auc:.6f}")
log(f"  OOF Log Loss: {oof_ll:.6f}")
log(f"  OOF AP:       {oof_ap:.6f}")

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
log("\n## [8] CatBoost 피처 중요도 (상위 30)")

if cb_importances is not None:
    feature_names = X_train.columns.tolist()
    imp_df = pd.DataFrame(
        {"feature": feature_names, "importance": cb_importances}
    ).sort_values("importance", ascending=False)

    cat_set = set(cat_found)
    for rank_i, (_, row) in enumerate(imp_df.head(30).iterrows(), 1):
        marks = " [cat]" if row["feature"] in cat_set else ""
        if row["feature"] in new_found:
            marks += " ★new"
        log(f"  {rank_i}. {row['feature']}{marks}: {row['importance']:.2f}")

    log("\n### 신규 피처 중요도")
    for f in new_found:
        if f in imp_df["feature"].values:
            v = imp_df[imp_df["feature"] == f]["importance"].values[0]
            r = list(imp_df["feature"].values).index(f) + 1
            log(f"  {f}: {v:.2f} (#{r})")

    zero_imp = imp_df[imp_df["importance"] == 0]["feature"].tolist()
    log(
        f"\n### 중요도 0 피처 ({len(zero_imp)}개): "
        f"{zero_imp[:20]}{'...' if len(zero_imp) > 20 else ''}"
    )

    imp_df.to_csv(
        os.path.join(RESULT_DIR, f"feature_importance_{VERSION}.csv"), index=False
    )

# ============================================================
# [9] 버전 비교
# ============================================================
log("\n## [9] 버전 비교")
log("| 버전 | OOF AUC | 비고 |")
log("|------|---------|------|")
log("| v17                    | 0.7408 | 팀원피처+frozen+CW |")
log("| v17_plus_light         | 0.7406 | frozen 정리 + 신규 9개 |")
log(f"| **{VERSION}** | **{cb_auc:.4f}** | **frozen만 제거 + 신규 6개** |")

# ============================================================
# [10] 최종 요약
# ============================================================
total_time_sec = time.time() - t0
run_end = datetime.datetime.now()

log(f"\n{'='*60}")
log("## 최종 요약")
log(f"{'='*60}")
log("- 변경사항:")
log("  [제거] frozen 피처 7개 제거 (frozen_day_interaction만 유지)")
log("  [유지] is_ivf, is_di, has_infertility, egg_sperm_combo")
log("  [추가] 배아 이식 경과일_missing_flag, age_midpoint, created_minus_transferred")
log("  [추가] embryo_usage_combo, treatment_source_combo_v3, ivf_transfer_ratio_signal_v3")
log(f"- CB seed ensemble OOF AUC: {cb_auc:.6f}")
log(f"  seed별: {[f'{a:.4f}' for a in seed_aucs]}")
log(f"- OOF LogLoss: {oof_ll:.6f}")
log(f"- OOF AP: {oof_ap:.6f}")
log(f"- 총 피처 수: {len(X_train.columns)} (범주형={len(cat_idx)})")
log(f"- 데이터 로드 시간: {load_time_sec:.2f}초")
log(f"- 피처 엔지니어링 시간: {fe_time_sec/60:.2f}분 ({fe_time_sec:.1f}초)")
log(f"- 전체 학습 시간: {train_time_sec/60:.2f}분 ({train_time_sec:.1f}초)")
log(f"- 총 실행 시간: {total_time_sec/60:.2f}분 ({total_time_sec:.1f}초)")
log(f"- 로그: {LOG_PATH}")
log(f"{'='*60}")

log("=" * 70)
log(f"실행 종료: {run_end.strftime('%Y-%m-%d %H:%M:%S')}")
log(f"총 실행 시간: {total_time_sec/60:.2f}분 ({total_time_sec:.1f}초)")
log("=" * 70)

save_log()
print(f"\n완료! 로그: {LOG_PATH}")