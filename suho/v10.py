#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v10.py - v9 기반 + 팀원 의료 파생변수 흡수 + 나이 중간값 + XGB depth 하향
  - v9 전처리/IVF/DI 분기 피처 유지
  - 팀원 make_medical_features 핵심 변수 추가
  - 나이: ordinal → 구간 중간값 변환
  - XGB depth 8→5 (일반화 강화)
  - CatBoost 3-seed 앙상블 유지
  - 다양한 평가지표 출력 (AUC, F1, Precision, Recall, AP, LogLoss)
"""

import os, sys, warnings, time, re
import numpy as np, pandas as pd
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    average_precision_score,
    log_loss,
    confusion_matrix,
    precision_recall_curve,
)
from xgboost import XGBClassifier
from catboost import CatBoostClassifier, Pool

warnings.filterwarnings("ignore")

VERSION = "v10"
SEED = 42
SEEDS_CB = [42, 2026, 2604]
N_FOLDS = 5
TARGET = "임신 성공 여부"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
RESULT_DIR = os.path.join(BASE_DIR, "result")
os.makedirs(RESULT_DIR, exist_ok=True)
NOW = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_PATH = os.path.join(RESULT_DIR, f"log_{VERSION}.md")


class Logger:
    def __init__(self, fp):
        self.terminal = sys.stdout
        self.log = open(fp, "w", encoding="utf-8")

    def write(self, msg):
        self.terminal.write(msg)
        clean = msg.replace("\r", "")
        if clean.strip():
            self.log.write(clean + "\n")

    def flush(self):
        self.terminal.flush()
        self.log.flush()


sys.stdout = Logger(LOG_PATH)

start_all = time.time()
print(f"# {VERSION} - v9 + 팀원 의료 파생변수 + 나이 중간값 + XGB depth=5")
print(f"시각: {NOW}")
print("=" * 60)

# ============================================================
# 1. 데이터 로드
# ============================================================
print("\n## [1] 데이터 로드")
train_raw = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test_raw = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
sub = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

y = train_raw[TARGET].values
test_ids = test_raw["ID"].values
print(f"- train: {train_raw.shape}, test: {test_raw.shape}")
print(f"- 타겟: 0={np.sum(y==0)}, 1={np.sum(y==1)}, 양성비율={y.mean()*100:.1f}%")

# ============================================================
# 2. 상수
# ============================================================
DROP_COLS = [
    "ID",
    TARGET,
    "PGD 시술 여부",
    "PGS 시술 여부",
    "난자 해동 경과일",
    "착상 전 유전 검사 사용 여부",
]

REMOVE_ZERO_IMP = [
    "저장된 신선 난자 수",
    "불임 원인 - 정자 형태",
    "불임 원인 - 정자 운동성",
    "불임 원인 - 정자 면역학적 요인",
    "불임 원인 - 자궁경부 문제",
    "난자 채취 경과일",
    "불임 원인 - 여성 요인",
]

COUNT_MAP = {"0회": 0, "1회": 1, "2회": 2, "3회": 3, "4회": 4, "5회": 5, "6회 이상": 6}

# v10: 나이 → 구간 중간값 (팀원 방식)
AGE_MIDPOINT_MAP = {
    "만 18-34세": 26.0,
    "만 35-37세": 36.0,
    "만 38-39세": 38.5,
    "만 40-42세": 41.0,
    "만 43-44세": 43.5,
    "만 45-50세": 47.5,
    "알 수 없음": np.nan,
    "Unknown": np.nan,
}


# fallback: regex 파싱
def age_str_to_midpoint(x):
    if pd.isna(x):
        return np.nan
    x = str(x).strip()
    mapped = AGE_MIDPOINT_MAP.get(x)
    if mapped is not None or (mapped != mapped if mapped is not None else False):
        return mapped
    # 직접 매핑 실패 시 regex
    nums = re.findall(r"\d+", x)
    if len(nums) >= 2:
        return (float(nums[0]) + float(nums[1])) / 2
    if len(nums) == 1:
        return float(nums[0])
    return np.nan


PROCEDURE_MAP = {"IVF": 1, "DI": 0}
EGG_SOURCE_MAP = {"본인 제공": 0, "기증 제공": 1, "알 수 없음": 2}
SPERM_SOURCE_MAP = {
    "배우자 제공": 0,
    "기증 제공": 1,
    "미할당": 2,
    "배우자 및 기증 제공": 3,
}

TE_COLS = ["특정 시술 유형", "배아 생성 주요 이유", "시술 시기 코드", "배란 유도 유형"]
TE_COLS_NUMERIC = ["시술 당시 나이"]

BOOL_MAP = {
    "TRUE": 1,
    "FALSE": 0,
    True: 1,
    False: 0,
    "예": 1,
    "아니오": 0,
    "1": 1,
    "0": 0,
    "Y": 1,
    "N": 0,
    "Yes": 1,
    "No": 0,
}

IMPLANT_BOOL_COLS = [
    "단일 배아 이식 여부",
    "신선 배아 사용 여부",
    "동결 배아 사용 여부",
    "기증 배아 사용 여부",
]


def safe_div(a, b, fill=0.0):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.where(b > 0, a / b, fill)


def safe_div_nan(a, b):
    """팀원 스타일: 0이면 NaN"""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.where((np.isnan(b)) | (b == 0), np.nan, a / b)


# ============================================================
# 3. 전처리 함수
# ============================================================


def step1_drop_and_fillna(df):
    drop = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=drop, errors="ignore")
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna("Unknown")
        else:
            df[col] = df[col].fillna(0)
    return df


def step2_count_to_int(df):
    for col in df.columns:
        if "횟수" in col and df[col].dtype == "object":
            df[col] = df[col].map(COUNT_MAP).fillna(0).astype(int)
    return df


def step3_age_to_midpoint(df):
    """v10: 나이를 구간 중간값으로 변환"""
    for col in ["시술 당시 나이", "난자 기증자 나이", "정자 기증자 나이"]:
        if col not in df.columns or df[col].dtype != "object":
            continue
        df[col] = df[col].apply(age_str_to_midpoint)
        # NaN은 전체 중간값으로 채움 (나중에 모델이 처리)
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val if not np.isnan(median_val) else 0)
    return df


def step4_manual_encode(df):
    if "시술 유형" in df.columns:
        df["시술 유형"] = df["시술 유형"].map(PROCEDURE_MAP).fillna(-1).astype(int)
    if "난자 출처" in df.columns:
        df["난자 출처"] = df["난자 출처"].map(EGG_SOURCE_MAP).fillna(-1).astype(int)
    if "정자 출처" in df.columns:
        df["정자 출처"] = df["정자 출처"].map(SPERM_SOURCE_MAP).fillna(-1).astype(int)
    return df


def step5_bool_to_int(df):
    for col in df.columns:
        if df[col].dtype != "object":
            continue
        unique_vals = set(df[col].dropna().unique())
        bool_vals = {
            "TRUE",
            "FALSE",
            "True",
            "False",
            "예",
            "아니오",
            "Y",
            "N",
            "Yes",
            "No",
        }
        if unique_vals.issubset(bool_vals | {"Unknown", "_missing_"}):
            df[col] = df[col].map(BOOL_MAP).fillna(0).astype(int)
    return df


def step6_feature_engineering(df):
    """v8 + v9 IVF/DI 분기 + v10 팀원 의료 파생변수"""

    # === v8 기존 ===
    if "특정 시술 유형" in df.columns:
        df["is_blastocyst"] = (
            df["특정 시술 유형"]
            .apply(
                lambda x: 1 if isinstance(x, str) and "BLASTOCYST" in x.upper() else 0
            )
            .astype(int)
        )

    embryo_total = (
        pd.to_numeric(df.get("총 생성 배아 수", 0), errors="coerce").fillna(0).values
    )
    mixed = pd.to_numeric(df.get("혼합된 난자 수", 0), errors="coerce").fillna(0).values
    transferred = (
        pd.to_numeric(df.get("이식된 배아 수", 0), errors="coerce").fillna(0).values
    )
    stored = (
        pd.to_numeric(df.get("저장된 배아 수", 0), errors="coerce").fillna(0).values
    )
    fresh_egg = (
        pd.to_numeric(df.get("수집된 신선 난자 수", 0), errors="coerce")
        .fillna(0)
        .values
    )
    icsi_egg = (
        pd.to_numeric(df.get("미세주입된 난자 수", 0), errors="coerce").fillna(0).values
    )
    icsi_embryo = (
        pd.to_numeric(df.get("미세주입에서 생성된 배아 수", 0), errors="coerce")
        .fillna(0)
        .values
    )

    df["수정_성공률"] = safe_div(embryo_total, mixed)

    for col in IMPLANT_BOOL_COLS:
        if col in df.columns and df[col].dtype == "object":
            df[col] = df[col].map(BOOL_MAP).fillna(0).astype(int)

    if all(c in df.columns for c in IMPLANT_BOOL_COLS):
        df["이식유형_단일신선"] = (
            (df["단일 배아 이식 여부"] == 1) & (df["신선 배아 사용 여부"] == 1)
        ).astype(int)
        df["이식유형_단일동결"] = (
            (df["단일 배아 이식 여부"] == 1) & (df["동결 배아 사용 여부"] == 1)
        ).astype(int)
        df["이식유형_복수신선"] = (
            (df["단일 배아 이식 여부"] == 0) & (df["신선 배아 사용 여부"] == 1)
        ).astype(int)
        df["이식유형_복수동결"] = (
            (df["단일 배아 이식 여부"] == 0) & (df["동결 배아 사용 여부"] == 1)
        ).astype(int)
        df["이식유형_기증"] = df["기증 배아 사용 여부"].astype(int)

    donor_flag = np.zeros(len(df))
    if "난자 출처" in df.columns:
        donor_flag = np.where(df["난자 출처"] == 1, 1, donor_flag)
    if "난자 기증자 나이" in df.columns:
        donor_age = (
            pd.to_numeric(df["난자 기증자 나이"], errors="coerce").fillna(0).values
        )
        donor_flag = np.where((donor_age > 0), 1, donor_flag)
    df["is_donor_egg"] = donor_flag.astype(int)

    total_cycles = (
        pd.to_numeric(df.get("총 시술 횟수", 0), errors="coerce").fillna(0).values
    )
    clinic_cycles = (
        pd.to_numeric(df.get("클리닉 내 총 시술 횟수", 0), errors="coerce")
        .fillna(0)
        .values
    )
    df["타클리닉_시술"] = np.clip(total_cycles - clinic_cycles, 0, None).astype(int)

    df["실제이식여부"] = (transferred > 0).astype(int)
    df["배아_이용률"] = safe_div(transferred, embryo_total)
    df["배아_잉여율"] = safe_div(embryo_total - transferred, embryo_total)
    df["난자_배아_전환율"] = safe_div(embryo_total, fresh_egg)
    df["total_embryo_ratio"] = safe_div(transferred, embryo_total + 1)

    age_val = (
        pd.to_numeric(df.get("시술 당시 나이", 0), errors="coerce").fillna(0).values
    )

    if "배아 생성 주요 이유" in df.columns:
        reason = df["배아 생성 주요 이유"].astype(str)
        df["purpose_zero"] = (
            reason.str.contains("저장용|기증용", na=False)
            & ~reason.str.contains("현재 시술용", na=False)
        ).astype(int)

    # === v9 IVF/DI 분기 ===
    if "시술 유형" in df.columns:
        is_ivf = (df["시술 유형"] == 1).values
        is_di = (df["시술 유형"] == 0).values

        df["is_ivf"] = is_ivf.astype(int)
        df["is_di"] = is_di.astype(int)

        total_preg = (
            pd.to_numeric(df.get("총 임신 횟수", 0), errors="coerce").fillna(0).values
        )

        df["ivf_total_treatment_load"] = np.where(is_ivf, total_cycles, 0)
        df["di_total_treatment_load"] = np.where(is_di, total_cycles, 0)
        df["ivf_pregnancy_load"] = np.where(is_ivf, total_preg, 0)
        df["di_pregnancy_load"] = np.where(is_di, total_preg, 0)

        transfer_ratio = safe_div(transferred, embryo_total)
        storage_ratio = safe_div(stored, embryo_total)
        df["ivf_transfer_ratio"] = np.where(is_ivf, transfer_ratio, 0)
        df["di_transfer_ratio"] = np.where(is_di, transfer_ratio, 0)
        df["ivf_storage_ratio"] = np.where(is_ivf, storage_ratio, 0)
        df["di_storage_ratio"] = np.where(is_di, storage_ratio, 0)

        embryo_days = (
            pd.to_numeric(df.get("배아 이식 경과일", 0), errors="coerce")
            .fillna(0)
            .values
        )
        df["ivf_embryo_age_signal"] = np.where(is_ivf, embryo_days * age_val, 0)
        df["di_embryo_age_signal"] = np.where(is_di, embryo_days * age_val, 0)

        egg_src = (
            pd.to_numeric(df.get("난자 출처", 0), errors="coerce").fillna(0).values
        )
        sperm_src = (
            pd.to_numeric(df.get("정자 출처", 0), errors="coerce").fillna(0).values
        )
        df["treatment_history_combo"] = df["시술 유형"] * 10 + total_cycles
        df["treatment_source_combo"] = df["시술 유형"] * 100 + egg_src * 10 + sperm_src
        df["treatment_age_history_combo"] = (
            df["시술 유형"] * 100 + age_val * 10 + total_cycles
        )

    # === v10 신규: 팀원 의료 파생변수 ===

    # ICSI 수정 효율
    df["ICSI수정효율"] = safe_div(icsi_embryo, icsi_egg)

    # 난자 활용률
    df["난자활용률"] = safe_div(icsi_egg, fresh_egg)

    # 난자 대비 이식 배아 수
    df["난자대비이식배아수"] = safe_div(transferred, fresh_egg)

    # 저장 대비 이식 비율
    df["저장_대비_이식비율"] = safe_div(stored, transferred + 0.001)

    # 이식 배아 수 구간 (0, 1, 2, 3+)
    df["이식배아수_구간"] = np.clip(transferred, 0, 3).astype(int)

    # 과거 시술 이력 기반
    total_preg_val = (
        pd.to_numeric(df.get("총 임신 횟수", 0), errors="coerce").fillna(0).values
    )
    total_birth = (
        pd.to_numeric(df.get("총 출산 횟수", 0), errors="coerce").fillna(0).values
    )
    ivf_cycles = (
        pd.to_numeric(df.get("IVF 시술 횟수", 0), errors="coerce").fillna(0).values
    )
    ivf_preg = (
        pd.to_numeric(df.get("IVF 임신 횟수", 0), errors="coerce").fillna(0).values
    )
    ivf_birth = (
        pd.to_numeric(df.get("IVF 출산 횟수", 0), errors="coerce").fillna(0).values
    )
    di_cycles = (
        pd.to_numeric(df.get("DI 시술 횟수", 0), errors="coerce").fillna(0).values
    )
    di_preg = pd.to_numeric(df.get("DI 임신 횟수", 0), errors="coerce").fillna(0).values

    # 전체 임신률
    df["전체임신률"] = safe_div(total_preg_val, total_cycles)

    # IVF 임신률
    df["IVF임신률"] = safe_div(ivf_preg, ivf_cycles)

    # DI 임신률
    df["DI임신률"] = safe_div(di_preg, di_cycles)

    # 임신 유지율 (출산/임신)
    df["임신유지율"] = safe_div(total_birth, total_preg_val)

    # IVF 임신 유지율
    df["IVF임신유지율"] = safe_div(ivf_birth, ivf_preg)

    # 총 실패 횟수
    df["총실패횟수"] = np.clip(total_cycles - total_preg_val, 0, None)

    # IVF 실패 횟수
    df["IVF실패횟수"] = np.clip(ivf_cycles - ivf_preg, 0, None)

    # 반복 IVF 실패 여부 (3회 이상)
    df["반복IVF실패_여부"] = (df["IVF실패횟수"] >= 3).astype(int)

    # 클리닉 집중도
    df["클리닉집중도"] = safe_div(clinic_cycles, total_cycles)

    # IVF 시술 비율
    df["IVF시술비율"] = safe_div(ivf_cycles, total_cycles)

    # 임신/출산 경험
    df["임신경험있음"] = (total_preg_val > 0).astype(int)
    df["출산경험있음"] = (total_birth > 0).astype(int)

    # 나이 기반 (v10: 중간값이므로 의미 있음)
    df["나이_제곱"] = age_val**2
    df["고령_여부"] = (age_val >= 35).astype(int)
    df["초고령_여부"] = (age_val >= 40).astype(int)
    df["극고령_여부"] = (age_val >= 42).astype(int)

    # 나이 × 이력 상호작용
    df["나이X총시술"] = age_val * total_cycles
    df["나이XIVF실패"] = age_val * df["IVF실패횟수"].values
    df["나이XIVF임신률"] = age_val * df["IVF임신률"].values
    df["초고령X반복실패"] = df["초고령_여부"].values * df["반복IVF실패_여부"].values

    # 복합 위험도 점수
    df["복합위험도점수"] = (
        df["고령_여부"].values
        + df["초고령_여부"].values
        + df["반복IVF실패_여부"].values
        + (1 - df["임신경험있음"].values)
    )

    return df


def preprocess_xgb(df, label):
    df = step1_drop_and_fillna(df)
    print(f"  [{label}] step1: {df.shape}, obj={df.select_dtypes('object').shape[1]}")
    df = step2_count_to_int(df)
    print(f"  [{label}] step2: obj={df.select_dtypes('object').shape[1]}")
    df = step3_age_to_midpoint(df)
    print(f"  [{label}] step3 (나이 중간값): obj={df.select_dtypes('object').shape[1]}")
    df = step4_manual_encode(df)
    print(f"  [{label}] step4: obj={df.select_dtypes('object').shape[1]}")
    df = step5_bool_to_int(df)
    print(f"  [{label}] step5: obj={df.select_dtypes('object').shape[1]}")
    df = step6_feature_engineering(df)
    print(
        f"  [{label}] step6: 컬럼={df.shape[1]}, obj={df.select_dtypes('object').shape[1]}"
    )
    return df


# ============================================================
# 4. CatBoost용 피처
# ============================================================


def build_catboost_features(df):
    out = df.copy()

    embryo_total = (
        pd.to_numeric(out.get("총 생성 배아 수", 0), errors="coerce").fillna(0).values
    )
    mixed = (
        pd.to_numeric(out.get("혼합된 난자 수", 0), errors="coerce").fillna(0).values
    )
    transferred = (
        pd.to_numeric(out.get("이식된 배아 수", 0), errors="coerce").fillna(0).values
    )
    stored = (
        pd.to_numeric(out.get("저장된 배아 수", 0), errors="coerce").fillna(0).values
    )
    fresh_egg = (
        pd.to_numeric(out.get("수집된 신선 난자 수", 0), errors="coerce")
        .fillna(0)
        .values
    )
    icsi_egg = (
        pd.to_numeric(out.get("미세주입된 난자 수", 0), errors="coerce")
        .fillna(0)
        .values
    )
    icsi_embryo = (
        pd.to_numeric(out.get("미세주입에서 생성된 배아 수", 0), errors="coerce")
        .fillna(0)
        .values
    )

    # v8 파생
    out["수정_성공률"] = safe_div(embryo_total, mixed)
    out["배아_이용률"] = safe_div(transferred, embryo_total)
    out["배아_잉여율"] = safe_div(embryo_total - transferred, embryo_total)
    out["난자_배아_전환율"] = safe_div(embryo_total, fresh_egg)
    out["실제이식여부"] = (transferred > 0).astype(int)
    out["total_embryo_ratio"] = safe_div(transferred, embryo_total + 1)

    if "배아 생성 주요 이유" in out.columns:
        reason = out["배아 생성 주요 이유"].astype(str)
        out["purpose_zero"] = (
            reason.str.contains("저장용|기증용", na=False)
            & ~reason.str.contains("현재 시술용", na=False)
        ).astype(int)

    # 나이 중간값 (CB용)
    for col in ["시술 당시 나이", "난자 기증자 나이", "정자 기증자 나이"]:
        if col in out.columns and out[col].dtype == "object":
            out[col + "_mid"] = out[col].apply(age_str_to_midpoint)
            median_val = out[col + "_mid"].median()
            out[col + "_mid"] = out[col + "_mid"].fillna(
                median_val if not np.isnan(median_val) else 0
            )

    age_col = "시술 당시 나이_mid" if "시술 당시 나이_mid" in out.columns else None
    age_val = out[age_col].values if age_col else np.zeros(len(out))

    # v9 IVF/DI
    if "시술 유형" in out.columns:
        proc_str = out["시술 유형"].astype(str)
        is_ivf = proc_str.str.upper().str.contains("IVF", na=False).values
        is_di = proc_str.str.upper().str.contains("DI", na=False).values
        out["is_ivf"] = is_ivf.astype(int)
        out["is_di"] = is_di.astype(int)

        def to_num(col_name):
            if col_name not in out.columns:
                return np.zeros(len(out))
            vals = out[col_name]
            if vals.dtype == "object":
                vals = vals.map(COUNT_MAP).fillna(vals)
            return pd.to_numeric(vals, errors="coerce").fillna(0).values

        total_cycles = to_num("총 시술 횟수")
        total_preg = to_num("총 임신 횟수")
        ivf_cycles = to_num("IVF 시술 횟수")
        ivf_preg = to_num("IVF 임신 횟수")
        ivf_birth = to_num("IVF 출산 횟수")
        total_birth = to_num("총 출산 횟수")
        clinic_cycles = to_num("클리닉 내 총 시술 횟수")
        embryo_days = (
            pd.to_numeric(out.get("배아 이식 경과일", 0), errors="coerce")
            .fillna(0)
            .values
        )

        out["ivf_total_treatment_load"] = np.where(is_ivf, total_cycles, 0)
        out["di_total_treatment_load"] = np.where(is_di, total_cycles, 0)
        out["ivf_pregnancy_load"] = np.where(is_ivf, total_preg, 0)
        out["di_pregnancy_load"] = np.where(is_di, total_preg, 0)

        transfer_ratio = safe_div(transferred, embryo_total)
        storage_ratio = safe_div(stored, embryo_total)
        out["ivf_transfer_ratio"] = np.where(is_ivf, transfer_ratio, 0)
        out["di_transfer_ratio"] = np.where(is_di, transfer_ratio, 0)
        out["ivf_storage_ratio"] = np.where(is_ivf, storage_ratio, 0)
        out["di_storage_ratio"] = np.where(is_di, storage_ratio, 0)

        out["ivf_embryo_age_signal"] = np.where(is_ivf, embryo_days * age_val, 0)
        out["di_embryo_age_signal"] = np.where(is_di, embryo_days * age_val, 0)

        # v10 의료 파생변수 (CB용)
        out["ICSI수정효율"] = safe_div(icsi_embryo, icsi_egg)
        out["난자활용률"] = safe_div(icsi_egg, fresh_egg)
        out["난자대비이식배아수"] = safe_div(transferred, fresh_egg)
        out["전체임신률"] = safe_div(total_preg, total_cycles)
        out["IVF임신률"] = safe_div(ivf_preg, ivf_cycles)
        out["임신유지율"] = safe_div(total_birth, total_preg)
        out["IVF임신유지율"] = safe_div(ivf_birth, ivf_preg)
        out["IVF실패횟수"] = np.clip(ivf_cycles - ivf_preg, 0, None)
        out["반복IVF실패_여부"] = (out["IVF실패횟수"] >= 3).astype(int)
        out["클리닉집중도"] = safe_div(clinic_cycles, total_cycles)
        out["임신경험있음"] = (total_preg > 0).astype(int)

        out["나이_제곱"] = age_val**2
        out["고령_여부"] = (age_val >= 35).astype(int)
        out["초고령_여부"] = (age_val >= 40).astype(int)
        out["나이X총시술"] = age_val * total_cycles
        out["나이XIVF실패"] = age_val * out["IVF실패횟수"].values
        out["복합위험도점수"] = (
            out["고령_여부"].values
            + out["초고령_여부"].values
            + out["반복IVF실패_여부"].values
            + (1 - out["임신경험있음"].values)
        )

    return out


# ============================================================
# 5. 전처리 실행
# ============================================================
print("\n## [2] XGBoost 전처리 (v9 + 팀원 의료 파생)")
train_xgb_df = preprocess_xgb(train_raw.copy(), "train")
test_xgb_df = preprocess_xgb(test_raw.copy(), "test")

te_cols_exist = [c for c in TE_COLS if c in train_xgb_df.columns]
obj_cols = train_xgb_df.select_dtypes("object").columns.tolist()
drop_obj = [c for c in obj_cols if c not in te_cols_exist]
if drop_obj:
    train_xgb_df = train_xgb_df.drop(columns=drop_obj)
    test_xgb_df = test_xgb_df.drop(columns=drop_obj)

V10_NEW = [
    "ICSI수정효율",
    "난자활용률",
    "난자대비이식배아수",
    "저장_대비_이식비율",
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
    "고령_여부",
    "초고령_여부",
    "극고령_여부",
    "나이X총시술",
    "나이XIVF실패",
    "나이XIVF임신률",
    "초고령X반복실패",
    "복합위험도점수",
]

# ============================================================
# 6. Target Encoding (v8 동일)
# ============================================================
print(f"\n## [3] Target Encoding (K-Fold)")
global_mean = y.mean()
alpha = 10
te_mappings = {}
skf_te = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

for col in te_cols_exist:
    col_str = train_xgb_df[col].astype(str)
    print(f"  - {col} (고유값: {col_str.nunique()}) → 덮어쓰기")
    oof_te = np.full(len(y), global_mean)
    for tr_i, va_i in skf_te.split(train_xgb_df, y):
        agg_df = pd.DataFrame({"val": col_str.iloc[tr_i].values, "target": y[tr_i]})
        agg = agg_df.groupby("val")["target"].agg(["mean", "count"])
        agg["smoothed"] = (agg["count"] * agg["mean"] + alpha * global_mean) / (
            agg["count"] + alpha
        )
        mapping = agg["smoothed"].to_dict()
        oof_te[va_i] = col_str.iloc[va_i].map(mapping).fillna(global_mean).values
    train_xgb_df[col] = oof_te
    full_df = pd.DataFrame({"val": col_str.values, "target": y})
    full_agg = full_df.groupby("val")["target"].agg(["mean", "count"])
    full_agg["smoothed"] = (
        full_agg["count"] * full_agg["mean"] + alpha * global_mean
    ) / (full_agg["count"] + alpha)
    te_mappings[col] = full_agg["smoothed"].to_dict()
    test_xgb_df[col] = (
        test_xgb_df[col].astype(str).map(te_mappings[col]).fillna(global_mean)
    )

for col in TE_COLS_NUMERIC:
    if col not in train_xgb_df.columns:
        continue
    col_str = train_xgb_df[col].astype(str)
    te_name = f"{col}_te"
    print(f"  - {col} (고유값: {col_str.nunique()}) → {te_name}")
    oof_te = np.full(len(y), global_mean)
    for tr_i, va_i in skf_te.split(train_xgb_df, y):
        agg_df = pd.DataFrame({"val": col_str.iloc[tr_i].values, "target": y[tr_i]})
        agg = agg_df.groupby("val")["target"].agg(["mean", "count"])
        agg["smoothed"] = (agg["count"] * agg["mean"] + alpha * global_mean) / (
            agg["count"] + alpha
        )
        mapping = agg["smoothed"].to_dict()
        oof_te[va_i] = col_str.iloc[va_i].map(mapping).fillna(global_mean).values
    train_xgb_df[te_name] = oof_te
    full_df = pd.DataFrame({"val": col_str.values, "target": y})
    full_agg = full_df.groupby("val")["target"].agg(["mean", "count"])
    full_agg["smoothed"] = (
        full_agg["count"] * full_agg["mean"] + alpha * global_mean
    ) / (full_agg["count"] + alpha)
    te_mappings[col] = full_agg["smoothed"].to_dict()
    test_xgb_df[te_name] = (
        test_xgb_df[col].astype(str).map(te_mappings[col]).fillna(global_mean)
    )

for name, df_ref in [("train", train_xgb_df), ("test", test_xgb_df)]:
    obj_r = df_ref.select_dtypes("object").columns.tolist()
    if obj_r:
        if name == "train":
            train_xgb_df = train_xgb_df.drop(columns=obj_r)
        else:
            test_xgb_df = test_xgb_df.drop(columns=obj_r)

remove_exist = [c for c in REMOVE_ZERO_IMP if c in train_xgb_df.columns]
train_xgb_df = train_xgb_df.drop(columns=remove_exist, errors="ignore")
test_xgb_df = test_xgb_df.drop(columns=remove_exist, errors="ignore")

# NaN 처리 (팀원 스타일 safe_div_nan 결과)
train_xgb_df = train_xgb_df.fillna(0)
test_xgb_df = test_xgb_df.fillna(0)

common_xgb = sorted(set(train_xgb_df.columns) & set(test_xgb_df.columns))
train_xgb_df = train_xgb_df[common_xgb]
test_xgb_df = test_xgb_df[common_xgb]

X_train_xgb = train_xgb_df.values.astype(np.float32)
X_test_xgb = test_xgb_df.values.astype(np.float32)
xgb_features = list(common_xgb)

v10_in_xgb = [f for f in V10_NEW if f in xgb_features]
print(f"\n## [4] XGBoost 피처 정리")
print(f"- 전체 피처: {len(xgb_features)}개")
print(f"- v10 신규 ({len(v10_in_xgb)}개): {v10_in_xgb}")

# ============================================================
# 7. CatBoost 데이터
# ============================================================
print(f"\n## [5] CatBoost 데이터")

train_cb_df = train_raw.drop(columns=["ID", TARGET]).copy()
test_cb_df = test_raw.drop(columns=["ID"]).copy()

drop_cb = [c for c in DROP_COLS if c in train_cb_df.columns and c not in ["ID", TARGET]]
train_cb_df = train_cb_df.drop(columns=drop_cb, errors="ignore")
test_cb_df = test_cb_df.drop(columns=drop_cb, errors="ignore")

remove_cb = [c for c in REMOVE_ZERO_IMP if c in train_cb_df.columns]
train_cb_df = train_cb_df.drop(columns=remove_cb, errors="ignore")
test_cb_df = test_cb_df.drop(columns=remove_cb, errors="ignore")

train_cb_df = build_catboost_features(train_cb_df)
test_cb_df = build_catboost_features(test_cb_df)

cat_cols_cb = []
for col in train_cb_df.columns:
    if train_cb_df[col].dtype == "object":
        cat_cols_cb.append(col)
        train_cb_df[col] = train_cb_df[col].fillna("_missing_").astype(str)
        test_cb_df[col] = test_cb_df[col].fillna("_missing_").astype(str)
    else:
        train_cb_df[col] = train_cb_df[col].fillna(0)
        test_cb_df[col] = test_cb_df[col].fillna(0)

common_cb = sorted(set(train_cb_df.columns) & set(test_cb_df.columns))
train_cb_df = train_cb_df[common_cb]
test_cb_df = test_cb_df[common_cb]
cat_idx_cb = [common_cb.index(c) for c in cat_cols_cb if c in common_cb]

print(f"- CB 피처: {len(common_cb)}개, 카테고리: {len(cat_idx_cb)}개")

# ============================================================
# 8. CatBoost 3-seed 앙상블
# ============================================================
print(f"\n## [6] CatBoost 3-seed 앙상블 (seed={SEEDS_CB})")
print(f"### 파라미터 (v8 동일): iterations=5000, lr=0.01, depth=8, l2=3")

oof_cb_ensemble = np.zeros(len(y))
test_cb_ensemble = np.zeros(len(test_ids))
seed_cb_aucs = []

for seed_idx, seed_val in enumerate(SEEDS_CB):
    print(f"\n  --- CatBoost Seed {seed_val} ({seed_idx + 1}/{len(SEEDS_CB)}) ---")

    oof_cb_seed = np.zeros(len(y))
    test_cb_seed = np.zeros(len(test_ids))
    fold_aucs = []

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed_val)

    for fold, (tr_i, va_i) in enumerate(skf.split(train_cb_df, y), 1):
        fold_start = time.time()

        tr_pool = Pool(train_cb_df.iloc[tr_i], y[tr_i], cat_features=cat_idx_cb)
        va_pool = Pool(train_cb_df.iloc[va_i], y[va_i], cat_features=cat_idx_cb)
        te_pool = Pool(test_cb_df, cat_features=cat_idx_cb)

        cb_model = CatBoostClassifier(
            iterations=5000,
            learning_rate=0.01,
            depth=8,
            l2_leaf_reg=3,
            min_data_in_leaf=20,
            bootstrap_type="Bernoulli",
            subsample=0.8,
            eval_metric="AUC",
            random_seed=seed_val + fold,
            early_stopping_rounds=200,
            verbose=0,
            task_type="GPU",
            devices="0",
        )
        cb_model.fit(tr_pool, eval_set=va_pool, use_best_model=True)

        oof_cb_seed[va_i] = cb_model.predict_proba(va_pool)[:, 1]
        test_cb_seed += cb_model.predict_proba(te_pool)[:, 1] / N_FOLDS

        fold_auc = roc_auc_score(y[va_i], oof_cb_seed[va_i])
        fold_aucs.append(fold_auc)
        fold_time = time.time() - fold_start
        print(
            f"    Fold {fold}: AUC={fold_auc:.4f}, iter={cb_model.best_iteration_}, "
            f"소요={fold_time/60:.1f}분"
        )

    seed_auc = roc_auc_score(y, oof_cb_seed)
    seed_cb_aucs.append(seed_auc)
    print(f"  Seed {seed_val} OOF AUC: {seed_auc:.4f}")

    oof_cb_ensemble += oof_cb_seed / len(SEEDS_CB)
    test_cb_ensemble += test_cb_seed / len(SEEDS_CB)

cb_ensemble_auc = roc_auc_score(y, oof_cb_ensemble)
print(f"\n  === CatBoost 3-seed 앙상블 OOF AUC: {cb_ensemble_auc:.4f} ===")
print(f"  개별 seed: {[f'{a:.4f}' for a in seed_cb_aucs]}")

# ============================================================
# 9. XGBoost 5-Fold (depth=5, 일반화 강화)
# ============================================================
print(f"\n## [7] XGBoost 5-Fold (v10: depth=5, 일반화 강화)")

oof_xgb = np.zeros(len(y))
test_xgb_pred = np.zeros(len(test_ids))
xgb_aucs = []

xgb_params = {
    "n_estimators": 5000,
    "learning_rate": 0.02,
    "max_depth": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "colsample_bylevel": 0.7,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "min_child_weight": 3,
    "gamma": 0.0,
    "eval_metric": "auc",
    "tree_method": "gpu_hist",
    "gpu_id": 0,
    "random_state": SEED,
    "verbosity": 0,
    "early_stopping_rounds": 50,
}

print(
    f"### XGB 파라미터: depth={xgb_params['max_depth']}, lr={xgb_params['learning_rate']}, "
    f"min_child={xgb_params['min_child_weight']}, gamma={xgb_params['gamma']}, "
    f"reg_alpha={xgb_params['reg_alpha']}, reg_lambda={xgb_params['reg_lambda']}"
)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

for fold, (tr_i, va_i) in enumerate(skf.split(X_train_xgb, y), 1):
    fold_start = time.time()
    print(f"\n  [XGB] Fold {fold}/{N_FOLDS} 학습 중...")

    xgb_model = XGBClassifier(**xgb_params)
    xgb_model.fit(
        X_train_xgb[tr_i],
        y[tr_i],
        eval_set=[(X_train_xgb[va_i], y[va_i])],
        verbose=0,
    )
    oof_xgb[va_i] = xgb_model.predict_proba(X_train_xgb[va_i])[:, 1]
    test_xgb_pred += xgb_model.predict_proba(X_test_xgb)[:, 1] / N_FOLDS

    fold_auc = roc_auc_score(y[va_i], oof_xgb[va_i])
    xgb_aucs.append(fold_auc)
    xgb_iter = (
        xgb_model.best_iteration if hasattr(xgb_model, "best_iteration") else "N/A"
    )
    fold_time = time.time() - fold_start
    print(
        f"  [XGB] Fold {fold}: AUC={fold_auc:.4f}, iter={xgb_iter}, 소요={fold_time/60:.1f}분"
    )

oof_auc_xgb = roc_auc_score(y, oof_xgb)
print(f"\n  === XGBoost OOF AUC: {oof_auc_xgb:.4f} ===")

# ============================================================
# 10. 블렌딩
# ============================================================
print(f"\n## [8] 블렌딩 가중치 탐색")

best_w = 0.0
best_blend_auc = 0

for w in np.arange(0.0, 1.01, 0.05):
    blend = w * oof_xgb + (1 - w) * oof_cb_ensemble
    auc_val = roc_auc_score(y, blend)
    if auc_val > best_blend_auc:
        best_blend_auc = auc_val
        best_w = w

best_w_fine = best_w
best_auc_fine = best_blend_auc
for w in np.arange(max(0, best_w - 0.05), min(1.0, best_w + 0.06), 0.01):
    blend = w * oof_xgb + (1 - w) * oof_cb_ensemble
    auc_val = roc_auc_score(y, blend)
    if auc_val > best_auc_fine:
        best_auc_fine = auc_val
        best_w_fine = w

best_w = best_w_fine
best_blend_auc = best_auc_fine
print(f"  최적: XGB={best_w:.2f}, CB={1-best_w:.2f}, AUC={best_blend_auc:.4f}")

if cb_ensemble_auc >= best_blend_auc:
    final_oof_auc = cb_ensemble_auc
    test_final = test_cb_ensemble
    oof_final = oof_cb_ensemble
    use_blend = False
    print(f"  → CB 단독 사용 ({cb_ensemble_auc:.4f})")
else:
    final_oof_auc = best_blend_auc
    test_final = best_w * test_xgb_pred + (1 - best_w) * test_cb_ensemble
    oof_final = best_w * oof_xgb + (1 - best_w) * oof_cb_ensemble
    use_blend = True
    print(f"  → 블렌딩 사용 ({best_blend_auc:.4f})")

# ============================================================
# 11. 다양한 평가지표
# ============================================================
print(f"\n## [9] 종합 평가지표")

# OOF 기반 지표
oof_auc_final = roc_auc_score(y, oof_final)
oof_logloss = log_loss(y, oof_final)
oof_ap = average_precision_score(y, oof_final)

# threshold별 지표
print(f"\n### 확률 기반 지표")
print(f"  OOF AUC:      {oof_auc_final:.6f}")
print(f"  OOF Log Loss: {oof_logloss:.6f}")
print(f"  OOF AP (PR-AUC): {oof_ap:.6f}")

print(f"\n### Threshold별 분류 지표")
print(
    f"  {'Threshold':>10} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Specificity':>10}"
)
print(f"  {'-'*65}")

best_f1 = 0
best_f1_th = 0.5

for th in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
    pred_label = (oof_final >= th).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred_label).ravel()
    acc = (tp + tn) / (tp + tn + fp + fn)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    if f1 > best_f1:
        best_f1 = f1
        best_f1_th = th

    print(
        f"  {th:>10.2f} {acc:>10.4f} {prec:>10.4f} {rec:>10.4f} {f1:>10.4f} {spec:>10.4f}"
    )

print(f"\n  최적 F1: {best_f1:.4f} (threshold={best_f1_th})")

# 개별 모델 지표
print(f"\n### 개별 모델 지표")
print(f"  {'모델':>15} {'AUC':>10} {'Log Loss':>10} {'AP':>10}")
print(f"  {'-'*50}")
xgb_ll = log_loss(y, oof_xgb)
xgb_ap = average_precision_score(y, oof_xgb)
cb_ll = log_loss(y, oof_cb_ensemble)
cb_ap = average_precision_score(y, oof_cb_ensemble)
print(f"  {'XGBoost':>15} {oof_auc_xgb:>10.4f} {xgb_ll:>10.4f} {xgb_ap:>10.4f}")
print(f"  {'CB 3-seed':>15} {cb_ensemble_auc:>10.4f} {cb_ll:>10.4f} {cb_ap:>10.4f}")
if use_blend:
    blend_ll = log_loss(y, oof_final)
    blend_ap = average_precision_score(y, oof_final)
    print(f"  {'Blend':>15} {final_oof_auc:>10.4f} {blend_ll:>10.4f} {blend_ap:>10.4f}")

# ============================================================
# 12. 제출
# ============================================================
print(f"\n## [10] 제출 파일")

submission = pd.DataFrame({"ID": test_ids, "probability": test_final})
main_path = os.path.join(RESULT_DIR, f"sample_submission_{VERSION}_{NOW}.csv")
submission.to_csv(main_path, index=False)

print(f"- 파일: {main_path}")
print(
    f"- 확률: mean={test_final.mean():.4f}, std={test_final.std():.4f}, "
    f"min={test_final.min():.4f}, max={test_final.max():.4f}"
)
print(f"- 예시:")
print(submission.head(5).to_string(index=False))

# ============================================================
# 13. 피처 중요도
# ============================================================
print(f"\n## [11] 피처 중요도")

print(f"\n### XGBoost 상위 30")
fi_xgb = xgb_model.feature_importances_
fi_xgb_df = pd.DataFrame({"feature": xgb_features, "imp": fi_xgb}).sort_values(
    "imp", ascending=False
)
for i, (_, r) in enumerate(fi_xgb_df.head(30).iterrows()):
    marker = " ★v10" if r["feature"] in V10_NEW else ""
    print(f"  {i+1}. {r['feature']}{marker}: {r['imp']:.4f}")

print(f"\n### CatBoost 상위 30 (마지막 모델)")
fi_cb = cb_model.get_feature_importance()
fi_cb_df = pd.DataFrame({"feature": common_cb, "imp": fi_cb}).sort_values(
    "imp", ascending=False
)
for i, (_, r) in enumerate(fi_cb_df.head(30).iterrows()):
    marker = " ★v10" if r["feature"] in V10_NEW else ""
    print(f"  {i+1}. {r['feature']}{marker}: {r['imp']:.2f}")

# ============================================================
# 14. 버전 비교
# ============================================================
print(f"\n## [12] 버전 비교")
print(f"| 버전 | 모델 | OOF AUC | Log Loss | AP | 비고 |")
print(f"|------|------|---------|----------|----|------|")
print(f"| v1 | CB원본 | 0.7403 | - | - | 베이스라인 |")
print(f"| v8 | XGB+CB | 0.7401 | - | - | +파생변수 |")
print(f"| v9 | XGB+CB 3seed | 0.7405 | - | - | +IVF/DI분기 |")
print(
    f"| v10-CB | CB 3seed | {cb_ensemble_auc:.4f} | {cb_ll:.4f} | {cb_ap:.4f} | +의료파생+나이중간값 |"
)
print(
    f"| v10-XGB | XGB depth5 | {oof_auc_xgb:.4f} | {xgb_ll:.4f} | {xgb_ap:.4f} | +팀원파라미터 |"
)
print(
    f"| v10 | {'blend' if use_blend else 'CB단독'} | {final_oof_auc:.4f} | {oof_logloss:.4f} | {oof_ap:.4f} | 최종 |"
)

# ============================================================
# 15. 요약
# ============================================================
total_time = time.time() - start_all
print(f"\n{'='*60}")
print(f"## 최종 요약")
print(f"{'='*60}")
print(f"- v10 핵심 변경:")
print(f"  1. 팀원 의료 파생변수 {len(v10_in_xgb)}개 추가")
print(f"  2. 나이: ordinal → 구간 중간값 (26, 36, 38.5, 41, 43.5, 47.5)")
print(f"  3. XGB depth 7→5, lr 0.01→0.02, 정규화 완화 (팀원 설정)")
print(f"- CatBoost 3-seed: {cb_ensemble_auc:.4f}")
print(f"  - seed별: {[f'{a:.4f}' for a in seed_cb_aucs]}")
print(f"  - 피처: {len(common_cb)}개 (cat={len(cat_idx_cb)})")
print(f"- XGBoost: {oof_auc_xgb:.4f} (depth=5, 피처={len(xgb_features)})")
print(f"- 블렌딩: XGB={best_w:.2f}, CB={1-best_w:.2f}")
print(
    f"- **최종 OOF AUC: {final_oof_auc:.4f}** (v9 대비 {final_oof_auc - 0.7405:+.4f})"
)
print(f"- OOF Log Loss: {oof_logloss:.4f}")
print(f"- OOF AP: {oof_ap:.4f}")
print(f"- 최적 F1: {best_f1:.4f} (th={best_f1_th})")
print(f"- 데이터 누수: 없음")
print(f"- 소요: {total_time/60:.1f}분")
print(f"- 로그: {LOG_PATH}")
print(f"{'='*60}")
