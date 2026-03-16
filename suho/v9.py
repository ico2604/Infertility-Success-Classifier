#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v9.py - v8 기반 + IVF/DI 분기 피처 추가 (팀원 인사이트 반영)
  - v8 전처리/파라미터 100% 유지
  - IVF/DI 분기 피처 16개 추가
  - CatBoost 3-seed 앙상블
  - XGB + CB 블렌딩
"""

import os, sys, warnings, time, re
import numpy as np, pandas as pd
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier, Pool

warnings.filterwarnings("ignore")

VERSION = "v9"
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
print(f"# {VERSION} - v8 기반 + IVF/DI 분기 피처 + CatBoost 3-seed 앙상블")
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
# 2. 상수 (v8 동일)
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

AGE_MAP = {}
for prefix in ["", "만"]:
    for s in ["세", ""]:
        AGE_MAP[f"{prefix}18-34{s}"] = 1
        AGE_MAP[f"{prefix}18{s}-34{s}"] = 1
        AGE_MAP[f"{prefix}35-37{s}"] = 2
        AGE_MAP[f"{prefix}35{s}-37{s}"] = 2
        AGE_MAP[f"{prefix}38-39{s}"] = 3
        AGE_MAP[f"{prefix}38{s}-39{s}"] = 3
        AGE_MAP[f"{prefix}40-42{s}"] = 4
        AGE_MAP[f"{prefix}40{s}-42{s}"] = 4
        AGE_MAP[f"{prefix}43-44{s}"] = 5
        AGE_MAP[f"{prefix}43{s}-44{s}"] = 5
        AGE_MAP[f"{prefix}45-50{s}"] = 7
        AGE_MAP[f"{prefix}45{s}-50{s}"] = 7
AGE_MAP["알 수 없음"] = 0
AGE_MAP["Unknown"] = 0

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
    return np.where(b > 0, a / b, fill)


# ============================================================
# 3. 전처리 함수 (v8 동일 + IVF/DI 분기 피처 추가)
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


def step3_age_to_int(df):
    for col in ["시술 당시 나이", "난자 기증자 나이", "정자 기증자 나이"]:
        if col not in df.columns or df[col].dtype != "object":
            continue
        mapped = df[col].map(AGE_MAP)
        unmapped = mapped.isna() & (df[col] != "Unknown")
        if unmapped.any():
            for idx in df.index[unmapped]:
                val = str(df.at[idx, col])
                nums = re.findall(r"\d+", val)
                if nums:
                    age = int(nums[0])
                    if age <= 34:
                        mapped.at[idx] = 1
                    elif age <= 37:
                        mapped.at[idx] = 2
                    elif age <= 39:
                        mapped.at[idx] = 3
                    elif age <= 42:
                        mapped.at[idx] = 4
                    elif age <= 44:
                        mapped.at[idx] = 5
                    elif age <= 50:
                        mapped.at[idx] = 7
                    else:
                        mapped.at[idx] = 0
        df[col] = mapped.fillna(0).astype(int)
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
    """v8 피처 + v9 IVF/DI 분기 피처"""

    # === v8 기존 피처 (동일) ===
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
        donor_flag = np.where(
            (df["난자 기증자 나이"] != 0) & (df["난자 기증자 나이"] != -1),
            1,
            donor_flag,
        )
    df["is_donor_egg"] = donor_flag.astype(int)

    if "총 시술 횟수" in df.columns and "클리닉 내 총 시술 횟수" in df.columns:
        tc = pd.to_numeric(df["총 시술 횟수"], errors="coerce").fillna(0)
        cc = pd.to_numeric(df["클리닉 내 총 시술 횟수"], errors="coerce").fillna(0)
        df["타클리닉_시술"] = (tc - cc).clip(0).astype(int)

    transferred = (
        pd.to_numeric(df.get("이식된 배아 수", 0), errors="coerce").fillna(0).values
    )
    df["실제이식여부"] = (transferred > 0).astype(int)

    df["배아_이용률"] = safe_div(transferred, embryo_total)
    df["배아_잉여율"] = safe_div(embryo_total - transferred, embryo_total)
    fresh_egg = (
        pd.to_numeric(df.get("수집된 신선 난자 수", 0), errors="coerce")
        .fillna(0)
        .values
    )
    df["난자_배아_전환율"] = safe_div(embryo_total, fresh_egg)

    if "시술 당시 나이" in df.columns:
        age = pd.to_numeric(df["시술 당시 나이"], errors="coerce").fillna(0).values
        df["age_u_curve"] = (age == 7).astype(int)
        df["age_peak"] = (age <= 1).astype(int)

    if "배아 생성 주요 이유" in df.columns:
        reason = df["배아 생성 주요 이유"].astype(str)
        df["purpose_zero"] = (
            reason.str.contains("저장용|기증용", na=False)
            & ~reason.str.contains("현재 시술용", na=False)
        ).astype(int)

    df["total_embryo_ratio"] = safe_div(transferred, embryo_total + 1)

    # === v9 신규: IVF/DI 분기 피처 ===
    if "시술 유형" in df.columns:
        is_ivf = (df["시술 유형"] == 1).values  # IVF=1
        is_di = (df["시술 유형"] == 0).values  # DI=0

        # 플래그
        df["is_ivf"] = is_ivf.astype(int)
        df["is_di"] = is_di.astype(int)

        # 수치 분리: 총 시술 횟수
        total_cycles = (
            pd.to_numeric(df.get("총 시술 횟수", 0), errors="coerce").fillna(0).values
        )
        df["ivf_total_treatment_load"] = np.where(is_ivf, total_cycles, 0)
        df["di_total_treatment_load"] = np.where(is_di, total_cycles, 0)

        # 수치 분리: 총 임신 횟수
        total_preg = (
            pd.to_numeric(df.get("총 임신 횟수", 0), errors="coerce").fillna(0).values
        )
        df["ivf_pregnancy_load"] = np.where(is_ivf, total_preg, 0)
        df["di_pregnancy_load"] = np.where(is_di, total_preg, 0)

        # 비율 분리: 배아 이식 비율 (이식배아/총배아)
        transfer_ratio = safe_div(transferred, embryo_total)
        df["ivf_transfer_ratio"] = np.where(is_ivf, transfer_ratio, 0)
        df["di_transfer_ratio"] = np.where(is_di, transfer_ratio, 0)

        # 비율 분리: 배아 저장 비율 (저장배아/총배아)
        stored = (
            pd.to_numeric(df.get("저장된 배아 수", 0), errors="coerce").fillna(0).values
        )
        storage_ratio = safe_div(stored, embryo_total)
        df["ivf_storage_ratio"] = np.where(is_ivf, storage_ratio, 0)
        df["di_storage_ratio"] = np.where(is_di, storage_ratio, 0)

        # 상호작용: 시술유형 × 배아이식경과일 × 나이
        embryo_days = (
            pd.to_numeric(df.get("배아 이식 경과일", 0), errors="coerce")
            .fillna(0)
            .values
        )
        age_val = (
            pd.to_numeric(df.get("시술 당시 나이", 0), errors="coerce").fillna(0).values
        )
        df["ivf_embryo_age_signal"] = np.where(is_ivf, embryo_days * age_val, 0)
        df["di_embryo_age_signal"] = np.where(is_di, embryo_days * age_val, 0)

        # 조합: 시술유형 × 총시술횟수 (상호작용)
        df["treatment_history_combo"] = df["시술 유형"] * 10 + total_cycles

        # 조합: 시술유형 × 난자출처 × 정자출처
        egg_src = (
            pd.to_numeric(df.get("난자 출처", 0), errors="coerce").fillna(0).values
        )
        sperm_src = (
            pd.to_numeric(df.get("정자 출처", 0), errors="coerce").fillna(0).values
        )
        df["treatment_source_combo"] = df["시술 유형"] * 100 + egg_src * 10 + sperm_src

        # 조합: 시술유형 × 나이 × 총시술횟수
        df["treatment_age_history_combo"] = (
            df["시술 유형"] * 100 + age_val * 10 + total_cycles
        )

    return df


def preprocess_xgb(df, label):
    df = step1_drop_and_fillna(df)
    print(f"  [{label}] step1: {df.shape}, obj={df.select_dtypes('object').shape[1]}")
    df = step2_count_to_int(df)
    print(f"  [{label}] step2: obj={df.select_dtypes('object').shape[1]}")
    df = step3_age_to_int(df)
    print(f"  [{label}] step3: obj={df.select_dtypes('object').shape[1]}")
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
# 4. CatBoost용 피처 (v8 + IVF/DI 분기)
# ============================================================


def build_catboost_features(df):
    """CatBoost용: 원본 카테고리 유지 + 수치 파생변수 + IVF/DI 분기"""
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
    fresh_egg = (
        pd.to_numeric(out.get("수집된 신선 난자 수", 0), errors="coerce")
        .fillna(0)
        .values
    )
    stored = (
        pd.to_numeric(out.get("저장된 배아 수", 0), errors="coerce").fillna(0).values
    )

    # v8 파생변수
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

    if "시술 당시 나이" in out.columns:
        age_str = out["시술 당시 나이"].astype(str)
        out["age_u_curve"] = age_str.str.contains("45-50|45세-50세", na=False).astype(
            int
        )
        out["age_peak"] = age_str.str.contains("18-34|18세-34세", na=False).astype(int)

    # v9 IVF/DI 분기 피처
    if "시술 유형" in out.columns:
        proc_str = out["시술 유형"].astype(str)
        is_ivf = proc_str.str.upper().str.contains("IVF", na=False).values
        is_di = proc_str.str.upper().str.contains("DI", na=False).values

        out["is_ivf"] = is_ivf.astype(int)
        out["is_di"] = is_di.astype(int)

        # 횟수 컬럼은 아직 문자열일 수 있으므로 변환
        def to_num(col_name):
            if col_name not in out.columns:
                return np.zeros(len(out))
            vals = out[col_name]
            if vals.dtype == "object":
                vals = vals.map(COUNT_MAP).fillna(vals)
            return pd.to_numeric(vals, errors="coerce").fillna(0).values

        total_cycles = to_num("총 시술 횟수")
        total_preg = to_num("총 임신 횟수")

        # 나이도 아직 문자열일 수 있음
        age_val_raw = out.get("시술 당시 나이", pd.Series(np.zeros(len(out))))
        if age_val_raw.dtype == "object":
            age_num = age_val_raw.map(AGE_MAP).fillna(0)
        else:
            age_num = age_val_raw.fillna(0)
        age_num = pd.to_numeric(age_num, errors="coerce").fillna(0).values

        embryo_days = (
            pd.to_numeric(out.get("배아 이식 경과일", 0), errors="coerce")
            .fillna(0)
            .values
        )

        # 수치 분리
        out["ivf_total_treatment_load"] = np.where(is_ivf, total_cycles, 0)
        out["di_total_treatment_load"] = np.where(is_di, total_cycles, 0)
        out["ivf_pregnancy_load"] = np.where(is_ivf, total_preg, 0)
        out["di_pregnancy_load"] = np.where(is_di, total_preg, 0)

        # 비율 분리
        transfer_ratio = safe_div(transferred, embryo_total)
        storage_ratio = safe_div(stored, embryo_total)
        out["ivf_transfer_ratio"] = np.where(is_ivf, transfer_ratio, 0)
        out["di_transfer_ratio"] = np.where(is_di, transfer_ratio, 0)
        out["ivf_storage_ratio"] = np.where(is_ivf, storage_ratio, 0)
        out["di_storage_ratio"] = np.where(is_di, storage_ratio, 0)

        # 상호작용
        out["ivf_embryo_age_signal"] = np.where(is_ivf, embryo_days * age_num, 0)
        out["di_embryo_age_signal"] = np.where(is_di, embryo_days * age_num, 0)

    return out


# ============================================================
# 5. 전처리 실행
# ============================================================
print("\n## [2] XGBoost 전처리 (v8 + IVF/DI 분기)")
train_xgb_df = preprocess_xgb(train_raw.copy(), "train")
test_xgb_df = preprocess_xgb(test_raw.copy(), "test")

te_cols_exist = [c for c in TE_COLS if c in train_xgb_df.columns]
obj_cols = train_xgb_df.select_dtypes("object").columns.tolist()
drop_obj = [c for c in obj_cols if c not in te_cols_exist]
if drop_obj:
    train_xgb_df = train_xgb_df.drop(columns=drop_obj)
    test_xgb_df = test_xgb_df.drop(columns=drop_obj)

# v9 신규 피처 목록
V9_NEW_FEATURES = [
    "is_ivf",
    "is_di",
    "ivf_total_treatment_load",
    "di_total_treatment_load",
    "ivf_pregnancy_load",
    "di_pregnancy_load",
    "ivf_transfer_ratio",
    "di_transfer_ratio",
    "ivf_storage_ratio",
    "di_storage_ratio",
    "ivf_embryo_age_signal",
    "di_embryo_age_signal",
    "treatment_history_combo",
    "treatment_source_combo",
    "treatment_age_history_combo",
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

# 잔여 object 제거
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

common_xgb = sorted(set(train_xgb_df.columns) & set(test_xgb_df.columns))
train_xgb_df = train_xgb_df[common_xgb]
test_xgb_df = test_xgb_df[common_xgb]

X_train_xgb = train_xgb_df.values.astype(np.float32)
X_test_xgb = test_xgb_df.values.astype(np.float32)
xgb_features = list(common_xgb)

v9_in_xgb = [f for f in V9_NEW_FEATURES if f in xgb_features]
print(f"\n## [4] XGBoost 피처 정리")
print(f"- 전체 피처 수: {len(xgb_features)}")
print(f"- v9 신규 피처 ({len(v9_in_xgb)}개): {v9_in_xgb}")

# ============================================================
# 7. CatBoost 데이터 (v8 + IVF/DI 분기)
# ============================================================
print(f"\n## [5] CatBoost 데이터 (네이티브 카테고리 + v8 파생 + IVF/DI 분기)")

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

v9_in_cb = [f for f in V9_NEW_FEATURES if f in common_cb]
print(f"- CB 피처: {len(common_cb)}개, 카테고리: {len(cat_idx_cb)}개")
print(f"- v9 신규 피처 ({len(v9_in_cb)}개): {v9_in_cb}")

# ============================================================
# 8. CatBoost 3-seed 앙상블 (v8 파라미터)
# ============================================================
print(f"\n## [6] CatBoost 3-seed 앙상블 (seed={SEEDS_CB})")
print(
    f"### 파라미터 : iterations=6000, lr=0.02, depth=6, l2_leaf_reg=10, min_data_in_leaf=20, random_strength=1.2, subsample=0.8, order_count=128, early_stopping_rounds=300"
)

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
            iterations=6000,
            learning_rate=0.02,
            depth=6,
            l2_leaf_reg=10,
            min_data_in_leaf=20,
            random_strength=1.2,
            # bagging_temperature=0.8,  <-- 이 줄을 삭제하세요.
            border_count=128,
            bootstrap_type="Bernoulli",  # Bernoulli를 쓸 때는 subsample만 사용
            subsample=0.8,
            eval_metric="AUC",
            random_seed=seed_val + fold,
            early_stopping_rounds=300,
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
# 9. XGBoost 5-Fold (v8 동일)
# ============================================================
print(f"\n## [7] XGBoost 5-Fold (v8 파라미터, depth=7)")

oof_xgb = np.zeros(len(y))
test_xgb_pred = np.zeros(len(test_ids))
xgb_aucs = []

xgb_params = {
    "n_estimators": 5000,
    "learning_rate": 0.01,
    "max_depth": 7,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.3,
    "reg_lambda": 0.3,
    "min_child_weight": 20,
    "eval_metric": "auc",
    "tree_method": "gpu_hist",
    "gpu_id": 0,
    "random_state": SEED,
    "verbosity": 0,
    "early_stopping_rounds": 200,
}

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

print(f"  1차: XGB={best_w:.2f}, CB={1-best_w:.2f}, AUC={best_blend_auc:.4f}")

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
print(f"  미세조정: XGB={best_w:.2f}, CB={1-best_w:.2f}, AUC={best_blend_auc:.4f}")

if cb_ensemble_auc >= best_blend_auc:
    print(
        f"  → CB 3-seed({cb_ensemble_auc:.4f}) >= 블렌딩({best_blend_auc:.4f}) → CB 단독"
    )
    final_oof_auc = cb_ensemble_auc
    test_final = test_cb_ensemble
    use_blend = False
else:
    print(
        f"  → 블렌딩({best_blend_auc:.4f}) > CB 3-seed({cb_ensemble_auc:.4f}) → 블렌딩"
    )
    final_oof_auc = best_blend_auc
    test_final = best_w * test_xgb_pred + (1 - best_w) * test_cb_ensemble
    use_blend = True

# ============================================================
# 11. 제출
# ============================================================
print(f"\n## [9] 제출 파일")

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
# 12. 피처 중요도
# ============================================================
print(f"\n## [10] 피처 중요도")

print(f"\n### XGBoost 상위 30")
fi_xgb = xgb_model.feature_importances_
fi_xgb_df = pd.DataFrame({"feature": xgb_features, "imp": fi_xgb}).sort_values(
    "imp", ascending=False
)
for i, (_, r) in enumerate(fi_xgb_df.head(30).iterrows()):
    marker = " ★v9" if r["feature"] in V9_NEW_FEATURES else ""
    print(f"  {i+1}. {r['feature']}{marker}: {r['imp']:.4f}")

print(f"\n### CatBoost 상위 30 (마지막 모델)")
fi_cb = cb_model.get_feature_importance()
fi_cb_df = pd.DataFrame({"feature": common_cb, "imp": fi_cb}).sort_values(
    "imp", ascending=False
)
for i, (_, r) in enumerate(fi_cb_df.head(30).iterrows()):
    marker = " ★v9" if r["feature"] in V9_NEW_FEATURES else ""
    print(f"  {i+1}. {r['feature']}{marker}: {r['imp']:.2f}")

# ============================================================
# 13. 버전 비교
# ============================================================
print(f"\n## [11] 버전 비교")
print(f"| 버전 | 모델 | OOF AUC | 비고 |")
print(f"|------|------|---------|------|")
print(f"| v1 | CB원본 | 0.7403 | 베이스라인 |")
print(f"| v4 | XGB | 0.7392 | 컬럼복원 |")
print(f"| v7 | XGB+CB | 0.7402 | 블렌딩 |")
print(f"| v8 | XGB+CB | 0.7401 | +파생변수 |")
print(f"| v9-CB | CB 3seed | {cb_ensemble_auc:.4f} | +IVF/DI분기 |")
print(f"| v9-XGB | XGB | {oof_auc_xgb:.4f} | +IVF/DI분기 |")
print(f"| v9 | {'blend' if use_blend else 'CB단독'} | {final_oof_auc:.4f} | 최종 |")

# ============================================================
# 14. 요약
# ============================================================
total_time = time.time() - start_all
print(f"\n{'='*60}")
print(f"## 최종 요약")
print(f"{'='*60}")
print(f"- v9 핵심: IVF/DI 분기 피처 {len(v9_in_cb)}개 추가 (팀원 인사이트)")
print(f"- CatBoost 3-seed 앙상블: {cb_ensemble_auc:.4f}")
print(f"  - seed별: {[f'{a:.4f}' for a in seed_cb_aucs]}")
print(f"  - 피처: {len(common_cb)}개 (cat={len(cat_idx_cb)})")
print(f"- XGBoost: {oof_auc_xgb:.4f} (depth=7, 피처={len(xgb_features)})")
print(f"- 블렌딩: XGB={best_w:.2f}, CB={1-best_w:.2f}")
print(
    f"- **최종 OOF AUC: {final_oof_auc:.4f}** (v8 대비 {final_oof_auc - 0.7401:+.4f})"
)
print(f"- 데이터 누수: 없음 (행 단위 연산만 사용)")
print(f"- 소요: {total_time/60:.1f}분")
print(f"- 로그: {LOG_PATH}")
print(f"{'='*60}")
