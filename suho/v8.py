#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v8.py - v7 베이스 + Age U-Curve + Purpose Zero Logic + Total Embryo Ratio
       + CB에 파생변수 추가 + XGB depth 조정
       평가지표: ROC-AUC (확률 제출)
"""

import os, sys, warnings, time, re
import numpy as np, pandas as pd
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier, Pool
from tqdm import tqdm

warnings.filterwarnings("ignore")

VERSION = "v8"
SEED = 42
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
print(
    f"# {VERSION} - Age U-Curve + Purpose Zero + Embryo Ratio + CB파생 + XGB depth조정"
)
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

# [아이디어1] Age U-Curve: 45-50세를 별도 코드 7로 분리
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
        # 45-50: 기존 6 → 7 (U-curve 반등 구간)
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
# 3. XGBoost용 전처리 함수
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
                        mapped.at[idx] = 7  # U-curve
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
    """v7 피처 + v8 신규 3개"""

    # --- v7 기존 ---
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

    # 효율성 지표 (v7)
    df["배아_이용률"] = safe_div(transferred, embryo_total)
    df["배아_잉여율"] = safe_div(embryo_total - transferred, embryo_total)
    fresh_egg = (
        pd.to_numeric(df.get("수집된 신선 난자 수", 0), errors="coerce")
        .fillna(0)
        .values
    )
    df["난자_배아_전환율"] = safe_div(embryo_total, fresh_egg)

    # --- v8 신규 ---

    # [아이디어1] Age U-Curve 플래그
    if "시술 당시 나이" in df.columns:
        age = pd.to_numeric(df["시술 당시 나이"], errors="coerce").fillna(0).values
        df["age_u_curve"] = (age == 7).astype(int)  # 45-50세 = 코드 7
        df["age_peak"] = (age <= 1).astype(int)  # 18-34세 최고 성공률

    # [아이디어2] Purpose Zero Logic
    if "배아 생성 주요 이유" in df.columns:
        reason = df["배아 생성 주요 이유"].astype(str)
        df["purpose_zero"] = (
            reason.str.contains("저장용|기증용", na=False)
            & ~reason.str.contains("현재 시술용", na=False)
        ).astype(int)

    # [아이디어3] Total Embryo Ratio
    df["total_embryo_ratio"] = safe_div(transferred, embryo_total + 1)

    return df


def step7_drop_remaining_object(df, te_cols_to_keep):
    obj_cols = df.select_dtypes(include="object").columns.tolist()
    to_drop = [c for c in obj_cols if c not in te_cols_to_keep]
    if to_drop:
        print(f"  - object 삭제: {to_drop}")
        df = df.drop(columns=to_drop)
    return df


# ============================================================
# 4. CatBoost용 파생변수 함수 (원본 카테고리 유지 + 수치 파생)
# ============================================================


def build_catboost_features(df):
    """CatBoost용: 원본 카테고리 유지 + 수치형 파생변수 추가"""
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

    # 효율성 지표
    out["수정_성공률"] = safe_div(embryo_total, mixed)
    out["배아_이용률"] = safe_div(transferred, embryo_total)
    out["배아_잉여율"] = safe_div(embryo_total - transferred, embryo_total)
    out["난자_배아_전환율"] = safe_div(embryo_total, fresh_egg)

    # 실제이식여부
    out["실제이식여부"] = (transferred > 0).astype(int)

    # v8 신규
    out["total_embryo_ratio"] = safe_div(transferred, embryo_total + 1)

    # purpose_zero
    if "배아 생성 주요 이유" in out.columns:
        reason = out["배아 생성 주요 이유"].astype(str)
        out["purpose_zero"] = (
            reason.str.contains("저장용|기증용", na=False)
            & ~reason.str.contains("현재 시술용", na=False)
        ).astype(int)

    # age_u_curve (원본 나이 컬럼에서 직접)
    if "시술 당시 나이" in out.columns:
        age_str = out["시술 당시 나이"].astype(str)
        out["age_u_curve"] = age_str.str.contains("45-50|45세-50세", na=False).astype(
            int
        )
        out["age_peak"] = age_str.str.contains("18-34|18세-34세", na=False).astype(int)

    return out


# ============================================================
# 5. 전처리 실행
# ============================================================
print("\n## [3] XGBoost 전처리")


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


train_xgb_df = preprocess_xgb(train_raw.copy(), "train")
test_xgb_df = preprocess_xgb(test_raw.copy(), "test")

te_cols_exist = [c for c in TE_COLS if c in train_xgb_df.columns]
train_xgb_df = step7_drop_remaining_object(train_xgb_df, te_cols_exist)
test_xgb_df = step7_drop_remaining_object(test_xgb_df, te_cols_exist)

# ============================================================
# 6. Target Encoding (XGBoost용)
# ============================================================
print(f"\n## [4] Target Encoding (K-Fold)")

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

# 잔여 object
for name, df in [("train", train_xgb_df), ("test", test_xgb_df)]:
    obj_r = df.select_dtypes("object").columns.tolist()
    if obj_r:
        print(f"  {name} 잔여 object 삭제: {obj_r}")
        if name == "train":
            train_xgb_df = train_xgb_df.drop(columns=obj_r)
        else:
            test_xgb_df = test_xgb_df.drop(columns=obj_r)

# 피처 정리
remove_exist = [c for c in REMOVE_ZERO_IMP if c in train_xgb_df.columns]
train_xgb_df = train_xgb_df.drop(columns=remove_exist, errors="ignore")
test_xgb_df = test_xgb_df.drop(columns=remove_exist, errors="ignore")

common_xgb = sorted(set(train_xgb_df.columns) & set(test_xgb_df.columns))
train_xgb_df = train_xgb_df[common_xgb]
test_xgb_df = test_xgb_df[common_xgb]

X_train_xgb = train_xgb_df.values.astype(np.float32)
X_test_xgb = test_xgb_df.values.astype(np.float32)
xgb_features = list(common_xgb)

print(f"\n## [5] XGBoost 피처 정리")
print(f"- 피처 수: {len(xgb_features)}")
print(f"- v8 신규: age_u_curve, age_peak, purpose_zero, total_embryo_ratio")

# ============================================================
# 7. CatBoost용 데이터 (원본 카테고리 + 파생변수)
# ============================================================
print(f"\n## [6] CatBoost 데이터 (네이티브 카테고리 + 파생변수)")

train_cb_df = train_raw.drop(columns=["ID", TARGET]).copy()
test_cb_df = test_raw.drop(columns=["ID"]).copy()

drop_cb = [c for c in DROP_COLS if c in train_cb_df.columns and c not in ["ID", TARGET]]
train_cb_df = train_cb_df.drop(columns=drop_cb, errors="ignore")
test_cb_df = test_cb_df.drop(columns=drop_cb, errors="ignore")

remove_cb = [c for c in REMOVE_ZERO_IMP if c in train_cb_df.columns]
train_cb_df = train_cb_df.drop(columns=remove_cb, errors="ignore")
test_cb_df = test_cb_df.drop(columns=remove_cb, errors="ignore")

# 파생변수 추가
train_cb_df = build_catboost_features(train_cb_df)
test_cb_df = build_catboost_features(test_cb_df)

# 카테고리 컬럼
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
print(f"- CB 파생변수: 수정_성공률, 배아_이용률, 배아_잉여율, 난자_배아_전환율,")
print(
    f"              실제이식여부, total_embryo_ratio, purpose_zero, age_u_curve, age_peak"
)

# ============================================================
# 8. 5-Fold 학습
# ============================================================
print(f"\n## [7] {N_FOLDS}-Fold: XGBoost(depth=7) + CatBoost(GPU 네이티브)")

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

oof_xgb = np.zeros(len(y))
oof_cb_pred = np.zeros(len(y))
test_xgb_pred = np.zeros(len(test_ids))
test_cb_pred = np.zeros(len(test_ids))

# XGB: depth 8→7로 조정 (과적합 방지, 블렌딩 다양성)
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

print(f"\n### XGBoost: depth={xgb_params['max_depth']}, 피처={len(xgb_features)}")
print(f"### CatBoost: depth=8, 피처={len(common_cb)}, cat={len(cat_idx_cb)}")

folds_list = list(skf.split(X_train_xgb, y))

for fold, (tr_i, va_i) in enumerate(folds_list):
    print(f"\n  ===== Fold {fold+1}/{N_FOLDS} =====")
    fold_start = time.time()

    # --- XGBoost ---
    print(f"    [XGB] 학습 중...")
    xgb_model = XGBClassifier(**xgb_params)
    xgb_model.fit(
        X_train_xgb[tr_i],
        y[tr_i],
        eval_set=[(X_train_xgb[va_i], y[va_i])],
        verbose=0,
    )
    oof_xgb[va_i] = xgb_model.predict_proba(X_train_xgb[va_i])[:, 1]
    test_xgb_pred += xgb_model.predict_proba(X_test_xgb)[:, 1] / N_FOLDS
    auc_xgb = roc_auc_score(y[va_i], oof_xgb[va_i])
    xgb_iter = (
        xgb_model.best_iteration if hasattr(xgb_model, "best_iteration") else "N/A"
    )
    print(f"    [XGB] AUC: {auc_xgb:.4f}, iter: {xgb_iter}")

    # --- CatBoost ---
    print(f"    [CB] 학습 중...")
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
        random_seed=SEED + fold,
        early_stopping_rounds=200,
        verbose=0,
        task_type="GPU",
        devices="0",
    )
    cb_model.fit(tr_pool, eval_set=va_pool, use_best_model=True)
    oof_cb_pred[va_i] = cb_model.predict_proba(va_pool)[:, 1]
    test_cb_pred += cb_model.predict_proba(te_pool)[:, 1] / N_FOLDS
    auc_cb = roc_auc_score(y[va_i], oof_cb_pred[va_i])
    print(f"    [CB] AUC: {auc_cb:.4f}, iter: {cb_model.best_iteration_}")

    elapsed = time.time() - fold_start
    print(f"    소요: {elapsed/60:.1f}분")

# 개별 OOF
oof_auc_xgb = roc_auc_score(y, oof_xgb)
oof_auc_cb = roc_auc_score(y, oof_cb_pred)
print(f"\n  === 개별 OOF AUC ===")
print(f"    XGBoost:  {oof_auc_xgb:.4f}")
print(f"    CatBoost: {oof_auc_cb:.4f}")

# ============================================================
# 9. 블렌딩
# ============================================================
print(f"\n## [8] 블렌딩 가중치 탐색")

best_auc = 0
best_w = 0.5

for w in np.arange(0.0, 1.01, 0.05):
    blend = w * oof_xgb + (1 - w) * oof_cb_pred
    auc = roc_auc_score(y, blend)
    if auc > best_auc:
        best_auc = auc
        best_w = round(w, 2)

print(f"  1차: XGB={best_w}, CB={round(1-best_w, 2)}, AUC={best_auc:.4f}")

for w in np.arange(max(0, best_w - 0.05), min(1.0, best_w + 0.06), 0.01):
    blend = w * oof_xgb + (1 - w) * oof_cb_pred
    auc = roc_auc_score(y, blend)
    if auc > best_auc:
        best_auc = auc
        best_w = round(w, 2)

print(f"  최종: XGB={best_w}, CB={round(1-best_w, 2)}, AUC={best_auc:.4f}")

oof_final = best_w * oof_xgb + (1 - best_w) * oof_cb_pred
test_final = best_w * test_xgb_pred + (1 - best_w) * test_cb_pred

# ============================================================
# 10. 제출
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
# 11. 피처 중요도
# ============================================================
print(f"\n## [10] 피처 중요도")

v8_new = ["age_u_curve", "age_peak", "purpose_zero", "total_embryo_ratio"]

print(f"\n### XGBoost 상위 25")
fi_xgb = xgb_model.feature_importances_
fi_xgb_df = pd.DataFrame({"feature": xgb_features, "imp": fi_xgb}).sort_values(
    "imp", ascending=False
)
for i, (_, r) in enumerate(fi_xgb_df.head(25).iterrows()):
    marker = " ★v8" if r["feature"] in v8_new else ""
    print(f"  {i+1}. {r['feature']}{marker}: {r['imp']:.4f}")

print(f"\n### CatBoost 상위 25")
fi_cb = cb_model.get_feature_importance()
fi_cb_df = pd.DataFrame({"feature": common_cb, "imp": fi_cb}).sort_values(
    "imp", ascending=False
)
for i, (_, r) in enumerate(fi_cb_df.head(25).iterrows()):
    marker = " ★v8" if r["feature"] in v8_new else ""
    print(f"  {i+1}. {r['feature']}{marker}: {r['imp']:.2f}")

# ============================================================
# 12. 버전 비교
# ============================================================
print(f"\n## [11] 버전 비교")
print(
    f"""
| 버전 | 모델 | OOF AUC | 비고 |
|------|------|---------|------|
| v1 | CB원본 | 0.7403 | 베이스라인 |
| v4 | XGB | 0.7392 | 컬럼복원 |
| v7 | XGB+CB | 0.7402 | 블렌딩 |
| v8 | XGB+CB | {best_auc:.4f} | U-curve+Purpose+Ratio |
| v8-XGB | XGB | {oof_auc_xgb:.4f} | depth=7 |
| v8-CB | CB | {oof_auc_cb:.4f} | +파생변수 |
"""
)

# ============================================================
# 13. 요약
# ============================================================
total_time = time.time() - start_all
print(f"{'='*60}")
print(f"## 최종 요약")
print(f"{'='*60}")
print(f"- XGB OOF AUC: {oof_auc_xgb:.4f} (depth=7, 피처={len(xgb_features)})")
print(f"- CB OOF AUC:  {oof_auc_cb:.4f} (네이티브cat+파생, 피처={len(common_cb)})")
print(f"- 블렌딩: XGB={best_w}, CB={round(1-best_w, 2)}")
print(f"- **최종 OOF AUC: {best_auc:.4f}**")
print(f"- v7 대비: {best_auc - 0.7402:+.4f}")
print(f"- 소요: {total_time/60:.1f}분")
print(f"- 로그: {LOG_PATH}")
print(f"{'='*60}")
