#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v5.py - v4 개선: DI결측활용 + 시술유형정보추출 + 출처조합
       + 나이x이식상호작용 + 배아목적단순화 + 불임원인총합
       평가지표: ROC-AUC (확률 제출)
"""

import os, sys, warnings, time, re
import numpy as np, pandas as pd
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score
from xgboost import XGBClassifier
from tqdm import tqdm

warnings.filterwarnings("ignore")

VERSION = "v5"
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
print(f"# {VERSION} - v4 + DI결측 + 시술추출 + 출처조합 + 상호작용 + 목적단순화")
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
# 2. 상수 정의
# ============================================================

DROP_COLS = [
    "ID",
    TARGET,
    "PGD 시술 여부",
    "PGS 시술 여부",
    "난자 해동 경과일",
    "착상 전 유전 검사 사용 여부",
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
        AGE_MAP[f"{prefix}45-50{s}"] = 6
        AGE_MAP[f"{prefix}45{s}-50{s}"] = 6
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

# DI일 때 100% 결측인 컬럼 (EDA 기반)
DI_MISSING_COLS = [
    "단일 배아 이식 여부",
    "착상 전 유전 진단 사용 여부",
    "배아 생성 주요 이유",
    "총 생성 배아 수",
    "미세주입된 난자 수",
    "미세주입에서 생성된 배아 수",
    "이식된 배아 수",
    "미세주입 배아 이식 수",
    "저장된 배아 수",
    "미세주입 후 저장된 배아 수",
    "해동된 배아 수",
    "해동 난자 수",
    "수집된 신선 난자 수",
    "저장된 신선 난자 수",
    "혼합된 난자 수",
    "파트너 정자와 혼합된 난자 수",
    "기증자 정자와 혼합된 난자 수",
    "동결 배아 사용 여부",
    "신선 배아 사용 여부",
    "기증 배아 사용 여부",
    "대리모 여부",
    "배아 이식 경과일",
    "난자 채취 경과일",
    "난자 혼합 경과일",
    "배아 해동 경과일",
]


def safe_div(a, b, fill=0.0):
    return np.where(b > 0, a / b, fill)


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


def step3_age_to_int(df):
    for col in ["시술 당시 나이", "난자 기증자 나이", "정자 기증자 나이"]:
        if col not in df.columns:
            continue
        if df[col].dtype == "object":
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
                            mapped.at[idx] = 6
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
    """v4 피처 + v5 신규 피처"""

    # === v4 기존 피처 ===

    # is_blastocyst
    if "특정 시술 유형" in df.columns:
        df["is_blastocyst"] = (
            df["특정 시술 유형"]
            .apply(
                lambda x: 1 if isinstance(x, str) and "BLASTOCYST" in x.upper() else 0
            )
            .astype(int)
        )

    # 수정_성공률
    embryo = (
        pd.to_numeric(df.get("총 생성 배아 수", 0), errors="coerce").fillna(0).values
    )
    mixed = pd.to_numeric(df.get("혼합된 난자 수", 0), errors="coerce").fillna(0).values
    df["수정_성공률"] = safe_div(embryo, mixed)

    # 이식 유형 One-Hot
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

    # is_donor_egg
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

    # 타클리닉 시술
    if "총 시술 횟수" in df.columns and "클리닉 내 총 시술 횟수" in df.columns:
        tc = pd.to_numeric(df["총 시술 횟수"], errors="coerce").fillna(0)
        cc = pd.to_numeric(df["클리닉 내 총 시술 횟수"], errors="coerce").fillna(0)
        df["타클리닉_시술"] = (tc - cc).clip(0).astype(int)

    # 실제이식여부
    transferred = (
        pd.to_numeric(df.get("이식된 배아 수", 0), errors="coerce").fillna(0).values
    )
    df["실제이식여부"] = (transferred > 0).astype(int)

    # === v5 신규 피처 ===

    # [개선1] DI 구조적 결측 활용: is_DI + DI 결측 개수
    if "시술 유형" in df.columns:
        df["is_DI"] = (df["시술 유형"] == 0).astype(int)

    di_miss_exist = [c for c in DI_MISSING_COLS if c in df.columns]
    if di_miss_exist:
        df["DI_결측수"] = 0
        for c in di_miss_exist:
            col_val = pd.to_numeric(df[c], errors="coerce").fillna(0)
            df["DI_결측수"] += (col_val == 0).astype(int)

    # [개선2] 특정 시술 유형에서 정보 추출
    if "특정 시술 유형" in df.columns:
        stype = df["특정 시술 유형"].astype(str)
        # 반복 시술 (콜론 포함 = 여러 시술 조합, 성공률 ~1%)
        df["is_repeat_procedure"] = stype.str.contains(":", na=False).astype(int)
        # AH (Assisted Hatching) 포함
        df["is_AH"] = stype.str.contains("AH", case=False, na=False).astype(int)
        # IUI (DI 관련 시술)
        df["is_IUI"] = stype.str.contains("IUI", case=False, na=False).astype(int)
        # ICSI 여부
        df["is_ICSI"] = stype.apply(
            lambda x: (
                1 if isinstance(x, str) and "ICSI" in x.upper() and ":" not in x else 0
            )
        ).astype(int)

    # [개선3] 난자×정자 출처 조합
    if "난자 출처" in df.columns and "정자 출처" in df.columns:
        egg = df["난자 출처"].fillna(-1).astype(int)
        sperm = df["정자 출처"].fillna(-1).astype(int)
        # 조합을 고유 정수로: egg * 10 + sperm
        df["출처_조합"] = egg * 10 + sperm

        # 주요 조합 플래그
        df["본인난자_배우자정자"] = ((egg == 0) & (sperm == 0)).astype(int)
        df["기증난자_배우자정자"] = ((egg == 1) & (sperm == 0)).astype(int)
        df["본인난자_기증정자"] = ((egg == 0) & (sperm == 1)).astype(int)

    # [개선4] 나이 × 이식배아 상호작용
    if "시술 당시 나이" in df.columns:
        age = pd.to_numeric(df["시술 당시 나이"], errors="coerce").fillna(0).values
        df["나이x이식배아"] = age * transferred
        df["나이x총배아"] = (
            age
            * pd.to_numeric(df.get("총 생성 배아 수", 0), errors="coerce")
            .fillna(0)
            .values
        )

        # 고령 플래그
        df["고령"] = (age >= 4).astype(int)  # 40세 이상
        df["젊음"] = (age <= 1).astype(int)  # 18-34세
        df["젊음x배아많음"] = (df["젊음"] * (embryo >= 5)).astype(int)
        df["고령x이식함"] = (df["고령"] * df["실제이식여부"]).astype(int)

    # [개선5] 배아 생성 목적 단순화
    if "배아 생성 주요 이유" in df.columns:
        reason = df["배아 생성 주요 이유"].astype(str)
        # "현재 시술용"이 포함되면 실제 이식 목적
        df["시술목적_이식있음"] = reason.str.contains("현재 시술용", na=False).astype(
            int
        )
        # "기증용, 현재 시술용"은 성공률 38%로 가장 높음
        df["시술목적_기증시술"] = reason.str.contains(
            "기증용.*현재 시술용", na=False
        ).astype(int)
        # 저장/기증 전용 (성공률 0~0.1%)
        df["시술목적_저장전용"] = (
            reason.str.contains("저장용|기증용", na=False)
            & ~reason.str.contains("현재 시술용", na=False)
        ).astype(int)

    # [추가] 불임원인 총합
    inf_cols = [c for c in df.columns if c.startswith("불임 원인 - ")]
    if inf_cols:
        inf_values = df[inf_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
        df["불임원인_총합"] = inf_values.sum(axis=1).astype(int)
        df["불임원인_없음"] = (df["불임원인_총합"] == 0).astype(int)
        df["불임원인_다수"] = (df["불임원인_총합"] >= 3).astype(int)

    return df


def step7_drop_remaining_object(df, te_cols_to_keep):
    obj_cols = df.select_dtypes(include="object").columns.tolist()
    to_drop = [c for c in obj_cols if c not in te_cols_to_keep]
    if to_drop:
        print(f"  - object 삭제: {to_drop}")
        df = df.drop(columns=to_drop)
    return df


# ============================================================
# 3. 전처리 실행
# ============================================================
print("\n## [3] 전처리 실행")


def preprocess(df, label):
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


train_df = preprocess(train_raw.copy(), "train")
test_df = preprocess(test_raw.copy(), "test")

te_cols_exist = [c for c in TE_COLS if c in train_df.columns]
train_df = step7_drop_remaining_object(train_df, te_cols_exist)
test_df = step7_drop_remaining_object(test_df, te_cols_exist)
print(
    f"  step7 후 - train: {train_df.shape}, obj={train_df.select_dtypes('object').shape[1]}"
)
print(
    f"  step7 후 - test: {test_df.shape}, obj={test_df.select_dtypes('object').shape[1]}"
)

# ============================================================
# 4. Target Encoding
# ============================================================
print(f"\n## [4] Target Encoding (K-Fold)")

global_mean = y.mean()
alpha = 10
te_mappings = {}
skf_te = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

for col in te_cols_exist:
    print(f"  - {col} (고유값: {train_df[col].nunique()})")
    col_str = train_df[col].astype(str)
    oof_te = np.full(len(y), global_mean)

    for tr_i, va_i in skf_te.split(train_df, y):
        fold_vals = col_str.iloc[tr_i].values
        fold_y = y[tr_i]
        agg_df = pd.DataFrame({"val": fold_vals, "target": fold_y})
        agg = agg_df.groupby("val")["target"].agg(["mean", "count"])
        agg["smoothed"] = (agg["count"] * agg["mean"] + alpha * global_mean) / (
            agg["count"] + alpha
        )
        mapping = agg["smoothed"].to_dict()
        oof_te[va_i] = col_str.iloc[va_i].map(mapping).fillna(global_mean).values

    train_df[col] = oof_te

    full_df = pd.DataFrame({"val": col_str.values, "target": y})
    full_agg = full_df.groupby("val")["target"].agg(["mean", "count"])
    full_agg["smoothed"] = (
        full_agg["count"] * full_agg["mean"] + alpha * global_mean
    ) / (full_agg["count"] + alpha)
    te_mappings[col] = full_agg["smoothed"].to_dict()
    test_df[col] = test_df[col].astype(str).map(te_mappings[col]).fillna(global_mean)

# 잔여 object
for name, df in [("train", train_df), ("test", test_df)]:
    obj_r = df.select_dtypes("object").columns.tolist()
    if obj_r:
        print(f"  {name} 잔여 object 삭제: {obj_r}")
        if name == "train":
            train_df = train_df.drop(columns=obj_r)
        else:
            test_df = test_df.drop(columns=obj_r)

# ============================================================
# 5. 최종 피처 정리
# ============================================================
print(f"\n## [5] 최종 피처 정리")

common_cols = sorted(set(train_df.columns) & set(test_df.columns))
train_df = train_df[common_cols]
test_df = test_df[common_cols]

X_train = train_df.values.astype(np.float32)
X_test = test_df.values.astype(np.float32)
feature_names = list(common_cols)

print(f"- X_train: {X_train.shape}")
print(f"- X_test: {X_test.shape}")
print(f"- 피처 수: {len(feature_names)}")
print(f"- NaN: train={np.isnan(X_train).sum()}, test={np.isnan(X_test).sum()}")

# v5 신규 피처 확인
v5_new = [
    "is_DI",
    "DI_결측수",
    "is_repeat_procedure",
    "is_AH",
    "is_IUI",
    "is_ICSI",
    "출처_조합",
    "본인난자_배우자정자",
    "기증난자_배우자정자",
    "본인난자_기증정자",
    "나이x이식배아",
    "나이x총배아",
    "고령",
    "젊음",
    "젊음x배아많음",
    "고령x이식함",
    "시술목적_이식있음",
    "시술목적_기증시술",
    "시술목적_저장전용",
    "불임원인_총합",
    "불임원인_없음",
    "불임원인_다수",
]
v5_actual = [c for c in v5_new if c in feature_names]
print(f"- v5 신규 피처 ({len(v5_actual)}개): {v5_actual}")

print(f"- 전체 피처 목록:")
for i, f in enumerate(feature_names):
    marker = " ★" if f in v5_actual else ""
    print(f"  {i+1}. {f}{marker}")

# ============================================================
# 6. XGBoost 5-Fold (GPU)
# ============================================================
print(f"\n## [6] XGBoost {N_FOLDS}-Fold (GPU)")

xgb_params = {
    "n_estimators": 5000,
    "learning_rate": 0.01,
    "max_depth": 8,
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
print(f"- 파라미터:")
for k, v in xgb_params.items():
    print(f"  {k}: {v}")

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof_pred = np.zeros(len(y))
test_pred = np.zeros(len(X_test))
fold_results = []

for fold, (tr_i, va_i) in enumerate(
    tqdm(list(skf.split(X_train, y)), desc="XGB Folds", ncols=80)
):
    print(f"\n### Fold {fold+1}/{N_FOLDS}")
    fold_start = time.time()

    model = XGBClassifier(**xgb_params)
    model.fit(
        X_train[tr_i],
        y[tr_i],
        eval_set=[(X_train[va_i], y[va_i])],
        verbose=500,
    )

    pred = model.predict_proba(X_train[va_i])[:, 1]
    oof_pred[va_i] = pred
    test_pred += model.predict_proba(X_test)[:, 1] / N_FOLDS

    auc = roc_auc_score(y[va_i], pred)
    f1 = f1_score(y[va_i], (pred >= 0.5).astype(int))
    best_iter = model.best_iteration if hasattr(model, "best_iteration") else "N/A"
    elapsed = time.time() - fold_start

    fold_results.append(
        {
            "fold": fold + 1,
            "auc": auc,
            "f1": f1,
            "best_iter": best_iter,
            "time": elapsed,
        }
    )
    print(
        f"  AUC: {auc:.4f}, F1: {f1:.4f}, iter: {best_iter}, 소요: {elapsed/60:.1f}분"
    )

# ============================================================
# 7. 전체 성능
# ============================================================
print(f"\n## [7] 전체 성능")

oof_auc = roc_auc_score(y, oof_pred)
oof_f1 = f1_score(y, (oof_pred >= 0.5).astype(int))

print(f"- **OOF AUC: {oof_auc:.4f}**")
print(f"- OOF F1 (th=0.5): {oof_f1:.4f}")
for r in fold_results:
    print(
        f"  Fold {r['fold']}: AUC={r['auc']:.4f}, F1={r['f1']:.4f}, "
        f"iter={r['best_iter']}, {r['time']/60:.1f}분"
    )
aucs = [r["auc"] for r in fold_results]
print(f"- 평균 AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")

# ============================================================
# 8. 피처 중요도
# ============================================================
print(f"\n## [8] 피처 중요도")
fi = model.feature_importances_
fi_df = pd.DataFrame({"feature": feature_names, "importance": fi})
fi_df = fi_df.sort_values("importance", ascending=False)
print(f"| 순위 | 피처 | 중요도 |")
print(f"|------|------|--------|")
for i, (_, r) in enumerate(fi_df.head(30).iterrows()):
    marker = " ★v5" if r["feature"] in v5_actual else ""
    print(f"| {i+1} | {r['feature']}{marker} | {r['importance']:.4f} |")

# 중요도 0 피처
zero_fi = fi_df[fi_df["importance"] == 0]["feature"].tolist()
if zero_fi:
    print(f"\n중요도 0 피처 ({len(zero_fi)}개): {zero_fi}")

# ============================================================
# 9. 제출 파일
# ============================================================
print(f"\n## [9] 제출 파일")

submission = pd.DataFrame({"ID": test_ids, "probability": test_pred})
main_path = os.path.join(RESULT_DIR, f"sample_submission_{VERSION}_{NOW}.csv")
submission.to_csv(main_path, index=False)

print(f"- 파일: {main_path}")
print(
    f"- 확률: mean={test_pred.mean():.4f}, std={test_pred.std():.4f}, "
    f"min={test_pred.min():.4f}, max={test_pred.max():.4f}"
)
print(f"- 예시:")
print(submission.head(5).to_string(index=False))

# ============================================================
# 10. v4 vs v5 비교
# ============================================================
print(f"\n## [10] v4 → v5 변경사항")
print(
    f"""
| 항목 | v4 | v5 |
|------|-----|-----|
| 피처 수 | 73 | {len(feature_names)} |
| is_DI / DI_결측수 | 없음 | 추가 |
| 시술유형 추출 (반복/AH/IUI/ICSI) | 없음 | 추가 |
| 난자×정자 출처 조합 | 없음 | 추가 |
| 나이×이식배아 상호작용 | 없음 | 추가 |
| 배아목적 단순화 (이식/기증시술/저장) | 없음 | 추가 |
| 불임원인 총합/없음/다수 | 없음 | 추가 |
| OOF AUC | 0.7392 | {oof_auc:.4f} |
| 변화 | - | {oof_auc - 0.7392:+.4f} |
"""
)

# ============================================================
# 11. 최종 요약
# ============================================================
total_time = time.time() - start_all
print(f"{'='*60}")
print(f"## 최종 요약")
print(f"{'='*60}")
print(f"- 모델: XGBoost (GPU)")
print(
    f"- 피처 수: {len(feature_names)} (v4: 73 → v5: {len(feature_names)}, +{len(feature_names)-73})"
)
print(f"- **OOF AUC: {oof_auc:.4f}**")
print(f"- v4 대비: {oof_auc - 0.7392:+.4f}")
print(f"- v1(CatBoost) 대비: {oof_auc - 0.7403:+.4f}")
print(f"- Data Leakage: 없음")
print(f"- 총 소요시간: {total_time/60:.1f}분")
print(f"- 로그: {LOG_PATH}")
print(f"{'='*60}")
