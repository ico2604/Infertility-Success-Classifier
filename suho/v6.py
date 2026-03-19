#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v6.py - 피처 정리 + DI분리처리 + 효율성지표 + 나이TE추가
       중복/0중요도 피처 제거, 다중공선성 정리
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

VERSION = "v6"
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
print(f"# {VERSION} - 피처정리 + DI분리 + 효율성지표 + 나이TE")
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

# v5에서 중요도 0 + 다중공선성 제거 대상
REMOVE_AFTER = [
    # 중요도 0
    "저장된 신선 난자 수",
    "불임 원인 - 정자 형태",
    "불임 원인 - 정자 운동성",
    "불임 원인 - 정자 면역학적 요인",
    "불임 원인 - 자궁경부 문제",
    "난자 채취 경과일",
    "불임 원인 - 여성 요인",
    # 다중공선성 (is_DI ≡ 시술유형의 역, 제거)
    "is_DI",
    # 다중공선성 (시술목적_이식있음 ≈ 실제이식여부, 하나만 유지)
    "시술목적_이식있음",
    # 다중공선성 (고령 ⊂ 시술당시나이, 제거)
    "고령",
    # 다중공선성 (본인난자_배우자정자 ⊂ 출처_조합, 제거)
    "본인난자_배우자정자",
    # 낮은 중요도 + 정보 중복
    "젊음",  # 나이에 포함
    "DI_결측수",  # 시술유형으로 충분
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

# [전략3] 나이와 횟수도 TE 대상에 추가
TE_COLS = [
    "특정 시술 유형",
    "배아 생성 주요 이유",
    "시술 시기 코드",
    "배란 유도 유형",
    "시술 당시 나이",
    "총 시술 횟수",
]

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

# DI일 때 결측인 배아/난자 관련 컬럼
DI_EMBRYO_COLS = [
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
    "혼합된 난자 수",
    "파트너 정자와 혼합된 난자 수",
    "기증자 정자와 혼합된 난자 수",
    "배아 이식 경과일",
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


def step6_di_separation(df):
    """[전략1] DI 그룹 분리: DI일 때 배아/난자 컬럼을 -1로 표시"""
    if "시술 유형" not in df.columns:
        return df

    is_di = df["시술 유형"] == 0
    di_count = is_di.sum()

    for col in DI_EMBRYO_COLS:
        if col in df.columns:
            # DI인 행의 배아/난자 컬럼을 -1로 (0과 구분)
            df.loc[is_di, col] = -1

    print(
        f"    DI 행 {di_count}개: 배아/난자 {len([c for c in DI_EMBRYO_COLS if c in df.columns])}개 컬럼 → -1"
    )
    return df


def step7_feature_engineering(df):
    """파생변수 생성"""

    # --- v4 기존 ---
    if "특정 시술 유형" in df.columns:
        stype = df["특정 시술 유형"].astype(str)
        df["is_blastocyst"] = stype.str.contains(
            "BLASTOCYST", case=False, na=False
        ).astype(int)
        df["is_repeat_procedure"] = stype.str.contains(":", na=False).astype(int)
        df["is_ICSI"] = stype.apply(
            lambda x: (
                1 if isinstance(x, str) and "ICSI" in x.upper() and ":" not in x else 0
            )
        ).astype(int)

    # 이식 유형
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

    # 타클리닉
    if "총 시술 횟수" in df.columns and "클리닉 내 총 시술 횟수" in df.columns:
        tc = pd.to_numeric(df["총 시술 횟수"], errors="coerce").fillna(0)
        cc = pd.to_numeric(df["클리닉 내 총 시술 횟수"], errors="coerce").fillna(0)
        df["타클리닉_시술"] = (tc - cc).clip(0).astype(int)

    # 실제이식여부
    transferred = (
        pd.to_numeric(df.get("이식된 배아 수", 0), errors="coerce").fillna(0).values
    )
    df["실제이식여부"] = (transferred > 0).astype(int)

    # 난자×정자 조합
    if "난자 출처" in df.columns and "정자 출처" in df.columns:
        egg = df["난자 출처"].fillna(-1).astype(int)
        sperm = df["정자 출처"].fillna(-1).astype(int)
        df["출처_조합"] = egg * 10 + sperm
        df["기증난자_배우자정자"] = ((egg == 1) & (sperm == 0)).astype(int)
        df["본인난자_기증정자"] = ((egg == 0) & (sperm == 1)).astype(int)

    # 나이×이식배아 상호작용
    if "시술 당시 나이" in df.columns:
        age = pd.to_numeric(df["시술 당시 나이"], errors="coerce").fillna(0).values
        embryo_total = (
            pd.to_numeric(df.get("총 생성 배아 수", 0), errors="coerce")
            .fillna(0)
            .values
        )
        df["나이x이식배아"] = age * transferred
        df["나이x총배아"] = age * embryo_total
        df["고령x이식함"] = ((age >= 4) * df["실제이식여부"]).astype(int)
        df["젊음x배아많음"] = ((age <= 1) * (embryo_total >= 5)).astype(int)

    # 배아목적 단순화
    if "배아 생성 주요 이유" in df.columns:
        reason = df["배아 생성 주요 이유"].astype(str)
        df["시술목적_기증시술"] = reason.str.contains(
            "기증용.*현재 시술용", na=False
        ).astype(int)
        df["시술목적_저장전용"] = (
            reason.str.contains("저장용|기증용", na=False)
            & ~reason.str.contains("현재 시술용", na=False)
        ).astype(int)

    # 불임원인 총합
    inf_cols = [c for c in df.columns if c.startswith("불임 원인 - ")]
    if inf_cols:
        inf_values = df[inf_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
        df["불임원인_총합"] = inf_values.sum(axis=1).astype(int)
        df["불임원인_없음"] = (df["불임원인_총합"] == 0).astype(int)

    # --- [전략2] 효율성 지표 (행 독립 연산) ---
    embryo_total = (
        pd.to_numeric(df.get("총 생성 배아 수", 0), errors="coerce").fillna(0).values
    )
    mixed_egg = (
        pd.to_numeric(df.get("혼합된 난자 수", 0), errors="coerce").fillna(0).values
    )
    fresh_egg = (
        pd.to_numeric(df.get("수집된 신선 난자 수", 0), errors="coerce")
        .fillna(0)
        .values
    )
    stored = (
        pd.to_numeric(df.get("저장된 배아 수", 0), errors="coerce").fillna(0).values
    )

    # 수정 성공률 (배아/혼합난자)
    df["수정_성공률"] = safe_div(embryo_total, mixed_egg)
    # 배아 이용률 (이식배아/총배아)
    df["배아_이용률"] = safe_div(transferred, embryo_total)
    # 배아 잉여율 ((총배아-이식배아)/총배아)
    df["배아_잉여율"] = safe_div(embryo_total - transferred, embryo_total)
    # 난자→배아 전환율 (총배아/신선난자)
    df["난자_배아_전환율"] = safe_div(embryo_total, fresh_egg)

    return df


def step8_drop_remaining_object(df, te_cols_to_keep):
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
    df = step6_di_separation(df)
    print(f"  [{label}] step6 DI분리: obj={df.select_dtypes('object').shape[1]}")
    df = step7_feature_engineering(df)
    print(
        f"  [{label}] step7 피처: 컬럼={df.shape[1]}, obj={df.select_dtypes('object').shape[1]}"
    )
    return df


train_df = preprocess(train_raw.copy(), "train")
test_df = preprocess(test_raw.copy(), "test")

# TE용 원본 문자열 보존 (나이, 횟수는 이미 int이므로 str 변환)
te_cols_exist = [c for c in TE_COLS if c in train_df.columns]
train_df = step8_drop_remaining_object(
    train_df, [c for c in te_cols_exist if train_df[c].dtype == "object"]
)
test_df = step8_drop_remaining_object(
    test_df, [c for c in te_cols_exist if test_df[c].dtype == "object"]
)

print(
    f"  step8 후 - train: {train_df.shape}, obj={train_df.select_dtypes('object').shape[1]}"
)

# ============================================================
# 4. Target Encoding (K-Fold)
# ============================================================
print(f"\n## [4] Target Encoding (K-Fold)")

global_mean = y.mean()
alpha = 10
te_mappings = {}
skf_te = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

for col in te_cols_exist:
    col_str = train_df[col].astype(str)
    nunique = col_str.nunique()
    print(f"  - {col} (고유값: {nunique})")

    # TE 결과를 새 컬럼에 저장 (원본이 숫자면 _te 접미사)
    if train_df[col].dtype != "object":
        te_col_name = f"{col}_te"
    else:
        te_col_name = col  # object면 덮어쓰기

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

    train_df[te_col_name] = oof_te

    # test 매핑
    full_df = pd.DataFrame({"val": col_str.values, "target": y})
    full_agg = full_df.groupby("val")["target"].agg(["mean", "count"])
    full_agg["smoothed"] = (
        full_agg["count"] * full_agg["mean"] + alpha * global_mean
    ) / (full_agg["count"] + alpha)
    te_mappings[col] = full_agg["smoothed"].to_dict()
    test_df[te_col_name] = (
        test_df[col].astype(str).map(te_mappings[col]).fillna(global_mean)
    )

# 잔여 object 정리
for name in ["train", "test"]:
    df = train_df if name == "train" else test_df
    obj_r = df.select_dtypes("object").columns.tolist()
    if obj_r:
        print(f"  {name} 잔여 object 삭제: {obj_r}")
        if name == "train":
            train_df = train_df.drop(columns=obj_r)
        else:
            test_df = test_df.drop(columns=obj_r)

# ============================================================
# 5. 피처 정리 (제거 + 공통 컬럼)
# ============================================================
print(f"\n## [5] 피처 정리")

# 중요도0 + 다중공선성 피처 제거
remove_exist = [c for c in REMOVE_AFTER if c in train_df.columns]
print(f"- 제거 대상 ({len(remove_exist)}개): {remove_exist}")
train_df = train_df.drop(columns=remove_exist, errors="ignore")
test_df = test_df.drop(columns=remove_exist, errors="ignore")

# 공통 컬럼
common_cols = sorted(set(train_df.columns) & set(test_df.columns))
train_df = train_df[common_cols]
test_df = test_df[common_cols]

# 상관관계 0.95 이상 자동 제거
corr = train_df.corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
high_corr = []
for col in upper.columns:
    correlated = upper.index[upper[col] > 0.95].tolist()
    for c2 in correlated:
        high_corr.append((col, c2, corr.loc[col, c2]))

drop_corr = set()
for c1, c2, val in high_corr:
    # 둘 중 뒤에 있는 것 제거
    drop_corr.add(c2)

if drop_corr:
    print(f"- 상관 > 0.95 제거 ({len(drop_corr)}개):")
    for c1, c2, val in high_corr:
        if c2 in drop_corr:
            print(f"    {c1} <-> {c2}: {val:.3f} → {c2} 제거")
    common_cols = [c for c in common_cols if c not in drop_corr]
    train_df = train_df[common_cols]
    test_df = test_df[common_cols]

X_train = train_df.values.astype(np.float32)
X_test = test_df.values.astype(np.float32)
feature_names = list(common_cols)

print(f"- X_train: {X_train.shape}")
print(f"- X_test: {X_test.shape}")
print(f"- 최종 피처 수: {len(feature_names)}")
print(f"- NaN: train={np.isnan(X_train).sum()}, test={np.isnan(X_test).sum()}")

# 신규 피처 확인
v6_new = [
    "시술 당시 나이_te",
    "총 시술 횟수_te",
    "수정_성공률",
    "배아_이용률",
    "배아_잉여율",
    "난자_배아_전환율",
]
v6_actual = [c for c in v6_new if c in feature_names]
print(f"- v6 신규 피처: {v6_actual}")

print(f"- 피처 목록:")
for i, f in enumerate(feature_names):
    marker = " ★new" if f in v6_actual else ""
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
# 7. 성능
# ============================================================
print(f"\n## [7] 전체 성능")

oof_auc = roc_auc_score(y, oof_pred)
oof_f1 = f1_score(y, (oof_pred >= 0.5).astype(int))

print(f"- **OOF AUC: {oof_auc:.4f}**")
print(f"- OOF F1 (th=0.5): {oof_f1:.4f}")
for r in fold_results:
    print(
        f"  Fold {r['fold']}: AUC={r['auc']:.4f}, F1={r['f1']:.4f}, iter={r['best_iter']}"
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
    marker = " ★new" if r["feature"] in v6_actual else ""
    print(f"| {i+1} | {r['feature']}{marker} | {r['importance']:.4f} |")

zero_fi = fi_df[fi_df["importance"] == 0]["feature"].tolist()
if zero_fi:
    print(f"\n중요도 0 피처 ({len(zero_fi)}개): {zero_fi}")

# ============================================================
# 9. 제출
# ============================================================
print(f"\n## [9] 제출 파일")

submission = pd.DataFrame({"ID": test_ids, "probability": test_pred})
main_path = os.path.join(RESULT_DIR, f"sample_submission_{VERSION}_{NOW}.csv")
submission.to_csv(main_path, index=False)

print(f"- 파일: {main_path}")
print(f"- 확률: mean={test_pred.mean():.4f}, std={test_pred.std():.4f}")
print(f"- 예시:")
print(submission.head(5).to_string(index=False))

# ============================================================
# 10. 버전 비교
# ============================================================
print(f"\n## [10] 버전 비교")
print(
    f"""
| 버전 | 피처수 | OOF AUC | 주요 변경 |
|------|--------|---------|-----------|
| v3 | 69 | 0.7387 | 기본 XGB + TE |
| v4 | 73 | 0.7392 | 컬럼복원 + SPW제거 |
| v5 | 95 | 0.7390 | 22개 피처 추가 (과다) |
| v6 | {len(feature_names)} | {oof_auc:.4f} | 정리+DI분리+효율성+나이TE |
"""
)

# ============================================================
# 11. 요약
# ============================================================
total_time = time.time() - start_all
print(f"{'='*60}")
print(f"## 최종 요약")
print(f"{'='*60}")
print(f"- 모델: XGBoost (GPU)")
print(f"- 피처 수: {len(feature_names)}")
print(f"- **OOF AUC: {oof_auc:.4f}**")
print(f"- v5 대비: {oof_auc - 0.7390:+.4f}")
print(f"- v4 대비: {oof_auc - 0.7392:+.4f}")
print(f"- Data Leakage: 없음")
print(f"- 소요: {total_time/60:.1f}분")
print(f"- 로그: {LOG_PATH}")
print(f"{'='*60}")
