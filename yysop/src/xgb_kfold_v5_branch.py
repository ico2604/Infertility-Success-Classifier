# =========================================================
# XGBoost Stratified K-Fold Validation v5 — IVF/DI Branch Features (Deduplicated)
# ---------------------------------------------------------
# 목적
# 1) train/test 데이터를 불러온다.
# 2) 기본 수치 변환(횟수형/나이형)을 수행한다.
# 3) 기존 v2 의료 도메인 파생변수를 생성한다.
# 4) IVF / DI branch feature를 추가 생성한다. (중복 변수 제거 반영)
# 5) 범주형 컬럼을 안전하게 숫자형으로 인코딩한다.
# 6) Stratified K-Fold + OOF 방식으로 내부 검증을 수행한다.
# 7) Optuna best에 가까운 XGB 설정으로 재학습한다.
# 8) Feature Importance / OOF / Submission / Log를 저장한다.
#
# 실행 방법
#   python yysop/src/xgb_kfold_v5_branch.py
# =========================================================

import os
import re
import warnings
warnings.filterwarnings("ignore")

from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix
)

from xgboost import XGBClassifier


# =========================================================
# 0. 경로 / 기본 설정
# =========================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")

TARGET_COL = "임신 성공 여부"
ID_COL = "ID"

SEED = 42
N_FOLDS = 5

IMBALANCE_THRESHOLD = 0.30
SCALE_POS_WEIGHT_MODE = "off"   # "off" / "auto" / "manual"
MANUAL_SCALE_POS_WEIGHT = 1.0

EXPERIMENT_NAME = "optuna_branch_v5_dedup"


# =========================================================
# 1. XGBoost 파라미터
# - Optuna best 결과 반영
# =========================================================
XGB_TUNED_PARAMS = dict(
    n_estimators=4500,
    learning_rate=0.0308963469304573,
    max_depth=5,
    min_child_weight=1,
    gamma=0.14065784182145028,
    subsample=0.8707471118408655,
    colsample_bytree=0.832381623264227,
    colsample_bylevel=0.8531543518708955,
    reg_alpha=0.0531939083047116,
    reg_lambda=2.7719302220914796,
    early_stopping_rounds=50,
)


# =========================================================
# 2. 보조 함수
# =========================================================
def print_section(title: str):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def parse_korean_count(x):
    if pd.isna(x):
        return np.nan
    x = str(x).strip()
    if re.fullmatch(r"\d+", x):
        return float(x)
    m = re.search(r"(\d+)", x)
    if m:
        return float(m.group(1))
    return np.nan


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


def safe_div(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    return np.where((np.isnan(b)) | (b == 0), np.nan, a / b)


def decide_scale_pos_weight(y, mode="auto", threshold=0.30, manual_value=1.0):
    neg = (y == 0).sum()
    pos = (y == 1).sum()
    pos_ratio = pos / len(y)

    if mode == "off":
        spw = 1.0
    elif mode == "manual":
        spw = manual_value
    else:
        spw = neg / pos if pos_ratio < threshold else 1.0

    return spw, neg, pos, pos_ratio


# =========================================================
# 3. 데이터 로드
# =========================================================
print_section("3. 데이터 로드")

raw_train = pd.read_csv(TRAIN_PATH)
raw_test = pd.read_csv(TEST_PATH)

train = raw_train.copy()
test = raw_test.copy()
original_columns = raw_train.columns.tolist()

print("train shape:", train.shape)
print("test  shape:", test.shape)


# =========================================================
# 4. 기본 수치 변환
# =========================================================
print_section("4. 기본 수치 변환")

COUNT_COLS = [
    "총 시술 횟수", "클리닉 내 총 시술 횟수",
    "IVF 시술 횟수", "DI 시술 횟수",
    "총 임신 횟수", "IVF 임신 횟수", "DI 임신 횟수",
    "총 출산 횟수", "IVF 출산 횟수", "DI 출산 횟수",
]

AGE_COLS = [
    "시술 당시 나이",
    "난자 기증자 나이",
    "정자 기증자 나이",
]


def basic_numeric_convert(df):
    for col in COUNT_COLS:
        if col in df.columns:
            df[col + "_num"] = df[col].apply(parse_korean_count)
    for col in AGE_COLS:
        if col in df.columns:
            df[col + "_num"] = df[col].apply(age_to_numeric)
    return df


train = basic_numeric_convert(train)
test = basic_numeric_convert(test)

print("기본 수치 변환 완료")


# =========================================================
# 5. 기존 의료 도메인 파생변수
# =========================================================
print_section("5. 기존 의료 도메인 파생변수 생성")


def make_medical_features(df):
    def has(*cols):
        return all(c in df.columns for c in cols)

    # [A] 배아 처리 단계별 전환율
    if has("총 생성 배아 수", "수집된 신선 난자 수"):
        df["배아생성효율"] = safe_div(df["총 생성 배아 수"], df["수집된 신선 난자 수"])
    if has("미세주입에서 생성된 배아 수", "미세주입된 난자 수"):
        df["ICSI수정효율"] = safe_div(df["미세주입에서 생성된 배아 수"], df["미세주입된 난자 수"])
    if has("이식된 배아 수", "총 생성 배아 수"):
        df["배아이식비율"] = safe_div(df["이식된 배아 수"], df["총 생성 배아 수"])
    if has("저장된 배아 수", "총 생성 배아 수"):
        df["배아저장비율"] = safe_div(df["저장된 배아 수"], df["총 생성 배아 수"])
    if has("미세주입된 난자 수", "수집된 신선 난자 수"):
        df["난자활용률"] = safe_div(df["미세주입된 난자 수"], df["수집된 신선 난자 수"])
    if has("저장된 배아 수", "이식된 배아 수"):
        df["저장_대비_이식비율"] = safe_div(df["저장된 배아 수"], df["이식된 배아 수"])
    if has("이식된 배아 수", "수집된 신선 난자 수"):
        df["난자대비이식배아수"] = safe_div(df["이식된 배아 수"], df["수집된 신선 난자 수"])
    if "이식된 배아 수" in df.columns:
        df["이식배아수_구간"] = pd.cut(
            df["이식된 배아 수"],
            bins=[-1, 0, 1, 2, 100],
            labels=[0, 1, 2, 3]
        ).astype(float)

    # [B] 과거 시술 이력
    if has("총 임신 횟수_num", "총 시술 횟수_num"):
        df["전체임신률"] = safe_div(df["총 임신 횟수_num"], df["총 시술 횟수_num"])
    if has("IVF 임신 횟수_num", "IVF 시술 횟수_num"):
        df["IVF임신률"] = safe_div(df["IVF 임신 횟수_num"], df["IVF 시술 횟수_num"])
    if has("DI 임신 횟수_num", "DI 시술 횟수_num"):
        df["DI임신률"] = safe_div(df["DI 임신 횟수_num"], df["DI 시술 횟수_num"])
    if has("총 출산 횟수_num", "총 임신 횟수_num"):
        df["임신유지율"] = safe_div(df["총 출산 횟수_num"], df["총 임신 횟수_num"])
    if has("IVF 출산 횟수_num", "IVF 임신 횟수_num"):
        df["IVF임신유지율"] = safe_div(df["IVF 출산 횟수_num"], df["IVF 임신 횟수_num"])
    if has("DI 출산 횟수_num", "DI 임신 횟수_num"):
        df["DI임신유지율"] = safe_div(df["DI 출산 횟수_num"], df["DI 임신 횟수_num"])
    if has("총 시술 횟수_num", "총 임신 횟수_num"):
        df["총실패횟수"] = (
            df["총 시술 횟수_num"].fillna(0) - df["총 임신 횟수_num"].fillna(0)
        ).clip(lower=0)
    if has("IVF 시술 횟수_num", "IVF 임신 횟수_num"):
        df["IVF실패횟수"] = (
            df["IVF 시술 횟수_num"].fillna(0) - df["IVF 임신 횟수_num"].fillna(0)
        ).clip(lower=0)
    if "IVF실패횟수" in df.columns:
        df["반복IVF실패_여부"] = (df["IVF실패횟수"] >= 3).astype(float)
    if has("클리닉 내 총 시술 횟수_num", "총 시술 횟수_num"):
        df["클리닉집중도"] = safe_div(df["클리닉 내 총 시술 횟수_num"], df["총 시술 횟수_num"])
    if has("IVF 임신 횟수_num", "DI 임신 횟수_num"):
        df["IVF_DI_임신합"] = (
            df["IVF 임신 횟수_num"].fillna(0) + df["DI 임신 횟수_num"].fillna(0)
        )
    if has("IVF 출산 횟수_num", "DI 출산 횟수_num"):
        df["IVF_DI_출산합"] = (
            df["IVF 출산 횟수_num"].fillna(0) + df["DI 출산 횟수_num"].fillna(0)
        )
    if has("IVF 시술 횟수_num", "총 시술 횟수_num"):
        df["IVF시술비율"] = safe_div(df["IVF 시술 횟수_num"], df["총 시술 횟수_num"])
    if "총 임신 횟수_num" in df.columns:
        df["임신경험있음"] = (df["총 임신 횟수_num"] > 0).astype(float)
    if "총 출산 횟수_num" in df.columns:
        df["출산경험있음"] = (df["총 출산 횟수_num"] > 0).astype(float)

    # [C] 나이 기반 위험도
    if "시술 당시 나이_num" in df.columns:
        age = df["시술 당시 나이_num"]
        df["나이_제곱"] = age ** 2
        df["나이_임상구간"] = pd.cut(
            age, bins=[0, 35, 40, 45, 100], labels=[0, 1, 2, 3], right=False
        ).astype(float)
        df["고령_여부"] = (age >= 35).astype(float)
        df["초고령_여부"] = (age >= 40).astype(float)
        df["극고령_여부"] = (age >= 42).astype(float)

    # [D] 나이 × 이력 상호작용
    if has("시술 당시 나이_num", "총 시술 횟수_num"):
        df["나이X총시술"] = df["시술 당시 나이_num"] * df["총 시술 횟수_num"]
    if has("시술 당시 나이_num", "IVF실패횟수"):
        df["나이XIVF실패"] = df["시술 당시 나이_num"] * df["IVF실패횟수"]
    if has("시술 당시 나이_num", "IVF임신률"):
        df["나이XIVF임신률"] = df["시술 당시 나이_num"] * df["IVF임신률"].fillna(0)
    if has("초고령_여부", "반복IVF실패_여부"):
        df["초고령X반복실패"] = df["초고령_여부"] * df["반복IVF실패_여부"]

    # [E] 복합 위험도 점수
    risk_components = []
    if "고령_여부" in df.columns:
        risk_components.append(df["고령_여부"].fillna(0))
    if "초고령_여부" in df.columns:
        risk_components.append(df["초고령_여부"].fillna(0))
    if "반복IVF실패_여부" in df.columns:
        risk_components.append(df["반복IVF실패_여부"].fillna(0))
    if "임신경험있음" in df.columns:
        risk_components.append(1 - df["임신경험있음"].fillna(1))
    if len(risk_components) >= 2:
        df["복합위험도점수"] = sum(risk_components)

    return df


# =========================================================
# 6. IVF / DI Branch Feature
# - 중복 제거 반영:
#   제거: IVF실패횟수_v5, IVF성공률_v5, IVF출산유지율_v5,
#        DI성공률_v5, DI출산유지율_v5
#   유지: DI실패횟수_v5 (기존 동일 변수 없음)
# =========================================================
print_section("6. IVF / DI Branch Feature 생성")


def add_ivf_di_branch_features(df):
    def has(*cols):
        return all(c in df.columns for c in cols)

    # 1) IVF / DI 경험 여부
    if "IVF 시술 횟수_num" in df.columns:
        df["IVF여부_v5"] = (df["IVF 시술 횟수_num"].fillna(0) > 0).astype(float)

    if "DI 시술 횟수_num" in df.columns:
        df["DI여부_v5"] = (df["DI 시술 횟수_num"].fillna(0) > 0).astype(float)

    # 2) IVF dominant / DI dominant
    if has("IVF 시술 횟수_num", "DI 시술 횟수_num"):
        ivf_cnt = df["IVF 시술 횟수_num"].fillna(0)
        di_cnt = df["DI 시술 횟수_num"].fillna(0)

        df["IVFdominant_v5"] = (ivf_cnt > di_cnt).astype(float)
        df["DIdominant_v5"] = (di_cnt > ivf_cnt).astype(float)
        df["IVF_DI_균형시술_v5"] = (ivf_cnt == di_cnt).astype(float)

    # 3) IVF / DI 비중
    if has("IVF 시술 횟수_num", "총 시술 횟수_num"):
        df["IVF비중_v5"] = safe_div(df["IVF 시술 횟수_num"], df["총 시술 횟수_num"])

    if has("DI 시술 횟수_num", "총 시술 횟수_num"):
        df["DI비중_v5"] = safe_div(df["DI 시술 횟수_num"], df["총 시술 횟수_num"])

    # 4) DI 실패횟수 (기존 동일 변수 없음)
    if has("DI 시술 횟수_num", "DI 임신 횟수_num"):
        df["DI실패횟수_v5"] = (
            df["DI 시술 횟수_num"].fillna(0) - df["DI 임신 횟수_num"].fillna(0)
        ).clip(lower=0)

    # 5) 나이와 IVF / DI 상호작용
    if has("시술 당시 나이_num", "IVF여부_v5"):
        df["나이XIVF여부_v5"] = (
            df["시술 당시 나이_num"].fillna(0) * df["IVF여부_v5"].fillna(0)
        )

    if has("시술 당시 나이_num", "DI여부_v5"):
        df["나이XDI여부_v5"] = (
            df["시술 당시 나이_num"].fillna(0) * df["DI여부_v5"].fillna(0)
        )

    if has("시술 당시 나이_num", "IVF비중_v5"):
        df["나이XIVF비중_v5"] = (
            df["시술 당시 나이_num"].fillna(0) * pd.Series(df["IVF비중_v5"]).fillna(0)
        )

    if has("시술 당시 나이_num", "DI비중_v5"):
        df["나이XDI비중_v5"] = (
            df["시술 당시 나이_num"].fillna(0) * pd.Series(df["DI비중_v5"]).fillna(0)
        )

    # 6) IVF / DI 경험 조합
    if has("IVF여부_v5", "DI여부_v5"):
        df["IVF만경험_v5"] = ((df["IVF여부_v5"] == 1) & (df["DI여부_v5"] == 0)).astype(float)
        df["DI만경험_v5"] = ((df["IVF여부_v5"] == 0) & (df["DI여부_v5"] == 1)).astype(float)
        df["IVF_DI모두경험_v5"] = ((df["IVF여부_v5"] == 1) & (df["DI여부_v5"] == 1)).astype(float)

    # 7) 문자열 플래그
    # 범주형 인코딩 전에 실행되어야 함
    if "시술 유형" in df.columns:
        s = df["시술 유형"].fillna("").astype(str).str.upper()
        df["시술유형_IVF플래그_v5"] = s.str.contains("IVF", na=False).astype(float)
        df["시술유형_DI플래그_v5"] = s.str.contains("DI", na=False).astype(float)

    if "특정 시술 유형" in df.columns:
        s2 = df["특정 시술 유형"].fillna("").astype(str).str.upper()
        df["특정시술_ICSI플래그_v5"] = s2.str.contains("ICSI", na=False).astype(float)
        df["특정시술_FET플래그_v5"] = s2.str.contains("FET", na=False).astype(float)
        df["특정시술_IUI플래그_v5"] = s2.str.contains("IUI", na=False).astype(float)

    return df


train = make_medical_features(train)
test = make_medical_features(test)

train = add_ivf_di_branch_features(train)
test = add_ivf_di_branch_features(test)

new_feat_cols = [c for c in train.columns if c not in original_columns]
print(f"파생변수 생성 완료: {len(new_feat_cols)}개")
print(new_feat_cols)


# =========================================================
# 7. 범주형 컬럼 인코딩
# =========================================================
print_section("7. 범주형 컬럼 인코딩")

exclude_encode = [TARGET_COL, ID_COL]

cat_cols = []
for c in train.columns:
    if c in exclude_encode:
        continue
    if not (
        pd.api.types.is_numeric_dtype(train[c]) or
        pd.api.types.is_bool_dtype(train[c]) or
        isinstance(train[c].dtype, pd.CategoricalDtype)
    ):
        cat_cols.append(c)

print(f"범주형 인코딩 대상 컬럼 수: {len(cat_cols)}")
print(cat_cols)

for col in cat_cols:
    combined = pd.concat([
        train[col].fillna("MISSING").astype(str),
        test[col].fillna("MISSING").astype(str)
    ], axis=0)

    codes, _ = pd.factorize(combined)
    train[col] = codes[:len(train)]
    test[col] = codes[len(train):]

print(f"범주형 인코딩 완료: {len(cat_cols)}개 컬럼")

remaining_non_numeric = [
    c for c in train.columns
    if c not in [TARGET_COL, ID_COL]
    and not (
        pd.api.types.is_numeric_dtype(train[c]) or
        pd.api.types.is_bool_dtype(train[c]) or
        isinstance(train[c].dtype, pd.CategoricalDtype)
    )
]
print("\n[인코딩 후 남아있는 비수치형 컬럼]")
print(remaining_non_numeric)


# =========================================================
# 8. 원본 문자열 컬럼 제거
# =========================================================
print_section("8. 원본 문자열 컬럼 제거 및 결측 확인")

drop_cols = [c for c in COUNT_COLS + AGE_COLS if c in train.columns]
train = train.drop(columns=drop_cols, errors="ignore")
test = test.drop(columns=drop_cols, errors="ignore")

print(f"원본 문자열 컬럼 {len(drop_cols)}개 제거 완료")
missing_summary = train.isnull().sum()
print(f"결측 있는 컬럼 수: {(missing_summary > 0).sum()}")


# =========================================================
# 9. 입력 데이터 구성
# =========================================================
print_section("9. 입력 데이터 구성")

X = train.drop(columns=[TARGET_COL, ID_COL], errors="ignore")
y = train[TARGET_COL]
X_test = test.drop(columns=[ID_COL], errors="ignore")

common_cols = [c for c in X.columns if c in X_test.columns]
X = X[common_cols]
X_test = X_test[common_cols]

print(f"X shape      : {X.shape}")
print(f"y shape      : {y.shape}")
print(f"X_test shape : {X_test.shape}")

non_numeric_in_X = [
    c for c in X.columns
    if not (
        pd.api.types.is_numeric_dtype(X[c]) or
        pd.api.types.is_bool_dtype(X[c]) or
        isinstance(X[c].dtype, pd.CategoricalDtype)
    )
]
print("\n[X 안의 비수치형 컬럼]")
print(non_numeric_in_X)

scale_pos_weight, neg, pos, pos_ratio = decide_scale_pos_weight(
    y,
    mode=SCALE_POS_WEIGHT_MODE,
    threshold=IMBALANCE_THRESHOLD,
    manual_value=MANUAL_SCALE_POS_WEIGHT
)

print(f"\n음성(0): {neg}개  /  양성(1): {pos}개  /  양성 비율: {pos_ratio:.2%}")
print(f"scale_pos_weight mode: {SCALE_POS_WEIGHT_MODE}")
print(f"scale_pos_weight     : {scale_pos_weight:.4f}")


# =========================================================
# 10. XGBoost 파라미터 정의
# =========================================================
print_section("10. XGBoost 모델 파라미터 정의")

XGB_PARAMS = dict(
    **XGB_TUNED_PARAMS,
    scale_pos_weight=scale_pos_weight,
    objective="binary:logistic",
    eval_metric="auc",
    tree_method="hist",
    random_state=SEED,
    n_jobs=-1,
)

for k, v in XGB_PARAMS.items():
    print(f"  {k:30s}: {v}")


# =========================================================
# 11. Stratified K-Fold 학습
# =========================================================
print_section("11. Stratified K-Fold 학습")

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))
fold_aucs = []
best_iters = []
models = []

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
    print(f"\n{'─' * 60}")
    print(f"  Fold {fold} / {N_FOLDS}  [{EXPERIMENT_NAME}]")
    print(f"{'─' * 60}")

    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

    model = XGBClassifier(**XGB_PARAMS)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        verbose=200
    )

    va_pred = model.predict_proba(X_va)[:, 1]
    te_pred = model.predict_proba(X_test)[:, 1]

    oof_preds[va_idx] = va_pred
    test_preds += te_pred / N_FOLDS

    fold_auc = roc_auc_score(y_va, va_pred)
    best_iter = getattr(model, "best_iteration", None)

    fold_aucs.append(fold_auc)
    best_iters.append(best_iter)
    models.append(model)

    print(f"  Fold {fold} AUC       : {fold_auc:.6f}")
    print(f"  Best Iteration        : {best_iter}")


# =========================================================
# 12. 성능 요약
# =========================================================
print_section("12. 성능 요약")

oof_auc = roc_auc_score(y, oof_preds)
mean_auc = np.mean(fold_aucs)
std_auc = np.std(fold_aucs)
ap_score = average_precision_score(y, oof_preds)

print(f"\n{'폴드':>6}  {'AUC':>10}  {'Best Iter':>10}")
print("-" * 34)
for i, (auc, bi) in enumerate(zip(fold_aucs, best_iters)):
    print(f"Fold {i + 1:>2}  {auc:>10.6f}  {str(bi):>10}")
print("-" * 34)
print(f"{'Mean':>6}  {mean_auc:>10.6f}")
print(f"{'Std':>6}  {std_auc:>10.6f}")
print(f"\n최종 OOF AUC : {oof_auc:.6f}")
print(f"PR-AUC       : {ap_score:.6f}")

if std_auc > 0.01:
    print("⚠ 폴드 간 AUC 편차가 큽니다.")
else:
    print("✓ 폴드 간 AUC 편차가 안정적입니다.")


# =========================================================
# 13. 결과 로그 저장
# =========================================================
print_section("13. 결과 로그 저장")

log_path = os.path.join(OUTPUT_DIR, "xgb_v5_branch_results_log.csv")

log_row = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "experiment_name": EXPERIMENT_NAME,
    "scale_pos_weight_mode": SCALE_POS_WEIGHT_MODE,
    "scale_pos_weight": round(float(scale_pos_weight), 6),
    "n_features": X.shape[1],
    "n_new_features": len(new_feat_cols),
    "oof_auc": round(float(oof_auc), 6),
    "mean_auc": round(float(mean_auc), 6),
    "std_auc": round(float(std_auc), 6),
    "pr_auc": round(float(ap_score), 6),
    "fold1_auc": round(float(fold_aucs[0]), 6),
    "fold2_auc": round(float(fold_aucs[1]), 6),
    "fold3_auc": round(float(fold_aucs[2]), 6),
    "fold4_auc": round(float(fold_aucs[3]), 6),
    "fold5_auc": round(float(fold_aucs[4]), 6),
    "mean_best_iter": round(np.mean([b for b in best_iters if b is not None]), 1) if any(b is not None for b in best_iters) else None,
    **{f"param_{k}": v for k, v in XGB_TUNED_PARAMS.items()},
}

log_df = pd.DataFrame([log_row])
if os.path.exists(log_path):
    old = pd.read_csv(log_path)
    log_df = pd.concat([old, log_df], ignore_index=True)

log_df.to_csv(log_path, index=False, encoding="utf-8-sig")
print(f"[저장] {log_path}")


# =========================================================
# 14. Feature Importance
# =========================================================
print_section("14. Feature Importance")

importance_matrix = np.array([m.feature_importances_ for m in models])
mean_importance = importance_matrix.mean(axis=0)
std_importance = importance_matrix.std(axis=0)

importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance": mean_importance,
    "std": std_importance
}).sort_values("importance", ascending=False).reset_index(drop=True)

print(importance_df.head(40).to_string(index=False))

imp_path = os.path.join(OUTPUT_DIR, f"xgb_v5_{EXPERIMENT_NAME}_feature_importance.csv")
importance_df.to_csv(imp_path, index=False, encoding="utf-8-sig")
print(f"[저장] {imp_path}")


# =========================================================
# 15. ROC / PR Curve
# =========================================================
print_section("15. ROC / PR Curve")

fpr, tpr, _ = roc_curve(y, oof_preds)
precision, recall, _ = precision_recall_curve(y, oof_preds)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC={oof_auc:.4f})")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"OOF ROC Curve [{EXPERIMENT_NAME}]")
plt.legend()
plt.tight_layout()

roc_path = os.path.join(OUTPUT_DIR, f"xgb_v5_{EXPERIMENT_NAME}_roc_curve.png")
plt.savefig(roc_path, bbox_inches="tight")
plt.close()
print(f"[저장] {roc_path}")

plt.figure(figsize=(6, 5))
plt.plot(recall, precision, label=f"PR Curve (AP={ap_score:.4f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title(f"OOF PR Curve [{EXPERIMENT_NAME}]")
plt.legend()
plt.tight_layout()

pr_path = os.path.join(OUTPUT_DIR, f"xgb_v5_{EXPERIMENT_NAME}_pr_curve.png")
plt.savefig(pr_path, bbox_inches="tight")
plt.close()
print(f"[저장] {pr_path}")


# =========================================================
# 16. OOF 저장
# =========================================================
print_section("16. OOF 예측값 저장")

oof_df = pd.DataFrame({
    ID_COL: train[ID_COL].values,
    TARGET_COL: y.values,
    "oof_pred_prob": oof_preds,
    "oof_pred_label": (oof_preds >= 0.5).astype(int),
})
oof_df["correct"] = (oof_df[TARGET_COL] == oof_df["oof_pred_label"]).astype(int)

oof_path = os.path.join(OUTPUT_DIR, f"xgb_v5_{EXPERIMENT_NAME}_oof_predictions.csv")
oof_df.to_csv(oof_path, index=False, encoding="utf-8-sig")
print(f"[저장] {oof_path}")


# =========================================================
# 17. Threshold 분석
# =========================================================
print_section("17. Threshold 분석")

threshold_list = [0.30, 0.40, 0.50, 0.60, 0.70]
threshold_rows = []

for th in threshold_list:
    pred_label = (oof_preds >= th).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred_label).ravel()

    acc = (tp + tn) / (tp + tn + fp + fn)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

    threshold_rows.append({
        "threshold": th,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "specificity": spec,
        "f1": f1
    })

threshold_df = pd.DataFrame(threshold_rows)
threshold_path = os.path.join(OUTPUT_DIR, f"xgb_v5_{EXPERIMENT_NAME}_threshold_analysis.csv")
threshold_df.to_csv(threshold_path, index=False, encoding="utf-8-sig")
print(f"[저장] {threshold_path}")


# =========================================================
# 18. 제출 파일 생성
# =========================================================
print_section("18. 제출 파일 생성")

submission = pd.DataFrame({
    ID_COL: test[ID_COL].values,
    TARGET_COL: test_preds
})

sub_path = os.path.join(OUTPUT_DIR, f"xgb_v5_{EXPERIMENT_NAME}_submission.csv")
submission.to_csv(sub_path, index=False, encoding="utf-8-sig")
print(f"[저장] {sub_path}")
print(submission.head(10))


# =========================================================
# 19. 종료 요약
# =========================================================
print_section("19. 완료")

print(f"실험명     : {EXPERIMENT_NAME}")
print(f"OOF AUC    : {oof_auc:.6f}")
print(f"Mean AUC   : {mean_auc:.6f}  ±  {std_auc:.6f}")
print(f"PR-AUC     : {ap_score:.6f}")

print("\n저장 파일:")
print(f"  1. {log_path}")
print(f"  2. {imp_path}")
print(f"  3. {roc_path}")
print(f"  4. {pr_path}")
print(f"  5. {oof_path}")
print(f"  6. {threshold_path}")
print(f"  7. {sub_path}")