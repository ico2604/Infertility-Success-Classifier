# =========================================================
# XGBoost Stratified K-Fold Validation v1
# ---------------------------------------------------------
# 목적
# 1) train/test 데이터를 불러온다.
# 2) 기본 수치 변환(횟수형/나이형)을 수행한다.
# 3) IVF 의료 도메인 기반 파생변수를 생성한다.
# 4) Stratified K-Fold + OOF 방식으로 안정적인 내부 검증을 수행한다.
# 5) OOF AUC, 폴드별 AUC, feature importance를 확인한다.
# 6) ROC Curve / PR Curve / Threshold 분석을 수행한다.
# 7) test 예측 및 제출 파일을 생성한다.
#
# 주요 특징
# 1) Stratified K-Fold (5-fold OOF) → 분산이 낮은 안정적 검증
# 2) Early Stopping per fold        → 폴드별 최적 트리 수 자동 결정
# 3) 의료 도메인 파생변수 확장      → IVF 임상 흐름 기반
# 4) raw_train / raw_test 원본 보존 → 중복 파일 I/O 제거
# 5) original_columns 저장          → 신규 파생변수 추적 가능
# 6) scale_pos_weight 자동 결정     → 클래스 비율 확인 후 조건부 적용
# 7) importance reset_index         → CSV 저장 시 보기 좋은 인덱스 정리
# 8) OOF 저장                       → 스태킹/앙상블 재활용 가능
#
# 실행 방법
#   python yysop/src/xgb_kfold_v1.py
#
# 참고
# - xgboost 3.2.0 기준:
#   early_stopping_rounds는 XGBClassifier 생성자에 넣는 방식으로 작성
# - 범주형 인코딩:
#   숫자형 / bool / category가 아닌 모든 컬럼을 factorize 처리
# =========================================================

import os
import re
import warnings
warnings.filterwarnings("ignore")

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

# 양성 클래스 비율이 이 값보다 작으면 불균형으로 보고 scale_pos_weight 적용
IMBALANCE_THRESHOLD = 0.30

# scale_pos_weight 실험 모드
# "auto"   : 양성 비율 보고 자동 결정
# "off"    : 무조건 1.0 사용
# "manual" : 아래 MANUAL_SCALE_POS_WEIGHT 사용
SCALE_POS_WEIGHT_MODE = "auto"
MANUAL_SCALE_POS_WEIGHT = 1.0  # mode="manual"일 때만 사용


# =========================================================
# 1. 보조 함수
# =========================================================
def print_section(title: str):
    """콘솔 출력 시 각 단계를 보기 쉽게 구분선으로 표시"""
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def parse_korean_count(x):
    """
    '0회', '1회', '3회 이상' 같은 문자열을 숫자로 변환한다.
    - '3회 이상' -> 3.0 (하한값 근사)
    - NaN        -> NaN
    """
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
    """
    나이 구간 문자열을 숫자(구간 중간값)로 변환한다.
    - '만18-34세' -> 26.0
    - '만35-39세' -> 37.0
    - '만40-44세' -> 42.0
    - '알 수 없음' -> NaN
    """
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
    """
    분모가 0이거나 NaN이면 NaN을 반환하는 안전한 나눗셈 함수.
    XGBoost는 NaN을 native하게 처리하므로 -999 대신 NaN 유지.
    """
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    return np.where((np.isnan(b)) | (b == 0), np.nan, a / b)


def decide_scale_pos_weight(y, mode="auto", threshold=0.30, manual_value=1.0):
    """
    클래스 비율에 따라 scale_pos_weight를 결정한다.

    mode:
    - auto   : 양성 비율이 threshold보다 낮으면 neg/pos, 아니면 1.0
    - off    : 무조건 1.0
    - manual : manual_value 사용

    반환: (scale_pos_weight, 음성 수, 양성 수, 양성 비율)
    """
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
# 2. 데이터 로드
# =========================================================
print_section("2. 데이터 로드")

# raw 원본은 절대 수정하지 않음
raw_train = pd.read_csv(TRAIN_PATH)
raw_test = pd.read_csv(TEST_PATH)

# 실제 전처리/파생변수 생성은 복사본에 수행
train = raw_train.copy()
test = raw_test.copy()

# 원본 컬럼명 저장 → 신규 파생변수 추적에 사용
original_columns = raw_train.columns.tolist()

print("train shape:", train.shape)
print("test  shape:", test.shape)
print(f"\n[원본 컬럼 수] {len(original_columns)}")


# =========================================================
# 3. 기본 수치 변환 (횟수형 / 나이형)
# =========================================================
print_section("3. 기본 수치 변환")

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
    """COUNT_COLS, AGE_COLS를 각각 _num 파생 컬럼으로 변환한다."""
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
# 4. 의료 도메인 파생변수 생성
# =========================================================
print_section("4. 의료 도메인 파생변수 생성")


def make_medical_features(df):
    """
    train/test 모두에 동일하게 적용할 IVF 의료 도메인 파생변수 생성 함수.

    [A] 배아 처리 단계별 전환율
    [B] 과거 시술 이력
    [C] 나이 기반 위험도
    [D] 나이 × 이력 상호작용
    [E] 복합 위험도 점수
    """

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
        )
        df["이식배아수_구간"] = df["이식배아수_구간"].astype(float)

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
        df["클리닉집중도"] = safe_div(
            df["클리닉 내 총 시술 횟수_num"], df["총 시술 횟수_num"]
        )

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
            age,
            bins=[0, 35, 40, 45, 100],
            labels=[0, 1, 2, 3],
            right=False
        )
        df["나이_임상구간"] = df["나이_임상구간"].astype(float)

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


train = make_medical_features(train)
test = make_medical_features(test)

new_feat_cols = [c for c in train.columns if c not in original_columns]
print(f"파생변수 생성 완료: {len(new_feat_cols)}개")
print(new_feat_cols)


# =========================================================
# 5. 범주형 컬럼 인코딩 (train + test 동시 factorize)
# =========================================================
print_section("5. 범주형 컬럼 인코딩")

# 숫자형 / bool / category가 아닌 컬럼은 전부 범주형으로 보고 인코딩
exclude_encode = [TARGET_COL, ID_COL]

cat_cols = []
for c in train.columns:
    if c in exclude_encode:
        continue

    if not (
        pd.api.types.is_numeric_dtype(train[c]) or
        pd.api.types.is_bool_dtype(train[c]) or
        pd.api.types.is_categorical_dtype(train[c])
    ):
        cat_cols.append(c)

print(f"범주형 인코딩 대상 컬럼 수: {len(cat_cols)}")
print(cat_cols)

for col in cat_cols:
    combined = pd.concat([
        train[col].astype(str).fillna("MISSING"),
        test[col].astype(str).fillna("MISSING")
    ], axis=0)

    codes, _ = pd.factorize(combined)
    train[col] = codes[:len(train)]
    test[col] = codes[len(train):]

print(f"범주형 인코딩 완료: {len(cat_cols)}개 컬럼")

# 인코딩 후 비수치형 컬럼이 남아있는지 최종 점검
remaining_non_numeric = [
    c for c in train.columns
    if c not in [TARGET_COL, ID_COL]
    and not (
        pd.api.types.is_numeric_dtype(train[c]) or
        pd.api.types.is_bool_dtype(train[c]) or
        pd.api.types.is_categorical_dtype(train[c])
    )
]

print("\n[인코딩 후 남아있는 비수치형 컬럼]")
print(remaining_non_numeric)


# =========================================================
# 6. 원본 문자열 컬럼 제거
# =========================================================
print_section("6. 원본 문자열 컬럼 제거 및 결측 확인")

# 이미 _num으로 변환한 원본 문자열 컬럼 제거
drop_cols = [c for c in COUNT_COLS + AGE_COLS if c in train.columns]
train = train.drop(columns=drop_cols, errors="ignore")
test = test.drop(columns=drop_cols, errors="ignore")

print(f"원본 문자열 컬럼 {len(drop_cols)}개 제거 완료")

missing_summary = train.isnull().sum()
print(f"\n결측 있는 컬럼 수: {(missing_summary > 0).sum()}")
print(missing_summary[missing_summary > 0].sort_values(ascending=False).head(15))


# =========================================================
# 7. 입력 데이터 구성
# =========================================================
print_section("7. 입력 데이터 구성")

X = train.drop(columns=[TARGET_COL, ID_COL], errors="ignore")
y = train[TARGET_COL]
X_test = test.drop(columns=[ID_COL], errors="ignore")

# train/test 공통 컬럼만 사용
common_cols = [c for c in X.columns if c in X_test.columns]
X = X[common_cols]
X_test = X_test[common_cols]

print(f"X shape      : {X.shape}")
print(f"y shape      : {y.shape}")
print(f"X_test shape : {X_test.shape}")

# 최종 입력에서 비수치형 컬럼이 없는지 한 번 더 점검
non_numeric_in_X = [
    c for c in X.columns
    if not (
        pd.api.types.is_numeric_dtype(X[c]) or
        pd.api.types.is_bool_dtype(X[c]) or
        pd.api.types.is_categorical_dtype(X[c])
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

print("\n[클래스 비율]")
print(f"음성(0): {neg}개")
print(f"양성(1): {pos}개")
print(f"양성 비율: {pos_ratio:.2%}")
print(f"scale_pos_weight mode: {SCALE_POS_WEIGHT_MODE}")
print(f"scale_pos_weight 값  : {scale_pos_weight:.6f}")


# =========================================================
# 8. XGBoost 모델 파라미터 정의
# =========================================================
print_section("8. XGBoost 모델 파라미터 정의")

# xgboost 3.2.0 기준: early_stopping_rounds를 생성자에 전달
XGB_PARAMS = dict(
    # 트리 구조
    n_estimators=3000,        # early stopping이 실제 트리 수 결정
    learning_rate=0.02,       # 낮은 lr + early stopping 조합 권장
    max_depth=5,              # 깊이 제한 → 과적합 방지
    min_child_weight=5,       # 리프 최소 샘플 → 과적합 방지
    gamma=0.1,                # 분할 최소 손실 감소량

    # 샘플링
    subsample=0.8,
    colsample_bytree=0.7,
    colsample_bylevel=0.7,

    # 정규화
    reg_alpha=0.1,            # L1
    reg_lambda=1.5,           # L2

    # 클래스 불균형 대응
    scale_pos_weight=scale_pos_weight,

    # 기타
    objective="binary:logistic",
    eval_metric="auc",
    tree_method="hist",
    random_state=SEED,
    n_jobs=-1,

    # early stopping: valid AUC가 50라운드 동안 개선 없으면 종료
    early_stopping_rounds=50
)

print("파라미터 설정 완료")
for k, v in XGB_PARAMS.items():
    print(f"  {k:30s}: {v}")


# =========================================================
# 9. Stratified K-Fold 학습 + OOF 예측
# =========================================================
print_section("9. Stratified K-Fold 학습")

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

oof_preds = np.zeros(len(X))        # 전체 train OOF 예측값
test_preds = np.zeros(len(X_test))  # 폴드별 test 예측 누적 평균

fold_aucs = []
best_iters = []
models = []

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
    print(f"\n{'─' * 60}")
    print(f"  Fold {fold + 1} / {N_FOLDS}")
    print(f"{'─' * 60}")

    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

    # 폴드마다 새 인스턴스 생성 (폴드 간 독립성 보장)
    model = XGBClassifier(**XGB_PARAMS)

    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        verbose=200
    )

    # OOF 예측: 이 폴드의 검증셋에 예측값 저장
    oof_preds[va_idx] = model.predict_proba(X_va)[:, 1]

    # test 예측: 폴드별 예측을 누적 후 N_FOLDS로 나눠 평균
    test_preds += model.predict_proba(X_test)[:, 1] / N_FOLDS

    fold_auc = roc_auc_score(y_va, oof_preds[va_idx])
    best_iter = getattr(model, "best_iteration", None)

    fold_aucs.append(fold_auc)
    best_iters.append(best_iter)
    models.append(model)

    print(f"  Fold {fold + 1} AUC     : {fold_auc:.6f}")
    print(f"  Best Iteration : {best_iter}")


# =========================================================
# 10. 최종 성능 요약
# =========================================================
print_section("10. 성능 요약")

# OOF AUC: 전체 train 데이터에 대한 out-of-fold 예측 기반 AUC
oof_auc = roc_auc_score(y, oof_preds)

print(f"\n{'폴드':>6}  {'AUC':>10}  {'Best Iter':>10}")
print("-" * 34)
for i, (auc, bi) in enumerate(zip(fold_aucs, best_iters)):
    print(f"Fold {i + 1:>2}  {auc:>10.6f}  {str(bi):>10}")
print("-" * 34)
print(f"{'Mean':>6}  {np.mean(fold_aucs):>10.6f}")
print(f"{'Std':>6}  {np.std(fold_aucs):>10.6f}")
print(f"\n최종 OOF AUC : {oof_auc:.6f}")

# 폴드 간 AUC 편차 진단
if np.std(fold_aucs) > 0.01:
    print("⚠ 폴드 간 AUC 편차가 큽니다. 데이터 분포 또는 파생변수 안정성을 확인하세요.")
else:
    print("✓ 폴드 간 AUC 편차가 안정적입니다.")


# =========================================================
# 11. Feature Importance (폴드 평균)
# =========================================================
print_section("11. Feature Importance (폴드 평균)")

# 단일 폴드 importance는 분할에 따라 흔들릴 수 있으므로 폴드 평균 사용
importance_matrix = np.array([m.feature_importances_ for m in models])
mean_importance = importance_matrix.mean(axis=0)
std_importance = importance_matrix.std(axis=0)

importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance": mean_importance,
    "std": std_importance       # std가 크면 폴드별로 중요도가 불안정한 변수
}).sort_values("importance", ascending=False).reset_index(drop=True)

print(importance_df.head(30).to_string(index=False))

# 중요도 하위 10%: 다음 실험에서 제거 후보
low_imp = importance_df[
    importance_df["importance"] < importance_df["importance"].quantile(0.1)
]
print(f"\n중요도 하위 10% ({len(low_imp)}개) — 제거 후보:")
print(low_imp["feature"].tolist())

imp_path = os.path.join(OUTPUT_DIR, "xgb_v1_feature_importance.csv")
importance_df.to_csv(imp_path, index=False, encoding="utf-8-sig")
print(f"\n[저장] {imp_path}")


# =========================================================
# 12. ROC Curve / PR Curve / Threshold 분석
# =========================================================
print_section("12. ROC Curve / PR Curve / Threshold 분석")

# ROC curve
fpr, tpr, roc_thresholds = roc_curve(y, oof_preds)

# PR curve
precision, recall, pr_thresholds = precision_recall_curve(y, oof_preds)

# PR-AUC (Average Precision)
ap_score = average_precision_score(y, oof_preds)

print(f"OOF ROC-AUC          : {oof_auc:.6f}")
print(f"OOF PR-AUC (AP Score): {ap_score:.6f}")

# 12-1. ROC Curve 시각화
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {oof_auc:.6f})")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random Baseline")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("OOF ROC Curve")
plt.legend()
plt.tight_layout()

roc_plot_path = os.path.join(OUTPUT_DIR, "xgb_v1_roc_curve.png")
plt.savefig(roc_plot_path, bbox_inches="tight")
plt.show()
plt.close()
print(f"[저장] {roc_plot_path}")

# 12-2. PR Curve 시각화
plt.figure(figsize=(6, 5))
plt.plot(recall, precision, label=f"PR Curve (AP = {ap_score:.6f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("OOF Precision-Recall Curve")
plt.legend()
plt.tight_layout()

pr_plot_path = os.path.join(OUTPUT_DIR, "xgb_v1_pr_curve.png")
plt.savefig(pr_plot_path, bbox_inches="tight")
plt.show()
plt.close()
print(f"[저장] {pr_plot_path}")

# 12-3. Threshold별 confusion matrix 요약
# ROC-AUC 대회에서 threshold 자체는 제출에 사용하지 않지만,
# 모델이 어느 구간에서 잘 구분하는지 분석하는 데 유용함
threshold_list = [0.30, 0.40, 0.50, 0.60, 0.70]
threshold_rows = []

for th in threshold_list:
    pred_label = (oof_preds >= th).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred_label).ravel()

    acc  = (tp + tn) / (tp + tn + fp + fn)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1   = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

    threshold_rows.append({
        "threshold": th,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "specificity": spec,
        "f1": f1
    })

threshold_df = pd.DataFrame(threshold_rows)

print("\n[Threshold별 분류 성능 요약]")
print(threshold_df.to_string(index=False))

threshold_path = os.path.join(OUTPUT_DIR, "xgb_v1_threshold_analysis.csv")
threshold_df.to_csv(threshold_path, index=False, encoding="utf-8-sig")
print(f"\n[저장] {threshold_path}")


# =========================================================
# 13. OOF 예측값 저장
# =========================================================
print_section("13. OOF 예측값 저장")

# OOF 예측은 나중에 스태킹/앙상블의 메타 피처로 재활용 가능
oof_df = pd.DataFrame({
    ID_COL: train[ID_COL].values,
    TARGET_COL: y.values,
    "oof_pred_prob": oof_preds,
    # 0.5 threshold label은 참고용 분석용이다.
    # 대회 평가는 ROC-AUC(확률값 기준)이므로 핵심은 oof_pred_prob이다.
    "oof_pred_label": (oof_preds >= 0.5).astype(int),
})
oof_df["correct"] = (oof_df[TARGET_COL] == oof_df["oof_pred_label"]).astype(int)

oof_path = os.path.join(OUTPUT_DIR, "xgb_v1_oof_predictions.csv")
oof_df.to_csv(oof_path, index=False, encoding="utf-8-sig")
print(f"[저장] {oof_path}")


# =========================================================
# 14. test 예측 및 제출 파일 생성
# =========================================================
print_section("14. 제출 파일 생성")

# test_preds는 5개 폴드 예측의 평균 (폴드 앙상블)
# round() 없이 float 그대로 저장 → ranking 손실 방지
submission = pd.DataFrame({
    ID_COL: test[ID_COL].values,
    TARGET_COL: test_preds,
})

sub_path = os.path.join(OUTPUT_DIR, "xgb_v1_submission.csv")
submission.to_csv(sub_path, index=False, encoding="utf-8-sig")
print(f"[저장] {sub_path}")
print(submission.head(10))


# =========================================================
# 15. 종료 요약
# =========================================================
print_section("15. 완료")

print(f"OOF AUC  : {oof_auc:.6f}")
print(f"Fold AUC : {[round(a, 6) for a in fold_aucs]}")
print(f"Mean AUC : {np.mean(fold_aucs):.6f}  ±  {np.std(fold_aucs):.6f}")
print(f"PR-AUC   : {ap_score:.6f}")

print("\n저장 파일:")
print(f"  1. {imp_path}")
print(f"  2. {roc_plot_path}")
print(f"  3. {pr_plot_path}")
print(f"  4. {threshold_path}")
print(f"  5. {oof_path}")
print(f"  6. {sub_path}")