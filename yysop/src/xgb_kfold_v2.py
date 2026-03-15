# =========================================================
# XGBoost Stratified K-Fold Validation v2 — 파라미터 수동 튜닝
# ---------------------------------------------------------
# 목적
# 1) train/test 데이터를 불러온다.
# 2) 기본 수치 변환(횟수형/나이형)을 수행한다.
# 3) IVF 의료 도메인 기반 파생변수를 생성한다.
# 4) 범주형 컬럼을 안전하게 숫자형으로 인코딩한다.
# 5) Stratified K-Fold + OOF 방식으로 안정적인 내부 검증을 수행한다.
# 6) 실험 결과를 누적 로그로 저장한다.
# 7) Feature Importance / ROC Curve / OOF / Submission 파일을 저장한다.
#
# v2 핵심 포인트
# 1) EXPERIMENT_NAME만 바꾸면 여러 파라미터 실험 가능
# 2) 결과가 outputs/xgb_v2_results_log.csv에 누적 저장됨
# 3) 실험별 OOF AUC / Fold Mean / Std를 쉽게 비교 가능
# 4) XGBoost 3.2.0 기준 early_stopping_rounds는 생성자에 넣음
#
# 현재 기본 추천 설정
# - EXPERIMENT_NAME = "reg_relax"
# - SCALE_POS_WEIGHT_MODE = "off"
#
# 이유
# - baseline이 다소 보수적으로 보여 표현력을 조금 풀어주는 방향이 유리할 수 있음
# - 현재 데이터 불균형은 있지만 아주 극단적이지 않아 scale_pos_weight를 끈 버전도
#   반드시 비교해볼 가치가 큼
# =========================================================

import os
import re
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
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

TARGET_COL = "임신 성공 여부"   # 학습/OOF 평가용
ID_COL = "ID"

SUBMIT_ID_COL = "ID"
SUBMIT_TARGET_COL = "probability"

SEED = 42
N_FOLDS = 5

# 양성 클래스 비율이 이 값보다 작으면 auto 모드에서 neg/pos 적용
IMBALANCE_THRESHOLD = 0.30

# scale_pos_weight 실험 모드
# "auto"   : 양성 비율 보고 자동 결정
# "off"    : 무조건 1.0 사용
# "manual" : 아래 MANUAL_SCALE_POS_WEIGHT 사용
#
# 현재는 "off"를 기본 추천값으로 둠.
# 이유: 현재 데이터는 불균형이 있긴 하지만 아주 극단적이지 않아
# auto보다 off가 더 잘 나올 가능성이 있음.
SCALE_POS_WEIGHT_MODE = "off"
MANUAL_SCALE_POS_WEIGHT = 1.0  # mode="manual"일 때만 사용


# =========================================================
# 1. 실험 설정
# =========================================================
# 실험 순서 추천
# 1) baseline   : 기준점
# 2) reg_relax  : 규제 완화 (가장 우선 추천)
# 3) depth_up   : 표현력 확대
# 4) lr_down    : 더 세밀한 학습
# 5) sampling   : 샘플링 완화
# 6) best_combo : 좋은 것들 조합
#
# 현재 기본 추천:
#   EXPERIMENT_NAME = "reg_relax"
EXPERIMENT_NAME = "reg_relax"

EXPERIMENTS = {
    # -----------------------------------------------------
    # 기준점
    # -----------------------------------------------------
    "baseline": dict(
        n_estimators=3000,
        learning_rate=0.02,
        max_depth=5,
        min_child_weight=5,
        gamma=0.1,
        subsample=0.8,
        colsample_bytree=0.7,
        colsample_bylevel=0.7,
        reg_alpha=0.1,
        reg_lambda=1.5,
        early_stopping_rounds=50,
    ),

    # -----------------------------------------------------
    # 깊이 확대
    # max_depth 5 -> 6
    # 상호작용 학습 능력 강화
    # -----------------------------------------------------
    "depth_up": dict(
        n_estimators=3000,
        learning_rate=0.02,
        max_depth=6,
        min_child_weight=5,
        gamma=0.1,
        subsample=0.8,
        colsample_bytree=0.7,
        colsample_bylevel=0.7,
        reg_alpha=0.1,
        reg_lambda=1.5,
        early_stopping_rounds=50,
    ),

    # -----------------------------------------------------
    # 학습률 낮추기
    # 더 세밀하게 수렴, 대신 학습 시간 증가
    # -----------------------------------------------------
    "lr_down": dict(
        n_estimators=5000,
        learning_rate=0.01,
        max_depth=5,
        min_child_weight=5,
        gamma=0.1,
        subsample=0.8,
        colsample_bytree=0.7,
        colsample_bylevel=0.7,
        reg_alpha=0.1,
        reg_lambda=1.5,
        early_stopping_rounds=50,
    ),

    # -----------------------------------------------------
    # 규제 완화
    # 현재 baseline이 보수적인 편이라 가장 먼저 추천
    # -----------------------------------------------------
    "reg_relax": dict(
        n_estimators=3000,
        learning_rate=0.02,
        max_depth=5,
        min_child_weight=3,
        gamma=0.0,
        subsample=0.8,
        colsample_bytree=0.7,
        colsample_bylevel=0.7,
        reg_alpha=0.0,
        reg_lambda=1.0,
        early_stopping_rounds=50,
    ),

    # -----------------------------------------------------
    # 샘플링 비율 완화
    # 트리가 더 많은 행/열을 보게 해서 표현력 확대
    # -----------------------------------------------------
    "sampling": dict(
        n_estimators=3000,
        learning_rate=0.02,
        max_depth=5,
        min_child_weight=5,
        gamma=0.1,
        subsample=0.9,
        colsample_bytree=0.8,
        colsample_bylevel=0.8,
        reg_alpha=0.1,
        reg_lambda=1.5,
        early_stopping_rounds=50,
    ),

    # -----------------------------------------------------
    # 조합 예시
    # 실제로는 log 보고 좋은 것들만 조합 추천
    # -----------------------------------------------------
    "best_combo": dict(
        n_estimators=5000,
        learning_rate=0.01,
        max_depth=6,
        min_child_weight=3,
        gamma=0.0,
        subsample=0.8,
        colsample_bytree=0.8,
        colsample_bylevel=0.7,
        reg_alpha=0.0,
        reg_lambda=1.0,
        early_stopping_rounds=50,
    ),
}

assert EXPERIMENT_NAME in EXPERIMENTS, f"EXPERIMENT_NAME '{EXPERIMENT_NAME}'이 EXPERIMENTS에 없습니다."
TUNE_PARAMS = EXPERIMENTS[EXPERIMENT_NAME]

print(f"\n실험 이름: {EXPERIMENT_NAME}")
print("파라미터:")
for k, v in TUNE_PARAMS.items():
    print(f"  {k:30s}: {v}")


# =========================================================
# 2. 보조 함수
# =========================================================
def print_section(title: str):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def parse_korean_count(x):
    """
    '0회', '1회', '3회 이상' -> 숫자 변환
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
    '만18-34세' -> 26.0 같은 구간 중간값으로 변환
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
    0으로 나누거나 NaN이면 NaN 유지
    """
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    return np.where((np.isnan(b)) | (b == 0), np.nan, a / b)


def decide_scale_pos_weight(y, mode="auto", threshold=0.30, manual_value=1.0):
    """
    scale_pos_weight 결정 함수
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
# 5. 의료 도메인 파생변수 생성
# =========================================================
print_section("5. 의료 도메인 파생변수 생성")


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
# 6. 범주형 컬럼 인코딩
# =========================================================
print_section("6. 범주형 컬럼 인코딩")

# 숫자형 / bool / category가 아닌 컬럼은 전부 범주형으로 처리
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
        train[col].astype(str).fillna("MISSING"),
        test[col].astype(str).fillna("MISSING")
    ], axis=0)
    codes, _ = pd.factorize(combined)
    train[col] = codes[:len(train)]
    test[col] = codes[len(train):]

print(f"범주형 인코딩 완료: {len(cat_cols)}개 컬럼")

# 인코딩 후 비수치형 컬럼 점검
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
# 7. 원본 문자열 컬럼 제거
# =========================================================
print_section("7. 원본 문자열 컬럼 제거 및 결측 확인")

drop_cols = [c for c in COUNT_COLS + AGE_COLS if c in train.columns]
train = train.drop(columns=drop_cols, errors="ignore")
test = test.drop(columns=drop_cols, errors="ignore")

print(f"원본 문자열 컬럼 {len(drop_cols)}개 제거 완료")
missing_summary = train.isnull().sum()
print(f"결측 있는 컬럼 수: {(missing_summary > 0).sum()}")


# =========================================================
# 8. 입력 데이터 구성
# =========================================================
print_section("8. 입력 데이터 구성")

X = train.drop(columns=[TARGET_COL, ID_COL], errors="ignore")
y = train[TARGET_COL]
X_test = test.drop(columns=[ID_COL], errors="ignore")

common_cols = [c for c in X.columns if c in X_test.columns]
X = X[common_cols]
X_test = X_test[common_cols]

print(f"X shape      : {X.shape}")
print(f"y shape      : {y.shape}")
print(f"X_test shape : {X_test.shape}")

# 최종 입력에서 비수치형 컬럼이 없는지 점검
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
# 9. XGBoost 모델 파라미터 정의
# =========================================================
print_section("9. XGBoost 모델 파라미터 정의")

XGB_PARAMS = dict(
    **TUNE_PARAMS,
    scale_pos_weight=scale_pos_weight,
    objective="binary:logistic",
    eval_metric="auc",
    tree_method="hist",
    random_state=SEED,
    n_jobs=-1,
)

print(f"실험: [{EXPERIMENT_NAME}]")
for k, v in XGB_PARAMS.items():
    print(f"  {k:30s}: {v}")


# =========================================================
# 10. Stratified K-Fold 학습 + OOF 예측
# =========================================================
print_section("10. Stratified K-Fold 학습")

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))
fold_aucs = []
best_iters = []
models = []

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
    print(f"\n{'─' * 60}")
    print(f"  Fold {fold + 1} / {N_FOLDS}  [{EXPERIMENT_NAME}]")
    print(f"{'─' * 60}")

    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

    model = XGBClassifier(**XGB_PARAMS)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=200)

    oof_preds[va_idx] = model.predict_proba(X_va)[:, 1]
    test_preds += model.predict_proba(X_test)[:, 1] / N_FOLDS

    fold_auc = roc_auc_score(y_va, oof_preds[va_idx])
    best_iter = getattr(model, "best_iteration", None)

    fold_aucs.append(fold_auc)
    best_iters.append(best_iter)
    models.append(model)

    print(f"  Fold {fold + 1} AUC     : {fold_auc:.6f}")
    print(f"  Best Iteration : {best_iter}")


# =========================================================
# 11. 성능 요약
# =========================================================
print_section("11. 성능 요약")

oof_auc = roc_auc_score(y, oof_preds)
mean_auc = np.mean(fold_aucs)
std_auc = np.std(fold_aucs)

print(f"\n{'폴드':>6}  {'AUC':>10}  {'Best Iter':>10}")
print("-" * 34)
for i, (auc, bi) in enumerate(zip(fold_aucs, best_iters)):
    print(f"Fold {i + 1:>2}  {auc:>10.6f}  {str(bi):>10}")
print("-" * 34)
print(f"{'Mean':>6}  {mean_auc:>10.6f}")
print(f"{'Std':>6}  {std_auc:>10.6f}")
print(f"\n최종 OOF AUC : {oof_auc:.6f}")

if std_auc > 0.01:
    print("⚠ 폴드 간 AUC 편차가 큽니다.")
else:
    print("✓ 폴드 간 AUC 편차가 안정적입니다.")


# =========================================================
# 12. 실험 결과 누적 로그 저장
# =========================================================
print_section("12. 실험 결과 로그 저장")

log_path = os.path.join(OUTPUT_DIR, "xgb_v2_results_log.csv")

log_row = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "experiment": EXPERIMENT_NAME,
    "scale_pos_weight_mode": SCALE_POS_WEIGHT_MODE,
    "scale_pos_weight": round(scale_pos_weight, 6),
    "oof_auc": round(oof_auc, 6),
    "mean_auc": round(mean_auc, 6),
    "std_auc": round(std_auc, 6),
    "fold1_auc": round(fold_aucs[0], 6),
    "fold2_auc": round(fold_aucs[1], 6),
    "fold3_auc": round(fold_aucs[2], 6),
    "fold4_auc": round(fold_aucs[3], 6),
    "fold5_auc": round(fold_aucs[4], 6),
    "mean_best_iter": round(np.mean([b for b in best_iters if b is not None]), 1) if any(b is not None for b in best_iters) else None,
    **{f"param_{k}": v for k, v in TUNE_PARAMS.items()},
}

log_df = pd.DataFrame([log_row])

if os.path.exists(log_path):
    existing = pd.read_csv(log_path)
    log_df = pd.concat([existing, log_df], ignore_index=True)

log_df.to_csv(log_path, index=False, encoding="utf-8-sig")
print(f"[저장] {log_path}")
print("\n[현재까지 실험 요약]")
print(log_df[["timestamp", "experiment", "scale_pos_weight_mode", "oof_auc", "mean_auc", "std_auc"]].to_string(index=False))


# =========================================================
# 13. Feature Importance (폴드 평균)
# =========================================================
print_section("13. Feature Importance")

importance_matrix = np.array([m.feature_importances_ for m in models])
mean_importance = importance_matrix.mean(axis=0)
std_importance = importance_matrix.std(axis=0)

importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance": mean_importance,
    "std": std_importance
}).sort_values("importance", ascending=False).reset_index(drop=True)

print(importance_df.head(30).to_string(index=False))

low_imp = importance_df[
    importance_df["importance"] < importance_df["importance"].quantile(0.1)
]
print(f"\n중요도 하위 10% ({len(low_imp)}개) — 제거 후보:")
print(low_imp["feature"].tolist())

imp_path = os.path.join(OUTPUT_DIR, f"xgb_v2_{EXPERIMENT_NAME}_feature_importance.csv")
importance_df.to_csv(imp_path, index=False, encoding="utf-8-sig")
print(f"\n[저장] {imp_path}")


# =========================================================
# 14. ROC / PR Curve
# =========================================================
print_section("14. ROC / PR Curve")

# ROC
fpr, tpr, _ = roc_curve(y, oof_preds)

# PR
precision, recall, _ = precision_recall_curve(y, oof_preds)
ap_score = average_precision_score(y, oof_preds)

# ROC Curve 저장
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC={oof_auc:.4f}) [{EXPERIMENT_NAME}]")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"OOF ROC Curve [{EXPERIMENT_NAME}]")
plt.legend()
plt.tight_layout()

roc_path = os.path.join(OUTPUT_DIR, f"xgb_v2_{EXPERIMENT_NAME}_roc_curve.png")
plt.savefig(roc_path, bbox_inches="tight")
plt.show()
plt.close()
print(f"[저장] {roc_path}")

# PR Curve 저장
plt.figure(figsize=(6, 5))
plt.plot(recall, precision, label=f"PR Curve (AP={ap_score:.4f}) [{EXPERIMENT_NAME}]")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title(f"OOF PR Curve [{EXPERIMENT_NAME}]")
plt.legend()
plt.tight_layout()

pr_path = os.path.join(OUTPUT_DIR, f"xgb_v2_{EXPERIMENT_NAME}_pr_curve.png")
plt.savefig(pr_path, bbox_inches="tight")
plt.show()
plt.close()
print(f"[저장] {pr_path}")

print(f"PR-AUC (AP Score): {ap_score:.6f}")


# =========================================================
# 15. OOF 예측값 저장
# =========================================================
print_section("15. OOF 예측값 저장")

oof_df = pd.DataFrame({
    ID_COL: train[ID_COL].values,
    TARGET_COL: y.values,
    "oof_pred_prob": oof_preds,
    # 0.5 threshold label은 참고용
    "oof_pred_label": (oof_preds >= 0.5).astype(int),
})
oof_df["correct"] = (oof_df[TARGET_COL] == oof_df["oof_pred_label"]).astype(int)

oof_path = os.path.join(OUTPUT_DIR, f"xgb_v2_{EXPERIMENT_NAME}_oof_predictions.csv")
oof_df.to_csv(oof_path, index=False, encoding="utf-8-sig")
print(f"[저장] {oof_path}")


# =========================================================
# 16. Threshold 분석
# =========================================================
print_section("16. Threshold 분석")

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
print(threshold_df.to_string(index=False))

threshold_path = os.path.join(OUTPUT_DIR, f"xgb_v2_{EXPERIMENT_NAME}_threshold_analysis.csv")
threshold_df.to_csv(threshold_path, index=False, encoding="utf-8-sig")
print(f"[저장] {threshold_path}")


# =========================================================
# 17. 제출 파일 생성
# =========================================================
print_section("17. 제출 파일 생성")

submission = pd.DataFrame({
    SUBMIT_ID_COL: test[ID_COL].values,
    SUBMIT_TARGET_COL: test_preds,
})

sub_path = os.path.join(OUTPUT_DIR, f"xgb_v2_{EXPERIMENT_NAME}_submission.csv")
submission.to_csv(sub_path, index=False, encoding="utf-8-sig")
print(f"[저장] {sub_path}")
print(submission.head(10))


# =========================================================
# 18. 종료 요약
# =========================================================
print_section("18. 완료")

print(f"실험명  : {EXPERIMENT_NAME}")
print(f"spw mode: {SCALE_POS_WEIGHT_MODE}")
print(f"OOF AUC : {oof_auc:.6f}")
print(f"Mean AUC: {mean_auc:.6f}  ±  {std_auc:.6f}")
print(f"PR-AUC  : {ap_score:.6f}")

print("\n저장 파일:")
print(f"  1. {log_path}          ← 실험 누적 비교표")
print(f"  2. {imp_path}")
print(f"  3. {roc_path}")
print(f"  4. {pr_path}")
print(f"  5. {oof_path}")
print(f"  6. {threshold_path}")
print(f"  7. {sub_path}")

print("\n다음 추천 실험:")
print("  - EXPERIMENT_NAME = 'baseline'  + SCALE_POS_WEIGHT_MODE = 'off'")
print("  - EXPERIMENT_NAME = 'depth_up'  + SCALE_POS_WEIGHT_MODE = 'off'")
print("  - EXPERIMENT_NAME = 'lr_down'   + SCALE_POS_WEIGHT_MODE = 'off'")