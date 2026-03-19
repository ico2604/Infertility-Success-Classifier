# =========================================================
# CatBoost Stratified K-Fold Validation + Optuna v3
# ---------------------------------------------------------
# 목적
# 1) train/test 데이터를 불러온다.
# 2) XGBoost v2와 동일한 기본 수치 변환(횟수형/나이형)을 수행한다.
# 3) XGBoost v2와 동일한 IVF 의료 도메인 파생변수를 생성한다.
# 4) CatBoost native categorical 처리에 맞게 범주형 컬럼을 유지한다.
# 5) Optuna로 CatBoost 하이퍼파라미터를 탐색한다.
# 6) Best Params로 Stratified K-Fold + OOF 방식의 최종 학습을 수행한다.
# 7) 실험 결과를 누적 로그로 저장한다.
# 8) Feature Importance / OOF / Submission 파일을 저장한다.
#
# 핵심 포인트
# - 기존 catboost_v2 pipeline 최대한 유지
# - COUNT/AGE 원본 문자열 컬럼 제거
# - 일반 범주형 문자열은 CatBoost에 그대로 전달
# - Optuna 탐색 후 best params 기반 최종 OOF / submission 생성
# - OOF 파일과 제출 파일의 target 컬럼명을 분리 관리
#   · OOF 평가용 컬럼명      : "임신 성공 여부"
#   · 제출 파일 확률 컬럼명  : "probability"
# - Optuna CV split은 trial마다 새로 만들지 않고 고정해서 재사용
# - N_TRIALS = 50으로 확장
# =========================================================

import os
import re
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import optuna

from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix
)

from catboost import CatBoostClassifier, Pool


# =========================================================
# 0. 경로 / 기본 설정
# =========================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")

# 원본 학습 데이터 기준 컬럼
TARGET_COL = "임신 성공 여부"   # OOF 평가용
ID_COL = "ID"
OOF_PRED_COL = "oof_pred_prob"

# 대회 제출 형식
SUBMIT_ID_COL = "ID"
SUBMIT_TARGET_COL = "probability"

SEED = 42
N_FOLDS = 5

EXPERIMENT_NAME = "catboost_optuna_v3"

# Optuna 설정
N_TRIALS = 50
TIMEOUT = None          # 예: 3600
OPTUNA_DIRECTION = "maximize"

# 최종 학습 시 verbose
FINAL_VERBOSE = 200


# =========================================================
# 1. 보조 함수
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


# =========================================================
# 2. 데이터 로드
# =========================================================
print_section("2. 데이터 로드")

raw_train = pd.read_csv(TRAIN_PATH)
raw_test = pd.read_csv(TEST_PATH)

train = raw_train.copy()
test = raw_test.copy()
original_columns = raw_train.columns.tolist()

print("train shape:", train.shape)
print("test  shape:", test.shape)


# =========================================================
# 3. 기본 수치 변환
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
# 5. CatBoost용 전처리
# =========================================================
print_section("5. CatBoost용 전처리")

# XGB와 동일하게 원본 COUNT/AGE 문자열 컬럼 제거
drop_cols = [c for c in COUNT_COLS + AGE_COLS if c in train.columns]
train = train.drop(columns=drop_cols, errors="ignore")
test = test.drop(columns=drop_cols, errors="ignore")

print(f"원본 문자열 컬럼 {len(drop_cols)}개 제거 완료")

# bool -> int
for df in [train, test]:
    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
    for c in bool_cols:
        df[c] = df[c].astype(int)

# categorical 후보 선정
exclude_cols = [TARGET_COL, ID_COL]
cat_cols = []

for c in train.columns:
    if c in exclude_cols:
        continue
    if not pd.api.types.is_numeric_dtype(train[c]):
        cat_cols.append(c)

print(f"CatBoost categorical 컬럼 수: {len(cat_cols)}")
print(cat_cols)

# categorical 결측 처리
for col in cat_cols:
    train[col] = train[col].fillna("MISSING").astype(str)
    test[col] = test[col].fillna("MISSING").astype(str)

# numeric coercion
num_cols = [c for c in train.columns if c not in cat_cols + [TARGET_COL, ID_COL]]
for col in num_cols:
    train[col] = pd.to_numeric(train[col], errors="coerce")
    test[col] = pd.to_numeric(test[col], errors="coerce")

missing_summary = train.isnull().sum()
print(f"결측 있는 컬럼 수: {(missing_summary > 0).sum()}")


# =========================================================
# 6. 입력 데이터 구성
# =========================================================
print_section("6. 입력 데이터 구성")

X = train.drop(columns=[TARGET_COL, ID_COL], errors="ignore")
y = train[TARGET_COL]
X_test = test.drop(columns=[ID_COL], errors="ignore")

common_cols = [c for c in X.columns if c in X_test.columns]
X = X[common_cols]
X_test = X_test[common_cols]
cat_cols = [c for c in cat_cols if c in common_cols]

print(f"X shape      : {X.shape}")
print(f"y shape      : {y.shape}")
print(f"X_test shape : {X_test.shape}")
print(f"cat cols     : {len(cat_cols)}")

non_numeric_in_X = [
    c for c in X.columns
    if c not in cat_cols and not pd.api.types.is_numeric_dtype(X[c])
]
print("\n[X 안의 비수치형 컬럼 중 cat_cols에 없는 컬럼]")
print(non_numeric_in_X)


# =========================================================
# 7. Optuna 목적 함수
# =========================================================
print_section("7. Optuna 설정")

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Optuna용 CV split 고정
cv_splits = list(
    StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=SEED
    ).split(X, y)
)


def build_cat_params(trial):
    params = {
        "iterations": trial.suggest_categorical("iterations", [3000, 4000, 5000, 6000]),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.05, log=True),
        "depth": trial.suggest_int("depth", 4, 8),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "random_strength": trial.suggest_float("random_strength", 0.0, 5.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 3.0),
        "border_count": trial.suggest_int("border_count", 64, 255),
        "leaf_estimation_iterations": trial.suggest_int("leaf_estimation_iterations", 1, 10),
        "eval_metric": "AUC",
        "loss_function": "Logloss",
        "random_seed": SEED,
        "verbose": 0,
        "allow_writing_files": False,
        "early_stopping_rounds": 200,
    }
    return params


def objective(trial):
    params = build_cat_params(trial)
    fold_aucs = []

    for fold, (tr_idx, va_idx) in enumerate(cv_splits):
        X_tr, X_va = X.iloc[tr_idx].copy(), X.iloc[va_idx].copy()
        y_tr, y_va = y.iloc[tr_idx].copy(), y.iloc[va_idx].copy()

        train_pool = Pool(X_tr, y_tr, cat_features=cat_cols)
        valid_pool = Pool(X_va, y_va, cat_features=cat_cols)

        model = CatBoostClassifier(**params)
        model.fit(
            train_pool,
            eval_set=valid_pool,
            use_best_model=True
        )

        va_pred = model.predict_proba(valid_pool)[:, 1]
        fold_auc = roc_auc_score(y_va, va_pred)
        fold_aucs.append(fold_auc)

    mean_auc = float(np.mean(fold_aucs))
    trial.set_user_attr("fold_aucs", [round(x, 6) for x in fold_aucs])
    return mean_auc


study = optuna.create_study(
    direction=OPTUNA_DIRECTION,
    sampler=optuna.samplers.TPESampler(seed=SEED)
)

print(f"N_TRIALS : {N_TRIALS}")
print(f"TIMEOUT  : {TIMEOUT}")

study.optimize(objective, n_trials=N_TRIALS, timeout=TIMEOUT, show_progress_bar=True)

best_trial = study.best_trial
BEST_PARAMS = build_cat_params(best_trial)
BEST_SCORE = best_trial.value

print("\n[Best Trial]")
print(f"Trial Number : {best_trial.number}")
print(f"Best AUC     : {BEST_SCORE:.6f}")
print("Best Params:")
for k, v in BEST_PARAMS.items():
    print(f"  {k:30s}: {v}")


# =========================================================
# 8. Best Params 저장
# =========================================================
print_section("8. Best Params 저장")

best_params_json_path = os.path.join(OUTPUT_DIR, f"{EXPERIMENT_NAME}_best_params.json")
study_summary_path = os.path.join(OUTPUT_DIR, f"{EXPERIMENT_NAME}_study_trials.csv")

with open(best_params_json_path, "w", encoding="utf-8") as f:
    json.dump(
        {
            "experiment_name": EXPERIMENT_NAME,
            "best_trial_number": best_trial.number,
            "best_cv_auc": BEST_SCORE,
            "best_params": BEST_PARAMS,
            "best_trial_fold_aucs": best_trial.user_attrs.get("fold_aucs", [])
        },
        f,
        ensure_ascii=False,
        indent=2
    )

trial_rows = []
for t in study.trials:
    row = {
        "trial_number": t.number,
        "value": t.value,
        "state": str(t.state),
    }
    for k, v in t.params.items():
        row[f"param_{k}"] = v
    trial_rows.append(row)

study_df = pd.DataFrame(trial_rows).sort_values("value", ascending=False)
study_df.to_csv(study_summary_path, index=False, encoding="utf-8-sig")

print(f"[저장] {best_params_json_path}")
print(f"[저장] {study_summary_path}")


# =========================================================
# 9. Best Params로 최종 Stratified K-Fold 학습 + OOF
# =========================================================
print_section("9. Best Params 최종 학습")

FINAL_CAT_PARAMS = BEST_PARAMS.copy()
FINAL_CAT_PARAMS["verbose"] = FINAL_VERBOSE

skf_final = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))
fold_aucs = []
best_iters = []
models = []

for fold, (tr_idx, va_idx) in enumerate(skf_final.split(X, y)):
    print(f"\n{'─' * 60}")
    print(f"  Fold {fold + 1} / {N_FOLDS}  [{EXPERIMENT_NAME}]")
    print(f"{'─' * 60}")

    X_tr, X_va = X.iloc[tr_idx].copy(), X.iloc[va_idx].copy()
    y_tr, y_va = y.iloc[tr_idx].copy(), y.iloc[va_idx].copy()

    train_pool = Pool(X_tr, y_tr, cat_features=cat_cols)
    valid_pool = Pool(X_va, y_va, cat_features=cat_cols)
    test_pool = Pool(X_test, cat_features=cat_cols)

    model = CatBoostClassifier(**FINAL_CAT_PARAMS)
    model.fit(
        train_pool,
        eval_set=valid_pool,
        use_best_model=True
    )

    va_pred = model.predict_proba(valid_pool)[:, 1]
    te_pred = model.predict_proba(test_pool)[:, 1]

    oof_preds[va_idx] = va_pred
    test_preds += te_pred / N_FOLDS

    fold_auc = roc_auc_score(y_va, va_pred)
    best_iter = model.get_best_iteration()

    fold_aucs.append(fold_auc)
    best_iters.append(best_iter)
    models.append(model)

    print(f"  Fold {fold + 1} AUC     : {fold_auc:.6f}")
    print(f"  Best Iteration : {best_iter}")


# =========================================================
# 10. 성능 요약
# =========================================================
print_section("10. 성능 요약")

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
# 11. 실험 결과 누적 로그 저장
# =========================================================
print_section("11. 실험 결과 로그 저장")

log_path = os.path.join(OUTPUT_DIR, "catboost_optuna_results_log.csv")

log_row = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "experiment_name": EXPERIMENT_NAME,
    "n_trials": N_TRIALS,
    "best_trial_number": best_trial.number,
    "optuna_best_cv_auc": round(BEST_SCORE, 6),
    "final_oof_auc": round(oof_auc, 6),
    "mean_auc": round(mean_auc, 6),
    "std_auc": round(std_auc, 6),
    "fold1_auc": round(fold_aucs[0], 6),
    "fold2_auc": round(fold_aucs[1], 6),
    "fold3_auc": round(fold_aucs[2], 6),
    "fold4_auc": round(fold_aucs[3], 6),
    "fold5_auc": round(fold_aucs[4], 6),
    "mean_best_iter": round(np.mean([b for b in best_iters if b is not None]), 1) if any(b is not None for b in best_iters) else None,
    **{f"param_{k}": v for k, v in BEST_PARAMS.items()},
}

log_df = pd.DataFrame([log_row])

if os.path.exists(log_path):
    existing = pd.read_csv(log_path)
    log_df = pd.concat([existing, log_df], ignore_index=True)

log_df.to_csv(log_path, index=False, encoding="utf-8-sig")
print(f"[저장] {log_path}")
print("\n[현재까지 실험 요약]")
print(
    log_df[
        ["timestamp", "experiment_name", "optuna_best_cv_auc", "final_oof_auc", "mean_auc", "std_auc"]
    ].to_string(index=False)
)


# =========================================================
# 12. Feature Importance (폴드 평균)
# =========================================================
print_section("12. Feature Importance")

importance_list = []
for i, model in enumerate(models, start=1):
    fold_imp = pd.DataFrame({
        "feature": X.columns,
        "importance": model.get_feature_importance(type="FeatureImportance"),
        "fold": i
    })
    importance_list.append(fold_imp)

importance_df = pd.concat(importance_list, ignore_index=True)
importance_summary = (
    importance_df
    .groupby("feature", as_index=False)
    .agg(
        importance=("importance", "mean"),
        std=("importance", "std")
    )
    .sort_values("importance", ascending=False)
    .reset_index(drop=True)
)

print(importance_summary.head(30).to_string(index=False))

low_imp = importance_summary[
    importance_summary["importance"] < importance_summary["importance"].quantile(0.1)
]
print(f"\n중요도 하위 10% ({len(low_imp)}개) — 제거 후보:")
print(low_imp["feature"].tolist())

imp_path = os.path.join(OUTPUT_DIR, f"{EXPERIMENT_NAME}_feature_importance.csv")
importance_summary.to_csv(imp_path, index=False, encoding="utf-8-sig")
print(f"\n[저장] {imp_path}")


# =========================================================
# 13. PR Curve 요약값
# =========================================================
print_section("13. PR Curve 요약값")

precision, recall, _ = precision_recall_curve(y, oof_preds)
ap_score = average_precision_score(y, oof_preds)

print(f"PR-AUC (AP Score): {ap_score:.6f}")


# =========================================================
# 14. OOF 예측값 저장
# =========================================================
print_section("14. OOF 예측값 저장")

oof_df = pd.DataFrame({
    ID_COL: train[ID_COL].values,
    TARGET_COL: y.values,
    OOF_PRED_COL: oof_preds,
    "oof_pred_label": (oof_preds >= 0.5).astype(int),
})
oof_df["correct"] = (oof_df[TARGET_COL] == oof_df["oof_pred_label"]).astype(int)

oof_path = os.path.join(OUTPUT_DIR, f"{EXPERIMENT_NAME}_oof_predictions.csv")
oof_df.to_csv(oof_path, index=False, encoding="utf-8-sig")
print(f"[저장] {oof_path}")


# =========================================================
# 15. Threshold 분석
# =========================================================
print_section("15. Threshold 분석")

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

threshold_path = os.path.join(OUTPUT_DIR, f"{EXPERIMENT_NAME}_threshold_analysis.csv")
threshold_df.to_csv(threshold_path, index=False, encoding="utf-8-sig")
print(f"[저장] {threshold_path}")


# =========================================================
# 16. 제출 파일 생성
# =========================================================
print_section("16. 제출 파일 생성")

submission = pd.DataFrame({
    SUBMIT_ID_COL: test[ID_COL].values,
    SUBMIT_TARGET_COL: test_preds
})

sub_path = os.path.join(OUTPUT_DIR, f"{EXPERIMENT_NAME}_submission.csv")
submission.to_csv(sub_path, index=False, encoding="utf-8-sig")
print(f"[저장] {sub_path}")
print(submission.head(10))


# =========================================================
# 17. 종료 요약
# =========================================================
print_section("17. 완료")

print(f"실험명               : {EXPERIMENT_NAME}")
print(f"Optuna Trial 수      : {N_TRIALS}")
print(f"Best Trial Number    : {best_trial.number}")
print(f"Optuna Best CV AUC   : {BEST_SCORE:.6f}")
print(f"최종 OOF AUC         : {oof_auc:.6f}")
print(f"Mean AUC             : {mean_auc:.6f}  ±  {std_auc:.6f}")
print(f"PR-AUC               : {ap_score:.6f}")

print("\nBest Params:")
for k, v in BEST_PARAMS.items():
    print(f"  - {k}: {v}")

print("\n컬럼 설정:")
print(f"  - TARGET_COL        : {TARGET_COL}")
print(f"  - ID_COL            : {ID_COL}")
print(f"  - OOF_PRED_COL      : {OOF_PRED_COL}")
print(f"  - SUBMIT_ID_COL     : {SUBMIT_ID_COL}")
print(f"  - SUBMIT_TARGET_COL : {SUBMIT_TARGET_COL}")

print("\n저장 파일:")
print(f"  1. {best_params_json_path}")
print(f"  2. {study_summary_path}")
print(f"  3. {log_path}")
print(f"  4. {imp_path}")
print(f"  5. {oof_path}")
print(f"  6. {threshold_path}")
print(f"  7. {sub_path}")