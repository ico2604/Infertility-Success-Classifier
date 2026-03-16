#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v3.py - XGBoost GPU 파이프라인
       엄격한 전처리: 변환 후 object 컬럼 0개
       K-Fold Target Encoding (leakage-free)
       목표 AUC: 0.75~0.76
"""

import os, sys, warnings, time, re
import numpy as np, pandas as pd
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score
from xgboost import XGBClassifier
from tqdm import tqdm

warnings.filterwarnings("ignore")

VERSION = "v3"
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
print(f"# {VERSION} - XGBoost GPU Pipeline (strict preprocessing)")
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
# 2. 전처리 함수 정의
# ============================================================

# --- 상수 정의 ---
DROP_COLS = [
    "ID",
    TARGET,
    "PGD 시술 여부",
    "PGS 시술 여부",
    "난자 해동 경과일",
    "착상 전 유전 검사 사용 여부",
]

COUNT_COLS = [c for c in train_raw.columns if "횟수" in c]

AGE_COLS = ["시술 당시 나이", "난자 기증자 나이", "정자 기증자 나이"]

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


def safe_div(a, b, fill=0.0):
    return np.where(b > 0, a / b, fill)


def step1_drop_and_fillna(df):
    """1단계: 컬럼 삭제 + 결측 채우기"""
    drop = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=drop, errors="ignore")

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna("Unknown")
        else:
            df[col] = df[col].fillna(0)
    return df


def step2_count_to_int(df):
    """2단계: 횟수 컬럼 → 정수"""
    for col in df.columns:
        if "횟수" in col:
            if df[col].dtype == "object":
                df[col] = df[col].map(COUNT_MAP)
                df[col] = df[col].fillna(0).astype(int)
            else:
                df[col] = df[col].fillna(0).astype(int)
    return df


def step3_age_to_int(df):
    """3단계: 나이 컬럼 → 정수 (순서형)"""
    for col in AGE_COLS:
        if col not in df.columns:
            continue
        if df[col].dtype == "object":
            mapped = df[col].map(AGE_MAP)
            # 매핑 안 된 값 처리 (숫자 추출 시도)
            unmapped = mapped.isna() & (df[col] != "Unknown")
            if unmapped.any():
                for idx in df.index[unmapped]:
                    val = str(df.at[idx, col])
                    nums = re.findall(r"\d+", val)
                    if nums:
                        age_val = int(nums[0])
                        if age_val <= 34:
                            mapped.at[idx] = 1
                        elif age_val <= 37:
                            mapped.at[idx] = 2
                        elif age_val <= 39:
                            mapped.at[idx] = 3
                        elif age_val <= 42:
                            mapped.at[idx] = 4
                        elif age_val <= 44:
                            mapped.at[idx] = 5
                        elif age_val <= 50:
                            mapped.at[idx] = 6
                        else:
                            mapped.at[idx] = 0
            df[col] = mapped.fillna(0).astype(int)
    return df


def step4_bool_to_int(df):
    """4단계: TRUE/FALSE 류 컬럼 → 0/1 정수"""
    for col in df.columns:
        if df[col].dtype != "object":
            continue
        unique_vals = set(df[col].dropna().unique())
        # TRUE/FALSE 패턴 감지
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


def step5_feature_engineering(df):
    """5단계: 파생변수 생성 (행 독립 연산)"""

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
    embryo = df.get("총 생성 배아 수", pd.Series(0, index=df.index))
    mixed = df.get("혼합된 난자 수", pd.Series(0, index=df.index))
    embryo = pd.to_numeric(embryo, errors="coerce").fillna(0).values
    mixed = pd.to_numeric(mixed, errors="coerce").fillna(0).values
    df["수정_성공률"] = safe_div(embryo, mixed)

    # 이식_유형 One-Hot
    for col in IMPLANT_BOOL_COLS:
        if col in df.columns:
            if df[col].dtype == "object":
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
        donor_flag = np.where(
            df["난자 출처"].astype(str).str.contains("기증", na=False), 1, donor_flag
        )
    if "난자 기증자 나이" in df.columns:
        donor_flag = np.where(
            (df["난자 기증자 나이"] != 0) & (df["난자 기증자 나이"] != "Unknown"),
            1,
            donor_flag,
        )
    df["is_donor_egg"] = donor_flag.astype(int)

    # 타클리닉 시술
    if "총 시술 횟수" in df.columns and "클리닉 내 총 시술 횟수" in df.columns:
        tc = pd.to_numeric(df["총 시술 횟수"], errors="coerce").fillna(0)
        cc = pd.to_numeric(df["클리닉 내 총 시술 횟수"], errors="coerce").fillna(0)
        df["타클리닉_시술"] = (tc - cc).clip(0).astype(int)

    return df


def step6_drop_remaining_object(df, te_cols_to_keep):
    """6단계: TE 대상 외 남은 object 컬럼 강제 삭제"""
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

# --- train ---
train_df = train_raw.copy()
train_df = step1_drop_and_fillna(train_df)
print(
    f"  step1 (삭제+결측): {train_df.shape}, object={train_df.select_dtypes('object').shape[1]}"
)

train_df = step2_count_to_int(train_df)
print(f"  step2 (횟수→int): object={train_df.select_dtypes('object').shape[1]}")

train_df = step3_age_to_int(train_df)
print(f"  step3 (나이→int): object={train_df.select_dtypes('object').shape[1]}")

train_df = step4_bool_to_int(train_df)
print(f"  step4 (bool→int): object={train_df.select_dtypes('object').shape[1]}")

train_df = step5_feature_engineering(train_df)
print(
    f"  step5 (파생변수): 컬럼={train_df.shape[1]}, object={train_df.select_dtypes('object').shape[1]}"
)

# TE 전에 남은 object 중 TE 대상이 아닌 것 삭제
te_cols_exist = [c for c in TE_COLS if c in train_df.columns]
train_df = step6_drop_remaining_object(train_df, te_cols_exist)
print(
    f"  step6 (object삭제): 컬럼={train_df.shape[1]}, object={train_df.select_dtypes('object').shape[1]}"
)

# --- test (동일 과정) ---
test_df = test_raw.copy()
test_df = step1_drop_and_fillna(test_df)
test_df = step2_count_to_int(test_df)
test_df = step3_age_to_int(test_df)
test_df = step4_bool_to_int(test_df)
test_df = step5_feature_engineering(test_df)
test_df = step6_drop_remaining_object(test_df, te_cols_exist)
print(
    f"  test 전처리 완료: {test_df.shape}, object={test_df.select_dtypes('object').shape[1]}"
)

# ============================================================
# 4. Target Encoding (K-Fold, train만 사용)
# ============================================================
print(f"\n## [4] Target Encoding (K-Fold, leakage-free)")

global_mean = y.mean()
alpha = 10  # smoothing
te_mappings = {}  # test용 매핑 저장

skf_te = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

for col in te_cols_exist:
    print(f"  - {col} (고유값: {train_df[col].nunique()})")

    col_str = train_df[col].astype(str)
    oof_te = np.full(len(y), global_mean)

    # OOF 방식 (train 내부 fold 분리)
    for tr_i, va_i in skf_te.split(train_df, y):
        fold_vals = col_str.iloc[tr_i].values
        fold_y = y[tr_i]

        agg_df = pd.DataFrame({"val": fold_vals, "target": fold_y})
        agg = agg_df.groupby("val")["target"].agg(["mean", "count"])
        agg["smoothed"] = (agg["count"] * agg["mean"] + alpha * global_mean) / (
            agg["count"] + alpha
        )
        mapping = agg["smoothed"].to_dict()

        va_vals = col_str.iloc[va_i]
        oof_te[va_i] = va_vals.map(mapping).fillna(global_mean).values

    # train 컬럼 덮어쓰기 (숫자)
    train_df[col] = oof_te

    # test용 매핑 (train 전체 기준)
    full_df = pd.DataFrame({"val": col_str.values, "target": y})
    full_agg = full_df.groupby("val")["target"].agg(["mean", "count"])
    full_agg["smoothed"] = (
        full_agg["count"] * full_agg["mean"] + alpha * global_mean
    ) / (full_agg["count"] + alpha)
    te_mappings[col] = full_agg["smoothed"].to_dict()

    # test 컬럼 덮어쓰기
    test_col_str = test_df[col].astype(str)
    test_df[col] = test_col_str.map(te_mappings[col]).fillna(global_mean)

# object 컬럼 최종 확인
obj_remain_tr = train_df.select_dtypes("object").columns.tolist()
obj_remain_te = test_df.select_dtypes("object").columns.tolist()
print(f"  TE 후 object 컬럼 - train: {obj_remain_tr}, test: {obj_remain_te}")

if obj_remain_tr:
    train_df = train_df.drop(columns=obj_remain_tr)
    print(f"  train 잔여 object 삭제: {obj_remain_tr}")
if obj_remain_te:
    test_df = test_df.drop(columns=obj_remain_te)
    print(f"  test 잔여 object 삭제: {obj_remain_te}")

# ============================================================
# 5. 최종 피처 정리
# ============================================================
print(f"\n## [5] 최종 피처 정리")

# train-test 공통 컬럼만
common_cols = sorted(set(train_df.columns) & set(test_df.columns))
train_df = train_df[common_cols]
test_df = test_df[common_cols]

# float32 변환
X_train = train_df.values.astype(np.float32)
X_test = test_df.values.astype(np.float32)
feature_names = list(common_cols)

print(f"- X_train: {X_train.shape}")
print(f"- X_test: {X_test.shape}")
print(f"- 피처 수: {len(feature_names)}")
print(f"- object 컬럼: 0개 (확인: {train_df.select_dtypes('object').shape[1]})")
print(f"- NaN 확인 - train: {np.isnan(X_train).sum()}, test: {np.isnan(X_test).sum()}")
print(f"- 피처 목록:")
for i, f in enumerate(feature_names):
    print(f"  {i+1}. {f}")

# ============================================================
# 6. XGBoost 5-Fold 학습 (GPU)
# ============================================================
print(f"\n## [6] XGBoost {N_FOLDS}-Fold 학습 (GPU)")

spw = np.sum(y == 0) / np.sum(y == 1)
xgb_params = {
    "n_estimators": 5000,
    "learning_rate": 0.01,
    "max_depth": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.3,
    "reg_lambda": 0.3,
    "min_child_weight": 20,
    "scale_pos_weight": 3,
    "eval_metric": "auc",
    "tree_method": "gpu_hist",
    "gpu_id": 0,
    "random_state": SEED,
    "verbosity": 0,
    "early_stopping_rounds": 200,
}
print(f"- scale_pos_weight: {xgb_params['scale_pos_weight']} (실제 비율: {spw:.2f})")
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
print(f"- Fold별:")
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
print(f"\n## [8] 피처 중요도 (마지막 fold)")
fi = model.feature_importances_
fi_df = pd.DataFrame({"feature": feature_names, "importance": fi})
fi_df = fi_df.sort_values("importance", ascending=False)
print(f"| 순위 | 피처 | 중요도 |")
print(f"|------|------|--------|")
for i, (_, r) in enumerate(fi_df.head(20).iterrows()):
    print(f"| {i+1} | {r['feature']} | {r['importance']:.4f} |")

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
# 10. 최종 요약
# ============================================================
total_time = time.time() - start_all
print(f"\n{'='*60}")
print(f"## 최종 요약")
print(f"{'='*60}")
print(f"- 모델: XGBoost (GPU, tree_method=gpu_hist)")
print(f"- 피처 수: {len(feature_names)}")
print(f"- **OOF AUC: {oof_auc:.4f}**")
print(f"- OOF F1: {oof_f1:.4f}")
print(f"- scale_pos_weight: {xgb_params['scale_pos_weight']}")
print(f"- Data Leakage: 없음")
print(f"  - 횟수/나이: 고정 매핑 (데이터 무관)")
print(f"  - Target Encoding: K-Fold OOF (train만 사용)")
print(f"  - 결측: 수치→0, 범주→Unknown (행 독립)")
print(f"  - test: train 기준 TE 매핑만 적용")
print(f"- 총 소요시간: {total_time/60:.1f}분")
print(f"- 로그: {LOG_PATH}")
print(f"{'='*60}")
