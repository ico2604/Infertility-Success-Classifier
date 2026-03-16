#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v11.py - 피처 정리(감소) + v9 CB 3seed × 팀원 XGB 크로스 블렌딩
  - CatBoost: 원본 카테고리 + 검증된 파생변수만 (피처 축소)
  - XGB: 팀원 스타일 (make_medical_features + factorize + depth=5)
  - 서로 다른 전처리 파이프라인으로 블렌딩 다양성 극대화
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
)
from xgboost import XGBClassifier
from catboost import CatBoostClassifier, Pool

warnings.filterwarnings("ignore")

VERSION = "v11"
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
print(f"# {VERSION} - 피처 정리 + CB 3seed × 팀원 XGB 크로스 블렌딩")
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


# ############################################################
# PIPELINE A: CatBoost (원본 카테고리 + 최소 검증 파생변수)
# ############################################################
print("\n" + "=" * 60)
print("## PIPELINE A: CatBoost (피처 축소, 원본 카테고리 유지)")
print("=" * 60)

DROP_COLS_CB = [
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


def safe_div(a, b, fill=0.0):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.where(b > 0, a / b, fill)


def build_cb_minimal(df):
    """CatBoost용: 원본 카테고리 유지 + v8~v10에서 검증된 파생변수만"""
    out = df.copy()

    # 불필요 컬럼 제거
    drop = [c for c in DROP_COLS_CB if c in out.columns]
    out = out.drop(columns=drop, errors="ignore")
    drop2 = [c for c in REMOVE_ZERO_IMP if c in out.columns]
    out = out.drop(columns=drop2, errors="ignore")

    # 결측 채움
    for c in out.select_dtypes("object").columns:
        out[c] = out[c].fillna("_missing_")
    for c in out.select_dtypes("number").columns:
        out[c] = out[c].fillna(0)

    # 횟수 변환 (숫자형만)
    count_map = {
        "0회": 0,
        "1회": 1,
        "2회": 2,
        "3회": 3,
        "4회": 4,
        "5회": 5,
        "6회 이상": 6,
    }
    for c in out.columns:
        if "횟수" in c and out[c].dtype == "object":
            out[c] = out[c].map(count_map).fillna(0).astype(int)

    # 수치 추출
    embryo_total = (
        pd.to_numeric(out.get("총 생성 배아 수", 0), errors="coerce").fillna(0).values
    )
    transferred = (
        pd.to_numeric(out.get("이식된 배아 수", 0), errors="coerce").fillna(0).values
    )
    mixed = (
        pd.to_numeric(out.get("혼합된 난자 수", 0), errors="coerce").fillna(0).values
    )
    stored = (
        pd.to_numeric(out.get("저장된 배아 수", 0), errors="coerce").fillna(0).values
    )
    fresh_egg = (
        pd.to_numeric(out.get("수집된 신선 난자 수", 0), errors="coerce")
        .fillna(0)
        .values
    )

    # 검증된 파생변수만 (v8~v10 상위 중요도)
    out["실제이식여부"] = (transferred > 0).astype(int)
    out["total_embryo_ratio"] = safe_div(transferred, embryo_total + 1)
    out["배아_이용률"] = safe_div(transferred, embryo_total)
    out["수정_성공률"] = safe_div(embryo_total, mixed)
    out["난자_배아_전환율"] = safe_div(embryo_total, fresh_egg)
    out["배아_잉여율"] = safe_div(embryo_total - transferred, embryo_total)

    # IVF/DI 분기 (v9에서 검증)
    if "시술 유형" in out.columns:
        proc = out["시술 유형"].astype(str).str.upper()
        is_ivf = proc.str.contains("IVF", na=False).values
        is_di = proc.str.contains("DI", na=False).values

        total_cycles_raw = out.get("총 시술 횟수", pd.Series(np.zeros(len(out))))
        total_cycles = pd.to_numeric(total_cycles_raw, errors="coerce").fillna(0).values
        embryo_days = (
            pd.to_numeric(out.get("배아 이식 경과일", 0), errors="coerce")
            .fillna(0)
            .values
        )

        # 나이 중간값 (CB용 별도 컬럼)
        age_mid_map = {
            "만 18-34세": 26.0,
            "만 35-37세": 36.0,
            "만 38-39세": 38.5,
            "만 40-42세": 41.0,
            "만 43-44세": 43.5,
            "만 45-50세": 47.5,
        }
        if "시술 당시 나이" in out.columns and out["시술 당시 나이"].dtype == "object":
            age_mid = out["시술 당시 나이"].map(age_mid_map).fillna(0).values
        else:
            age_mid = (
                pd.to_numeric(out.get("시술 당시 나이", 0), errors="coerce")
                .fillna(0)
                .values
            )

        storage_ratio = safe_div(stored, embryo_total)
        transfer_ratio = safe_div(transferred, embryo_total)

        out["ivf_storage_ratio"] = np.where(is_ivf, storage_ratio, 0)
        out["ivf_transfer_ratio"] = np.where(is_ivf, transfer_ratio, 0)
        out["ivf_embryo_age_signal"] = np.where(is_ivf, embryo_days * age_mid, 0)

    return out


print("\n### CatBoost 전처리")
train_cb = build_cb_minimal(train_raw)
test_cb = build_cb_minimal(test_raw)

# 카테고리 식별
cat_cols_cb = []
for c in train_cb.columns:
    if train_cb[c].dtype == "object":
        cat_cols_cb.append(c)
        train_cb[c] = train_cb[c].astype(str)
        test_cb[c] = test_cb[c].astype(str)

common_cb = sorted(set(train_cb.columns) & set(test_cb.columns))
train_cb = train_cb[common_cb]
test_cb = test_cb[common_cb]
cat_idx_cb = [common_cb.index(c) for c in cat_cols_cb if c in common_cb]
cb_features = list(common_cb)

print(f"- CB 피처: {len(cb_features)}개, 카테고리: {len(cat_idx_cb)}개")
print(f"- 카테고리: {[c for c in cat_cols_cb if c in common_cb]}")


# ============================================================
# CatBoost 3-seed 학습
# ============================================================
print(f"\n### CatBoost 3-seed 앙상블 (seed={SEEDS_CB})")
print(f"### 파라미터: iterations=5000, lr=0.01, depth=8, l2=3")

oof_cb = np.zeros(len(y))
test_cb_pred = np.zeros(len(test_ids))
seed_cb_aucs = []

for seed_idx, seed_val in enumerate(SEEDS_CB):
    print(f"\n  --- Seed {seed_val} ({seed_idx+1}/{len(SEEDS_CB)}) ---")

    oof_seed = np.zeros(len(y))
    test_seed = np.zeros(len(test_ids))
    fold_aucs = []

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed_val)

    for fold, (tr_i, va_i) in enumerate(skf.split(train_cb, y), 1):
        fold_start = time.time()

        tr_pool = Pool(train_cb.iloc[tr_i], y[tr_i], cat_features=cat_idx_cb)
        va_pool = Pool(train_cb.iloc[va_i], y[va_i], cat_features=cat_idx_cb)
        te_pool = Pool(test_cb, cat_features=cat_idx_cb)

        model = CatBoostClassifier(
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
        model.fit(tr_pool, eval_set=va_pool, use_best_model=True)

        oof_seed[va_i] = model.predict_proba(va_pool)[:, 1]
        test_seed += model.predict_proba(te_pool)[:, 1] / N_FOLDS

        auc = roc_auc_score(y[va_i], oof_seed[va_i])
        fold_aucs.append(auc)
        ft = time.time() - fold_start
        print(
            f"    Fold {fold}: AUC={auc:.4f}, iter={model.best_iteration_}, 소요={ft/60:.1f}분"
        )

    seed_auc = roc_auc_score(y, oof_seed)
    seed_cb_aucs.append(seed_auc)
    print(f"  Seed {seed_val} OOF AUC: {seed_auc:.4f}")

    oof_cb += oof_seed / len(SEEDS_CB)
    test_cb_pred += test_seed / len(SEEDS_CB)

cb_auc = roc_auc_score(y, oof_cb)
print(f"\n  === CatBoost 3-seed OOF AUC: {cb_auc:.4f} ===")
print(f"  seed별: {[f'{a:.4f}' for a in seed_cb_aucs]}")


# ############################################################
# PIPELINE B: XGBoost (팀원 스타일 - make_medical_features + factorize)
# ############################################################
print("\n" + "=" * 60)
print("## PIPELINE B: XGBoost (팀원 스타일 전처리)")
print("=" * 60)

COUNT_COLS = [
    "총 시술 횟수",
    "클리닉 내 총 시술 횟수",
    "IVF 시술 횟수",
    "DI 시술 횟수",
    "총 임신 횟수",
    "IVF 임신 횟수",
    "DI 임신 횟수",
    "총 출산 횟수",
    "IVF 출산 횟수",
    "DI 출산 횟수",
]
AGE_COLS = ["시술 당시 나이", "난자 기증자 나이", "정자 기증자 나이"]


def parse_korean_count(x):
    if pd.isna(x):
        return np.nan
    x = str(x).strip()
    if re.fullmatch(r"\d+", x):
        return float(x)
    m = re.search(r"(\d+)", x)
    return float(m.group(1)) if m else np.nan


def age_to_midpoint(x):
    if pd.isna(x):
        return np.nan
    x = str(x).strip()
    if x in ["알 수 없음", "미상", "Unknown"]:
        return np.nan
    nums = re.findall(r"\d+", x)
    if len(nums) >= 2:
        return (float(nums[0]) + float(nums[1])) / 2
    if len(nums) == 1:
        return float(nums[0])
    return np.nan


def safe_div_nan(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.where((np.isnan(b)) | (b == 0), np.nan, a / b)


def make_medical_features(df):
    """팀원 스타일 의료 파생변수"""

    def has(*cols):
        return all(c in df.columns for c in cols)

    # 배아 효율성
    if has("총 생성 배아 수", "수집된 신선 난자 수"):
        df["배아생성효율"] = safe_div_nan(
            df["총 생성 배아 수"], df["수집된 신선 난자 수"]
        )
    if has("미세주입에서 생성된 배아 수", "미세주입된 난자 수"):
        df["ICSI수정효율"] = safe_div_nan(
            df["미세주입에서 생성된 배아 수"], df["미세주입된 난자 수"]
        )
    if has("이식된 배아 수", "총 생성 배아 수"):
        df["배아이식비율"] = safe_div_nan(df["이식된 배아 수"], df["총 생성 배아 수"])
    if has("저장된 배아 수", "총 생성 배아 수"):
        df["배아저장비율"] = safe_div_nan(df["저장된 배아 수"], df["총 생성 배아 수"])
    if has("미세주입된 난자 수", "수집된 신선 난자 수"):
        df["난자활용률"] = safe_div_nan(
            df["미세주입된 난자 수"], df["수집된 신선 난자 수"]
        )
    if has("이식된 배아 수", "수집된 신선 난자 수"):
        df["난자대비이식배아수"] = safe_div_nan(
            df["이식된 배아 수"], df["수집된 신선 난자 수"]
        )
    if "이식된 배아 수" in df.columns:
        df["이식배아수_구간"] = pd.cut(
            df["이식된 배아 수"], bins=[-1, 0, 1, 2, 100], labels=[0, 1, 2, 3]
        )
        df["이식배아수_구간"] = df["이식배아수_구간"].astype(float)

    # 시술 이력
    if has("총 임신 횟수_num", "총 시술 횟수_num"):
        df["전체임신률"] = safe_div_nan(df["총 임신 횟수_num"], df["총 시술 횟수_num"])
    if has("IVF 임신 횟수_num", "IVF 시술 횟수_num"):
        df["IVF임신률"] = safe_div_nan(df["IVF 임신 횟수_num"], df["IVF 시술 횟수_num"])
    if has("총 출산 횟수_num", "총 임신 횟수_num"):
        df["임신유지율"] = safe_div_nan(df["총 출산 횟수_num"], df["총 임신 횟수_num"])
    if has("IVF 출산 횟수_num", "IVF 임신 횟수_num"):
        df["IVF임신유지율"] = safe_div_nan(
            df["IVF 출산 횟수_num"], df["IVF 임신 횟수_num"]
        )
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
        df["클리닉집중도"] = safe_div_nan(
            df["클리닉 내 총 시술 횟수_num"], df["총 시술 횟수_num"]
        )
    if has("IVF 시술 횟수_num", "총 시술 횟수_num"):
        df["IVF시술비율"] = safe_div_nan(
            df["IVF 시술 횟수_num"], df["총 시술 횟수_num"]
        )
    if "총 임신 횟수_num" in df.columns:
        df["임신경험있음"] = (df["총 임신 횟수_num"] > 0).astype(float)
    if "총 출산 횟수_num" in df.columns:
        df["출산경험있음"] = (df["총 출산 횟수_num"] > 0).astype(float)

    # 나이 기반
    if "시술 당시 나이_num" in df.columns:
        age = df["시술 당시 나이_num"]
        df["나이_제곱"] = age**2
        df["나이_임상구간"] = pd.cut(
            age, bins=[0, 35, 40, 45, 100], labels=[0, 1, 2, 3], right=False
        )
        df["나이_임상구간"] = df["나이_임상구간"].astype(float)
        df["고령_여부"] = (age >= 35).astype(float)
        df["초고령_여부"] = (age >= 40).astype(float)
        df["극고령_여부"] = (age >= 42).astype(float)

    # 상호작용
    if has("시술 당시 나이_num", "총 시술 횟수_num"):
        df["나이X총시술"] = df["시술 당시 나이_num"] * df["총 시술 횟수_num"]
    if has("시술 당시 나이_num", "IVF실패횟수"):
        df["나이XIVF실패"] = df["시술 당시 나이_num"] * df["IVF실패횟수"]
    if has("초고령_여부", "반복IVF실패_여부"):
        df["초고령X반복실패"] = df["초고령_여부"] * df["반복IVF실패_여부"]

    # 복합 위험도
    risk = []
    if "고령_여부" in df.columns:
        risk.append(df["고령_여부"].fillna(0))
    if "초고령_여부" in df.columns:
        risk.append(df["초고령_여부"].fillna(0))
    if "반복IVF실패_여부" in df.columns:
        risk.append(df["반복IVF실패_여부"].fillna(0))
    if "임신경험있음" in df.columns:
        risk.append(1 - df["임신경험있음"].fillna(1))
    if len(risk) >= 2:
        df["복합위험도점수"] = sum(risk)

    return df


print("\n### XGBoost 전처리 (팀원 스타일)")

train_xgb = train_raw.copy()
test_xgb = test_raw.copy()

# 횟수/나이 수치 변환
for c in COUNT_COLS:
    if c in train_xgb.columns:
        train_xgb[c + "_num"] = train_xgb[c].apply(parse_korean_count)
        test_xgb[c + "_num"] = test_xgb[c].apply(parse_korean_count)
for c in AGE_COLS:
    if c in train_xgb.columns:
        train_xgb[c + "_num"] = train_xgb[c].apply(age_to_midpoint)
        test_xgb[c + "_num"] = test_xgb[c].apply(age_to_midpoint)

# 의료 파생변수
train_xgb = make_medical_features(train_xgb)
test_xgb = make_medical_features(test_xgb)

new_feats = [c for c in train_xgb.columns if c not in train_raw.columns]
print(f"- 파생변수: {len(new_feats)}개")

# 범주형 factorize (팀원 스타일: train+test 합쳐서)
exclude = [TARGET, "ID"]
cat_cols_xgb = []
for c in train_xgb.columns:
    if c in exclude:
        continue
    if not (
        pd.api.types.is_numeric_dtype(train_xgb[c])
        or pd.api.types.is_bool_dtype(train_xgb[c])
    ):
        cat_cols_xgb.append(c)

print(f"- 범주형 인코딩: {len(cat_cols_xgb)}개")

for c in cat_cols_xgb:
    combined = pd.concat(
        [
            train_xgb[c].astype(str).fillna("MISSING"),
            test_xgb[c].astype(str).fillna("MISSING"),
        ]
    )
    codes, _ = pd.factorize(combined)
    train_xgb[c] = codes[: len(train_xgb)]
    test_xgb[c] = codes[len(train_xgb) :]

# 원본 문자열 컬럼 제거
drop_str = [c for c in COUNT_COLS + AGE_COLS if c in train_xgb.columns]
train_xgb = train_xgb.drop(columns=drop_str, errors="ignore")
test_xgb = test_xgb.drop(columns=drop_str, errors="ignore")

# X, X_test 구성
X_xgb = train_xgb.drop(columns=[TARGET, "ID"], errors="ignore")
X_test_xgb = test_xgb.drop(columns=["ID"], errors="ignore")

common_xgb = [c for c in X_xgb.columns if c in X_test_xgb.columns]
X_xgb = X_xgb[common_xgb]
X_test_xgb = X_test_xgb[common_xgb]

xgb_features = list(common_xgb)
print(f"- XGB 피처: {len(xgb_features)}개")

# ============================================================
# XGBoost 학습 (팀원 파라미터)
# ============================================================
print(f"\n### XGBoost 5-Fold (팀원 파라미터: depth=5, lr=0.02)")

xgb_params = {
    "n_estimators": 3000,
    "learning_rate": 0.02,
    "max_depth": 5,
    "min_child_weight": 3,
    "gamma": 0.0,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "colsample_bylevel": 0.7,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "scale_pos_weight": 1.0,
    "eval_metric": "auc",
    "tree_method": "gpu_hist",
    "random_state": SEED,
    "verbosity": 0,
    "early_stopping_rounds": 50,
}

print(
    f"### 파라미터: depth={xgb_params['max_depth']}, lr={xgb_params['learning_rate']}, "
    f"n_est={xgb_params['n_estimators']}, min_child={xgb_params['min_child_weight']}"
)

oof_xgb = np.zeros(len(y))
test_xgb_pred = np.zeros(len(test_ids))
xgb_aucs = []

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

for fold, (tr_i, va_i) in enumerate(skf.split(X_xgb, y), 1):
    fold_start = time.time()
    print(f"\n  [XGB] Fold {fold}/{N_FOLDS} 학습 중...")

    xgb_model = XGBClassifier(**xgb_params)
    xgb_model.fit(
        X_xgb.iloc[tr_i],
        y[tr_i],
        eval_set=[(X_xgb.iloc[va_i], y[va_i])],
        verbose=0,
    )
    oof_xgb[va_i] = xgb_model.predict_proba(X_xgb.iloc[va_i])[:, 1]
    test_xgb_pred += xgb_model.predict_proba(X_test_xgb)[:, 1] / N_FOLDS

    auc = roc_auc_score(y[va_i], oof_xgb[va_i])
    xgb_aucs.append(auc)
    best_iter = (
        xgb_model.best_iteration if hasattr(xgb_model, "best_iteration") else "N/A"
    )
    ft = time.time() - fold_start
    print(f"  [XGB] Fold {fold}: AUC={auc:.4f}, iter={best_iter}, 소요={ft/60:.1f}분")

xgb_auc = roc_auc_score(y, oof_xgb)
print(f"\n  === XGBoost OOF AUC: {xgb_auc:.4f} ===")


# ############################################################
# 크로스 블렌딩
# ############################################################
print("\n" + "=" * 60)
print("## [7] 크로스 블렌딩 (CB 3seed × 팀원 XGB)")
print("=" * 60)

best_w = 0.0
best_blend_auc = 0

for w in np.arange(0.0, 1.01, 0.05):
    blend = w * oof_xgb + (1 - w) * oof_cb
    auc_val = roc_auc_score(y, blend)
    if auc_val > best_blend_auc:
        best_blend_auc = auc_val
        best_w = w

# 미세조정
best_w_fine = best_w
best_auc_fine = best_blend_auc
for w in np.arange(max(0, best_w - 0.05), min(1.0, best_w + 0.06), 0.01):
    blend = w * oof_xgb + (1 - w) * oof_cb
    auc_val = roc_auc_score(y, blend)
    if auc_val > best_auc_fine:
        best_auc_fine = auc_val
        best_w_fine = w

best_w = best_w_fine
best_blend_auc = best_auc_fine
print(f"  최적: XGB={best_w:.2f}, CB={1-best_w:.2f}, AUC={best_blend_auc:.4f}")

# 최종 결정
candidates = {
    "CB 단독": (cb_auc, oof_cb, test_cb_pred),
    "XGB 단독": (xgb_auc, oof_xgb, test_xgb_pred),
    "블렌딩": (
        best_blend_auc,
        best_w * oof_xgb + (1 - best_w) * oof_cb,
        best_w * test_xgb_pred + (1 - best_w) * test_cb_pred,
    ),
}

best_name = max(candidates, key=lambda k: candidates[k][0])
final_oof_auc, oof_final, test_final = candidates[best_name]
print(f"  → {best_name} 선택 (AUC={final_oof_auc:.4f})")

for name, (auc, _, _) in candidates.items():
    tag = " ★" if name == best_name else ""
    print(f"    {name}: {auc:.4f}{tag}")


# ============================================================
# 8. 종합 평가지표
# ============================================================
print(f"\n## [8] 종합 평가지표")

oof_ll = log_loss(y, oof_final)
oof_ap = average_precision_score(y, oof_final)

print(f"\n### 확률 기반 지표")
print(f"  OOF AUC:      {final_oof_auc:.6f}")
print(f"  OOF Log Loss: {oof_ll:.6f}")
print(f"  OOF AP:       {oof_ap:.6f}")

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
    acc = (tp + tn) / len(y)
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

# 개별 모델
print(f"\n### 개별 모델 비교")
print(f"  {'모델':>15} {'AUC':>10} {'LogLoss':>10} {'AP':>10}")
print(f"  {'-'*50}")
cb_ll = log_loss(y, oof_cb)
cb_ap_val = average_precision_score(y, oof_cb)
xgb_ll = log_loss(y, oof_xgb)
xgb_ap_val = average_precision_score(y, oof_xgb)
print(f"  {'CB 3seed':>15} {cb_auc:>10.4f} {cb_ll:>10.4f} {cb_ap_val:>10.4f}")
print(f"  {'XGB 팀원':>15} {xgb_auc:>10.4f} {xgb_ll:>10.4f} {xgb_ap_val:>10.4f}")
print(f"  {'최종':>15} {final_oof_auc:>10.4f} {oof_ll:>10.4f} {oof_ap:>10.4f}")

# OOF 상관관계 (블렌딩 다양성 확인)
corr = np.corrcoef(oof_cb, oof_xgb)[0, 1]
print(f"\n  CB↔XGB OOF 상관계수: {corr:.4f}")
print(f"  (낮을수록 블렌딩 효과 큼, 0.95 이하가 이상적)")


# ============================================================
# 9. 제출
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
# 10. 피처 중요도
# ============================================================
print(f"\n## [10] 피처 중요도")

print(f"\n### XGBoost 상위 30 (팀원 스타일)")
fi = xgb_model.feature_importances_
fi_df = pd.DataFrame({"feature": xgb_features, "imp": fi}).sort_values(
    "imp", ascending=False
)
for i, (_, r) in enumerate(fi_df.head(30).iterrows()):
    print(f"  {i+1}. {r['feature']}: {r['imp']:.4f}")

print(f"\n### CatBoost 상위 30 (피처 축소)")
fi_cb_arr = model.get_feature_importance()
fi_cb_df = pd.DataFrame({"feature": cb_features, "imp": fi_cb_arr}).sort_values(
    "imp", ascending=False
)
for i, (_, r) in enumerate(fi_cb_df.head(30).iterrows()):
    cat_tag = " [cat]" if r["feature"] in cat_cols_cb else ""
    print(f"  {i+1}. {r['feature']}{cat_tag}: {r['imp']:.2f}")


# ============================================================
# 11. 버전 비교
# ============================================================
print(f"\n## [11] 버전 비교")
print(f"| 버전 | 모델 | OOF AUC | LogLoss | AP | 비고 |")
print(f"|------|------|---------|---------|------|------|")
print(f"| v1 | CB원본 | 0.7403 | - | - | 베이스라인 |")
print(f"| v8 | XGB+CB | 0.7401 | - | - | +파생변수 |")
print(f"| v9 | XGB+CB 3seed | 0.7405 | - | - | +IVF/DI분기 |")
print(f"| v10 | XGB+CB 3seed | 0.7403 | 0.4878 | 0.4514 | +의료파생(과다) |")
print(
    f"| v11-CB | CB 3seed | {cb_auc:.4f} | {cb_ll:.4f} | {cb_ap_val:.4f} | 피처축소 |"
)
print(
    f"| v11-XGB | XGB 팀원 | {xgb_auc:.4f} | {xgb_ll:.4f} | {xgb_ap_val:.4f} | 팀원파이프라인 |"
)
print(
    f"| v11 | {best_name} | {final_oof_auc:.4f} | {oof_ll:.4f} | {oof_ap:.4f} | 최종 |"
)


# ============================================================
# 12. 요약
# ============================================================
total_time = time.time() - start_all
print(f"\n{'='*60}")
print(f"## 최종 요약")
print(f"{'='*60}")
print(f"- v11 핵심 전략:")
print(f"  1. CatBoost: 피처 축소 (95→{len(cb_features)}개), 원본 카테고리 유지")
print(f"  2. XGBoost: 팀원 파이프라인 재현 (factorize + 의료 파생)")
print(f"  3. 서로 다른 전처리 파이프라인으로 크로스 블렌딩")
print(f"- CB 3-seed: {cb_auc:.4f} (seed별: {[f'{a:.4f}' for a in seed_cb_aucs]})")
print(f"- XGB 팀원: {xgb_auc:.4f}")
print(f"- CB↔XGB 상관: {corr:.4f}")
print(f"- 블렌딩: XGB={best_w:.2f}, CB={1-best_w:.2f}")
print(
    f"- **최종: {best_name}, OOF AUC={final_oof_auc:.4f}** (v9 대비 {final_oof_auc-0.7405:+.4f})"
)
print(f"- OOF LogLoss: {oof_ll:.4f}, AP: {oof_ap:.4f}")
print(f"- 최적 F1: {best_f1:.4f} (th={best_f1_th})")
print(f"- 소요: {total_time/60:.1f}분")
print(f"- 로그: {LOG_PATH}")
print(f"{'='*60}")
