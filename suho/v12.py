#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v12.py - v11 기반 + TabNet 추가 → 3모델 크로스 블렌딩
  - CatBoost 3-seed (v11 Pipeline A 동일)
  - XGBoost 팀원 스타일 (v11 Pipeline B 동일)
  - TabNet (신경망 기반, 트리와 다른 학습 방식)
  - 3모델 블렌딩으로 다양성 극대화
"""

import os, sys, warnings, time, re
import numpy as np, pandas as pd
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    log_loss,
    confusion_matrix,
)
from xgboost import XGBClassifier
from catboost import CatBoostClassifier, Pool
from pytorch_tabnet.tab_model import TabNetClassifier
import torch

warnings.filterwarnings("ignore")

VERSION = "v12"
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
print(f"# {VERSION} - CB 3seed + XGB 팀원 + TabNet → 3모델 블렌딩")
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


def safe_div(a, b, fill=0.0):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.where(b > 0, a / b, fill)


def safe_div_nan(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.where((np.isnan(b)) | (b == 0), np.nan, a / b)


# ############################################################
# PIPELINE A: CatBoost (v11 동일)
# ############################################################
print("\n" + "=" * 60)
print("## PIPELINE A: CatBoost (피처 축소, 원본 카테고리)")
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


def build_cb_minimal(df):
    out = df.copy()
    drop = [c for c in DROP_COLS_CB if c in out.columns]
    out = out.drop(columns=drop, errors="ignore")
    drop2 = [c for c in REMOVE_ZERO_IMP if c in out.columns]
    out = out.drop(columns=drop2, errors="ignore")

    for c in out.select_dtypes("object").columns:
        out[c] = out[c].fillna("_missing_")
    for c in out.select_dtypes("number").columns:
        out[c] = out[c].fillna(0)

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

    out["실제이식여부"] = (transferred > 0).astype(int)
    out["total_embryo_ratio"] = safe_div(transferred, embryo_total + 1)
    out["배아_이용률"] = safe_div(transferred, embryo_total)
    out["수정_성공률"] = safe_div(embryo_total, mixed)
    out["난자_배아_전환율"] = safe_div(embryo_total, fresh_egg)
    out["배아_잉여율"] = safe_div(embryo_total - transferred, embryo_total)

    if "시술 유형" in out.columns:
        proc = out["시술 유형"].astype(str).str.upper()
        is_ivf = proc.str.contains("IVF", na=False).values

        total_cycles_raw = out.get("총 시술 횟수", pd.Series(np.zeros(len(out))))
        total_cycles = pd.to_numeric(total_cycles_raw, errors="coerce").fillna(0).values
        embryo_days = (
            pd.to_numeric(out.get("배아 이식 경과일", 0), errors="coerce")
            .fillna(0)
            .values
        )

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

# CatBoost 3-seed 학습
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

        cb_model = CatBoostClassifier(
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
        cb_model.fit(tr_pool, eval_set=va_pool, use_best_model=True)
        oof_seed[va_i] = cb_model.predict_proba(va_pool)[:, 1]
        test_seed += cb_model.predict_proba(te_pool)[:, 1] / N_FOLDS
        auc = roc_auc_score(y[va_i], oof_seed[va_i])
        fold_aucs.append(auc)
        print(
            f"    Fold {fold}: AUC={auc:.4f}, iter={cb_model.best_iteration_}, 소요={( time.time()-fold_start)/60:.1f}분"
        )

    seed_auc = roc_auc_score(y, oof_seed)
    seed_cb_aucs.append(seed_auc)
    print(f"  Seed {seed_val} OOF AUC: {seed_auc:.4f}")
    oof_cb += oof_seed / len(SEEDS_CB)
    test_cb_pred += test_seed / len(SEEDS_CB)

cb_auc = roc_auc_score(y, oof_cb)
print(f"\n  === CatBoost 3-seed OOF AUC: {cb_auc:.4f} ===")


# ############################################################
# PIPELINE B: XGBoost 팀원 스타일 (v11 동일)
# ############################################################
print("\n" + "=" * 60)
print("## PIPELINE B: XGBoost (팀원 스타일)")
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


def make_medical_features(df):
    def has(*cols):
        return all(c in df.columns for c in cols)

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
    if "총 임신 횟수_num" in df.columns:
        df["임신경험있음"] = (df["총 임신 횟수_num"] > 0).astype(float)
    if "총 출산 횟수_num" in df.columns:
        df["출산경험있음"] = (df["총 출산 횟수_num"] > 0).astype(float)

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

    if has("시술 당시 나이_num", "총 시술 횟수_num"):
        df["나이X총시술"] = df["시술 당시 나이_num"] * df["총 시술 횟수_num"]
    if has("시술 당시 나이_num", "IVF실패횟수"):
        df["나이XIVF실패"] = df["시술 당시 나이_num"] * df["IVF실패횟수"]
    if has("초고령_여부", "반복IVF실패_여부"):
        df["초고령X반복실패"] = df["초고령_여부"] * df["반복IVF실패_여부"]

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


print("\n### XGBoost 전처리")
train_xgb = train_raw.copy()
test_xgb = test_raw.copy()

for c in COUNT_COLS:
    if c in train_xgb.columns:
        train_xgb[c + "_num"] = train_xgb[c].apply(parse_korean_count)
        test_xgb[c + "_num"] = test_xgb[c].apply(parse_korean_count)
for c in AGE_COLS:
    if c in train_xgb.columns:
        train_xgb[c + "_num"] = train_xgb[c].apply(age_to_midpoint)
        test_xgb[c + "_num"] = test_xgb[c].apply(age_to_midpoint)

train_xgb = make_medical_features(train_xgb)
test_xgb = make_medical_features(test_xgb)

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

drop_str = [c for c in COUNT_COLS + AGE_COLS if c in train_xgb.columns]
train_xgb = train_xgb.drop(columns=drop_str, errors="ignore")
test_xgb = test_xgb.drop(columns=drop_str, errors="ignore")

X_xgb = train_xgb.drop(columns=[TARGET, "ID"], errors="ignore")
X_test_xgb = test_xgb.drop(columns=["ID"], errors="ignore")
common_xgb = [c for c in X_xgb.columns if c in X_test_xgb.columns]
X_xgb = X_xgb[common_xgb]
X_test_xgb = X_test_xgb[common_xgb]
xgb_features = list(common_xgb)
print(f"- XGB 피처: {len(xgb_features)}개")

# XGBoost 학습
print(f"\n### XGBoost 5-Fold (depth=5, lr=0.02)")

oof_xgb = np.zeros(len(y))
test_xgb_pred = np.zeros(len(test_ids))
xgb_aucs = []

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

for fold, (tr_i, va_i) in enumerate(skf.split(X_xgb, y), 1):
    fold_start = time.time()
    print(f"\n  [XGB] Fold {fold}/{N_FOLDS} 학습 중...")

    xgb_model = XGBClassifier(
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
        scale_pos_weight=1.0,
        eval_metric="auc",
        tree_method="gpu_hist",
        random_state=SEED,
        verbosity=0,
        early_stopping_rounds=50,
    )
    xgb_model.fit(
        X_xgb.iloc[tr_i], y[tr_i], eval_set=[(X_xgb.iloc[va_i], y[va_i])], verbose=0
    )
    oof_xgb[va_i] = xgb_model.predict_proba(X_xgb.iloc[va_i])[:, 1]
    test_xgb_pred += xgb_model.predict_proba(X_test_xgb)[:, 1] / N_FOLDS
    auc = roc_auc_score(y[va_i], oof_xgb[va_i])
    xgb_aucs.append(auc)
    best_iter = (
        xgb_model.best_iteration if hasattr(xgb_model, "best_iteration") else "N/A"
    )
    print(
        f"  [XGB] Fold {fold}: AUC={auc:.4f}, iter={best_iter}, 소요={(time.time()-fold_start)/60:.1f}분"
    )

xgb_auc = roc_auc_score(y, oof_xgb)
print(f"\n  === XGBoost OOF AUC: {xgb_auc:.4f} ===")


# ############################################################
# PIPELINE C: TabNet (신경망)
# ############################################################
print("\n" + "=" * 60)
print("## PIPELINE C: TabNet (신경망 기반)")
print("=" * 60)

# TabNet 전처리: XGB 데이터 재사용 + StandardScaler
print("\n### TabNet 전처리 (XGB 피처 + 스케일링)")

X_tab = X_xgb.copy().fillna(0)
X_test_tab = X_test_xgb.copy().fillna(0)

# inf 처리
X_tab = X_tab.replace([np.inf, -np.inf], 0)
X_test_tab = X_test_tab.replace([np.inf, -np.inf], 0)

tab_features = list(X_tab.columns)
print(f"- TabNet 피처: {len(tab_features)}개")

# TabNet 학습
print(f"\n### TabNet 5-Fold 학습")
print(f"### 파라미터: n_d=32, n_a=32, n_steps=5, gamma=1.5, lr=0.02")

oof_tab = np.zeros(len(y))
test_tab_pred = np.zeros(len(test_ids))
tab_aucs = []

skf_tab = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

for fold, (tr_i, va_i) in enumerate(skf_tab.split(X_tab, y), 1):
    fold_start = time.time()
    print(f"\n  [TabNet] Fold {fold}/{N_FOLDS} 학습 중...")

    # fold별 스케일링 (train fold만으로 fit)
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tab.iloc[tr_i])
    X_va_scaled = scaler.transform(X_tab.iloc[va_i])
    X_te_scaled = scaler.transform(X_test_tab)

    tab_model = TabNetClassifier(
        n_d=32,
        n_a=32,
        n_steps=5,
        gamma=1.5,
        lambda_sparse=1e-4,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=0.02, weight_decay=1e-5),
        scheduler_fn=torch.optim.lr_scheduler.CosineAnnealingLR,
        scheduler_params=dict(T_max=50, eta_min=1e-4),
        mask_type="entmax",
        seed=SEED + fold,
        verbose=0,
        device_name="cuda" if torch.cuda.is_available() else "cpu",
    )

    tab_model.fit(
        X_train=X_tr_scaled,
        y_train=y[tr_i],
        eval_set=[(X_va_scaled, y[va_i])],
        eval_metric=["auc"],
        max_epochs=200,
        patience=30,
        batch_size=4096,
        virtual_batch_size=512,
        drop_last=False,
    )

    oof_tab[va_i] = tab_model.predict_proba(X_va_scaled)[:, 1]
    test_tab_pred += tab_model.predict_proba(X_te_scaled)[:, 1] / N_FOLDS

    auc = roc_auc_score(y[va_i], oof_tab[va_i])
    tab_aucs.append(auc)
    ft = time.time() - fold_start
    print(f"  [TabNet] Fold {fold}: AUC={auc:.4f}, 소요={ft/60:.1f}분")

tab_auc = roc_auc_score(y, oof_tab)
print(f"\n  === TabNet OOF AUC: {tab_auc:.4f} ===")


# ############################################################
# 3모델 블렌딩
# ############################################################
print("\n" + "=" * 60)
print("## [7] 3모델 블렌딩")
print("=" * 60)

# 상관계수 확인
corr_cb_xgb = np.corrcoef(oof_cb, oof_xgb)[0, 1]
corr_cb_tab = np.corrcoef(oof_cb, oof_tab)[0, 1]
corr_xgb_tab = np.corrcoef(oof_xgb, oof_tab)[0, 1]

print(f"\n### OOF 상관계수")
print(f"  CB ↔ XGB:    {corr_cb_xgb:.4f}")
print(f"  CB ↔ TabNet:  {corr_cb_tab:.4f}")
print(f"  XGB ↔ TabNet: {corr_xgb_tab:.4f}")

# 그리드 탐색: w_cb, w_xgb, w_tab (합=1)
print(f"\n### 가중치 그리드 탐색")

best_blend_auc = 0
best_weights = (0.33, 0.33, 0.34)

for w_cb in np.arange(0.0, 1.01, 0.05):
    for w_xgb in np.arange(0.0, 1.01 - w_cb, 0.05):
        w_tab = round(1.0 - w_cb - w_xgb, 2)
        if w_tab < 0:
            continue
        blend = w_cb * oof_cb + w_xgb * oof_xgb + w_tab * oof_tab
        auc_val = roc_auc_score(y, blend)
        if auc_val > best_blend_auc:
            best_blend_auc = auc_val
            best_weights = (round(w_cb, 2), round(w_xgb, 2), round(w_tab, 2))

# 미세조정
w_cb_best, w_xgb_best, w_tab_best = best_weights
for dw_cb in np.arange(-0.05, 0.06, 0.01):
    for dw_xgb in np.arange(-0.05, 0.06, 0.01):
        wc = round(w_cb_best + dw_cb, 2)
        wx = round(w_xgb_best + dw_xgb, 2)
        wt = round(1.0 - wc - wx, 2)
        if wc < 0 or wx < 0 or wt < 0:
            continue
        blend = wc * oof_cb + wx * oof_xgb + wt * oof_tab
        auc_val = roc_auc_score(y, blend)
        if auc_val > best_blend_auc:
            best_blend_auc = auc_val
            best_weights = (wc, wx, wt)

w_cb_f, w_xgb_f, w_tab_f = best_weights
print(f"  최적: CB={w_cb_f}, XGB={w_xgb_f}, TabNet={w_tab_f}")
print(f"  블렌딩 AUC: {best_blend_auc:.4f}")

# 후보 비교
candidates = {
    "CB 단독": (cb_auc, oof_cb, test_cb_pred),
    "XGB 단독": (xgb_auc, oof_xgb, test_xgb_pred),
    "TabNet 단독": (tab_auc, oof_tab, test_tab_pred),
    "CB+XGB": None,
    "3모델 블렌딩": (
        best_blend_auc,
        w_cb_f * oof_cb + w_xgb_f * oof_xgb + w_tab_f * oof_tab,
        w_cb_f * test_cb_pred + w_xgb_f * test_xgb_pred + w_tab_f * test_tab_pred,
    ),
}

# CB+XGB 2모델 블렌딩도 비교
best_2m_auc = 0
best_2m_w = 0.5
for w in np.arange(0.0, 1.01, 0.01):
    b = w * oof_xgb + (1 - w) * oof_cb
    a = roc_auc_score(y, b)
    if a > best_2m_auc:
        best_2m_auc = a
        best_2m_w = w
candidates["CB+XGB"] = (
    best_2m_auc,
    best_2m_w * oof_xgb + (1 - best_2m_w) * oof_cb,
    best_2m_w * test_xgb_pred + (1 - best_2m_w) * test_cb_pred,
)

best_name = max(candidates, key=lambda k: candidates[k][0])
final_oof_auc, oof_final, test_final = candidates[best_name]

print(f"\n### 후보 비교:")
for name, (auc, _, _) in candidates.items():
    tag = " ★" if name == best_name else ""
    print(f"  {name}: {auc:.4f}{tag}")
print(f"\n  → {best_name} 선택 (AUC={final_oof_auc:.4f})")


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
for name, arr in [("CB 3seed", oof_cb), ("XGB 팀원", oof_xgb), ("TabNet", oof_tab)]:
    ll = log_loss(y, arr)
    ap = average_precision_score(y, arr)
    auc = roc_auc_score(y, arr)
    print(f"  {name:>15} {auc:>10.4f} {ll:>10.4f} {ap:>10.4f}")
print(f"  {'최종':>15} {final_oof_auc:>10.4f} {oof_ll:>10.4f} {oof_ap:>10.4f}")


# ============================================================
# 9. 제출
# ============================================================
print(f"\n## [9] 제출 파일")

submission = pd.DataFrame({"ID": test_ids, "probability": test_final})
main_path = os.path.join(RESULT_DIR, f"sample_submission_{VERSION}_{NOW}.csv")
submission.to_csv(main_path, index=False)

# XGB 단독 제출도 생성 (팀원 스타일 test 일반화 확인용)
sub_xgb = pd.DataFrame({"ID": test_ids, "probability": test_xgb_pred})
xgb_path = os.path.join(RESULT_DIR, f"sample_submission_{VERSION}_xgb_only_{NOW}.csv")
sub_xgb.to_csv(xgb_path, index=False)

print(f"- 메인: {main_path}")
print(f"- XGB단독: {xgb_path}")
print(f"- 확률: mean={test_final.mean():.4f}, std={test_final.std():.4f}")
print(f"- 예시:")
print(submission.head(5).to_string(index=False))


# ============================================================
# 10. 피처 중요도
# ============================================================
print(f"\n## [10] 피처 중요도")

print(f"\n### XGBoost 상위 20")
fi = xgb_model.feature_importances_
fi_df = pd.DataFrame({"feature": xgb_features, "imp": fi}).sort_values(
    "imp", ascending=False
)
for i, (_, r) in enumerate(fi_df.head(20).iterrows()):
    print(f"  {i+1}. {r['feature']}: {r['imp']:.4f}")

print(f"\n### CatBoost 상위 20")
fi_cb_arr = cb_model.get_feature_importance()
fi_cb_df = pd.DataFrame({"feature": cb_features, "imp": fi_cb_arr}).sort_values(
    "imp", ascending=False
)
for i, (_, r) in enumerate(fi_cb_df.head(20).iterrows()):
    cat_tag = " [cat]" if r["feature"] in cat_cols_cb else ""
    print(f"  {i+1}. {r['feature']}{cat_tag}: {r['imp']:.2f}")

print(f"\n### TabNet Feature Importance 상위 20")
try:
    tab_imp = tab_model.feature_importances_
    tab_fi_df = pd.DataFrame({"feature": tab_features, "imp": tab_imp}).sort_values(
        "imp", ascending=False
    )
    for i, (_, r) in enumerate(tab_fi_df.head(20).iterrows()):
        print(f"  {i+1}. {r['feature']}: {r['imp']:.4f}")
except:
    print("  (TabNet feature importance 추출 불가)")


# ============================================================
# 11. 버전 비교
# ============================================================
print(f"\n## [11] 버전 비교")
print(f"| 버전 | 모델 | OOF AUC | LogLoss | AP | 비고 |")
print(f"|------|------|---------|---------|------|------|")
print(f"| v1 | CB원본 | 0.7403 | - | - | 베이스라인 |")
print(f"| v9 | XGB+CB 3seed | 0.7405 | - | - | +IVF/DI분기 |")
print(f"| v11 | CB+XGB 크로스 | 0.7405 | 0.4875 | 0.4516 | 팀원XGB |")
print(
    f"| v12-CB | CB 3seed | {cb_auc:.4f} | {log_loss(y,oof_cb):.4f} | {average_precision_score(y,oof_cb):.4f} | 피처축소 |"
)
print(
    f"| v12-XGB | XGB 팀원 | {xgb_auc:.4f} | {log_loss(y,oof_xgb):.4f} | {average_precision_score(y,oof_xgb):.4f} | depth=5 |"
)
print(
    f"| v12-Tab | TabNet | {tab_auc:.4f} | {log_loss(y,oof_tab):.4f} | {average_precision_score(y,oof_tab):.4f} | 신경망 |"
)
print(
    f"| v12 | {best_name} | {final_oof_auc:.4f} | {oof_ll:.4f} | {oof_ap:.4f} | 최종 |"
)


# ============================================================
# 12. 요약
# ============================================================
total_time = time.time() - start_all
print(f"\n{'='*60}")
print(f"## 최종 요약")
print(f"{'='*60}")
print(f"- v12 핵심: CatBoost + XGBoost + TabNet 3모델 블렌딩")
print(f"- CatBoost 3-seed: {cb_auc:.4f}")
print(f"- XGBoost 팀원: {xgb_auc:.4f}")
print(f"- TabNet: {tab_auc:.4f}")
print(
    f"- OOF 상관: CB↔XGB={corr_cb_xgb:.4f}, CB↔Tab={corr_cb_tab:.4f}, XGB↔Tab={corr_xgb_tab:.4f}"
)
print(f"- 블렌딩: CB={w_cb_f}, XGB={w_xgb_f}, Tab={w_tab_f}")
print(
    f"- **최종: {best_name}, OOF AUC={final_oof_auc:.4f}** (v11 대비 {final_oof_auc-0.7405:+.4f})"
)
print(f"- OOF LogLoss: {oof_ll:.4f}, AP: {oof_ap:.4f}")
print(f"- 최적 F1: {best_f1:.4f} (th={best_f1_th})")
print(f"- XGB 단독 제출 파일도 생성 (test 일반화 확인용)")
print(f"- 소요: {total_time/60:.1f}분")
print(f"- 로그: {LOG_PATH}")
print(f"{'='*60}")
