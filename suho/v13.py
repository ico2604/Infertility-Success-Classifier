#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v13 - v12 기반 + 비이식 클리핑 후처리 (Conditional Probability Post-processing)
=============================================================================
핵심 변경:
  - 모델 학습은 v12와 동일 (전체 데이터, 전체 피처)
  - 제출 직전에 test의 '이식된 배아 수 == 0' 행의 예측값을 0.001로 클리핑
  - train OOF에도 동일 클리핑 적용하여 OOF AUC 비교
"""

import os
import time
import datetime
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    log_loss,
    average_precision_score,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
)
from catboost import CatBoostClassifier, Pool

warnings.filterwarnings("ignore")

# ============================================================
# 설정
# ============================================================
VERSION = "v13"
NOW = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
RESULT_DIR = "result"
LOG_PATH = os.path.join(RESULT_DIR, f"log_{VERSION}.md")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
RESULT_DIR = os.path.join(BASE_DIR, "result")
os.makedirs(RESULT_DIR, exist_ok=True)

SEEDS = [42, 2026, 2604]
N_FOLDS = 5
CLIP_PROB = 0.001  # 비이식 그룹 클리핑 확률

log_lines = []


def log(msg=""):
    print(msg)
    log_lines.append(msg)


def save_log():
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))


# ============================================================
log(f"# {VERSION} - 비이식 클리핑 후처리 (Conditional Probability Post-processing)")
log(f"시각: {NOW}")
log("=" * 60)

# ============================================================
# [1] 데이터 로드
# ============================================================
log("\n## [1] 데이터 로드")
t0 = time.time()

train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
sub = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

target_col = "임신 성공 여부"
y = train[target_col].values
train_id = train.iloc[:, 0]
test_id = test.iloc[:, 0]

log(f"- train: {train.shape}, test: {test.shape}")
log(f"- 타겟: 0={np.sum(y==0)}, 1={np.sum(y==1)}, 양성비율={np.mean(y)*100:.1f}%")

# ============================================================
# [2] CatBoost 전처리 (v1/v8 동일 – 원본 카테고리 20개 유지)
# ============================================================
log("\n## [2] CatBoost 전처리 (원본 카테고리 유지)")


def preprocess_catboost(df, is_train=True):
    """CatBoost 네이티브 카테고리 유지 전처리"""
    df = df.copy()

    # ID, 타겟 제거
    drop_cols = [df.columns[0]]
    if is_train and target_col in df.columns:
        drop_cols.append(target_col)
    df = df.drop(columns=drop_cols, errors="ignore")

    # 카테고리 vs 수치 분리
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    num_cols = df.select_dtypes(exclude=["object"]).columns.tolist()

    # 카테고리: NaN → "Unknown"
    for c in cat_cols:
        df[c] = df[c].fillna("Unknown").astype(str)

    # 수치: NaN → 0
    for c in num_cols:
        df[c] = df[c].fillna(0)

    # 파생변수 (v8 핵심: total_embryo_ratio)
    embryo_col = "이식된 배아 수"
    total_col = "총 생성 배아 수"
    if embryo_col in df.columns and total_col in df.columns:
        df["total_embryo_ratio"] = df[embryo_col] / (df[total_col] + 1)

    # 실제이식여부
    if embryo_col in df.columns:
        df["실제이식여부"] = (df[embryo_col] > 0).astype(int)

    # 수정 성공률, 배아 이용률, 배아 잉여율, 난자_배아_전환율
    fresh_egg_col = "수집된 신선 난자 수"
    stored_col = "저장된 배아 수"
    mixed_col = "혼합된 난자 수"

    if total_col in df.columns and fresh_egg_col in df.columns:
        df["수정_성공률"] = df[total_col] / (df[fresh_egg_col] + 1)
        df["난자_배아_전환율"] = df[total_col] / (
            df[fresh_egg_col] + df.get(mixed_col, pd.Series(0, index=df.index)) + 1
        )

    if embryo_col in df.columns and total_col in df.columns:
        df["배아_이용률"] = df[embryo_col] / (df[total_col] + 1)

    if stored_col in df.columns and total_col in df.columns:
        df["배아_잉여율"] = df[stored_col] / (df[total_col] + 1)

    # IVF/DI 분기 피처 (v9)
    type_col = "시술 유형"
    if type_col in df.columns:
        df["is_ivf"] = (df[type_col] == "IVF").astype(int)
        df["is_di"] = (df[type_col] == "DI").astype(int)

        ivf_mask = df["is_ivf"] == 1

        # IVF 전용 비율
        if embryo_col in df.columns and total_col in df.columns:
            df["ivf_transfer_ratio"] = 0.0
            df.loc[ivf_mask, "ivf_transfer_ratio"] = df.loc[ivf_mask, embryo_col] / (
                df.loc[ivf_mask, total_col] + 1
            )

        if stored_col in df.columns and total_col in df.columns:
            df["ivf_storage_ratio"] = 0.0
            df.loc[ivf_mask, "ivf_storage_ratio"] = df.loc[ivf_mask, stored_col] / (
                df.loc[ivf_mask, total_col] + 1
            )

    # 카테고리 재확인
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    return df, cat_cols


log("  전처리 중...")
X_train_cb, cat_features_names = preprocess_catboost(train, is_train=True)
X_test_cb, _ = preprocess_catboost(test, is_train=True)

# cat_features 인덱스
cat_features_idx = [X_train_cb.columns.get_loc(c) for c in cat_features_names]
feature_names_cb = X_train_cb.columns.tolist()

log(f"- CB 피처: {len(feature_names_cb)}개, 카테고리: {len(cat_features_idx)}개")
log(f"- 카테고리: {cat_features_names}")

# ============================================================
# [3] 비이식 그룹 식별 (train & test)
# ============================================================
log("\n## [3] 비이식 그룹 식별")

embryo_col_raw = "이식된 배아 수"

# train 원본에서 이식 여부 확인
train_zero_mask = (train[embryo_col_raw].fillna(0) == 0).values
test_zero_mask = (test[embryo_col_raw].fillna(0) == 0).values

train_zero_count = train_zero_mask.sum()
test_zero_count = test_zero_mask.sum()
train_zero_positive = y[train_zero_mask].sum()
train_zero_rate = y[train_zero_mask].mean() if train_zero_count > 0 else 0

log(f"- train 비이식: {train_zero_count}건 ({train_zero_count/len(y)*100:.1f}%)")
log(f"  - 양성: {train_zero_positive}건, 양성률: {train_zero_rate*100:.2f}%")
log(f"- test 비이식: {test_zero_count}건 ({test_zero_count/len(test)*100:.1f}%)")
log(f"- 클리핑 확률: {CLIP_PROB}")

# ============================================================
# [4] CatBoost 3-seed 앙상블
# ============================================================
log("\n## [4] CatBoost 3-seed 앙상블")
log(f"### 파라미터: iterations=5000, lr=0.01, depth=8, l2=3")
log(f"### Seeds: {SEEDS}")

oof_cb = np.zeros(len(y))
test_cb = np.zeros(len(test))
seed_cb_aucs = []
cb_importances = None

total_models = len(SEEDS) * N_FOLDS
model_count = 0

for si, seed_val in enumerate(SEEDS):
    log(f"\n  --- Seed {seed_val} ({si+1}/{len(SEEDS)}) ---")
    oof_seed = np.zeros(len(y))
    test_seed = np.zeros(len(test))

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed_val)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train_cb, y)):
        model_count += 1
        t_fold = time.time()

        X_tr = X_train_cb.iloc[tr_idx]
        X_va = X_train_cb.iloc[va_idx]
        y_tr = y[tr_idx]
        y_va = y[va_idx]

        cb_model = CatBoostClassifier(
            iterations=5000,
            learning_rate=0.01,
            depth=8,
            l2_leaf_reg=3,
            min_data_in_leaf=20,
            eval_metric="AUC",
            random_seed=seed_val + fold,
            early_stopping_rounds=200,
            verbose=0,
            task_type="GPU",
            devices="0",
        )

        cb_model.fit(
            X_tr,
            y_tr,
            eval_set=(X_va, y_va),
            cat_features=cat_features_idx,
            verbose=0,
        )

        va_pred = cb_model.predict_proba(X_va)[:, 1]
        oof_seed[va_idx] = va_pred
        test_seed += cb_model.predict_proba(X_test_cb)[:, 1] / N_FOLDS

        va_auc = roc_auc_score(y_va, va_pred)
        elapsed = (time.time() - t_fold) / 60

        log(
            f"    Fold {fold+1}: AUC={va_auc:.4f}, iter={cb_model.best_iteration_}, "
            f"소요={elapsed:.1f}분 [{model_count}/{total_models}]"
        )

        # 마지막 시드의 마지막 폴드에서 중요도 저장
        if si == len(SEEDS) - 1 and fold == N_FOLDS - 1:
            cb_importances = cb_model.get_feature_importance()

    seed_auc = roc_auc_score(y, oof_seed)
    seed_cb_aucs.append(seed_auc)
    log(f"  Seed {seed_val} OOF AUC: {seed_auc:.4f}")

    oof_cb += oof_seed / len(SEEDS)
    test_cb += test_seed / len(SEEDS)

cb_auc_raw = roc_auc_score(y, oof_cb)
log(f"\n  === CatBoost 3-seed OOF AUC (클리핑 전): {cb_auc_raw:.4f} ===")
log(f"  개별 seed: {[f'{a:.4f}' for a in seed_cb_aucs]}")

# ============================================================
# [5] 비이식 클리핑 적용
# ============================================================
log("\n## [5] 비이식 클리핑 후처리 적용")

# --- OOF 클리핑 ---
oof_cb_clipped = oof_cb.copy()
oof_cb_clipped[train_zero_mask] = np.minimum(oof_cb_clipped[train_zero_mask], CLIP_PROB)

cb_auc_clipped = roc_auc_score(y, oof_cb_clipped)
log(f"- OOF AUC (클리핑 전): {cb_auc_raw:.6f}")
log(f"- OOF AUC (클리핑 후): {cb_auc_clipped:.6f}")
log(f"- 차이: {cb_auc_clipped - cb_auc_raw:+.6f}")

# --- OOF 클리핑 전후 비이식 그룹 예측값 통계 ---
log(f"\n### 비이식 그룹 예측값 변화 (train)")
log(
    f"  클리핑 전: mean={oof_cb[train_zero_mask].mean():.6f}, "
    f"max={oof_cb[train_zero_mask].max():.6f}, "
    f"min={oof_cb[train_zero_mask].min():.6f}"
)
log(
    f"  클리핑 후: mean={oof_cb_clipped[train_zero_mask].mean():.6f}, "
    f"max={oof_cb_clipped[train_zero_mask].max():.6f}, "
    f"min={oof_cb_clipped[train_zero_mask].min():.6f}"
)

# --- Test 클리핑 ---
test_cb_clipped = test_cb.copy()
test_cb_clipped[test_zero_mask] = np.minimum(test_cb_clipped[test_zero_mask], CLIP_PROB)

log(f"\n### 비이식 그룹 예측값 변화 (test)")
log(
    f"  클리핑 전: mean={test_cb[test_zero_mask].mean():.6f}, "
    f"max={test_cb[test_zero_mask].max():.6f}"
)
log(
    f"  클리핑 후: mean={test_cb_clipped[test_zero_mask].mean():.6f}, "
    f"max={test_cb_clipped[test_zero_mask].max():.6f}"
)

# ============================================================
# [6] 다양한 클리핑 값 비교
# ============================================================
log("\n## [6] 클리핑 값 민감도 분석")

clip_values = [0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01]
log(f"  {'클리핑값':>10}  {'OOF AUC':>12}  {'Δ vs 원본':>12}")
log(f"  {'-'*10}  {'-'*12}  {'-'*12}")

best_clip = None
best_clip_auc = 0

for cv in clip_values:
    oof_temp = oof_cb.copy()
    oof_temp[train_zero_mask] = np.minimum(oof_temp[train_zero_mask], cv)
    temp_auc = roc_auc_score(y, oof_temp)
    delta = temp_auc - cb_auc_raw
    log(f"  {cv:>10.4f}  {temp_auc:>12.6f}  {delta:>+12.6f}")
    if temp_auc > best_clip_auc:
        best_clip_auc = temp_auc
        best_clip = cv

log(f"\n  최적 클리핑값: {best_clip} (AUC={best_clip_auc:.6f})")

# 최적값으로 재클리핑
oof_final = oof_cb.copy()
oof_final[train_zero_mask] = np.minimum(oof_final[train_zero_mask], best_clip)

test_final = test_cb.copy()
test_final[test_zero_mask] = np.minimum(test_final[test_zero_mask], best_clip)

final_auc = roc_auc_score(y, oof_final)

# ============================================================
# [7] 종합 평가지표
# ============================================================
log("\n## [7] 종합 평가지표")

oof_ll = log_loss(y, oof_final)
oof_ap = average_precision_score(y, oof_final)

log(f"\n### 확률 기반 지표")
log(f"  OOF AUC:      {final_auc:.6f}")
log(f"  OOF Log Loss: {oof_ll:.6f}")
log(f"  OOF AP:       {oof_ap:.6f}")

# Threshold별 분류 지표
log(f"\n### Threshold별 분류 지표")
log(
    f"  {'Threshold':>10}  {'Accuracy':>10}  {'Precision':>10}  {'Recall':>10}  {'F1':>10}  {'Specificity':>12}"
)
log(f"  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*12}")

best_f1 = 0
best_f1_th = 0.25
for th in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
    y_pred_bin = (oof_final >= th).astype(int)
    acc = accuracy_score(y, y_pred_bin)
    prec = precision_score(y, y_pred_bin, zero_division=0)
    rec = recall_score(y, y_pred_bin, zero_division=0)
    f1 = f1_score(y, y_pred_bin, zero_division=0)
    spec = np.sum((y == 0) & (y_pred_bin == 0)) / np.sum(y == 0)
    log(
        f"  {th:>10.2f}  {acc:>10.4f}  {prec:>10.4f}  {rec:>10.4f}  {f1:>10.4f}  {spec:>12.4f}"
    )
    if f1 > best_f1:
        best_f1 = f1
        best_f1_th = th

log(f"\n  최적 F1: {best_f1:.4f} (threshold={best_f1_th})")

# ============================================================
# [8] 이식 그룹 vs 비이식 그룹 별도 AUC
# ============================================================
log("\n## [8] 그룹별 AUC 분석")

# 이식 그룹
implant_mask = ~train_zero_mask
if implant_mask.sum() > 0 and y[implant_mask].sum() > 0:
    implant_auc = roc_auc_score(y[implant_mask], oof_final[implant_mask])
    log(
        f"- 이식 그룹 ({implant_mask.sum()}건): AUC={implant_auc:.6f}, 양성률={y[implant_mask].mean()*100:.1f}%"
    )

# 비이식 그룹
if train_zero_count > 0 and y[train_zero_mask].sum() > 0:
    zero_auc = roc_auc_score(y[train_zero_mask], oof_final[train_zero_mask])
    log(
        f"- 비이식 그룹 ({train_zero_count}건): AUC={zero_auc:.6f}, 양성률={train_zero_rate*100:.2f}%"
    )
else:
    log(f"- 비이식 그룹: 양성이 {y[train_zero_mask].sum()}건으로 AUC 계산 불가")

# ============================================================
# [9] 제출 파일 생성
# ============================================================
log("\n## [9] 제출 파일 생성")

# 메인 제출 (클리핑 적용)
sub_main = sub.copy()
sub_main["probability"] = test_final
main_path = os.path.join(RESULT_DIR, f"sample_submission_{VERSION}_{NOW}.csv")
sub_main.to_csv(main_path, index=False)

# 클리핑 없는 버전도 저장 (비교용)
sub_raw = sub.copy()
sub_raw["probability"] = test_cb
raw_path = os.path.join(RESULT_DIR, f"sample_submission_{VERSION}_raw_{NOW}.csv")
sub_raw.to_csv(raw_path, index=False)

log(f"- 메인 (클리핑): {main_path}")
log(
    f"  확률: mean={test_final.mean():.4f}, std={test_final.std():.4f}, "
    f"min={test_final.min():.6f}, max={test_final.max():.4f}"
)
log(f"- 비교용 (원본): {raw_path}")
log(
    f"  확률: mean={test_cb.mean():.4f}, std={test_cb.std():.4f}, "
    f"min={test_cb.min():.6f}, max={test_cb.max():.4f}"
)

log(f"\n- 예시:")
log(f"  {'ID':>10}  {'클리핑':>12}  {'원본':>12}")
for i in range(5):
    log(f"  {sub_main.iloc[i, 0]:>10}  {test_final[i]:>12.6f}  {test_cb[i]:>12.6f}")

# ============================================================
# [10] 피처 중요도
# ============================================================
log("\n## [10] CatBoost 피처 중요도 (상위 30)")

if cb_importances is not None:
    imp_df = pd.DataFrame(
        {"feature": feature_names_cb, "importance": cb_importances}
    ).sort_values("importance", ascending=False)

    cat_set = set(cat_features_names)
    for i, row in imp_df.head(30).iterrows():
        rank = imp_df.index.get_loc(i) + 1 if i == imp_df.head(30).index[0] else None
        cat_mark = " [cat]" if row["feature"] in cat_set else ""
        log(
            f"  {imp_df.head(30).values.tolist().index([row['feature'], row['importance']])+1:>3}. "
            f"{row['feature']}{cat_mark}: {row['importance']:.2f}"
        )

    # 재정렬 출력
    log("")
    for rank_i, (_, row) in enumerate(imp_df.head(30).iterrows(), 1):
        cat_mark = " [cat]" if row["feature"] in cat_set else ""
        log(f"  {rank_i}. {row['feature']}{cat_mark}: {row['importance']:.2f}")

# ============================================================
# [11] 버전 비교
# ============================================================
log("\n## [11] 버전 비교")
log("| 버전 | 모델 | OOF AUC | 비고 |")
log("|------|------|---------|------|")
log("| v1 | CB원본 | 0.7403 | 베이스라인 (67피처) |")
log("| v4 | XGB | 0.7392 | 컬럼복원+이식플래그 |")
log("| v7 | XGB+CB blend | 0.7402 | 블렌딩 |")
log("| v8 | XGB+CB blend | 0.7401 | +파생변수 |")
log("| v8-seed | CB 3seed | 0.7403 | seed앙상블 |")
log("| v9 | XGB+CB 3seed | 0.7405 | +IVF/DI분기 |")
log("| v10 | XGB+CB 3seed | 0.7403 | +의료파생(과다) |")
log("| v11 | CB+XGB cross | 0.7405 | 크로스블렌딩 |")
log("| v12 | CB+XGB+TabNet | 0.7406 | 3모델블렌딩 |")
log(f"| v13-raw | CB 3seed | {cb_auc_raw:.4f} | 클리핑 전 |")
log(
    f"| **v13** | **CB 3seed+클리핑** | **{final_auc:.4f}** | **비이식 clip={best_clip}** |"
)

# ============================================================
# [12] 최종 요약
# ============================================================
total_time = time.time() - t0
log(f"\n{'='*60}")
log(f"## 최종 요약")
log(f"{'='*60}")
log(f"- v13 핵심 변경: 비이식 그룹(이식배아수=0) 예측값을 {best_clip}로 클리핑")
log(f"- 모델: CatBoost 3-seed 앙상블 (학습은 전체 데이터, 전체 피처)")
log(f"- 학습 파라미터: iterations=5000, lr=0.01, depth=8, l2=3")
log(f"- CatBoost 3-seed OOF AUC (클리핑 전): {cb_auc_raw:.6f}")
log(f"- CatBoost 3-seed OOF AUC (클리핑 후): {final_auc:.6f}")
log(f"- 클리핑 효과: {final_auc - cb_auc_raw:+.6f}")
log(f"- 비이식 그룹: train {train_zero_count}건, test {test_zero_count}건")
log(f"- 최적 클리핑값: {best_clip}")
log(f"- OOF LogLoss: {oof_ll:.6f}")
log(f"- OOF AP: {oof_ap:.6f}")
log(f"- 최적 F1: {best_f1:.4f} (th={best_f1_th})")
log(f"- 제출 파일: {main_path}")
log(f"- 데이터 누수: 없음")
log(f"  - 클리핑은 행 단위 (이식배아수 == 0 여부)로 적용, 타겟 정보 미사용")
log(f"  - CatBoost: K-Fold OOF 학습, test에는 예측만 수행")
log(f"- 소요: {total_time/60:.1f}분")
log(f"- 로그: {LOG_PATH}")
log(f"{'='*60}")

save_log()
print(f"\n완료! 로그: {LOG_PATH}")
