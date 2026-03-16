#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v14 - 이식 그룹 전용 피처 강화 + CatBoost 3-seed 앙상블
======================================================================
v13 분석 결과:
  - 비이식 클리핑 → AUC 하락 (0.7405 → 0.7395), 폐기
  - 핵심 병목: 이식 그룹(213,516건) AUC = 0.6745 (전체 0.7405)
  - 비이식 그룹(42,835건) AUC = 0.909로 이미 충분

v14 핵심 변경:
  1. 이식 그룹 전용 피처 10개 추가 (비이식 그룹에서는 0)
     - transfer_day_optimal (이식 경과일 2~5일 최적 구간 플래그)
     - transfer_day_cat (이식 경과일 구간: 0/fresh_early/fresh_optimal/fresh_late/frozen)
     - embryo_day_interaction (이식 배아 수 × 이식 경과일)
     - fresh_transfer_ratio (신선 배아 이식 비율)
     - micro_transfer_quality (미세주입 배아 이식 품질)
     - single_good_embryo (단일 배아 + 최적 경과일 조합)
     - frozen_embryo_signal (동결 배아 사용 × 해동 배아 수 비율)
     - embryo_surplus_after_transfer (이식 후 잔여 배아 수)
     - transfer_intensity (이식 배아 / 수집 난자)
     - age_transfer_interaction (나이 × 이식 배아 수 상호작용)
  2. 클리핑 제거 (원본 확률 그대로 제출)
  3. CatBoost 3-seed 앙상블 유지
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
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

# ============================================================
# 설정
# ============================================================
VERSION = "v14"
NOW = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
RESULT_DIR = "result"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
RESULT_DIR = os.path.join(BASE_DIR, "result")
LOG_PATH = os.path.join(RESULT_DIR, f"log_{VERSION}.md")
os.makedirs(RESULT_DIR, exist_ok=True)

SEEDS = [42, 2026, 2604]
N_FOLDS = 5

log_lines = []


def log(msg=""):
    print(msg)
    log_lines.append(msg)


def save_log():
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))


# ============================================================
log(f"# {VERSION} - 이식 그룹 전용 피처 강화 + CatBoost 3-seed")
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
# [2] 전처리 + 이식 그룹 전용 피처
# ============================================================
log("\n## [2] CatBoost 전처리 + 이식 그룹 전용 피처")


def preprocess_catboost(df, is_train=True):
    """CatBoost 네이티브 카테고리 유지 + v14 이식 전용 피처"""
    df = df.copy()

    # --- ID / 타겟 제거 ---
    drop_cols = [df.columns[0]]
    if is_train and target_col in df.columns:
        drop_cols.append(target_col)
    df = df.drop(columns=drop_cols, errors="ignore")

    # --- 카테고리 / 수치 분리 ---
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    num_cols = df.select_dtypes(exclude=["object"]).columns.tolist()

    for c in cat_cols:
        df[c] = df[c].fillna("Unknown").astype(str)
    for c in num_cols:
        df[c] = df[c].fillna(0)

    # ========================================
    # 기존 파생변수 (v8/v9 계승)
    # ========================================
    embryo_col = "이식된 배아 수"
    total_col = "총 생성 배아 수"
    fresh_egg_col = "수집된 신선 난자 수"
    stored_col = "저장된 배아 수"
    mixed_col = "혼합된 난자 수"
    day_col = "배아 이식 경과일"
    micro_transfer_col = "미세주입 배아 이식 수"
    thawed_col = "해동된 배아 수"
    single_col = "단일 배아 이식 여부"
    frozen_use_col = "동결 배아 사용 여부"
    fresh_use_col = "신선 배아 사용 여부"
    age_col = "시술 당시 나이"
    type_col = "시술 유형"

    # total_embryo_ratio
    if embryo_col in df.columns and total_col in df.columns:
        df["total_embryo_ratio"] = df[embryo_col] / (df[total_col] + 1)

    # 실제이식여부
    if embryo_col in df.columns:
        df["실제이식여부"] = (df[embryo_col] > 0).astype(int)

    # 수정 성공률
    if total_col in df.columns and fresh_egg_col in df.columns:
        df["수정_성공률"] = df[total_col] / (df[fresh_egg_col] + 1)

    # 난자_배아_전환율
    if (
        total_col in df.columns
        and fresh_egg_col in df.columns
        and mixed_col in df.columns
    ):
        df["난자_배아_전환율"] = df[total_col] / (df[fresh_egg_col] + df[mixed_col] + 1)

    # 배아_이용률
    if embryo_col in df.columns and total_col in df.columns:
        df["배아_이용률"] = df[embryo_col] / (df[total_col] + 1)

    # 배아_잉여율
    if stored_col in df.columns and total_col in df.columns:
        df["배아_잉여율"] = df[stored_col] / (df[total_col] + 1)

    # IVF/DI 분기 피처 (v9)
    if type_col in df.columns:
        df["is_ivf"] = (df[type_col] == "IVF").astype(int)
        df["is_di"] = (df[type_col] == "DI").astype(int)
        ivf_mask = df["is_ivf"] == 1

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

    # ========================================
    # ★ v14 신규: 이식 그룹 전용 피처 10개
    # ========================================
    implanted = (
        df[embryo_col] > 0
        if embryo_col in df.columns
        else pd.Series(False, index=df.index)
    )

    # 1) transfer_day_optimal: 이식 경과일 2~5일 최적 구간 플래그
    if day_col in df.columns:
        df["transfer_day_optimal"] = 0
        df.loc[implanted, "transfer_day_optimal"] = (
            (df.loc[implanted, day_col] >= 2) & (df.loc[implanted, day_col] <= 5)
        ).astype(int)

    # 2) transfer_day_cat: 이식 경과일 구간화
    #    0: 비이식, 1: fresh_early(0~1일), 2: fresh_optimal(2~3일),
    #    3: fresh_late(4~5일), 4: frozen(6일+)
    if day_col in df.columns:
        df["transfer_day_cat"] = 0
        days = df.loc[implanted, day_col]
        df.loc[implanted & (df[day_col] <= 1), "transfer_day_cat"] = 1
        df.loc[
            implanted & (df[day_col] >= 2) & (df[day_col] <= 3), "transfer_day_cat"
        ] = 2
        df.loc[
            implanted & (df[day_col] >= 4) & (df[day_col] <= 5), "transfer_day_cat"
        ] = 3
        df.loc[implanted & (df[day_col] >= 6), "transfer_day_cat"] = 4

    # 3) embryo_day_interaction: 이식 배아 수 × 이식 경과일
    if embryo_col in df.columns and day_col in df.columns:
        df["embryo_day_interaction"] = 0.0
        df.loc[implanted, "embryo_day_interaction"] = (
            df.loc[implanted, embryo_col] * df.loc[implanted, day_col]
        )

    # 4) fresh_transfer_ratio: 신선 배아 이식 비율 (신선 사용 시 이식배아/총배아)
    if (
        fresh_use_col in df.columns
        and embryo_col in df.columns
        and total_col in df.columns
    ):
        df["fresh_transfer_ratio"] = 0.0
        fresh_implant = implanted & (df[fresh_use_col] == 1)
        df.loc[fresh_implant, "fresh_transfer_ratio"] = df.loc[
            fresh_implant, embryo_col
        ] / (df.loc[fresh_implant, total_col] + 1)

    # 5) micro_transfer_quality: 미세주입 배아 이식 품질
    if micro_transfer_col in df.columns and embryo_col in df.columns:
        df["micro_transfer_quality"] = 0.0
        df.loc[implanted, "micro_transfer_quality"] = df.loc[
            implanted, micro_transfer_col
        ] / (df.loc[implanted, embryo_col] + 0.001)

    # 6) single_good_embryo: 단일 배아 이식 + 최적 경과일 조합
    if single_col in df.columns and day_col in df.columns:
        df["single_good_embryo"] = 0
        df.loc[implanted, "single_good_embryo"] = (
            (df.loc[implanted, single_col] == 1)
            & (df.loc[implanted, day_col] >= 2)
            & (df.loc[implanted, day_col] <= 5)
        ).astype(int)

    # 7) frozen_embryo_signal: 동결 배아 사용 × (해동 배아 수 / 이식 배아 수)
    if (
        frozen_use_col in df.columns
        and thawed_col in df.columns
        and embryo_col in df.columns
    ):
        df["frozen_embryo_signal"] = 0.0
        frozen_implant = implanted & (df[frozen_use_col] == 1)
        df.loc[frozen_implant, "frozen_embryo_signal"] = df.loc[
            frozen_implant, thawed_col
        ] / (df.loc[frozen_implant, embryo_col] + 0.001)

    # 8) embryo_surplus_after_transfer: 이식 후 잔여 배아 수
    if (
        total_col in df.columns
        and embryo_col in df.columns
        and stored_col in df.columns
    ):
        df["embryo_surplus_after_transfer"] = 0.0
        df.loc[implanted, "embryo_surplus_after_transfer"] = (
            df.loc[implanted, total_col]
            - df.loc[implanted, embryo_col]
            - df.loc[implanted, stored_col]
        ).clip(lower=0)

    # 9) transfer_intensity: 이식 배아 / 수집 난자 (이식 강도)
    if embryo_col in df.columns and fresh_egg_col in df.columns:
        df["transfer_intensity"] = 0.0
        df.loc[implanted, "transfer_intensity"] = df.loc[implanted, embryo_col] / (
            df.loc[implanted, fresh_egg_col] + 1
        )

    # 10) age_transfer_interaction: 나이 구간 × 이식 배아 수
    if age_col in df.columns and embryo_col in df.columns:
        # 나이를 수치로 변환 (카테고리인 경우 중간값 매핑)
        age_map = {
            "18-34": 26,
            "35-37": 36,
            "38-39": 38.5,
            "40-42": 41,
            "43-44": 43.5,
            "45-50": 47.5,
        }
        if df[age_col].dtype == object:
            age_numeric = df[age_col].map(age_map).fillna(35)
        else:
            age_numeric = df[age_col]

        df["age_transfer_interaction"] = 0.0
        df.loc[implanted, "age_transfer_interaction"] = (
            age_numeric.loc[implanted] * df.loc[implanted, embryo_col]
        )

    # --- 카테고리 재확인 ---
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    return df, cat_cols


log("  전처리 중...")
X_train_cb, cat_features_names = preprocess_catboost(train, is_train=True)
X_test_cb, _ = preprocess_catboost(test, is_train=True)

cat_features_idx = [X_train_cb.columns.get_loc(c) for c in cat_features_names]
feature_names_cb = X_train_cb.columns.tolist()

# v14 신규 피처 목록
v14_new_features = [
    "transfer_day_optimal",
    "transfer_day_cat",
    "embryo_day_interaction",
    "fresh_transfer_ratio",
    "micro_transfer_quality",
    "single_good_embryo",
    "frozen_embryo_signal",
    "embryo_surplus_after_transfer",
    "transfer_intensity",
    "age_transfer_interaction",
]
v14_found = [f for f in v14_new_features if f in feature_names_cb]

log(f"- CB 피처: {len(feature_names_cb)}개, 카테고리: {len(cat_features_idx)}개")
log(f"- 카테고리: {cat_features_names}")
log(f"- v14 신규 피처 ({len(v14_found)}개): {v14_found}")

# 신규 피처 기초 통계
log(f"\n### v14 신규 피처 통계 (train)")
for feat in v14_found:
    vals = X_train_cb[feat]
    nonzero = (vals != 0).sum()
    log(
        f"  {feat}: mean={vals.mean():.4f}, std={vals.std():.4f}, "
        f"nonzero={nonzero} ({nonzero/len(vals)*100:.1f}%)"
    )

# ============================================================
# [3] 비이식/이식 그룹 확인
# ============================================================
log("\n## [3] 이식/비이식 그룹 확인")

embryo_col_raw = "이식된 배아 수"
train_zero_mask = (train[embryo_col_raw].fillna(0) == 0).values
train_impl_mask = ~train_zero_mask

log(
    f"- 이식 그룹: {train_impl_mask.sum()}건 ({train_impl_mask.sum()/len(y)*100:.1f}%), "
    f"양성률={y[train_impl_mask].mean()*100:.1f}%"
)
log(
    f"- 비이식 그룹: {train_zero_mask.sum()}건 ({train_zero_mask.sum()/len(y)*100:.1f}%), "
    f"양성률={y[train_zero_mask].mean()*100:.2f}%"
)

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

        # 마지막 시드 마지막 폴드 → 중요도
        if si == len(SEEDS) - 1 and fold == N_FOLDS - 1:
            cb_importances = cb_model.get_feature_importance()

    seed_auc = roc_auc_score(y, oof_seed)
    seed_cb_aucs.append(seed_auc)
    log(f"  Seed {seed_val} OOF AUC: {seed_auc:.4f}")

    oof_cb += oof_seed / len(SEEDS)
    test_cb += test_seed / len(SEEDS)

cb_auc = roc_auc_score(y, oof_cb)
log(f"\n  === CatBoost 3-seed OOF AUC: {cb_auc:.4f} ===")
log(f"  개별 seed: {[f'{a:.4f}' for a in seed_cb_aucs]}")

# ============================================================
# [5] 그룹별 AUC 분석
# ============================================================
log("\n## [5] 그룹별 AUC 분석")

oof_final = oof_cb.copy()
test_final = test_cb.copy()
final_auc = cb_auc

# 이식 그룹
if train_impl_mask.sum() > 0 and y[train_impl_mask].sum() > 0:
    impl_auc = roc_auc_score(y[train_impl_mask], oof_final[train_impl_mask])
    log(f"- 이식 그룹 ({train_impl_mask.sum()}건): AUC={impl_auc:.6f}")
else:
    impl_auc = 0
    log(f"- 이식 그룹: AUC 계산 불가")

# 비이식 그룹
if train_zero_mask.sum() > 0 and y[train_zero_mask].sum() > 0:
    zero_auc = roc_auc_score(y[train_zero_mask], oof_final[train_zero_mask])
    log(f"- 비이식 그룹 ({train_zero_mask.sum()}건): AUC={zero_auc:.6f}")
else:
    zero_auc = 0
    log(f"- 비이식 그룹: AUC 계산 불가")

log(f"- 전체: AUC={final_auc:.6f}")
log(
    f"- v13 이식 그룹 AUC 대비: v13=0.6745, v14={impl_auc:.4f} (Δ={impl_auc-0.6745:+.4f})"
)

# ============================================================
# [6] 종합 평가지표
# ============================================================
log("\n## [6] 종합 평가지표")

oof_ll = log_loss(y, oof_final)
oof_ap = average_precision_score(y, oof_final)

log(f"\n### 확률 기반 지표")
log(f"  OOF AUC:      {final_auc:.6f}")
log(f"  OOF Log Loss: {oof_ll:.6f}")
log(f"  OOF AP:       {oof_ap:.6f}")

log(f"\n### Threshold별 분류 지표")
log(
    f"  {'Threshold':>10}  {'Accuracy':>10}  {'Precision':>10}  "
    f"{'Recall':>10}  {'F1':>10}  {'Specificity':>12}"
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
        f"  {th:>10.2f}  {acc:>10.4f}  {prec:>10.4f}  "
        f"{rec:>10.4f}  {f1:>10.4f}  {spec:>12.4f}"
    )
    if f1 > best_f1:
        best_f1 = f1
        best_f1_th = th

log(f"\n  최적 F1: {best_f1:.4f} (threshold={best_f1_th})")

# ============================================================
# [7] 제출 파일 생성
# ============================================================
log("\n## [7] 제출 파일 생성")

sub_main = sub.copy()
sub_main["probability"] = test_final
main_path = os.path.join(RESULT_DIR, f"sample_submission_{VERSION}_{NOW}.csv")
sub_main.to_csv(main_path, index=False)

log(f"- 파일: {main_path}")
log(
    f"- 확률: mean={test_final.mean():.4f}, std={test_final.std():.4f}, "
    f"min={test_final.min():.6f}, max={test_final.max():.4f}"
)
log(f"- 예시:")
log(f"  {'ID':>10}  {'probability':>12}")
for i in range(5):
    log(f"  {sub_main.iloc[i, 0]:>10}  {test_final[i]:>12.6f}")

# ============================================================
# [8] 피처 중요도
# ============================================================
log("\n## [8] CatBoost 피처 중요도 (상위 30)")

if cb_importances is not None:
    imp_df = pd.DataFrame(
        {"feature": feature_names_cb, "importance": cb_importances}
    ).sort_values("importance", ascending=False)

    cat_set = set(cat_features_names)
    for rank_i, (_, row) in enumerate(imp_df.head(30).iterrows(), 1):
        cat_mark = " [cat]" if row["feature"] in cat_set else ""
        v14_mark = " ★v14" if row["feature"] in v14_new_features else ""
        log(
            f"  {rank_i}. {row['feature']}{cat_mark}{v14_mark}: {row['importance']:.2f}"
        )

    # v14 신규 피처 중요도 별도 출력
    log(f"\n### v14 신규 피처 중요도")
    v14_imp = imp_df[imp_df["feature"].isin(v14_new_features)].sort_values(
        "importance", ascending=False
    )
    for _, row in v14_imp.iterrows():
        rank_in_all = (imp_df["feature"] == row["feature"]).values.argmax() + 1
        log(f"  {row['feature']}: {row['importance']:.2f} (전체 {rank_in_all}위)")

# ============================================================
# [9] 버전 비교
# ============================================================
log("\n## [9] 버전 비교")
log("| 버전 | 모델 | OOF AUC | 이식그룹AUC | 비고 |")
log("|------|------|---------|------------|------|")
log("| v1 | CB원본 | 0.7403 | - | 베이스라인 (67피처) |")
log("| v4 | XGB | 0.7392 | - | 컬럼복원 |")
log("| v7 | XGB+CB | 0.7402 | - | 블렌딩 |")
log("| v8 | XGB+CB | 0.7401 | - | +파생변수 |")
log("| v9 | XGB+CB 3seed | 0.7405 | - | +IVF/DI분기 |")
log("| v11 | CB+XGB cross | 0.7405 | - | 크로스블렌딩 |")
log("| v12 | CB+XGB+TabNet | 0.7406 | - | 3모델블렌딩 |")
log(f"| v13 | CB 3seed | 0.7405 | 0.6745 | 클리핑 실패 |")
log(
    f"| **v14** | **CB 3seed** | **{final_auc:.4f}** | **{impl_auc:.4f}** | **이식전용피처** |"
)

# ============================================================
# [10] 최종 요약
# ============================================================
total_time = time.time() - t0
log(f"\n{'='*60}")
log(f"## 최종 요약")
log(f"{'='*60}")
log(f"- v14 핵심: 이식 그룹 전용 피처 {len(v14_found)}개 추가")
log(f"  {v14_found}")
log(f"- 모델: CatBoost 3-seed 앙상블")
log(f"- 파라미터: iterations=5000, lr=0.01, depth=8, l2=3")
log(f"- CB 3-seed OOF AUC: {final_auc:.6f}")
log(f"  seed별: {[f'{a:.4f}' for a in seed_cb_aucs]}")
log(f"- 이식 그룹 AUC: {impl_auc:.6f} (v13: 0.6745, Δ={impl_auc-0.6745:+.4f})")
log(f"- 비이식 그룹 AUC: {zero_auc:.6f} (v13: 0.909)")
log(f"- OOF LogLoss: {oof_ll:.6f}")
log(f"- OOF AP: {oof_ap:.6f}")
log(f"- 최적 F1: {best_f1:.4f} (th={best_f1_th})")
log(f"- 클리핑: 없음 (v13에서 하락 확인, 폐기)")
log(f"- 데이터 누수: 없음")
log(f"  - 모든 파생변수: 행 단위 연산 (타겟/test 정보 미사용)")
log(f"  - CatBoost: K-Fold OOF 학습, test 예측만 수행")
log(
    f"- 피처 수: {len(feature_names_cb)}개 (v13: 77개 → v14: {len(feature_names_cb)}개, "
    f"+{len(feature_names_cb)-77}개)"
)
log(f"- 제출: {main_path}")
log(f"- 소요: {total_time/60:.1f}분")
log(f"- 로그: {LOG_PATH}")
log(f"{'='*60}")

save_log()
print(f"\n완료! 로그: {LOG_PATH}")
