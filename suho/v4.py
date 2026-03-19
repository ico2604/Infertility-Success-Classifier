#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v4.py - v3 개선: 삭제 컬럼 복원 + scale_pos_weight 제거 + 실제이식여부 플래그
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

VERSION = "v4"
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
print(f"# {VERSION} - v3 개선: 컬럼 복원 + SPW 제거 + 이식플래그")
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

# 결측 99% 이상만 삭제 (시술유형, 난자출처, 정자출처는 유지!)
DROP_COLS = [
    "ID",
    TARGET,
    "PGD 시술 여부",
    "PGS 시술 여부",
    "난자 해동 경과일",
    "착상 전 유전 검사 사용 여부",
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

# --- 아이디어1: 시술유형, 난자출처, 정자출처 수동 인코딩 매핑 ---
PROCEDURE_MAP = {"IVF": 1, "DI": 0}
EGG_SOURCE_MAP = {"본인 제공": 0, "기증 제공": 1, "알 수 없음": 2}
SPERM_SOURCE_MAP = {
    "배우자 제공": 0,
    "기증 제공": 1,
    "미할당": 2,
    "배우자 및 기증 제공": 3,
}

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
    """컬럼 삭제 + 결측 채우기"""
    drop = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=drop, errors="ignore")
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna("Unknown")
        else:
            df[col] = df[col].fillna(0)
    return df


def step2_count_to_int(df):
    """횟수 → 정수"""
    for col in df.columns:
        if "횟수" in col and df[col].dtype == "object":
            df[col] = df[col].map(COUNT_MAP).fillna(0).astype(int)
    return df


def step3_age_to_int(df):
    """나이 → 정수 (순서형)"""
    for col in ["시술 당시 나이", "난자 기증자 나이", "정자 기증자 나이"]:
        if col not in df.columns:
            continue
        if df[col].dtype == "object":
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
    """[개선1] 시술유형, 난자출처, 정자출처 → 수동 인코딩 (삭제하지 않음!)"""

    if "시술 유형" in df.columns:
        df["시술 유형"] = df["시술 유형"].map(PROCEDURE_MAP).fillna(-1).astype(int)

    if "난자 출처" in df.columns:
        df["난자 출처"] = df["난자 출처"].map(EGG_SOURCE_MAP).fillna(-1).astype(int)

    if "정자 출처" in df.columns:
        df["정자 출처"] = df["정자 출처"].map(SPERM_SOURCE_MAP).fillna(-1).astype(int)

    return df


def step5_bool_to_int(df):
    """TRUE/FALSE → 0/1"""
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


def step6_feature_engineering(df):
    """파생변수 (행 독립 연산)"""

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
    embryo = (
        pd.to_numeric(df.get("총 생성 배아 수", 0), errors="coerce").fillna(0).values
    )
    mixed = pd.to_numeric(df.get("혼합된 난자 수", 0), errors="coerce").fillna(0).values
    df["수정_성공률"] = safe_div(embryo, mixed)

    # 이식 유형 One-Hot
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
        donor_flag = np.where(df["난자 출처"] == 1, 1, donor_flag)  # 기증=1
    if "난자 기증자 나이" in df.columns:
        donor_flag = np.where(
            (df["난자 기증자 나이"] != 0) & (df["난자 기증자 나이"] != -1),
            1,
            donor_flag,
        )
    df["is_donor_egg"] = donor_flag.astype(int)

    # 타클리닉 시술
    if "총 시술 횟수" in df.columns and "클리닉 내 총 시술 횟수" in df.columns:
        tc = pd.to_numeric(df["총 시술 횟수"], errors="coerce").fillna(0)
        cc = pd.to_numeric(df["클리닉 내 총 시술 횟수"], errors="coerce").fillna(0)
        df["타클리닉_시술"] = (tc - cc).clip(0).astype(int)

    # [개선3] 실제 이식 여부 플래그 (이식배아 0개 = 이식 안 함)
    if "이식된 배아 수" in df.columns:
        transferred = pd.to_numeric(df["이식된 배아 수"], errors="coerce").fillna(0)
        df["실제이식여부"] = (transferred > 0).astype(int)

    return df


def step7_drop_remaining_object(df, te_cols_to_keep):
    """TE 대상 외 남은 object 컬럼 삭제"""
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
    print(
        f"  [{label}] step1 삭제+결측: {df.shape}, obj={df.select_dtypes('object').shape[1]}"
    )

    df = step2_count_to_int(df)
    print(f"  [{label}] step2 횟수→int: obj={df.select_dtypes('object').shape[1]}")

    df = step3_age_to_int(df)
    print(f"  [{label}] step3 나이→int: obj={df.select_dtypes('object').shape[1]}")

    df = step4_manual_encode(df)
    print(
        f"  [{label}] step4 수동인코딩(시술유형,출처): obj={df.select_dtypes('object').shape[1]}"
    )

    df = step5_bool_to_int(df)
    print(f"  [{label}] step5 bool→int: obj={df.select_dtypes('object').shape[1]}")

    df = step6_feature_engineering(df)
    print(
        f"  [{label}] step6 파생변수: 컬럼={df.shape[1]}, obj={df.select_dtypes('object').shape[1]}"
    )

    return df


train_df = preprocess(train_raw.copy(), "train")
test_df = preprocess(test_raw.copy(), "test")

# TE 대상 외 object 삭제
te_cols_exist = [c for c in TE_COLS if c in train_df.columns]
train_df = step7_drop_remaining_object(train_df, te_cols_exist)
test_df = step7_drop_remaining_object(test_df, te_cols_exist)

print(
    f"  step7 후 - train: {train_df.shape}, obj={train_df.select_dtypes('object').shape[1]}"
)
print(
    f"  step7 후 - test: {test_df.shape}, obj={test_df.select_dtypes('object').shape[1]}"
)

# ============================================================
# 4. Target Encoding (K-Fold, train만 사용)
# ============================================================
print(f"\n## [4] Target Encoding (K-Fold, leakage-free)")

global_mean = y.mean()
alpha = 10
te_mappings = {}

skf_te = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

for col in te_cols_exist:
    print(f"  - {col} (고유값: {train_df[col].nunique()})")

    col_str = train_df[col].astype(str)
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

    train_df[col] = oof_te

    full_df = pd.DataFrame({"val": col_str.values, "target": y})
    full_agg = full_df.groupby("val")["target"].agg(["mean", "count"])
    full_agg["smoothed"] = (
        full_agg["count"] * full_agg["mean"] + alpha * global_mean
    ) / (full_agg["count"] + alpha)
    te_mappings[col] = full_agg["smoothed"].to_dict()

    test_col_str = test_df[col].astype(str)
    test_df[col] = test_col_str.map(te_mappings[col]).fillna(global_mean)

# 잔여 object 최종 정리
for df_name, df in [("train", train_df), ("test", test_df)]:
    obj_remain = df.select_dtypes("object").columns.tolist()
    if obj_remain:
        print(f"  {df_name} 잔여 object 삭제: {obj_remain}")
        if df_name == "train":
            train_df = train_df.drop(columns=obj_remain)
        else:
            test_df = test_df.drop(columns=obj_remain)

# ============================================================
# 5. 최종 피처 정리
# ============================================================
print(f"\n## [5] 최종 피처 정리")

common_cols = sorted(set(train_df.columns) & set(test_df.columns))
train_df = train_df[common_cols]
test_df = test_df[common_cols]

X_train = train_df.values.astype(np.float32)
X_test = test_df.values.astype(np.float32)
feature_names = list(common_cols)

print(f"- X_train: {X_train.shape}")
print(f"- X_test: {X_test.shape}")
print(f"- 피처 수: {len(feature_names)}")
print(f"- object: {train_df.select_dtypes('object').shape[1]}개")
print(f"- NaN: train={np.isnan(X_train).sum()}, test={np.isnan(X_test).sum()}")

# v3과 비교: 복원된 컬럼 확인
restored = [c for c in ["시술 유형", "난자 출처", "정자 출처"] if c in feature_names]
new_features = [c for c in ["실제이식여부"] if c in feature_names]
print(f"- [개선1] 복원된 컬럼: {restored}")
print(f"- [개선3] 새 피처: {new_features}")
print(f"- 피처 목록:")
for i, f in enumerate(feature_names):
    print(f"  {i+1}. {f}")

# ============================================================
# 6. XGBoost 5-Fold (GPU, scale_pos_weight 제거)
# ============================================================
print(f"\n## [6] XGBoost {N_FOLDS}-Fold (GPU, scale_pos_weight=1)")

xgb_params = {
    "n_estimators": 5000,
    "learning_rate": 0.01,
    "max_depth": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.3,
    "reg_lambda": 0.3,
    "min_child_weight": 20,
    # [개선2] scale_pos_weight 제거 (기본값 1)
    "eval_metric": "auc",
    "tree_method": "gpu_hist",
    "gpu_id": 0,
    "random_state": SEED,
    "verbosity": 0,
    "early_stopping_rounds": 200,
}
print(f"- [개선2] scale_pos_weight: 미설정 (기본 1)")
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
print(f"\n## [8] 피처 중요도")
fi = model.feature_importances_
fi_df = pd.DataFrame({"feature": feature_names, "importance": fi})
fi_df = fi_df.sort_values("importance", ascending=False)
print(f"| 순위 | 피처 | 중요도 |")
print(f"|------|------|--------|")
for i, (_, r) in enumerate(fi_df.head(25).iterrows()):
    marker = ""
    if r["feature"] in restored:
        marker = " ★복원"
    elif r["feature"] in new_features:
        marker = " ★신규"
    print(f"| {i+1} | {r['feature']}{marker} | {r['importance']:.4f} |")

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
# 10. v3 vs v4 비교
# ============================================================
print(f"\n## [10] v3 → v4 변경사항")
print(
    f"""
| 항목 | v3 | v4 |
|------|-----|-----|
| 시술 유형 | 삭제됨 | IVF=1, DI=0 으로 복원 |
| 난자 출처 | 삭제됨 | 본인=0, 기증=1, 알수없음=2 복원 |
| 정자 출처 | 삭제됨 | 배우자=0, 기증=1, 미할당=2, 둘다=3 복원 |
| scale_pos_weight | 3 | 1 (기본값, 미설정) |
| 실제이식여부 | 없음 | 이식배아>0 이면 1 |
| OOF AUC | 0.7387 | {oof_auc:.4f} |
| 변화 | - | {oof_auc - 0.7387:+.4f} |
"""
)

# ============================================================
# 11. 최종 요약
# ============================================================
total_time = time.time() - start_all
print(f"{'='*60}")
print(f"## 최종 요약")
print(f"{'='*60}")
print(f"- 모델: XGBoost (GPU, tree_method=gpu_hist)")
print(f"- 피처 수: {len(feature_names)}")
print(f"- **OOF AUC: {oof_auc:.4f}**")
print(f"- OOF F1: {oof_f1:.4f}")
print(f"- scale_pos_weight: 1 (기본)")
print(f"- Data Leakage: 없음")
print(f"- 총 소요시간: {total_time/60:.1f}분")
print(f"- 로그: {LOG_PATH}")
print(f"{'='*60}")
