#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v15 피처 중요도 패치 - 모델 1개만 학습해서 중요도 추출 후 log_v15.md에 추가
"""
import os, time, warnings, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
RESULT_DIR = os.path.join(BASE_DIR, "result")
LOG_PATH = os.path.join(RESULT_DIR, "log_v15.md")

# ── 기존 로그 읽기 ──
with open(LOG_PATH, "r", encoding="utf-8") as f:
    existing_log = f.read()

# 이미 [8] 섹션이 있으면 그 앞까지만 유지
marker = "## [8] CatBoost 피처 중요도"
if marker in existing_log:
    existing_log = existing_log[: existing_log.index(marker)].rstrip() + "\n\n"

new_lines = []


def log(msg=""):
    print(msg)
    new_lines.append(msg)


# ── 데이터 & 전처리 (v15.py 동일) ──
print("데이터 로드 중...")
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
target_col = "임신 성공 여부"
y = train[target_col].values


# ── v15 전처리 함수 (v15.py에서 그대로 복사) ──
def preprocess_and_engineer(df, is_train=True):
    df = df.copy()
    drop_cols = [df.columns[0]]
    if is_train and target_col in df.columns:
        drop_cols.append(target_col)
    df = df.drop(columns=drop_cols, errors="ignore")
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    num_cols = df.select_dtypes(exclude=["object"]).columns.tolist()
    for c in cat_cols:
        df[c] = df[c].fillna("Unknown").astype(str)
    for c in num_cols:
        df[c] = df[c].fillna(0)

    embryo_col = "이식된 배아 수"
    total_col = "총 생성 배아 수"
    fresh_egg_col = "수집된 신선 난자 수"
    stored_col = "저장된 배아 수"
    mixed_col = "혼합된 난자 수"
    partner_col = "파트너 정자와 혼합된 난자 수"
    micro_transfer_col = "미세주입 배아 이식 수"
    transfer_day_col = "배아 이식 경과일"
    type_col = "시술 유형"
    single_col = "단일 배아 이식 여부"
    thawed_col = "해동된 배아 수"
    purpose_col = "배아 생성 주요 이유"
    egg_source_col = "난자 출처"
    sperm_source_col = "정자 출처"
    age_col = "시술 당시 나이"
    total_proc_col = "총 시술 횟수"

    if embryo_col in df.columns:
        df["실제이식여부"] = (df[embryo_col] > 0).astype(int)
    if embryo_col in df.columns and total_col in df.columns:
        df["total_embryo_ratio"] = df[embryo_col] / (df[total_col] + 1)
    if total_col in df.columns and fresh_egg_col in df.columns:
        df["수정_성공률"] = df[total_col] / (df[fresh_egg_col] + 1)
    if embryo_col in df.columns and total_col in df.columns:
        df["배아_이용률"] = df[embryo_col] / (df[total_col] + 1)
    if stored_col in df.columns and total_col in df.columns:
        df["배아_잉여율"] = df[stored_col] / (df[total_col] + 1)
    if (
        total_col in df.columns
        and fresh_egg_col in df.columns
        and mixed_col in df.columns
    ):
        df["난자_배아_전환율"] = df[total_col] / (df[fresh_egg_col] + df[mixed_col] + 1)
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
    if transfer_day_col in df.columns:
        df["transfer_day_optimal"] = df[transfer_day_col].apply(
            lambda x: 1 if 2 <= x <= 5 else 0
        )
        day_map = {0: 0, 1: 1, 2: 2, 3: 2, 4: 3, 5: 3, 6: 4, 7: 4}
        df["transfer_day_cat"] = df[transfer_day_col].map(day_map).fillna(0).astype(int)
    if embryo_col in df.columns and transfer_day_col in df.columns:
        df["embryo_day_interaction"] = df[embryo_col] * df[transfer_day_col]
    if embryo_col in df.columns and fresh_egg_col in df.columns:
        df["fresh_transfer_ratio"] = df[embryo_col] / (df[fresh_egg_col] + 1)
    if micro_transfer_col in df.columns and embryo_col in df.columns:
        df["micro_transfer_quality"] = df[micro_transfer_col] / (df[embryo_col] + 0.001)
        df.loc[df[embryo_col] == 0, "micro_transfer_quality"] = 0
    if single_col in df.columns and total_col in df.columns:
        df["single_good_embryo"] = (
            (df[single_col] == 1) & (df[total_col] >= 3)
        ).astype(int)
    if thawed_col in df.columns and embryo_col in df.columns:
        df["frozen_embryo_signal"] = df[thawed_col] * df[embryo_col]
    if stored_col in df.columns:
        df["embryo_surplus_after_transfer"] = df[stored_col]
    if embryo_col in df.columns and total_col in df.columns:
        df["transfer_intensity"] = (
            df[embryo_col] / (df[total_col] + 1) * (df[total_col] > 0).astype(int)
        )
    # v15
    if purpose_col in df.columns:
        df["purpose_current"] = (
            df[purpose_col].str.contains("현재 시술용", na=False).astype(int)
        )
    infertility_cols = [c for c in df.columns if c.startswith("불임 원인 -")]
    if len(infertility_cols) > 0:
        df["infertility_count"] = df[infertility_cols].sum(axis=1)
        df["has_infertility"] = (df["infertility_count"] > 0).astype(int)
    if egg_source_col in df.columns:
        df["donor_egg_flag"] = (df[egg_source_col] == "기증 제공").astype(int)
    if egg_source_col in df.columns and sperm_source_col in df.columns:
        egg_map = {"본인 제공": 0, "기증 제공": 1, "알 수 없음": 2, "Unknown": 2}
        sperm_map = {
            "배우자 제공": 0,
            "기증 제공": 1,
            "미할당": 2,
            "배우자 및 기증 제공": 3,
            "Unknown": 2,
        }
        df["egg_source_num"] = df[egg_source_col].map(egg_map).fillna(2).astype(int)
        df["sperm_source_num"] = (
            df[sperm_source_col].map(sperm_map).fillna(2).astype(int)
        )
        df["egg_sperm_combo"] = df["egg_source_num"] * 10 + df["sperm_source_num"]
        df = df.drop(columns=["egg_source_num", "sperm_source_num"])
    if total_col in df.columns and mixed_col in df.columns:
        df["embryo_per_egg"] = df[total_col] / (df[mixed_col] + 1)
    if mixed_col in df.columns and fresh_egg_col in df.columns:
        df["mixed_egg_ratio"] = df[mixed_col] / (df[fresh_egg_col] + 1)
    if partner_col in df.columns and mixed_col in df.columns:
        df["fresh_egg_dominance"] = df[partner_col] / (df[mixed_col] + 1)
    if transfer_day_col in df.columns:
        df["transfer_day_sq"] = df[transfer_day_col] ** 2
    if total_proc_col in df.columns:
        proc_map = {
            "0회": 0,
            "1회": 1,
            "2회": 2,
            "3회": 3,
            "4회": 4,
            "5회": 5,
            "6회 이상": 6,
        }
        proc_num = df[total_proc_col].map(proc_map)
        if proc_num.isna().all():
            proc_num = pd.to_numeric(df[total_proc_col], errors="coerce").fillna(0)
        else:
            proc_num = proc_num.fillna(0)
        df["first_treatment"] = (proc_num == 0).astype(int)
    if age_col in df.columns and embryo_col in df.columns:
        age_mid_map = {
            "만18-34세": 26,
            "만35-37세": 36,
            "만38-39세": 38.5,
            "만40-42세": 41,
            "만43-44세": 43.5,
            "만45-50세": 47.5,
            "알 수 없음": 35,
        }
        age_num = df[age_col].map(age_mid_map)
        if age_num.isna().all():
            age_num = pd.to_numeric(df[age_col], errors="coerce").fillna(35)
        else:
            age_num = age_num.fillna(35)
        df["age_transfer_interaction"] = age_num * df[embryo_col]
        df["age_mid"] = age_num

    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    return df, cat_cols


print("전처리 중...")
X_train_cb, cat_features_names = preprocess_and_engineer(train, is_train=True)
cat_features_idx = [X_train_cb.columns.get_loc(c) for c in cat_features_names]
feature_names_cb = X_train_cb.columns.tolist()

# ── v15 로그에서 Optuna best params 읽기 ──
# 로그에서 파싱하거나, 직접 입력 (로그에 출력된 값을 여기에 넣어라)
# TODO: 아래 값을 log_v15.md의 [3] Optuna 결과에서 복사해 넣어라
BEST_PARAMS = {
    "learning_rate": 0.01,  # ← log에서 확인한 값으로 교체
    "depth": 8,  # ← log에서 확인한 값으로 교체
    "l2_leaf_reg": 3.0,  # ← log에서 확인한 값으로 교체
    "min_data_in_leaf": 20,  # ← log에서 확인한 값으로 교체
    "random_strength": 1.0,  # ← log에서 확인한 값으로 교체
    "bagging_temperature": 0.8,  # ← log에서 확인한 값으로 교체
    "border_count": 128,  # ← log에서 확인한 값으로 교체
}

print(f"피처 중요도 추출용 모델 1개 학습 중 (약 2분)...")
t1 = time.time()

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
tr_idx, va_idx = next(iter(skf.split(X_train_cb, y)))

cb_model = CatBoostClassifier(
    iterations=5000,
    learning_rate=BEST_PARAMS["learning_rate"],
    depth=BEST_PARAMS["depth"],
    l2_leaf_reg=BEST_PARAMS["l2_leaf_reg"],
    min_data_in_leaf=BEST_PARAMS["min_data_in_leaf"],
    random_strength=BEST_PARAMS["random_strength"],
    bagging_temperature=BEST_PARAMS["bagging_temperature"],
    border_count=BEST_PARAMS["border_count"],
    eval_metric="AUC",
    random_seed=42,
    early_stopping_rounds=200,
    verbose=100,
    task_type="GPU",
    devices="0",
)

cb_model.fit(
    X_train_cb.iloc[tr_idx],
    y[tr_idx],
    eval_set=(X_train_cb.iloc[va_idx], y[va_idx]),
    cat_features=cat_features_idx,
)

va_auc = roc_auc_score(y[va_idx], cb_model.predict_proba(X_train_cb.iloc[va_idx])[:, 1])
print(
    f"모델 학습 완료: AUC={va_auc:.4f}, iter={cb_model.best_iteration_}, 소요={(time.time()-t1)/60:.1f}분"
)

cb_importances = cb_model.get_feature_importance()

# ── 피처 중요도 출력 ──
log(f"## [8] CatBoost 피처 중요도 (상위 30)")
log(f"(패치: 모델 1개 재학습으로 추출, Fold1 AUC={va_auc:.4f})")

imp_df = pd.DataFrame(
    {"feature": feature_names_cb, "importance": cb_importances}
).sort_values("importance", ascending=False)

cat_set = set(cat_features_names)
v15_new_names = [
    "purpose_current",
    "infertility_count",
    "has_infertility",
    "donor_egg_flag",
    "egg_sperm_combo",
    "embryo_per_egg",
    "mixed_egg_ratio",
    "fresh_egg_dominance",
    "transfer_day_sq",
    "first_treatment",
    "age_mid",
]
v15_set = set([f for f in v15_new_names if f in feature_names_cb])
v14_features = set(
    [
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
)

for rank_i, (_, row) in enumerate(imp_df.head(30).iterrows(), 1):
    marks = ""
    if row["feature"] in cat_set:
        marks += " [cat]"
    if row["feature"] in v15_set:
        marks += " ★v15"
    elif row["feature"] in v14_features:
        marks += " ★v14"
    log(f"  {rank_i}. {row['feature']}{marks}: {row['importance']:.2f}")

log(f"\n### v15 신규 피처 중요도")
feat_list = list(imp_df["feature"].values)
for f in sorted(v15_set):
    if f in feat_list:
        imp_val = imp_df[imp_df["feature"] == f]["importance"].values[0]
        rank = feat_list.index(f) + 1
        log(f"  {f}: {imp_val:.2f} (전체 {rank}위)")

# ── [9] 버전 비교도 추가 ──
# 기존 로그에서 OOF AUC 값 가져오기
log(f"\n## [9] 버전 비교")
log("| 버전 | 모델 | OOF AUC | 비고 |")
log("|------|------|---------|------|")
log("| v1 | CB원본 | 0.7403 | 베이스라인 |")
log("| v7 | XGB+CB | 0.7402 | 블렌딩 |")
log("| v9 | XGB+CB 3seed | 0.7405 | +IVF/DI분기 |")
log("| v12 | CB+XGB+TabNet | 0.7406 | 3모델블렌딩 |")
log("| v14 | CB 3seed | 0.7406 | 이식전용피처 |")
log("| v15 | CB 5seed+Optuna | (로그참조) | EDA피처+HP탐색 |")
log("")
log("(v15 OOF AUC는 기존 로그 [4] 섹션 참조)")

# ── 로그 저장 ──
with open(LOG_PATH, "w", encoding="utf-8") as f:
    f.write(existing_log + "\n".join(new_lines))

print(f"\n완료! 로그 업데이트: {LOG_PATH}")
