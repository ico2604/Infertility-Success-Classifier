import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

print("=" * 60)
print("  Step 10: Codex CatBoost + Blend + Submit (Soyoon Local)")
print("=" * 60)

try:
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    from scipy.stats import rankdata
    from scipy.optimize import minimize
    from catboost import CatBoostClassifier, Pool
    print("  패키지 로드 완료!")
except ImportError as e:
    print(f"  [오류] 패키지 없음: {e}")
    print("  필요 패키지: pip install numpy pandas scikit-learn scipy catboost")
    sys.exit(1)

# ============================================================
# 설정 (소윤님 로컬 경로용)
# ============================================================
BASE = Path(__file__).resolve().parent.parent
TRAIN_PATH = BASE / "data" / "train.csv"
TEST_PATH = BASE / "data" / "test.csv"
SAMPLE_CSV = BASE / "data" / "sample_submission.csv"
FALLBACK_SAMPLE = BASE / "ensemble_v4_submission_소윤.csv"
MODELS_DIR = BASE / "models"
SUBMISSION_DIR = BASE / "submissions"
TEAM_EXISTING_DIR = BASE / "team_existing_oofs"

MODELS_DIR.mkdir(exist_ok=True)
SUBMISSION_DIR.mkdir(exist_ok=True)
TEAM_EXISTING_DIR.mkdir(exist_ok=True)

N_SPLITS = 5
SEEDS = [42, 123, 2024, 2026, 2604]

# 기존 OOF 경로: 없으면 자동 스킵
EXISTING_OOFS = {
    "CX_V31": {
        "oof": TEAM_EXISTING_DIR / "oof_predictions_cx_v31.csv",
        "test": TEAM_EXISTING_DIR / "submission_cx_v31.csv",
        "fmt": "codex",
        "oof_col": "oof_probability",
        "test_col": "probability",
    },
    "CX_LGB": {
        "oof": TEAM_EXISTING_DIR / "oof_predictions_cx_lgb.csv",
        "test": TEAM_EXISTING_DIR / "submission_cx_lgb.csv",
        "fmt": "codex",
        "oof_col": "oof_probability",
        "test_col": "probability",
    },
    "CL_XGB": {
        "oof": MODELS_DIR / "oof_v3_xgb.npy",
        "test": MODELS_DIR / "test_v3_xgb.npy",
        "fmt": "npy",
    },
}

INFERTILITY_CAUSE_COLS = [
    "불임 원인 - 난관 질환",
    "불임 원인 - 남성 요인",
    "불임 원인 - 배란 장애",
    "불임 원인 - 여성 요인",
    "불임 원인 - 자궁경부 문제",
    "불임 원인 - 자궁내막증",
    "불임 원인 - 정자 농도",
    "불임 원인 - 정자 면역학적 요인",
    "불임 원인 - 정자 운동성",
    "불임 원인 - 정자 형태",
]

AGE_ORDINAL = {
    "만18-34세": 0, "만35-37세": 1, "만38-39세": 2,
    "만40-42세": 3, "만43-44세": 4, "만45-50세": 5,
    "알 수 없음": -1,
}
DONOR_AGE_ORDINAL = {
    "만20세 이하": 0, "만21-25세": 1, "만26-30세": 2,
    "만31-35세": 3, "알 수 없음": -1,
}
SPERM_DONOR_AGE_ORDINAL = {
    "만20세 이하": 0, "만21-25세": 1, "만26-30세": 2,
    "만31-35세": 3, "만36-40세": 4, "만41-45세": 5,
    "알 수 없음": -1,
}


def parse_count_column_v2(series):
    def _parse(val):
        if pd.isna(val):
            return np.nan
        val_str = str(val).strip().replace("이상", "").replace("회", "").strip()
        try:
            return float(val_str)
        except ValueError:
            return np.nan
    return series.apply(_parse)


def create_missing_indicators(df):
    df = df.copy()
    df["행_결측_비율"] = df.isna().mean(axis=1)
    configs = {
        "임신 시도 또는 마지막 임신 경과 연수": "임신경과연수_있음",
        "착상 전 유전 검사 사용 여부": "유전검사_있음",
        "PGD 시술 여부": "PGD_있음",
        "PGS 시술 여부": "PGS_있음",
        "난자 해동 경과일": "난자해동일_있음",
        "배아 해동 경과일": "배아해동일_있음",
        "배아 이식 경과일": "배아이식일_있음",
        "난자 채취 경과일": "난자채취일_있음",
    }
    for col, new_col in configs.items():
        if col in df.columns:
            df[new_col] = (~df[col].isna()).astype(int)
    return df


def extract_procedure_features(df):
    df = df.copy()
    col = "특정 시술 유형"
    if col not in df.columns:
        return df
    raw = df[col].fillna("Unknown").astype(str)

    def get_base(val):
        first = val.split(":")[0].strip()
        if "ICSI" in first:
            return 0
        elif "IVF" in first:
            return 1
        elif "IUI" in first:
            return 2
        return 3

    df["시술_기본유형"] = raw.apply(get_base)
    df["블라스토시스트_여부"] = raw.str.contains("BLASTOCYST", na=False).astype(int)
    df["보조부화_여부"] = raw.str.contains("/ AH", na=False).astype(int)
    df["이중시술_여부"] = raw.str.contains(":", na=False).astype(int)
    return df


def extract_egg_source_feature(df):
    df = df.copy()
    col = "난자 출처"
    if col in df.columns:
        df["난자기증_사용"] = df[col].astype(str).str.contains("기증", na=False).astype(int)
    return df


def add_ivf_di_split_features(df):
    df = df.copy()
    col = "시술 유형"
    if col not in df.columns:
        return df

    treatment_str = df[col].fillna("Unknown").astype(str)
    is_ivf = (treatment_str == "IVF").astype(float)
    is_di = (treatment_str == "DI").astype(float)
    df["is_ivf_flag"] = is_ivf
    df["is_di_flag"] = is_di

    total_treat = df.get("총 시술 횟수", pd.Series(0, index=df.index))
    total_treat = pd.to_numeric(total_treat, errors="coerce").fillna(0)

    total_preg = df.get("총 임신 횟수", pd.Series(0, index=df.index))
    total_preg = pd.to_numeric(total_preg, errors="coerce").fillna(0)

    df["ivf_시술부하"] = is_ivf * total_treat
    df["di_시술부하"] = is_di * total_treat
    df["ivf_임신부하"] = is_ivf * total_preg
    df["di_임신부하"] = is_di * total_preg

    total_created = pd.to_numeric(df.get("총 생성 배아 수", pd.Series(0, index=df.index)), errors="coerce")
    transferred = pd.to_numeric(df.get("이식된 배아 수", pd.Series(0, index=df.index)), errors="coerce")
    transfer_ratio = (transferred / total_created.replace(0, np.nan)).fillna(0)
    df["ivf_이식비율"] = is_ivf * transfer_ratio
    df["di_이식비율"] = is_di * transfer_ratio

    stored = pd.to_numeric(df.get("저장된 배아 수", pd.Series(0, index=df.index)), errors="coerce")
    storage_ratio = (stored / total_created.replace(0, np.nan)).fillna(0)
    df["ivf_저장비율"] = is_ivf * storage_ratio
    df["di_저장비율"] = is_di * storage_ratio

    age_ord = pd.to_numeric(df.get("시술 당시 나이", pd.Series(0, index=df.index)), errors="coerce").fillna(0)
    embryo_days = pd.to_numeric(df.get("배아 이식 경과일", pd.Series(0, index=df.index)), errors="coerce").fillna(0)
    embryo_age = embryo_days * age_ord
    df["ivf_배아나이"] = is_ivf * embryo_age
    df["di_배아나이"] = is_di * embryo_age
    return df


def add_infertility_bundle_features(df):
    df = df.copy()
    cause_cols = [c for c in INFERTILITY_CAUSE_COLS if c in df.columns]
    if not cause_cols:
        return df

    cause_count = df[cause_cols].sum(axis=1)
    age_val = pd.to_numeric(df.get("시술 당시 나이", pd.Series(0, index=df.index)), errors="coerce").fillna(0)
    treat_count = pd.to_numeric(df.get("총 시술 횟수", pd.Series(0, index=df.index)), errors="coerce").fillna(0)

    df["원인수_x_나이"] = cause_count * (age_val + 1.0)
    df["원인수_x_시술횟수"] = cause_count * (treat_count + 1.0)
    df["복합원인플래그"] = (cause_count >= 2).astype(float)
    return df


def add_domain_features_v2(df):
    df = df.copy()

    if "총 생성 배아 수" in df.columns and "이식된 배아 수" in df.columns:
        total = pd.to_numeric(df["총 생성 배아 수"], errors="coerce").replace(0, np.nan)
        df["배아_이식_비율"] = pd.to_numeric(df["이식된 배아 수"], errors="coerce") / total

    if "IVF 시술 횟수" in df.columns and "DI 시술 횟수" in df.columns:
        df["총_시술_경험"] = (
            pd.to_numeric(df["IVF 시술 횟수"], errors="coerce").fillna(0)
            + pd.to_numeric(df["DI 시술 횟수"], errors="coerce").fillna(0)
        )

    if "총 임신 횟수" in df.columns and "총 시술 횟수" in df.columns:
        denom = pd.to_numeric(df["총 시술 횟수"], errors="coerce").replace(0, np.nan)
        df["이전_임신_성공률"] = pd.to_numeric(df["총 임신 횟수"], errors="coerce") / denom

    if "혼합된 난자 수" in df.columns and "수집된 신선 난자 수" in df.columns:
        fresh = pd.to_numeric(df["수집된 신선 난자 수"], errors="coerce").replace(0, np.nan)
        df["난자_활용_효율"] = pd.to_numeric(df["혼합된 난자 수"], errors="coerce") / fresh

    infertility_cols = [c for c in df.columns if c.startswith("불임 원인")]
    if infertility_cols:
        df["불임_원인_합계"] = df[infertility_cols].sum(axis=1)

    if "미세주입된 난자 수" in df.columns and "수집된 신선 난자 수" in df.columns:
        fresh = pd.to_numeric(df["수집된 신선 난자 수"], errors="coerce").replace(0, np.nan)
        df["미세주입_비율"] = pd.to_numeric(df["미세주입된 난자 수"], errors="coerce") / fresh

    if "저장된 배아 수" in df.columns and "이식된 배아 수" in df.columns:
        total_e = (
            pd.to_numeric(df["저장된 배아 수"], errors="coerce").fillna(0)
            + pd.to_numeric(df["이식된 배아 수"], errors="coerce").fillna(0)
        )
        total_e = total_e.replace(0, np.nan)
        df["이식_저장_비율"] = pd.to_numeric(df["이식된 배아 수"], errors="coerce") / total_e

    if "총 출산 횟수" in df.columns:
        df["출산_경험"] = (pd.to_numeric(df["총 출산 횟수"], errors="coerce").fillna(0) > 0).astype(int)

    if "시술 당시 나이" in df.columns and "이식된 배아 수" in df.columns:
        df["나이_x_배아이식수"] = (
            pd.to_numeric(df["시술 당시 나이"], errors="coerce").fillna(0)
            * pd.to_numeric(df["이식된 배아 수"], errors="coerce").fillna(0)
        )

    if "불임 원인 - 남성 요인" in df.columns and "시술_기본유형" in df.columns:
        is_icsi = (df["시술_기본유형"] == 0).astype(int)
        df["남성요인_x_ICSI"] = (
            pd.to_numeric(df["불임 원인 - 남성 요인"], errors="coerce").fillna(0) * is_icsi
        )

    if "총 생성 배아 수" in df.columns and "혼합된 난자 수" in df.columns:
        mixed = pd.to_numeric(df["혼합된 난자 수"], errors="coerce").replace(0, np.nan)
        df["수정률"] = pd.to_numeric(df["총 생성 배아 수"], errors="coerce") / mixed

    return df


def add_features_v3(df):
    df = df.copy()

    if "저장된 배아 수" in df.columns and "총 생성 배아 수" in df.columns:
        df["배아품질_proxy"] = (
            pd.to_numeric(df["저장된 배아 수"], errors="coerce").fillna(0)
            / (pd.to_numeric(df["총 생성 배아 수"], errors="coerce").fillna(0) + 1)
        )

    if "시술 당시 나이" in df.columns:
        age_num = pd.to_numeric(df["시술 당시 나이"], errors="coerce").fillna(0)
        df["나이_제곱"] = age_num ** 2

    if "총 시술 횟수" in df.columns and "총 임신 횟수" in df.columns:
        treat = pd.to_numeric(df["총 시술 횟수"], errors="coerce").fillna(0)
        preg = pd.to_numeric(df["총 임신 횟수"], errors="coerce").fillna(0)
        df["반복실패"] = ((treat >= 3) & (preg == 0)).astype(int)

    if "총 출산 횟수" in df.columns and "총 임신 횟수" in df.columns:
        birth = pd.to_numeric(df["총 출산 횟수"], errors="coerce").fillna(0)
        preg = pd.to_numeric(df["총 임신 횟수"], errors="coerce").fillna(0)
        df["이전성공_경험"] = ((birth > 0) & (preg > 0)).astype(int)

    if "시술 당시 나이" in df.columns and "난자기증_사용" in df.columns:
        age_num = pd.to_numeric(df["시술 당시 나이"], errors="coerce").fillna(0)
        df["나이_x_기증난자"] = age_num * pd.to_numeric(df["난자기증_사용"], errors="coerce").fillna(0)

    if "불임 원인 - 남성 요인" in df.columns and "불임_원인_합계" in df.columns:
        male = pd.to_numeric(df["불임 원인 - 남성 요인"], errors="coerce").fillna(0)
        infert_sum = pd.to_numeric(df["불임_원인_합계"], errors="coerce").fillna(0)
        tubal_col = [c for c in df.columns if "난관" in c]
        unexplained_col = [c for c in df.columns if "원인불명" in c or "설명되지" in c]

        df["불임원인_복합도"] = 0
        df.loc[(male == 1) & (infert_sum == 1), "불임원인_복합도"] = 1

        if tubal_col:
            tubal = pd.to_numeric(df[tubal_col[0]], errors="coerce").fillna(0)
            df.loc[(tubal == 1) & (infert_sum == 1), "불임원인_복합도"] = 2

        if unexplained_col:
            unexp = pd.to_numeric(df[unexplained_col[0]], errors="coerce").fillna(0)
            df.loc[(unexp == 1) & (infert_sum == 1), "불임원인_복합도"] = 3

    return df


def detect_categorical_columns(df):
    cat_cols = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()

    # 혹시 dtype은 object가 아닌데 문자열이 섞인 컬럼이 있으면 추가 탐지
    extra_cat_cols = []
    for col in df.columns:
        if col in cat_cols:
            continue
        sample = df[col].dropna().head(20)
        if len(sample) > 0 and any(isinstance(v, str) for v in sample):
            extra_cat_cols.append(col)

    cat_cols = sorted(list(set(cat_cols + extra_cat_cols)))
    return cat_cols


def align_train_test_columns(train, test):
    train = train.copy()
    test = test.copy()

    for col in train.columns:
        if col not in test.columns:
            test[col] = np.nan

    for col in test.columns:
        if col not in train.columns:
            train[col] = np.nan

    test = test[train.columns]
    return train, test


def preprocess_for_catboost(train_raw, test_raw):
    train = train_raw.copy()
    test = test_raw.copy()

    target_col = "임신 성공 여부"
    id_col = "ID"

    y = train[target_col].copy()
    train = train.drop(columns=[target_col])

    train_ids = train[id_col].copy()
    test_ids = test[id_col].copy()

    train = train.drop(columns=[id_col])
    test = test.drop(columns=[id_col])

    print("    [1/9] 결측 지표 생성...")
    train = create_missing_indicators(train)
    test = create_missing_indicators(test)

    print("    [2/9] 시술 유형 분해...")
    train = extract_procedure_features(train)
    test = extract_procedure_features(test)

    print("    [3/9] 난자 기증 피처...")
    train = extract_egg_source_feature(train)
    test = extract_egg_source_feature(test)

    print("    [4/9] '회' 컬럼 변환...")
    count_cols = [c for c in train.columns if train[c].astype(str).str.contains("회").any()]
    for col in count_cols:
        train[col] = parse_count_column_v2(train[col])
        if col in test.columns:
            test[col] = parse_count_column_v2(test[col])

    print("    [5/9] 나이 Ordinal 인코딩...")
    ordinal_configs = {
        "시술 당시 나이": AGE_ORDINAL,
        "난자 기증자 나이": DONOR_AGE_ORDINAL,
        "정자 기증자 나이": SPERM_DONOR_AGE_ORDINAL,
    }
    for col, mapping in ordinal_configs.items():
        if col in train.columns:
            train[col] = train[col].map(mapping).fillna(-1).astype(int)
        if col in test.columns:
            test[col] = test[col].map(mapping).fillna(-1).astype(int)

    print("    [6/9] IVF/DI 분기 피처 12개...")
    train = add_ivf_di_split_features(train)
    test = add_ivf_di_split_features(test)

    print("    [7/9] 불임 번들 피처 3개...")
    train = add_infertility_bundle_features(train)
    test = add_infertility_bundle_features(test)

    print("    [8/9] 도메인 피처 v2 + v3...")
    train = add_domain_features_v2(train)
    test = add_domain_features_v2(test)
    train = add_features_v3(train)
    test = add_features_v3(test)

    print("    [9/9] CatBoost 범주형 처리...")
    train, test = align_train_test_columns(train, test)

    cat_cols = detect_categorical_columns(train)

    for col in cat_cols:
        train[col] = train[col].astype("string").fillna("_MISSING_")
        test[col] = test[col].astype("string").fillna("_MISSING_")

    # 숫자형 컬럼 정리
    num_cols = [c for c in train.columns if c not in cat_cols]
    for col in num_cols:
        train[col] = pd.to_numeric(train[col], errors="coerce")
        test[col] = pd.to_numeric(test[col], errors="coerce")

    print("\n  전처리 완료!")
    print(f"    피처 수: {train.shape[1]}")
    print(f"    범주형: {len(cat_cols)}개")
    print(f"    숫자형: {train.shape[1] - len(cat_cols)}개")

    # 디버그용
    non_numeric_cols = train.select_dtypes(exclude=["number"]).columns.tolist()
    missing_in_cat = [c for c in non_numeric_cols if c not in cat_cols]
    if missing_in_cat:
        print("    [경고] non-numeric인데 cat_cols에 없는 컬럼:", missing_in_cat)

    return train, test, y, train.columns.tolist(), cat_cols, train_ids, test_ids


def train_catboost_5seed(X_train, X_test, y_train, cat_cols, feature_names):
    print("\n" + "=" * 60)
    print("  [Step 1] CatBoost 5-Seed 학습")
    print("=" * 60)

    n_samples = len(y_train)
    oof_all_seeds = np.zeros(n_samples)
    test_all_seeds = np.zeros(len(X_test))

    cat_params = {
        "iterations": 6000,
        "learning_rate": 0.02,
        "depth": 6,
        "l2_leaf_reg": 10.0,
        "random_strength": 1.2,
        "bagging_temperature": 0.8,
        "rsm": 0.8,
        "border_count": 128,
        "auto_class_weights": "Balanced",
        "eval_metric": "AUC",
        "verbose": 200,
        "task_type": "CPU",
        "thread_count": -1,
        "loss_function": "Logloss",
    }

    print("\n  [DEBUG] categorical columns sample:")
    print(f"    total={len(cat_cols)}")
    print(f"    {cat_cols[:20]}")

    print("\n  [DEBUG] first 10 columns:")
    print(f"    {X_train.columns[:10].tolist()}")

    for seed_idx, seed in enumerate(SEEDS):
        print(f"\n  --- Seed {seed_idx + 1}/{len(SEEDS)}: {seed} ---")
        t_seed = time.time()
        oof_pred = np.zeros(n_samples)
        test_pred = np.zeros(len(X_test))

        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)

        for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train)):
            t_fold = time.time()
            print(f"    Fold {fold + 1}/{N_SPLITS}...", end=" ", flush=True)

            X_tr = X_train.iloc[tr_idx].copy()
            X_va = X_train.iloc[va_idx].copy()
            y_tr = y_train.iloc[tr_idx]
            y_va = y_train.iloc[va_idx]

            fold_cat_cols = [c for c in cat_cols if c in X_tr.columns]

            train_pool = Pool(X_tr, y_tr, cat_features=fold_cat_cols)
            val_pool = Pool(X_va, y_va, cat_features=fold_cat_cols)
            test_pool = Pool(X_test, cat_features=fold_cat_cols)

            model = CatBoostClassifier(**cat_params, random_seed=seed)
            model.fit(
                train_pool,
                eval_set=val_pool,
                early_stopping_rounds=200,
                verbose=False
            )

            va_pred = model.predict_proba(val_pool)[:, 1]
            oof_pred[va_idx] = va_pred
            fold_auc = roc_auc_score(y_va, va_pred)

            test_pred += model.predict_proba(test_pool)[:, 1] / N_SPLITS
            print(f"AUC={fold_auc:.6f} ({time.time() - t_fold:.0f}초)")

        seed_auc = roc_auc_score(y_train, oof_pred)
        print(f"    Seed {seed} OOF AUC: {seed_auc:.6f} ({time.time() - t_seed:.0f}초)")
        oof_all_seeds += oof_pred / len(SEEDS)
        test_all_seeds += test_pred / len(SEEDS)

    final_auc = roc_auc_score(y_train, oof_all_seeds)
    print(f"\n  ★ 5-Seed 평균 OOF AUC: {final_auc:.6f}")

    np.save(MODELS_DIR / "oof_v5_cat.npy", oof_all_seeds)
    np.save(MODELS_DIR / "test_v5_cat.npy", test_all_seeds)
    print(f"  저장: {MODELS_DIR / 'oof_v5_cat.npy'}")
    print(f"  저장: {MODELS_DIR / 'test_v5_cat.npy'}")

    return oof_all_seeds, test_all_seeds, final_auc


def load_existing_oofs(y):
    loaded = {}
    for name, info in EXISTING_OOFS.items():
        oof_path = Path(info["oof"])
        test_path = Path(info["test"])

        if not oof_path.exists() or not test_path.exists():
            print(f"    {name}: 파일 없음 - 스킵")
            continue

        if info["fmt"] == "codex":
            oof_df = pd.read_csv(oof_path, encoding="utf-8-sig")
            test_df = pd.read_csv(test_path, encoding="utf-8-sig")
            oof = oof_df[info.get("oof_col", "oof_probability")].values
            test = test_df[info.get("test_col", "probability")].values
        else:
            oof = np.load(oof_path).astype(np.float64)
            test = np.load(test_path).astype(np.float64)

        auc = roc_auc_score(y, oof)
        print(f"    {name:10s}: OOF AUC = {auc:.6f}")
        loaded[name] = {"oof": oof, "test": test, "auc": auc}
    return loaded


def scipy_weights(oofs, y, n_starts=50):
    n = len(oofs)

    def neg_auc(w):
        w = np.abs(w)
        s = w.sum()
        if s < 1e-10:
            return 0.0
        w = w / s
        blend = sum(wi * oof for wi, oof in zip(w, oofs))
        return -roc_auc_score(y, blend)

    best = None
    for _ in range(n_starts):
        x0 = np.random.dirichlet(np.ones(n))
        result = minimize(
            neg_auc,
            x0,
            method="Nelder-Mead",
            options={"maxiter": 5000, "xatol": 1e-7}
        )
        if best is None or result.fun < best.fun:
            best = result

    w = np.abs(best.x)
    w = w / w.sum()
    return w, -best.fun


def grid_search_4(oofs, y, step=0.05):
    best_score = 0
    best_w = None
    steps = np.arange(0, 1.01, step)

    for w0 in steps:
        for w1 in np.arange(0, 1.01 - w0, step):
            for w2 in np.arange(0, 1.01 - w0 - w1, step):
                w3 = 1.0 - w0 - w1 - w2
                if w3 < -0.001:
                    continue
                blend = w0 * oofs[0] + w1 * oofs[1] + w2 * oofs[2] + w3 * oofs[3]
                s = roc_auc_score(y, blend)
                if s > best_score:
                    best_score = s
                    best_w = [w0, w1, w2, w3]

    if best_w is not None:
        center = list(best_w)
        fine = np.arange(-step, step + 0.001, 0.01)
        for d0 in fine:
            for d1 in fine:
                for d2 in fine:
                    w0 = center[0] + d0
                    w1 = center[1] + d1
                    w2 = center[2] + d2
                    w3 = 1.0 - w0 - w1 - w2
                    if w0 < 0 or w1 < 0 or w2 < 0 or w3 < 0:
                        continue
                    blend = w0 * oofs[0] + w1 * oofs[1] + w2 * oofs[2] + w3 * oofs[3]
                    s = roc_auc_score(y, blend)
                    if s > best_score:
                        best_score = s
                        best_w = [w0, w1, w2, w3]
    return best_w, best_score


def blend_optimization(new_oof, new_test, y, existing):
    print("\n" + "=" * 60)
    print("  [Step 2] 블렌드 최적화")
    print("=" * 60)

    all_names = list(existing.keys()) + ["NEW_Cat"]
    all_oofs = [existing[n]["oof"] for n in existing] + [new_oof]
    all_tests = [existing[n]["test"] for n in existing] + [new_test]

    n = len(all_names)
    print(f"\n  사용 모델 {n}개:")
    for name, oof in zip(all_names, all_oofs):
        print(f"    {name:10s}: {roc_auc_score(y, oof):.6f}")

    results = []
    n_train = len(all_oofs[0])
    n_test = len(all_tests[0])

    print(f"\n  [방법 1] 전체 {n}모델 랭크 평균")
    oof_ranks = [rankdata(o) / n_train for o in all_oofs]
    test_ranks = [rankdata(t) / n_test for t in all_tests]
    ra_oof = np.mean(oof_ranks, axis=0)
    ra_test = np.mean(test_ranks, axis=0)
    ra_auc = roc_auc_score(y, ra_oof)
    print(f"    AUC = {ra_auc:.6f}")
    results.append(("rank_avg_all", ra_auc, None, all_names, ra_oof, ra_test))

    print(f"\n  [방법 2] 전체 {n}모델 Scipy 가중치")
    np.random.seed(42)
    sw, ss = scipy_weights(all_oofs, y)
    print(f"    AUC = {ss:.6f}")
    for nm, wi in zip(all_names, sw):
        if wi > 0.005:
            print(f"      {nm:10s}: {wi:.4f}")
    sp_test = sum(wi * t for wi, t in zip(sw, all_tests))
    sp_oof = sum(wi * o for wi, o in zip(sw, all_oofs))
    results.append(("scipy_all", ss, sw, all_names, sp_oof, sp_test))

    print(f"\n  [방법 3] 전체 가중 랭크 (Scipy 가중치)")
    wr_oof = sum(wi * r for wi, r in zip(sw, oof_ranks))
    wr_test = sum(wi * r for wi, r in zip(sw, test_ranks))
    wr_auc = roc_auc_score(y, wr_oof)
    print(f"    AUC = {wr_auc:.6f}")
    results.append(("weighted_rank_all", wr_auc, sw, all_names, wr_oof, wr_test))

    if n >= 4:
        auc_list = [(nm, roc_auc_score(y, o)) for nm, o in zip(all_names, all_oofs)]
        auc_list.sort(key=lambda x: -x[1])
        top4_names = [x[0] for x in auc_list[:4]]
        top4_idx = [all_names.index(nm) for nm in top4_names]
        top4_oofs = [all_oofs[i] for i in top4_idx]
        top4_tests = [all_tests[i] for i in top4_idx]

        print(f"\n  [방법 4] 상위 4모델 그리드서치: {top4_names}")
        gw, gs = grid_search_4(top4_oofs, y)
        print(f"    AUC = {gs:.6f}")
        for nm, wi in zip(top4_names, gw):
            print(f"      {nm:10s}: {wi:.4f}")
        g_oof = sum(wi * o for wi, o in zip(gw, top4_oofs))
        g_test = sum(wi * t for wi, t in zip(gw, top4_tests))
        results.append(("grid_top4", gs, gw, top4_names, g_oof, g_test))

    from itertools import combinations
    print(f"\n  [방법 5] 모든 2~3모델 조합 탐색...")
    best_combo = None

    for size in [2, 3]:
        for combo in combinations(range(n), size):
            combo_names = [all_names[i] for i in combo]
            combo_oofs = [all_oofs[i] for i in combo]
            combo_tests = [all_tests[i] for i in combo]

            c_oof_ranks = [rankdata(o) / n_train for o in combo_oofs]
            c_test_ranks = [rankdata(t) / n_test for t in combo_tests]
            c_ra_oof = np.mean(c_oof_ranks, axis=0)
            c_ra_test = np.mean(c_test_ranks, axis=0)
            c_auc = roc_auc_score(y, c_ra_oof)

            if best_combo is None or c_auc > best_combo[1]:
                best_combo = (
                    f"rank_{'+'.join(combo_names)}",
                    c_auc,
                    None,
                    combo_names,
                    c_ra_oof,
                    c_ra_test,
                )

            np.random.seed(42)
            c_sw, c_ss = scipy_weights(combo_oofs, y, n_starts=20)
            if c_ss > best_combo[1]:
                c_sp_oof = sum(wi * o for wi, o in zip(c_sw, combo_oofs))
                c_sp_test = sum(wi * t for wi, t in zip(c_sw, combo_tests))
                best_combo = (
                    f"scipy_{'+'.join(combo_names)}",
                    c_ss,
                    c_sw,
                    combo_names,
                    c_sp_oof,
                    c_sp_test,
                )

    if best_combo is not None:
        print(f"    최고 조합: {best_combo[0]} AUC={best_combo[1]:.6f}")
        results.append(best_combo)

    results.sort(key=lambda x: -x[1])
    print(f"\n  === 전체 블렌드 결과 ===")
    for i, (method, score, _, _, _, _) in enumerate(results):
        marker = " ★ BEST" if i == 0 else ""
        print(f"    {method:35s}: AUC = {score:.6f}{marker}")

    return results


def resolve_test_ids():
    if SAMPLE_CSV.exists():
        sample_df = pd.read_csv(SAMPLE_CSV)
        return sample_df["ID"], sample_df

    if FALLBACK_SAMPLE.exists():
        fallback = pd.read_csv(FALLBACK_SAMPLE)
        return fallback["ID"], fallback[["ID", "probability"]]

    test_df = pd.read_csv(TEST_PATH)
    fallback = pd.DataFrame({"ID": test_df["ID"], "probability": 0.0})
    return test_df["ID"], fallback


def create_submission(results, test_ids, sample_like):
    print("\n" + "=" * 60)
    print("  [Step 3] 제출파일 생성")
    print("=" * 60)

    best = results[0]
    method, score, weights, names, oof, test_pred = best

    sub = pd.DataFrame({"ID": test_ids, "probability": test_pred})
    sub_path = SUBMISSION_DIR / "v5_best.csv"
    sub.to_csv(sub_path, index=False, encoding="utf-8")
    print(f"  저장: {sub_path}")
    print(f"  방법: {method}")
    print(f"  OOF AUC: {score:.6f}")

    rank_path = None
    rank_results = [r for r in results if "rank" in r[0]]
    if rank_results:
        rank_results.sort(key=lambda x: -x[1])
        r_best = rank_results[0]
        sub_rank = pd.DataFrame({"ID": test_ids, "probability": r_best[5]})
        rank_path = SUBMISSION_DIR / "v5_rank.csv"
        sub_rank.to_csv(rank_path, index=False, encoding="utf-8")
        print(f"\n  저장: {rank_path}")
        print(f"  방법: {r_best[0]}")
        print(f"  OOF AUC: {r_best[1]:.6f}")

    print("\n  제출파일 검증...")
    paths_to_check = [(sub_path, "best")]
    if rank_path is not None:
        paths_to_check.append((rank_path, "rank"))

    for path, label in paths_to_check:
        check = pd.read_csv(path)
        errors = []

        if check.shape != sample_like.shape:
            errors.append(f"크기 불일치: {check.shape} vs {sample_like.shape}")
        if not check.columns.equals(sample_like.columns):
            errors.append("컬럼 불일치")
        if not check["ID"].equals(sample_like["ID"]):
            errors.append("ID 불일치")
        if check.isnull().sum().sum() > 0:
            errors.append("결측치 있음")
        if check["probability"].min() < 0 or check["probability"].max() > 1:
            errors.append("확률 범위 오류")

        if errors:
            print(f"    [{label}] 오류: {errors}")
        else:
            print(f"    [{label}] 검증 통과!")

    return sub_path


if __name__ == "__main__":
    total_start = time.time()
    np.random.seed(42)

    print("\n  데이터 로드...")
    if not TRAIN_PATH.exists() or not TEST_PATH.exists():
        raise FileNotFoundError(
            f"train/test 파일을 찾을 수 없습니다.\n"
            f"- 기대 경로: {TRAIN_PATH}\n"
            f"- 기대 경로: {TEST_PATH}"
        )

    train_raw = pd.read_csv(TRAIN_PATH)
    test_raw = pd.read_csv(TEST_PATH)
    test_ids, sample_like = resolve_test_ids()
    print(f"  Train: {train_raw.shape}, Test: {test_raw.shape}")

    print("\n  전처리 시작...")
    X_train, X_test, y, feature_names, cat_cols, train_ids, test_ids_proc = preprocess_for_catboost(
        train_raw, test_raw
    )

    new_oof, new_test, new_auc = train_catboost_5seed(
        X_train, X_test, y, cat_cols, feature_names
    )

    print("\n  기존 OOF 로드...")
    existing = load_existing_oofs(y)

    if existing:
        blend_results = blend_optimization(new_oof, new_test, y, existing)
    else:
        blend_results = [("new_cat_only", new_auc, [1.0], ["NEW_Cat"], new_oof, new_test)]

    sub_path = create_submission(blend_results, test_ids, sample_like)

    total_time = time.time() - total_start
    print("\n" + "=" * 60)
    print("  ★ Step 10 최종 결과 ★")
    print("=" * 60)
    print(f"\n  새 CatBoost 단일 OOF AUC: {new_auc:.6f}")
    print(f"\n  최고 블렌드 OOF AUC: {blend_results[0][1]:.6f}")
    print(f"\n  제출 파일: {sub_path}")
    print(f"  총 소요 시간: {total_time / 60:.1f}분")
    print("=" * 60)