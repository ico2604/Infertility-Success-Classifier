# =========================================================
# 난임 환자 임신 성공 여부 예측 EDA
# ---------------------------------------------------------
# 이 스크립트의 목적
# 1) train/test 데이터를 불러온다.
# 2) 컬럼 구조, 결측치, 범주형/수치형 분포를 확인한다.
# 3) 타겟(임신 성공 여부)과 관련 있는 패턴을 찾는다.
# 4) 모델링 전에 유용한 파생변수 후보를 만든다.
# 5) 결과를 CSV/PNG 파일로 outputs/eda 폴더에 저장한다.
#
# 실행 환경
# - VSCode
# - Python 3.x
# - data/train.csv, data/test.csv 가 존재해야 함
#
# 실행 방법
# - VSCode 터미널에서:
#   python src/eda.py
# =========================================================

import os
import re
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier


# =========================================================
# 0. 기본 환경 설정
# =========================================================
# 한글이 포함된 그래프 제목/축 라벨이 깨지지 않도록 설정
# Windows 환경에서는 보통 "Malgun Gothic"이 잘 동작함
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 120

# 현재 파일(src/eda.py)의 상위 폴더를 기준으로 프로젝트 루트 계산
# 예: yysop/src/eda.py -> yysop/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 데이터 폴더와 결과 저장 폴더 경로 설정
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "eda")

# 결과 저장 폴더가 없으면 생성
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 데이터 파일 경로
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")

# 타겟 컬럼명 / ID 컬럼명
TARGET_COL = "임신 성공 여부"
ID_COL = "ID"

# 재현성을 위한 random seed
SEED = 42


# =========================================================
# 1. 자주 쓰는 유틸 함수 정의
# =========================================================
def print_section(title: str):
    """
    콘솔 출력 시 구간이 잘 보이도록 제목을 크게 표시하는 함수
    """
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def save_csv(df: pd.DataFrame, filename: str):
    """
    DataFrame을 outputs/eda/ 아래에 csv로 저장하는 함수
    - utf-8-sig로 저장해 엑셀에서 한글 깨짐을 줄임
    """
    path = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[저장 완료] {path}")


def save_figure(filename: str):
    """
    현재 matplotlib figure를 파일로 저장하는 함수
    """
    path = os.path.join(OUTPUT_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    print(f"[그림 저장] {path}")
    plt.close()


def safe_divide(a, b):
    """
    0으로 나누는 문제를 피하면서 비율형 파생변수를 만들기 위한 함수

    예:
    - 총 생성 배아 수 / 수집된 신선 난자 수
    - IVF 임신 횟수 / IVF 시술 횟수

    분모가 0이거나 결측이면 결과를 NaN으로 둔다.
    """
    a = pd.Series(a)
    b = pd.Series(b)
    result = np.where((b.isna()) | (b == 0), np.nan, a / b)
    return result


def parse_korean_count(x):
    """
    '0회', '1회', '2회', '3회 이상'처럼
    숫자처럼 보이지만 문자열(object)로 저장된 컬럼을 숫자로 바꾼다.

    예:
    - '0회' -> 0
    - '1회' -> 1
    - '3회 이상' -> 3
    - '없음' -> 0
    - NaN -> NaN

    주의:
    - '3회 이상'을 3으로 단순 치환하는 건 완벽한 해석은 아니지만,
      EDA와 baseline 모델링 단계에서는 꽤 유용하다.
    """
    if pd.isna(x):
        return np.nan

    x = str(x).strip()

    # 값이 숫자만으로 이루어진 경우
    if re.fullmatch(r"\d+", x):
        return float(x)

    # 문자열 내부에서 숫자를 찾음
    m = re.search(r"(\d+)", x)
    if m:
        return float(m.group(1))

    # 0으로 해석 가능한 표현들
    zero_tokens = ["없음", "무", "미실시", "해당없음"]
    if x in zero_tokens:
        return 0.0

    return np.nan


def age_to_numeric(x):
    """
    나이 범주형 문자열을 수치형으로 근사 변환한다.

    예시:
    - '만35-39세' -> 37
    - '35세 이상' -> 35
    - '34세 이하' -> 34
    - '알 수 없음' -> NaN

    이유:
    - '시술 당시 나이'가 object일 경우, 순서 정보가 손실됨
    - 모델링 전에 대략적인 연령 축을 만들면 신호를 보기 쉬워짐
    """
    if pd.isna(x):
        return np.nan

    x = str(x).strip()

    unknown_tokens = ["알 수 없음", "미상", "불명", "unknown", "Unknown"]
    if x in unknown_tokens:
        return np.nan

    nums = re.findall(r"\d+", x)

    # 범위가 잡힌 경우: 예) 35-39 -> 평균 37로 근사
    if len(nums) >= 2:
        return (float(nums[0]) + float(nums[1])) / 2

    # 숫자가 하나면 그 값을 사용
    if len(nums) == 1:
        return float(nums[0])

    return np.nan


def detect_binary_columns(df: pd.DataFrame):
    """
    0/1 형태의 이진 컬럼을 자동 탐색하는 함수

    예:
    - 배란 자극 여부
    - 남성 주 불임 원인
    - 불임 원인 - 배란 장애
    - 동결 배아 사용 여부

    이런 컬럼들은 별도로 보면 타겟과의 관계를 쉽게 확인할 수 있다.
    """
    binary_cols = []

    for col in df.columns:
        nunique = df[col].dropna().nunique()
        unique_vals = set(df[col].dropna().unique().tolist())

        # 결측 제외하고 값이 2개 이하이며, 그 값이 0/1로만 구성되어 있는 경우
        if nunique <= 2 and unique_vals.issubset({0, 1, 0.0, 1.0}):
            binary_cols.append(col)

    return binary_cols


# =========================================================
# 2. 데이터 로드
# =========================================================
print_section("2. 데이터 로드")

# train/test 파일 읽기
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

print("train shape:", train.shape)
print("test shape :", test.shape)

print("\n[train 상위 5행]")
print(train.head())

print("\n[test 상위 5행]")
print(test.head())


# =========================================================
# 3. 기본 정보 확인
# =========================================================
print_section("3. 기본 정보 확인")

# info()는 컬럼별 dtype, non-null 개수를 확인할 수 있어
# 어떤 컬럼이 object인지, 어떤 컬럼이 numeric인지 빠르게 파악 가능
print("[train info]")
print(train.info())

print("\n[test info]")
print(test.info())

# 타겟 분포 확인
# 임신 성공(1) / 실패(0) 비율을 확인해 class imbalance 정도를 파악
print("\n[타겟 분포 - 개수]")
print(train[TARGET_COL].value_counts(dropna=False))

print("\n[타겟 분포 - 비율]")
print(train[TARGET_COL].value_counts(normalize=True, dropna=False))

# 타겟 분포 시각화
plt.figure(figsize=(6, 4))
train[TARGET_COL].value_counts().sort_index().plot(kind="bar")
plt.title("임신 성공 여부 분포")
plt.xlabel("임신 성공 여부")
plt.ylabel("Count")
save_figure("01_target_distribution.png")


# =========================================================
# 4. 컬럼 요약 테이블 생성
# =========================================================
print_section("4. 컬럼 요약 테이블 생성")

# 모든 컬럼에 대해:
# - dtype
# - 고유값 개수
# - 결측 개수 / 결측 비율
# - 샘플 값
# 을 정리해두면, 이후 데이터 이해가 훨씬 쉬워진다.
summary_rows = []

for col in train.columns:
    non_null_values = train[col].dropna().astype(str).tolist()

    sample1 = non_null_values[0] if len(non_null_values) > 0 else None
    sample2 = non_null_values[1] if len(non_null_values) > 1 else None

    summary_rows.append({
        "column": col,
        "dtype": str(train[col].dtype),
        "nunique_including_nan": train[col].nunique(dropna=False),
        "missing_count": train[col].isna().sum(),
        "missing_ratio_percent": round(train[col].isna().mean() * 100, 4),
        "sample1": sample1,
        "sample2": sample2,
    })

summary_df = pd.DataFrame(summary_rows).sort_values(
    by=["missing_ratio_percent", "nunique_including_nan"],
    ascending=[False, False]
)

print(summary_df.head(20))
save_csv(summary_df, "02_column_summary.csv")


# =========================================================
# 5. 결측치 분석
# =========================================================
print_section("5. 결측치 분석")

# 의료 데이터에서 결측은 단순 누락이 아니라
# "해당 시술을 하지 않음"을 의미하는 경우가 많다.
# 따라서 결측률 자체를 먼저 정리해두는 것이 중요하다.
missing_df = pd.DataFrame({
    "column": train.columns,
    "missing_count": train.isnull().sum().values,
    "missing_ratio_percent": (train.isnull().mean().values * 100)
}).sort_values("missing_ratio_percent", ascending=False)

print(missing_df.head(30))
save_csv(missing_df, "03_missing_summary.csv")

# 결측률 막대그래프
plt.figure(figsize=(10, 14))
plt.barh(missing_df["column"], missing_df["missing_ratio_percent"])
plt.gca().invert_yaxis()
plt.title("컬럼별 결측 비율")
plt.xlabel("Missing Ratio (%)")
save_figure("04_missing_ratio_barh.png")


# =========================================================
# 6. 컬럼 타입 분리
# =========================================================
print_section("6. 컬럼 타입 분리")

# object 컬럼: 범주형 / 문자열 숫자형 / 코드형 가능성이 큼
object_cols = train.select_dtypes(include=["object"]).columns.tolist()

# numeric 컬럼: int64, float64
numeric_cols = train.select_dtypes(include=["int64", "float64"]).columns.tolist()

# 타겟은 따로 다룰 예정이므로 수치형 리스트에서 제외
if TARGET_COL in numeric_cols:
    numeric_cols.remove(TARGET_COL)

print("object 컬럼 수 :", len(object_cols))
print("numeric 컬럼 수:", len(numeric_cols))

print("\n[object 컬럼 목록]")
print(object_cols)

print("\n[numeric 컬럼 목록]")
print(numeric_cols)


# =========================================================
# 7. 범주형(object) 컬럼 상위 값 확인
# =========================================================
print_section("7. 범주형(object) 컬럼 상위 값 확인")

# object 컬럼은 실제 의미를 봐야 한다.
# 예를 들어:
# - '시술 당시 나이'는 순서가 있는 범주형일 수 있고
# - '총 시술 횟수'는 사실상 숫자형일 수 있으며
# - '시술 유형'은 순수 범주형일 수 있다.
category_preview_rows = []

for col in object_cols:
    vc = train[col].value_counts(dropna=False).head(10)

    category_preview_rows.append({
        "column": col,
        "nunique": train[col].nunique(dropna=False),
        "top_values": " | ".join([str(idx) for idx in vc.index.tolist()[:10]])
    })

    print(f"\n[{col}] 상위 10개 값")
    print(vc)

category_preview_df = pd.DataFrame(category_preview_rows).sort_values("nunique", ascending=False)
save_csv(category_preview_df, "05_object_column_preview.csv")


# =========================================================
# 8. 문자열 숫자형 컬럼 변환
# =========================================================
print_section("8. 문자열 숫자형 컬럼 변환")

# 이런 컬럼들은 object로 읽혀도 실제 의미는 "횟수"
# 따라서 숫자로 바꿔두면 훨씬 분석하기 좋아진다.
count_like_cols = [
    "총 시술 횟수",
    "클리닉 내 총 시술 횟수",
    "IVF 시술 횟수",
    "DI 시술 횟수",
    "총 임신 횟수",
    "IVF 임신 횟수",
    "DI 임신 횟수",
    "총 출산 횟수",
    "IVF 출산 횟수",
    "DI 출산 횟수"
]

for col in count_like_cols:
    if col in train.columns:
        train[col + "_num"] = train[col].apply(parse_korean_count)
        test[col + "_num"] = test[col].apply(parse_korean_count)
        print(f"[변환 완료] {col} -> {col + '_num'}")

# 나이형 범주를 수치형으로 근사 변환
age_like_cols = [
    "시술 당시 나이",
    "난자 기증자 나이",
    "정자 기증자 나이"
]

for col in age_like_cols:
    if col in train.columns:
        train[col + "_num"] = train[col].apply(age_to_numeric)
        test[col + "_num"] = test[col].apply(age_to_numeric)
        print(f"[변환 완료] {col} -> {col + '_num'}")


# =========================================================
# 9. 숫자형 컬럼 기초 통계
# =========================================================
print_section("9. 숫자형 컬럼 기초 통계")

# 수치형 컬럼에 대해 count, mean, std, min, max 등을 확인
# 어떤 변수가 심하게 치우쳤는지, 값 범위가 큰지 확인 가능
numeric_describe = train.describe(include=[np.number]).T.reset_index()
numeric_describe = numeric_describe.rename(columns={"index": "column"})

print(numeric_describe.head(20))
save_csv(numeric_describe, "06_numeric_describe.csv")


# =========================================================
# 10. 숫자형 컬럼 분포 시각화
# =========================================================
print_section("10. 숫자형 컬럼 분포 시각화")

# 모든 수치형 컬럼에 대해 히스토그램 저장
# 분포가 한쪽에 몰려 있는지, 이상치가 많은지, 대부분 0인지 확인할 수 있다.
plot_numeric_cols = [col for col in train.select_dtypes(include=[np.number]).columns if col != TARGET_COL]

for col in plot_numeric_cols:
    if train[col].notna().sum() == 0:
        continue

    plt.figure(figsize=(6, 4))
    train[col].dropna().hist(bins=30)
    plt.title(f"{col} 분포")
    plt.xlabel(col)
    plt.ylabel("Count")

    # 파일명에 사용할 수 없거나 보기 불편한 문자를 "_"로 치환
    safe_filename = re.sub(r"[\\/:*?\"<>| ]+", "_", col)
    save_figure(f"hist_{safe_filename}.png")


# =========================================================
# 11. 타겟별 숫자형 평균 비교
# =========================================================
print_section("11. 타겟별 숫자형 평균 비교")

# 임신 성공(1)과 실패(0) 그룹에서 평균 차이를 보면
# 어떤 수치형 변수가 예측 신호를 가지는지 빠르게 확인 가능
numeric_compare_rows = []

for col in plot_numeric_cols:
    grp = train.groupby(TARGET_COL)[col].mean()
    mean_0 = grp.get(0, np.nan)
    mean_1 = grp.get(1, np.nan)

    numeric_compare_rows.append({
        "column": col,
        "target_0_mean": mean_0,
        "target_1_mean": mean_1,
        "diff_1_minus_0": mean_1 - mean_0 if pd.notna(mean_0) and pd.notna(mean_1) else np.nan
    })

numeric_compare_df = pd.DataFrame(numeric_compare_rows).sort_values("diff_1_minus_0", ascending=False)
print(numeric_compare_df.head(30))
save_csv(numeric_compare_df, "07_numeric_target_compare.csv")


# =========================================================
# 12. 숫자형 변수 타겟별 박스플롯
# =========================================================
print_section("12. 숫자형 변수 타겟별 박스플롯")

# 박스플롯은 타겟 0/1별 분포 차이를 보기 좋다.
# 단, 값 종류가 너무 적은 컬럼은 박스플롯 의미가 약해서 제외한다.
for col in plot_numeric_cols:
    if train[col].dropna().nunique() < 3:
        continue

    tmp = train[[col, TARGET_COL]].dropna()
    if len(tmp) == 0:
        continue

    plt.figure(figsize=(6, 4))
    tmp.boxplot(column=col, by=TARGET_COL)
    plt.title(f"{col} vs {TARGET_COL}")
    plt.suptitle("")

    safe_filename = re.sub(r"[\\/:*?\"<>| ]+", "_", col)
    save_figure(f"box_{safe_filename}.png")


# =========================================================
# 13. 범주형 컬럼별 성공률 분석
# =========================================================
print_section("13. 범주형 컬럼별 성공률 분석")

# 범주형 컬럼에서 각 카테고리별:
# - count
# - success_rate(임신 성공률)
# 을 계산한다.
#
# 이 분석은 특히 중요하다.
# 예:
# - 시술 유형별 성공률
# - 시술 당시 나이 구간별 성공률
# - 난자 출처 / 정자 출처별 성공률
cat_stats_all = []

for col in object_cols:
    tmp = train[[col, TARGET_COL]].copy()

    # 결측을 "MISSING"으로 명시적으로 넣어 결측 자체도 하나의 범주처럼 본다.
    tmp[col] = tmp[col].fillna("MISSING").astype(str)

    stats = tmp.groupby(col)[TARGET_COL].agg(["count", "mean"]).reset_index()
    stats.columns = [col, "count", "success_rate"]
    stats = stats.sort_values("count", ascending=False)

    # 나중에 통합 csv로 저장하기 위해 컬럼명 통일
    stats["column_name"] = col
    cat_stats_all.append(stats.rename(columns={col: "category"}))

    print(f"\n[{col}] 상위 10개 카테고리")
    print(stats.head(10))

    # 자주 등장하는 상위 20개 카테고리만 그림으로 저장
    top_stats = stats.head(20)

    plt.figure(figsize=(10, 4))
    plt.bar(top_stats[col].astype(str), top_stats["success_rate"])
    plt.xticks(rotation=45, ha="right")
    plt.title(f"{col} 상위 카테고리별 임신 성공률")
    plt.xlabel(col)
    plt.ylabel("Success Rate")

    safe_filename = re.sub(r"[\\/:*?\"<>| ]+", "_", col)
    save_figure(f"cat_success_{safe_filename}.png")

cat_stats_df = pd.concat(cat_stats_all, ignore_index=True)
save_csv(cat_stats_df, "08_categorical_success_rate.csv")


# =========================================================
# 14. 이진(binary) 컬럼 분석
# =========================================================
print_section("14. 이진(binary) 컬럼 분석")

# 0/1 컬럼들은 의료적으로 해석이 쉽다.
# 예:
# - 배란 자극 여부
# - 동결 배아 사용 여부
# - 남성/여성 불임 원인 관련 플래그
#
# 값이 0인 집단 / 1인 집단의 성공률 차이를 보면 의미를 빠르게 파악할 수 있다.
binary_cols = detect_binary_columns(train)
binary_cols = [col for col in binary_cols if col not in [TARGET_COL]]

binary_result_rows = []

for col in binary_cols:
    grp = train.groupby(col)[TARGET_COL].agg(["count", "mean"]).reset_index()
    grp["column"] = col
    binary_result_rows.append(grp)

    print(f"\n[{col}]")
    print(grp)

binary_result_df = pd.concat(binary_result_rows, ignore_index=True)
save_csv(binary_result_df, "09_binary_column_analysis.csv")


# =========================================================
# 15. 결측 자체의 의미 확인
# =========================================================
print_section("15. 결측 자체의 의미 확인")

# 의료 데이터에서는 결측이 "정보"일 수 있다.
# 예:
# - PGS/PGD 관련 컬럼이 결측인 것은 해당 시술을 안 했다는 뜻일 수 있음
# - 배아 해동 관련 컬럼 결측은 동결/해동 절차가 없었다는 뜻일 수 있음
#
# 따라서 각 컬럼에 대해 "결측 여부(missing_flag)" 자체가
# 타겟과 관련이 있는지 확인한다.
missing_signal_rows = []

for col in train.columns:
    if col in [ID_COL, TARGET_COL]:
        continue

    miss_flag = train[col].isna().astype(int)

    # 전부 결측이거나 전부 비결측이면 비교 의미가 없음
    if miss_flag.nunique() < 2:
        continue

    tmp = pd.DataFrame({
        "missing_flag": miss_flag,
        TARGET_COL: train[TARGET_COL]
    })

    grp = tmp.groupby("missing_flag")[TARGET_COL].agg(["count", "mean"]).reset_index()
    grp["column"] = col
    missing_signal_rows.append(grp)

missing_signal_df = pd.concat(missing_signal_rows, ignore_index=True)
print(missing_signal_df.head(30))
save_csv(missing_signal_df, "10_missing_signal_analysis.csv")


# =========================================================
# 16. 파생변수 생성
# =========================================================
print_section("16. 파생변수 생성")

# IVF/난임 데이터에서는 원시 count보다 "비율"이 더 직관적인 경우가 많다.
# 예:
# - 배아가 몇 개 생성됐는가? 보다
#   난자 대비 배아 생성 효율이 어땠는가?
# - IVF를 몇 번 했는가? 보다
#   과거 IVF 성공률이 어땠는가?
#
# 이런 비율형 feature는 모델 성능에 도움될 가능성이 높다.

# 16-1. 총 생성 배아 수 / 수집된 신선 난자 수
# 난자 대비 배아 생성 효율의 거친 proxy
if {"총 생성 배아 수", "수집된 신선 난자 수"}.issubset(train.columns):
    train["배아생성효율"] = safe_divide(train["총 생성 배아 수"], train["수집된 신선 난자 수"])
    test["배아생성효율"] = safe_divide(test["총 생성 배아 수"], test["수집된 신선 난자 수"])

# 16-2. 미세주입에서 생성된 배아 수 / 미세주입된 난자 수
# ICSI 관련 수정 효율 proxy
if {"미세주입에서 생성된 배아 수", "미세주입된 난자 수"}.issubset(train.columns):
    train["미세주입수정효율"] = safe_divide(train["미세주입에서 생성된 배아 수"], train["미세주입된 난자 수"])
    test["미세주입수정효율"] = safe_divide(test["미세주입에서 생성된 배아 수"], test["미세주입된 난자 수"])

# 16-3. 이식된 배아 수 / 총 생성 배아 수
# 생성된 배아 중 실제 이식으로 이어진 비율
if {"이식된 배아 수", "총 생성 배아 수"}.issubset(train.columns):
    train["배아이식비율"] = safe_divide(train["이식된 배아 수"], train["총 생성 배아 수"])
    test["배아이식비율"] = safe_divide(test["이식된 배아 수"], test["총 생성 배아 수"])

# 16-4. 저장된 배아 수 / 총 생성 배아 수
# 생성된 배아 중 동결/저장으로 이어진 비율
if {"저장된 배아 수", "총 생성 배아 수"}.issubset(train.columns):
    train["배아저장비율"] = safe_divide(train["저장된 배아 수"], train["총 생성 배아 수"])
    test["배아저장비율"] = safe_divide(test["저장된 배아 수"], test["총 생성 배아 수"])

# 16-5. 과거 IVF 임신률 = IVF 임신 횟수 / IVF 시술 횟수
if {"IVF 임신 횟수_num", "IVF 시술 횟수_num"}.issubset(train.columns):
    train["과거IVF임신률"] = safe_divide(train["IVF 임신 횟수_num"], train["IVF 시술 횟수_num"])
    test["과거IVF임신률"] = safe_divide(test["IVF 임신 횟수_num"], test["IVF 시술 횟수_num"])

# 16-6. 전체 시술 대비 임신률 = 총 임신 횟수 / 총 시술 횟수
if {"총 임신 횟수_num", "총 시술 횟수_num"}.issubset(train.columns):
    train["전체시술대비임신률"] = safe_divide(train["총 임신 횟수_num"], train["총 시술 횟수_num"])
    test["전체시술대비임신률"] = safe_divide(test["총 임신 횟수_num"], test["총 시술 횟수_num"])

# 생성된 파생변수 리스트
engineered_cols = [
    "배아생성효율",
    "미세주입수정효율",
    "배아이식비율",
    "배아저장비율",
    "과거IVF임신률",
    "전체시술대비임신률"
]
engineered_cols = [col for col in engineered_cols if col in train.columns]

# 파생변수도 타겟 0/1별 평균 차이 확인
eng_compare_rows = []

for col in engineered_cols:
    grp = train.groupby(TARGET_COL)[col].mean()
    eng_compare_rows.append({
        "column": col,
        "target_0_mean": grp.get(0, np.nan),
        "target_1_mean": grp.get(1, np.nan),
        "diff_1_minus_0": grp.get(1, np.nan) - grp.get(0, np.nan)
    })

eng_compare_df = pd.DataFrame(eng_compare_rows).sort_values("diff_1_minus_0", ascending=False)
print(eng_compare_df)
save_csv(eng_compare_df, "11_engineered_feature_compare.csv")


# =========================================================
# 17. 수치형 변수와 타겟의 상관관계
# =========================================================
print_section("17. 수치형 변수와 타겟의 상관관계")

# 수치형 변수와 타겟의 단순 상관관계를 본다.
# 주의:
# - 상관계수는 선형 관계만 일부 반영하므로 절대 기준은 아님
# - 그래도 EDA에서 대략적인 신호를 파악하는 데는 유용함
numeric_all = train.select_dtypes(include=[np.number]).columns.tolist()

if TARGET_COL in numeric_all:
    corr_df = train[numeric_all].corr(numeric_only=True)
    target_corr = corr_df[TARGET_COL].drop(TARGET_COL).sort_values(ascending=False)

    corr_result = target_corr.reset_index()
    corr_result.columns = ["column", "correlation_with_target"]

    print(corr_result.head(30))
    save_csv(corr_result, "12_target_correlation_numeric.csv")

    plt.figure(figsize=(8, 10))
    corr_result.head(25).sort_values("correlation_with_target").plot(
        kind="barh", x="column", y="correlation_with_target", legend=False
    )
    plt.title("타겟과의 상관관계 상위 25개")
    save_figure("13_target_correlation_top25.png")


# =========================================================
# 18. 간단한 baseline 모델로 feature signal 확인
# =========================================================
print_section("18. 간단한 baseline 모델")

# 이 단계의 목적은 "최종 모델 만들기"가 아니라,
# 현재 feature들만으로 어느 정도 ROC-AUC가 나오는지,
# 그리고 어떤 feature가 중요하게 보이는지 거칠게 확인하는 것이다.
eda_model_df = train.copy()

# 결측 처리
# - object는 "MISSING"이라는 문자열로 채움
# - numeric은 -999 같은 sentinel 값으로 채움
# 정교한 전처리는 아니지만, EDA용 baseline에는 충분하다.
for col in eda_model_df.columns:
    if col == TARGET_COL:
        continue

    if eda_model_df[col].dtype == "object":
        eda_model_df[col] = eda_model_df[col].fillna("MISSING")
    else:
        eda_model_df[col] = eda_model_df[col].fillna(-999)

# object 컬럼을 factorize로 정수 인코딩
# 주의:
# - 이 방식은 CatBoost용 최적 인코딩은 아니고,
#   빠른 baseline 확인용이다.
for col in eda_model_df.columns:
    if col not in [ID_COL, TARGET_COL] and eda_model_df[col].dtype == "object":
        eda_model_df[col] = pd.factorize(eda_model_df[col])[0]

# 입력 X / 타겟 y 구성
X = eda_model_df.drop(columns=[ID_COL, TARGET_COL], errors="ignore")
y = eda_model_df[TARGET_COL]

# RandomForest는 빠르게 중요도를 볼 수 있어 EDA 단계에서 자주 사용됨
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_leaf=3,
    random_state=SEED,
    n_jobs=-1
)

# ROC-AUC 기준 5-fold CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
scores = cross_val_score(rf_model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)

print("CV ROC-AUC scores:", scores)
print("Mean ROC-AUC     :", scores.mean())

# 전체 train으로 다시 학습 후 feature importance 확인
rf_model.fit(X, y)

feature_importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance": rf_model.feature_importances_
}).sort_values("importance", ascending=False)

print(feature_importance_df.head(30))
save_csv(feature_importance_df, "14_rf_feature_importance.csv")

plt.figure(figsize=(8, 10))
feature_importance_df.head(30).sort_values("importance").plot(
    kind="barh", x="feature", y="importance", legend=False
)
plt.title("RandomForest Feature Importance Top 30")
save_figure("15_rf_feature_importance_top30.png")


# =========================================================
# 19. train/test 컬럼 차이 확인
# =========================================================
print_section("19. train/test 컬럼 차이 확인")

# 실제 대회에서는 train에만 타겟이 있고 test에는 없는 게 정상
# 그 외에 다른 컬럼 차이가 없는지 미리 확인해두면 나중에 오류를 줄일 수 있다.
train_cols = set(train.columns)
test_cols = set(test.columns)

only_in_train = sorted(list(train_cols - test_cols))
only_in_test = sorted(list(test_cols - train_cols))

print("[train에만 있는 컬럼]")
print(only_in_train)

print("\n[test에만 있는 컬럼]")
print(only_in_test)

col_diff_df = pd.DataFrame({
    "only_in_train": pd.Series(only_in_train),
    "only_in_test": pd.Series(only_in_test)
})
save_csv(col_diff_df, "16_train_test_column_diff.csv")


# =========================================================
# 20. 가공된 train snapshot 저장
# =========================================================
print_section("20. 가공된 train snapshot 저장")

# EDA 과정에서 생성한 _num 컬럼, 파생변수들을 포함한 snapshot 저장
# 이후 전처리/모델링 코드 작성 시 참고용으로 사용할 수 있다.
processed_snapshot_path = os.path.join(OUTPUT_DIR, "17_train_eda_snapshot.csv")
train.to_csv(processed_snapshot_path, index=False, encoding="utf-8-sig")
print(f"[저장 완료] {processed_snapshot_path}")


# =========================================================
# 21. 종료 메시지
# =========================================================
print_section("21. EDA 완료")
print(f"결과물 저장 폴더: {OUTPUT_DIR}")
print("주요 확인 파일:")
print("- 02_column_summary.csv")
print("- 03_missing_summary.csv")
print("- 07_numeric_target_compare.csv")
print("- 08_categorical_success_rate.csv")
print("- 10_missing_signal_analysis.csv")
print("- 11_engineered_feature_compare.csv")
print("- 14_rf_feature_importance.csv")