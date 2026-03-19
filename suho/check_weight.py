import numpy as np
import os
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

RESULT_DIR = "./result"

# 1. 데이터 로드
oof_full = np.load(os.path.join(RESULT_DIR, "oof_full.npy"))
oof_nt = np.load(os.path.join(RESULT_DIR, "oof_nt.npy"))
y_true = np.load(os.path.join(RESULT_DIR, "y_true.npy"))
is_tr = np.load(os.path.join(RESULT_DIR, "is_tr_mask.npy"))

# ==========================================
# 2. 가중치 직접 지정 (여기를 수정하세요!)
# ==========================================
w_nt_expert = 0.2  # NT_EXPERT의 비중 (0.0 ~ 1.0)
w_full = 1 - w_nt_expert  # 나머지 FULL_CB의 비중
# ==========================================

# 3. 결합 계산
final_oof = np.zeros_like(oof_full)

# 이식 그룹: Full 모델 100%
final_oof[is_tr] = oof_full[is_tr]

# 비이식 그룹: 지정한 가중치로 Rank Blending
# (점수 격차 문제를 막기 위해 Rank를 씁니다)
full_nt_rank = rankdata(oof_full[~is_tr])
expert_nt_rank = rankdata(oof_nt[~is_tr])

final_oof[~is_tr] = (full_nt_rank * w_full) + (expert_nt_rank * w_nt_expert)

# 4. 결과 출력
current_auc = roc_auc_score(y_true, final_oof)

print("=" * 50)
print(f"설정된 가중치: Full({w_full:.2f}) : NT_Expert({w_nt_expert:.2f})")
print(f"현재 조합의 Combined AUC: {current_auc:.6f}")
print("=" * 50)
