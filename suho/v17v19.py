import os
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss, average_precision_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(BASE_DIR, "result")
y = np.load(os.path.join(RESULT_DIR, "y_train_v19.npy"))
oof_v17 = np.load(os.path.join(RESULT_DIR, "oof_v17_final.npy"))
oof_v19 = np.load(os.path.join(RESULT_DIR, "oof_v19_final.npy"))

best = None
for w in np.arange(0.0, 1.01, 0.05):
    blend = w * oof_v17 + (1 - w) * oof_v19
    auc = roc_auc_score(y, blend)
    ll = log_loss(y, np.clip(blend, 1e-7, 1 - 1e-7))
    ap = average_precision_score(y, blend)
    print(f"w_v17={w:.2f}, auc={auc:.6f}, logloss={ll:.6f}, ap={ap:.6f}")
    if best is None or auc > best[1]:
        best = (w, auc, ll, ap)

print("\nBEST:", best)
