from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import auc, roc_curve

TABICL_PRED_PATH = Path('/Users/ac/comforter/artifacts/tabicl_death_final/test_predictions_final.csv')
AUTOGLUON_PRED_PATH = Path('/Users/ac/comforter/artifacts/autogluon_death/test_predictions.csv')
GROUPLASSO_PRED_PATH = Path('/Users/ac/comforter/artifacts/group_lasso_death_tuned/test_predictions.csv')
GROUPRIDGE_PRED_PATH = Path('/Users/ac/comforter/artifacts/group_ridge_death_tuned/test_predictions.csv')
RANDOM_FOREST_PRED_PATH = Path('/Users/ac/comforter/artifacts/random_forest_death_tuned/test_predictions.csv')
OUT_PATH = Path('/Users/ac/comforter/artifacts/death_roc_curve_final_models.png')

# TabICL final predictions
tabicl_df = pd.read_csv(TABICL_PRED_PATH)
autogluon_df = pd.read_csv(AUTOGLUON_PRED_PATH)
group_lasso_df = pd.read_csv(GROUPLASSO_PRED_PATH)
group_ridge_df = pd.read_csv(GROUPRIDGE_PRED_PATH)
rf_df = pd.read_csv(RANDOM_FOREST_PRED_PATH)

# Outcome is binary {0,1}; use model-positive-class probability

y_true_tabicl = tabicl_df['y_true'].to_numpy()

# If AutoGluon model columns change later, fall back gracefully.
y_col = 'proba_death' if 'proba_death' in autogluon_df.columns else 'proba_positive'
y_true_ag = autogluon_df['y_true'].to_numpy()

fpr_tabicl, tpr_tabicl, _ = roc_curve(y_true_tabicl, tabicl_df['proba_death'].to_numpy())
auc_tabicl = auc(fpr_tabicl, tpr_tabicl)

fpr_ag, tpr_ag, _ = roc_curve(y_true_ag, autogluon_df[y_col].to_numpy())
auc_ag = auc(fpr_ag, tpr_ag)

fpr_gl, tpr_gl, _ = roc_curve(group_lasso_df['y_true'].to_numpy(), group_lasso_df['proba_death'].to_numpy())
auc_gl = auc(fpr_gl, tpr_gl)

fpr_gr, tpr_gr, _ = roc_curve(group_ridge_df['y_true'].to_numpy(), group_ridge_df['proba_death'].to_numpy())
auc_gr = auc(fpr_gr, tpr_gr)

fpr_rf, tpr_rf, _ = roc_curve(rf_df['y_true'].to_numpy(), rf_df['proba_death'].to_numpy())
auc_rf = auc(fpr_rf, tpr_rf)

plt.figure(figsize=(6.8, 6.2))
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Chance')
plt.plot(fpr_tabicl, tpr_tabicl, label=f'TabICL final (AUC={auc_tabicl:.4f})', linewidth=2)
plt.plot(fpr_ag, tpr_ag, label=f'AutoGluon selected (AUC={auc_ag:.4f})', linewidth=2)
plt.plot(fpr_gl, tpr_gl, label=f'Group Lasso tuned (AUC={auc_gl:.4f})', linewidth=2)
plt.plot(fpr_gr, tpr_gr, label=f'Group Ridge tuned (AUC={auc_gr:.4f})', linewidth=2)
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest tuned (AUC={auc_rf:.4f})', linewidth=2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve on Test Set: DeathHospDisch')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid(alpha=0.3)
plt.legend(loc='lower right')
plt.tight_layout()
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT_PATH, dpi=220)
print(f'Saved {OUT_PATH}')
print(f'TabICL AUC={auc_tabicl:.6f}')
print(f'AutoGluon AUC={auc_ag:.6f}')
print(f'Group Lasso AUC={auc_gl:.6f}')
print(f'Group Ridge AUC={auc_gr:.6f}')
print(f'Random Forest AUC={auc_rf:.6f}')
