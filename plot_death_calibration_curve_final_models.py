from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import calibration_curve

TABICL_PRED_PATH = Path('/Users/ac/comforter/artifacts/tabicl_death_final/test_predictions_final.csv')
AUTOGLUON_PRED_PATH = Path('/Users/ac/comforter/artifacts/autogluon_death/test_predictions.csv')
GROUPLASSO_PRED_PATH = Path('/Users/ac/comforter/artifacts/group_lasso_death_tuned/test_predictions.csv')
GROUPRIDGE_PRED_PATH = Path('/Users/ac/comforter/artifacts/group_ridge_death_tuned/test_predictions.csv')
RANDOM_FOREST_PRED_PATH = Path('/Users/ac/comforter/artifacts/random_forest_death_tuned/test_predictions.csv')
OUT_PATH = Path('/Users/ac/comforter/artifacts/death_calibration_curve_final_models.png')

# Load prediction artifacts
	

tabicl_df = pd.read_csv(TABICL_PRED_PATH)
autogluon_df = pd.read_csv(AUTOGLUON_PRED_PATH)
group_lasso_df = pd.read_csv(GROUPLASSO_PRED_PATH)
group_ridge_df = pd.read_csv(GROUPRIDGE_PRED_PATH)
rf_df = pd.read_csv(RANDOM_FOREST_PRED_PATH)

# Positive-class probabilities for DeathHospDisch
proba_col_tabicl = 'proba_death'
proba_col_ag = 'proba_death'

y_true_tabicl = tabicl_df['y_true'].to_numpy()
prob_tabicl = tabicl_df[proba_col_tabicl].to_numpy()

y_true_ag = autogluon_df['y_true'].to_numpy()
prob_ag = autogluon_df[proba_col_ag].to_numpy()
y_true_gl = group_lasso_df['y_true'].to_numpy()
prob_gl = group_lasso_df['proba_death'].to_numpy()
y_true_gr = group_ridge_df['y_true'].to_numpy()
prob_gr = group_ridge_df['proba_death'].to_numpy()
y_true_rf = rf_df['y_true'].to_numpy()
prob_rf = rf_df['proba_death'].to_numpy()

frac_pos_tabicl, mean_pred_tabicl = calibration_curve(
    y_true_tabicl,
    prob_tabicl,
    n_bins=10,
    strategy='quantile',
)
frac_pos_ag, mean_pred_ag = calibration_curve(
    y_true_ag,
    prob_ag,
    n_bins=10,
    strategy='quantile',
)
frac_pos_gl, mean_pred_gl = calibration_curve(
    y_true_gl,
    prob_gl,
    n_bins=10,
    strategy='quantile',
)
frac_pos_gr, mean_pred_gr = calibration_curve(
    y_true_gr,
    prob_gr,
    n_bins=10,
    strategy='quantile',
)
frac_pos_rf, mean_pred_rf = calibration_curve(
    y_true_rf,
    prob_rf,
    n_bins=10,
    strategy='quantile',
)

plt.figure(figsize=(6.8, 6.2))
plt.plot([0, 1], [0, 1], '--', color='gray', linewidth=1.2, label='Perfect calibration')
plt.plot(mean_pred_tabicl, frac_pos_tabicl, marker='o', linewidth=1.8, label='TabICL final')
plt.plot(mean_pred_ag, frac_pos_ag, marker='o', linewidth=1.8, label='AutoGluon selected')
plt.plot(mean_pred_gl, frac_pos_gl, marker='o', linewidth=1.8, label='Group Lasso tuned')
plt.plot(mean_pred_gr, frac_pos_gr, marker='o', linewidth=1.8, label='Group Ridge tuned')
plt.plot(mean_pred_rf, frac_pos_rf, marker='o', linewidth=1.8, label='Random Forest tuned')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel('Mean predicted probability')
plt.ylabel('Observed frequency')
plt.title('DeathHospDisch Test Calibration (10 quantile bins)')
plt.grid(alpha=0.3)
plt.legend(loc='upper left')
plt.tight_layout()
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT_PATH, dpi=220)
print(f'Saved {OUT_PATH}')
