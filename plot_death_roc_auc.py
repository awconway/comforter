from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import json

# Inputs
TABICL_EXP_PATH = Path('/Users/ac/comforter/artifacts/tabicl_death_experiments/leaderboard.csv')
TABICL_STAGE2_PATH = Path('/Users/ac/comforter/artifacts/tabicl_death_stage2/stage2_leaderboard.csv')
TABICL_FINAL_PATH = Path('/Users/ac/comforter/artifacts/tabicl_death_final/final_report.json')
AUTOGLUON_PATH = Path('/Users/ac/comforter/artifacts/autogluon_death/leaderboard_test.csv')
GROUPLASSO_TUNED_PATH = Path('/Users/ac/comforter/artifacts/group_lasso_death_tuned/report.json')
GROUPRIDGE_TUNED_PATH = Path('/Users/ac/comforter/artifacts/group_ridge_death_tuned/report.json')
RANDOM_FOREST_PATH = Path('/Users/ac/comforter/artifacts/random_forest_death_tuned/report.json')

OUT_PATH = Path('/Users/ac/comforter/artifacts/death_roc_auc_comparison.png')


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors='coerce')


# TabICL experiment sweep (all configs)
exp_df = pd.read_csv(TABICL_EXP_PATH)
exp_df['roc_auc'] = _safe_numeric(exp_df['test_roc_auc'])
exp_df = exp_df.dropna(subset=['roc_auc']).sort_values('roc_auc', ascending=False)
# Keep top 8 to avoid chart clutter
exp_top = exp_df.head(8).copy()
exp_top['model'] = 'TabICL_EXP: ' + exp_top['experiment'].astype(str)
exp_top['source'] = 'TabICL exp'

# TabICL stage-2 sweep (smaller set)
stage2_df = pd.read_csv(TABICL_STAGE2_PATH)
stage2_df['roc_auc'] = _safe_numeric(stage2_df['test_roc_auc'])
stage2_df = stage2_df.dropna(subset=['roc_auc']).sort_values('roc_auc', ascending=False)
stage2_top = stage2_df.head(4).copy()
stage2_top['model'] = 'TabICL_S2: ' + stage2_top['experiment'].astype(str)
stage2_top['source'] = 'TabICL stage2'

# TabICL final locked model
final_report = json.loads(TABICL_FINAL_PATH.read_text(encoding='utf-8'))
final_row = {
    'model': 'TabICL_FINAL: Train+Val Locked',
    'roc_auc': float(final_report['test_probability_metrics']['roc_auc']),
    'source': 'TabICL final',
}
final_df = pd.DataFrame([final_row])

# AutoGluon leaderboard (test set)
ag_df = pd.read_csv(AUTOGLUON_PATH)
ag_df['roc_auc'] = _safe_numeric(ag_df['score_test'])
ag_df = ag_df.dropna(subset=['roc_auc']).sort_values('roc_auc', ascending=False)
ag_top = ag_df.head(8).copy()
ag_top['model'] = 'AutoGluon: ' + ag_top['model'].astype(str)
ag_top['source'] = 'AutoGluon'

# Group Lasso tuned result (test set)
group_lasso_df = pd.DataFrame([])
if GROUPLASSO_TUNED_PATH.exists():
    gl_report = json.loads(GROUPLASSO_TUNED_PATH.read_text(encoding='utf-8'))
    gl_auc = gl_report.get('test_probability_metrics', {}).get('roc_auc')
    if gl_auc is not None:
        group_lasso_df = pd.DataFrame([
            {
                'model': 'GroupLasso_Tuned: Train+Val best',
                'roc_auc': float(gl_auc),
                'source': 'GroupLasso tuned',
            }
        ])

# Group Ridge tuned result (test set)
group_ridge_df = pd.DataFrame([])
if GROUPRIDGE_TUNED_PATH.exists():
    rg_report = json.loads(GROUPRIDGE_TUNED_PATH.read_text(encoding='utf-8'))
    rg_auc = rg_report.get('test_probability_metrics', {}).get('roc_auc')
    if rg_auc is not None:
        group_ridge_df = pd.DataFrame([
            {
                'model': 'GroupRidge_Tuned: Train+Val best',
                'roc_auc': float(rg_auc),
                'source': 'GroupRidge tuned',
            }
        ])

# Random Forest tuned result (test set)
random_forest_df = pd.DataFrame([])
if RANDOM_FOREST_PATH.exists():
    rf_report = json.loads(RANDOM_FOREST_PATH.read_text(encoding='utf-8'))
    rf_auc = rf_report.get('test_probability_metrics', {}).get('roc_auc')
    if rf_auc is not None:
        random_forest_df = pd.DataFrame([
            {
                'model': 'RandomForest_Tuned: Train+Val best',
                'roc_auc': float(rf_auc),
                'source': 'RandomForest tuned',
            }
        ])

# Combine
combined = pd.concat([
    exp_top[['model', 'roc_auc', 'source']],
    stage2_top[['model', 'roc_auc', 'source']],
    final_df,
    ag_top[['model', 'roc_auc', 'source']],
    group_lasso_df[['model', 'roc_auc', 'source']],
    group_ridge_df[['model', 'roc_auc', 'source']],
    random_forest_df[['model', 'roc_auc', 'source']],
], ignore_index=True)
combined = combined.sort_values('roc_auc', ascending=False).reset_index(drop=True)

# Plot
plt.figure(figsize=(13, max(6, 0.35 * len(combined))))
ax = plt.gca()
bar_positions = range(len(combined))
colors = combined['source'].map(
    {
        'TabICL final': '#2ca02c',
        'TabICL exp': '#1f77b4',
        'TabICL stage2': '#17becf',
        'AutoGluon': '#ff7f0e',
        'GroupLasso tuned': '#9467bd',
        'GroupRidge tuned': '#8c564b',
        'RandomForest tuned': '#e377c2',
    }
)

ax.barh(
    bar_positions,
    combined['roc_auc'],
    color=colors,
)
ax.set_yticks(list(bar_positions))
ax.set_yticklabels(combined['model'])
ax.invert_yaxis()
ax.set_xlim(0.0, 1.0)
ax.set_xlabel('Test ROC-AUC')
ax.set_title('DeathHospDisch Test ROC-AUC Across Models')
ax.grid(axis='x', alpha=0.3)

# Add value labels
for pos, score in zip(bar_positions, combined['roc_auc']):
    ax.text(score + 0.003, pos, f'{score:.3f}', va='center', fontsize=8)

# Compact legend
legend_handles = [
    plt.Rectangle((0, 0), 1, 1, color='#2ca02c', label='TabICL final'),
    plt.Rectangle((0, 0), 1, 1, color='#1f77b4', label='TabICL exp'),
    plt.Rectangle((0, 0), 1, 1, color='#17becf', label='TabICL stage2'),
    plt.Rectangle((0, 0), 1, 1, color='#ff7f0e', label='AutoGluon'),
    plt.Rectangle((0, 0), 1, 1, color='#9467bd', label='GroupLasso tuned'),
    plt.Rectangle((0, 0), 1, 1, color='#8c564b', label='GroupRidge tuned'),
    plt.Rectangle((0, 0), 1, 1, color='#e377c2', label='RandomForest tuned'),
]
ax.legend(handles=legend_handles, loc='lower right')

plt.tight_layout()
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT_PATH, dpi=220)
plt.close()

print(f'Saved {OUT_PATH}')
