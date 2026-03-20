from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

RUNS = {
    'baseline_shift': Path('/Users/ac/comforter/artifacts/internal_external_validation_death_stability_baseline_shift'),
    'drop_MET': Path('/Users/ac/comforter/artifacts/internal_external_validation_death_stability_drop_met'),
    'drop_MET+Sex': Path('/Users/ac/comforter/artifacts/internal_external_validation_death_stability_drop_MET_sex'),
    'transform_MET_binary': Path('/Users/ac/comforter/artifacts/internal_external_validation_death_stability_transformed_met_binary'),
}
LOGIT_EPS = 1e-6


def cal_metrics(y_true, probs):
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(probs, dtype=float), LOGIT_EPS, 1.0 - LOGIT_EPS)
    brier = float(brier_score_loss(y, p))
    logits = np.log(p / (1.0 - p)).reshape(-1, 1)
    lr = LogisticRegression(C=1e12, solver="lbfgs", fit_intercept=True, max_iter=2000)
    lr.fit(logits, y)
    citl = float(lr.intercept_[0])
    slope = float(lr.coef_[0, 0])
    return dict(brier=brier, citl=citl, slope=slope)

rows = []
for run_name, base in RUNS.items():
    pred_path = base / 'all_predictions.csv'
    if not pred_path.exists():
        print(f'missing {pred_path}')
        continue
    df = pd.read_csv(pred_path)
    for model, mdf in df.groupby('model'):
        mdf = mdf.sort_values(['test_site', 'row_index'])
        y = mdf['y_true'].to_numpy()
        p = mdf['proba_death'].to_numpy()
        over = cal_metrics(y, p)
        rows.append({
            'run': run_name,
            'model': model,
            'scope': 'overall_holdout',
            'test_site': 'pooled',
            'n': int(len(mdf)),
            'events': int(y.sum()),
            'prevalence': float(y.mean()),
            'brier': over['brier'],
            'citl': over['citl'],
            'abs_citl': abs(over['citl']),
            'slope': over['slope'],
            'abs_slope_dev': abs(over['slope'] - 1.0),
        })

        for site, sdf in mdf.groupby('test_site'):
            y_site = sdf['y_true'].to_numpy()
            p_site = sdf['proba_death'].to_numpy()
            site_metrics = cal_metrics(y_site, p_site)
            rows.append({
                'run': run_name,
                'model': model,
                'scope': 'site',
                'test_site': site,
                'n': int(len(sdf)),
                'events': int(y_site.sum()),
                'prevalence': float(y_site.mean()),
                'brier': site_metrics['brier'],
                'citl': site_metrics['citl'],
                'abs_citl': abs(site_metrics['citl']),
                'slope': site_metrics['slope'],
                'abs_slope_dev': abs(site_metrics['slope'] - 1.0),
            })

metrics_df = pd.DataFrame(rows)

pooled = metrics_df[metrics_df['scope'] == 'overall_holdout'].copy()
site = metrics_df[metrics_df['scope'] == 'site'].copy()

print('OVERALL HOLDOUT CALIBRATION (pooled across sites)')
print(
    pooled[[
        'run',
         'model',
         'n',
         'events',
         'prevalence',
         'brier',
         'citl',
         'abs_citl',
         'slope',
         'abs_slope_dev',
    ]]
    .sort_values(['model', 'run'])
    .to_string(index=False, float_format='%.4f')
)

print('\nSITE-LEVEL CALIBRATION VARIABILITY (lower is better)')
site_summary = []
for (run_name, model), g in site.groupby(['run', 'model']):
    site_summary.append({
        'run': run_name,
        'model': model,
        'brier_mean': float(g['brier'].mean()),
        'brier_range': float(g['brier'].max() - g['brier'].min()),
        'abs_citl_mean': float(g['abs_citl'].mean()),
        'abs_citl_max': float(g['abs_citl'].max()),
        'abs_slope_dev_mean': float(g['abs_slope_dev'].mean()),
        'abs_slope_dev_max': float(g['abs_slope_dev'].max()),
    })

site_summary_df = pd.DataFrame(site_summary).sort_values(['model', 'run'])
print(site_summary_df.to_string(index=False, float_format='%.4f'))

outdir = Path('/Users/ac/comforter/artifacts/predictor_stability_death/calibration_sensitivity')
outdir.mkdir(parents=True, exist_ok=True)
metrics_df.to_csv(outdir / 'site_and_pooled_calibration_metrics.csv', index=False)
pooled.to_csv(outdir / 'pooled_calibration_summary.csv', index=False)
site_summary_df.to_csv(outdir / 'site_calibration_aggregation.csv', index=False)
print(f'\nSaved artifacts -> {outdir}')
