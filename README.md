# comforter

`comforter` contains tabular-model experiments built around `NewComfModelData.csv`.
The root project is centered on predicting `DeathHospDisch` and comparing several
model families:

- TabICL
- Group lasso + logistic regression
- Group-penalized ridge + logistic regression
- Random forest
- AutoGluon CPU baselines from `autogluon_cpu/`

Generated reports, prediction files, and figures are written under `artifacts/`.

There is also a facility-holdout internal-external validation workflow for
`NewModelV3.csv`, where `Facility` is used only to define the leave-one-site-out
splits and is excluded from the predictor set. The pooled ROC-AUC summaries use
random-effects meta-analysis with Sidik-Jonkman heterogeneity estimation and
Hartung-Knapp confidence and prediction intervals.

## Data and path assumptions

- The dataset is expected at `NewComfModelData.csv` in the repository root.
- Most scripts hard-code `/Users/ac/comforter/...` paths for both input data and
  output directories.
- The configurable exceptions are `train_tabicl.py` and
  `run_tabicl_experiments.py`, which expose CLI flags for data, target, and
  output paths.
- If you move this repository, update the `DATA_PATH` and `OUTPUT_DIR`
  constants at the top of the fixed-path scripts before running them.

## Environment

Root project:

```bash
uv sync
```

AutoGluon subproject:

```bash
cd autogluon_cpu
uv sync
cd ..
```

Notes:

- The root `pyproject.toml` currently declares `python >= 3.14`.
- `autogluon_cpu/pyproject.toml` declares `python >= 3.12`.
- The first TabICL run may download the checkpoint
  `tabicl-classifier-v2-20260212.ckpt`.

## Quick start

For a single configurable TabICL train/validation/test run:

```bash
uv run python train_tabicl.py \
  --data NewComfModelData.csv \
  --target DeathHospDisch \
  --exclude ID Mort30d ICUWithin48h METWithin48h \
  --test-size 0.20 \
  --val-size 0.20 \
  --seed 42 \
  --output-dir artifacts/tabicl
```

Optional CPU override:

```bash
uv run python train_tabicl.py --device cpu
```

Main outputs:

- `artifacts/tabicl/metrics.json`
- `artifacts/tabicl/val_predictions.csv`
- `artifacts/tabicl/test_predictions.csv`

## Root workflow

The usual death-prediction workflow in this repository is:

| Script | Purpose | Main outputs |
| --- | --- | --- |
| `train_tabicl.py` | Single configurable TabICL run with train/val/test split. | `artifacts/tabicl/metrics.json` |
| `run_tabicl_experiments.py` | Broad TabICL sweep across a small parameter grid. | `artifacts/tabicl_death_experiments/leaderboard.csv`, `best_experiment.json` |
| `run_tabicl_death_stage2.py` | Narrower TabICL sweep with threshold analysis. | `artifacts/tabicl_death_stage2/stage2_leaderboard.csv`, `stage2_best.json` |
| `run_tabicl_death_final.py` | Locked final TabICL model on train+val, evaluated on held-out test. | `artifacts/tabicl_death_final/final_report.json`, `test_predictions_final.csv` |
| `run_tabicl_death_calibration.py` | Raw vs Platt vs isotonic calibration comparison for TabICL. | `artifacts/tabicl_death_calibration/calibration_report.json` |
| `run_group_lasso_death.py` | Untuned group lasso baseline. | `artifacts/group_lasso_death/report.json` |
| `run_group_lasso_death_tune.py` | Group lasso hyperparameter sweep and final test evaluation. | `artifacts/group_lasso_death_tuned/tuning_results.csv`, `report.json` |
| `run_group_ridge_death_tune.py` | Group-penalized ridge sweep and final test evaluation. | `artifacts/group_ridge_death_tuned/tuning_results.csv`, `report.json` |
| `run_random_forest_death_tune.py` | Random forest sweep and final test evaluation. | `artifacts/random_forest_death_tuned/tuning_results.csv`, `report.json` |
| `run_internal_external_validation_death.py` | Facility-holdout internal-external validation for TabICL, group lasso, group ridge, and random forest on `NewModelV3.csv`. | `artifacts/internal_external_validation_death/summary_auc.csv`, `report.json` |
| `plot_death_roc_auc.py` | Bar chart comparing ROC-AUC across final and sweep outputs. | `artifacts/death_roc_auc_comparison.png` |
| `plot_death_roc_curve_final_models.py` | ROC curves for final selected models. | `artifacts/death_roc_curve_final_models.png` |
| `plot_death_calibration_curve_final_models.py` | Calibration curves for final selected models. | `artifacts/death_calibration_curve_final_models.png` |

## Suggested run order

If you want to reproduce the death-model comparison from scratch, run:

```bash
uv run python run_tabicl_experiments.py
uv run python run_tabicl_death_stage2.py
uv run python run_tabicl_death_final.py
uv run python run_tabicl_death_calibration.py
uv run python run_group_lasso_death_tune.py
uv run python run_group_ridge_death_tune.py
uv run python run_random_forest_death_tune.py
```

Then build the AutoGluon death baseline:

```bash
cd autogluon_cpu
uv run python run_autogluon_death.py
cd ..
```

Finally, generate the comparison figures:

```bash
uv run python plot_death_roc_auc.py
uv run python plot_death_roc_curve_final_models.py
uv run python plot_death_calibration_curve_final_models.py
```

For facility-holdout internal-external validation on `NewModelV3.csv`:

```bash
uv run python run_internal_external_validation_death.py
uv run --project autogluon_cpu python autogluon_cpu/run_autogluon_death_internal_external.py
```

Main outputs:

- `artifacts/internal_external_validation_death/fold_auc.csv`
- `artifacts/internal_external_validation_death/summary_auc.csv`
- `artifacts/internal_external_validation_death/report.json`
- `artifacts/autogluon_death_internal_external/fold_auc.csv`
- `artifacts/autogluon_death_internal_external/summary_auc.csv`
- `artifacts/autogluon_death_internal_external/report.json`

## AutoGluon subproject

`autogluon_cpu/` is a separate `uv` project used for CPU-only AutoGluon baselines.
It currently includes:

- `run_autogluon_death.py`
- `run_autogluon_met.py`
- `run_autogluon_icu.py`

Those scripts write their outputs under:

- `artifacts/autogluon_death/`
- `artifacts/autogluon_metwithin48h/`
- `artifacts/autogluon_icuwithin48h/`

The root plotting scripts expect the death outputs from
`autogluon_cpu/run_autogluon_death.py` to exist before they run.

## Practical notes

- Most scripts use `seed = 42`.
- The TabICL experiment scripts use stratified splits.
- Several model families use different hard-coded exclusion lists. Check the
  constants at the top of each script if you want exact feature parity across
  models.
