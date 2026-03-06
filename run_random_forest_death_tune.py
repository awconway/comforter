from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.calibration import calibration_curve

DATA_PATH = Path('/Users/ac/comforter/NewComfModelData.csv')
OUTPUT_DIR = Path('/Users/ac/comforter/artifacts/random_forest_death_tuned')
TARGET = 'DeathHospDisch'
EXCLUDE = [
    'Mort30d',
    'ICUWithin48h',
    'METWithin48h',
]
MANDATORY_EXCLUDES = ['Id']

RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.25

RF_N_ESTIMATORS_GRID = [200, 400]
RF_MAX_DEPTH_GRID = [None, 8, 16]
RF_MIN_SAMPLES_SPLIT_GRID = [2, 5]
RF_MIN_SAMPLES_LEAF_GRID = [1, 2]
RF_MAX_FEATURES_GRID = ['sqrt', 0.8]


def _resolve_target_column(columns: list[str], target: str) -> str:
    if target in columns:
        return target
    lower_map = {c.lower(): c for c in columns}
    if target.lower() in lower_map:
        return lower_map[target.lower()]
    raise ValueError(f"Target column '{target}' not found in data.")


def _resolve_excluded_columns(columns: list[str], requested: list[str]) -> list[str]:
    requested_lower = {x.lower() for x in requested + MANDATORY_EXCLUDES}
    return [col for col in columns if col.lower() in requested_lower]


def _ece_quantile(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> dict[str, float | int]:
    qbin = pd.qcut(pd.Series(prob), q=n_bins, labels=False, duplicates='drop')
    work = pd.DataFrame({'y_true': y_true, 'prob': prob, 'bin': qbin})
    agg = (
        work.groupby('bin', observed=True)
        .agg(n=('y_true', 'size'), mean_pred=('prob', 'mean'), obs_rate=('y_true', 'mean'))
        .reset_index(drop=True)
    )
    gap = np.abs(agg['mean_pred'] - agg['obs_rate'])
    return {
        'ece_q10': float((agg['n'] * gap).sum() / agg['n'].sum()),
        'mce_q10': float(gap.max()),
        'effective_bins': int(len(agg)),
        'min_bin_count': int(agg['n'].min()),
        'max_bin_count': int(agg['n'].max()),
    }


def _confusion_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'specificity': float(spec),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
    }


def _plot_calibration(y_true: np.ndarray, prob: np.ndarray, out_path: Path) -> None:
    frac_pos, mean_pred = calibration_curve(y_true, prob, n_bins=10, strategy='quantile')
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.plot([0, 1], [0, 1], '--', color='gray', linewidth=1.2, label='Perfect calibration')
    ax.plot(mean_pred, frac_pos, marker='o', linewidth=1.8, label='RandomForest')
    ax.set_xlabel('Predicted probability')
    ax.set_ylabel('Observed probability')
    ax.set_title('Random Forest DeathHospDisch Tuned Test Calibration (10 quantile bins)')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)


def _build_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series, str, list[str], pd.Series]:
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.lstrip('\ufeff') for c in df.columns]

    target_col = _resolve_target_column(df.columns.tolist(), TARGET)
    excluded_cols = _resolve_excluded_columns(df.columns.tolist(), EXCLUDE)

    X = df.drop(columns=[target_col] + excluded_cols, errors='ignore').copy()
    y = df[target_col].astype(int).copy()

    model_df = pd.concat([X, y], axis=1).dropna(subset=[target_col])
    y = model_df[target_col].astype(int)
    X = model_df.drop(columns=[target_col])

    train_val_X, test_X, train_val_y, test_y = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    train_X, val_X, train_y, val_y = train_test_split(
        train_val_X,
        train_val_y,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=train_val_y,
    )

    return (
        train_X,
        val_X,
        test_X,
        train_val_X,
        train_val_y,
        train_y,
        val_y,
        test_y,
        target_col,
        excluded_cols,
        model_df[target_col].astype(int),
    )


def _build_preprocess(train_X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = [c for c in train_X.columns if pd.api.types.is_numeric_dtype(train_X[c])]
    categorical_cols = [c for c in train_X.columns if c not in numeric_cols]

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore')),
    ])

    return ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols),
        ],
    )


def _fit_predict_prob(train_X: pd.DataFrame, train_y: pd.Series, X_eval: pd.DataFrame, params: dict[str, float | int | None | str]) -> np.ndarray:
    preprocessor = _build_preprocess(train_X)
    clf = RandomForestClassifier(
        n_estimators=int(params['n_estimators']),
        max_depth=params['max_depth'],
        min_samples_split=int(params['min_samples_split']),
        min_samples_leaf=int(params['min_samples_leaf']),
        max_features=params['max_features'],
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight='balanced',
    )
    model = Pipeline(steps=[
        ('preprocess', preprocessor),
        ('model', clf),
    ])
    model.fit(train_X, train_y.to_numpy())
    return model.predict_proba(X_eval)[:, 1]


def _evaluate_candidate(train_X: pd.DataFrame, train_y: pd.Series, val_X: pd.DataFrame, val_y: pd.Series, params: dict[str, float | int | None | str]) -> dict[str, float | int | str]:
    val_prob = _fit_predict_prob(train_X, train_y, val_X, params)
    val_pred = (val_prob >= 0.5).astype(int)
    val_y_arr = val_y.to_numpy()

    return {
        'n_estimators': int(params['n_estimators']),
        'max_depth': str(params['max_depth']) if params['max_depth'] is not None else 'None',
        'min_samples_split': int(params['min_samples_split']),
        'min_samples_leaf': int(params['min_samples_leaf']),
        'max_features': str(params['max_features']),
        'val_roc_auc': float(roc_auc_score(val_y_arr, val_prob)),
        'val_pr_auc': float(average_precision_score(val_y_arr, val_prob)),
        'val_brier': float(brier_score_loss(val_y_arr, val_prob)),
        'val_log_loss': float(log_loss(val_y_arr, val_prob, labels=[0, 1])),
        'val_accuracy': float(accuracy_score(val_y_arr, val_pred)),
        'val_balanced_accuracy': float(balanced_accuracy_score(val_y_arr, val_pred)),
    }


def main() -> None:
    train_X, val_X, test_X, train_val_X, train_val_y, train_y, val_y, test_y, target_col, excluded_cols, full_y = _build_splits()

    grid = [
        {
            'n_estimators': n,
            'max_depth': d,
            'min_samples_split': s,
            'min_samples_leaf': l,
            'max_features': mf,
        }
        for n, d, s, l, mf in product(
            RF_N_ESTIMATORS_GRID,
            RF_MAX_DEPTH_GRID,
            RF_MIN_SAMPLES_SPLIT_GRID,
            RF_MIN_SAMPLES_LEAF_GRID,
            RF_MAX_FEATURES_GRID,
        )
    ]

    tune_rows: list[dict[str, float | int | str]] = []
    for params in grid:
        tune_rows.append(_evaluate_candidate(train_X, train_y, val_X, val_y, params))

    tune_df = pd.DataFrame(tune_rows)
    tune_df = tune_df.sort_values(['val_roc_auc', 'val_brier'], ascending=[False, True]).reset_index(drop=True)
    tune_df['rank'] = tune_df.index + 1

    best_row = tune_df.iloc[0]
    best_params = {
        'n_estimators': int(best_row['n_estimators']),
        'max_depth': None if best_row['max_depth'] == 'None' else int(float(best_row['max_depth'])),
        'min_samples_split': int(best_row['min_samples_split']),
        'min_samples_leaf': int(best_row['min_samples_leaf']),
        'max_features': best_row['max_features'],
    }

    # Refit on train+val with best params
    test_prob = _fit_predict_prob(
        train_val_X,
        train_val_y,
        test_X,
        {
            'n_estimators': best_params['n_estimators'],
            'max_depth': best_params['max_depth'],
            'min_samples_split': best_params['min_samples_split'],
            'min_samples_leaf': best_params['min_samples_leaf'],
            'max_features': float(best_params['max_features']) if best_params['max_features'] not in {'sqrt', 'log2'} else best_params['max_features'],
        },
    )
    test_pred = (test_prob >= 0.5).astype(int)
    test_true = test_y.to_numpy()

    test_prob_metrics = {
        'roc_auc': float(roc_auc_score(test_true, test_prob)),
        'pr_auc': float(average_precision_score(test_true, test_prob)),
        'brier': float(brier_score_loss(test_true, test_prob)),
        'log_loss': float(log_loss(test_true, test_prob, labels=[0, 1])),
    } | _ece_quantile(test_true, test_prob)

    test_metrics = _confusion_metrics(test_true, test_pred)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tune_df_path = OUTPUT_DIR / 'tuning_results.csv'
    tuning_summary_path = OUTPUT_DIR / 'tuning_best.json'
    report_path = OUTPUT_DIR / 'report.json'
    pred_path = OUTPUT_DIR / 'test_predictions.csv'
    cal_path = OUTPUT_DIR / 'test_calibration_curve.png'

    tune_df.to_csv(tune_df_path, index=False)
    tuning_summary_path.write_text(
        json.dumps(
            {
                'best_params': best_params,
                'best_val_roc_auc': float(best_row['val_roc_auc']),
                'best_val_pr_auc': float(best_row['val_pr_auc']),
                'best_val_brier': float(best_row['val_brier']),
                'best_val_log_loss': float(best_row['val_log_loss']),
                'grid_size': int(len(tune_df)),
            },
            indent=2,
        ),
        encoding='utf-8',
    )

    report = {
        'metadata': {
            'data_path': str(DATA_PATH),
            'target': target_col,
            'excluded_predictors': excluded_cols,
            'predictor_count': len(train_X.columns),
            'split_counts': {
                'train': int(len(train_X)),
                'val': int(len(val_X)),
                'test': int(len(test_X)),
            },
            'seed': RANDOM_STATE,
            'hyperparameters': {
                'n_estimators': best_params['n_estimators'],
                'max_depth': best_params['max_depth'],
                'min_samples_split': best_params['min_samples_split'],
                'min_samples_leaf': best_params['min_samples_leaf'],
                'max_features': best_params['max_features'],
                'class_weight': 'balanced',
            },
            'prevalence': {
                'full': float(full_y.mean()),
                'train': float(train_y.mean()),
                'val': float(val_y.mean()),
                'test': float(test_y.mean()),
            },
        },
        'best_validation_metrics': {
            'val_roc_auc': float(best_row['val_roc_auc']),
            'val_pr_auc': float(best_row['val_pr_auc']),
            'val_brier': float(best_row['val_brier']),
            'val_log_loss': float(best_row['val_log_loss']),
            'val_accuracy': float(best_row['val_accuracy']),
            'val_balanced_accuracy': float(best_row['val_balanced_accuracy']),
        },
        'test_metrics_threshold_0_5': {
            'accuracy': float(test_metrics['accuracy']),
            'balanced_accuracy': float(test_metrics['balanced_accuracy']),
            'precision': float(test_metrics['precision']),
            'recall': float(test_metrics['recall']),
            'specificity': float(test_metrics['specificity']),
            'f1': float(test_metrics['f1']),
            'tn': int(test_metrics['tn']),
            'fp': int(test_metrics['fp']),
            'fn': int(test_metrics['fn']),
            'tp': int(test_metrics['tp']),
        },
        'test_probability_metrics': test_prob_metrics,
        'tuning': {
            'grid_size': int(len(tune_df)),
            'top_10_by_val_roc_auc': tune_df.head(10).to_dict(orient='records'),
        },
    }

    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')

    pd.DataFrame({'y_true': test_true, 'proba_death': test_prob, 'y_pred_threshold_0_5': test_pred}).to_csv(
        pred_path,
        index=False,
    )

    _plot_calibration(test_true, test_prob, cal_path)

    print(f'Saved {tune_df_path}')
    print(f'Saved {tuning_summary_path}')
    print(f'Saved {report_path}')
    print(f'Saved {pred_path}')
    print(f'Saved {cal_path}')
    print(f'Best params: {best_params}')


if __name__ == '__main__':
    main()
