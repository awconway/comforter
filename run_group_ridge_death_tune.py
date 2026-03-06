from __future__ import annotations

import json
from itertools import combinations, product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
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
from sklearn.preprocessing import SplineTransformer, StandardScaler

from group_penalized_logistic import fit_group_ridge_logistic

DATA_PATH = Path('/Users/ac/comforter/NewComfModelData.csv')
OUTPUT_DIR = Path('/Users/ac/comforter/artifacts/group_ridge_death_tuned')
TARGET = 'DeathHospDisch'
EXCLUDE = [
    'Cardiovascular',
    'LungDisease',
    'CKD',
    'LiverDisease',
    'Diabetes',
    'Malignancy',
    'CognitionMentalHealth',
    'FunctionalDependency',
    'NutritionMetabolism',
    'SensoryImpairment',
    'Mort30d',
    'ICUWithin48h',
    'METWithin48h',
]
MANDATORY_EXCLUDES = ['Id']

SPLINE_VARS = ['Age', 'NAS', 'NEWS', 'ICD', 'SOFA']
INTERACTION_VARS = ['NAS', 'NEWS', 'ICD', 'SOFA']
RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.25

MAX_ITER = 2000
TOL = 1e-8
GROUP_WEIGHT_MODE = 'none'

RIDGE_REG_GRID = [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
SPLINE_KNOTS_GRID = [3, 4, 5]
SPLINE_DEGREE_GRID = [2, 3]


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
    ax.plot(mean_pred, frac_pos, marker='o', linewidth=1.8, label='Group Ridge + Logistic')
    ax.set_xlabel('Predicted probability')
    ax.set_ylabel('Observed probability')
    ax.set_title('Group Ridge DeathHospDisch Tuned Test Calibration (10 quantile bins)')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)


class GroupLassoDesign:
    def __init__(self, spline_vars: list[str], interaction_vars: list[str], spline_knots: int, spline_degree: int):
        self.spline_vars = spline_vars
        self.interaction_vars = interaction_vars
        self.spline_knots = spline_knots
        self.spline_degree = spline_degree

        self.numeric_cols: list[str] = []
        self.binary_numeric_cols: list[str] = []
        self.linear_numeric_cols: list[str] = []
        self.cat_cols: list[str] = []
        self.spline_vars_fit: list[str] = []
        self.interactions: list[tuple[str, str]] = []
        self.spline_transformers: dict[str, SplineTransformer] = {}
        self.num_medians: dict[str, float] = {}
        self.cat_fill: dict[str, str] = {}
        self.cat_dummy_cols: list[str] = []
        self.feature_names: list[str] = []
        self.group_ids: list[int] = []
        self.fitted = False

    def fit(self, df: pd.DataFrame) -> 'GroupLassoDesign':
        df_local = df.copy()
        self.numeric_cols = [c for c in df_local.columns if pd.api.types.is_numeric_dtype(df_local[c])]
        self.cat_cols = [c for c in df_local.columns if c not in self.numeric_cols]

        self.num_medians = df_local[self.numeric_cols].median().to_dict()
        self.cat_fill = {}
        for col in self.cat_cols:
            mode = df_local[col].mode(dropna=True)
            self.cat_fill[col] = mode.iloc[0] if len(mode) else 'missing'

        self.spline_vars_fit = [c for c in self.spline_vars if c in self.numeric_cols]
        non_spline_numeric = [c for c in self.numeric_cols if c not in self.spline_vars_fit]
        self.binary_numeric_cols = [
            c for c in non_spline_numeric if df_local[c].dropna().nunique() <= 2
        ]
        self.linear_numeric_cols = [c for c in non_spline_numeric if c not in self.binary_numeric_cols]

        int_candidates = [c for c in self.interaction_vars if c in self.numeric_cols]
        self.interactions = list(combinations(int_candidates, 2))

        self.spline_transformers = {}
        for col in self.spline_vars_fit:
            st = SplineTransformer(
                degree=self.spline_degree,
                n_knots=self.spline_knots,
                include_bias=False,
            )
            col_train = df_local[[col]].fillna(self.num_medians[col]).to_numpy()
            st.fit(col_train)
            self.spline_transformers[col] = st

        if self.cat_cols:
            cat_imputed = df_local[self.cat_cols].fillna(self.cat_fill)
            cat_dummies = pd.get_dummies(
                cat_imputed,
                columns=self.cat_cols,
                prefix_sep='__',
                dummy_na=False,
                dtype=float,
            )
            self.cat_dummy_cols = cat_dummies.columns.tolist()
        else:
            self.cat_dummy_cols = []

        names: list[str] = []
        groups: list[int] = []
        group_id = 1

        for col in self.spline_vars_fit:
            n_spl = self.spline_transformers[col].transform(df_local[[col]].fillna(self.num_medians[col]).to_numpy()).shape[1]
            for i in range(n_spl):
                names.append(f'{col}_spline_{i + 1}')
                groups.append(group_id)
            group_id += 1

        for col in self.linear_numeric_cols + self.binary_numeric_cols:
            names.append(col)
            groups.append(group_id)
            group_id += 1

        for left, right in self.interactions:
            names.append(f'{left}_x_{right}')
            groups.append(group_id)
            group_id += 1

        for c in self.cat_dummy_cols:
            names.append(c)
        if self.cat_dummy_cols:
            for c in self.cat_cols:
                cols_for_var = [x for x in self.cat_dummy_cols if x.split('__', 1)[0] == c]
                groups.extend([group_id] * len(cols_for_var))
                group_id += 1

        self.feature_names = names
        self.group_ids = groups
        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError('Call fit() first.')

        num = df[self.numeric_cols].copy()
        cat = df[self.cat_cols].copy() if self.cat_cols else pd.DataFrame(index=df.index)
        num = num.fillna(self.num_medians)

        blocks: list[np.ndarray] = []

        for col in self.spline_vars_fit:
            blocks.append(self.spline_transformers[col].transform(num[[col]].to_numpy()))

        if self.linear_numeric_cols:
            blocks.append(num[self.linear_numeric_cols].to_numpy(dtype=float))

        if self.binary_numeric_cols:
            blocks.append(num[self.binary_numeric_cols].to_numpy(dtype=float))

        for left, right in self.interactions:
            blocks.append((num[left] * num[right]).to_numpy(dtype=float).reshape(-1, 1))

        if self.cat_cols:
            cat = cat.fillna(self.cat_fill)
            cat_d = pd.get_dummies(
                cat,
                columns=self.cat_cols,
                prefix_sep='__',
                dummy_na=False,
                dtype=float,
            )
            cat_d = cat_d.reindex(columns=self.cat_dummy_cols, fill_value=0.0)
            blocks.append(cat_d.to_numpy(dtype=float))

        if not blocks:
            return np.empty((len(df), 0))

        return np.hstack(blocks)


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


def _fit_model(
    train_X: pd.DataFrame,
    train_y: pd.Series,
    params: dict[str, float | int],
) -> tuple[GroupLassoDesign, StandardScaler, object]:
    design = GroupLassoDesign(
        spline_vars=SPLINE_VARS,
        interaction_vars=INTERACTION_VARS,
        spline_knots=int(params['spline_knots']),
        spline_degree=int(params['spline_degree']),
    )
    design.fit(train_X)

    X_train_raw = design.transform(train_X)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)

    model = fit_group_ridge_logistic(
        X_train,
        train_y.to_numpy(dtype=float),
        group_ids=np.array(design.group_ids),
        ridge_reg=float(params['ridge_reg']),
        group_weight_mode=GROUP_WEIGHT_MODE,
        max_iter=MAX_ITER,
        tol=TOL,
    )

    return design, scaler, model


def _predict_prob(
    design: GroupLassoDesign,
    scaler: StandardScaler,
    model: object,
    X: pd.DataFrame,
) -> np.ndarray:
    X_raw = design.transform(X)
    X_scaled = scaler.transform(X_raw)
    return model.predict_proba(X_scaled)


def _evaluate_candidate(
    train_X: pd.DataFrame,
    train_y: pd.Series,
    val_X: pd.DataFrame,
    val_y: pd.Series,
    params: dict[str, float | int],
) -> dict[str, float | int]:
    design, scaler, model = _fit_model(train_X, train_y, params)
    val_prob = _predict_prob(design, scaler, model, val_X)
    val_pred = (val_prob >= 0.5).astype(int)
    val_y_arr = val_y.to_numpy()

    sel = int(np.sum(model.active_mask_)) if model.active_mask_.size else 0
    return {
        'ridge_reg': float(params['ridge_reg']),
        'spline_knots': int(params['spline_knots']),
        'spline_degree': int(params['spline_degree']),
        'val_roc_auc': float(roc_auc_score(val_y_arr, val_prob)),
        'val_pr_auc': float(average_precision_score(val_y_arr, val_prob)),
        'val_brier': float(brier_score_loss(val_y_arr, val_prob)),
        'val_log_loss': float(log_loss(val_y_arr, val_prob, labels=[0, 1])),
        'val_accuracy': float(accuracy_score(val_y_arr, val_pred)),
        'val_balanced_accuracy': float(balanced_accuracy_score(val_y_arr, val_pred)),
        'selected_features': int(sel),
        'selected_feature_proportion': float(float(sel) / len(model.active_mask_)) if len(model.active_mask_) else 0.0,
        'selected_groups': int(len(model.active_groups_)),
        'design_groups': int(design.group_ids[-1]) if design.group_ids else 0,
        'converged': bool(model.converged_),
        'n_iter': int(model.n_iter_),
    }


def main() -> None:
    train_X, val_X, test_X, train_val_X, train_val_y, train_y, val_y, test_y, target_col, excluded_cols, full_y = _build_splits()

    grid = [
        {
            'ridge_reg': g,
            'spline_knots': k,
            'spline_degree': d,
        }
        for g, k, d in product(RIDGE_REG_GRID, SPLINE_KNOTS_GRID, SPLINE_DEGREE_GRID)
    ]

    tune_rows: list[dict] = []
    for params in grid:
        tune_rows.append(_evaluate_candidate(train_X, train_y, val_X, val_y, params))

    tune_df = pd.DataFrame(tune_rows)
    tune_df = tune_df.sort_values(['val_roc_auc', 'val_brier'], ascending=[False, True]).reset_index(drop=True)
    tune_df['rank'] = tune_df.index + 1

    best_row = tune_df.iloc[0]
    best_params = {
        'ridge_reg': float(best_row['ridge_reg']),
        'spline_knots': int(best_row['spline_knots']),
        'spline_degree': int(best_row['spline_degree']),
    }

    # Refit on train+val with best params
    best_design, best_scaler, best_model = _fit_model(train_val_X, train_val_y, best_params)

    test_prob = _predict_prob(best_design, best_scaler, best_model, test_X)
    test_pred = (test_prob >= 0.5).astype(int)
    test_true = test_y.to_numpy()

    test_prob_metrics = {
        'roc_auc': float(roc_auc_score(test_true, test_prob)),
        'pr_auc': float(average_precision_score(test_true, test_prob)),
        'brier': float(brier_score_loss(test_true, test_prob)),
        'log_loss': float(log_loss(test_true, test_prob, labels=[0, 1])),
    } | _ece_quantile(test_true, test_prob)

    test_metrics = _confusion_metrics(test_true, test_pred)

    selected_features = [f for f, k in zip(best_design.feature_names, best_model.active_mask_) if k]
    selected_groups = [int(group) for group in best_model.active_groups_]

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
                'best_params': {
                    **best_params,
                },
                'best_val_roc_auc': float(best_row['val_roc_auc']),
                'best_val_pr_auc': float(best_row['val_pr_auc']),
                'best_val_brier': float(best_row['val_brier']),
                'best_val_log_loss': float(best_row['val_log_loss']),
                'grid_size': int(len(tune_df)),
                'group_weight_mode': GROUP_WEIGHT_MODE,
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
            'predictor_count': len(best_design.feature_names),
            'split_counts': {
                'train': int(len(train_X)),
                'val': int(len(val_X)),
                'test': int(len(test_X)),
            },
            'seed': RANDOM_STATE,
            'spline_vars': SPLINE_VARS,
            'interaction_vars': INTERACTION_VARS,
            'interaction_terms': [f'{a}_x_{b}' for a, b in best_design.interactions],
            'hyperparameters': {
                'ridge_reg': best_params['ridge_reg'],
                'spline_knots': best_params['spline_knots'],
                'spline_degree': best_params['spline_degree'],
                'group_weight_mode': GROUP_WEIGHT_MODE,
                'max_iter': MAX_ITER,
                'tol': TOL,
                'objective': 'logistic + 0.5 * lambda * sum_g ||beta_g||_2^2',
            },
            'selected_feature_count': int(len(selected_features)),
            'selected_group_count': int(len(selected_groups)),
            'selected_features': selected_features,
            'selected_groups': selected_groups,
            'group_norms': {str(k): float(v) for k, v in best_model.group_norms_.items()},
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
            'selected_features': int(best_row['selected_features']),
            'selected_feature_proportion': float(best_row['selected_feature_proportion']),
            'selected_groups': int(best_row['selected_groups']),
            'converged': bool(best_row['converged']),
            'n_iter': int(best_row['n_iter']),
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
        'optimization': {
            'converged': bool(best_model.converged_),
            'n_iter': int(best_model.n_iter_),
            'objective': float(best_model.objective_),
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
