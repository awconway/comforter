from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from group_lasso import GroupLasso
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
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

DATA_PATH = Path('/Users/ac/comforter/NewComfModelData.csv')
OUTPUT_DIR = Path('/Users/ac/comforter/artifacts/group_lasso_death')
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
INTERACTION_VARS = ['Age', 'NAS', 'NEWS', 'ICD', 'SOFA']
SPLINE_KNOTS = 4
SPLINE_DEGREE = 3
GROUP_REG = 0.05
L1_REG = 0.02
RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.25


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


class GroupLassoDesign:
    def __init__(self, spline_vars: list[str], interaction_vars: list[str]):
        self.spline_vars = spline_vars
        self.interaction_vars = interaction_vars
        self.numeric_cols: list[str] = []
        self.binary_numeric_cols: list[str] = []
        self.linear_numeric_cols: list[str] = []
        self.cat_cols: list[str] = []
        self.spline_vars_fit: list[str] = []
        self.interactions: list[tuple[str, str]] = []
        self.spline_transformers: dict[str, SplineTransformer] = {}
        self.num_medians: dict[str, float] = {}
        self.numeric_means: dict[str, float] = {}
        self.cat_fill: dict[str, str] = {}
        self.cat_dummy_cols: list[str] = []
        self.feature_names: list[str] = []
        self.group_ids: list[int] = []
        self.fitted = False

    def _center_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'Age' in df.columns and 'Age' in self.numeric_means:
            df = df.copy()
            df['Age'] = df['Age'] - self.numeric_means['Age']
        return df

    def fit(self, df: pd.DataFrame) -> 'GroupLassoDesign':
        df_local = df.copy()
        self.numeric_cols = [c for c in df_local.columns if pd.api.types.is_numeric_dtype(df_local[c])]
        self.cat_cols = [c for c in df_local.columns if c not in self.numeric_cols]

        # imputation
        self.num_medians = df_local[self.numeric_cols].median().to_dict()
        num_imputed = df_local[self.numeric_cols].fillna(self.num_medians)
        self.numeric_means = num_imputed.mean().to_dict()
        df_local[self.numeric_cols] = self._center_numeric(num_imputed)
        for col in self.cat_cols:
            mode = df_local[col].mode(dropna=True)
            self.cat_fill[col] = mode.iloc[0] if len(mode) else 'missing'

        # continuous spline terms
        self.spline_vars_fit = [c for c in self.spline_vars if c in self.numeric_cols]
        non_spline_numeric = [c for c in self.numeric_cols if c not in self.spline_vars_fit]
        self.binary_numeric_cols = [
            c for c in non_spline_numeric if df_local[c].dropna().nunique() <= 2
        ]
        self.linear_numeric_cols = [c for c in non_spline_numeric if c not in self.binary_numeric_cols]

        # interaction terms
        int_candidates = [c for c in self.interaction_vars if c in self.numeric_cols]
        self.interactions = list(combinations(int_candidates, 2))

        # fit spline transformers
        self.spline_transformers = {}
        for col in self.spline_vars_fit:
            st = SplineTransformer(degree=SPLINE_DEGREE, n_knots=SPLINE_KNOTS, include_bias=False)
            col_train = df_local[[col]].to_numpy()
            st.fit(col_train)
            self.spline_transformers[col] = st

        # categorical dummy template
        cat_imputed = df_local[self.cat_cols].fillna(self.cat_fill) if self.cat_cols else pd.DataFrame(index=df_local.index)
        if self.cat_cols:
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

        # build feature names and group ids in deterministic order
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
            # one group per original categorical variable
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
        num = self._center_numeric(num.fillna(self.num_medians))
        cat = cat.fillna(self.cat_fill)

        blocks: list[np.ndarray] = []

        for col in self.spline_vars_fit:
            tr = self.spline_transformers[col].transform(num[[col]].to_numpy())
            blocks.append(tr)

        if self.linear_numeric_cols:
            blocks.append(num[self.linear_numeric_cols].to_numpy(dtype=float))

        if self.binary_numeric_cols:
            blocks.append(num[self.binary_numeric_cols].to_numpy(dtype=float))

        for left, right in self.interactions:
            blocks.append((num[left] * num[right]).to_numpy(dtype=float).reshape(-1, 1))

        if self.cat_cols:
            cat_imp = cat.fillna(self.cat_fill)
            cat_d = pd.get_dummies(
                cat_imp,
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



def main() -> None:
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.lstrip('\ufeff') for c in df.columns]

    target_col = _resolve_target_column(df.columns.tolist(), TARGET)
    excluded_cols = _resolve_excluded_columns(df.columns.tolist(), EXCLUDE)

    X = df.drop(columns=[target_col] + excluded_cols, errors='ignore').copy()
    y = df[target_col].astype(int).copy()

    # ensure aligned rows
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

    design = GroupLassoDesign(SPLINE_VARS, INTERACTION_VARS)
    design.fit(train_X)
    feature_names = design.feature_names
    group_ids = design.group_ids

    X_train_raw = design.transform(train_X)
    X_val_raw = design.transform(val_X)
    X_test_raw = design.transform(test_X)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    X_test = scaler.transform(X_test_raw)

    gl = GroupLasso(
        groups=np.array(group_ids),
        group_reg=GROUP_REG,
        l1_reg=L1_REG,
        n_iter=5000,
        tol=1e-5,
        scale_reg='none',
        fit_intercept=True,
        random_state=RANDOM_STATE,
        supress_warning=True,
    )

    gl.fit(X_train, train_y.to_numpy())
    keep = gl.sparsity_mask_
    X_train_sel = gl.transform(X_train)
    if X_train_sel.shape[1] == 0:
        X_train_sel = X_train
        keep = np.ones(X_train.shape[1], dtype=bool)

    selected_features = [f for f, k in zip(feature_names, keep) if k]
    selected_groups = sorted(set(g for g, k in zip(group_ids, keep) if k))

    final_lr = LogisticRegression(max_iter=1000, solver='liblinear', random_state=RANDOM_STATE)
    final_lr.fit(X_train_sel, train_y.to_numpy())

    X_val_sel = gl.transform(X_val)
    if X_val_sel.shape[1] == 0:
        X_val_sel = X_val
    val_prob = final_lr.predict_proba(X_val_sel)[:, 1]
    val_pred = (val_prob >= 0.5).astype(int)
    val_metrics = _confusion_metrics(val_y.to_numpy(), val_pred)
    val_metrics.update({
        'roc_auc': float(roc_auc_score(val_y.to_numpy(), val_prob)),
        'pr_auc': float(average_precision_score(val_y.to_numpy(), val_prob)),
    })

    # Final refit on train+val
    X_trainval = np.vstack([X_train, X_val])
    y_trainval = np.concatenate([train_y.to_numpy(), val_y.to_numpy()])
    gl.fit(X_trainval, y_trainval)

    X_trainval_sel = gl.transform(X_trainval)
    if X_trainval_sel.shape[1] == 0:
        X_trainval_sel = X_trainval

    final_lr.fit(X_trainval_sel, y_trainval)

    X_test_sel = gl.transform(X_test)
    if X_test_sel.shape[1] == 0:
        X_test_sel = X_test
    test_prob = final_lr.predict_proba(X_test_sel)[:, 1]
    test_pred = (test_prob >= 0.5).astype(int)
    test_metrics = _confusion_metrics(test_y.to_numpy(), test_pred)
    test_prob_metrics = {
        'roc_auc': float(roc_auc_score(test_y.to_numpy(), test_prob)),
        'pr_auc': float(average_precision_score(test_y.to_numpy(), test_prob)),
        'brier': float(brier_score_loss(test_y.to_numpy(), test_prob)),
        'log_loss': float(log_loss(test_y.to_numpy(), test_prob, labels=[0, 1])),
    } | _ece_quantile(test_y.to_numpy(), test_prob)

    frac_pos, mean_pred = calibration_curve(test_y.to_numpy(), test_prob, n_bins=10, strategy='quantile')
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.plot([0, 1], [0, 1], '--', color='gray', linewidth=1.2, label='Perfect calibration')
    ax.plot(mean_pred, frac_pos, marker='o', linewidth=1.8, label='Group Lasso + Logistic')
    ax.set_xlabel('Predicted probability')
    ax.set_ylabel('Observed probability')
    ax.set_title('Group Lasso DeathHospDisch Test Calibration (10 quantile bins)')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    fig.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cal_path = OUTPUT_DIR / 'test_calibration_curve.png'
    fig.savefig(cal_path, dpi=220)

    report = {
        'metadata': {
            'data_path': str(DATA_PATH),
            'target': target_col,
            'excluded_predictors': excluded_cols,
            'predictor_count': len(feature_names),
            'split_counts': {
                'train': int(len(train_X)),
                'val': int(len(val_X)),
                'test': int(len(test_X)),
            },
            'seed': RANDOM_STATE,
            'spline_vars': SPLINE_VARS,
            'interaction_vars': INTERACTION_VARS,
            'interaction_terms': [f'{a}_x_{b}' for a, b in design.interactions],
            'hyperparameters': {
                'group_reg': GROUP_REG,
                'l1_reg': L1_REG,
                'spline_knots': SPLINE_KNOTS,
                'spline_degree': SPLINE_DEGREE,
                'scale_reg': 'none',
            },
            'selected_feature_count': int(len(selected_features)),
            'selected_group_count': int(len(selected_groups)),
            'selected_features': selected_features,
            'selected_groups': selected_groups,
            'prevalence': {
                'full': float(y.mean()),
                'train': float(train_y.mean()),
                'val': float(val_y.mean()),
                'test': float(test_y.mean()),
            },
        },
        'validation_metrics_threshold_0_5': {
            'accuracy': val_metrics['accuracy'],
            'balanced_accuracy': val_metrics['balanced_accuracy'],
            'precision': val_metrics['precision'],
            'recall': val_metrics['recall'],
            'specificity': val_metrics['specificity'],
            'f1': val_metrics['f1'],
            'roc_auc': val_metrics['roc_auc'],
            'pr_auc': val_metrics['pr_auc'],
            'tn': val_metrics['tn'],
            'fp': val_metrics['fp'],
            'fn': val_metrics['fn'],
            'tp': val_metrics['tp'],
        },
        'test_metrics_threshold_0_5': {
            'accuracy': test_metrics['accuracy'],
            'balanced_accuracy': test_metrics['balanced_accuracy'],
            'precision': test_metrics['precision'],
            'recall': test_metrics['recall'],
            'specificity': test_metrics['specificity'],
            'f1': test_metrics['f1'],
            'tn': test_metrics['tn'],
            'fp': test_metrics['fp'],
            'fn': test_metrics['fn'],
            'tp': test_metrics['tp'],
        },
        'test_probability_metrics': test_prob_metrics,
    }

    report_path = OUTPUT_DIR / 'report.json'
    pred_path = OUTPUT_DIR / 'test_predictions.csv'

    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    pd.DataFrame({'y_true': test_y.to_numpy(), 'proba_death': test_prob, 'y_pred_threshold_0_5': test_pred}).to_csv(
        pred_path,
        index=False,
    )

    print(f'Saved {report_path}')
    print(f'Saved {pred_path}')
    print(f'Saved {cal_path}')


if __name__ == '__main__':
    main()
