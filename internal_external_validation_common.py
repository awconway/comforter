from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.metrics import roc_auc_score


DEFAULT_TARGET = "DeathHospDisch"
DEFAULT_FACILITY = "Facility"
DEFAULT_ID = "Id"
LOGIT_EPS = 1e-6


def load_dataset(data_path: Path) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    df.columns = [c.lstrip("\ufeff") for c in df.columns]
    return df


def prepare_site_holdout_data(
    data_path: Path,
    *,
    target_col: str = DEFAULT_TARGET,
    facility_col: str = DEFAULT_FACILITY,
    id_col: str = DEFAULT_ID,
    extra_excludes: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    df = load_dataset(data_path)
    missing = [col for col in [target_col, facility_col] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {data_path}: {missing}")

    exclude = {target_col, facility_col}
    if id_col in df.columns:
        exclude.add(id_col)
    if extra_excludes:
        exclude.update(extra_excludes)

    model_df = df.dropna(subset=[target_col, facility_col]).copy()
    model_df[target_col] = model_df[target_col].astype(int)

    feature_cols = [col for col in model_df.columns if col not in exclude]
    facility_levels = sorted(model_df[facility_col].astype(str).unique().tolist())
    return model_df, feature_cols, facility_levels


def iter_site_splits(
    model_df: pd.DataFrame,
    feature_cols: list[str],
    *,
    target_col: str = DEFAULT_TARGET,
    facility_col: str = DEFAULT_FACILITY,
    id_col: str = DEFAULT_ID,
) -> list[dict[str, Any]]:
    facility_levels = sorted(model_df[facility_col].astype(str).unique().tolist())
    splits: list[dict[str, Any]] = []
    for test_site in facility_levels:
        test_mask = model_df[facility_col].astype(str) == test_site
        train_df = model_df.loc[~test_mask].copy()
        test_df = model_df.loc[test_mask].copy()
        train_sites = sorted(train_df[facility_col].astype(str).unique().tolist())

        split = {
            "test_site": test_site,
            "train_sites": train_sites,
            "train_X": train_df[feature_cols].copy(),
            "train_y": train_df[target_col].astype(int).copy(),
            "test_X": test_df[feature_cols].copy(),
            "test_y": test_df[target_col].astype(int).copy(),
            "test_index": test_df.index.to_numpy(),
            "test_ids": test_df[id_col].copy() if id_col in test_df.columns else pd.Series(test_df.index, index=test_df.index),
        }
        splits.append(split)
    return splits


def auc_with_hanley_mcneil_variance(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float | int]:
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)
    n_pos = int(np.sum(y_true_arr == 1))
    n_neg = int(np.sum(y_true_arr == 0))
    if n_pos == 0 or n_neg == 0:
        raise ValueError("ROC-AUC variance requires both outcome classes in the evaluation set.")

    auc = float(roc_auc_score(y_true_arr, y_prob_arr))
    q1 = auc / (2.0 - auc) if auc < 2.0 else 1.0
    q2 = (2.0 * auc * auc) / (1.0 + auc) if auc > -1.0 else 0.0
    variance = (
        auc * (1.0 - auc)
        + (n_pos - 1) * (q1 - auc * auc)
        + (n_neg - 1) * (q2 - auc * auc)
    ) / (n_pos * n_neg)
    variance = float(max(variance, 0.0))
    return {
        "auc": auc,
        "auc_variance": variance,
        "auc_se": float(math.sqrt(variance)),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_test": int(n_pos + n_neg),
    }


def _clip_auc(auc: np.ndarray) -> np.ndarray:
    return np.clip(auc, LOGIT_EPS, 1.0 - LOGIT_EPS)


def _logit(x: np.ndarray) -> np.ndarray:
    return np.log(x / (1.0 - x))


def _expit(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def sidik_jonkman_tau2(effects: np.ndarray, variances: np.ndarray) -> float:
    k = len(effects)
    if k <= 1:
        return 0.0

    tau2_0 = float(np.var(effects, ddof=1) * (k - 1) / k)
    if tau2_0 <= 0.0:
        return 0.0

    weights_0 = 1.0 / (variances + tau2_0)
    mean_0 = float(np.sum(weights_0 * effects) / np.sum(weights_0))
    rss = float(np.sum(weights_0 * (effects - mean_0) ** 2))
    tau2 = tau2_0 * rss / (k - 1)
    return float(max(tau2, 0.0))


def pool_auc_hksj(fold_df: pd.DataFrame) -> dict[str, Any]:
    if fold_df.empty:
        raise ValueError("fold_df must contain at least one site result.")

    auc = fold_df["auc"].to_numpy(dtype=float)
    auc_var = fold_df["auc_variance"].to_numpy(dtype=float)
    auc_clip = _clip_auc(auc)

    effects = _logit(auc_clip)
    variances = auc_var / (auc_clip**2 * (1.0 - auc_clip) ** 2)
    tau2 = sidik_jonkman_tau2(effects, variances)

    weights_re = 1.0 / (variances + tau2)
    pooled_effect = float(np.sum(weights_re * effects) / np.sum(weights_re))

    df = max(len(effects) - 1, 1)
    hk_scale = float(np.sum(weights_re * (effects - pooled_effect) ** 2) / df)
    hk_var = float(max(hk_scale / np.sum(weights_re), 0.0))
    hk_se = float(math.sqrt(hk_var))
    crit = float(t.ppf(0.975, df))

    ci_low = pooled_effect - crit * hk_se
    ci_high = pooled_effect + crit * hk_se

    pred_se = float(math.sqrt(max(tau2 + hk_var, 0.0)))
    pi_low = pooled_effect - crit * pred_se
    pi_high = pooled_effect + crit * pred_se

    weights_fe = 1.0 / variances
    pooled_fe = float(np.sum(weights_fe * effects) / np.sum(weights_fe))
    q = float(np.sum(weights_fe * (effects - pooled_fe) ** 2))
    i2 = 0.0 if q <= 0.0 else max(0.0, (q - (len(effects) - 1)) / q) * 100.0

    return {
        "method": "random-effects meta-analysis",
        "tau2_estimator": "Sidik-Jonkman",
        "ci_method": "Hartung-Knapp-Sidik-Jonkman",
        "prediction_interval_method": "Hartung-Knapp t-distribution",
        "effect_scale": "logit_auc",
        "within_site_variance": "Hanley-McNeil AUC variance with delta-method to logit(AUC)",
        "k_sites": int(len(effects)),
        "pooled_auc": float(_expit(np.array([pooled_effect]))[0]),
        "pooled_auc_ci_95": {
            "lower": float(_expit(np.array([ci_low]))[0]),
            "upper": float(_expit(np.array([ci_high]))[0]),
        },
        "pooled_auc_prediction_interval_95": {
            "lower": float(_expit(np.array([pi_low]))[0]),
            "upper": float(_expit(np.array([pi_high]))[0]),
        },
        "pooled_logit_auc": pooled_effect,
        "pooled_logit_auc_se_hksj": hk_se,
        "tau2_logit_auc": tau2,
        "i2_percent": float(i2),
        "q_statistic": q,
        "degrees_of_freedom": int(df),
    }
