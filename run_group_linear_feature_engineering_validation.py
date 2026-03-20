from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import run_group_lasso_death_tune as group_lasso
import run_group_ridge_death_tune as group_ridge
from group_penalized_logistic import fit_group_lasso_logistic, fit_group_ridge_logistic


DATA_PATH = Path("/Users/ac/comforter/NewComfModelData.csv")
OUTPUT_DIR = Path("/Users/ac/comforter/artifacts/group_linear_feature_engineering_validation")
TARGET = "DeathHospDisch"
SEED = 42

BASE_FEATURES = [
    "Age",
    "Sex",
    "ARP",
    "AdmitPrev24h",
    "SurgPrev24h",
    "ICUDischPrev24h",
    "METWithinPrev24h",
    "NAS",
    "NEWS",
    "ICD",
    "SOFA",
]

FEATURE_SETS = [
    {
        "name": "raw_core_plus_nas",
        "use_summary": False,
        "use_flags": False,
        "spline_vars": ["Age", "NAS", "NEWS", "ICD", "SOFA"],
        "interaction_vars": ["Age", "NAS", "NEWS", "ICD", "SOFA"],
    },
    {
        "name": "raw_plus_summary_flags",
        "use_summary": True,
        "use_flags": True,
        "spline_vars": ["Age", "NAS", "NEWS", "ICD", "SOFA"],
        "interaction_vars": ["Age", "NAS", "NEWS", "ICD", "SOFA", "AcuteInstabilityCount", "HighAcuity"],
    },
]

GROUP_LASSO_TUNING_PATH = Path("/Users/ac/comforter/artifacts/group_lasso_death_tuned/tuning_best.json")
GROUP_RIDGE_TUNING_PATH = Path("/Users/ac/comforter/artifacts/group_ridge_death_tuned/tuning_best.json")


def _resolve_target_column(columns: list[str], target: str) -> str:
    if target in columns:
        return target
    lower_map = {c.lower(): c for c in columns}
    if target.lower() in lower_map:
        return lower_map[target.lower()]
    raise ValueError(f"Target column '{target}' not found.")


def _to_met_count(series: pd.Series) -> pd.Series:
    normalized = series.fillna("nil").astype(str).str.strip().str.lower()
    mapped = normalized.map(
        {
            "nil": 0,
            "none": 0,
            "0": 0,
            "one": 1,
            "1": 1,
            "twoplus": 2,
            "two_plus": 2,
            "two plus": 2,
            "2": 2,
            "2+": 2,
        }
    )
    if mapped.isna().any():
        unknown = sorted(normalized[mapped.isna()].unique().tolist())
        raise ValueError(f"Unsupported METWithinPrev24h values: {unknown}")
    return mapped.astype(int)


class FeatureEngineer:
    def __init__(self, config: dict[str, Any]):
        self.config = config

    def fit(self, df: pd.DataFrame) -> "FeatureEngineer":
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df[BASE_FEATURES].copy()
        met_count = _to_met_count(out["METWithinPrev24h"])

        if self.config["use_summary"]:
            any_met = (met_count >= 1).astype(int)
            two_plus_met = (met_count >= 2).astype(int)
            instability_count = (
                out["AdmitPrev24h"].astype(int)
                + out["SurgPrev24h"].astype(int)
                + out["ICUDischPrev24h"].astype(int)
                + any_met
            )
            instability_weighted_count = (
                out["AdmitPrev24h"].astype(int)
                + out["SurgPrev24h"].astype(int)
                + out["ICUDischPrev24h"].astype(int)
                + met_count
            )

            out["METWithinPrev24hCount"] = met_count
            out["AnyMETWithinPrev24h"] = any_met
            out["TwoPlusMETWithinPrev24h"] = two_plus_met
            out["AcuteInstabilityCount"] = instability_count
            out["AcuteInstabilityWeightedCount"] = instability_weighted_count
            out["AnyRecentInstability"] = (instability_count >= 1).astype(int)
            out["MultiRecentInstability"] = (instability_count >= 2).astype(int)

        if self.config["use_flags"]:
            out["AgeGE75"] = (out["Age"] >= 75).astype(int)
            out["AgeGE85"] = (out["Age"] >= 85).astype(int)
            out["NEWSGE5"] = (out["NEWS"] >= 5).astype(int)
            out["NEWSGE7"] = (out["NEWS"] >= 7).astype(int)
            out["NEWSGE10"] = (out["NEWS"] >= 10).astype(int)
            out["SOFAGE2"] = (out["SOFA"] >= 2).astype(int)
            out["SOFAGE4"] = (out["SOFA"] >= 4).astype(int)
            out["SOFAGE6"] = (out["SOFA"] >= 6).astype(int)
            out["HighAcuity"] = ((out["NEWS"] >= 7) | (out["SOFA"] >= 4)).astype(int)

        return out


def _evaluate_probabilities(y_true: pd.Series, y_prob: np.ndarray) -> dict[str, float]:
    y_arr = y_true.astype(int).to_numpy()
    return {
        "roc_auc": float(roc_auc_score(y_arr, y_prob)),
        "pr_auc": float(average_precision_score(y_arr, y_prob)),
        "brier": float(brier_score_loss(y_arr, y_prob)),
        "log_loss": float(log_loss(y_arr, y_prob, labels=[0, 1])),
    }


def _load_group_lasso_params() -> dict[str, Any]:
    data = json.loads(GROUP_LASSO_TUNING_PATH.read_text(encoding="utf-8"))
    return dict(data["best_params"])


def _load_group_ridge_params() -> dict[str, Any]:
    data = json.loads(GROUP_RIDGE_TUNING_PATH.read_text(encoding="utf-8"))
    return dict(data["best_params"])


def _fit_lasso_with_design(
    train_X: pd.DataFrame,
    train_y: pd.Series,
    feature_config: dict[str, Any],
    params: dict[str, Any],
) -> tuple[group_lasso.GroupLassoDesign, StandardScaler, Any]:
    design = group_lasso.GroupLassoDesign(
        spline_vars=feature_config["spline_vars"],
        interaction_vars=feature_config["interaction_vars"],
        spline_knots=int(params["spline_knots"]),
        spline_degree=int(params["spline_degree"]),
    )
    design.fit(train_X)
    X_train_raw = design.transform(train_X)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    model = fit_group_lasso_logistic(
        X_train,
        train_y.to_numpy(dtype=float),
        group_reg=float(params["group_reg"]),
        group_ids=np.array(design.group_ids),
        group_weight_mode=group_lasso.GROUP_WEIGHT_MODE,
        max_iter=group_lasso.MAX_ITER,
        tol=group_lasso.TOL,
    )
    return design, scaler, model


def _fit_ridge_with_design(
    train_X: pd.DataFrame,
    train_y: pd.Series,
    feature_config: dict[str, Any],
    params: dict[str, Any],
) -> tuple[group_ridge.GroupLassoDesign, StandardScaler, Any]:
    design = group_ridge.GroupLassoDesign(
        spline_vars=feature_config["spline_vars"],
        interaction_vars=feature_config["interaction_vars"],
        spline_knots=int(params["spline_knots"]),
        spline_degree=int(params["spline_degree"]),
    )
    design.fit(train_X)
    X_train_raw = design.transform(train_X)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    model = fit_group_ridge_logistic(
        X_train,
        train_y.to_numpy(dtype=float),
        group_ids=np.array(design.group_ids),
        ridge_reg=float(params["ridge_reg"]),
        group_weight_mode=group_ridge.GROUP_WEIGHT_MODE,
        max_iter=group_ridge.MAX_ITER,
        tol=group_ridge.TOL,
    )
    return design, scaler, model


def _predict(design: Any, scaler: StandardScaler, model: Any, X: pd.DataFrame) -> np.ndarray:
    X_raw = design.transform(X)
    X_scaled = scaler.transform(X_raw)
    return model.predict_proba(X_scaled)


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.lstrip("\ufeff") for c in df.columns]
    target_col = _resolve_target_column(df.columns.tolist(), TARGET)

    missing_base = [col for col in BASE_FEATURES if col not in df.columns]
    if missing_base:
        raise ValueError(f"Missing expected base features: {missing_base}")

    model_df = df[BASE_FEATURES + [target_col]].copy()
    model_df = model_df.dropna(subset=[target_col]).reset_index(drop=True)

    X = model_df[BASE_FEATURES]
    y = model_df[target_col].astype(int)

    X_train_val, X_test_unused, y_train_val, y_test_unused = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=SEED,
        stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.25,
        random_state=SEED,
        stratify=y_train_val,
    )

    lasso_params = _load_group_lasso_params()
    ridge_params = _load_group_ridge_params()

    rows: list[dict[str, Any]] = []
    for feature_config in FEATURE_SETS:
        engineer = FeatureEngineer(feature_config).fit(X_train)
        train_feat = engineer.transform(X_train)
        val_feat = engineer.transform(X_val)
        added_features = [col for col in train_feat.columns if col not in BASE_FEATURES]

        for model_name in ["group_lasso", "group_ridge"]:
            start = time.perf_counter()
            if model_name == "group_lasso":
                design, scaler, model = _fit_lasso_with_design(train_feat, y_train, feature_config, lasso_params)
                prob = _predict(design, scaler, model, val_feat)
                selected_features = int(np.sum(model.active_mask_))
                selected_groups = int(len(model.active_groups_))
                hyperparameters = lasso_params
            else:
                design, scaler, model = _fit_ridge_with_design(train_feat, y_train, feature_config, ridge_params)
                prob = _predict(design, scaler, model, val_feat)
                selected_features = int(np.sum(model.active_mask_))
                selected_groups = int(len(model.active_groups_))
                hyperparameters = ridge_params
            elapsed = time.perf_counter() - start

            metrics = _evaluate_probabilities(y_val, prob)
            row = {
                "model": model_name,
                "experiment": feature_config["name"],
                "runtime_sec": round(elapsed, 3),
                "feature_count": int(train_feat.shape[1]),
                "added_feature_count": int(len(added_features)),
                "added_features": added_features,
                "design_feature_count": int(len(design.feature_names)),
                "design_group_count": int(design.group_ids[-1]) if design.group_ids else 0,
                "selected_feature_count": selected_features,
                "selected_group_count": selected_groups,
                "spline_vars": feature_config["spline_vars"],
                "interaction_vars": feature_config["interaction_vars"],
                "validation_metrics": metrics,
                "hyperparameters": hyperparameters,
            }
            rows.append(row)
            print(
                f"{model_name} | {feature_config['name']}: "
                f"val_pr_auc={metrics['pr_auc']:.4f}, val_roc_auc={metrics['roc_auc']:.4f}, "
                f"val_brier={metrics['brier']:.4f}, design_features={len(design.feature_names)}, runtime={elapsed:.1f}s"
            )

    leaderboard = pd.DataFrame(
        [
            {
                "model": row["model"],
                "experiment": row["experiment"],
                "feature_count": row["feature_count"],
                "design_feature_count": row["design_feature_count"],
                "selected_feature_count": row["selected_feature_count"],
                "selected_group_count": row["selected_group_count"],
                "val_pr_auc": row["validation_metrics"]["pr_auc"],
                "val_roc_auc": row["validation_metrics"]["roc_auc"],
                "val_brier": row["validation_metrics"]["brier"],
                "val_log_loss": row["validation_metrics"]["log_loss"],
                "runtime_sec": row["runtime_sec"],
            }
            for row in rows
        ]
    ).sort_values(by=["model", "val_pr_auc", "val_roc_auc", "val_brier"], ascending=[True, False, False, True])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "experiments_full.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    leaderboard.to_csv(OUTPUT_DIR / "leaderboard.csv", index=False)
    (OUTPUT_DIR / "metadata.json").write_text(
        json.dumps(
            {
                "data_path": str(DATA_PATH),
                "target": target_col,
                "base_features": BASE_FEATURES,
                "split_counts": {
                    "train": int(len(X_train)),
                    "val": int(len(X_val)),
                    "held_out_test_unused": int(len(X_test_unused)),
                },
                "feature_sets": [config["name"] for config in FEATURE_SETS],
                "seed": SEED,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Saved {OUTPUT_DIR / 'leaderboard.csv'}")


if __name__ == "__main__":
    main()
