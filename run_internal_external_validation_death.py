from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from tabicl import TabICLClassifier

import run_group_lasso_death_tune as group_lasso
import run_group_ridge_death_tune as group_ridge
import run_random_forest_death_tune as random_forest
from internal_external_validation_common import (
    DEFAULT_FACILITY,
    DEFAULT_ID,
    DEFAULT_TARGET,
    auc_with_hanley_mcneil_variance,
    iter_site_splits,
    pool_auc_hksj,
    prepare_site_holdout_data,
)


DATA_PATH = Path("/Users/ac/comforter/NewModelV3.csv")
OUTPUT_DIR = Path("/Users/ac/comforter/artifacts/internal_external_validation_death")
SEED = 42

GROUP_LASSO_REPORT = Path("/Users/ac/comforter/artifacts/group_lasso_death_tuned/report.json")
GROUP_RIDGE_REPORT = Path("/Users/ac/comforter/artifacts/group_ridge_death_tuned/report.json")
RANDOM_FOREST_REPORT = Path("/Users/ac/comforter/artifacts/random_forest_death_tuned/report.json")
TABICL_REPORT = Path("/Users/ac/comforter/artifacts/tabicl_death_final/final_report.json")

GROUP_LASSO_DEFAULT = {"group_reg": 0.1, "spline_knots": 3, "spline_degree": 2}
GROUP_RIDGE_DEFAULT = {"ridge_reg": 0.3, "spline_knots": 3, "spline_degree": 3}
RANDOM_FOREST_DEFAULT = {
    "n_estimators": 400,
    "max_depth": 8,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
}
TABICL_DEFAULT = {
    "n_estimators": 8,
    "norm_methods": None,
    "feat_shuffle_method": "shift",
    "class_shuffle_method": "shift",
    "outlier_threshold": 4.0,
    "softmax_temperature": 0.9,
    "average_logits": True,
    "support_many_classes": True,
    "batch_size": 8,
    "checkpoint_version": "tabicl-classifier-v2-20260212.ckpt",
    "device": "cpu",
}

TABICL_FEATURE_SETS = ["raw_core", "raw_plus_instability_summary"]
TABICL_INSTABILITY_REQUIREMENTS = {"AdmitPrev24h", "SurgPrev24h", "ICUDischPrev24h", "METWithinPrev24h"}
INNER_CV_SPLITS = 5


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_group_lasso_config() -> tuple[dict[str, Any], str]:
    report = _load_json(GROUP_LASSO_REPORT)
    if report and isinstance(report.get("best_params"), dict):
        return dict(report["best_params"]), str(GROUP_LASSO_REPORT)
    return dict(GROUP_LASSO_DEFAULT), "fallback_defaults"


def _load_group_ridge_config() -> tuple[dict[str, Any], str]:
    report = _load_json(GROUP_RIDGE_REPORT)
    hyper = report.get("metadata", {}).get("hyperparameters") if report else None
    if isinstance(hyper, dict):
        return {
            "ridge_reg": float(hyper["ridge_reg"]),
            "spline_knots": int(hyper["spline_knots"]),
            "spline_degree": int(hyper["spline_degree"]),
        }, str(GROUP_RIDGE_REPORT)
    return dict(GROUP_RIDGE_DEFAULT), "fallback_defaults"


def _load_random_forest_config() -> tuple[dict[str, Any], str]:
    report = _load_json(RANDOM_FOREST_REPORT)
    hyper = report.get("metadata", {}).get("hyperparameters") if report else None
    if isinstance(hyper, dict):
        return {
            "n_estimators": int(hyper["n_estimators"]),
            "max_depth": None if hyper["max_depth"] is None else int(hyper["max_depth"]),
            "min_samples_split": int(hyper["min_samples_split"]),
            "min_samples_leaf": int(hyper["min_samples_leaf"]),
            "max_features": hyper["max_features"],
        }, str(RANDOM_FOREST_REPORT)
    return dict(RANDOM_FOREST_DEFAULT), "fallback_defaults"


def _load_tabicl_config(feat_shuffle_override: str | None = None) -> tuple[dict[str, Any], str]:
    report = _load_json(TABICL_REPORT)
    config = report.get("metadata", {}).get("model_config") if report else None
    if isinstance(config, dict):
        merged = dict(TABICL_DEFAULT)
        merged.update(config)
        source = str(TABICL_REPORT)
    else:
        merged = dict(TABICL_DEFAULT)
        source = "fallback_defaults"
    if feat_shuffle_override is not None:
        merged["feat_shuffle_method"] = feat_shuffle_override
        source = f"{source} + feat_shuffle_method={feat_shuffle_override}"
    return merged, source


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


def _to_binary_flag(series: pd.Series) -> pd.Series:
    normalized = series.fillna("no").astype(str).str.strip().str.lower()
    mapped = normalized.map(
        {
            "0": 0,
            "1": 1,
            "false": 0,
            "true": 1,
            "no": 0,
            "yes": 1,
            "n": 0,
            "y": 1,
        }
    )
    if mapped.isna().any():
        unknown = sorted(normalized[mapped.isna()].unique().tolist())
        raise ValueError(f"Unsupported binary values: {unknown}")
    return mapped.astype(int)


def _apply_tabicl_feature_set(df: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    if feature_set not in TABICL_FEATURE_SETS:
        raise ValueError(f"Unsupported TabICL feature set: {feature_set}")

    out = df.copy()
    if feature_set == "raw_core":
        return out

    met_count = _to_met_count(out["METWithinPrev24h"])
    admit_prev = _to_binary_flag(out["AdmitPrev24h"])
    surg_prev = _to_binary_flag(out["SurgPrev24h"])
    icu_disch_prev = _to_binary_flag(out["ICUDischPrev24h"])
    any_met = (met_count >= 1).astype(int)
    two_plus_met = (met_count >= 2).astype(int)
    instability_count = (
        admit_prev
        + surg_prev
        + icu_disch_prev
        + any_met
    )
    instability_weighted_count = (
        admit_prev
        + surg_prev
        + icu_disch_prev
        + met_count
    )

    out["AcuteInstabilityCount"] = instability_count
    out["AcuteInstabilityWeightedCount"] = instability_weighted_count
    out["METWithinPrev24hCount"] = met_count
    out["AnyMETWithinPrev24h"] = any_met
    out["TwoPlusMETWithinPrev24h"] = two_plus_met
    out["AnyRecentInstability"] = (instability_count >= 1).astype(int)
    out["MultiRecentInstability"] = (instability_count >= 2).astype(int)
    return out


def _predict_group_lasso(train_X: pd.DataFrame, train_y: pd.Series, test_X: pd.DataFrame, config: dict[str, Any]) -> pd.Series:
    design, scaler, model = group_lasso._fit_gl_model(train_X, train_y, config)
    prob = group_lasso._predict_prob(design, scaler, model, test_X)
    return pd.Series(prob, index=test_X.index, dtype=float)


def _predict_group_ridge(train_X: pd.DataFrame, train_y: pd.Series, test_X: pd.DataFrame, config: dict[str, Any]) -> pd.Series:
    design, scaler, model = group_ridge._fit_model(train_X, train_y, config)
    prob = group_ridge._predict_prob(design, scaler, model, test_X)
    return pd.Series(prob, index=test_X.index, dtype=float)


def _predict_random_forest(train_X: pd.DataFrame, train_y: pd.Series, test_X: pd.DataFrame, config: dict[str, Any]) -> pd.Series:
    prob = random_forest._fit_predict_prob(train_X, train_y, test_X, config)
    return pd.Series(prob, index=test_X.index, dtype=float)


def _effective_cv_splits(y: pd.Series, requested_splits: int) -> int:
    counts = y.value_counts()
    if counts.empty:
        return 0
    min_class_n = int(counts.min())
    return int(max(0, min(requested_splits, min_class_n)))


def _select_lasso_lambda_inner_cv(
    train_X: pd.DataFrame,
    train_y: pd.Series,
    base_config: dict[str, Any],
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidates = sorted({float(v) for v in list(group_lasso.GROUP_REG_GRID) + [float(base_config["group_reg"])]})
    n_splits = _effective_cv_splits(train_y, INNER_CV_SPLITS)
    if n_splits < 2:
        return dict(base_config), {
            "status": "skipped",
            "reason": "insufficient_class_counts_for_cv",
            "selected_group_reg": float(base_config["group_reg"]),
            "n_splits": int(n_splits),
        }

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y_arr = train_y.to_numpy(dtype=int)
    rows: list[dict[str, float | int]] = []
    for reg in candidates:
        auc_scores: list[float] = []
        brier_scores: list[float] = []
        for fit_idx, val_idx in skf.split(train_X, y_arr):
            fold_train_X = train_X.iloc[fit_idx]
            fold_train_y = train_y.iloc[fit_idx]
            fold_val_X = train_X.iloc[val_idx]
            fold_val_y = train_y.iloc[val_idx]
            cfg = dict(base_config)
            cfg["group_reg"] = float(reg)
            design, scaler, model = group_lasso._fit_gl_model(fold_train_X, fold_train_y, cfg)
            prob = group_lasso._predict_prob(design, scaler, model, fold_val_X)
            y_true = fold_val_y.to_numpy(dtype=int)
            auc_scores.append(float(roc_auc_score(y_true, prob)))
            brier_scores.append(float(brier_score_loss(y_true, prob)))
        rows.append(
            {
                "group_reg": float(reg),
                "cv_mean_auc": float(np.mean(auc_scores)),
                "cv_mean_brier": float(np.mean(brier_scores)),
                "cv_folds": int(n_splits),
            }
        )
    cv_df = pd.DataFrame(rows).sort_values(["cv_mean_auc", "cv_mean_brier"], ascending=[False, True]).reset_index(drop=True)
    best = cv_df.iloc[0]
    out = dict(base_config)
    out["group_reg"] = float(best["group_reg"])
    return out, {
        "status": "selected",
        "selected_group_reg": float(best["group_reg"]),
        "n_splits": int(n_splits),
        "candidate_results": cv_df.to_dict(orient="records"),
    }


def _select_ridge_lambda_inner_cv(
    train_X: pd.DataFrame,
    train_y: pd.Series,
    base_config: dict[str, Any],
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidates = sorted({float(v) for v in list(group_ridge.RIDGE_REG_GRID) + [float(base_config["ridge_reg"])]})
    n_splits = _effective_cv_splits(train_y, INNER_CV_SPLITS)
    if n_splits < 2:
        return dict(base_config), {
            "status": "skipped",
            "reason": "insufficient_class_counts_for_cv",
            "selected_ridge_reg": float(base_config["ridge_reg"]),
            "n_splits": int(n_splits),
        }

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y_arr = train_y.to_numpy(dtype=int)
    rows: list[dict[str, float | int]] = []
    for reg in candidates:
        auc_scores: list[float] = []
        brier_scores: list[float] = []
        for fit_idx, val_idx in skf.split(train_X, y_arr):
            fold_train_X = train_X.iloc[fit_idx]
            fold_train_y = train_y.iloc[fit_idx]
            fold_val_X = train_X.iloc[val_idx]
            fold_val_y = train_y.iloc[val_idx]
            cfg = dict(base_config)
            cfg["ridge_reg"] = float(reg)
            design, scaler, model = group_ridge._fit_model(fold_train_X, fold_train_y, cfg)
            prob = group_ridge._predict_prob(design, scaler, model, fold_val_X)
            y_true = fold_val_y.to_numpy(dtype=int)
            auc_scores.append(float(roc_auc_score(y_true, prob)))
            brier_scores.append(float(brier_score_loss(y_true, prob)))
        rows.append(
            {
                "ridge_reg": float(reg),
                "cv_mean_auc": float(np.mean(auc_scores)),
                "cv_mean_brier": float(np.mean(brier_scores)),
                "cv_folds": int(n_splits),
            }
        )
    cv_df = pd.DataFrame(rows).sort_values(["cv_mean_auc", "cv_mean_brier"], ascending=[False, True]).reset_index(drop=True)
    best = cv_df.iloc[0]
    out = dict(base_config)
    out["ridge_reg"] = float(best["ridge_reg"])
    return out, {
        "status": "selected",
        "selected_ridge_reg": float(best["ridge_reg"]),
        "n_splits": int(n_splits),
        "candidate_results": cv_df.to_dict(orient="records"),
    }


def _predict_tabicl(
    train_X: pd.DataFrame,
    train_y: pd.Series,
    test_X: pd.DataFrame,
    config: dict[str, Any],
    seed: int,
    feature_set: str,
) -> pd.Series:
    train_features = _apply_tabicl_feature_set(train_X, feature_set)
    test_features = _apply_tabicl_feature_set(test_X, feature_set)
    clf = TabICLClassifier(
        n_estimators=int(config["n_estimators"]),
        norm_methods=config.get("norm_methods"),
        feat_shuffle_method=config.get("feat_shuffle_method"),
        class_shuffle_method=config.get("class_shuffle_method"),
        outlier_threshold=float(config.get("outlier_threshold", 4.0)),
        softmax_temperature=float(config.get("softmax_temperature", 0.9)),
        average_logits=bool(config.get("average_logits", True)),
        support_many_classes=bool(config.get("support_many_classes", True)),
        batch_size=int(config.get("batch_size", 8)),
        checkpoint_version=config.get("checkpoint_version"),
        device=config.get("device", "cpu"),
        random_state=seed,
        verbose=False,
    )
    clf.fit(train_features, train_y)
    prob = clf.predict_proba(test_features)[:, 1]
    return pd.Series(prob, index=test_X.index, dtype=float)


def _build_model_registry(seed: int, tabicl_feat_shuffle_method: str | None = None) -> dict[str, dict[str, Any]]:
    gl_config, gl_source = _load_group_lasso_config()
    gr_config, gr_source = _load_group_ridge_config()
    rf_config, rf_source = _load_random_forest_config()
    tabicl_config, tabicl_source = _load_tabicl_config(tabicl_feat_shuffle_method)

    return {
        "tabicl": {
            "config": {**tabicl_config, "feature_set": "raw_core"},
            "config_source": tabicl_source,
            "predict_fn": lambda train_X, train_y, test_X, config: _predict_tabicl(
                train_X, train_y, test_X, config, seed, "raw_core"
            ),
        },
        "tabicl_instability_summary": {
            "config": {**tabicl_config, "feature_set": "raw_plus_instability_summary"},
            "config_source": f"{tabicl_source} + feature_set=raw_plus_instability_summary",
            "predict_fn": lambda train_X, train_y, test_X, config: _predict_tabicl(
                train_X, train_y, test_X, config, seed, "raw_plus_instability_summary"
            ),
        },
        "group_lasso": {
            "config": gl_config,
            "config_source": gl_source,
            "predict_fn": _predict_group_lasso,
        },
        "group_ridge": {
            "config": gr_config,
            "config_source": gr_source,
            "predict_fn": _predict_group_ridge,
        },
        "random_forest": {
            "config": rf_config,
            "config_source": rf_source,
            "predict_fn": _predict_random_forest,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Facility-holdout internal-external validation for DeathHospDisch models.")
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["tabicl", "tabicl_instability_summary", "group_lasso", "group_ridge", "random_forest"],
        default=["tabicl", "group_lasso", "group_ridge", "random_forest"],
    )
    parser.add_argument("--target", type=str, default=DEFAULT_TARGET)
    parser.add_argument("--facility-col", type=str, default=DEFAULT_FACILITY)
    parser.add_argument("--id-col", type=str, default=DEFAULT_ID)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--tabicl-feat-shuffle-method",
        choices=["none", "shift", "random", "latin"],
        default=None,
        help="Optional override for TabICL feat_shuffle_method during this run.",
    )
    parser.add_argument(
        "--drop-predictors",
        nargs="*",
        default=[],
        help="Predictor columns to remove before model fitting (site-holdout internal-external).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    random_forest.RANDOM_STATE = args.seed
    drop_predictors = [col for col in args.drop_predictors if col]
    if drop_predictors:
        print(f"Dropping predictors from this run: {', '.join(drop_predictors)}")

    if args.models and drop_predictors:
        if "tabicl_instability_summary" in args.models:
            if set(drop_predictors) & TABICL_INSTABILITY_REQUIREMENTS:
                print(
                    "Note: tabicl_instability_summary requires raw instability components; "
                    "skipping this model because one or more instability components are dropped."
                )
                args.models = [m for m in args.models if m != "tabicl_instability_summary"]
            if not args.models:
                raise ValueError("No models selected for evaluation after applying predictor exclusions.")

    model_df, feature_cols, facility_levels = prepare_site_holdout_data(
        args.data,
        target_col=args.target,
        facility_col=args.facility_col,
        id_col=args.id_col,
        extra_excludes=drop_predictors,
    )
    site_splits = iter_site_splits(
        model_df,
        feature_cols,
        target_col=args.target,
        facility_col=args.facility_col,
        id_col=args.id_col,
    )

    model_registry = _build_model_registry(args.seed, args.tabicl_feat_shuffle_method)
    selected_models = {name: model_registry[name] for name in args.models}

    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    report_models: dict[str, Any] = {}

    for model_name, model_spec in selected_models.items():
        predict_fn: Callable[[pd.DataFrame, pd.Series, pd.DataFrame, dict[str, Any]], pd.Series] = model_spec["predict_fn"]
        model_fold_rows: list[dict[str, Any]] = []

        model_tuning_rows: list[dict[str, Any]] = []
        for split_idx, split in enumerate(site_splits):
            run_config = dict(model_spec["config"])
            tuning_info: dict[str, Any] | None = None
            inner_seed = int(args.seed + 1000 + split_idx)
            if model_name == "group_lasso":
                run_config, tuning_info = _select_lasso_lambda_inner_cv(
                    split["train_X"],
                    split["train_y"],
                    run_config,
                    seed=inner_seed,
                )
            elif model_name == "group_ridge":
                run_config, tuning_info = _select_ridge_lambda_inner_cv(
                    split["train_X"],
                    split["train_y"],
                    run_config,
                    seed=inner_seed,
                )

            y_prob = predict_fn(split["train_X"], split["train_y"], split["test_X"], run_config)
            auc_stats = auc_with_hanley_mcneil_variance(split["test_y"].to_numpy(), y_prob.to_numpy())

            fold_row = {
                "model": model_name,
                "test_site": split["test_site"],
                "train_sites": "|".join(split["train_sites"]),
                "auc": float(auc_stats["auc"]),
                "auc_variance": float(auc_stats["auc_variance"]),
                "auc_se": float(auc_stats["auc_se"]),
                "n_test": int(auc_stats["n_test"]),
                "n_pos": int(auc_stats["n_pos"]),
                "n_neg": int(auc_stats["n_neg"]),
                "prevalence": float(split["test_y"].mean()),
            }
            if model_name == "group_lasso":
                fold_row["selected_group_reg"] = float(run_config["group_reg"])
            if model_name == "group_ridge":
                fold_row["selected_ridge_reg"] = float(run_config["ridge_reg"])
            fold_rows.append(fold_row)
            model_fold_rows.append(fold_row)
            if tuning_info is not None:
                model_tuning_rows.append(
                    {
                        "test_site": split["test_site"],
                        "train_sites": "|".join(split["train_sites"]),
                        "selected_config": run_config,
                        "inner_cv": tuning_info,
                    }
                )

            prediction_rows.extend(
                {
                    "model": model_name,
                    "test_site": split["test_site"],
                    "train_sites": "|".join(split["train_sites"]),
                    "row_index": int(row_index),
                    "id": int(row_id) if str(row_id).isdigit() else row_id,
                    "y_true": int(y_true),
                    "proba_death": float(prob),
                    "proba_survival": float(1.0 - prob),
                }
                for row_index, row_id, y_true, prob in zip(
                    split["test_index"],
                    split["test_ids"],
                    split["test_y"].to_numpy(),
                    y_prob.to_numpy(),
                    strict=False,
                )
            )

        model_fold_df = pd.DataFrame(model_fold_rows).sort_values("test_site").reset_index(drop=True)
        pooled = pool_auc_hksj(model_fold_df)
        report_models[model_name] = {
            "config_source": model_spec["config_source"],
            "model_config": model_spec["config"],
            "fold_auc": model_fold_df.to_dict(orient="records"),
            "pooled_auc_random_effects": pooled,
        }
        if model_tuning_rows:
            report_models[model_name]["fold_specific_inner_cv_tuning"] = model_tuning_rows

    args.output_dir.mkdir(parents=True, exist_ok=True)

    fold_df = pd.DataFrame(fold_rows).sort_values(["model", "test_site"]).reset_index(drop=True)
    prediction_df = pd.DataFrame(prediction_rows).sort_values(["model", "test_site", "row_index"]).reset_index(drop=True)
    summary_df = pd.DataFrame(
        [
            {
                "model": model_name,
                "pooled_auc": model_report["pooled_auc_random_effects"]["pooled_auc"],
                "ci_95_lower": model_report["pooled_auc_random_effects"]["pooled_auc_ci_95"]["lower"],
                "ci_95_upper": model_report["pooled_auc_random_effects"]["pooled_auc_ci_95"]["upper"],
                "pi_95_lower": model_report["pooled_auc_random_effects"]["pooled_auc_prediction_interval_95"]["lower"],
                "pi_95_upper": model_report["pooled_auc_random_effects"]["pooled_auc_prediction_interval_95"]["upper"],
                "tau2_logit_auc": model_report["pooled_auc_random_effects"]["tau2_logit_auc"],
                "i2_percent": model_report["pooled_auc_random_effects"]["i2_percent"],
            }
            for model_name, model_report in report_models.items()
        ]
    ).sort_values("pooled_auc", ascending=False, ignore_index=True)

    report = {
        "metadata": {
            "data_path": str(args.data),
            "output_dir": str(args.output_dir),
            "target": args.target,
            "facility_column": args.facility_col,
            "id_column": args.id_col,
            "seed": int(args.seed),
            "feature_columns": feature_cols,
        "facility_levels": facility_levels,
        "dropped_predictors": drop_predictors,
        "model_count": int(len(report_models)),
        "models": list(report_models.keys()),
    },
        "models": report_models,
    }

    fold_df.to_csv(args.output_dir / "fold_auc.csv", index=False)
    prediction_df.to_csv(args.output_dir / "all_predictions.csv", index=False)
    summary_df.to_csv(args.output_dir / "summary_auc.csv", index=False)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Saved {args.output_dir / 'fold_auc.csv'}")
    print(f"Saved {args.output_dir / 'all_predictions.csv'}")
    print(f"Saved {args.output_dir / 'summary_auc.csv'}")
    print(f"Saved {args.output_dir / 'report.json'}")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
