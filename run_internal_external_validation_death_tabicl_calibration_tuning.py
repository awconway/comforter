from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from tabicl import TabICLClassifier

from internal_external_validation_common import (
    LOGIT_EPS,
    auc_with_hanley_mcneil_variance,
    iter_site_splits,
    load_dataset,
    prepare_site_holdout_data,
)


DATA_PATH = Path("/Users/ac/comforter/NewModelV3.csv")
TARGET = "DeathHospDisch"
FACILITY_COL = "Facility"
ID_COL = "Id"
DEFAULT_OUTPUT_DIR = Path("/Users/ac/comforter/artifacts/internal_external_validation_death_tabicl_calibration_tuning")
SEED = 42


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
    "random_state": SEED,
    "verbose": False,
}


TABICL_FEATURE_SETS = ["raw_core", "raw_plus_instability_summary"]
TABICL_INSTABILITY_REQUIREMENTS = {"AdmitPrev24h", "SurgPrev24h", "ICUDischPrev24h", "METWithinPrev24h"}


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
    instability_count = admit_prev + surg_prev + icu_disch_prev + any_met
    instability_weighted_count = admit_prev + surg_prev + icu_disch_prev + met_count

    out["AcuteInstabilityCount"] = instability_count
    out["AcuteInstabilityWeightedCount"] = instability_weighted_count
    out["METWithinPrev24hCount"] = met_count
    out["AnyMETWithinPrev24h"] = any_met
    out["TwoPlusMETWithinPrev24h"] = two_plus_met
    out["AnyRecentInstability"] = (instability_count >= 1).astype(int)
    out["MultiRecentInstability"] = (instability_count >= 2).astype(int)
    return out


def _fit_tabicl(
    train_X: pd.DataFrame,
    train_y: pd.Series,
    config: dict[str, Any],
    seed: int,
    feature_set: str,
) -> Callable[[pd.DataFrame], pd.Series]:
    train_features = _apply_tabicl_feature_set(train_X, feature_set)
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

    def predict(test_X: pd.DataFrame) -> pd.Series:
        test_features = _apply_tabicl_feature_set(test_X, feature_set)
        prob = clf.predict_proba(test_features)[:, 1]
        return pd.Series(prob, index=test_X.index, dtype=float)

    return predict


def _calibration_metrics(y_true: np.ndarray, probs: np.ndarray) -> tuple[float, float, float]:
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(probs, dtype=float), LOGIT_EPS, 1.0 - LOGIT_EPS)
    brier = float(brier_score_loss(y, p))
    logits = np.log(p / (1.0 - p)).reshape(-1, 1)
    lr = LogisticRegression(C=1e12, solver="lbfgs", fit_intercept=True, max_iter=2000)
    lr.fit(logits, y)
    citl = float(lr.intercept_[0])
    slope = float(lr.coef_[0, 0])
    return brier, citl, slope


def _build_default_experiments() -> list[dict[str, Any]]:
    return [
        {"name": "baseline", "params": {}},
        {"name": "n_estimators_16", "params": {"n_estimators": 16}},
        {"name": "n_estimators_32", "params": {"n_estimators": 32}},
        {"name": "norm_none_power_robust", "params": {"norm_methods": ["none", "power", "robust"]}},
        {"name": "norm_none_power_quantile", "params": {"norm_methods": ["none", "power", "quantile"]}},
        {"name": "feat_shuffle_latin", "params": {"feat_shuffle_method": "latin"}},
        {"name": "feat_shuffle_random", "params": {"feat_shuffle_method": "random"}},
        {"name": "outlier_3", "params": {"outlier_threshold": 3.0}},
        {"name": "outlier_6", "params": {"outlier_threshold": 6.0}},
        {"name": "softmax_temp_0_7", "params": {"softmax_temperature": 0.7}},
        {"name": "softmax_temp_1_1", "params": {"softmax_temperature": 1.1}},
        {"name": "average_logits_false", "params": {"average_logits": False}},
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune TabICL in internal-external validation with a calibration-penalized objective."
    )
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--target", type=str, default=TARGET)
    parser.add_argument("--facility-col", type=str, default=FACILITY_COL)
    parser.add_argument("--id-col", type=str, default=ID_COL)
    parser.add_argument("--drop-predictors", nargs="*", default=[], help="Predictors to remove before training.")
    parser.add_argument("--tabicl-feature-set", choices=TABICL_FEATURE_SETS, default="raw_core")
    parser.add_argument(
        "--weight-brier",
        type=float,
        default=10.0,
        help="Scale multiplier for mean Brier in objective.",
    )
    parser.add_argument(
        "--weight-citl",
        type=float,
        default=1.0,
        help="Scale multiplier for mean abs(CITL) in objective.",
    )
    parser.add_argument(
        "--weight-slope",
        type=float,
        default=2.0,
        help="Scale multiplier for mean abs(slope-1) in objective.",
    )
    parser.add_argument(
        "--weight-slope-instability",
        type=float,
        default=2.0,
        help="Scale multiplier for across-site slope instability (range) in objective.",
    )
    parser.add_argument(
        "--experiments-json",
        type=Path,
        default=None,
        help="Optional JSON file containing list of experiment dicts with name/params.",
    )
    return parser.parse_args()


def _validate_feature_set_support(tabicl_feature_set: str, drop_predictors: list[str]) -> None:
    if tabicl_feature_set == "raw_plus_instability_summary":
        if any(col in set(drop_predictors) for col in TABICL_INSTABILITY_REQUIREMENTS):
            raise ValueError(
                "raw_plus_instability_summary requires instability components "
                "(AdmitPrev24h, SurgPrev24h, ICUDischPrev24h, METWithinPrev24h)."
            )


def main() -> None:
    args = parse_args()
    seed = args.seed
    drop_predictors = [col for col in args.drop_predictors if col]
    _validate_feature_set_support(args.tabicl_feature_set, drop_predictors)

    model_df = load_dataset(args.data)
    if args.target not in model_df.columns:
        raise ValueError(f"Target column '{args.target}' not found in {args.data}")
    if args.facility_col not in model_df.columns:
        raise ValueError(f"Facility column '{args.facility_col}' not found in {args.data}")
    if args.id_col not in model_df.columns:
        raise ValueError(f"Id column '{args.id_col}' not found in {args.data}")

    if drop_predictors:
        missing = [col for col in drop_predictors if col not in model_df.columns]
        if missing:
            raise ValueError(f"Dropped predictors missing from data: {missing}")

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

    if args.experiments_json:
        experiments = json.loads(args.experiments_json.read_text(encoding="utf-8"))
        if not isinstance(experiments, list):
            raise ValueError("--experiments-json must contain a list of experiment dicts")
    else:
        experiments = _build_default_experiments()

    config_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []

    for exp in experiments:
        if not isinstance(exp, dict) or "name" not in exp:
            raise ValueError("Each experiment must be a dict with at least a 'name' key.")

        name = str(exp["name"])
        params = dict(TABICL_DEFAULT)
        params.update(exp.get("params", {}))

        fold_metrics: list[dict[str, Any]] = []
        site_brier: list[float] = []
        site_citl: list[float] = []
        site_slope_dev: list[float] = []

        for split in site_splits:
            pred_fn = _fit_tabicl(split["train_X"], split["train_y"], params, seed, args.tabicl_feature_set)
            y_prob = pred_fn(split["test_X"])
            auc_stats = auc_with_hanley_mcneil_variance(split["test_y"].to_numpy(), y_prob.to_numpy())
            brier, citl, slope = _calibration_metrics(split["test_y"].to_numpy(), y_prob.to_numpy())
            abs_citl = abs(citl)
            abs_slope_dev = abs(slope - 1.0)

            fold_rows.append(
                {
                    "experiment": name,
                    "test_site": split["test_site"],
                    "train_sites": "|".join(split["train_sites"]),
                    "auc": float(auc_stats["auc"]),
                    "auc_variance": float(auc_stats["auc_variance"]),
                    "auc_se": float(auc_stats["auc_se"]),
                    "brier": float(brier),
                    "citl": float(citl),
                    "abs_citl": float(abs_citl),
                    "slope": float(slope),
                    "abs_slope_dev": float(abs_slope_dev),
                    "n_test": int(auc_stats["n_test"]),
                    "n_pos": int(auc_stats["n_pos"]),
                    "n_neg": int(auc_stats["n_neg"]),
                    "prevalence": float(split["test_y"].mean()),
                }
            )

            fold_metrics.append(
                {
                    "test_site": split["test_site"],
                    "auc": float(auc_stats["auc"]),
                    "brier": float(brier),
                    "citl": float(citl),
                    "abs_citl": float(abs_citl),
                    "slope": float(slope),
                    "abs_slope_dev": float(abs_slope_dev),
                }
            )
            site_brier.append(float(brier))
            site_citl.append(float(abs_citl))
            site_slope_dev.append(float(abs_slope_dev))

            pred_rows.extend(
                {
                    "experiment": name,
                    "test_site": split["test_site"],
                    "id": int(row_id) if str(row_id).isdigit() else row_id,
                    "row_index": int(row_index),
                    "y_true": int(y_true),
                    "proba_death": float(prob),
                }
                for row_index, row_id, y_true, prob in zip(
                    split["test_index"],
                    split["test_ids"],
                    split["test_y"].to_numpy(),
                    y_prob.to_numpy(),
                    strict=False,
                )
            )

        fold_df = pd.DataFrame(fold_metrics)
        site_brier_arr = np.asarray(site_brier, dtype=float)
        site_citl_arr = np.asarray(site_citl, dtype=float)
        site_slope_dev_arr = np.asarray(site_slope_dev, dtype=float)

        mean_brier = float(np.mean(site_brier_arr))
        mean_abs_citl = float(np.mean(site_citl_arr))
        mean_abs_slope_dev = float(np.mean(site_slope_dev_arr))
        site_slope_range = float(np.max(site_slope_dev_arr) - np.min(site_slope_dev_arr))
        site_brier_range = float(np.max(site_brier_arr) - np.min(site_brier_arr))
        site_citl_range = float(np.max(site_citl_arr) - np.min(site_citl_arr))

        objective = (
            args.weight_brier * mean_brier
            + args.weight_citl * mean_abs_citl
            + args.weight_slope * mean_abs_slope_dev
            + args.weight_slope_instability * site_slope_range
        )

        config_rows.append(
            {
                "experiment": name,
                "n_estimators": params["n_estimators"],
                "norm_methods": str(params.get("norm_methods")),
                "feat_shuffle_method": params["feat_shuffle_method"],
                "class_shuffle_method": params["class_shuffle_method"],
                "outlier_threshold": params["outlier_threshold"],
                "softmax_temperature": params["softmax_temperature"],
                "average_logits": params["average_logits"],
                "support_many_classes": params["support_many_classes"],
                "batch_size": params["batch_size"],
                "checkpoint_version": params["checkpoint_version"],
                "device": params["device"],
                "tabicl_feature_set": args.tabicl_feature_set,
                "mean_brier": mean_brier,
                "mean_abs_citl": mean_abs_citl,
                "mean_abs_slope_dev": mean_abs_slope_dev,
                "site_brier_range": site_brier_range,
                "site_citl_range": site_citl_range,
                "site_slope_range": site_slope_range,
                "calibration_objective": objective,
            }
        )

    config_df = pd.DataFrame(config_rows).sort_values("calibration_objective", ignore_index=True)
    fold_df = pd.DataFrame(fold_rows).sort_values(["experiment", "test_site"]).reset_index(drop=True)
    pred_df = pd.DataFrame(pred_rows).sort_values(["experiment", "test_site", "row_index"]).reset_index(drop=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_df.to_csv(args.output_dir / "calibration_objective_summary.csv", index=False)
    fold_df.to_csv(args.output_dir / "fold_calibration_metrics.csv", index=False)
    pred_df.to_csv(args.output_dir / "all_predictions.csv", index=False)

    best = config_df.iloc[0]
    payload = {
        "metadata": {
            "data_path": str(args.data),
            "output_dir": str(args.output_dir),
            "target": args.target,
            "facility_column": args.facility_col,
            "id_column": args.id_col,
            "seed": seed,
            "facility_levels": facility_levels,
            "dropped_predictors": drop_predictors,
            "tabicl_feature_set": args.tabicl_feature_set,
            "weight_brier": args.weight_brier,
            "weight_citl": args.weight_citl,
            "weight_slope": args.weight_slope,
            "weight_slope_instability": args.weight_slope_instability,
            "experiments": experiments,
        },
        "best_experiment": best.to_dict(),
    }
    (args.output_dir / "tuning_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Saved {args.output_dir / 'calibration_objective_summary.csv'}")
    print(f"Saved {args.output_dir / 'fold_calibration_metrics.csv'}")
    print(f"Saved {args.output_dir / 'all_predictions.csv'}")
    print(f"Saved {args.output_dir / 'tuning_report.json'}")
    print("Best configuration (lowest calibration_objective):")
    print(json.dumps(best.to_dict(), indent=2, default=str))


if __name__ == "__main__":
    main()
