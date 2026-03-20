from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from tabicl import TabICLClassifier

from internal_external_validation_common import (
    LOGIT_EPS,
    auc_with_hanley_mcneil_variance,
    iter_site_splits,
    load_dataset,
)

DATA_PATH = Path("/Users/ac/comforter/NewModelV3.csv")
OUTPUT_DIR = Path("/Users/ac/comforter/artifacts/internal_external_validation_death_tabicl_feature_search")
TARGET = "DeathHospDisch"
FACILITY_COL = "Facility"
ID_COL = "Id"
SEED = 42

TABICL_BASE_PARAMS = {
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

DEFAULT_FEATURE_SETS = [
    {
        "name": "raw_core",
        "added_features": [],
    },
    {
        "name": "raw_plus_instability_summary",
        "added_features": [
            "METWithinPrev24hCount",
            "AnyMETWithinPrev24h",
            "TwoPlusMETWithinPrev24h",
            "AcuteInstabilityCount",
            "AcuteInstabilityWeightedCount",
            "AnyRecentInstability",
            "MultiRecentInstability",
        ],
    },
    {
        "name": "raw_plus_acute_instability_count",
        "added_features": ["AcuteInstabilityCount"],
    },
    {
        "name": "raw_plus_acute_instability_weighted_count",
        "added_features": ["AcuteInstabilityWeightedCount"],
    },
    {
        "name": "raw_plus_high_acuity",
        "added_features": ["HighAcuity"],
    },
    {
        "name": "raw_plus_instability_count_high_acuity",
        "added_features": ["AcuteInstabilityCount", "HighAcuity"],
    },
    {
        "name": "raw_plus_summary_flags",
        "added_features": [
            "METWithinPrev24hCount",
            "AnyMETWithinPrev24h",
            "TwoPlusMETWithinPrev24h",
            "AcuteInstabilityCount",
            "AcuteInstabilityWeightedCount",
            "AnyRecentInstability",
            "MultiRecentInstability",
            "AgeGE75",
            "AgeGE85",
            "NEWSGE5",
            "NEWSGE7",
            "NEWSGE10",
            "SOFAGE2",
            "SOFAGE4",
            "SOFAGE6",
            "HighAcuity",
        ],
    },
]


def _to_met_count(series: pd.Series) -> pd.Series:
    normalized = series.fillna("nil").astype(str).str.strip().str.lower()
    mapped = normalized.map(
        {
            "nil": 0,
            "none": 0,
            "0": 0,
            "1": 1,
            "one": 1,
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
    normalized = series.fillna("0").astype(str).str.strip().str.lower()
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


def _apply_feature_set(df: pd.DataFrame, feature_set: dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    met_count = _to_met_count(out["METWithinPrev24h"])
    admit_prev = _to_binary_flag(out["AdmitPrev24h"])
    surg_prev = _to_binary_flag(out["SurgPrev24h"])
    icu_disch_prev = _to_binary_flag(out["ICUDischPrev24h"])
    any_met = (met_count >= 1).astype(int)
    two_plus_met = (met_count >= 2).astype(int)
    instability_count = admit_prev + surg_prev + icu_disch_prev + any_met
    instability_weighted_count = admit_prev + surg_prev + icu_disch_prev + met_count
    news = out["NEWS"].astype(float)
    sofa = out["SOFA"].astype(float)

    if "METWithinPrev24hCount" in feature_set["added_features"]:
        out["METWithinPrev24hCount"] = met_count
    if "AnyMETWithinPrev24h" in feature_set["added_features"]:
        out["AnyMETWithinPrev24h"] = any_met
    if "TwoPlusMETWithinPrev24h" in feature_set["added_features"]:
        out["TwoPlusMETWithinPrev24h"] = two_plus_met
    if "AcuteInstabilityCount" in feature_set["added_features"]:
        out["AcuteInstabilityCount"] = instability_count
    if "AcuteInstabilityWeightedCount" in feature_set["added_features"]:
        out["AcuteInstabilityWeightedCount"] = instability_weighted_count
    if "AnyRecentInstability" in feature_set["added_features"]:
        out["AnyRecentInstability"] = (instability_count >= 1).astype(int)
    if "MultiRecentInstability" in feature_set["added_features"]:
        out["MultiRecentInstability"] = (instability_count >= 2).astype(int)
    if "AgeGE75" in feature_set["added_features"]:
        out["AgeGE75"] = (out["Age"] >= 75).astype(int)
    if "AgeGE85" in feature_set["added_features"]:
        out["AgeGE85"] = (out["Age"] >= 85).astype(int)
    if "NEWSGE5" in feature_set["added_features"]:
        out["NEWSGE5"] = (news >= 5).astype(int)
    if "NEWSGE7" in feature_set["added_features"]:
        out["NEWSGE7"] = (news >= 7).astype(int)
    if "NEWSGE10" in feature_set["added_features"]:
        out["NEWSGE10"] = (news >= 10).astype(int)
    if "SOFAGE2" in feature_set["added_features"]:
        out["SOFAGE2"] = (sofa >= 2).astype(int)
    if "SOFAGE4" in feature_set["added_features"]:
        out["SOFAGE4"] = (sofa >= 4).astype(int)
    if "SOFAGE6" in feature_set["added_features"]:
        out["SOFAGE6"] = (sofa >= 6).astype(int)
    if "HighAcuity" in feature_set["added_features"]:
        out["HighAcuity"] = ((news >= 7) | (sofa >= 4)).astype(int)
    return out


def _positive_probability(clf: TabICLClassifier, X: pd.DataFrame) -> np.ndarray:
    prob = clf.predict_proba(X)
    classes = list(clf.classes_)
    if 1 in classes:
        pos_idx = classes.index(1)
    elif 1.0 in classes:
        pos_idx = classes.index(1.0)
    else:
        raise ValueError(f"Positive class not found in model classes {classes}")
    return np.asarray(prob[:, pos_idx], dtype=float)


def _calibration_stats(y_true: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(probs, dtype=float), LOGIT_EPS, 1.0 - LOGIT_EPS)
    brier = float(brier_score_loss(y, p))
    if len(np.unique(y)) < 2:
        return {
            "brier": brier,
            "citl": float("nan"),
            "slope": float("nan"),
            "abs_citl": float("nan"),
            "abs_slope_dev": float("nan"),
        }
    logits = np.log(p / (1.0 - p)).reshape(-1, 1)
    lr = LogisticRegression(C=1e12, solver="lbfgs", fit_intercept=True, max_iter=2000)
    lr.fit(logits, y)
    citl = float(lr.intercept_[0])
    slope = float(lr.coef_[0, 0])
    return {
        "brier": brier,
        "citl": citl,
        "slope": slope,
        "abs_citl": abs(citl),
        "abs_slope_dev": abs(slope - 1.0),
    }


def _probability_metrics(y_true: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    if len(np.unique(y)) < 2:
        return {"roc_auc": float("nan"), "pr_auc": float("nan"), "n_pos": int(np.sum(y)), "n_neg": int(np.sum(1 - y))}
    return {
        "roc_auc": float(roc_auc_score(y, probs)),
        "pr_auc": float(average_precision_score(y, probs)),
        "n_pos": int(np.sum(y)),
        "n_neg": int(np.sum(1 - y)),
    }


def _safe_mean(values: list[float]) -> float:
    vals = np.asarray(values, dtype=float)
    return float(np.mean(vals[~np.isnan(vals)])) if len(vals) else float("nan")


def _split_data_for_site(
    train_X: pd.DataFrame,
    train_y: pd.Series,
    val_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(
        train_X,
        train_y,
        test_size=val_size,
        random_state=seed,
        stratify=train_y,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Nested internal-external validation for TabICL using inner-site-train "
            "validation split to tune predictor sets per holdout site."
        )
    )
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--target", type=str, default=TARGET)
    parser.add_argument("--facility-col", type=str, default=FACILITY_COL)
    parser.add_argument("--id-col", type=str, default=ID_COL)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--inner-val-size",
        type=float,
        default=0.25,
        help="Fraction of training-site data used as inner validation within each outer split.",
    )
    parser.add_argument("--weight-brier", type=float, default=10.0)
    parser.add_argument("--weight-citl", type=float, default=1.0)
    parser.add_argument("--weight-slope", type=float, default=2.0)
    parser.add_argument(
        "--feature-set-json",
        type=Path,
        default=None,
        help="Optional JSON file with feature-set list. Each entry: {name, added_features}.",
    )
    return parser.parse_args()


def _prepare_feature_sets(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return DEFAULT_FEATURE_SETS
    data = json.loads(path.read_text(encoding="utf-8"))
    normalized: list[dict[str, Any]] = []
    for item in data:
        normalized.append(
            {
                "name": str(item["name"]),
                "added_features": [str(x) for x in item.get("added_features", [])],
            }
        )
    return normalized


def _prepare_site_data(args: argparse.Namespace) -> tuple[pd.DataFrame, list[str], list[str]]:
    df = load_dataset(args.data)
    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found in {args.data}")
    if args.facility_col not in df.columns:
        raise ValueError(f"Facility column '{args.facility_col}' not found in {args.data}")

    model_df = df.dropna(subset=[args.target, args.facility_col]).copy()
    model_df[args.target] = model_df[args.target].astype(int)
    feature_cols = [c for c in model_df.columns if c not in {args.target, args.facility_col, args.id_col}]
    facility_levels = sorted(model_df[args.facility_col].astype(str).unique().tolist())
    if args.id_col not in model_df.columns:
        model_df[args.id_col] = model_df.index
    return model_df, feature_cols, facility_levels


def main() -> None:
    args = parse_args()
    if not (0 < args.inner_val_size < 1):
        raise ValueError("--inner-val-size must be in (0,1).")

    model_df, feature_cols, facility_levels = _prepare_site_data(args)
    feature_sets = _prepare_feature_sets(args.feature_set_json)
    site_splits = iter_site_splits(
        model_df,
        feature_cols,
        target_col=args.target,
        facility_col=args.facility_col,
        id_col=args.id_col,
    )

    candidate_rows: list[dict[str, Any]] = []
    outer_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []

    for split_idx, split in enumerate(site_splits, start=1):
        train_X = split["train_X"].copy()
        train_y = split["train_y"].copy()
        test_X = split["test_X"].copy()
        test_y = split["test_y"].copy()
        inner_seed = args.seed + split_idx * 17

        inner_X, inner_val_X, inner_y, inner_val_y = _split_data_for_site(
            train_X, train_y, args.inner_val_size, inner_seed
        )

        config_rows: list[dict[str, Any]] = []
        for feature_set in feature_sets:
            set_name = feature_set["name"]
            params = {**TABICL_BASE_PARAMS}
            params["random_state"] = args.seed

            tuned_train = _apply_feature_set(inner_X, feature_set)
            tuned_val = _apply_feature_set(inner_val_X, feature_set)
            tuned_test = _apply_feature_set(test_X, feature_set)

            clf = TabICLClassifier(**params)
            clf.fit(tuned_train, inner_y)
            val_prob = _positive_probability(clf, tuned_val)
            test_prob = _positive_probability(clf, tuned_test)

            val_cal = _calibration_stats(inner_val_y.to_numpy(), val_prob)
            val_prob_metrics = _probability_metrics(inner_val_y.to_numpy(), val_prob)
            objective = (
                args.weight_brier * val_cal["brier"]
                + args.weight_citl * val_cal["abs_citl"]
                + args.weight_slope * val_cal["abs_slope_dev"]
            )
            config_rows.append(
                {
                    "test_site": split["test_site"],
                    "feature_set": set_name,
                    "added_feature_count": int(len(feature_set["added_features"])),
                    "val_brier": val_cal["brier"],
                    "val_citl": val_cal["citl"],
                    "val_abs_citl": val_cal["abs_citl"],
                    "val_slope": val_cal["slope"],
                    "val_abs_slope_dev": val_cal["abs_slope_dev"],
                    "val_roc_auc": val_prob_metrics["roc_auc"],
                    "val_pr_auc": val_prob_metrics["pr_auc"],
                    "val_n_pos": val_prob_metrics["n_pos"],
                    "val_n_neg": val_prob_metrics["n_neg"],
                    "validation_objective": float(objective),
                }
            )

        ranked = sorted(config_rows, key=lambda r: r["validation_objective"])
        best = ranked[0]
        best_set_name = best["feature_set"]
        best_set = next(fs for fs in feature_sets if fs["name"] == best_set_name)

        best_train = _apply_feature_set(train_X, best_set)
        best_test = _apply_feature_set(test_X, best_set)
        best_clf = TabICLClassifier(
            **{**TABICL_BASE_PARAMS, "random_state": args.seed + 101}
        )
        best_clf.fit(best_train, train_y)
        best_prob = _positive_probability(best_clf, best_test)

        hold_auc = auc_with_hanley_mcneil_variance(test_y.to_numpy(), best_prob)
        hold_prob_metrics = _probability_metrics(test_y.to_numpy(), best_prob)
        hold_cal = _calibration_stats(test_y.to_numpy(), best_prob)

        outer_rows.append(
            {
                "test_site": split["test_site"],
                "train_sites": "|".join(split["train_sites"]),
                "best_feature_set": best_set_name,
                "selected_added_features": ", ".join(best_set["added_features"]),
                "best_objective": float(best["validation_objective"]),
                "inner_val_brier": best["val_brier"],
                "inner_val_citl": best["val_citl"],
                "inner_val_slope": best["val_slope"],
                "hold_auc": hold_auc["auc"],
                "hold_auc_se": hold_auc["auc_se"],
                "hold_pr_auc": hold_prob_metrics["pr_auc"],
                "hold_brier": hold_cal["brier"],
                "hold_citl": hold_cal["citl"],
                "hold_slope": hold_cal["slope"],
                "hold_n_test": hold_auc["n_test"],
                "hold_n_pos": hold_auc["n_pos"],
                "hold_n_neg": hold_auc["n_neg"],
                "hold_prevalence": float(test_y.mean()),
            }
        )

        for row in ranked:
            candidate_rows.append(row)

        prediction_rows.extend(
            {
                "test_site": split["test_site"],
                "train_sites": "|".join(split["train_sites"]),
                "feature_set": best_set_name,
                "row_index": int(row_id),
                "id": int(row_id) if str(row_id).isdigit() else row_id,
                "y_true": int(y_true),
                "proba_death": float(prob),
                "proba_survival": float(1.0 - prob),
            }
            for row_id, y_true, prob in zip(
                split["test_index"],
                test_y.to_numpy(),
                best_prob,
                strict=False,
            )
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidate_df = pd.DataFrame(candidate_rows).sort_values(["test_site", "validation_objective"], ignore_index=True)
    outer_df = pd.DataFrame(outer_rows).sort_values("test_site", ignore_index=True)
    pred_df = pd.DataFrame(prediction_rows).sort_values(["test_site", "row_index"], ignore_index=True)

    candidate_df.to_csv(args.output_dir / "candidate_metrics_inner_validation.csv", index=False)
    outer_df.to_csv(args.output_dir / "feature_set_selection_by_site.csv", index=False)
    pred_df.to_csv(args.output_dir / "all_predictions.csv", index=False)

    candidate_summary = (
        candidate_df.groupby("feature_set", as_index=False)
        .agg(
            mean_inner_objective=("validation_objective", "mean"),
            mean_inner_val_brier=("val_brier", "mean"),
            mean_inner_val_abs_citl=("val_abs_citl", "mean"),
            mean_inner_val_abs_slope_dev=("val_abs_slope_dev", "mean"),
            n_sites=("test_site", "nunique"),
        )
        .sort_values("mean_inner_objective")
    )

    candidate_summary.to_csv(
        args.output_dir / "feature_set_summary_global_scores.csv",
        index=False,
    )

    metadata = {
        "data_path": str(args.data),
        "output_dir": str(args.output_dir),
        "target": args.target,
        "facility_column": args.facility_col,
        "id_column": args.id_col,
        "seed": int(args.seed),
        "facility_levels": facility_levels,
        "feature_set_count": int(len(feature_sets)),
        "feature_sets": [fs["name"] for fs in feature_sets],
        "inner_val_size": args.inner_val_size,
        "weights": {
            "brier": args.weight_brier,
            "citl": args.weight_citl,
            "slope": args.weight_slope,
        },
        "model_params": TABICL_BASE_PARAMS,
        "best_summary_by_site": outer_df.to_dict(orient="records"),
    }
    (args.output_dir / "report.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved {args.output_dir / 'candidate_metrics_inner_validation.csv'}")
    print(f"Saved {args.output_dir / 'feature_set_selection_by_site.csv'}")
    print(f"Saved {args.output_dir / 'feature_set_summary_global_scores.csv'}")
    print(f"Saved {args.output_dir / 'all_predictions.csv'}")
    print(f"Saved {args.output_dir / 'report.json'}")
    print(outer_df.to_string(index=False))


if __name__ == "__main__":
    main()
