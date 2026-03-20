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

from internal_external_validation_common import LOGIT_EPS, iter_site_splits, load_dataset

DATA_PATH = Path("/Users/ac/comforter/NewModelV3.csv")
OUTPUT_DIR = Path("/Users/ac/comforter/artifacts/internal_external_validation_death_tabicl_feature_search_simple")
TARGET = "DeathHospDisch"
FACILITY_COL = "Facility"
ID_COL = "Id"
SEED = 42

TABICL_PARAMS = {
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

BASE_CANDIDATE_FEATURE_SETS = [
    {"name": "news", "feature_cols": ["NEWS"]},
    {"name": "icd", "feature_cols": ["ICD"]},
    {"name": "sofa", "feature_cols": ["SOFA"]},
    {"name": "news_icd", "feature_cols": ["NEWS", "ICD"]},
    {"name": "news_sofa", "feature_cols": ["NEWS", "SOFA"]},
    {"name": "icd_sofa", "feature_cols": ["ICD", "SOFA"]},
    {"name": "news_icd_sofa", "feature_cols": ["NEWS", "ICD", "SOFA"]},
    {"name": "news_icd_sofa_age", "feature_cols": ["NEWS", "ICD", "SOFA", "Age"]},
    {"name": "news_icd_sofa_sex", "feature_cols": ["NEWS", "ICD", "SOFA", "Sex"]},
    {"name": "news_icd_sofa_arp", "feature_cols": ["NEWS", "ICD", "SOFA", "ARP"]},
    {"name": "news_icd_sofa_met", "feature_cols": ["NEWS", "ICD", "SOFA", "METWithinPrev24h"]},
    {"name": "news_icd_sofa_admitprev", "feature_cols": ["NEWS", "ICD", "SOFA", "AdmitPrev24h"]},
    {"name": "news_icd_sofa_surgprev", "feature_cols": ["NEWS", "ICD", "SOFA", "SurgPrev24h"]},
    {"name": "news_icd_sofa_icudisch", "feature_cols": ["NEWS", "ICD", "SOFA", "ICUDischPrev24h"]},
]


def _prepare_data(
    data_path: Path,
    target_col: str,
    facility_col: str,
    id_col: str,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    df = load_dataset(data_path)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in {data_path}")
    if facility_col not in df.columns:
        raise ValueError(f"Facility column '{facility_col}' not found in {data_path}")
    if id_col not in df.columns:
        raise ValueError(f"Id column '{id_col}' not found in {data_path}")

    model_df = df.dropna(subset=[target_col, facility_col]).copy()
    model_df[target_col] = model_df[target_col].astype(int)
    feature_cols = [c for c in model_df.columns if c not in {target_col, facility_col, id_col}]
    facility_levels = sorted(model_df[facility_col].astype(str).unique().tolist())
    return model_df, feature_cols, facility_levels


def _to_positive_probs(clf: TabICLClassifier, X: pd.DataFrame) -> np.ndarray:
    prob = clf.predict_proba(X)
    classes = list(clf.classes_)
    if 1 in classes:
        idx = classes.index(1)
    elif 1.0 in classes:
        idx = classes.index(1.0)
    else:
        raise ValueError(f"Positive class not found in {classes}")
    return np.asarray(prob[:, idx], dtype=float)


def _calibration(y_true: np.ndarray, prob: np.ndarray) -> tuple[float, float, float, float, float]:
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(prob, dtype=float), LOGIT_EPS, 1.0 - LOGIT_EPS)
    brier = float(brier_score_loss(y, p))
    logits = np.log(p / (1.0 - p)).reshape(-1, 1)
    lr = LogisticRegression(C=1e12, solver="lbfgs", fit_intercept=True, max_iter=2000)
    lr.fit(logits, y)
    citl = float(lr.intercept_[0])
    slope = float(lr.coef_[0, 0])
    return brier, citl, slope, abs(citl), abs(slope - 1.0)


def _validate_discrete_metrics(y_true: np.ndarray, prob: np.ndarray) -> tuple[float, float]:
    y = np.asarray(y_true, dtype=int)
    return float(roc_auc_score(y, prob)), float(average_precision_score(y, prob))


def _build_candidate_sets(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return BASE_CANDIDATE_FEATURE_SETS
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        {"name": str(item["name"]), "feature_cols": [str(c) for c in item["feature_cols"]]}
        for item in data
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Nested internal-external validation for TabICL using an inner train/validation split "
            "within each outer training fold to select feature subsets without holdout-site leakage."
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
        help="Fraction of outer-train data used for inner validation split.",
    )
    parser.add_argument("--weight-brier", type=float, default=10.0)
    parser.add_argument("--weight-citl", type=float, default=1.0)
    parser.add_argument("--weight-slope", type=float, default=2.0)
    parser.add_argument(
        "--feature-set-json",
        type=Path,
        default=None,
        help="Optional JSON with feature-set candidates. Each entry: {name, feature_cols}.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not (0 < args.inner_val_size < 1):
        raise ValueError("--inner-val-size must be in (0,1).")

    model_df, _, facility_levels = _prepare_data(args.data, args.target, args.facility_col, args.id_col)
    feature_sets = _build_candidate_sets(args.feature_set_json)
    all_feature_cols = [c for c in model_df.columns if c not in {args.target, args.facility_col, args.id_col}]
    for fs in feature_sets:
        missing = [c for c in fs["feature_cols"] if c not in all_feature_cols]
        if missing:
            raise ValueError(f"Feature set {fs['name']} contains missing columns: {missing}")

    site_splits = iter_site_splits(
        model_df,
        all_feature_cols,
        target_col=args.target,
        facility_col=args.facility_col,
        id_col=args.id_col,
    )

    candidate_rows: list[dict[str, Any]] = []
    per_site_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []

    for split_index, split in enumerate(site_splits, start=1):
        inner_seed = args.seed + split_index * 13
        train_X = split["train_X"]
        train_y = split["train_y"]
        test_X = split["test_X"]
        test_y = split["test_y"]

        inner_train_X, inner_val_X, inner_train_y, inner_val_y = train_test_split(
            train_X,
            train_y,
            test_size=args.inner_val_size,
            random_state=inner_seed,
            stratify=train_y,
        )

        split_candidates: list[dict[str, Any]] = []
        for fs in feature_sets:
            cols = fs["feature_cols"]
            inner_train_sub = inner_train_X[cols].copy()
            inner_val_sub = inner_val_X[cols].copy()
            test_sub = test_X[cols].copy()

            clf = TabICLClassifier(**{**TABICL_PARAMS, "random_state": args.seed + split_index})
            clf.fit(inner_train_sub, inner_train_y)
            val_prob = _to_positive_probs(clf, inner_val_sub)
            brier, citl, slope, abs_citl, abs_slope_dev = _calibration(inner_val_y.to_numpy(), val_prob)
            val_roc, val_pr = _validate_discrete_metrics(inner_val_y.to_numpy(), val_prob)
            inner_objective = (
                args.weight_brier * brier
                + args.weight_citl * abs_citl
                + args.weight_slope * abs_slope_dev
            )
            split_candidates.append(
                {
                    "test_site": split["test_site"],
                    "feature_set": fs["name"],
                    "feature_count": len(cols),
                    "feature_cols": ", ".join(cols),
                    "val_auc": val_roc,
                    "val_pr_auc": val_pr,
                    "val_brier": brier,
                    "val_citl": citl,
                    "val_abs_citl": abs_citl,
                    "val_slope": slope,
                    "val_abs_slope_dev": abs_slope_dev,
                    "inner_objective": float(inner_objective),
                }
            )

        best = sorted(split_candidates, key=lambda row: row["inner_objective"])[0]
        best_set = next(fs for fs in feature_sets if fs["name"] == best["feature_set"])
        best_cols = best_set["feature_cols"]

        best_train_sub = train_X[best_cols]
        best_test_sub = test_X[best_cols]
        best_clf = TabICLClassifier(**{**TABICL_PARAMS, "random_state": args.seed + split_index + 100})
        best_clf.fit(best_train_sub, train_y)
        hold_prob = _to_positive_probs(best_clf, best_test_sub)

        hold_auc, hold_pr = _validate_discrete_metrics(test_y.to_numpy(), hold_prob)
        hold_brier, hold_citl, hold_slope, hold_abs_citl, hold_abs_slope_dev = _calibration(
            test_y.to_numpy(), hold_prob
        )
        hold_auc_n = len(test_y)
        hold_n_pos = int(np.sum(test_y == 1))
        hold_n_neg = int(np.sum(test_y == 0))

        per_site_rows.append(
            {
                "test_site": split["test_site"],
                "train_sites": "|".join(split["train_sites"]),
                "best_feature_set": best["feature_set"],
                "best_feature_cols": ", ".join(best_cols),
                "inner_objective": float(best["inner_objective"]),
                "inner_val_auc": float(best["val_auc"]),
                "inner_val_pr_auc": float(best["val_pr_auc"]),
                "inner_val_brier": float(best["val_brier"]),
                "inner_val_citl": float(best["val_citl"]),
                "inner_val_slope": float(best["val_slope"]),
                "hold_auc": float(hold_auc),
                "hold_pr_auc": float(hold_pr),
                "hold_brier": float(hold_brier),
                "hold_citl": float(hold_citl),
                "hold_slope": float(hold_slope),
                "hold_n_test": int(hold_auc_n),
                "hold_n_pos": int(hold_n_pos),
                "hold_n_neg": int(hold_n_neg),
                "hold_prevalence": float(test_y.mean()),
            }
        )

        for rec in split_candidates:
            candidate_rows.append(rec)

        prediction_rows.extend(
            {
                "test_site": split["test_site"],
                "train_sites": "|".join(split["train_sites"]),
                "feature_set": best["feature_set"],
                "feature_cols": ", ".join(best_cols),
                "row_index": int(index),
                "id": int(row_id) if str(row_id).isdigit() else row_id,
                "y_true": int(y_true),
                "proba_death": float(prob),
                "proba_survival": float(1.0 - prob),
            }
            for index, (row_id, y_true, prob) in enumerate(
                zip(split["test_index"], test_y.to_numpy(), hold_prob, strict=False),
                start=0
            )
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    candidate_df = pd.DataFrame(candidate_rows).sort_values(["test_site", "inner_objective"], ignore_index=True)
    per_site_df = pd.DataFrame(per_site_rows).sort_values("test_site", ignore_index=True)
    pred_df = pd.DataFrame(prediction_rows).sort_values(["test_site", "row_index"], ignore_index=True)

    candidate_df.to_csv(args.output_dir / "candidate_metrics_inner_validation.csv", index=False)
    per_site_df.to_csv(args.output_dir / "feature_set_selection_by_site.csv", index=False)
    pred_df.to_csv(args.output_dir / "all_predictions.csv", index=False)

    summary = (
        candidate_df.groupby("feature_set", as_index=False)
        .agg(
            n_sites=("test_site", "nunique"),
            mean_inner_objective=("inner_objective", "mean"),
            mean_inner_val_auc=("val_auc", "mean"),
            mean_inner_val_pr_auc=("val_pr_auc", "mean"),
            mean_inner_val_brier=("val_brier", "mean"),
            mean_inner_val_abs_citl=("val_abs_citl", "mean"),
            mean_inner_val_abs_slope_dev=("val_abs_slope_dev", "mean"),
        )
        .sort_values("mean_inner_objective")
    )
    summary.to_csv(args.output_dir / "feature_set_summary_global_scores.csv", index=False)

    report = {
        "data_path": str(args.data),
        "output_dir": str(args.output_dir),
        "target": args.target,
        "facility_column": args.facility_col,
        "id_column": args.id_col,
        "seed": int(args.seed),
        "facility_levels": facility_levels,
        "feature_set_count": int(len(feature_sets)),
        "feature_set_definitions": [
            {"name": fs["name"], "feature_cols": fs["feature_cols"]} for fs in feature_sets
        ],
        "inner_val_size": args.inner_val_size,
        "weights": {"brier": args.weight_brier, "citl": args.weight_citl, "slope": args.weight_slope},
        "best_feature_set_by_site": per_site_df.to_dict(orient="records"),
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Saved {args.output_dir / 'feature_set_selection_by_site.csv'}")
    print(f"Saved {args.output_dir / 'candidate_metrics_inner_validation.csv'}")
    print(f"Saved {args.output_dir / 'feature_set_summary_global_scores.csv'}")
    print(f"Saved {args.output_dir / 'all_predictions.csv'}")
    print(f"Saved {args.output_dir / 'report.json'}")
    print(per_site_df.to_string(index=False))


if __name__ == "__main__":
    main()
