from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.linear_model import LogisticRegression

from internal_external_validation_common import (
    DEFAULT_FACILITY,
    DEFAULT_ID,
    DEFAULT_TARGET,
    auc_with_hanley_mcneil_variance,
    iter_site_splits,
    pool_auc_hksj,
    prepare_site_holdout_data,
)

DATA_PATH = Path("/Users/ac/comforter/MergedData.csv")
OUTPUT_DIR = Path("/Users/ac/comforter/artifacts/internal_external_validation_death_news_logistic")
MODEL_NAME = "news_logistic"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Facility-holdout internal-external validation for a NEWS-only logistic regression model."
    )
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--target", type=str, default=DEFAULT_TARGET)
    parser.add_argument("--facility-col", type=str, default=DEFAULT_FACILITY)
    parser.add_argument("--id-col", type=str, default=DEFAULT_ID)
    parser.add_argument("--news-col", type=str, default="NEWS")
    parser.add_argument("--max-iter", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_df, _, facility_levels = prepare_site_holdout_data(
        args.data,
        target_col=args.target,
        facility_col=args.facility_col,
        id_col=args.id_col,
    )

    if args.news_col not in model_df.columns:
        raise ValueError(f"Missing NEWS predictor column: {args.news_col}")

    model_df = model_df.dropna(subset=[args.news_col]).copy()
    feature_cols = [args.news_col]
    site_splits = iter_site_splits(
        model_df,
        feature_cols,
        target_col=args.target,
        facility_col=args.facility_col,
        id_col=args.id_col,
    )

    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []

    for split in site_splits:
        train_x = split["train_X"][[args.news_col]].astype(float)
        train_y = split["train_y"].astype(int)
        test_x = split["test_X"][[args.news_col]].astype(float)
        test_y = split["test_y"].astype(int)

        model = LogisticRegression(solver="lbfgs", max_iter=args.max_iter)
        model.fit(train_x, train_y)
        y_prob = model.predict_proba(test_x)[:, 1]

        auc_stats = auc_with_hanley_mcneil_variance(test_y.to_numpy(), y_prob)
        fold_rows.append(
            {
                "model": MODEL_NAME,
                "test_site": split["test_site"],
                "train_sites": "|".join(split["train_sites"]),
                "auc": float(auc_stats["auc"]),
                "auc_variance": float(auc_stats["auc_variance"]),
                "auc_se": float(auc_stats["auc_se"]),
                "n_test": int(auc_stats["n_test"]),
                "n_pos": int(auc_stats["n_pos"]),
                "n_neg": int(auc_stats["n_neg"]),
                "prevalence": float(test_y.mean()),
            }
        )

        prediction_rows.extend(
            {
                "model": MODEL_NAME,
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
                test_y.to_numpy(),
                y_prob,
                strict=False,
            )
        )

    fold_df = pd.DataFrame(fold_rows).sort_values(["model", "test_site"]).reset_index(drop=True)
    prediction_df = pd.DataFrame(prediction_rows).sort_values(["model", "test_site", "row_index"]).reset_index(drop=True)
    pooled = pool_auc_hksj(fold_df)

    summary_df = pd.DataFrame(
        [
            {
                "model": MODEL_NAME,
                "pooled_auc": pooled["pooled_auc"],
                "ci_95_lower": pooled["pooled_auc_ci_95"]["lower"],
                "ci_95_upper": pooled["pooled_auc_ci_95"]["upper"],
                "pi_95_lower": pooled["pooled_auc_prediction_interval_95"]["lower"],
                "pi_95_upper": pooled["pooled_auc_prediction_interval_95"]["upper"],
                "tau2_logit_auc": pooled["tau2_logit_auc"],
                "i2_percent": pooled["i2_percent"],
            }
        ]
    )

    report = {
        "metadata": {
            "data_path": str(args.data),
            "output_dir": str(args.output_dir),
            "target": args.target,
            "facility_column": args.facility_col,
            "id_column": args.id_col,
            "feature_columns": feature_cols,
            "facility_levels": facility_levels,
            "models": [MODEL_NAME],
            "model_spec": {
                "type": "logistic_regression",
                "predictors": feature_cols,
                "solver": "lbfgs",
                "max_iter": int(args.max_iter),
            },
        },
        "models": {
            MODEL_NAME: {
                "config_source": "run_internal_external_validation_news_logistic.py defaults/CLI",
                "fold_auc": fold_df.to_dict(orient="records"),
                "pooled_auc_random_effects": pooled,
            }
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
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
