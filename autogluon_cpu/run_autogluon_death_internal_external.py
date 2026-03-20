from __future__ import annotations

import argparse
import importlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from autogluon.tabular import TabularPredictor

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from internal_external_validation_common import (  # noqa: E402
    DEFAULT_FACILITY,
    DEFAULT_ID,
    DEFAULT_TARGET,
    auc_with_hanley_mcneil_variance,
    iter_site_splits,
    pool_auc_hksj,
    prepare_site_holdout_data,
)


DATA_PATH = Path("/Users/ac/comforter/NewModelV3.csv")
OUTPUT_DIR = Path("/Users/ac/comforter/artifacts/autogluon_death_internal_external")
REPORT_PATH = Path("/Users/ac/comforter/artifacts/autogluon_death/report.json")
SEED = 42

AUTOGLUON_HYPERPARAMETERS = {
    "GBM": {},
    "CAT": {},
    "XGB": {},
    "RF": {},
    "XT": {},
    "KNN": {},
    "LR": {},
}


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_hyperparameters() -> tuple[dict[str, Any], str]:
    report = _load_json(REPORT_PATH)
    hyper = report.get("metadata", {}).get("autogluon_hyperparameters") if report else None
    if isinstance(hyper, dict):
        return hyper, str(REPORT_PATH)
    return AUTOGLUON_HYPERPARAMETERS, "fallback_defaults"


def _filter_available_hyperparameters(hyperparameters: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    dependency_by_family = {
        "GBM": "lightgbm",
        "XGB": "xgboost",
        "CAT": "catboost",
    }
    filtered: dict[str, Any] = {}
    skipped: dict[str, str] = {}

    for family, params in hyperparameters.items():
        dependency = dependency_by_family.get(family)
        if dependency is None:
            filtered[family] = params
            continue

        try:
            importlib.import_module(dependency)
            filtered[family] = params
        except Exception as exc:
            skipped[family] = f"{type(exc).__name__}: {exc}"

    return filtered, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Facility-holdout internal-external validation for the AutoGluon DeathHospDisch model.")
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--target", type=str, default=DEFAULT_TARGET)
    parser.add_argument("--facility-col", type=str, default=DEFAULT_FACILITY)
    parser.add_argument("--id-col", type=str, default=DEFAULT_ID)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--time-limit", type=int, default=900)
    parser.add_argument("--presets", type=str, default="good_quality")
    parser.add_argument("--num-bag-folds", type=int, default=5)
    parser.add_argument("--num-stack-levels", type=int, default=1)
    parser.add_argument("--verbosity", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_df, feature_cols, facility_levels = prepare_site_holdout_data(
        args.data,
        target_col=args.target,
        facility_col=args.facility_col,
        id_col=args.id_col,
    )
    site_splits = iter_site_splits(
        model_df,
        feature_cols,
        target_col=args.target,
        facility_col=args.facility_col,
        id_col=args.id_col,
    )
    hyperparameters, config_source = _load_hyperparameters()
    hyperparameters, skipped_families = _filter_available_hyperparameters(hyperparameters)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    leaderboard_paths: dict[str, str] = {}

    for split in site_splits:
        fold_dir = args.output_dir / split["test_site"].lower()
        predictor_dir = fold_dir / "predictor"
        if predictor_dir.exists():
            shutil.rmtree(predictor_dir)
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_df = split["train_X"].copy()
        train_df[args.target] = split["train_y"].to_numpy()
        test_df = split["test_X"].copy()
        test_df[args.target] = split["test_y"].to_numpy()

        predictor = TabularPredictor(
            label=args.target,
            problem_type="binary",
            eval_metric="roc_auc",
            path=str(predictor_dir),
        ).fit(
            train_data=train_df,
            presets=args.presets,
            hyperparameters=hyperparameters,
            num_bag_folds=args.num_bag_folds,
            num_stack_levels=args.num_stack_levels,
            refit_full=True,
            time_limit=args.time_limit,
            verbosity=args.verbosity,
        )

        leaderboard = predictor.leaderboard(test_df, silent=True)
        leaderboard_path = fold_dir / "leaderboard_test.csv"
        leaderboard.to_csv(leaderboard_path, index=False)
        leaderboard_paths[split["test_site"]] = str(leaderboard_path)

        proba = predictor.predict_proba(test_df, as_pandas=True)
        if 1 in proba.columns:
            prob_pos = proba[1].to_numpy(dtype=float)
            pos_label = 1
        elif "1" in proba.columns:
            prob_pos = proba["1"].to_numpy(dtype=float)
            pos_label = "1"
        else:
            prob_pos = proba.iloc[:, -1].to_numpy(dtype=float)
            pos_label = str(proba.columns[-1])

        auc_stats = auc_with_hanley_mcneil_variance(split["test_y"].to_numpy(), prob_pos)
        fold_rows.append(
            {
                "model": "autogluon",
                "test_site": split["test_site"],
                "train_sites": "|".join(split["train_sites"]),
                "auc": float(auc_stats["auc"]),
                "auc_variance": float(auc_stats["auc_variance"]),
                "auc_se": float(auc_stats["auc_se"]),
                "n_test": int(auc_stats["n_test"]),
                "n_pos": int(auc_stats["n_pos"]),
                "n_neg": int(auc_stats["n_neg"]),
                "prevalence": float(split["test_y"].mean()),
                "best_model": predictor.model_best,
                "pos_proba_column": str(pos_label),
            }
        )

        prediction_rows.extend(
            {
                "model": "autogluon",
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
                prob_pos,
                strict=False,
            )
        )

    fold_df = pd.DataFrame(fold_rows).sort_values("test_site").reset_index(drop=True)
    prediction_df = pd.DataFrame(prediction_rows).sort_values(["test_site", "row_index"]).reset_index(drop=True)
    pooled = pool_auc_hksj(fold_df)
    summary_df = pd.DataFrame(
        [
            {
                "model": "autogluon",
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
            "seed": int(args.seed),
            "feature_columns": feature_cols,
            "facility_levels": facility_levels,
            "hyperparameters_source": config_source,
            "autogluon_hyperparameters": hyperparameters,
            "skipped_model_families": skipped_families,
            "fit_settings": {
                "presets": args.presets,
                "num_bag_folds": int(args.num_bag_folds),
                "num_stack_levels": int(args.num_stack_levels),
                "refit_full": True,
                "time_limit": int(args.time_limit),
                "verbosity": int(args.verbosity),
            },
            "leaderboard_paths_by_site": leaderboard_paths,
        },
        "model": {
            "name": "autogluon",
            "fold_auc": fold_df.to_dict(orient="records"),
            "pooled_auc_random_effects": pooled,
        },
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
