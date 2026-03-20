from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from internal_external_validation_common import (
    DEFAULT_FACILITY,
    DEFAULT_ID,
    DEFAULT_TARGET,
    auc_with_hanley_mcneil_variance,
    iter_site_splits,
    pool_auc_hksj,
    prepare_site_holdout_data,
)

DATA_PATH = Path("/Users/ac/comforter/ComforterDatasetVal.csv")
COEFFICIENT_PATH = Path("/Users/ac/comforter/remain-coefficients.csv")
OUTPUT_DIR = Path("/Users/ac/comforter/artifacts/internal_external_validation_remain_logistic")


def _normalize_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).replace("\ufeff", "").replace("\xa0", " ").strip()


def _normalize_category(value: Any) -> str:
    clean = _normalize_text(value).upper()
    return "".join(ch for ch in clean if ch.isalnum())


def _parse_coefficients(
    path: Path,
) -> tuple[float, dict[str, float], dict[str, dict[str, float]], list[str], dict[str, int]]:
    coeff_df = pd.read_csv(path, encoding="utf-8")
    coeff_df.columns = [_normalize_text(c) for c in coeff_df.columns]

    if "variable" not in coeff_df.columns or "coefficient" not in coeff_df.columns:
        raise ValueError(f"Coefficients file must include 'variable' and 'coefficient': {path}")

    intercept = 0.0
    numeric_coefficients: dict[str, float] = {}
    group_coefficients: dict[str, dict[str, float]] = {}
    missing_groups: list[str] = []
    unknown_rows = 0

    for _, row in coeff_df.iterrows():
        variable = _normalize_text(row["variable"])
        if not variable:
            continue

        coefficient_raw = _normalize_text(row["coefficient"])
        if coefficient_raw.upper() == "REFERENT":
            continue
        if variable.lower() == "intercept":
            intercept = float(coefficient_raw)
            continue
        if variable in {"MCAge", "MCAgeSqd"}:
            numeric_coefficients[variable] = float(coefficient_raw)
            continue

        if not coefficient_raw:
            unknown_rows += 1
            continue

        group = _normalize_text(row.get("group", ""))
        if not group:
            missing_groups.append(variable)
            continue

        group_key = group.lower()
        group_coefficients.setdefault(group_key, {})[_normalize_category(variable)] = float(coefficient_raw)

    return (
        intercept,
        numeric_coefficients,
        group_coefficients,
        missing_groups,
        {"dropped_rows": unknown_rows},
    )


def _validate_required_columns(df: pd.DataFrame) -> None:
    required_cols = [
        "MCAge",
        "MCAgeSqd",
        "Sex",
        "Comorbidity",
        "ARP",
        "RemainProfile",
        "HospAdmitPrev24h",
        "SurgPrev24h",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for remain coefficients model: {missing}")


def _fit_platt_calibrator(
    raw_prob: np.ndarray,
    y: np.ndarray,
    random_state: int,
) -> LogisticRegression | None:
    if raw_prob.size == 0 or len(np.unique(y)) < 2:
        return None
    x = np.clip(raw_prob, 1e-6, 1.0 - 1e-6).reshape(-1, 1)
    model = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=random_state)
    model.fit(x, y.astype(int))
    return model


def _apply_platt_calibration(raw_prob: np.ndarray, calibrator: LogisticRegression | None) -> np.ndarray:
    if calibrator is None:
        return raw_prob
    x = np.clip(raw_prob, 1e-6, 1.0 - 1e-6).reshape(-1, 1)
    return calibrator.predict_proba(x)[:, 1]


def _predict_with_coefficients(
    df: pd.DataFrame,
    intercept: float,
    numeric_coefficients: dict[str, float],
    group_coefficients: dict[str, dict[str, float]],
) -> pd.Series:
    _validate_required_columns(df)

    score = np.full(len(df), float(intercept), dtype=float)
    score += df["MCAge"].astype(float).to_numpy() * numeric_coefficients.get("MCAge", 0.0)
    score += df["MCAgeSqd"].astype(float).to_numpy() * numeric_coefficients.get("MCAgeSqd", 0.0)

    for group, col in [
        ("Sex", "Sex"),
        ("Comorbidity", "Comorbidity"),
        ("ARP", "ARP"),
        ("RemainProfile", "RemainProfile"),
        ("HospAdmitPrev24h", "HospAdmitPrev24h"),
        ("SurgPrev24h", "SurgPrev24h"),
    ]:
        coef_map = group_coefficients.get(group.lower(), {})
        if not coef_map:
            continue
        normalized = df[col].map(_normalize_category)
        score += normalized.map(coef_map).fillna(0.0).to_numpy(dtype=float)

    return pd.Series(expit(score), index=df.index, name="proba_death", dtype=float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Facility-holdout internal-external validation for remain-coefficients logistic model."
    )
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--coefficients", type=Path, default=COEFFICIENT_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--target", type=str, default=DEFAULT_TARGET)
    parser.add_argument("--facility-col", type=str, default=DEFAULT_FACILITY)
    parser.add_argument("--id-col", type=str, default=DEFAULT_ID)
    parser.add_argument(
        "--recalibrate",
        action="store_true",
        help="Fit a Platt recalibration model on a split of each non-held-out training split.",
    )
    parser.add_argument(
        "--calibration-frac",
        type=float,
        default=0.2,
        help="Fraction of non-held-out training data reserved for recalibration.",
    )
    parser.add_argument(
        "--calibration-random-state",
        type=int,
        default=42,
        help="Random state for held-out-site recalibration split.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not 0.0 < args.calibration_frac < 1.0:
        raise ValueError(f"calibration-frac must be in (0, 1): {args.calibration_frac}")

    model_df, feature_cols, facility_levels = prepare_site_holdout_data(
        args.data,
        target_col=args.target,
        facility_col=args.facility_col,
        id_col=args.id_col,
    )

    intercept, numeric_coefficients, group_coefficients, missing_groups, meta = _parse_coefficients(args.coefficients)
    _validate_required_columns(model_df)

    site_splits = iter_site_splits(
        model_df,
        feature_cols,
        target_col=args.target,
        facility_col=args.facility_col,
        id_col=args.id_col,
    )

    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []

    model_name = "remain_coeff_logistic"
    recalibrate = bool(args.recalibrate)

    model_fold_rows: list[dict[str, Any]] = []

    for split in site_splits:
        train_prob_raw = _predict_with_coefficients(
            split["train_X"],
            intercept,
            numeric_coefficients,
            group_coefficients,
        ).to_numpy()

        test_prob_raw = _predict_with_coefficients(
            split["test_X"],
            intercept,
            numeric_coefficients,
            group_coefficients,
        ).to_numpy()

        calibrator = None
        if recalibrate:
            train_y = split["train_y"].to_numpy()
            _, calib_prob_raw, _, calib_y = train_test_split(
                train_prob_raw,
                train_y,
                test_size=args.calibration_frac,
                random_state=args.calibration_random_state,
                stratify=train_y,
            )
            calibrator = _fit_platt_calibrator(calib_prob_raw, calib_y, args.calibration_random_state)

        y_prob = _apply_platt_calibration(test_prob_raw, calibrator)
        auc_stats = auc_with_hanley_mcneil_variance(split["test_y"].to_numpy(), y_prob)

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
        fold_rows.append(fold_row)
        model_fold_rows.append(fold_row)

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
                y_prob,
                strict=False,
            )
        )

    model_fold_df = pd.DataFrame(model_fold_rows).sort_values("test_site").reset_index(drop=True)
    pooled = pool_auc_hksj(model_fold_df)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    fold_df = pd.DataFrame(fold_rows).sort_values(["model", "test_site"]).reset_index(drop=True)
    prediction_df = pd.DataFrame(prediction_rows).sort_values(["model", "test_site", "row_index"]).reset_index(drop=True)
    summary_df = pd.DataFrame(
        [
            {
                "model": model_name,
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
            "coefficients_path": str(args.coefficients),
            "output_dir": str(args.output_dir),
            "target": args.target,
            "facility_column": args.facility_col,
            "id_column": args.id_col,
            "feature_columns": feature_cols,
            "facility_levels": facility_levels,
            "models": [model_name],
            "parse_warnings": {
                "missing_group_rows": missing_groups,
                "parse_meta": meta,
            },
            "recalibration": {
                "enabled": recalibrate,
                "method": "platt_logistic" if recalibrate else "none",
                "calibration_frac": float(args.calibration_frac),
                "calibration_random_state": int(args.calibration_random_state),
            },
            "coefficients_intercept": intercept,
            "coefficients_by_group": {
                group: dict(vals) for group, vals in sorted(group_coefficients.items())
            },
            "coefficients_numeric": numeric_coefficients,
        },
        "models": {
            model_name: {
                "config_source": f"Fixed coefficients from {args.coefficients}",
                "model_config": {
                    "intercept": intercept,
                    "numeric_coefficients": numeric_coefficients,
                    "group_coefficients": {
                        group: dict(vals) for group, vals in sorted(group_coefficients.items())
                    },
                    "target": args.target,
                },
                "fold_auc": model_fold_df.to_dict(orient="records"),
                "pooled_auc_random_effects": pooled,
            }
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
