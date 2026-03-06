from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.calibration import calibration_curve
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


DATA_PATH = Path("/Users/ac/comforter/NewComfModelData.csv")
OUTPUT_DIR = Path("/Users/ac/comforter/artifacts/autogluon_death")
TARGET = "DeathHospDisch"
EXCLUDE = [
    "Id",
    "Cardiovascular",
    "LungDisease",
    "CKD",
    "LiverDisease",
    "Diabetes",
    "Malignancy",
    "CognitionMentalHealth",
    "FunctionalDependency",
    "NutritionMetabolism",
    "SensoryImpairment",
    "Mort30d",
    "ICUWithin48h",
    "METWithin48h",
]
SEED = 42


def _resolve_target_column(columns: list[str], target: str) -> str:
    if target in columns:
        return target
    lower_map = {c.lower(): c for c in columns}
    if target.lower() in lower_map:
        return lower_map[target.lower()]
    raise ValueError(f"Target column '{target}' not found in data.")


def _resolve_excluded_columns(columns: list[str], requested: list[str]) -> list[str]:
    requested_lower = {x.lower() for x in requested}
    return [col for col in columns if col.lower() in requested_lower]


def _ece_quantile(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> dict[str, float | int]:
    qbin = pd.qcut(pd.Series(prob), q=n_bins, labels=False, duplicates="drop")
    work = pd.DataFrame({"y_true": y_true, "prob": prob, "bin": qbin})
    agg = (
        work.groupby("bin", observed=True)
        .agg(n=("y_true", "size"), mean_pred=("prob", "mean"), obs_rate=("y_true", "mean"))
        .reset_index(drop=True)
    )
    gap = np.abs(agg["mean_pred"] - agg["obs_rate"])
    return {
        "ece_q10": float((agg["n"] * gap).sum() / agg["n"].sum()),
        "mce_q10": float(gap.max()),
        "effective_bins": int(len(agg)),
        "min_bin_count": int(agg["n"].min()),
        "max_bin_count": int(agg["n"].max()),
    }


def _plot_calibration(y_true: np.ndarray, prob: np.ndarray, out_path: Path) -> None:
    frac_pos, mean_pred = calibration_curve(y_true, prob, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1.2, label="Perfect calibration")
    ax.plot(mean_pred, frac_pos, marker="o", linewidth=1.8, label="AutoGluon (test)")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed probability")
    ax.set_title("AutoGluon Test Calibration Curve (10 quantile bins)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.lstrip("\ufeff") for c in df.columns]
    target_col = _resolve_target_column(df.columns.tolist(), TARGET)
    excluded_cols = _resolve_excluded_columns(df.columns.tolist(), EXCLUDE)
    feature_cols = [c for c in df.columns if c not in set(excluded_cols + [target_col])]

    model_df = df[feature_cols + [target_col]].copy()

    train_df, test_df = train_test_split(
        model_df,
        test_size=0.2,
        random_state=SEED,
        stratify=model_df[target_col],
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ag_path = OUTPUT_DIR / "predictor"

    # Explicitly exclude deep learning families by only listing classical tabular models.
    hyperparameters = {
        "GBM": {},
        "CAT": {},
        "XGB": {},
        "RF": {},
        "XT": {},
        "KNN": {},
        "LR": {},
    }

    predictor = TabularPredictor(
        label=target_col,
        problem_type="binary",
        eval_metric="roc_auc",
        path=str(ag_path),
    ).fit(
        train_data=train_df,
        presets="good_quality",
        hyperparameters=hyperparameters,
        num_bag_folds=5,
        num_stack_levels=1,
        refit_full=True,
        time_limit=900,
        verbosity=2,
    )

    leaderboard = predictor.leaderboard(test_df, silent=True)
    leaderboard.to_csv(OUTPUT_DIR / "leaderboard_test.csv", index=False)

    proba = predictor.predict_proba(test_df, as_pandas=True)
    # Binary-class probability column can be named 1 or True depending on encoder.
    if 1 in proba.columns:
        prob_pos = proba[1].to_numpy(dtype=float)
        pos_label = 1
    elif "1" in proba.columns:
        prob_pos = proba["1"].to_numpy(dtype=float)
        pos_label = "1"
    else:
        prob_pos = proba.iloc[:, -1].to_numpy(dtype=float)
        pos_label = str(proba.columns[-1])

    y_true = test_df[target_col].astype(int).to_numpy()
    y_pred = (prob_pos >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    spec = tn / (tn + fp)

    report = {
        "metadata": {
            "data_path": str(DATA_PATH),
            "target": target_col,
            "excluded_predictors": excluded_cols,
            "predictor_count": len(feature_cols),
            "split_counts": {"train": int(len(train_df)), "test": int(len(test_df))},
            "seed": SEED,
            "autogluon_hyperparameters": hyperparameters,
            "pos_proba_column": str(pos_label),
            "best_model": predictor.model_best,
            "prevalence": {
                "full": float(model_df[target_col].mean()),
                "train": float(train_df[target_col].mean()),
                "test": float(test_df[target_col].mean()),
            },
        },
        "test_metrics_threshold_0_5": {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "specificity": float(spec),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "test_probability_metrics": {
            "roc_auc": float(roc_auc_score(y_true, prob_pos)),
            "pr_auc": float(average_precision_score(y_true, prob_pos)),
            "brier": float(brier_score_loss(y_true, prob_pos)),
            "log_loss": float(log_loss(y_true, prob_pos, labels=[0, 1])),
        }
        | _ece_quantile(y_true, prob_pos),
    }

    (OUTPUT_DIR / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    preds = pd.DataFrame(
        {
            "row_index": test_df.index,
            "y_true": y_true,
            "y_pred_threshold_0_5": y_pred,
            "proba_death": prob_pos,
            "proba_survival": 1.0 - prob_pos,
        }
    )
    preds.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

    _plot_calibration(y_true, prob_pos, OUTPUT_DIR / "test_calibration_curve.png")

    print(f"Saved {OUTPUT_DIR / 'report.json'}")
    print(f"Saved {OUTPUT_DIR / 'leaderboard_test.csv'}")
    print(f"Saved {OUTPUT_DIR / 'test_predictions.csv'}")
    print(f"Saved {OUTPUT_DIR / 'test_calibration_curve.png'}")
    print(f"Best model: {predictor.model_best}")


if __name__ == "__main__":
    main()
