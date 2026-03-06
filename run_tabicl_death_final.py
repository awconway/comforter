from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
from tabicl import TabICLClassifier


DATA_PATH = Path("/Users/ac/comforter/NewComfModelData.csv")
OUTPUT_DIR = Path("/Users/ac/comforter/artifacts/tabicl_death_final")
TARGET = "DeathHospDisch"
EXCLUDE = ["ID", "Mort30d", "ICUWithin48h", "METWithin48h"]
SEED = 42


def _resolve_target_column(columns: list[str], target: str) -> str:
    if target in columns:
        return target
    lower_map = {c.lower(): c for c in columns}
    if target.lower() in lower_map:
        return lower_map[target.lower()]
    raise ValueError(f"Target column '{target}' not found.")


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
    ax.plot(mean_pred, frac_pos, marker="o", linewidth=1.8, label="Final model (test)")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed probability")
    ax.set_title("Final Test Calibration Curve (10 quantile bins)")
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

    X = df[feature_cols]
    y = df[target_col].astype(int)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=SEED,
        stratify=y,
    )

    clf = TabICLClassifier(
        n_estimators=8,
        norm_methods=None,
        feat_shuffle_method="shift",
        class_shuffle_method="shift",
        outlier_threshold=4.0,
        softmax_temperature=0.9,
        average_logits=True,
        support_many_classes=True,
        batch_size=8,
        checkpoint_version="tabicl-classifier-v2-20260212.ckpt",
        device="cpu",
        random_state=SEED,
        verbose=False,
    )
    clf.fit(X_train_val, y_train_val)

    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    spec = tn / (tn + fp)

    report = {
        "metadata": {
            "data_path": str(DATA_PATH),
            "target": target_col,
            "excluded_predictors": excluded_cols,
            "predictor_count": len(feature_cols),
            "predictors": feature_cols,
            "split": {"train_val": int(len(X_train_val)), "test": int(len(X_test))},
            "seed": SEED,
            "model_config": {
                "n_estimators": 8,
                "norm_methods": None,
                "feat_shuffle_method": "shift",
                "class_shuffle_method": "shift",
                "outlier_threshold": 4.0,
                "softmax_temperature": 0.9,
                "average_logits": True,
                "checkpoint_version": "tabicl-classifier-v2-20260212.ckpt",
                "device": "cpu",
            },
            "prevalence": {
                "full": float(y.mean()),
                "train_val": float(y_train_val.mean()),
                "test": float(y_test.mean()),
            },
        },
        "test_metrics_threshold_0_5": {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "specificity": float(spec),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "test_probability_metrics": {
            "roc_auc": float(roc_auc_score(y_test, y_prob)),
            "pr_auc": float(average_precision_score(y_test, y_prob)),
            "brier": float(brier_score_loss(y_test, y_prob)),
            "log_loss": float(log_loss(y_test, y_prob, labels=[0, 1])),
        }
        | _ece_quantile(y_test.to_numpy(), y_prob),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "final_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    preds = pd.DataFrame(
        {
            "row_index": X_test.index,
            "y_true": y_test.to_numpy(),
            "y_pred_threshold_0_5": y_pred,
            "proba_death": y_prob,
            "proba_survival": 1.0 - y_prob,
        }
    )
    preds.to_csv(OUTPUT_DIR / "test_predictions_final.csv", index=False)

    _plot_calibration(y_test.to_numpy(), y_prob, OUTPUT_DIR / "test_calibration_curve_final.png")

    print(f"Saved {OUTPUT_DIR / 'final_report.json'}")
    print(f"Saved {OUTPUT_DIR / 'test_predictions_final.csv'}")
    print(f"Saved {OUTPUT_DIR / 'test_calibration_curve_final.png'}")


if __name__ == "__main__":
    main()
