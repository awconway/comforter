from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from tabicl import TabICLClassifier


DATA_PATH = Path("/Users/ac/comforter/NewComfModelData.csv")
OUTPUT_DIR = Path("/Users/ac/comforter/artifacts/tabicl_death_calibration")
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


def _metrics(y_true: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, prob)),
        "pr_auc": float(average_precision_score(y_true, prob)),
        "brier": float(brier_score_loss(y_true, prob)),
        "log_loss": float(log_loss(y_true, prob, labels=[0, 1])),
    }


def _ece_quantile(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> dict[str, float | int]:
    qbin = pd.qcut(pd.Series(prob), q=n_bins, labels=False, duplicates="drop")
    work = pd.DataFrame({"y_true": y_true, "prob": prob, "bin": qbin})
    agg = (
        work.groupby("bin", observed=True)
        .agg(n=("y_true", "size"), mean_pred=("prob", "mean"), obs_rate=("y_true", "mean"))
        .reset_index(drop=True)
    )
    gap = np.abs(agg["mean_pred"] - agg["obs_rate"])
    ece = float((agg["n"] * gap).sum() / agg["n"].sum())
    mce = float(gap.max())
    return {
        "ece_q10": ece,
        "mce_q10": mce,
        "effective_bins": int(len(agg)),
        "min_bin_count": int(agg["n"].min()),
        "max_bin_count": int(agg["n"].max()),
    }


def _plot_reliability(y_true: np.ndarray, prob_map: dict[str, np.ndarray], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1.2, label="Perfect calibration")
    for name, probs in prob_map.items():
        frac_pos, mean_pred = calibration_curve(y_true, probs, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", linewidth=1.8, label=name)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed probability")
    ax.set_title("Test Calibration Curves (10 quantile bins)")
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

    drop_cols = set(excluded_cols + [target_col])
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols]
    y = df[target_col].astype(int)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=SEED, stratify=y_train_val
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
    clf.fit(X_train, y_train)

    val_raw = clf.predict_proba(X_val)[:, 1]
    test_raw = clf.predict_proba(X_test)[:, 1]

    # Platt scaling on validation probabilities.
    platt = LogisticRegression(solver="lbfgs", max_iter=1000)
    platt.fit(val_raw.reshape(-1, 1), y_val.to_numpy())
    test_platt = platt.predict_proba(test_raw.reshape(-1, 1))[:, 1]

    # Isotonic regression on validation probabilities.
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(val_raw, y_val.to_numpy())
    test_iso = iso.predict(test_raw)

    y_test_np = y_test.to_numpy()
    y_val_np = y_val.to_numpy()

    report = {
        "metadata": {
            "target": target_col,
            "excluded_predictors": excluded_cols,
            "predictor_count": len(feature_cols),
            "split_counts": {
                "train": int(len(X_train)),
                "val": int(len(X_val)),
                "test": int(len(X_test)),
            },
            "prevalence": {
                "val": float(y_val_np.mean()),
                "test": float(y_test_np.mean()),
            },
        },
        "validation_metrics": {
            "raw": _metrics(y_val_np, val_raw),
        },
        "test_metrics": {
            "raw": _metrics(y_test_np, test_raw) | _ece_quantile(y_test_np, test_raw),
            "platt": _metrics(y_test_np, test_platt) | _ece_quantile(y_test_np, test_platt),
            "isotonic": _metrics(y_test_np, test_iso) | _ece_quantile(y_test_np, test_iso),
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "calibration_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    preds = pd.DataFrame(
        {
            "row_index": X_test.index,
            "y_true": y_test_np,
            "proba_raw": test_raw,
            "proba_platt": test_platt,
            "proba_isotonic": test_iso,
        }
    )
    preds.to_csv(OUTPUT_DIR / "test_calibrated_predictions.csv", index=False)

    _plot_reliability(
        y_test_np,
        {"Raw": test_raw, "Platt": test_platt, "Isotonic": test_iso},
        OUTPUT_DIR / "test_calibration_overlay.png",
    )

    print(f"Saved {OUTPUT_DIR / 'calibration_report.json'}")
    print(f"Saved {OUTPUT_DIR / 'test_calibrated_predictions.csv'}")
    print(f"Saved {OUTPUT_DIR / 'test_calibration_overlay.png'}")


if __name__ == "__main__":
    main()
