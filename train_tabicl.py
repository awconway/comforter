from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from tabicl import TabICLClassifier


DEFAULT_EXCLUDE = ["ID", "Mort30d", "ICUWithin48h", "METWithin48h"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/evaluate a TabICL classifier to predict DeathHospDisch."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("NewComfModelData.csv"),
        help="Input CSV path.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="DeathHospDisch",
        help="Target column name.",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=DEFAULT_EXCLUDE,
        help="Columns to exclude as predictors (case-insensitive).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Fraction of total rows used for test split.",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.20,
        help="Fraction of total rows used for validation split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic splits and estimator behavior.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=8,
        help="TabICL ensemble size.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional TabICL device override, e.g. cpu or cuda.",
    )
    parser.add_argument(
        "--checkpoint-version",
        type=str,
        default="tabicl-classifier-v2-20260212.ckpt",
        help="TabICL checkpoint filename on Hugging Face.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/tabicl"),
        help="Directory for metrics and predictions.",
    )
    return parser.parse_args()


def _resolve_target_column(columns: list[str], target: str) -> str:
    if target in columns:
        return target
    lower_map = {c.lower(): c for c in columns}
    if target.lower() in lower_map:
        return lower_map[target.lower()]
    raise ValueError(f"Target column '{target}' was not found in CSV columns.")


def _resolve_excluded_columns(columns: list[str], requested: list[str]) -> list[str]:
    requested_lower = {col.lower() for col in requested}
    return [col for col in columns if col.lower() in requested_lower]


def _choose_positive_class(classes: np.ndarray) -> Any:
    priorities = [1, "1", True, "true", "True", "yes", "YES", "positive", "Positive"]
    for candidate in priorities:
        for klass in classes:
            if klass == candidate:
                return klass
    return classes[-1]


def _serialize_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _binary_view(y: pd.Series, positive_class: Any) -> np.ndarray:
    return (y.to_numpy() == positive_class).astype(int)


def _evaluate_split(
    split_name: str,
    model: TabICLClassifier,
    X_split: pd.DataFrame,
    y_split: pd.Series,
    positive_class: Any,
    output_dir: Path,
) -> dict[str, Any]:
    classes = model.classes_
    y_pred = model.predict(X_split)
    proba = model.predict_proba(X_split)
    class_to_idx = {klass: idx for idx, klass in enumerate(classes)}
    pos_idx = class_to_idx[positive_class]
    proba_pos = proba[:, pos_idx]

    y_binary = _binary_view(y_split, positive_class)
    pred_binary = (y_pred == positive_class).astype(int)

    metrics: dict[str, Any] = {
        "n_rows": int(len(y_split)),
        "accuracy": float(accuracy_score(y_split, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_split, y_pred)),
        "f1_positive": float(f1_score(y_binary, pred_binary, zero_division=0)),
        "precision_positive": float(precision_score(y_binary, pred_binary, zero_division=0)),
        "recall_positive": float(recall_score(y_binary, pred_binary, zero_division=0)),
        "positive_class": _serialize_value(positive_class),
    }

    if len(classes) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_binary, proba_pos))
        metrics["pr_auc"] = float(average_precision_score(y_binary, proba_pos))

    cm = confusion_matrix(y_split, y_pred, labels=classes)
    metrics["confusion_matrix"] = {
        "labels": [_serialize_value(v) for v in classes.tolist()],
        "matrix": cm.tolist(),
    }

    pred_df = pd.DataFrame(
        {
            "row_index": X_split.index,
            "y_true": y_split.to_numpy(),
            "y_pred": y_pred,
            "proba_positive_class": proba_pos,
        }
    )
    for idx, klass in enumerate(classes):
        pred_df[f"proba_class_{klass}"] = proba[:, idx]
    pred_df.to_csv(output_dir / f"{split_name}_predictions.csv", index=False)

    return metrics


def main() -> None:
    args = parse_args()
    if not (0 < args.test_size < 1):
        raise ValueError("--test-size must be between 0 and 1.")
    if not (0 < args.val_size < 1):
        raise ValueError("--val-size must be between 0 and 1.")
    if args.test_size + args.val_size >= 1:
        raise ValueError("test-size + val-size must be < 1.")

    df = pd.read_csv(args.data)
    df.columns = [col.lstrip("\ufeff") for col in df.columns]

    target_col = _resolve_target_column(df.columns.tolist(), args.target)
    if df[target_col].isna().any():
        raise ValueError(f"Target column '{target_col}' contains missing values.")

    excluded_cols = _resolve_excluded_columns(df.columns.tolist(), args.exclude)
    drop_cols = set(excluded_cols + [target_col])
    feature_cols = [col for col in df.columns if col not in drop_cols]
    if not feature_cols:
        raise ValueError("No predictor columns left after exclusions.")

    X = df[feature_cols]
    y = df[target_col]
    class_counts = y.value_counts()
    if (class_counts < 2).any():
        raise ValueError(
            "At least one target class has fewer than 2 rows; stratified splitting is not possible."
        )

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )

    val_relative = args.val_size / (1.0 - args.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_relative,
        random_state=args.seed,
        stratify=y_train_val,
    )

    model = TabICLClassifier(
        n_estimators=args.n_estimators,
        random_state=args.seed,
        device=args.device,
        checkpoint_version=args.checkpoint_version,
    )
    model.fit(X_train, y_train)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    positive_class = _choose_positive_class(model.classes_)
    val_metrics = _evaluate_split("val", model, X_val, y_val, positive_class, output_dir)
    test_metrics = _evaluate_split("test", model, X_test, y_test, positive_class, output_dir)

    metadata = {
        "data_path": str(args.data),
        "target": target_col,
        "excluded_requested": args.exclude,
        "excluded_matched": excluded_cols,
        "predictor_count": len(feature_cols),
        "predictors": feature_cols,
        "split_fraction": {
            "train": float(1.0 - args.test_size - args.val_size),
            "val": float(args.val_size),
            "test": float(args.test_size),
        },
        "split_counts": {
            "train": int(len(X_train)),
            "val": int(len(X_val)),
            "test": int(len(X_test)),
        },
        "target_distribution": {
            "full": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
            "train": {str(k): int(v) for k, v in y_train.value_counts().sort_index().items()},
            "val": {str(k): int(v) for k, v in y_val.value_counts().sort_index().items()},
            "test": {str(k): int(v) for k, v in y_test.value_counts().sort_index().items()},
        },
        "model": {
            "name": "TabICLClassifier",
            "n_estimators": args.n_estimators,
            "checkpoint_version": args.checkpoint_version,
            "classes": [_serialize_value(v) for v in model.classes_.tolist()],
            "positive_class": _serialize_value(positive_class),
        },
    }

    metrics = {"metadata": metadata, "validation": val_metrics, "test": test_metrics}
    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved metrics: {metrics_path}")
    print(f"Saved validation predictions: {output_dir / 'val_predictions.csv'}")
    print(f"Saved test predictions: {output_dir / 'test_predictions.csv'}")


if __name__ == "__main__":
    main()
