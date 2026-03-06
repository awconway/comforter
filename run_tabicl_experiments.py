from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
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
        description="Run a small TabICL hyperparameter sweep on a fixed split."
    )
    parser.add_argument("--data", type=Path, default=Path("NewComfModelData.csv"))
    parser.add_argument("--target", type=str, default="DeathHospDisch")
    parser.add_argument("--exclude", nargs="*", default=DEFAULT_EXCLUDE)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--val-size", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--checkpoint-version",
        type=str,
        default="tabicl-classifier-v2-20260212.ckpt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/tabicl_death_experiments"),
    )
    return parser.parse_args()


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


def _choose_positive_class(classes: np.ndarray) -> Any:
    for candidate in [1, "1", True]:
        for klass in classes:
            if klass == candidate:
                return klass
    return classes[-1]


def _evaluate_split(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_prob_pos: np.ndarray,
) -> dict[str, float | int]:
    y_true_bin = y_true.astype(int).to_numpy()
    y_pred_bin = y_pred.astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1]).ravel()

    return {
        "n_rows": int(len(y_true_bin)),
        "accuracy": float(accuracy_score(y_true_bin, y_pred_bin)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_bin, y_pred_bin)),
        "precision": float(precision_score(y_true_bin, y_pred_bin, zero_division=0)),
        "recall": float(recall_score(y_true_bin, y_pred_bin, zero_division=0)),
        "f1": float(f1_score(y_true_bin, y_pred_bin, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true_bin, y_prob_pos)),
        "pr_auc": float(average_precision_score(y_true_bin, y_prob_pos)),
        "brier": float(brier_score_loss(y_true_bin, y_prob_pos)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def build_experiments() -> list[dict[str, Any]]:
    return [
        {"name": "baseline", "params": {}},
        {"name": "n_estimators_16", "params": {"n_estimators": 16}},
        {"name": "n_estimators_32", "params": {"n_estimators": 32}},
        {"name": "norm_none_power_robust", "params": {"norm_methods": ["none", "power", "robust"]}},
        {"name": "norm_none_power_quantile", "params": {"norm_methods": ["none", "power", "quantile"]}},
        {"name": "feat_shuffle_random", "params": {"feat_shuffle_method": "random"}},
        {"name": "feat_shuffle_shift", "params": {"feat_shuffle_method": "shift"}},
        {"name": "outlier_threshold_3", "params": {"outlier_threshold": 3.0}},
        {"name": "outlier_threshold_6", "params": {"outlier_threshold": 6.0}},
        {"name": "softmax_temp_0_7", "params": {"softmax_temperature": 0.7}},
        {"name": "softmax_temp_1_1", "params": {"softmax_temperature": 1.1}},
        {"name": "average_logits_false", "params": {"average_logits": False}},
    ]


def main() -> None:
    args = parse_args()
    if not (0 < args.test_size < 1 and 0 < args.val_size < 1):
        raise ValueError("test-size and val-size must be in (0, 1).")
    if args.test_size + args.val_size >= 1:
        raise ValueError("test-size + val-size must be < 1.")

    df = pd.read_csv(args.data)
    df.columns = [col.lstrip("\ufeff") for col in df.columns]
    target_col = _resolve_target_column(df.columns.tolist(), args.target)

    excluded_cols = _resolve_excluded_columns(df.columns.tolist(), args.exclude)
    drop_cols = set(excluded_cols + [target_col])
    feature_cols = [col for col in df.columns if col not in drop_cols]
    X = df[feature_cols]
    y = df[target_col]

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

    base_params = {
        "n_estimators": 8,
        "norm_methods": None,
        "feat_shuffle_method": "latin",
        "class_shuffle_method": "shift",
        "outlier_threshold": 4.0,
        "softmax_temperature": 0.9,
        "average_logits": True,
        "support_many_classes": True,
        "batch_size": 8,
        "checkpoint_version": args.checkpoint_version,
        "device": args.device,
        "random_state": args.seed,
        "verbose": False,
    }

    results: list[dict[str, Any]] = []
    experiments = build_experiments()

    for idx, exp in enumerate(experiments, start=1):
        run_params = {**base_params, **exp["params"]}
        start = time.perf_counter()
        row: dict[str, Any] = {
            "experiment": exp["name"],
            "index": idx,
            "status": "ok",
            "params": run_params,
        }
        try:
            clf = TabICLClassifier(**run_params)
            clf.fit(X_train, y_train)
            positive_class = _choose_positive_class(clf.classes_)
            pos_idx = {k: i for i, k in enumerate(clf.classes_)}[positive_class]

            val_pred = clf.predict(X_val)
            val_prob = clf.predict_proba(X_val)[:, pos_idx]
            test_pred = clf.predict(X_test)
            test_prob = clf.predict_proba(X_test)[:, pos_idx]

            val_metrics = _evaluate_split(y_val, val_pred, val_prob)
            test_metrics = _evaluate_split(y_test, test_pred, test_prob)
            elapsed = time.perf_counter() - start

            row.update(
                {
                    "runtime_sec": round(elapsed, 3),
                    "positive_class": int(positive_class),
                    "val": val_metrics,
                    "test": test_metrics,
                }
            )
            print(
                f"[{idx:02d}/{len(experiments):02d}] {exp['name']}: "
                f"val_pr_auc={val_metrics['pr_auc']:.4f}, val_roc_auc={val_metrics['roc_auc']:.4f}, "
                f"test_pr_auc={test_metrics['pr_auc']:.4f}, test_roc_auc={test_metrics['roc_auc']:.4f}, "
                f"runtime={elapsed:.1f}s"
            )
        except Exception as exc:  # noqa: BLE001
            row.update({"status": "failed", "error": str(exc)})
            print(f"[{idx:02d}/{len(experiments):02d}] {exp['name']}: failed ({exc})")
        results.append(row)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "experiments_full.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    ok_rows = [r for r in results if r["status"] == "ok"]
    if not ok_rows:
        raise RuntimeError("All experiments failed. See experiments_full.json.")

    flat_rows = []
    for r in ok_rows:
        flat_rows.append(
            {
                "experiment": r["experiment"],
                "runtime_sec": r["runtime_sec"],
                "val_pr_auc": r["val"]["pr_auc"],
                "val_roc_auc": r["val"]["roc_auc"],
                "val_brier": r["val"]["brier"],
                "val_recall": r["val"]["recall"],
                "val_precision": r["val"]["precision"],
                "val_f1": r["val"]["f1"],
                "test_pr_auc": r["test"]["pr_auc"],
                "test_roc_auc": r["test"]["roc_auc"],
                "test_brier": r["test"]["brier"],
                "test_recall": r["test"]["recall"],
                "test_precision": r["test"]["precision"],
                "test_f1": r["test"]["f1"],
                "params_json": json.dumps(r["params"], sort_keys=True),
            }
        )

    leaderboard = pd.DataFrame(flat_rows).sort_values(
        by=["val_pr_auc", "val_roc_auc", "val_brier"],
        ascending=[False, False, True],
    )
    leaderboard.to_csv(output_dir / "leaderboard.csv", index=False)

    best_name = leaderboard.iloc[0]["experiment"]
    best_full = next(r for r in ok_rows if r["experiment"] == best_name)
    with (output_dir / "best_experiment.json").open("w", encoding="utf-8") as f:
        json.dump(best_full, f, indent=2)

    metadata = {
        "data_path": str(args.data),
        "target": target_col,
        "excluded_requested": args.exclude,
        "excluded_matched": excluded_cols,
        "predictor_count": len(feature_cols),
        "split_counts": {
            "train": int(len(X_train)),
            "val": int(len(X_val)),
            "test": int(len(X_test)),
        },
        "n_experiments": len(experiments),
        "n_successful": len(ok_rows),
    }
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved: {output_dir / 'leaderboard.csv'}")
    print(f"Saved: {output_dir / 'best_experiment.json'}")
    print(f"Best experiment: {best_name}")


if __name__ == "__main__":
    main()
