from __future__ import annotations

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


DATA_PATH = Path("/Users/ac/comforter/NewComfModelData.csv")
OUTPUT_DIR = Path("/Users/ac/comforter/artifacts/tabicl_death_stage2")
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


def _metrics_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float | int]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "threshold": float(threshold),
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
    }


def _best_threshold_by_f1(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float | int]:
    thresholds = np.unique(np.round(y_prob, 6))
    best = None
    for t in thresholds:
        m = _metrics_at_threshold(y_true, y_prob, float(t))
        key = (m["f1"], m["recall"], m["precision"])
        if best is None or key > best[0]:
            best = (key, m)
    assert best is not None
    return best[1]


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
        X,
        y,
        test_size=0.2,
        random_state=SEED,
        stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.25,  # 0.25 of 0.8 => 0.2 total
        random_state=SEED,
        stratify=y_train_val,
    )

    base_params: dict[str, Any] = {
        "n_estimators": 8,
        "norm_methods": None,
        "feat_shuffle_method": "latin",
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

    configs: list[dict[str, Any]] = [
        {"name": "baseline", "updates": {}},
        {"name": "shift_only", "updates": {"feat_shuffle_method": "shift"}},
        {"name": "robust_only", "updates": {"norm_methods": ["none", "power", "robust"]}},
        {"name": "quantile_only", "updates": {"norm_methods": ["none", "power", "quantile"]}},
        {
            "name": "shift_plus_robust",
            "updates": {"feat_shuffle_method": "shift", "norm_methods": ["none", "power", "robust"]},
        },
        {
            "name": "shift_plus_quantile",
            "updates": {"feat_shuffle_method": "shift", "norm_methods": ["none", "power", "quantile"]},
        },
        {"name": "robust_n16", "updates": {"n_estimators": 16, "norm_methods": ["none", "power", "robust"]}},
        {
            "name": "shift_robust_n16",
            "updates": {
                "n_estimators": 16,
                "feat_shuffle_method": "shift",
                "norm_methods": ["none", "power", "robust"],
            },
        },
    ]

    rows = []
    for idx, conf in enumerate(configs, 1):
        params = {**base_params, **conf["updates"]}
        start = time.perf_counter()
        clf = TabICLClassifier(**params)
        clf.fit(X_train, y_train)
        val_prob = clf.predict_proba(X_val)[:, 1]
        test_prob = clf.predict_proba(X_test)[:, 1]
        elapsed = time.perf_counter() - start

        val_thr05 = _metrics_at_threshold(y_val.to_numpy(), val_prob, 0.5)
        test_thr05 = _metrics_at_threshold(y_test.to_numpy(), test_prob, 0.5)
        val_best = _best_threshold_by_f1(y_val.to_numpy(), val_prob)
        test_at_valbest = _metrics_at_threshold(y_test.to_numpy(), test_prob, float(val_best["threshold"]))

        row = {
            "experiment": conf["name"],
            "runtime_sec": round(elapsed, 3),
            "params": params,
            "val_roc_auc": float(roc_auc_score(y_val, val_prob)),
            "val_pr_auc": float(average_precision_score(y_val, val_prob)),
            "val_brier": float(brier_score_loss(y_val, val_prob)),
            "test_roc_auc": float(roc_auc_score(y_test, test_prob)),
            "test_pr_auc": float(average_precision_score(y_test, test_prob)),
            "test_brier": float(brier_score_loss(y_test, test_prob)),
            "val_metrics_at_0_5": val_thr05,
            "test_metrics_at_0_5": test_thr05,
            "val_best_f1_threshold": val_best,
            "test_metrics_at_val_best_f1_threshold": test_at_valbest,
        }
        rows.append(row)
        print(
            f"[{idx:02d}/{len(configs):02d}] {conf['name']}: "
            f"val_pr_auc={row['val_pr_auc']:.4f}, test_pr_auc={row['test_pr_auc']:.4f}, "
            f"val_best_thr={val_best['threshold']:.3f}, test_f1@valbest={test_at_valbest['f1']:.4f}, "
            f"runtime={elapsed:.1f}s"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "stage2_full.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    table = pd.DataFrame(
        [
            {
                "experiment": r["experiment"],
                "runtime_sec": r["runtime_sec"],
                "val_pr_auc": r["val_pr_auc"],
                "val_roc_auc": r["val_roc_auc"],
                "val_brier": r["val_brier"],
                "test_pr_auc": r["test_pr_auc"],
                "test_roc_auc": r["test_roc_auc"],
                "test_brier": r["test_brier"],
                "test_f1_at_0_5": r["test_metrics_at_0_5"]["f1"],
                "test_recall_at_0_5": r["test_metrics_at_0_5"]["recall"],
                "test_precision_at_0_5": r["test_metrics_at_0_5"]["precision"],
                "val_best_f1_threshold": r["val_best_f1_threshold"]["threshold"],
                "test_f1_at_val_best_thr": r["test_metrics_at_val_best_f1_threshold"]["f1"],
                "test_recall_at_val_best_thr": r["test_metrics_at_val_best_f1_threshold"]["recall"],
                "test_precision_at_val_best_thr": r["test_metrics_at_val_best_f1_threshold"]["precision"],
                "params_json": json.dumps(r["params"], sort_keys=True),
            }
            for r in rows
        ]
    )
    leaderboard = table.sort_values(
        by=["val_pr_auc", "val_roc_auc", "val_brier"],
        ascending=[False, False, True],
    )
    leaderboard.to_csv(OUTPUT_DIR / "stage2_leaderboard.csv", index=False)

    best_name = leaderboard.iloc[0]["experiment"]
    best_row = next(r for r in rows if r["experiment"] == best_name)
    with (OUTPUT_DIR / "stage2_best.json").open("w", encoding="utf-8") as f:
        json.dump(best_row, f, indent=2)

    meta = {
        "data_path": str(DATA_PATH),
        "target": target_col,
        "excluded_matched": excluded_cols,
        "predictor_count": len(feature_cols),
        "split_counts": {
            "train": int(len(X_train)),
            "val": int(len(X_val)),
            "test": int(len(X_test)),
        },
        "n_configs": len(configs),
        "selected_best_by": ["val_pr_auc", "val_roc_auc", "val_brier"],
        "best_experiment": best_name,
    }
    with (OUTPUT_DIR / "stage2_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved: {OUTPUT_DIR / 'stage2_leaderboard.csv'}")
    print(f"Saved: {OUTPUT_DIR / 'stage2_best.json'}")
    print(f"Best experiment: {best_name}")


if __name__ == "__main__":
    main()
