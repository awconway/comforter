from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from tabicl import TabICLClassifier


DATA_PATH = Path("/Users/ac/comforter/NewComfModelData.csv")
OUTPUT_DIR = Path("/Users/ac/comforter/artifacts/tabicl_death_feature_engineering")
TARGET = "DeathHospDisch"
SEED = 42

BASE_FEATURES = [
    "Age",
    "Sex",
    "ARP",
    "AdmitPrev24h",
    "SurgPrev24h",
    "ICUDischPrev24h",
    "METWithinPrev24h",
    "NEWS",
    "ICD",
    "SOFA",
]

FEATURE_SETS = [
    {"name": "raw_core", "features": []},
    {
        "name": "raw_plus_instability_summary",
        "features": [
            "METWithinPrev24hCount",
            "AnyMETWithinPrev24h",
            "TwoPlusMETWithinPrev24h",
            "AcuteInstabilityCount",
            "AcuteInstabilityWeightedCount",
            "AnyRecentInstability",
            "MultiRecentInstability",
        ],
    },
    {"name": "raw_plus_acute_instability_count", "features": ["AcuteInstabilityCount"]},
    {"name": "raw_plus_acute_instability_weighted_count", "features": ["AcuteInstabilityWeightedCount"]},
    {"name": "raw_plus_high_acuity", "features": ["HighAcuity"]},
    {"name": "raw_plus_instability_count_high_acuity", "features": ["AcuteInstabilityCount", "HighAcuity"]},
    {
        "name": "raw_plus_summary_flags",
        "features": [
            "METWithinPrev24hCount",
            "AnyMETWithinPrev24h",
            "TwoPlusMETWithinPrev24h",
            "AcuteInstabilityCount",
            "AcuteInstabilityWeightedCount",
            "AnyRecentInstability",
            "MultiRecentInstability",
            "AgeGE75",
            "AgeGE85",
            "NEWSGE5",
            "NEWSGE7",
            "NEWSGE10",
            "SOFAGE2",
            "SOFAGE4",
            "SOFAGE6",
            "HighAcuity",
        ],
    },
]

TABICL_PARAMS: dict[str, Any] = {
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


def _resolve_target_column(columns: list[str], target: str) -> str:
    if target in columns:
        return target
    lower_map = {c.lower(): c for c in columns}
    if target.lower() in lower_map:
        return lower_map[target.lower()]
    raise ValueError(f"Target column '{target}' not found.")


def _to_met_count(series: pd.Series) -> pd.Series:
    normalized = series.fillna("nil").astype(str).str.strip().str.lower()
    mapped = normalized.map(
        {
            "nil": 0,
            "none": 0,
            "0": 0,
            "one": 1,
            "1": 1,
            "twoplus": 2,
            "two_plus": 2,
            "two plus": 2,
            "2": 2,
            "2+": 2,
        }
    )
    if mapped.isna().any():
        unknown = sorted(normalized[mapped.isna()].unique().tolist())
        raise ValueError(f"Unsupported METWithinPrev24h values: {unknown}")
    return mapped.astype(int)


class FeatureEngineer:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.continuous_means: dict[str, float] = {}

    def fit(self, df: pd.DataFrame) -> "FeatureEngineer":
        self.continuous_means = {
            col: float(df[col].astype(float).mean())
            for col in ["Age", "NEWS", "ICD", "SOFA"]
        }
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df[BASE_FEATURES].copy()
        selected = set(self.config["features"])
        met_count = _to_met_count(out["METWithinPrev24h"])

        any_met = (met_count >= 1).astype(int)
        two_plus_met = (met_count >= 2).astype(int)
        instability_count = (
            out["AdmitPrev24h"].astype(int)
            + out["SurgPrev24h"].astype(int)
            + out["ICUDischPrev24h"].astype(int)
            + any_met
        )
        instability_weighted_count = (
            out["AdmitPrev24h"].astype(int)
            + out["SurgPrev24h"].astype(int)
            + out["ICUDischPrev24h"].astype(int)
            + met_count
        )
        high_acuity = ((out["NEWS"] >= 7) | (out["SOFA"] >= 4)).astype(int)

        if "AcuteInstabilityCount" in selected:
            out["AcuteInstabilityCount"] = instability_count
        if "AcuteInstabilityWeightedCount" in selected:
            out["AcuteInstabilityWeightedCount"] = instability_weighted_count
        if "METWithinPrev24hCount" in selected:
            out["METWithinPrev24hCount"] = met_count
        if "AnyMETWithinPrev24h" in selected:
            out["AnyMETWithinPrev24h"] = any_met
        if "TwoPlusMETWithinPrev24h" in selected:
            out["TwoPlusMETWithinPrev24h"] = two_plus_met
        if "AnyRecentInstability" in selected:
            out["AnyRecentInstability"] = (instability_count >= 1).astype(int)
        if "MultiRecentInstability" in selected:
            out["MultiRecentInstability"] = (instability_count >= 2).astype(int)
        if "HighAcuity" in selected:
            out["HighAcuity"] = high_acuity
        if "AgeGE75" in selected:
            out["AgeGE75"] = (out["Age"] >= 75).astype(int)
        if "AgeGE85" in selected:
            out["AgeGE85"] = (out["Age"] >= 85).astype(int)
        if "NEWSGE5" in selected:
            out["NEWSGE5"] = (out["NEWS"] >= 5).astype(int)
        if "NEWSGE7" in selected:
            out["NEWSGE7"] = (out["NEWS"] >= 7).astype(int)
        if "NEWSGE10" in selected:
            out["NEWSGE10"] = (out["NEWS"] >= 10).astype(int)
        if "SOFAGE2" in selected:
            out["SOFAGE2"] = (out["SOFA"] >= 2).astype(int)
        if "SOFAGE4" in selected:
            out["SOFAGE4"] = (out["SOFA"] >= 4).astype(int)
        if "SOFAGE6" in selected:
            out["SOFAGE6"] = (out["SOFA"] >= 6).astype(int)

        return out


def _evaluate_probabilities(y_true: pd.Series, y_prob: np.ndarray) -> dict[str, float]:
    y_arr = y_true.astype(int).to_numpy()
    return {
        "roc_auc": float(roc_auc_score(y_arr, y_prob)),
        "pr_auc": float(average_precision_score(y_arr, y_prob)),
        "brier": float(brier_score_loss(y_arr, y_prob)),
        "log_loss": float(log_loss(y_arr, y_prob, labels=[0, 1])),
    }


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.lstrip("\ufeff") for c in df.columns]
    target_col = _resolve_target_column(df.columns.tolist(), TARGET)

    missing_base = [col for col in BASE_FEATURES if col not in df.columns]
    if missing_base:
        raise ValueError(f"Missing expected base features: {missing_base}")

    model_df = df[BASE_FEATURES + [target_col]].copy()
    model_df = model_df.dropna(subset=[target_col]).reset_index(drop=True)

    X = model_df[BASE_FEATURES]
    y = model_df[target_col].astype(int)

    X_train_val, X_test_unused, y_train_val, y_test_unused = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=SEED,
        stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.25,
        random_state=SEED,
        stratify=y_train_val,
    )

    rows: list[dict[str, Any]] = []
    for idx, feature_config in enumerate(FEATURE_SETS, start=1):
        engineer = FeatureEngineer(feature_config).fit(X_train)
        X_train_feat = engineer.transform(X_train)
        X_val_feat = engineer.transform(X_val)

        start = time.perf_counter()
        clf = TabICLClassifier(**TABICL_PARAMS)
        clf.fit(X_train_feat, y_train)
        val_prob = clf.predict_proba(X_val_feat)[:, 1]
        elapsed = time.perf_counter() - start

        added_features = [c for c in X_train_feat.columns if c not in BASE_FEATURES]
        metrics = _evaluate_probabilities(y_val, val_prob)
        row = {
            "experiment": feature_config["name"],
            "runtime_sec": round(elapsed, 3),
            "feature_count": int(X_train_feat.shape[1]),
            "added_feature_count": int(len(added_features)),
            "added_features": added_features,
            "validation_metrics": metrics,
        }
        rows.append(row)
        print(
            f"[{idx:02d}/{len(FEATURE_SETS):02d}] {feature_config['name']}: "
            f"val_pr_auc={metrics['pr_auc']:.4f}, val_roc_auc={metrics['roc_auc']:.4f}, "
            f"val_brier={metrics['brier']:.4f}, features={X_train_feat.shape[1]}, runtime={elapsed:.1f}s"
        )

    leaderboard = pd.DataFrame(
        [
            {
                "experiment": row["experiment"],
                "feature_count": row["feature_count"],
                "added_feature_count": row["added_feature_count"],
                "val_pr_auc": row["validation_metrics"]["pr_auc"],
                "val_roc_auc": row["validation_metrics"]["roc_auc"],
                "val_brier": row["validation_metrics"]["brier"],
                "val_log_loss": row["validation_metrics"]["log_loss"],
                "runtime_sec": row["runtime_sec"],
            }
            for row in rows
        ]
    ).sort_values(by=["val_pr_auc", "val_roc_auc", "val_brier"], ascending=[False, False, True])

    best_name = leaderboard.iloc[0]["experiment"]
    best_row = next(row for row in rows if row["experiment"] == best_name)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "feature_experiments_full.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    leaderboard.to_csv(OUTPUT_DIR / "feature_experiments_leaderboard.csv", index=False)
    (OUTPUT_DIR / "feature_experiments_best.json").write_text(json.dumps(best_row, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "feature_experiments_metadata.json").write_text(
        json.dumps(
            {
                "data_path": str(DATA_PATH),
                "target": target_col,
                "base_features": BASE_FEATURES,
                "model_config": TABICL_PARAMS,
                "split_counts": {
                    "train": int(len(X_train)),
                    "val": int(len(X_val)),
                    "held_out_test_unused": int(len(X_test_unused)),
                },
                "seed": SEED,
                "selection_rule": ["val_pr_auc", "val_roc_auc", "val_brier"],
                "best_experiment": best_name,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Saved {OUTPUT_DIR / 'feature_experiments_leaderboard.csv'}")
    print(f"Saved {OUTPUT_DIR / 'feature_experiments_best.json'}")
    print(f"Best experiment: {best_name}")


if __name__ == "__main__":
    main()
