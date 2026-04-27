from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tabicl import TabICLClassifier

FEATURES = [
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
    "sofa_cns",
    "sofa_cvs",
    "sofa_coag",
    "sofa_livr",
    "sofa_renl",
    "sofa_resp",
    "news_loc",
    "news_pulse",
    "news_sbp",
    "news_temp",
    "news_resp",
    "news_spo2",
    "news_o2",
    "ICD_nutrition",
    "ICD_continence",
    "ICD_hygiene",
    "ICD_mobilisation",
    "ICD_observation",
    "ICD_medication",
    "ICD_cognition",
    "ICD_skin"
]

TABICL_PARAMS = {
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
    "random_state": 42,
    "verbose": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit the expanded-feature-set TabICL model on a labelled support set and score a query set. "
            "This mirrors the archived internal-external validation settings for the 31-column expanded feature set."
        )
    )
    parser.add_argument("--support-csv", type=Path, required=True)
    parser.add_argument("--query-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--target-column", default="DeathHospDisch")
    parser.add_argument("--id-column", default="Id")
    parser.add_argument("--device", default=TABICL_PARAMS.get("device", "cpu"))
    parser.add_argument("--checkpoint-version", default=TABICL_PARAMS["checkpoint_version"])
    return parser.parse_args()


def require_columns(df: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def positive_probability(clf: TabICLClassifier, X: pd.DataFrame):
    prob = clf.predict_proba(X)
    classes = list(clf.classes_)
    if 1 in classes:
        pos_idx = classes.index(1)
    elif 1.0 in classes:
        pos_idx = classes.index(1.0)
    else:
        raise ValueError(f"Positive class not found in model classes {classes}")
    return prob[:, pos_idx]


def main() -> None:
    args = parse_args()
    support = pd.read_csv(args.support_csv)
    query = pd.read_csv(args.query_csv)

    require_columns(support, [*FEATURES, args.target_column], "Support CSV")
    require_columns(query, FEATURES, "Query CSV")

    X_support = support[FEATURES].copy()
    y_support = support[args.target_column].astype(int)
    X_query = query[FEATURES].copy()

    params = dict(TABICL_PARAMS)
    params["device"] = args.device
    params["checkpoint_version"] = args.checkpoint_version

    clf = TabICLClassifier(**params)
    clf.fit(X_support, y_support)
    y_prob = positive_probability(clf, X_query)

    out = pd.DataFrame(
        {
            "row_index": range(len(query)),
            "proba_death": y_prob,
            "proba_survival": 1.0 - y_prob,
        }
    )
    if args.id_column in query.columns:
        out.insert(0, args.id_column, query[args.id_column])

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    print(f"Saved {args.output_csv}")


if __name__ == "__main__":
    main()
