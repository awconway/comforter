from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
from tabicl import TabICLClassifier

ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = ROOT / "synthetic_demo"
SUPPORT_CSV = DEMO_DIR / "synthetic_support_expanded_feature_set.csv"
QUERY_CSV = DEMO_DIR / "synthetic_query_expanded_feature_set.csv"
FITTED_OBJECT = ROOT / "fitted_model" / "tabicl_expanded_feature_set_all_data.pkl"

FEATURES = [
    "Age", "Sex", "ARP", "AdmitPrev24h", "SurgPrev24h", "ICUDischPrev24h", "METWithinPrev24h",
    "NEWS", "ICD", "SOFA", "sofa_cns", "sofa_cvs", "sofa_coag", "sofa_livr", "sofa_renl",
    "sofa_resp", "news_loc", "news_pulse", "news_sbp", "news_temp", "news_resp", "news_spo2",
    "news_o2", "ICD_nutrition", "ICD_continence", "ICD_hygiene", "ICD_mobilisation",
    "ICD_observation", "ICD_medication", "ICD_cognition", "ICD_skin",
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
    support = pd.read_csv(SUPPORT_CSV)
    query = pd.read_csv(QUERY_CSV)

    X_support = support[FEATURES].copy()
    y_support = support["DeathHospDisch"].astype(int)
    X_query = query[FEATURES].copy()

    clf = TabICLClassifier(**TABICL_PARAMS)
    clf.fit(X_support, y_support)
    refit_prob = positive_probability(clf, X_query)

    with FITTED_OBJECT.open("rb") as fh:
        frozen_clf = pickle.load(fh)
    frozen_prob = positive_probability(frozen_clf, X_query)

    refit_out = pd.DataFrame({
        "Id": query["Id"],
        "row_index": range(len(query)),
        "proba_death": refit_prob,
        "proba_survival": 1.0 - refit_prob,
    })
    frozen_out = pd.DataFrame({
        "Id": query["Id"],
        "row_index": range(len(query)),
        "proba_death": frozen_prob,
        "proba_survival": 1.0 - frozen_prob,
    })

    refit_path = DEMO_DIR / "synthetic_predictions_from_refit.csv"
    frozen_path = DEMO_DIR / "synthetic_predictions_from_serialized_object.csv"
    refit_out.to_csv(refit_path, index=False)
    frozen_out.to_csv(frozen_path, index=False)

    print(f"Saved {refit_path}")
    print(f"Saved {frozen_path}")


if __name__ == "__main__":
    main()
