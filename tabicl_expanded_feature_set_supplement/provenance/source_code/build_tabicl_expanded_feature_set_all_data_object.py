from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd
from tabicl import TabICLClassifier

ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = ROOT / "MergedData.csv"
SUPPLEMENT_DIR = ROOT / "zenodo_tabicl_expanded_feature_set_supplement"
OUTPUT_DIR = SUPPLEMENT_DIR / "fitted_model"
FEATURE_COLUMNS = [
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
MODEL_CONFIG = {'n_estimators': 8, 'norm_methods': None, 'feat_shuffle_method': 'shift', 'class_shuffle_method': 'shift', 'outlier_threshold': 4.0, 'softmax_temperature': 0.9, 'average_logits': True, 'support_many_classes': True, 'batch_size': 8, 'checkpoint_version': 'tabicl-classifier-v2-20260212.ckpt', 'device': 'cpu', 'random_state': 42, 'verbose': False}


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["DeathHospDisch"]).copy()
    X = df[FEATURE_COLUMNS].copy()
    y = df["DeathHospDisch"].astype(int)

    clf = TabICLClassifier(**MODEL_CONFIG)
    clf.fit(X, y)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = OUTPUT_DIR / "tabicl_expanded_feature_set_all_data.pkl"
    meta_path = OUTPUT_DIR / "tabicl_expanded_feature_set_all_data_metadata.json"

    with model_path.open("wb") as fh:
        pickle.dump(clf, fh, protocol=pickle.HIGHEST_PROTOCOL)

    metadata = {
        "artifact_type": "serialized_fitted_tabicl_object",
        "artifact_path": str(model_path.relative_to(SUPPLEMENT_DIR)),
        "data_source": str(DATA_PATH),
        "n_rows_fit": int(len(df)),
        "n_positive": int(y.sum()),
        "n_negative": int((1 - y).sum()),
        "outcome_prevalence": float(y.mean()),
        "target_column": "DeathHospDisch",
        "feature_columns": FEATURE_COLUMNS,
        "predictor_count": len(FEATURE_COLUMNS),
        "model_config": MODEL_CONFIG,
        "notes": [
            "This object was fit on all rows in MergedData.csv with non-missing DeathHospDisch.",
            "No evaluation metrics were generated as part of this fitting step.",
        ],
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved {model_path}")
    print(f"Saved {meta_path}")


if __name__ == "__main__":
    main()
