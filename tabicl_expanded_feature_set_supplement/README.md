# TabICL Supplement

This folder is a supplement for the TabICL model with expanded feature set used in the COMFORTER internal-external validation analyses.

It was assembled to satisfy the intent of TRIPOD item 15 for a tabular foundation model. For TabICL, coefficient-style reporting is not applicable. Instead, this archive provides the exact predictor definitions, value encodings, checkpoint identifier, inference settings, and executable code required to generate predictions for new individuals.

## Scope of this archive

This archive is specific to the TabICL model that is comprised of 31 direct predictors:

- age, sex, and ARP;
- recent-admission, recent-surgery, recent-ICU-discharge, and recent-MET history variables;
- total NEWS, SOFA, and mICD scores; and
- component-level NEWS, SOFA, and mICD variables.

## Folder contents

- `model_specification/`
  - `tabicl_expanded_feature_set_model_spec.json`: frozen settings and implementation notes for the expanded-feature-set TabICL IECV analysis.
  - `predictor_dictionary_expanded_feature_set.csv`: predictor names, encodings, and observed ranges.
  - `iecv_tabicl_expanded_feature_set_fold_auc.csv`: site-specific ROC-AUC estimates.
  - `iecv_tabicl_expanded_feature_set_random_effects_summary.csv`: pooled random-effects ROC-AUC summary.
- `inference/`
  - `predict_tabicl_expanded_feature_set_from_csv.py`: standalone script to fit TabICL on a labelled support set and score a query set using the 31 expanded predictors.
  - `support_template_expanded_feature_set.csv` and `query_template_expanded_feature_set.csv`: header templates for the expected input files.
- `synthetic_demo/`
  - fully synthetic support and query CSVs with the correct 31-column schema;
  - `run_synthetic_demo.py`: end-to-end smoke test that produces probabilities from both the refit path and the serialized-object path; and
  - synthetic prediction CSVs generated from that demo.
- `fitted_model/`
  - `tabicl_expanded_feature_set_all_data.pkl`: serialized final TabICL object fit on all available rows of `MergedData.csv`.
  - `tabicl_expanded_feature_set_all_data_metadata.json`: metadata describing the fitting dataset, predictor set, and frozen settings.
- standalone `uv` files:
  - `pyproject.toml`
  - `uv.lock`
- `provenance/`
  - copied environment files;
  - copied source code for the local expanded-feature internal-external validation workflow; and
  - copied artifact reports from the local expanded-feature analyses.
- `USAGE.md`: step-by-step notes for generating predictions.
- `ZENODO_DEPOSIT_CHECKLIST.md`: final checks before public deposition.
- `SHA256SUMS.tsv`: checksums for the files in this folder.

## Important implementation note

TabICL predictions are support-set dependent. 

This archive includes a serialized fitted TabICL object created from all available rows of the COMFORTER dataset using the 31-predictor feature set.

## Provenance

All specifications in this folder were generated from the local project files on 2026-04-14.

## Standalone test

This archive includes a standalone `uv` project plus a fully synthetic demo dataset. After extracting the archive, a user can run:

```bash
uv sync
uv run python synthetic_demo/run_synthetic_demo.py
```

to verify that:

1. a schema-correct synthetic support set can be used to refit TabICL and generate probabilities; and
2. the archived serialized fitted object can also generate probabilities for a schema-correct synthetic query set.
