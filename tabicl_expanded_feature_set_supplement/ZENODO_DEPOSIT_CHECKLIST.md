# Zenodo Deposit Checklist

Use this checklist before uploading the folder to Zenodo.

## Required checks

1. Confirm that `README.md`, `USAGE.md`, and the expanded-feature predictor dictionary still match the manuscript wording.
2. Confirm that this archive is intended to describe the 31-predictor TabICL expanded feature set rather than the earlier core TabICL model.
3. Confirm that the included serialized fitted object in `fitted_model/tabicl_expanded_feature_set_all_data.pkl` is approved for deposit, given that it may embed patient-level support information.
4. If a de-identified support dataset will also be shared, add it and update `README.md` accordingly.
5. Confirm that the checkpoint identifier in `tabicl_expanded_feature_set_model_spec.json` is still the intended frozen version.
6. Confirm that no patient-level prediction files or other restricted outputs have been copied into the archive unintentionally.
7. Regenerate `SHA256SUMS.tsv` if any file is changed.

## Optional improvements before deposition

1. Add a license file if the deposit will be released publicly.
2. Add a citation file or plain-text citation note if the Zenodo record should cite the manuscript or repository.
3. Add a brief note describing how a final deployable expanded-feature TabICL object was created if one is included.
