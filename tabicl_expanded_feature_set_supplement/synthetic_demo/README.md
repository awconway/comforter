# Synthetic Demo

This directory contains a fully synthetic dataset that matches the 31-column schema of the expanded-feature-set TabICL model.

Files:
- `synthetic_support_expanded_feature_set.csv`: labelled synthetic support set used to refit TabICL.
- `synthetic_query_expanded_feature_set.csv`: synthetic query set to be scored.
- `run_synthetic_demo.py`: end-to-end demo that:
  1. refits TabICL on the synthetic support set; and
  2. loads the archived serialized fitted object and scores the same query set.
- `synthetic_predictions_from_refit.csv`: predictions produced by the refit path.
- `synthetic_predictions_from_serialized_object.csv`: predictions produced by the serialized-object path.

Run from the supplement root:

```bash
uv sync
uv run python synthetic_demo/run_synthetic_demo.py
```

The two prediction files are not expected to be identical:

- `synthetic_predictions_from_refit.csv` comes from refitting TabICL on the small synthetic support set.
- `synthetic_predictions_from_serialized_object.csv` comes from the archived all-data fitted object.
