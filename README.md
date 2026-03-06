## TabICL Prediction Workflow

This project predicts `DeathHospDisch` from `NewComfModelData.csv` using
[`tabicl`](https://github.com/soda-inria/tabicl), with stratified
train/validation/test splits.

Excluded predictors (case-insensitive): `ID`, `Mort30d`, `ICUWithin48h`, `METWithin48h`.

### Run

```bash
uv run python train_tabicl.py \
  --data NewComfModelData.csv \
  --target DeathHospDisch \
  --exclude ID Mort30d ICUWithin48h METWithin48h \
  --test-size 0.20 \
  --val-size 0.20 \
  --seed 42 \
  --output-dir artifacts/tabicl
```

Optional: force CPU inference.

```bash
uv run python train_tabicl.py --device cpu
```

### Outputs

- `artifacts/tabicl/metrics.json`: split metadata + validation/test metrics
- `artifacts/tabicl/val_predictions.csv`: row-level validation predictions/probabilities
- `artifacts/tabicl/test_predictions.csv`: row-level test predictions/probabilities
