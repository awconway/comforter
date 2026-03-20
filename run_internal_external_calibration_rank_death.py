from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss


RUNS = {
    "baseline_shift": Path("/Users/ac/comforter/artifacts/internal_external_validation_death_stability_baseline_shift"),
    "drop_MET": Path("/Users/ac/comforter/artifacts/internal_external_validation_death_stability_drop_met"),
    "drop_MET+Sex": Path("/Users/ac/comforter/artifacts/internal_external_validation_death_stability_drop_met_sex"),
    "transform_MET_binary": Path("/Users/ac/comforter/artifacts/internal_external_validation_death_stability_transformed_met_binary"),
}

EPS = 1e-6


def _cal_metrics(y_true: np.ndarray, probs: np.ndarray) -> tuple[float, float, float]:
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(probs, dtype=float), EPS, 1.0 - EPS)
    brier = float(brier_score_loss(y, p))
    logits = np.log(p / (1.0 - p)).reshape(-1, 1)
    lr = LogisticRegression(C=1e12, solver="lbfgs", fit_intercept=True, max_iter=2000)
    lr.fit(logits, y)
    citl = float(lr.intercept_[0])
    slope = float(lr.coef_[0, 0])
    return brier, citl, slope


def _collect_metrics(output_dir: Path, run_name: str) -> list[dict[str, float | str | int]]:
    pred_path = output_dir / "all_predictions.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing predictions file: {pred_path}")

    rows: list[dict[str, float | str | int]] = []
    pred_df = pd.read_csv(pred_path)
    for model, model_df in pred_df.groupby("model"):
        model_df = model_df.sort_values(["test_site", "row_index"]).reset_index(drop=True)
        brier, citl, slope = _cal_metrics(model_df["y_true"].to_numpy(), model_df["proba_death"].to_numpy())
        rows.append(
            {
                "run": run_name,
                "model": model,
                "scope": "overall_holdout",
                "test_site": "pooled",
                "n": int(len(model_df)),
                "events": int(model_df["y_true"].sum()),
                "prevalence": float(model_df["y_true"].mean()),
                "brier": float(brier),
                "citl": float(citl),
                "abs_citl": float(abs(citl)),
                "slope": float(slope),
                "abs_slope_dev": float(abs(slope - 1.0)),
            }
        )

        for site, site_df in model_df.groupby("test_site"):
            site_df = site_df.reset_index(drop=True)
            site_brier, site_citl, site_slope = _cal_metrics(
                site_df["y_true"].to_numpy(),
                site_df["proba_death"].to_numpy(),
            )
            rows.append(
                {
                    "run": run_name,
                    "model": model,
                    "scope": "site",
                    "test_site": site,
                    "n": int(len(site_df)),
                    "events": int(site_df["y_true"].sum()),
                    "prevalence": float(site_df["y_true"].mean()),
                    "brier": float(site_brier),
                    "citl": float(site_citl),
                    "abs_citl": float(abs(site_citl)),
                    "slope": float(site_slope),
                    "abs_slope_dev": float(abs(site_slope - 1.0)),
                }
            )
    return rows


def _minmax(series: pd.Series) -> pd.Series:
    min_val = float(series.min())
    max_val = float(series.max())
    if np.isclose(max_val, min_val):
        return pd.Series(0.0, index=series.index)
    return (series - min_val) / (max_val - min_val)


def _create_rankings(metrics_df: pd.DataFrame, out_dir: Path) -> None:
    pooled = metrics_df.query('scope == "overall_holdout"').copy().reset_index(drop=True)
    site = metrics_df.query('scope == "site"').copy().reset_index(drop=True)
    site_summary = (
        site.groupby(["run", "model"])
        .agg(
            site_brier_mean=("brier", "mean"),
            site_brier_range=("brier", lambda s: float(s.max() - s.min())),
            site_citl_mean=("abs_citl", "mean"),
            site_citl_max=("abs_citl", "max"),
            site_slope_dev_mean=("abs_slope_dev", "mean"),
            site_slope_dev_max=("abs_slope_dev", "max"),
        )
        .reset_index()
    )

    ranking = pooled.merge(site_summary, on=["run", "model"], how="left")
    ranking["calibration_gap"] = ranking["abs_slope_dev"] + ranking["abs_citl"] + ranking["brier"]

    norm = pd.DataFrame(
        {
            "brier_rank_norm": _minmax(ranking["brier"]),
            "citl_rank_norm": _minmax(ranking["abs_citl"]),
            "slope_rank_norm": _minmax(ranking["abs_slope_dev"]),
            "site_brier_range_norm": _minmax(ranking["site_brier_range"]),
            "site_citl_norm": _minmax(ranking["site_citl_max"]),
            "site_slope_norm": _minmax(ranking["site_slope_dev_max"]),
        },
        index=ranking.index,
    )
    ranking["calibration_composite"] = norm.mean(axis=1)
    ranking = ranking.sort_values("calibration_composite", kind="stable").reset_index(drop=True)
    ranking.insert(0, "rank", ranking.index + 1)

    # write all tables
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(out_dir / "site_and_pooled_calibration_metrics.csv", index=False)
    pooled.to_csv(out_dir / "pooled_calibration_summary.csv", index=False)
    site_summary.to_csv(out_dir / "site_calibration_aggregation.csv", index=False)
    ranking.to_csv(out_dir / "calibration_ranking.csv", index=False)

    # markdown snapshot for easy review
    markdown_lines = ["# Calibration-only Internal-External Ranking", "", "## Run/Model Ranking", ""]
    for model in ranking["model"].sort_values().unique():
        markdown_lines.append(f"### {model}")
        markdown_lines.append("")
        model_rank = ranking.loc[ranking["model"] == model, [
            "rank",
            "run",
            "brier",
            "abs_citl",
            "abs_slope_dev",
            "site_brier_range",
            "site_citl_max",
            "site_slope_dev_max",
            "calibration_composite",
        ]]
        markdown_lines.append("| Rank | Run | Brier | |CITL| | |slope-1| | Site Brier Range | Site max |Site slope max| Composite |")
        markdown_lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|")
        for row in model_rank.itertuples(index=False):
            markdown_lines.append(
                "| {rank:.0f} | {run} | {brier:.4f} | {abs_citl:.4f} | {abs_slope_dev:.4f} | "
                "{site_brier_range:.4f} | {site_citl_max:.4f} | {site_slope_dev_max:.4f} | {calibration_composite:.4f} |".format(
                    rank=float(row.rank),
                    run=row.run,
                    brier=float(row.brier),
                    abs_citl=float(row.abs_citl),
                    abs_slope_dev=float(row.abs_slope_dev),
                    site_brier_range=float(row.site_brier_range),
                    site_citl_max=float(row.site_citl_max),
                    site_slope_dev_max=float(row.site_slope_dev_max),
                    calibration_composite=float(row.calibration_composite),
                )
            )
        markdown_lines.append("")
    out_dir.joinpath("calibration_ranking.md").write_text(
        "\n".join(markdown_lines) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute hold-out calibration metrics and rank DeathHospDisch internal-external runs by calibration only."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/Users/ac/comforter/artifacts/predictor_stability_death/calibration_sensitivity"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, float | str | int]] = []
    for run_name, base in RUNS.items():
        rows.extend(_collect_metrics(base, run_name))
    metrics_df = pd.DataFrame(rows).sort_values(["model", "run", "scope", "test_site"]).reset_index(drop=True)
    _create_rankings(metrics_df, args.output_dir)
    print(f"Saved calibration-only outputs under {args.output_dir}")


if __name__ == "__main__":
    main()
