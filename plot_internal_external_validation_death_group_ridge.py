from __future__ import annotations

from pathlib import Path

import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


SITE_ORDER = ["Site_1", "Site_2", "Site_3", "Site_4"]
SITE_LABEL_MAP = {
    "Site_1": "Site 1",
    "Site_2": "Site 2",
    "Site_3": "Site 3",
    "Site_4": "Site 4",
}
SITE_COLOR_MAP = {
    "Site_1": "#4E79A7",
    "Site_2": "#F28E2B",
    "Site_3": "#59A14F",
    "Site_4": "#E15759",
}


def _apply_plot_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 320,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9.5,
            "font.size": 10.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.7,
        }
    )


def _save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _load_predictions(path: Path, model_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "model" not in df.columns:
        raise ValueError("Missing 'model' column in prediction file.")
    model_df = df.loc[df["model"] == model_name].copy()
    if model_df.empty:
        models = ", ".join(map(str, sorted(df["model"].dropna().unique().tolist())))
        raise ValueError(f"No predictions found for model={model_name!r}. Available models: {models}")

    model_df["test_site"] = pd.Categorical(model_df["test_site"], categories=SITE_ORDER, ordered=True)
    return model_df.sort_values(["test_site", "row_index"]).reset_index(drop=True)


def _plot_roc_by_site(predictions: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    baseline, = ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="#9CA3AF",
        linewidth=1.3,
        label="No-discrimination",
    )
    for site in SITE_ORDER:
        site_df = predictions.loc[predictions["test_site"] == site]
        y_true = site_df["y_true"].to_numpy(dtype=int)
        y_prob = site_df["proba_death"].to_numpy(dtype=float)
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = roc_auc_score(y_true, y_prob)
        ax.plot(
            fpr,
            tpr,
            color=SITE_COLOR_MAP[site],
            linewidth=2.4,
            label=f"{SITE_LABEL_MAP[site]} (AUC {roc_auc:.3f})",
        )

    ax.set(
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0),
        xlabel="False positive rate",
        ylabel="True positive rate",
        title="Receiver operating characteristics curves for in-hospital mortality using the Group ridge model",
    )
    ax.legend(
        loc="lower right",
        frameon=True,
        framealpha=0.92,
        fontsize=9.0,
        handlelength=1.9,
        ncol=1,
    )
    fig.tight_layout()
    _save_figure(fig, output_dir, "figure_group_ridge_external_roc_by_site")


def _plot_pr_by_site(predictions: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    prevalence = float(predictions["y_true"].mean())
    ax.plot(
        [0, 1],
        [prevalence, prevalence],
        linestyle="--",
        color="#9CA3AF",
        linewidth=1.3,
        label=f"Prevalence = {prevalence:.3f}",
    )

    for site in SITE_ORDER:
        site_df = predictions.loc[predictions["test_site"] == site]
        y_true = site_df["y_true"].to_numpy(dtype=int)
        y_prob = site_df["proba_death"].to_numpy(dtype=float)
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
        ax.plot(
            recall,
            precision,
            color=SITE_COLOR_MAP[site],
            linewidth=2.4,
            label=f"{SITE_LABEL_MAP[site]} (AP {pr_auc:.3f})",
        )

    ax.set(
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0),
        xlabel="Recall",
        ylabel="Precision",
        title="Precision-recall curves for in-hospital mortality using the Group ridge model",
    )
    ax.legend(
        loc="upper right",
        frameon=True,
        framealpha=0.92,
        fontsize=9.0,
        handlelength=1.9,
        ncol=1,
    )
    fig.tight_layout()
    _save_figure(fig, output_dir, "figure_group_ridge_external_pr_by_site")


def _plot_calibration_by_site(predictions: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="#9CA3AF",
        linewidth=1.2,
        label="Perfect calibration",
    )
    for site in SITE_ORDER:
        site_df = predictions.loc[predictions["test_site"] == site]
        pooled_y = site_df["y_true"].to_numpy(dtype=int)
        pooled_prob = site_df["proba_death"].to_numpy(dtype=float)
        frac_pos, mean_pred = calibration_curve(pooled_y, pooled_prob, n_bins=8, strategy="quantile")
        brier = brier_score_loss(pooled_y, pooled_prob)
        ax.plot(
            mean_pred,
            frac_pos,
            color=SITE_COLOR_MAP[site],
            linewidth=2.3,
            marker="o",
            markersize=5,
            alpha=0.92,
            label=f"{SITE_LABEL_MAP[site]} (Brier {brier:.3f})",
        )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Calibration plots for in-hospital mortality using the Group ridge model")
    ax.legend(
        loc="lower right",
        frameon=True,
        framealpha=0.92,
        fontsize=9.0,
        handlelength=1.9,
        ncol=1,
    )
    fig.tight_layout()
    _save_figure(fig, output_dir, "figure_group_ridge_external_calibration_by_site")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot group ridge internal-external validation curves by holdout site.")
    parser.add_argument(
        "--input-dir",
        default="/Users/ac/comforter/artifacts/internal_external_validation_death",
        help="Directory containing all_predictions.csv with group_ridge holdout predictions.",
    )
    parser.add_argument(
        "--output-dir",
        default="/Users/ac/comforter/artifacts/internal_external_validation_death_publication/publication_ready_group_ridge",
        help="Output directory for figures.",
    )
    parser.add_argument(
        "--model",
        default="group_ridge",
        help="Model key in all_predictions.csv",
    )

    args = parser.parse_args()
    predictions = _load_predictions(Path(args.input_dir) / "all_predictions.csv", args.model)

    _apply_plot_style()
    output_dir = Path(args.output_dir)
    _plot_roc_by_site(predictions, output_dir)
    _plot_pr_by_site(predictions, output_dir)
    _plot_calibration_by_site(predictions, output_dir)
    print(f"Saved group ridge internal-external figures to {output_dir}")


if __name__ == "__main__":
    main()
