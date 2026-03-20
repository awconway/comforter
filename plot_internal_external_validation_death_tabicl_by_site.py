from __future__ import annotations

from pathlib import Path

import argparse
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve, roc_auc_score, roc_curve


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


def _plot_roc_by_site(predictions: pd.DataFrame, output_dir: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    baseline, = ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="#9CA3AF",
        linewidth=1.3,
        label="No-discrimination",
    )
    site_lines: list[tuple[str, float]] = []

    for site in SITE_ORDER:
        site_df = predictions.loc[predictions["test_site"] == site].copy()
        y_true = site_df["y_true"].to_numpy(dtype=int)
        y_prob = site_df["proba_death"].to_numpy(dtype=float)
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = roc_auc_score(y_true, y_prob)
        line, = ax.plot(
            fpr,
            tpr,
            color=SITE_COLOR_MAP[site],
            linewidth=2.4,
            label=f"{SITE_LABEL_MAP[site]} (AUC {roc_auc:.3f})",
        )
        site_lines.append((site, roc_auc))

    ax.set(
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0),
        xlabel="False positive rate",
        ylabel="True positive rate",
        title=title,
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
    _save_figure(fig, output_dir, "figure_1_external_roc_by_site")


def _plot_pr_by_site(predictions: pd.DataFrame, output_dir: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    prevalence = float(predictions["y_true"].mean())
    baseline, = ax.plot(
        [0, 1],
        [prevalence, prevalence],
        linestyle="--",
        color="#9CA3AF",
        linewidth=1.3,
        label=f"Prevalence = {prevalence:.3f}",
    )
    for site in SITE_ORDER:
        site_df = predictions.loc[predictions["test_site"] == site].copy()
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
        title=title,
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
    _save_figure(fig, output_dir, "figure_1_external_pr_by_site")


def _plot_calibration_by_site(predictions: pd.DataFrame, output_dir: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#9CA3AF", linewidth=1.2, label="Perfect calibration")

    for site in SITE_ORDER:
        site_df = predictions.loc[predictions["test_site"] == site].copy()
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
    ax.set_title(title)
    ax.legend(
        loc="lower right",
        frameon=True,
        framealpha=0.92,
        fontsize=9.0,
        handlelength=1.9,
        ncol=1,
    )
    fig.tight_layout()
    _save_figure(fig, output_dir, "figure_2_external_calibration_by_site")


def _resolve_model_name(predictions: pd.DataFrame) -> str:
    if "model" not in predictions.columns:
        return "TabICL"
    model_values = sorted(predictions["model"].dropna().unique().tolist())
    if len(model_values) == 0:
        return "TabICL"
    if len(model_values) == 1:
        return "TabICL" if str(model_values[0]).lower() in {"tabicl", "tabicl_model"} else str(model_values[0])
    return "TabICL"


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot TabICL internal-external hold-out curves by site.")
    parser.add_argument(
        "--input-dir",
        default="/Users/ac/comforter/artifacts/internal_external_validation_death_tabicl_shift",
        help="Directory containing TabICL all_predictions.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="/Users/ac/comforter/artifacts/internal_external_validation_death_publication/publication_ready_tabicl_shift",
        help="Directory for output figures",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    predictions = pd.read_csv(input_dir / "all_predictions.csv")
    predictions["test_site"] = pd.Categorical(predictions["test_site"], categories=SITE_ORDER, ordered=True)
    predictions = predictions.sort_values(["test_site", "row_index"]).reset_index(drop=True)

    model_name = _resolve_model_name(predictions)
    _apply_plot_style()

    _plot_roc_by_site(
        predictions,
        output_dir,
        f"Receiver operating characteristics curves for in-hospital mortality using the {model_name} model",
    )
    _plot_pr_by_site(
        predictions,
        output_dir,
        f"Precision-recall curves for in-hospital mortality using the {model_name} model",
    )
    _plot_calibration_by_site(
        predictions,
        output_dir,
        f"Calibration plots for in-hospital mortality using the {model_name} model",
    )

    print(f"Saved plots to {output_dir}")


if __name__ == "__main__":
    main()
