from __future__ import annotations

import contextlib
import io
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve, roc_auc_score, roc_curve


ROOT = Path("/Users/ac/comforter")
PUB_ROOT = ROOT / "artifacts" / "internal_external_validation_death_publication"
REMAIN_DIR = PUB_ROOT / "publication_ready_remain_logistic"
MERGED_MODELS_DIR = ROOT / "artifacts" / "internal_external_validation_death_merged_selected_combined"
NEWS_LOGISTIC_DIR = ROOT / "artifacts" / "internal_external_validation_death_news_logistic"
OUTPUT_DIR = ROOT / "publication_quality_figures"
DPI = 1000

SITE_KEYS = ["Site_1", "Site_2", "Site_3", "Site_4"]
SITE_LABELS = {
    "Site_1": "Site 1",
    "Site_2": "Site 2",
    "Site_3": "Site 3",
    "Site_4": "Site 4",
}
SITE_STYLES = {
    "Site_1": {"color": "#111111", "linestyle": "-", "marker": "o"},
    "Site_2": {"color": "#444444", "linestyle": "--", "marker": "s"},
    "Site_3": {"color": "#666666", "linestyle": "-.", "marker": "^"},
    "Site_4": {"color": "#888888", "linestyle": ":", "marker": "D"},
}
MODEL_STYLES = {
    "TabICL": {"color": "#111111", "linestyle": "-", "marker": "o"},
    "NEWS logistic": {"color": "#666666", "linestyle": "--", "marker": "^"},
}


def _configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.55,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.2,
            "xtick.labelsize": 8.4,
            "ytick.labelsize": 8.4,
            "legend.fontsize": 8.1,
            "savefig.facecolor": "white",
        }
    )


def _style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.grid(True, which="major", color="#DDDDDD", linewidth=0.55)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#777777")
    ax.spines["bottom"].set_color("#777777")
    ax.tick_params(colors="#111111", length=3.0, width=0.55)
    ax.xaxis.label.set_color("#111111")
    ax.yaxis.label.set_color("#111111")
    ax.title.set_color("#111111")


def _save_figure(fig: plt.Figure, stem: str) -> list[Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_stem = OUTPUT_DIR / stem
    paths = [
        output_stem.with_suffix(".pdf"),
        output_stem.with_suffix(".eps"),
        output_stem.with_suffix(".png"),
        output_stem.with_suffix(".tiff"),
    ]
    fig.savefig(paths[0], format="pdf", bbox_inches=None, pad_inches=0)
    fig.savefig(paths[1], format="eps", bbox_inches=None, pad_inches=0)
    fig.savefig(paths[2], format="png", dpi=DPI, bbox_inches=None, pad_inches=0)
    plt.close(fig)

    with Image.open(paths[2]) as image:
        rgb = image.convert("RGB")
        rgb.save(paths[2], dpi=(DPI, DPI))
        rgb.save(paths[3], compression="tiff_lzw", dpi=(DPI, DPI))
    return paths


def _new_single_axis_figure() -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(7.48, 5.55), dpi=DPI)
    fig.subplots_adjust(left=0.105, right=0.972, bottom=0.115, top=0.925)
    _style_axis(ax)
    return fig, ax


def _load_remain_predictions() -> pd.DataFrame:
    return pd.read_csv(REMAIN_DIR / "combined_all_predictions.csv")


def _model_predictions(model_key: str, source_path: Path) -> pd.DataFrame:
    predictions = pd.read_csv(source_path)
    out = predictions.loc[predictions["model"] == model_key].copy()
    if out.empty:
        raise ValueError(f"No rows found for model={model_key!r} in {source_path}")
    return out


def _plot_remain_roc(predictions: pd.DataFrame) -> list[Path]:
    fig, ax = _new_single_axis_figure()
    ax.plot([0, 1], [0, 1], color="#999999", linestyle=(0, (4, 3)), linewidth=1.0, label="No discrimination")

    for site in SITE_KEYS:
        site_df = predictions.loc[predictions["test_site"] == site]
        if site_df.empty:
            continue
        y_true = site_df["y_true"].to_numpy(dtype=int)
        y_prob = site_df["proba_death"].to_numpy(dtype=float)
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        style = SITE_STYLES[site]
        ax.plot(
            fpr,
            tpr,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.7,
            label=f"{SITE_LABELS[site]} (AUC {auc:.2f})",
        )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("REMAIN model ROC curve", loc="left", fontweight="bold", pad=8)
    ax.legend(loc="lower right", frameon=False, handlelength=2.5)
    return _save_figure(fig, "figure_01_remain_roc_by_site")


def _plot_remain_pr(predictions: pd.DataFrame) -> list[Path]:
    fig, ax = _new_single_axis_figure()
    prevalence = float(predictions["y_true"].mean())
    ax.plot(
        [0, 1],
        [prevalence, prevalence],
        color="#999999",
        linestyle=(0, (4, 3)),
        linewidth=1.0,
        label=f"Prevalence = {prevalence:.2f}",
    )

    for site in SITE_KEYS:
        site_df = predictions.loc[predictions["test_site"] == site]
        if site_df.empty:
            continue
        y_true = site_df["y_true"].to_numpy(dtype=int)
        y_prob = site_df["proba_death"].to_numpy(dtype=float)
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        style = SITE_STYLES[site]
        ax.plot(
            recall,
            precision,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.7,
            label=f"{SITE_LABELS[site]} (AP {ap:.2f})",
        )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("REMAIN model precision-recall curve", loc="left", fontweight="bold", pad=8)
    ax.legend(loc="upper right", frameon=False, handlelength=2.5)
    return _save_figure(fig, "figure_02_remain_pr_by_site")


def _plot_remain_calibration(predictions: pd.DataFrame) -> list[Path]:
    fig, ax = _new_single_axis_figure()
    ax.plot([0, 1], [0, 1], color="#999999", linestyle=(0, (4, 3)), linewidth=1.0, label="Perfect calibration")

    for site in SITE_KEYS:
        site_df = predictions.loc[predictions["test_site"] == site]
        if site_df.empty:
            continue
        y_true = site_df["y_true"].to_numpy(dtype=int)
        y_prob = site_df["proba_death"].to_numpy(dtype=float)
        frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=8, strategy="quantile")
        brier = brier_score_loss(y_true, y_prob)
        style = SITE_STYLES[site]
        ax.plot(
            mean_pred,
            frac_pos,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.55,
            marker=style["marker"],
            markersize=4.3,
            markerfacecolor="white",
            markeredgecolor=style["color"],
            markeredgewidth=0.85,
            label=f"{SITE_LABELS[site]} (Brier {brier:.2f})",
        )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("REMAIN model calibration curve", loc="left", fontweight="bold", pad=8)
    ax.legend(loc="lower right", frameon=False, handlelength=2.5)
    return _save_figure(fig, "figure_03_remain_calibration_by_site")


def _plot_tabicl_news_comparison(tabicl_predictions: pd.DataFrame, news_predictions: pd.DataFrame) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(7.48, 3.85), dpi=DPI)
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.16, top=0.88, wspace=0.28)
    roc_ax, cal_ax = axes
    for ax in axes:
        _style_axis(ax)

    model_specs = [
        ("TabICL", tabicl_predictions),
        ("NEWS logistic", news_predictions),
    ]

    roc_ax.plot([0, 1], [0, 1], color="#AAAAAA", linestyle=(0, (4, 3)), linewidth=0.9, label="No discrimination")
    for label, predictions in model_specs:
        y_true = predictions["y_true"].to_numpy(dtype=int)
        y_prob = predictions["proba_death"].to_numpy(dtype=float)
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        style = MODEL_STYLES[label]
        roc_ax.plot(
            fpr,
            tpr,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.65,
            label=f"{label} (AUC {auc:.2f})",
        )

    roc_ax.set_xlim(0.0, 1.0)
    roc_ax.set_ylim(0.0, 1.0)
    roc_ax.set_xlabel("False positive rate")
    roc_ax.set_ylabel("True positive rate")
    roc_ax.set_title("A  ROC curve", loc="left", fontweight="bold", pad=7)
    roc_ax.legend(loc="lower right", frameon=False, handlelength=2.4)

    cal_ax.plot([0, 1], [0, 1], color="#AAAAAA", linestyle=(0, (4, 3)), linewidth=0.9, label="Perfect calibration")
    for label, predictions in model_specs:
        y_true = predictions["y_true"].to_numpy(dtype=int)
        y_prob = predictions["proba_death"].to_numpy(dtype=float)
        frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
        brier = brier_score_loss(y_true, y_prob)
        style = MODEL_STYLES[label]
        cal_ax.plot(
            mean_pred,
            frac_pos,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.55,
            marker=style["marker"],
            markersize=4.0,
            markerfacecolor="white",
            markeredgecolor=style["color"],
            markeredgewidth=0.8,
            label=f"{label} (Brier {brier:.2f})",
        )

    cal_ax.set_xlim(0.0, 1.0)
    cal_ax.set_ylim(0.0, 1.0)
    cal_ax.set_xlabel("Mean predicted probability")
    cal_ax.set_ylabel("Observed frequency")
    cal_ax.set_title("B  Calibration plot", loc="left", fontweight="bold", pad=7)
    cal_ax.legend(loc="lower right", frameon=False, handlelength=2.4)

    return _save_figure(fig, "figure_04_tabicl_vs_news_roc_calibration")


def _plot_auc_forest() -> list[Path]:
    import create_publication_auc_forest as auc_forest

    auc_forest.OUTPUT_DIR = OUTPUT_DIR
    with contextlib.redirect_stdout(io.StringIO()):
        auc_forest.main()
    stem = OUTPUT_DIR / "figure_internal_external_auc_forest_publication"
    return [stem.with_suffix(suffix) for suffix in [".pdf", ".eps", ".png", ".tiff"]]


def main() -> None:
    _configure_matplotlib()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    remain_predictions = _load_remain_predictions()
    tabicl_predictions = _model_predictions("tabicl", MERGED_MODELS_DIR / "all_predictions.csv")
    news_predictions = _model_predictions("news_logistic", NEWS_LOGISTIC_DIR / "all_predictions.csv")

    paths: list[Path] = []
    paths.extend(_plot_remain_roc(remain_predictions))
    paths.extend(_plot_remain_pr(remain_predictions))
    paths.extend(_plot_remain_calibration(remain_predictions))
    paths.extend(_plot_tabicl_news_comparison(tabicl_predictions, news_predictions))
    paths.extend(_plot_auc_forest())

    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
