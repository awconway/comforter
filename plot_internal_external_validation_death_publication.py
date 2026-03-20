from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
from matplotlib.ticker import PercentFormatter
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


BASELINE_DIR = Path("/Users/ac/comforter/artifacts/internal_external_validation_death_publication")
AUTOGLUON_DIR = Path("/Users/ac/comforter/artifacts/autogluon_death_internal_external")
OUTPUT_DIR = BASELINE_DIR / "publication_ready"

MODEL_SPECS = [
    {"key": "tabicl", "label": "TabICL", "color": "#4E79A7", "linestyle": "-", "marker": "o"},
    {"key": "group_lasso", "label": "Group lasso", "color": "#F28E2B", "linestyle": "--", "marker": "s"},
    {"key": "group_ridge", "label": "Group ridge", "color": "#59A14F", "linestyle": "-.", "marker": "^"},
    {"key": "autogluon", "label": "AutoGluon", "color": "#E15759", "linestyle": (0, (5, 1.5)), "marker": "D"},
    {"key": "random_forest", "label": "Random forest", "color": "#B07AA1", "linestyle": ":", "marker": "P"},
]

SITE_SPECS = [
    {"key": "Site_1", "label": "Site 1", "color": "#1B9E77"},
    {"key": "Site_2", "label": "Site 2", "color": "#D95F02"},
    {"key": "Site_3", "label": "Site 3", "color": "#7570B3"},
    {"key": "Site_4", "label": "Site 4", "color": "#E7298A"},
]

MODEL_LABEL_MAP = {spec["key"]: spec["label"] for spec in MODEL_SPECS}
MODEL_COLOR_MAP = {spec["key"]: spec["color"] for spec in MODEL_SPECS}
MODEL_LINESTYLE_MAP = {spec["key"]: spec["linestyle"] for spec in MODEL_SPECS}
MODEL_MARKER_MAP = {spec["key"]: spec["marker"] for spec in MODEL_SPECS}
SITE_LABEL_MAP = {spec["key"]: spec["label"] for spec in SITE_SPECS}
SITE_COLOR_MAP = {spec["key"]: spec["color"] for spec in SITE_SPECS}
MODEL_ORDER = [spec["key"] for spec in MODEL_SPECS]
SITE_ORDER = [spec["key"] for spec in SITE_SPECS]
LOGIT_EPS = 1e-6


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


def _load_predictions() -> pd.DataFrame:
    baseline = pd.read_csv(BASELINE_DIR / "all_predictions.csv")
    autogluon = pd.read_csv(AUTOGLUON_DIR / "all_predictions.csv")
    combined = pd.concat([baseline, autogluon], ignore_index=True)
    combined["model"] = pd.Categorical(combined["model"], categories=MODEL_ORDER, ordered=True)
    combined["test_site"] = pd.Categorical(combined["test_site"], categories=SITE_ORDER, ordered=True)
    return combined.sort_values(["model", "test_site", "row_index"]).reset_index(drop=True)


def _load_fold_auc() -> pd.DataFrame:
    baseline = pd.read_csv(BASELINE_DIR / "fold_auc.csv")
    autogluon = pd.read_csv(AUTOGLUON_DIR / "fold_auc.csv")
    combined = pd.concat([baseline, autogluon], ignore_index=True)
    combined["model"] = pd.Categorical(combined["model"], categories=MODEL_ORDER, ordered=True)
    combined["test_site"] = pd.Categorical(combined["test_site"], categories=SITE_ORDER, ordered=True)
    return combined.sort_values(["model", "test_site"]).reset_index(drop=True)


def _load_summary_auc() -> pd.DataFrame:
    baseline = pd.read_csv(BASELINE_DIR / "summary_auc.csv")
    autogluon = pd.read_csv(AUTOGLUON_DIR / "summary_auc.csv")
    combined = pd.concat([baseline, autogluon], ignore_index=True)
    combined["model"] = pd.Categorical(combined["model"], categories=MODEL_ORDER, ordered=True)
    return combined.sort_values("pooled_auc", ascending=False).reset_index(drop=True)


def _save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUTPUT_DIR / f"{stem}.png", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _format_float(value: float, decimals: int = 3) -> str:
    return f"{value:.{decimals}f}"


def _calibration_intercept_and_slope(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=float)
    probs = np.clip(y_pred, LOGIT_EPS, 1.0 - LOGIT_EPS)
    logits = np.log(probs / (1.0 - probs)).reshape(-1, 1)

    model = LogisticRegression(
        C=1e12,
        solver="lbfgs",
        fit_intercept=True,
        max_iter=2000,
    )
    model.fit(logits, y_true)
    intercept = float(model.intercept_[0])
    slope = float(model.coef_.ravel()[0])
    return intercept, slope


def _render_markdown_table(df: pd.DataFrame, path: Path) -> None:
    headers = [str(col) for col in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in df.itertuples(index=False, name=None):
        escaped = [str(cell).replace("|", "\\|") for cell in row]
        lines.append("| " + " | ".join(escaped) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_summary_tables(predictions: pd.DataFrame, summary_auc: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, float | str]] = []
    for model_key in MODEL_ORDER:
        model_df = predictions.loc[predictions["model"] == model_key].copy()
        y_true = model_df["y_true"].to_numpy(dtype=int)
        y_prob = model_df["proba_death"].to_numpy(dtype=float)
        metric_rows.append(
            {
                "model": model_key,
                "overall_holdout_roc_auc": float(roc_auc_score(y_true, y_prob)),
                "overall_holdout_pr_auc": float(average_precision_score(y_true, y_prob)),
                "overall_holdout_brier": float(brier_score_loss(y_true, y_prob)),
                "n_holdout": int(len(model_df)),
                "events": int(model_df["y_true"].sum()),
                "prevalence": float(model_df["y_true"].mean()),
            }
        )

    metric_df = pd.DataFrame(metric_rows)
    merged = summary_auc.merge(metric_df, on="model", how="left")
    merged["Model"] = merged["model"].map(MODEL_LABEL_MAP)

    summary_numeric = merged[
        [
            "Model",
            "pooled_auc",
            "ci_95_lower",
            "ci_95_upper",
            "pi_95_lower",
            "pi_95_upper",
            "overall_holdout_roc_auc",
            "overall_holdout_pr_auc",
            "overall_holdout_brier",
            "i2_percent",
            "n_holdout",
            "events",
            "prevalence",
        ]
    ].rename(
        columns={
            "pooled_auc": "Pooled ROC-AUC",
            "ci_95_lower": "CI lower",
            "ci_95_upper": "CI upper",
            "pi_95_lower": "PI lower",
            "pi_95_upper": "PI upper",
            "overall_holdout_roc_auc": "Overall holdout ROC-AUC",
            "overall_holdout_pr_auc": "Overall holdout PR-AUC",
            "overall_holdout_brier": "Overall holdout Brier",
            "i2_percent": "I2 (%)",
            "n_holdout": "Holdout n",
            "events": "Events",
            "prevalence": "Prevalence",
        }
    )

    summary_display = pd.DataFrame(
        {
            "Model": summary_numeric["Model"],
            "Pooled ROC-AUC": summary_numeric["Pooled ROC-AUC"].map(_format_float),
            "95% CI": [
                f"{_format_float(low)} to {_format_float(high)}"
                for low, high in zip(summary_numeric["CI lower"], summary_numeric["CI upper"], strict=False)
            ],
            "95% PI": [
                f"{_format_float(low)} to {_format_float(high)}"
                for low, high in zip(summary_numeric["PI lower"], summary_numeric["PI upper"], strict=False)
            ],
            "Overall holdout ROC-AUC": summary_numeric["Overall holdout ROC-AUC"].map(_format_float),
            "Overall holdout PR-AUC": summary_numeric["Overall holdout PR-AUC"].map(_format_float),
            "Overall holdout Brier": summary_numeric["Overall holdout Brier"].map(_format_float),
            "I2 (%)": summary_numeric["I2 (%)"].map(lambda value: _format_float(value, 1)),
            "Holdout n": summary_numeric["Holdout n"].map(lambda value: f"{int(value)}"),
            "Events": summary_numeric["Events"].map(lambda value: f"{int(value)}"),
            "Prevalence": summary_numeric["Prevalence"].map(_format_float),
        }
    )
    return summary_numeric, summary_display


def _build_site_level_tables(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, float | str | int]] = []
    for model_key in MODEL_ORDER:
        for site_key in SITE_ORDER:
            site_df = predictions.loc[
                (predictions["model"] == model_key) & (predictions["test_site"] == site_key)
            ].copy()
            y_true = site_df["y_true"].to_numpy(dtype=int)
            y_prob = site_df["proba_death"].to_numpy(dtype=float)
            cal_intercept, cal_slope = _calibration_intercept_and_slope(y_true, y_prob)
            rows.append(
                {
                    "Site": SITE_LABEL_MAP[site_key],
                    "Model": MODEL_LABEL_MAP[model_key],
                    "n": int(len(site_df)),
                    "Events": int(site_df["y_true"].sum()),
                    "Prevalence": float(site_df["y_true"].mean()),
                    "ROC-AUC": float(roc_auc_score(y_true, y_prob)),
                    "PR-AUC": float(average_precision_score(y_true, y_prob)),
                    "Brier": float(brier_score_loss(y_true, y_prob)),
                    "CITL": cal_intercept,
                    "Slope": cal_slope,
                }
            )

    site_numeric = pd.DataFrame(rows)
    site_display = site_numeric.copy()
    for column in ["Prevalence", "ROC-AUC", "PR-AUC", "Brier", "CITL", "Slope"]:
        site_display[column] = site_display[column].map(_format_float)
    site_display["n"] = site_display["n"].map(lambda value: f"{int(value)}")
    site_display["Events"] = site_display["Events"].map(lambda value: f"{int(value)}")
    return site_numeric, site_display


def _write_tables(predictions: pd.DataFrame, summary_auc: pd.DataFrame) -> None:
    summary_numeric, summary_display = _build_summary_tables(predictions, summary_auc)
    site_numeric, site_display = _build_site_level_tables(predictions)

    summary_numeric.to_csv(OUTPUT_DIR / "table_1_internal_external_summary.csv", index=False)
    site_numeric.to_csv(OUTPUT_DIR / "table_s1_site_level_metrics.csv", index=False)

    _render_markdown_table(summary_display, OUTPUT_DIR / "table_1_internal_external_summary.md")
    _render_markdown_table(site_display, OUTPUT_DIR / "table_s1_site_level_metrics.md")


def _plot_roc_pr(predictions: pd.DataFrame) -> None:
    prevalence = float(predictions["y_true"].mean())
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.2))
    roc_ax, pr_ax = axes

    roc_ax.plot([0, 1], [0, 1], linestyle="--", color="#9CA3AF", linewidth=1.3, label="Chance")
    pr_ax.axhline(prevalence, linestyle="--", color="#9CA3AF", linewidth=1.3, label=f"Prevalence = {prevalence:.3f}")

    for model_key in MODEL_ORDER:
        model_df = predictions.loc[predictions["model"] == model_key].copy()
        color = MODEL_COLOR_MAP[model_key]

        y_true = model_df["y_true"].to_numpy(dtype=int)
        y_prob = model_df["proba_death"].to_numpy(dtype=float)
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        roc_auc = roc_auc_score(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)

        roc_ax.plot(
            fpr,
            tpr,
            color=color,
            linewidth=2.4,
            linestyle="-",
            alpha=0.8,
            label=f"{MODEL_LABEL_MAP[model_key]} (AUC {roc_auc:.3f})",
            zorder=3,
        )
        pr_ax.plot(
            recall,
            precision,
            color=color,
            linewidth=2.4,
            linestyle="-",
            alpha=0.8,
            label=f"{MODEL_LABEL_MAP[model_key]} (AP {pr_auc:.3f})",
            zorder=3,
        )

    roc_ax.set(
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0),
        xlabel="False positive rate",
        ylabel="True positive rate",
        title="ROC curves across site holdouts",
    )
    pr_ax.set(
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0),
        xlabel="Recall",
        ylabel="Precision",
        title="Precision-recall curves across site holdouts",
    )
    roc_ax.legend(loc="lower right", frameon=True, framealpha=0.95)
    pr_ax.legend(loc="lower left", frameon=True, framealpha=0.95)

    fig.suptitle(
        "DeathHospDisch internal-external validation",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    fig.text(
        0.5,
        0.01,
        "Each curve is computed from the concatenated out-of-site holdout predictions across all facilities for that model.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout()
    _save_figure(fig, "figure_1_internal_external_roc_pr")


def _plot_calibration(predictions: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.1, 6.4))
    line_halo = [pe.Stroke(linewidth=4.0, foreground="white", alpha=0.85), pe.Normal()]
    ax.plot([0, 1], [0, 1], linestyle="--", color="#9CA3AF", linewidth=1.2, label="Perfect calibration")

    for model_key in MODEL_ORDER:
        model_df = predictions.loc[predictions["model"] == model_key].copy()
        pooled_y = model_df["y_true"].to_numpy(dtype=int)
        pooled_prob = model_df["proba_death"].to_numpy(dtype=float)
        frac_pos, mean_pred = calibration_curve(pooled_y, pooled_prob, n_bins=8, strategy="quantile")
        brier = brier_score_loss(pooled_y, pooled_prob)

        ax.plot(
            mean_pred,
            frac_pos,
            color=MODEL_COLOR_MAP[model_key],
            linewidth=2.3,
            linestyle=MODEL_LINESTYLE_MAP[model_key],
            marker=MODEL_MARKER_MAP[model_key],
            markersize=5,
            alpha=0.92,
            label=f"{MODEL_LABEL_MAP[model_key]} (Brier {brier:.3f})",
            path_effects=line_halo,
        )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Calibration across pooled out-of-site holdouts")
    ax.legend(loc="upper left", frameon=True, framealpha=0.95)

    fig.text(
        0.5,
        0.01,
        "Each curve is computed after concatenating all out-of-site holdout predictions for that model across the four facilities.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout()
    _save_figure(fig, "figure_2_internal_external_calibration")


def _plot_forest(summary_auc: pd.DataFrame, fold_auc: pd.DataFrame) -> None:
    ranked = summary_auc.copy().reset_index(drop=True)
    model_positions = np.arange(len(ranked))[::-1]
    model_y = {model_key: pos for model_key, pos in zip(ranked["model"], model_positions, strict=False)}

    fig, ax = plt.subplots(figsize=(10.5, 5.8))

    for _, row in ranked.iterrows():
        model_key = row["model"]
        y_pos = model_y[model_key]
        color = MODEL_COLOR_MAP[model_key]

        ax.hlines(
            y=y_pos,
            xmin=row["pi_95_lower"],
            xmax=row["pi_95_upper"],
            color=color,
            linewidth=8,
            alpha=0.22,
            zorder=1,
        )
        ax.hlines(
            y=y_pos,
            xmin=row["ci_95_lower"],
            xmax=row["ci_95_upper"],
            color=color,
            linewidth=4,
            alpha=0.8,
            zorder=2,
        )
        ax.scatter(
            row["pooled_auc"],
            y_pos,
            s=82,
            marker="s",
            color=color,
            edgecolor="#111827",
            linewidth=0.6,
            zorder=4,
        )

    ax.axvline(0.5, color="#9CA3AF", linestyle="--", linewidth=1.1)
    ax.set_yticks(model_positions)
    ax.set_yticklabels([MODEL_LABEL_MAP[model_key] for model_key in ranked["model"]])
    ax.set_xlabel("ROC-AUC")
    ax.set_ylabel("")
    ax.set_xlim(0.55, 0.96)
    ax.set_title("Pooled site-holdout ROC-AUC with confidence and prediction intervals")

    summary_handles = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#374151", markeredgecolor="#111827", markersize=8, label="Pooled estimate"),
        Line2D([0], [0], color="#374151", linewidth=4, alpha=0.8, label="95% confidence interval"),
        Line2D([0], [0], color="#374151", linewidth=8, alpha=0.22, label="95% prediction interval"),
    ]
    ax.legend(
        handles=summary_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=3,
        frameon=False,
    )

    fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
    _save_figure(fig, "figure_3_internal_external_forest")


def _plot_cumulative_lift(predictions: pd.DataFrame, groups: int = 10) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 6.4))
    ax.plot(
        [0, 1],
        [1, 1],
        linestyle="--",
        color="#9CA3AF",
        linewidth=1.3,
        label="Chance",
    )

    for model_key in MODEL_ORDER:
        model_df = predictions.loc[predictions["model"] == model_key].copy()
        model_df = model_df.sort_values("proba_death", ascending=False).reset_index(drop=True)
        y_true = model_df["y_true"].to_numpy(dtype=int)
        n = len(model_df)
        overall_event_rate = float(np.mean(y_true))
        fractions = np.linspace(1 / groups, 1.0, num=groups)
        cumulative_lift: list[float] = []
        for fraction in fractions:
            cutoff = max(1, int(np.ceil(fraction * n)))
            cumulative_event_rate = float(np.mean(y_true[:cutoff]))
            cumulative_lift.append(cumulative_event_rate / overall_event_rate)

        ax.plot(
            fractions,
            cumulative_lift,
            color=MODEL_COLOR_MAP[model_key],
            linewidth=2.5,
            alpha=0.88,
            marker="o",
            markersize=5,
            label=MODEL_LABEL_MAP[model_key],
        )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(bottom=0.9)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Patients screened by predicted risk rank")
    ax.set_ylabel("Cumulative lift vs overall death rate")
    ax.set_title("Cumulative lift for DeathHospDisch")
    ax.legend(loc="upper right", frameon=True, framealpha=0.95)

    fig.text(
        0.5,
        0.01,
        "Patients are ordered from highest to lowest predicted death risk. Each point shows the cumulative death rate up to that screening fraction divided by the overall death rate.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout()
    _save_figure(fig, "figure_4_internal_external_cumulative_lift")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _apply_plot_style()

    predictions = _load_predictions()
    fold_auc = _load_fold_auc()
    summary_auc = _load_summary_auc()

    predictions.to_csv(OUTPUT_DIR / "combined_all_predictions.csv", index=False)
    fold_auc.to_csv(OUTPUT_DIR / "combined_fold_auc.csv", index=False)
    summary_auc.to_csv(OUTPUT_DIR / "combined_summary_auc.csv", index=False)

    _write_tables(predictions, summary_auc)
    _plot_roc_pr(predictions)
    _plot_calibration(predictions)
    _plot_forest(summary_auc, fold_auc)
    _plot_cumulative_lift(predictions)

    print(f"Saved combined artifacts in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
