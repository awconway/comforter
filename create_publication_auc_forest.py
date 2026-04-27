from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon, Rectangle
from PIL import Image


ROOT = Path("/Users/ac/comforter")
PUB_ROOT = ROOT / "artifacts" / "internal_external_validation_death_publication"
MODELS_DIR = PUB_ROOT / "publication_ready"
MERGED_MODELS_DIR = ROOT / "artifacts" / "internal_external_validation_death_merged_selected_combined"
NEWS_LOGISTIC_DIR = ROOT / "artifacts" / "internal_external_validation_death_news_logistic"
OUTPUT_DIR = PUB_ROOT / "publication_ready"

SITE_LABEL = {
    "Site_1": "Site 1",
    "Site_2": "Site 2",
    "Site_3": "Site 3",
    "Site_4": "Site 4",
}
SITE_ORDER = {"Site_1": 1, "Site_2": 2, "Site_3": 3, "Site_4": 4}
MODEL_ORDER = {
    "NEWS logistic": 1,
    "TabICL": 2,
    "Group lasso": 3,
    "Group ridge": 4,
    "AutoGluon": 5,
    "Random forest": 6,
    "TabICL (Expanded feature set)": 7,
    "AutoGluon (Expanded feature set)": 8,
    "Random forest (Expanded feature set)": 9,
}
FOREST_MODELS = {
    "NEWS logistic",
    "Group lasso",
    "Group ridge",
    "TabICL (Expanded feature set)",
    "AutoGluon (Expanded feature set)",
    "Random forest (Expanded feature set)",
}
DISPLAY_LABEL_MAP = {
    "TabICL (Expanded feature set)": "TabICL",
    "AutoGluon (Expanded feature set)": "AutoGluon",
    "Random forest (Expanded feature set)": "Random forest",
}


def _site_auc_ci(auc: float, auc_variance: float) -> tuple[float, float]:
    auc_clip = float(np.clip(auc, 1e-6, 1.0 - 1e-6))
    variance = float(max(auc_variance, 0.0))
    logit_auc = float(np.log(auc_clip / (1.0 - auc_clip)))
    se_logit = float(np.sqrt(variance) / (auc_clip * (1.0 - auc_clip)))
    lower = 1.0 / (1.0 + np.exp(-(logit_auc - 1.96 * se_logit)))
    upper = 1.0 / (1.0 + np.exp(-(logit_auc + 1.96 * se_logit)))
    return float(lower), float(upper)


def _random_effects_weight_percent(site_rows: pd.DataFrame, tau2_logit_auc: float) -> np.ndarray:
    auc = np.clip(site_rows["auc"].to_numpy(dtype=float), 1e-6, 1.0 - 1e-6)
    auc_var = site_rows["auc_variance"].to_numpy(dtype=float)
    logit_var = auc_var / (auc**2 * (1.0 - auc) ** 2)
    weights = 1.0 / (logit_var + float(tau2_logit_auc))
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0.0:
        return np.full(len(site_rows), np.nan, dtype=float)
    return 100.0 * weights / total


def _load_forest_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    model_fold_base = pd.read_csv(MODELS_DIR / "combined_fold_auc.csv")
    model_label_base = {
        "tabicl": "TabICL",
        "group_lasso": "Group lasso",
        "group_ridge": "Group ridge",
        "autogluon": "AutoGluon",
        "random_forest": "Random forest",
    }
    model_fold_base["Model"] = model_fold_base["model"].map(model_label_base).fillna(model_fold_base["model"])
    model_meta_base = pd.read_csv(MODELS_DIR / "combined_summary_auc.csv")
    model_meta_base["Model"] = model_meta_base["model"].map(model_label_base).fillna(model_meta_base["model"])

    model_fold_merged = pd.read_csv(MERGED_MODELS_DIR / "fold_auc.csv")
    model_label_merged = {
        "tabicl": "TabICL (Expanded feature set)",
        "autogluon": "AutoGluon (Expanded feature set)",
        "random_forest": "Random forest (Expanded feature set)",
    }
    model_fold_merged["Model"] = model_fold_merged["model"].map(model_label_merged).fillna(model_fold_merged["model"])
    model_meta_merged = pd.read_csv(MERGED_MODELS_DIR / "summary_auc.csv")
    model_meta_merged["Model"] = model_meta_merged["model"].map(model_label_merged).fillna(model_meta_merged["model"])

    model_fold_news = pd.read_csv(NEWS_LOGISTIC_DIR / "fold_auc.csv")
    model_fold_news["Model"] = model_fold_news["model"].map({"news_logistic": "NEWS logistic"}).fillna(
        model_fold_news["model"]
    )
    model_meta_news = pd.read_csv(NEWS_LOGISTIC_DIR / "summary_auc.csv")
    model_meta_news["Model"] = model_meta_news["model"].map({"news_logistic": "NEWS logistic"}).fillna(
        model_meta_news["model"]
    )

    model_fold = pd.concat([model_fold_base, model_fold_merged, model_fold_news], ignore_index=True)
    model_meta = pd.concat([model_meta_base, model_meta_merged, model_meta_news], ignore_index=True)
    return (
        model_fold.loc[model_fold["Model"].isin(FOREST_MODELS)].copy(),
        model_meta.loc[model_meta["Model"].isin(FOREST_MODELS)].copy(),
    )


def _build_rows(fold_auc: pd.DataFrame, summary_auc: pd.DataFrame) -> list[dict[str, float | str]]:
    ranked = summary_auc.copy()
    ranked["model_rank"] = ranked["Model"].map(MODEL_ORDER)
    ranked = ranked.sort_values(["pooled_auc", "model_rank"], ascending=[False, True]).reset_index(drop=True)

    rows: list[dict[str, float | str]] = []
    for model in ranked["Model"]:
        pooled_row = ranked.loc[ranked["Model"] == model].iloc[0]
        display_model = DISPLAY_LABEL_MAP.get(model, model)
        site_rows = fold_auc.loc[fold_auc["Model"] == model].copy()
        site_rows["site_rank"] = site_rows["test_site"].map(SITE_ORDER)
        site_rows = site_rows.sort_values("site_rank")
        site_rows["weight_percent"] = _random_effects_weight_percent(site_rows, float(pooled_row["tau2_logit_auc"]))

        rows.append(
            {
                "kind": "model_header",
                "label": display_model,
                "tau2": float(pooled_row["tau2_logit_auc"]),
                "i2": float(pooled_row["i2_percent"]),
                "pi_lower": float(pooled_row["pi_95_lower"]),
                "pi_upper": float(pooled_row["pi_95_upper"]),
            }
        )
        for _, row in site_rows.iterrows():
            ci_lower, ci_upper = _site_auc_ci(float(row["auc"]), float(row["auc_variance"]))
            estimate = float(row["auc"])
            rows.append(
                {
                    "kind": "site",
                    "label": SITE_LABEL.get(row["test_site"], row["test_site"]),
                    "events": f"{int(row['n_pos'])}/{int(row['n_test'])}",
                    "estimate": estimate,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "weight": float(row["weight_percent"]),
                    "right_text": f"{estimate:.2f} ({ci_lower:.2f} to {ci_upper:.2f})",
                }
            )

        pooled_estimate = float(pooled_row["pooled_auc"])
        pooled_lower = float(pooled_row["ci_95_lower"])
        pooled_upper = float(pooled_row["ci_95_upper"])
        rows.append(
            {
                "kind": "pooled",
                "label": "Pooled",
                "events": f"{int(site_rows['n_pos'].sum())}/{int(site_rows['n_test'].sum())}",
                "estimate": pooled_estimate,
                "ci_lower": pooled_lower,
                "ci_upper": pooled_upper,
                "right_text": f"{pooled_estimate:.2f} ({pooled_lower:.2f} to {pooled_upper:.2f})",
            }
        )
    return rows


def _draw_diamond(ax: plt.Axes, cx: float, cy: float, half_w: float, half_h: float, *, fill: str) -> None:
    ax.add_patch(
        Polygon(
            [(cx, cy - half_h), (cx + half_w, cy), (cx, cy + half_h), (cx - half_w, cy)],
            closed=True,
            facecolor=fill,
            edgecolor="#111111",
            linewidth=0.45,
            joinstyle="miter",
            zorder=4,
        )
    )


def _draw_publication_forest(rows: list[dict[str, float | str]]) -> plt.Figure:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.5,
        }
    )

    intervals = [
        (float(row["ci_lower"]), float(row["ci_upper"]))
        for row in rows
        if row["kind"] in {"site", "pooled"}
    ]
    x_lower = min(lower for lower, _ in intervals)
    x_upper = max(upper for _, upper in intervals)
    x_min = max(0.5, np.floor((x_lower - 0.015) / 0.05) * 0.05)
    x_max = min(0.99, np.ceil((x_upper + 0.015) / 0.05) * 0.05)
    ticks = np.arange(x_min, x_max + 1e-9, 0.05)

    width = 1320
    header_bottom = 92
    row_h = 30
    group_header_h = 34
    pooled_h = 32
    group_gap = 12
    bottom_pad = 112
    plot_x = 392
    plot_w = 356
    label_x = 24
    site_label_x = 44
    events_x = 278
    weight_x = 820
    estimate_x = 910
    text = "#111111"
    border = "#9A9A9A"
    grid = "#D8D8D8"

    y = header_bottom + 20
    placed_rows: list[dict[str, float | str]] = []
    for row in rows:
        kind = str(row["kind"])
        height = group_header_h if kind == "model_header" else pooled_h if kind == "pooled" else row_h
        placed = row.copy()
        placed["y"] = y + height / 2.0
        placed["height"] = height
        placed_rows.append(placed)
        y += height
        if kind == "pooled":
            y += group_gap

    height = int(y + bottom_pad)
    plot_top = header_bottom + 14
    plot_bottom = height - bottom_pad + 12
    width_in = 7.48
    height_in = width_in * height / width

    fig = plt.figure(figsize=(width_in, height_in), dpi=1000, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")
    ax.patch.set_zorder(0)

    def x_pos(value: float) -> float:
        return plot_x + (float(value) - float(x_min)) / (float(x_max) - float(x_min)) * plot_w

    def add_text(
        x: float,
        y_pos: float,
        value: str,
        *,
        size: float,
        weight: str = "normal",
        ha: str = "left",
    ) -> None:
        ax.text(x, y_pos, value, ha=ha, va="center", fontsize=size, fontweight=weight, color=text)

    ax.add_patch(
        Rectangle((1, 1), width - 2, height - 2, facecolor="white", edgecolor=border, linewidth=0.6, zorder=0.1)
    )
    ax.plot([1, width - 1], [header_bottom, header_bottom], color=border, linewidth=0.6, zorder=2.2)

    add_text(label_x, 34, "Model / site", size=8.3, weight="bold")
    add_text(events_x, 34, "Deaths / total", size=8.3, weight="bold", ha="center")
    add_text(plot_x + plot_w / 2, 25, "ROC-AUC", size=8.3, weight="bold", ha="center")
    add_text(plot_x + plot_w / 2, 56, "(95% CI)", size=7.6, weight="bold", ha="center")
    add_text(weight_x, 34, "Weight (%)", size=8.3, weight="bold", ha="center")
    add_text(estimate_x, 25, "ROC-AUC", size=8.3, weight="bold")
    add_text(estimate_x, 56, "(95% CI)", size=7.6, weight="bold")

    ax.add_patch(
        Rectangle(
            (plot_x, plot_top),
            plot_w,
            plot_bottom - plot_top,
            facecolor="white",
            edgecolor=border,
            linewidth=0.55,
            zorder=0.2,
        )
    )
    for tick in ticks:
        tx = x_pos(float(tick))
        ax.plot([tx, tx], [plot_top, plot_bottom], color=grid, linewidth=0.38, zorder=1.0)
    ax.add_patch(
        Rectangle(
            (plot_x, plot_top),
            plot_w,
            plot_bottom - plot_top,
            facecolor="none",
            edgecolor=border,
            linewidth=0.55,
            zorder=2.0,
        )
    )

    for row in placed_rows:
        y_mid = float(row["y"])
        kind = str(row["kind"])
        if kind == "model_header":
            add_text(label_x, y_mid, str(row["label"]), size=7.4, weight="bold")
            heterogeneity = (
                f"Tau²={float(row['tau2']):.2f}; I²={float(row['i2']):.2f}%; "
                f"PI {float(row['pi_lower']):.2f}-{float(row['pi_upper']):.2f}"
            )
            add_text(estimate_x, y_mid, heterogeneity, size=6.2)
            continue

        is_pooled = kind == "pooled"
        lower = float(row["ci_lower"])
        upper = float(row["ci_upper"])
        estimate = float(row["estimate"])
        x_lower_ci = x_pos(lower)
        x_upper_ci = x_pos(upper)
        x_estimate = x_pos(estimate)
        line_width = 0.85 if is_pooled else 0.65
        ax.plot(
            [x_lower_ci, x_upper_ci],
            [y_mid, y_mid],
            color=text,
            linewidth=line_width,
            solid_capstyle="round",
            zorder=2.5,
        )
        if is_pooled:
            _draw_diamond(ax, x_estimate, y_mid, 7.6, 6.6, fill=text)
        else:
            marker_half = 5.0 + 0.08 * float(row["weight"])
            _draw_diamond(ax, x_estimate, y_mid, marker_half, marker_half, fill=text)

        body_weight = "bold" if is_pooled else "normal"
        add_text(site_label_x, y_mid, str(row["label"]), size=6.6, weight=body_weight)
        add_text(events_x, y_mid, str(row["events"]), size=6.6, ha="center")
        if not is_pooled:
            add_text(weight_x, y_mid, f"{float(row['weight']):.2f}", size=6.6, ha="center")
        add_text(estimate_x, y_mid, str(row["right_text"]), size=6.6, weight=body_weight)

    axis_y = plot_bottom + 20
    ax.plot([plot_x, plot_x + plot_w], [plot_bottom, plot_bottom], color=border, linewidth=0.45)
    for tick in ticks:
        tx = x_pos(float(tick))
        ax.plot([tx, tx], [plot_bottom, plot_bottom + 6], color=text, linewidth=0.38)
        add_text(tx, axis_y, f"{tick:.2f}", size=6.2, ha="center")
    add_text(plot_x + plot_w / 2, plot_bottom + 45, "ROC-AUC", size=7.2, ha="center")

    legend_y = height - 20
    legend_x = 34
    _draw_diamond(ax, legend_x, legend_y, 6.2, 6.2, fill=text)
    add_text(legend_x + 20, legend_y, "Site estimate", size=6.2)
    ax.plot([legend_x + 126, legend_x + 174], [legend_y, legend_y], color=text, linewidth=0.85, solid_capstyle="round")
    _draw_diamond(ax, legend_x + 150, legend_y, 6.6, 5.8, fill=text)
    add_text(legend_x + 190, legend_y, "Pooled estimate and 95% CI", size=6.2)

    return fig


def main() -> None:
    fold_auc, summary_auc = _load_forest_data()
    rows = _build_rows(fold_auc, summary_auc)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    stem = OUTPUT_DIR / "figure_internal_external_auc_forest_publication"
    fig = _draw_publication_forest(rows)
    fig.savefig(stem.with_suffix(".pdf"), format="pdf", bbox_inches=None, pad_inches=0)
    fig.savefig(stem.with_suffix(".eps"), format="eps", bbox_inches=None, pad_inches=0)
    fig.savefig(stem.with_suffix(".png"), format="png", dpi=1000, bbox_inches=None, pad_inches=0)
    plt.close(fig)

    with Image.open(stem.with_suffix(".png")) as image:
        rgb = image.convert("RGB")
        rgb.save(stem.with_suffix(".png"), dpi=(1000, 1000))
        rgb.save(stem.with_suffix(".tiff"), compression="tiff_lzw", dpi=(1000, 1000))

    print(stem.with_suffix(".pdf"))
    print(stem.with_suffix(".eps"))
    print(stem.with_suffix(".png"))
    print(stem.with_suffix(".tiff"))


if __name__ == "__main__":
    main()
