from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.linear_model import LogisticRegression
from scipy.stats import t

from internal_external_validation_common import sidik_jonkman_tau2

OUTPUT_ROOT = Path("/Users/ac/comforter/artifacts/internal_external_validation_death_publication")
OUTPUT_DIR = OUTPUT_ROOT / "publication_ready_remain_logistic"
INPUT_DIR = Path("/Users/ac/comforter/artifacts/internal_external_validation_remain_logistic")

SITE_ORDER = ["Site_1", "Site_2", "Site_3", "Site_4"]
TEMPORAL_VALIDATION_SITE = "Site_2"
EXTERNAL_VALIDATION_SITES = [site for site in SITE_ORDER if site != TEMPORAL_VALIDATION_SITE]
SITE_COLOR_MAP = {
    "Site_1": "#4E79A7",
    "Site_2": "#F28E2B",
    "Site_3": "#59A14F",
    "Site_4": "#E15759",
}
SITE_LABEL_MAP = {
    "Site_1": "Site 1",
    "Site_2": "Site 2",
    "Site_3": "Site 3",
    "Site_4": "Site 4",
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


def _load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictions = pd.read_csv(INPUT_DIR / "all_predictions.csv")
    summary_auc = pd.read_csv(INPUT_DIR / "summary_auc.csv")
    fold_auc = pd.read_csv(INPUT_DIR / "fold_auc.csv")

    predictions["test_site"] = pd.Categorical(predictions["test_site"], categories=SITE_ORDER, ordered=True)
    fold_auc["test_site"] = pd.Categorical(fold_auc["test_site"], categories=SITE_ORDER, ordered=True)
    summary_auc["model"] = "remain_coeff_logistic"
    predictions["model"] = "remain_coeff_logistic"
    fold_auc["model"] = "remain_coeff_logistic"

    return predictions, summary_auc, fold_auc


def _format_float(value: float, decimals: int = 3) -> str:
    return f"{value:.{decimals}f}"


def _logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 1e-9, 1.0 - 1e-9)
    return np.log(clipped / (1.0 - clipped))


def _expit(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def _bootstrap_metric_sd(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_fn,
    *,
    n_boot: int = 500,
    random_state: int = 42,
) -> tuple[float, float]:
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    if n < 2:
        return float("nan"), float("nan")

    estimates: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        sample_y = y_true[idx]
        sample_prob = y_prob[idx]
        if len(np.unique(sample_y)) < 2:
            continue
        metric = float(metric_fn(sample_y, sample_prob))
        if np.isfinite(metric):
            estimates.append(metric)

    if len(estimates) < 10:
        return float("nan"), float("nan")

    estimates_array = np.asarray(estimates, dtype=float)
    return float(estimates_array.mean()), float(estimates_array.var(ddof=1))


def _calibration_coefficients(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> tuple[float, float]:
    y = np.asarray(y_true, dtype=int)
    if len(y) < 2 or len(np.unique(y)) < 2:
        return float("nan"), float("nan")

    p = np.clip(np.asarray(y_prob, dtype=float), 1e-9, 1.0 - 1e-9)
    x = np.log(p / (1.0 - p)).reshape(-1, 1)

    model = LogisticRegression(
        fit_intercept=True,
        solver="lbfgs",
        max_iter=500,
        C=1e6,
    )
    try:
        model.fit(x, y)
    except Exception:
        return float("nan"), float("nan")

    return float(model.intercept_[0]), float(model.coef_[0, 0])


def _bootstrap_calibration_coeff_sd(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_boot: int = 500,
    random_state: int = 42,
) -> tuple[float, float, float, float]:
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    if n < 2:
        return float("nan"), float("nan"), float("nan"), float("nan")

    intercept_estimates: list[float] = []
    slope_estimates: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        sample_y = y_true[idx]
        sample_prob = y_prob[idx]
        if len(np.unique(sample_y)) < 2:
            continue
        intercept, slope = _calibration_coefficients(sample_y, sample_prob)
        if np.isfinite(intercept) and np.isfinite(slope):
            intercept_estimates.append(intercept)
            slope_estimates.append(slope)

    if len(intercept_estimates) < 10 or len(slope_estimates) < 10:
        return float("nan"), float("nan"), float("nan"), float("nan")

    intercept_array = np.asarray(intercept_estimates, dtype=float)
    slope_array = np.asarray(slope_estimates, dtype=float)
    return (
        float(intercept_array.mean()),
        float(intercept_array.var(ddof=1)),
        float(slope_array.mean()),
        float(slope_array.var(ddof=1)),
    )


def _pool_metric_hksj_unbounded(values: np.ndarray, ses: np.ndarray) -> dict[str, float | str]:
    values = np.asarray(values, dtype=float)
    ses = np.asarray(ses, dtype=float)
    mask = np.isfinite(values) & np.isfinite(ses) & (ses > 0)
    if not np.any(mask):
        raise ValueError("No finite metric values for pooling.")

    values = values[mask]
    ses = ses[mask]
    k = len(values)
    if k <= 1:
        point = float(values.mean())
        return {
            "pooled": point,
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "pi_lower": float("nan"),
            "pi_upper": float("nan"),
            "tau2": float("nan"),
            "i2": float("nan"),
        }

    variances = ses**2
    tau2 = sidik_jonkman_tau2(values, variances)
    weights_re = 1.0 / (variances + tau2)
    pooled_effect = float(np.sum(weights_re * values) / np.sum(weights_re))

    df = max(k - 1, 1)
    hk_scale = float(np.sum(weights_re * (values - pooled_effect) ** 2) / df)
    hk_var = float(max(hk_scale / np.sum(weights_re), 0.0))
    hk_se = float(np.sqrt(hk_var))
    crit = float(t.ppf(0.975, df))

    ci_low = pooled_effect - crit * hk_se
    ci_high = pooled_effect + crit * hk_se

    pred_se = float(np.sqrt(max(tau2 + hk_var, 0.0)))
    pi_low = pooled_effect - crit * pred_se
    pi_high = pooled_effect + crit * pred_se

    weights_fe = 1.0 / variances
    pooled_fe = float(np.sum(weights_fe * values) / np.sum(weights_fe))
    q = float(np.sum(weights_fe * (values - pooled_fe) ** 2))
    i2 = 0.0 if q <= 0.0 else float(max(0.0, (q - (k - 1)) / q) * 100.0)

    return {
        "pooled": pooled_effect,
        "ci_lower": ci_low,
        "ci_upper": ci_high,
        "pi_lower": pi_low,
        "pi_upper": pi_high,
        "tau2": float(tau2),
        "i2": float(i2),
    }


def _pool_metric_hksj(values: np.ndarray, ses: np.ndarray) -> dict[str, float | str]:
    values = np.asarray(values, dtype=float)
    ses = np.asarray(ses, dtype=float)
    mask = np.isfinite(values) & np.isfinite(ses) & (ses > 0)
    if not np.any(mask):
        raise ValueError("No finite metric values for pooling.")

    values = values[mask]
    ses = ses[mask]
    k = len(values)
    if k <= 1:
        point = float(values.mean())
        return {
            "pooled": point,
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "pi_lower": float("nan"),
            "pi_upper": float("nan"),
            "tau2": float("nan"),
            "i2": float("nan"),
        }

    clipped = np.clip(values, 1e-9, 1.0 - 1e-9)
    effects = _logit(clipped)
    variances = ses**2 / (clipped**2 * (1.0 - clipped) ** 2)

    tau2 = sidik_jonkman_tau2(effects, variances)
    weights_re = 1.0 / (variances + tau2)
    pooled_effect = float(np.sum(weights_re * effects) / np.sum(weights_re))

    df = max(k - 1, 1)
    hk_scale = float(np.sum(weights_re * (effects - pooled_effect) ** 2) / df)
    hk_var = float(max(hk_scale / np.sum(weights_re), 0.0))
    hk_se = float(np.sqrt(hk_var))
    crit = float(t.ppf(0.975, df))

    ci_low = pooled_effect - crit * hk_se
    ci_high = pooled_effect + crit * hk_se

    pred_se = float(np.sqrt(max(tau2 + hk_var, 0.0)))
    pi_low = pooled_effect - crit * pred_se
    pi_high = pooled_effect + crit * pred_se

    weights_fe = 1.0 / variances
    pooled_fe = float(np.sum(weights_fe * effects) / np.sum(weights_fe))
    q = float(np.sum(weights_fe * (effects - pooled_fe) ** 2))
    i2 = 0.0 if q <= 0.0 else float(max(0.0, (q - (k - 1)) / q) * 100.0)

    return {
        "pooled": float(_expit(np.array([pooled_effect]))[0]),
        "ci_lower": float(_expit(np.array([ci_low]))[0]),
        "ci_upper": float(_expit(np.array([ci_high]))[0]),
        "pi_lower": float(_expit(np.array([pi_low]))[0]),
        "pi_upper": float(_expit(np.array([pi_high]))[0]),
        "tau2": float(tau2),
        "i2": float(i2),
    }


def _save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUTPUT_DIR / f"{stem}.png", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _legend_group_header(label: str) -> Line2D:
    handle = Line2D([0], [0], linestyle="", marker="")
    handle.set_label(label)
    return handle


def _build_grouped_legend(
    baseline_entry: tuple[object, str] | None,
    site_entries: dict[str, tuple[object, str]],
    *,
    temporal_site: str = TEMPORAL_VALIDATION_SITE,
    external_sites: list[str] = EXTERNAL_VALIDATION_SITES,
) -> tuple[list[object], list[str]]:
    handles: list[object] = []
    labels: list[str] = []

    if baseline_entry is not None:
        handles.append(baseline_entry[0])
        labels.append(baseline_entry[1])

    if external_sites:
        handles.append(_legend_group_header("External validation"))
        labels.append("External validation")
        for site in external_sites:
            if site in site_entries:
                handles.append(site_entries[site][0])
                labels.append(site_entries[site][1])

    if temporal_site in site_entries:
        handles.append(_legend_group_header("Temporal validation"))
        labels.append("Temporal validation")
        handles.append(site_entries[temporal_site][0])
        labels.append(site_entries[temporal_site][1])

    return handles, labels


def _render_markdown_table(
    df: pd.DataFrame,
    path: Path,
    *,
    title: str | None = None,
    footnote: str | None = None,
) -> None:
    lines = []
    if title:
        lines.append(f"## {title}")
        lines.append("")

    headers = [str(col) for col in df.columns]
    lines.extend(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
    )
    for row in df.itertuples(index=False, name=None):
        escaped = [str(cell).replace("|", "\\|") for cell in row]
        lines.append("| " + " | ".join(escaped) + " |")

    if footnote:
        lines.append("")
        lines.append(f"**{footnote}**")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _render_site_level_markdown_grouped(
    df: pd.DataFrame,
    path: Path,
    *,
    title: str | None = None,
    footnote: str | None = None,
) -> None:
    if df.empty:
        lines = ["## Site-level validation metrics", "", "| No data |", "| --- |"]
        if footnote:
            lines.extend(["", f"**{footnote}**"])
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    ordered_sections = []
    for section, _ in df.groupby("Validation type", sort=False):
        ordered_sections.append(section)

    lines = []
    if title:
        lines.append(f"## {title}")
        lines.append("")

    base_cols = [col for col in df.columns if col != "Validation type"]
    headers = [str(col) for col in base_cols]
    lines.extend(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
    )

    for section in ordered_sections:
        section_rows = df.loc[df["Validation type"] == section, base_cols]
        if section_rows.empty:
            continue

        section_header = [f"**{section}**"] + [""] * (len(headers) - 1)
        lines.append("| " + " | ".join(section_header) + " |")

        for row in section_rows.itertuples(index=False, name=None):
            row_values = ["" if value is None else str(value) for value in row]
            escaped = [str(cell).replace("|", "\\|") for cell in row_values]
            lines.append("| " + " | ".join(escaped) + " |")

    if footnote:
        lines.append(f"**{footnote}**")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_interval(value: float, low: float, high: float, *, decimals: int = 3) -> str:
    if pd.isna(value) or pd.isna(low) or pd.isna(high):
        return "—"
    return f"{_format_float(value, decimals)} ({_format_float(low, decimals)} to {_format_float(high, decimals)})"


def _format_point(value: float, *, decimals: int = 3) -> str:
    if pd.isna(value):
        return "—"
    return _format_float(value, decimals)


def _compute_site_level_metrics_with_ci(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []

    for site_idx, site in enumerate(SITE_ORDER):
        site_df = predictions.loc[predictions["test_site"] == site].copy()
        y_true = site_df["y_true"].to_numpy(dtype=int)
        y_prob = site_df["proba_death"].to_numpy(dtype=float)

        if len(y_true) < 2:
            continue

        if len(np.unique(y_true)) < 2:
            roc = float("nan")
            roc_se = float("nan")
            brier = float("nan")
            brier_se = float("nan")
            citl = float("nan")
            slope = float("nan")
            slope_se = float("nan")
            citl_se = float("nan")
        else:
            roc = float(roc_auc_score(y_true, y_prob))
            roc_se = float(np.sqrt(_bootstrap_metric_sd(y_true, y_prob, roc_auc_score, n_boot=500, random_state=120 + site_idx)[1]))
            brier = float(brier_score_loss(y_true, y_prob))

            brier_error = (y_prob - y_true) ** 2
            if len(brier_error) > 1:
                brier_var = float(np.var(brier_error, ddof=1) / len(brier_error))
                brier_se = float(np.sqrt(brier_var))
            else:
                brier_se = float("nan")

            citl, slope = _calibration_coefficients(y_true, y_prob)
            _, citl_var, _, slope_var = _bootstrap_calibration_coeff_sd(
                y_true,
                y_prob,
                n_boot=500,
                random_state=220 + site_idx,
            )
            citl_se = float(np.sqrt(citl_var)) if np.isfinite(citl_var) else float("nan")
            slope_se = float(np.sqrt(slope_var)) if np.isfinite(slope_var) else float("nan")

        for metric, estimate, se, target in [
            ("Calibration-in-the-large (intercept)", citl, citl_se, 0.0),
            ("Calibration slope", slope, slope_se, 1.0),
            ("ROC-AUC", roc, roc_se, 0.5),
            ("Brier score", brier, brier_se, 0.0),
        ]:
            rows.append(
                {
                    "Metric": metric,
                    "Site": site,
                    "Estimate": estimate,
                    "SE": se,
                    "CI lower": estimate - 1.96 * se if np.isfinite(se) and np.isfinite(estimate) else float("nan"),
                    "CI upper": estimate + 1.96 * se if np.isfinite(se) and np.isfinite(estimate) else float("nan"),
                    "Ref": target,
                }
            )

    metrics_df = pd.DataFrame(rows)
    metrics_df["Metric"] = pd.Categorical(metrics_df["Metric"], categories=[
        "Calibration-in-the-large (intercept)",
        "Calibration slope",
        "ROC-AUC",
        "Brier score",
    ], ordered=True)
    return metrics_df


def _build_site_level_ci_table(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows_numeric: list[dict[str, float | int | str]] = []
    rows_display: list[dict[str, float | str | int]] = []

    ordered_sites = EXTERNAL_VALIDATION_SITES + [TEMPORAL_VALIDATION_SITE]
    for site in ordered_sites:
        site_idx = SITE_ORDER.index(site)
        site_df = predictions.loc[predictions["test_site"] == site].copy()
        y_true = site_df["y_true"].to_numpy(dtype=int)
        y_prob = site_df["proba_death"].to_numpy(dtype=float)
        validation_type = "Temporal validation" if site == TEMPORAL_VALIDATION_SITE else "External validation"

        if len(y_true) < 2 or len(np.unique(y_true)) < 2:
            continue

        citl, slope = _calibration_coefficients(y_true, y_prob)
        _, citl_var, _, slope_var = _bootstrap_calibration_coeff_sd(
            y_true,
            y_prob,
            n_boot=500,
            random_state=420 + site_idx,
        )
        citl_se = float(np.sqrt(citl_var)) if np.isfinite(citl_var) else float("nan")
        slope_se = float(np.sqrt(slope_var)) if np.isfinite(slope_var) else float("nan")
        roc = float(roc_auc_score(y_true, y_prob))
        brier = float(brier_score_loss(y_true, y_prob))
        pr_auc = float(average_precision_score(y_true, y_prob))

        _, roc_var = _bootstrap_metric_sd(y_true, y_prob, roc_auc_score, n_boot=500, random_state=420 + site_idx)
        _, brier_var = _bootstrap_metric_sd(y_true, y_prob, brier_score_loss, n_boot=500, random_state=520 + site_idx)
        _, pr_var = _bootstrap_metric_sd(y_true, y_prob, average_precision_score, n_boot=500, random_state=620 + site_idx)

        roc_se = float(np.sqrt(roc_var)) if np.isfinite(roc_var) else float("nan")
        brier_se = float(np.sqrt(brier_var)) if np.isfinite(brier_var) else float("nan")
        pr_se = float(np.sqrt(pr_var)) if np.isfinite(pr_var) else float("nan")

        roc_ci_low = roc - 1.96 * roc_se if np.isfinite(roc_se) else float("nan")
        roc_ci_high = roc + 1.96 * roc_se if np.isfinite(roc_se) else float("nan")
        brier_ci_low = brier - 1.96 * brier_se if np.isfinite(brier_se) else float("nan")
        brier_ci_high = brier + 1.96 * brier_se if np.isfinite(brier_se) else float("nan")
        pr_ci_low = pr_auc - 1.96 * pr_se if np.isfinite(pr_se) else float("nan")
        pr_ci_high = pr_auc + 1.96 * pr_se if np.isfinite(pr_se) else float("nan")

        rows_numeric.append(
            {
                "site": site,
                "validation_type": validation_type,
                "n": int(len(site_df)),
                "events": int(y_true.sum()),
                "prevalence": float(y_true.mean()),
                "citl": citl,
                "slope": slope,
                "roc_auc": roc,
                "roc_auc_ci_lower": roc_ci_low,
                "roc_auc_ci_upper": roc_ci_high,
                "pr_auc": pr_auc,
                "pr_auc_ci_lower": pr_ci_low,
                "pr_auc_ci_upper": pr_ci_high,
                "brier": brier,
                "brier_ci_lower": brier_ci_low,
                "brier_ci_upper": brier_ci_high,
            }
        )
        rows_display.append(
            {
                "Validation type": validation_type,
                "Site": SITE_LABEL_MAP.get(site, site),
                "N": f"{int(len(site_df))}",
                "Deaths": f"{int(y_true.sum())}",
                "CITL (intercept)": _format_point(citl),
                "Calibration slope": _format_point(slope),
                "ROC-AUC [95% CI]": _format_interval(roc, roc_ci_low, roc_ci_high),
                "PR-AUC [95% CI]": _format_interval(pr_auc, pr_ci_low, pr_ci_high),
                "Brier [95% CI]": _format_interval(brier, brier_ci_low, brier_ci_high),
            }
        )

    return pd.DataFrame(rows_numeric), pd.DataFrame(rows_display)


def _plot_site_forest_with_facets(predictions: pd.DataFrame) -> None:
    metrics_df = _compute_site_level_metrics_with_ci(predictions)
    if metrics_df.empty:
        return

    metric_order = [
        "Calibration-in-the-large (intercept)",
        "Calibration slope",
        "ROC-AUC",
        "Brier score",
    ]
    metric_display = {value: value for value in metric_order}

    fig, axes = plt.subplots(2, 2, figsize=(13.8, 9.6), sharey=True)
    for idx, metric in enumerate(metric_order):
        ax = axes[idx // 2, idx % 2]
        metric_rows = metrics_df.loc[metrics_df["Metric"] == metric].copy()
        metric_rows["Site"] = pd.Categorical(metric_rows["Site"], categories=SITE_ORDER, ordered=True)
        metric_rows = metric_rows.sort_values("Site")

        y_positions = np.arange(len(metric_rows))
        for y_pos, row in metric_rows.reset_index(drop=True).iterrows():
            site = str(row["Site"])
            color = SITE_COLOR_MAP[site]
            ax.hlines(
                y=y_pos,
                xmin=row["CI lower"],
                xmax=row["CI upper"],
                color=color,
                linewidth=7.0,
                alpha=0.28,
                zorder=1,
            )
            ax.hlines(
                y=y_pos,
                xmin=row["CI lower"],
                xmax=row["CI upper"],
                color=color,
                linewidth=3.0,
                alpha=0.7,
                zorder=2,
            )
            ax.scatter(
                row["Estimate"],
                y_pos,
                marker="o",
                s=72,
                edgecolor="#111827",
                linewidth=0.7,
                color=color,
                zorder=3,
            )

        ref = float(metric_rows["Ref"].iloc[0]) if not metric_rows.empty else 0.0
        ax.axvline(ref, linestyle="--", color="#9CA3AF", linewidth=1.1)
        ax.set_title(metric_display[metric], fontsize=11.5)
        ax.set_yticks(y_positions)
        ax.set_yticklabels([SITE_LABEL_MAP.get(str(site), str(site).replace("_", " ")) for site in metric_rows["Site"]])
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.24)

        est = metric_rows["Estimate"].to_numpy(dtype=float)
        valid = np.isfinite(est)
        if np.any(valid):
            lo = float(np.nanmin(metric_rows.loc[valid, "CI lower"].to_numpy(dtype=float)))
            hi = float(np.nanmax(metric_rows.loc[valid, "CI upper"].to_numpy(dtype=float)))
            span = hi - lo
            if not np.isfinite(span) or span <= 0:
                span = 0.2
            padding = 0.25 * span
            if metric in {"ROC-AUC", "Brier score"}:
                ax.set_xlim(max(0.0, lo - padding), min(1.0, hi + padding))
            else:
                ax.set_xlim(lo - padding, hi + padding)

    axes[1, 0].set_xlabel("Metric estimate")
    axes[1, 1].set_xlabel("Metric estimate")
    fig.suptitle("REMAIN model site-level metrics by metric", fontsize=15, fontweight="bold", y=0.99)
    fig.tight_layout()
    _save_figure(fig, "figure_5_internal_external_site_forest")


def _build_meta_summary(predictions: pd.DataFrame, summary_auc: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    if summary_auc.empty:
        raise ValueError("Summary AUC file is empty.")

    row = summary_auc.iloc[0]
    y_true = predictions["y_true"].to_numpy(dtype=int)
    n_total = len(predictions)
    n_events = int(predictions["y_true"].sum())
    prevalence = float(predictions["y_true"].mean())

    pooled_auc = float(row["pooled_auc"])
    ci_lower = float(row["ci_95_lower"])
    ci_upper = float(row["ci_95_upper"])
    pi_lower = float(row["pi_95_lower"])
    pi_upper = float(row["pi_95_upper"])
    tau2 = float(row.get("tau2_logit_auc", pd.NA)) if "tau2_logit_auc" in row.index else float("nan")
    i2 = float(row["i2_percent"])

    pr_site_estimates: list[float] = []
    pr_site_ses: list[float] = []
    brier_site_estimates: list[float] = []
    brier_site_ses: list[float] = []
    for site_idx, site in enumerate(SITE_ORDER):
        site_df = predictions.loc[predictions["test_site"] == site].copy()
        site_y = site_df["y_true"].to_numpy(dtype=int)
        site_p = site_df["proba_death"].to_numpy(dtype=float)

        if len(site_y) < 2:
            continue

        site_pr = float(average_precision_score(site_y, site_p))
        site_pr_var = float(_bootstrap_metric_sd(site_y, site_p, average_precision_score, n_boot=500, random_state=42 + site_idx)[1])
        site_brier = float(brier_score_loss(site_y, site_p))

        site_brier_error = (site_p - site_y) ** 2
        if len(site_brier_error) > 1:
            site_brier_var = float(np.var(site_brier_error, ddof=1) / len(site_brier_error))
        else:
            site_brier_var = float("nan")

        pr_site_estimates.append(site_pr)
        pr_site_ses.append(np.sqrt(site_pr_var) if np.isfinite(site_pr_var) else float("nan"))
        brier_site_estimates.append(site_brier)
        brier_site_ses.append(np.sqrt(site_brier_var) if np.isfinite(site_brier_var) else float("nan"))

    pr_pool = _pool_metric_hksj(np.asarray(pr_site_estimates, dtype=float), np.asarray(pr_site_ses, dtype=float))
    brier_pool = _pool_metric_hksj(
        np.asarray(brier_site_estimates, dtype=float),
        np.asarray(brier_site_ses, dtype=float),
    )
    summary_numeric = pd.DataFrame(
        [
            {
                "Metric": "ROC-AUC",
                "Estimate": pooled_auc,
                "CI lower": ci_lower,
                "CI upper": ci_upper,
                "PI lower": pi_lower,
                "PI upper": pi_upper,
                "Tau²": tau2,
                "I2 (%)": i2,
            },
            {
                "Metric": "PR-AUC (average precision)",
                "Estimate": pr_pool["pooled"],
                "CI lower": pr_pool["ci_lower"],
                "CI upper": pr_pool["ci_upper"],
                "PI lower": pr_pool["pi_lower"],
                "PI upper": pr_pool["pi_upper"],
                "Tau²": pr_pool["tau2"],
                "I2 (%)": pr_pool["i2"],
            },
            {
                "Metric": "Brier score",
                "Estimate": brier_pool["pooled"],
                "CI lower": brier_pool["ci_lower"],
                "CI upper": brier_pool["ci_upper"],
                "PI lower": brier_pool["pi_lower"],
                "PI upper": brier_pool["pi_upper"],
                "Tau²": brier_pool["tau2"],
                "I2 (%)": brier_pool["i2"],
            },
        ]
    )

    def _format_interval(lower: float, upper: float) -> str:
        if pd.isna(lower) or pd.isna(upper):
            return "—"
        return f"{_format_float(lower)} to {_format_float(upper)}"

    def _format_estimate_ci(estimate: float, lower: float, upper: float) -> str:
        if pd.isna(estimate) or pd.isna(lower) or pd.isna(upper):
            return "—"
        return f"{_format_float(estimate)} ({_format_float(lower)} to {_format_float(upper)})"

    summary_display = pd.DataFrame(
        {
            "Metric": summary_numeric["Metric"],
            "Pooled estimate (95% CI)": summary_numeric.apply(
                lambda row: _format_estimate_ci(row["Estimate"], row["CI lower"], row["CI upper"]),
                axis=1,
            ),
            "95% PI": summary_numeric.apply(
                lambda row: _format_interval(row["PI lower"], row["PI upper"]),
                axis=1,
            ),
            "Tau²": summary_numeric["Tau²"].apply(
                lambda value: _format_float(float(value)) if pd.notna(value) else "—"
            ),
            "I2 (%)": summary_numeric["I2 (%)"].apply(lambda value: f"{value:.1f}" if pd.notna(value) else "—"),
        }
    )

    footnote = f"Overall sample: N = {n_total}, deaths = {n_events}, prevalence = {_format_float(prevalence, 3)}."
    return summary_numeric, summary_display, footnote


def _plot_roc_by_site(predictions: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    baseline, = ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="#9CA3AF",
        linewidth=1.3,
        label="No-discrimination",
    )
    site_lines: dict[str, tuple[object, str]] = {}

    for site in SITE_ORDER:
        site_df = predictions.loc[predictions["test_site"] == site].copy()
        y_true = site_df["y_true"].to_numpy(dtype=int)
        y_prob = site_df["proba_death"].to_numpy(dtype=float)
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = roc_auc_score(y_true, y_prob)
        site_label = SITE_LABEL_MAP.get(site, site)
        site_legend_label = f"{site_label} (AUC {roc_auc:.3f})"

        line, = ax.plot(
            fpr,
            tpr,
            color=SITE_COLOR_MAP[site],
            linewidth=2.4,
            label=site_legend_label,
        )
        site_lines[site] = (line, site_legend_label)

    ax.set(
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0),
        xlabel="False positive rate",
        ylabel="True positive rate",
        title="Receiver operating characteristics curves for in-hospital mortality using the REMAIN model",
    )
    legend_handles, legend_labels = _build_grouped_legend(
        baseline_entry=(baseline, "No-discrimination"),
        site_entries=site_lines,
    )
    ax.legend(
        legend_handles,
        legend_labels,
        loc="lower right",
        frameon=True,
        framealpha=0.92,
        fontsize=9.0,
        handlelength=1.9,
        ncol=1,
    )
    fig.tight_layout()
    _save_figure(fig, "figure_1_external_roc_by_site")


def _plot_pr_by_site(predictions: pd.DataFrame) -> None:
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
    site_lines: dict[str, tuple[object, str]] = {}

    for site in SITE_ORDER:
        site_df = predictions.loc[predictions["test_site"] == site].copy()
        y_true = site_df["y_true"].to_numpy(dtype=int)
        y_prob = site_df["proba_death"].to_numpy(dtype=float)
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
        site_label = SITE_LABEL_MAP.get(site, site)
        site_legend_label = f"{site_label} (AP {pr_auc:.3f})"

        line, = ax.plot(
            recall,
            precision,
            color=SITE_COLOR_MAP[site],
            linewidth=2.4,
            label=site_legend_label,
        )
        site_lines[site] = (line, site_legend_label)

    ax.set(
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0),
        xlabel="Recall",
        ylabel="Precision",
        title="Precision-recall curves for in-hospital mortality using the REMAIN model",
    )
    legend_handles, legend_labels = _build_grouped_legend(
        baseline_entry=(baseline, f"Prevalence = {prevalence:.3f}"),
        site_entries=site_lines,
    )
    ax.legend(
        legend_handles,
        legend_labels,
        loc="upper right",
        frameon=True,
        framealpha=0.92,
        fontsize=9.0,
        handlelength=1.9,
        ncol=1,
    )
    fig.tight_layout()
    _save_figure(fig, "figure_1_external_pr_by_site")


def _plot_calibration(predictions: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.1, 6.4))
    baseline, = ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="#9CA3AF",
        linewidth=1.2,
        label="Perfect calibration",
    )
    site_lines: dict[str, tuple[object, str]] = {}

    for site in SITE_ORDER:
        site_df = predictions.loc[predictions["test_site"] == site].copy()
        pooled_y = site_df["y_true"].to_numpy(dtype=int)
        pooled_prob = site_df["proba_death"].to_numpy(dtype=float)
        frac_pos, mean_pred = calibration_curve(pooled_y, pooled_prob, n_bins=8, strategy="quantile")
        brier = brier_score_loss(pooled_y, pooled_prob)
        site_label = SITE_LABEL_MAP.get(site, site)
        site_legend_label = f"{site_label} (Brier {brier:.3f})"

        line, = ax.plot(
            mean_pred,
            frac_pos,
            color=SITE_COLOR_MAP[site],
            linewidth=2.3,
            marker="o",
            markersize=5,
            alpha=0.92,
            label=site_legend_label,
        )
        site_lines[site] = (line, site_legend_label)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Calibration plots for in-hospital mortality using the REMAIN model")
    legend_handles, legend_labels = _build_grouped_legend(
        baseline_entry=(baseline, "Perfect calibration"),
        site_entries=site_lines,
    )
    ax.legend(
        legend_handles,
        legend_labels,
        loc="lower right",
        frameon=True,
        framealpha=0.92,
        fontsize=9.0,
        handlelength=1.9,
        ncol=1,
    )
    fig.tight_layout()
    _save_figure(fig, "figure_2_external_calibration_by_site")


def _plot_cumulative_lift(predictions: pd.DataFrame, groups: int = 10) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 6.4))
    ax.plot([0, 1], [1, 1], linestyle="--", color="#9CA3AF", linewidth=1.3, label="Chance")

    for site in SITE_ORDER:
        site_df = predictions.loc[predictions["test_site"] == site].copy().sort_values("proba_death", ascending=False).reset_index(drop=True)
        y_true = site_df["y_true"].to_numpy(dtype=int)
        n = len(site_df)
        overall_event_rate = float(y_true.mean())
        fractions = [i / groups for i in range(1, groups + 1)]
        cumulative_lift: list[float] = []

        for fraction in fractions:
            cutoff = max(1, int((fraction * n)))
            cumulative_event_rate = float(y_true[:cutoff].mean())
            cumulative_lift.append(cumulative_event_rate / overall_event_rate)

        ax.plot(
            fractions,
            cumulative_lift,
            color=SITE_COLOR_MAP[site],
            linewidth=2.5,
            alpha=0.88,
            marker="o",
            markersize=5,
            label=SITE_LABEL_MAP.get(site, site),
        )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(bottom=0.9)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Patients screened by predicted risk rank")
    ax.set_ylabel("Cumulative lift vs overall death rate")
    ax.set_title("Cumulative lift for in-hospital mortality by site")
    ax.legend(loc="upper right", frameon=True, framealpha=0.95)

    fig.text(
        0.5,
        0.01,
        "Patients are ordered from highest to lowest predicted death risk in each site. Each point shows cumulative deaths captured (normalized by overall death rate).",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout()
    _save_figure(fig, "figure_4_internal_external_cumulative_lift")


def main() -> None:
    _apply_plot_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    predictions, summary_auc, fold_auc = _load_data()

    # Sort for stable artifacts and easier downstream use
    predictions = predictions.sort_values(["test_site", "row_index"]).reset_index(drop=True)
    fold_auc = fold_auc.sort_values(["model", "test_site"]).reset_index(drop=True)
    site_numeric, site_display = _build_site_level_ci_table(predictions)

    summary_numeric, summary_display, footnote = _build_meta_summary(predictions, summary_auc)
    # Keep concise, pooled RE-only meta-analysis summary for publication
    summary_numeric.to_csv(OUTPUT_DIR / "table_1_internal_external_summary.csv", index=False)
    site_numeric.to_csv(OUTPUT_DIR / "table_2_site_level_metrics_with_ci.csv", index=False)
    summary_auc.to_csv(OUTPUT_DIR / "combined_summary_auc.csv", index=False)
    fold_auc.to_csv(OUTPUT_DIR / "combined_fold_auc.csv", index=False)
    predictions.to_csv(OUTPUT_DIR / "combined_all_predictions.csv", index=False)
    _render_site_level_markdown_grouped(
        site_display,
        OUTPUT_DIR / "table_2_site_level_metrics_with_ci.md",
        title="REMAIN logistic (uncalibrated): Site-level validation metrics",
        footnote="Site-level 95% confidence intervals were estimated from 500 nonparametric bootstrap replicates.",
    )
    _render_markdown_table(
        summary_display,
        OUTPUT_DIR / "table_1_internal_external_summary.md",
        title="REMAIN logistic (uncalibrated): Random-effects meta-analysis",
        footnote=footnote,
    )

    _plot_roc_by_site(predictions)
    _plot_pr_by_site(predictions)
    _plot_calibration(predictions)
    _plot_cumulative_lift(predictions)
    _plot_site_forest_with_facets(predictions)

    print(f"Saved REMAIN logistic artifacts in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
