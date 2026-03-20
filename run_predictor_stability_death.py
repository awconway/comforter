from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.stats import f_oneway, ks_2samp


DATA_PATH = Path("/Users/ac/comforter/NewModelV3.csv")
OUTPUT_DIR = Path("/Users/ac/comforter/artifacts/predictor_stability_death")
TARGET = "DeathHospDisch"
SITE_COL = "Facility"

PREDICTORS = [
    "Age",
    "Sex",
    "ARP",
    "AdmitPrev24h",
    "SurgPrev24h",
    "ICUDischPrev24h",
    "METWithinPrev24h",
    "NEWS",
    "ICD",
    "SOFA",
]

BINARY_THRESH = 2


SITE_PREVALENCE_DELTA_WEAK = 0.06
SITE_MEAN_SD_DELTA_WEAK = 0.35
BINARY_SMD_WEAK = 0.35
BINARY_AUC_WEAK_LOWER = 0.47
BINARY_AUC_WEAK_UPPER = 0.53


def _ks_max_distance(values_by_site: dict[str, pd.Series]) -> float:
    sites = list(values_by_site.keys())
    max_distance = 0.0
    if len(sites) < 2:
        return 0.0

    for i in range(len(sites)):
        for j in range(i + 1, len(sites)):
            a = values_by_site[sites[i]].dropna().to_numpy()
            b = values_by_site[sites[j]].dropna().to_numpy()
            if len(a) < 3 or len(b) < 3:
                continue
            try:
                ks_stat = float(ks_2samp(a, b).statistic)
                max_distance = max(max_distance, ks_stat)
            except Exception:
                continue
    return max_distance


def _is_binary_like(series: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(series):
        return True
    if pd.api.types.is_numeric_dtype(series):
        unique = sorted(series.dropna().unique())
        return len(unique) <= BINARY_THRESH and all(v in {0, 1} for v in unique)
    unique_str = {str(v).strip().lower() for v in series.dropna().unique()}
    return unique_str <= {"0", "1", "true", "false", "yes", "no", "male", "female", "resuscitate", "dnar", "nil", "none", "one", "twoplus", "two_plus", "two plus", "2", "x"}


def _map_categorical_to_numeric(series: pd.Series) -> pd.Series:
    if series.dtype.kind in "biufc":
        return pd.to_numeric(series, errors="coerce")

    s = series.astype(str).str.strip().str.lower()
    if set(s.dropna().unique()) <= {"male", "female"}:
        return s.map({"male": 1, "female": 0}).astype(float)
    if set(s.dropna().unique()) <= {"resuscitate", "dnar"}:
        return s.map({"resuscitate": 1, "dnar": 0}).astype(float)
    if set(s.dropna().unique()) <= {"yes", "no", "1", "0", "y", "n", "true", "false"}:
        return s.map(
            {
                "yes": 1,
                "no": 0,
                "1": 1,
                "0": 0,
                "y": 1,
                "n": 0,
                "true": 1,
                "false": 0,
            }
        ).astype(float)

    if set(s.dropna().unique()) <= {"nil", "none", "no", "one", "twoplus", "2plus", "two_plus", "two plus", "1", "2", "0"}:
        return s.map(
            {
                "nil": 0,
                "none": 0,
                "0": 0,
                "no": 0,
                "one": 1,
                "1": 1,
                "twoplus": 2,
                "two_plus": 2,
                "two plus": 2,
                "2plus": 2,
                "2": 2,
            }
        ).astype(float)

    return pd.Series(pd.Categorical(s).codes, index=series.index).astype(float)


def _cohen_d_pairwise(values_by_site: dict[str, pd.Series]) -> float:
    sites = list(values_by_site.keys())
    max_d = 0.0
    for i in range(len(sites)):
        for j in range(i + 1, len(sites)):
            a = values_by_site[sites[i]].astype(float).dropna().to_numpy()
            b = values_by_site[sites[j]].astype(float).dropna().to_numpy()
            if len(a) < 2 or len(b) < 2:
                continue
            pooled_sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0)
            d = 0.0 if pooled_sd == 0 else abs(a.mean() - b.mean()) / pooled_sd
            if d > max_d:
                max_d = d
    return float(max_d)


def _pairwise_auc(values_by_site: dict[str, pd.Series], outcome_by_site: dict[str, pd.Series], y_true: pd.Series) -> list[float]:
    out: list[float] = []
    for site in values_by_site:
        y = outcome_by_site[site].to_numpy(dtype=int)
        p = values_by_site[site].to_numpy(dtype=float)
        if len(np.unique(y)) < 2 or len(np.unique(p)) < 2:
            continue
        from sklearn.metrics import roc_auc_score

        out.append(float(roc_auc_score(y, p)))
    return out


def _site_heterogeneity_feature(feature_vals: pd.Series, sites: pd.Series) -> dict[str, float]:
    tables = []
    groups = []
    by_site = {}
    for site, grp in feature_vals.groupby(sites):
        groups.append(grp.astype(float).to_numpy())
        by_site[site] = grp.astype(float)

    if len(groups) < 2:
        return {"site_sensitivity_smd_max": 0.0, "site_anova_p": 1.0}

    anova_p = 1.0
    try:
        anova_p = float(f_oneway(*groups, nan_policy="omit").pvalue)
    except Exception:
        anova_p = 1.0

    try:
        ks_p = float(_ks_max_distance(by_site) if groups else 1.0)
    except Exception:
        ks_p = 1.0
    return {
        "site_sensitivity_smd_max": _cohen_d_pairwise(by_site),
        "site_anova_p": anova_p,
        "site_ks_max": ks_p,
    }


def _site_counts(feature_vals: pd.Series, sites: pd.Series) -> dict[str, float]:
    by_site = feature_vals.groupby(sites).mean()
    return {
        "site_mean_min": float(by_site.min()),
        "site_mean_max": float(by_site.max()),
        "site_mean_range": float(by_site.max() - by_site.min()),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantify site-level predictor drift and univariate outcome signal."
    )
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data)
    df.columns = [c.lstrip("\ufeff") for c in df.columns]
    required = {TARGET, SITE_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    mapped = pd.DataFrame({col: _map_categorical_to_numeric(df[col]) for col in PREDICTORS})
    mapped[TARGET] = df[TARGET].astype(int)
    mapped[SITE_COL] = df[SITE_COL].astype(str)

    if mapped[TARGET].nunique() < 2:
        raise ValueError("Target must contain both outcomes.")

    from sklearn.metrics import roc_auc_score

    rows: list[dict[str, object]] = []
    dist_rows: list[dict[str, object]] = []
    for feature in PREDICTORS:
        x = mapped[feature]
        y = mapped[TARGET]
        site_col = mapped[SITE_COL]
        site_stats = _site_counts(x, site_col)
        hetero = _site_heterogeneity_feature(x, site_col)
        by_site = {
            site: x.loc[site_col == site]
            for site in sorted(site_col.dropna().unique())
        }
        by_site_y = {
            site: y.loc[site_col == site]
            for site in sorted(site_col.dropna().unique())
        }
        site_aucs = _pairwise_auc(by_site, by_site_y, y)

        overall_auc = (
            float(roc_auc_score(y, x))
            if len(x.dropna().unique()) > 1 and y.nunique() > 1
            else 0.5
        )
        is_binary = _is_binary_like(df[feature]) and x.nunique(dropna=True) <= BINARY_THRESH
        global_std = float(x.std(ddof=1, skipna=True)) if len(x.dropna()) > 1 else 0.0
        site_mean_range_std = (
            abs(site_stats["site_mean_max"] - site_stats["site_mean_min"]) / global_std
            if global_std and global_std > 0
            else 0.0
        )

        x_non_null = x.dropna()
        category_levels = int(x_non_null.nunique())
        rows.append(
            {
                "feature": feature,
                "n": int(x_non_null.shape[0]),
                "missing_pct": float((1 - (len(x_non_null) / len(df))) * 100.0),
                "n_levels_or_unique": category_levels,
                "global_mean_or_prevalence": float(x_non_null.mean()),
                "site_mean_min": site_stats["site_mean_min"],
                "site_mean_max": site_stats["site_mean_max"],
                "site_mean_range": site_stats["site_mean_range"],
                "site_mean_range_std": site_mean_range_std,
                "site_sensitivity_smd_max": hetero["site_sensitivity_smd_max"],
                "site_anova_p": hetero["site_anova_p"],
                "site_ks_max": hetero["site_ks_max"],
                "overall_auc": overall_auc,
                "site_auc_min": float(min(site_aucs)) if site_aucs else float("nan"),
                "site_auc_max": float(max(site_aucs)) if site_aucs else float("nan"),
                "site_auc_range": float(max(site_aucs) - min(site_aucs)) if site_aucs else float("nan"),
                "is_binary_like": bool(is_binary),
            }
        )

        by_site_numeric = by_site
        for site, vals in by_site_numeric.items():
            vals_clean = vals.dropna()
            if is_binary:
                level_vals = vals_clean.value_counts(dropna=True).to_dict()
                mode_val = next(iter(level_vals.keys()), None)
                mode_rate = float(level_vals.get(mode_val, 0)) / len(vals_clean) if len(vals_clean) else 0.0
            else:
                mode_rate = float("nan")
                mode_val = ""

            dist_rows.append(
                {
                    "feature": feature,
                    "site": site,
                    "n": int(len(vals_clean)),
                    "missing_pct": float((1 - (len(vals_clean) / len(vals))) * 100.0) if len(vals) else 0.0,
                    "mean_or_prevalence": float(vals_clean.mean()) if len(vals_clean) else float("nan"),
                    "median": float(vals_clean.median()) if len(vals_clean) else float("nan"),
                    "sd": float(vals_clean.std(ddof=1)) if len(vals_clean) > 1 else 0.0,
                    "min": float(vals_clean.min()) if len(vals_clean) else float("nan"),
                    "max": float(vals_clean.max()) if len(vals_clean) else float("nan"),
                    "p10": float(vals_clean.quantile(0.1)) if len(vals_clean) else float("nan"),
                    "p50": float(vals_clean.quantile(0.5)) if len(vals_clean) else float("nan"),
                    "p90": float(vals_clean.quantile(0.9)) if len(vals_clean) else float("nan"),
                    "mode": mode_val,
                    "mode_rate": mode_rate,
                }
            )

    report = pd.DataFrame(rows).sort_values(
        ["site_sensitivity_smd_max", "site_mean_range"], ascending=[False, False]
    )

    report["weak_signal"] = report["overall_auc"].between(BINARY_AUC_WEAK_LOWER, BINARY_AUC_WEAK_UPPER)
    report["site_sensitive"] = (
        (report["site_sensitivity_smd_max"] >= BINARY_SMD_WEAK)
        | (report["site_mean_range_std"] >= SITE_MEAN_SD_DELTA_WEAK)
        | (report["site_mean_range"] >= SITE_PREVALENCE_DELTA_WEAK)
    )
    report["recommend_remove"] = report["weak_signal"] & report["site_sensitive"]

    csv_path = args.output_dir / "predictor_stability_report.csv"
    report.to_csv(csv_path, index=False)
    (args.output_dir / "predictor_site_distributions.csv").write_text(
        pd.DataFrame(dist_rows).to_csv(index=False),
        encoding="utf-8",
    )

    md_path = args.output_dir / "predictor_stability_report.md"
    display = report.copy()
    for col in ["missing_pct", "global_mean_or_prevalence", "site_mean_min", "site_mean_max", "site_mean_range", "site_sensitivity_smd_max", "overall_auc", "site_auc_min", "site_auc_max", "site_auc_range", "site_anova_p"]:
        display[col] = display[col].map(lambda value: f"{value:.3f}" if pd.notna(value) else "")
    for col in ["site_mean_range_std", "site_ks_max"]:
        if col in display.columns:
            display[col] = display[col].map(lambda value: f"{value:.3f}" if pd.notna(value) else "")
    display = display[
        [
            "feature",
            "missing_pct",
            "n_levels_or_unique",
            "global_mean_or_prevalence",
            "site_mean_range",
            "site_mean_range_std",
            "site_sensitivity_smd_max",
            "site_ks_max",
            "site_anova_p",
            "overall_auc",
            "site_auc_range",
            "weak_signal",
            "site_sensitive",
            "recommend_remove",
        ]
    ]
    lines = ["| " + " | ".join(display.columns) + " |", "| " + " | ".join(["---"] * len(display.columns)) + " |"]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in display.columns) + " |")
    md_path.write_text("\\n".join(lines) + "\\n", encoding="utf-8")

    settings = {
        "data_path": str(args.data),
        "output_dir": str(args.output_dir),
        "target": TARGET,
        "facility_col": SITE_COL,
        "predictors": PREDICTORS,
        "recommended_for_removal": report.loc[report["recommend_remove"], "feature"].tolist(),
        "summary": {
            "n_features": len(report),
            "n_recommended": int(report["recommend_remove"].sum()),
        },
    }
    (args.output_dir / "predictor_stability_metadata.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")

    print(f"Saved predictor stability report: {csv_path}")
    print(f"Saved predictor stability report (markdown): {md_path}")
    print("Recommended removals:", settings["recommended_for_removal"])


if __name__ == "__main__":
    main()
