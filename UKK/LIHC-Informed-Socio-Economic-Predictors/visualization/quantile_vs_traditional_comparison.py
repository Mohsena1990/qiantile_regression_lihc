from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


COUNTRY_MAPPING = {
    1: "Bulgaria",
    2: "France",
    3: "Germany",
    4: "Hungary",
    5: "Italy",
    6: "Norway",
    7: "Poland",
    8: "Serbia",
    9: "Spain",
    10: "Ukraine",
    11: "United Kingdom",
}

COUNTRY_ORDER = [
    "Bulgaria",
    "France",
    "Germany",
    "Hungary",
    "Italy",
    "Norway",
    "Poland",
    "Serbia",
    "Spain",
    "Ukraine",
    "UK",
]

RISK_ORDER = ["No risk", "Income risk", "Expenditure risk", "Double risk"]

RISK_COLORS = {
    "No risk": "#2E8B57",
    "Income risk": "#FFD54F",
    "Expenditure risk": "#FB8C00",
    "Double risk": "#C62828",
}


def _prepare_risk_shares(
    df: pd.DataFrame,
    country_col: str = "Country",
    category_col: str = "risk_category",
) -> pd.DataFrame:
    share_df = df.copy()

    if pd.api.types.is_numeric_dtype(share_df[country_col]):
        share_df[country_col] = share_df[country_col].map(COUNTRY_MAPPING)

    share_df = share_df.dropna(subset=[country_col, category_col])

    category_counts = (
        share_df.groupby([country_col, category_col])
        .size()
        .unstack(fill_value=0)
        .reindex(index=COUNTRY_ORDER, fill_value=0)
    )

    for cat in RISK_ORDER:
        if cat not in category_counts.columns:
            category_counts[cat] = 0

    category_counts = category_counts[RISK_ORDER]
    return category_counts.div(category_counts.sum(axis=1), axis=0).fillna(0) * 100


def plot_quantile_vs_traditional_comparison(
    quantile_df: pd.DataFrame,
    traditional_df: pd.DataFrame,
    quantile_label: str = "HQRTM (q = 0.65)",
    output_path: str = "quantile_vs_traditional_lihc_comparison.png",
    country_col: str = "Country",
    category_col: str = "risk_category",
    dpi: int = 600,
) -> None:
    required_quantile_cols = {
        country_col,
        category_col,
        "equivalized_income",
        "total_expenditure",
        "expected_exp",
        "low_income",
    }
    required_traditional_cols = {
        country_col,
        category_col,
        "equivalized_income",
        "total_expenditure",
        "exp_threshold",
        "low_income",
    }

    missing_quantile = required_quantile_cols.difference(quantile_df.columns)
    missing_traditional = required_traditional_cols.difference(traditional_df.columns)

    if missing_quantile:
        raise KeyError(f"Missing quantile-regression columns: {sorted(missing_quantile)}")
    if missing_traditional:
        raise KeyError(f"Missing traditional LIHC columns: {sorted(missing_traditional)}")

    quantile_plot = quantile_df.copy()
    traditional_plot = traditional_df.copy()

    quantile_plot["expected_exp"] = pd.to_numeric(quantile_plot["expected_exp"], errors="coerce")
    traditional_plot["exp_threshold"] = pd.to_numeric(traditional_plot["exp_threshold"], errors="coerce")
    quantile_plot["total_expenditure"] = pd.to_numeric(quantile_plot["total_expenditure"], errors="coerce")
    traditional_plot["total_expenditure"] = pd.to_numeric(traditional_plot["total_expenditure"], errors="coerce")
    quantile_plot["equivalized_income"] = pd.to_numeric(quantile_plot["equivalized_income"], errors="coerce")
    traditional_plot["equivalized_income"] = pd.to_numeric(traditional_plot["equivalized_income"], errors="coerce")

    threshold_compare = pd.DataFrame(
        {
            "equivalized_income": quantile_plot["equivalized_income"],
            "actual_expenditure": quantile_plot["total_expenditure"],
            "quantile_threshold": quantile_plot["expected_exp"],
            "traditional_threshold": traditional_plot["exp_threshold"],
            "low_income": quantile_plot["low_income"].astype(str),
        }
    ).dropna()

    threshold_compare = threshold_compare.sort_values(
        ["equivalized_income", "actual_expenditure"]
    ).reset_index(drop=True)

    overall_compare = pd.DataFrame(
        {
            quantile_label: quantile_plot[category_col].value_counts(normalize=True).reindex(RISK_ORDER, fill_value=0) * 100,
            "Traditional LIHC": traditional_plot[category_col].value_counts(normalize=True).reindex(RISK_ORDER, fill_value=0) * 100,
        }
    )

    quantile_country_share = _prepare_risk_shares(quantile_plot, country_col, category_col)
    traditional_country_share = _prepare_risk_shares(traditional_plot, country_col, category_col)
    double_risk_delta = (
        quantile_country_share["Double risk"] - traditional_country_share["Double risk"]
    ).sort_values()

    plt.rcParams.update(
        {
            "font.size": 15,
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "axes.titlesize": 19,
            "axes.labelsize": 15,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 12,
            "legend.title_fontsize": 13,
        }
    )

    fig = plt.figure(figsize=(19, 12))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1], hspace=0.30, wspace=0.22)

    ax_thresholds = fig.add_subplot(gs[0, 0])
    x_idx = np.arange(len(threshold_compare))
    low_income_mask = threshold_compare["low_income"].str.lower().eq("true")

    ax_thresholds.scatter(
        x_idx[~low_income_mask],
        threshold_compare.loc[~low_income_mask, "actual_expenditure"],
        s=12,
        alpha=0.28,
        color="#90A4AE",
        label="Actual expenditure",
    )
    ax_thresholds.scatter(
        x_idx[low_income_mask],
        threshold_compare.loc[low_income_mask, "actual_expenditure"],
        s=22,
        alpha=0.78,
        color="#C62828",
        label="Actual expenditure (low-income households)",
    )
    ax_thresholds.plot(
        x_idx,
        threshold_compare["quantile_threshold"],
        color="#1565C0",
        linewidth=2.5,
        label=f"{quantile_label} adaptive threshold",
    )
    ax_thresholds.plot(
        x_idx,
        threshold_compare["traditional_threshold"],
        color="#EF6C00",
        linewidth=2.0,
        linestyle="--",
        label="Traditional LIHC fixed threshold",
    )
    ax_thresholds.set_title("How Quantile Regression Changes the High-Cost Threshold")
    ax_thresholds.set_xlabel("Households ranked by equivalized income")
    ax_thresholds.set_ylabel("Energy expenditure")
    ax_thresholds.grid(axis="y", linestyle="--", alpha=0.35)
    ax_thresholds.legend(loc="upper left", frameon=True, prop={"weight": "bold", "size": 11})

    expenditure_risk_delta = (
        quantile_country_share["Expenditure risk"] - traditional_country_share["Expenditure risk"]
    ).sort_values()

    ax_expenditure = fig.add_subplot(gs[0, 1])
    expenditure_colors = np.where(expenditure_risk_delta >= 0, "#1565C0", "#EF6C00")
    ax_expenditure.barh(
        expenditure_risk_delta.index,
        expenditure_risk_delta.values,
        color=expenditure_colors,
        edgecolor="black",
        linewidth=0.8,
    )
    ax_expenditure.axvline(0, color="black", linewidth=1.2)
    ax_expenditure.set_title("Country-Level Change in Expenditure-Risk Share")
    ax_expenditure.set_xlabel(f"Percentage-point change: {quantile_label} minus Traditional LIHC")
    ax_expenditure.grid(axis="x", linestyle="--", alpha=0.35)

    ax_overall = fig.add_subplot(gs[1, 0])
    x_pos = np.arange(len(RISK_ORDER))
    bar_width = 0.36
    ax_overall.bar(
        x_pos - bar_width / 2,
        overall_compare[quantile_label].values,
        width=bar_width,
        color=[RISK_COLORS[r] for r in RISK_ORDER],
        alpha=0.9,
        label=quantile_label,
        edgecolor="black",
        linewidth=0.7,
    )
    ax_overall.bar(
        x_pos + bar_width / 2,
        overall_compare["Traditional LIHC"].values,
        width=bar_width,
        color=[RISK_COLORS[r] for r in RISK_ORDER],
        alpha=0.42,
        label="Traditional LIHC",
        edgecolor="black",
        linewidth=0.7,
    )
    ax_overall.set_title("Overall Risk Category Mix")
    ax_overall.set_ylabel("Households (%)")
    ax_overall.set_xticks(x_pos)
    ax_overall.set_xticklabels(RISK_ORDER, rotation=15, ha="right")
    ax_overall.grid(axis="y", linestyle="--", alpha=0.35)

    method_legend = [
        Patch(facecolor="#263238", edgecolor="black", label=quantile_label),
        Patch(facecolor="#B0BEC5", edgecolor="black", label="Traditional LIHC"),
    ]
    category_legend = [
        Line2D([0], [0], color=RISK_COLORS[r], lw=10, label=r) for r in RISK_ORDER
    ]

    legend_methods = ax_overall.legend(
        handles=method_legend,
        title="Method",
        loc="upper left",
        frameon=True,
        prop={"weight": "bold", "size": 11},
    )
    ax_overall.add_artist(legend_methods)
    ax_overall.legend(
        handles=category_legend,
        title="Risk Category Color",
        loc="upper right",
        frameon=True,
        prop={"weight": "bold", "size": 11},
    )

    ax_country = fig.add_subplot(gs[1, 1])
    country_colors = np.where(double_risk_delta >= 0, "#1565C0", "#EF6C00")
    ax_country.barh(
        double_risk_delta.index,
        double_risk_delta.values,
        color=country_colors,
        edgecolor="black",
        linewidth=0.8,
    )
    ax_country.axvline(0, color="black", linewidth=1)
    ax_country.set_title("Country-Level Change in Double-Risk Share")
    ax_country.set_xlabel(f"Percentage-point change: {quantile_label} minus Traditional LIHC")
    ax_country.grid(axis="x", linestyle="--", alpha=0.35)

    for ax in [ax_thresholds, ax_expenditure, ax_overall, ax_country]:
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("bold")

    fig.suptitle(
        f"{quantile_label} vs Traditional LIHC: Threshold Logic and Risk Classification",
        fontsize=22,
        fontweight="bold",
        y=0.98,
    )

    plt.savefig(output_path, dpi=dpi, facecolor="white")
    plt.close(fig)
    print(f"Saved comparison dashboard: {output_path}")


def generate_all_quantile_comparisons(repo_root: Path, dpi: int = 900) -> None:
    traditional_df = pd.read_csv(repo_root / "df_lihc.csv", low_memory=False)
    quantile_configs = [
        ("df_hqrtm_60.csv", "HQRTM (q = 0.60)", "quantile_vs_traditional_lihc_q60.png"),
        ("df_hqrtm_65.csv", "HQRTM (q = 0.65)", "quantile_vs_traditional_lihc_q65.png"),
        ("df_hqrtm_70.csv", "HQRTM (q = 0.70)", "quantile_vs_traditional_lihc_q70.png"),
    ]

    for csv_name, label, output_name in quantile_configs:
        quantile_df = pd.read_csv(repo_root / csv_name, low_memory=False)
        plot_quantile_vs_traditional_comparison(
            quantile_df=quantile_df,
            traditional_df=traditional_df,
            quantile_label=label,
            output_path=str(repo_root / output_name),
            country_col="Country",
            category_col="risk_category",
            dpi=dpi,
        )


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[3]
    generate_all_quantile_comparisons(repo_root=repo_root, dpi=900)
