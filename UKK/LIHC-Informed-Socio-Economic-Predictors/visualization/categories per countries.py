import os
import pandas as pd
import matplotlib.pyplot as plt


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
    "United Kingdom",
]

RISK_ORDER = ["No risk", "Income risk", "Expenditure risk", "Double risk"]

RISK_COLORS = {
    "No risk": "#2E8B57",            # green
    "Income risk": "#FFD54F",        # warm yellow
    "Expenditure risk": "#FB8C00",   # orange
    "Double risk": "#C62828",        # red
}


def plot_country_risk_distribution(
    df: pd.DataFrame,
    country_col: str = "Country",
    category_col: str = "risk_category",
    title: str | None = None,
    output_path: str = "country_risk_distribution.png",
    width: int = 16,
    height: int = 9,
    dpi: int = 600,
) -> None:
    """
    Plot stacked 100% bar chart of energy poverty risk categories per country.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing country and risk category columns.
    country_col : str
        Country column name.
    category_col : str
        Risk category column name.
    title : str | None
        Custom chart title. If None, a dynamic title is generated from output_path.
    output_path : str
        File name for saved figure.
    width : int
        Figure width.
    height : int
        Figure height.
    dpi : int
        Export dpi.
    """
    plot_df = df.copy()

    if country_col not in plot_df.columns:
        raise KeyError(f"Column '{country_col}' not found.")
    if category_col not in plot_df.columns:
        raise KeyError(f"Column '{category_col}' not found.")

    # Map numeric country codes to names only when needed
    if pd.api.types.is_numeric_dtype(plot_df[country_col]):
        plot_df[country_col] = plot_df[country_col].map(COUNTRY_MAPPING)

    # Drop rows with missing country/category
    plot_df = plot_df.dropna(subset=[country_col, category_col])

    # Build counts and percentages
    category_counts = (
        plot_df.groupby([country_col, category_col])
        .size()
        .unstack(fill_value=0)
        .reindex(index=COUNTRY_ORDER, fill_value=0)
    )

    # Ensure consistent category order
    for cat in RISK_ORDER:
        if cat not in category_counts.columns:
            category_counts[cat] = 0
    category_counts = category_counts[RISK_ORDER]

    category_perc = category_counts.div(category_counts.sum(axis=1), axis=0).fillna(0) * 100

    # Dynamic title
    if title is None:
        base_name = os.path.splitext(os.path.basename(output_path))[0]
        pretty_name = base_name.replace("_", " ").replace("-", " ").title()
        title = f"{pretty_name}: Energy Poverty Risk Distribution by Country"

    # High-quality style
    plt.rcParams.update({
        "font.size": 15,
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "axes.titlesize": 22,
        "axes.labelsize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "legend.title_fontsize": 15,
    })

    fig, ax = plt.subplots(figsize=(width, height))

    category_perc.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        color=[RISK_COLORS[c] for c in category_perc.columns],
        edgecolor="black",
        linewidth=0.8,
        width=0.78,
    )

    ax.set_title(title, pad=18, fontweight="bold")
    ax.set_xlabel("Country", fontweight="bold", labelpad=10)
    ax.set_ylabel("Households (%)", fontweight="bold", labelpad=10)
    ax.set_ylim(0, 100)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", fontweight="bold")
    ax.set_yticks(range(0, 101, 10))

    for label in ax.get_yticklabels():
        label.set_fontweight("bold")

    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.5)
    ax.set_axisbelow(True)

    legend = ax.legend(
        title="Risk Category",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=True,
    )
    legend.get_frame().set_linewidth(0.8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved: {output_path}")
    print("\nCounts:")
    print(category_counts)
    print("\nPercentages:")
    print(category_perc.round(2))


if __name__ == "__main__":
    file_configs = [
        {
            "path": r"C:\Users\Ilani\OneDrive\Desktop\EP\LIHC-Informed-Socio-Economic-Predictors\df_hqrtm_60.csv",
            "title": "HQRTM (q = 0.60): Energy Poverty Risk Distribution by Country",
            "output": "country_risk_distribution_hqrtm_60.png",
        },
        {
            "path": r"C:\Users\Ilani\OneDrive\Desktop\EP\LIHC-Informed-Socio-Economic-Predictors\df_hqrtm_65.csv",
            "title": "HQRTM (q = 0.65): Energy Poverty Risk Distribution by Country",
            "output": "country_risk_distribution_hqrtm_65.png",
        },
        {
            "path": r"C:\Users\Ilani\OneDrive\Desktop\EP\LIHC-Informed-Socio-Economic-Predictors\df_hqrtm_70.csv",
            "title": "HQRTM (q = 0.70): Energy Poverty Risk Distribution by Country",
            "output": "country_risk_distribution_hqrtm_70.png",
        },
        {
            "path": r"C:\Users\Ilani\OneDrive\Desktop\EP\LIHC-Informed-Socio-Economic-Predictors\df_lihc.csv",
            "title": "Traditional LIHC: Energy Poverty Risk Distribution by Country",
            "output": "country_risk_distribution_traditional_lihc.png",
        },
    ]

    for cfg in file_configs:
        df_plot = pd.read_csv(cfg["path"])
        print(f"\nProcessing: {cfg['path']}")
        print("Unique countries:", df_plot["Country"].unique())
        print("Unique categories:", df_plot["risk_category"].unique())

        plot_country_risk_distribution(
            df=df_plot,
            country_col="Country",
            category_col="risk_category",
            title=cfg["title"],
            output_path=cfg["output"],
            width=16,
            height=9,
            dpi=600,
        )