from __future__ import annotations

from pathlib import Path
import re

import matplotlib.pyplot as plt
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
    "United Kingdom",
]

BURDEN_LEVELS = [0.10, 0.15, 0.20]
BURDEN_COLORS = {
    0.10: "#2A9D8F",
    0.15: "#E9C46A",
    0.20: "#E76F51",
}
METHOD_COLORS = {
    "Traditional LIHC": "#264653",
    "HQRTM q=0.60": "#2A9D8F",
    "HQRTM q=0.65": "#577590",
    "HQRTM q=0.70": "#E76F51",
}


def _extract_quantile_label(path: Path) -> str:
    match = re.search(r"(\d+)", path.stem)
    if not match:
        raise ValueError(f"Could not infer quantile from file name: {path.name}")
    return f"HQRTM q={int(match.group(1)) / 100:.2f}"


def _normalize_country(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.map(COUNTRY_MAPPING).fillna(series.astype(str))
    return series.astype(str)


def _load_frame(csv_path: Path, method: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False).copy()

    threshold_col = "exp_threshold" if "exp_threshold" in df.columns else "expected_exp"
    required = {"Country", "equivalized_income", "total_expenditure", threshold_col}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"{csv_path.name} is missing required columns: {sorted(missing)}")

    df["method"] = method
    df["Country_name"] = _normalize_country(df["Country"])
    df["equivalized_income"] = pd.to_numeric(df["equivalized_income"], errors="coerce")
    df["total_expenditure"] = pd.to_numeric(df["total_expenditure"], errors="coerce")
    df["threshold_value"] = pd.to_numeric(df[threshold_col], errors="coerce")
    df = df[(df["equivalized_income"] > 0) & (df["total_expenditure"] >= 0)].copy()
    df["energy_burden_pct"] = 100 * df["total_expenditure"] / df["equivalized_income"]
    df["threshold_burden_pct"] = 100 * df["threshold_value"] / df["equivalized_income"]
    return df


def load_methods(repo_root: Path) -> pd.DataFrame:
    frames = [_load_frame(repo_root / "df_lihc.csv", "Traditional LIHC")]
    for csv_path in sorted(repo_root.glob("df_hqrtm_*.csv")):
        frames.append(_load_frame(csv_path, _extract_quantile_label(csv_path)))
    return pd.concat(frames, ignore_index=True)


def _method_order(combined_df: pd.DataFrame) -> list[str]:
    methods = sorted(set(combined_df["method"]))
    return ["Traditional LIHC"] + [m for m in methods if m != "Traditional LIHC"]


def _sample_method_frame(method_df: pd.DataFrame, max_points: int = 2200) -> pd.DataFrame:
    if len(method_df) <= max_points:
        return method_df
    return method_df.sample(n=max_points, random_state=42)


def _country_burden_summary(combined_df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for burden in BURDEN_LEVELS:
        summary = (
            combined_df.assign(above_burden=combined_df["energy_burden_pct"] >= burden * 100)
            .groupby(["method", "Country_name"])["above_burden"]
            .mean()
            .mul(100)
            .reset_index(name="share_pct")
        )
        summary["burden_label"] = f">= {int(burden * 100)}%"
        frames.append(summary)
    result = pd.concat(frames, ignore_index=True)
    result["Country_name"] = pd.Categorical(result["Country_name"], categories=COUNTRY_ORDER, ordered=True)
    return result.sort_values(["burden_label", "Country_name", "method"])


def _method_burden_summary(combined_df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for burden in BURDEN_LEVELS:
        summary = (
            combined_df.assign(above_burden=combined_df["energy_burden_pct"] >= burden * 100)
            .groupby("method")["above_burden"]
            .mean()
            .mul(100)
            .reset_index(name="share_pct")
        )
        summary["burden_label"] = f">= {int(burden * 100)}%"
        frames.append(summary)
    return pd.concat(frames, ignore_index=True)


def plot_percentage_lines_dashboard(
    combined_df: pd.DataFrame,
    output_path: Path,
    summary_path: Path | None = None,
    dpi: int = 500,
) -> None:
    method_order = _method_order(combined_df)
    country_summary = _country_burden_summary(combined_df)
    method_summary = _method_burden_summary(combined_df)

    income_max = combined_df["equivalized_income"].quantile(0.98)
    expenditure_max = combined_df["total_expenditure"].quantile(0.98)
    x_line = np.linspace(0, income_max, 200)

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )

    fig = plt.figure(figsize=(20, 13))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.15], hspace=0.28, wspace=0.18)

    scatter_axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
    ]

    for ax, method in zip(scatter_axes, method_order):
        method_df = combined_df[combined_df["method"] == method].copy()
        sampled = _sample_method_frame(method_df)

        ax.scatter(
            sampled["equivalized_income"],
            sampled["total_expenditure"],
            s=10,
            alpha=0.22,
            color=METHOD_COLORS.get(method, "#3A3A3A"),
            edgecolors="none",
        )

        profile = (
            method_df.dropna(subset=["equivalized_income", "threshold_value"])
            .assign(income_bin=lambda df: pd.qcut(df["equivalized_income"], q=12, duplicates="drop"))
            .groupby("income_bin", observed=False)[["equivalized_income", "threshold_value"]]
            .median()
            .reset_index(drop=True)
        )
        ax.plot(
            profile["equivalized_income"],
            profile["threshold_value"],
            color="#1D3557",
            linewidth=2.5,
            label="Method threshold",
        )

        for burden in BURDEN_LEVELS:
            ax.plot(
                x_line,
                burden * x_line,
                linestyle="--",
                linewidth=1.5,
                color=BURDEN_COLORS[burden],
                label=f"{int(burden * 100)}% burden",
            )

        ax.set_xlim(0, income_max)
        ax.set_ylim(0, expenditure_max)
        ax.set_title(method)
        ax.set_xlabel("Equivalized income")
        ax.set_ylabel("Energy expenditure")
        ax.grid(linestyle="--", alpha=0.25)

    handles, labels = scatter_axes[0].get_legend_handles_labels()
    scatter_axes[0].legend(handles[:4], labels[:4], loc="upper left", frameon=True)

    ax_method = fig.add_subplot(gs[2, 0])
    burden_labels = [f">= {int(b * 100)}%" for b in BURDEN_LEVELS]
    x = np.arange(len(burden_labels))
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(method_order))
    for offset, method in zip(offsets, method_order):
        values = (
            method_summary[method_summary["method"] == method]
            .set_index("burden_label")
            .reindex(burden_labels)["share_pct"]
            .values
        )
        ax_method.bar(
            x + offset,
            values,
            width=width,
            label=method,
            color=METHOD_COLORS.get(method, "#3A3A3A"),
            edgecolor="black",
            linewidth=0.6,
        )
    ax_method.set_xticks(x)
    ax_method.set_xticklabels(burden_labels)
    ax_method.set_ylabel("Households (%)")
    ax_method.set_title("Overall Share Above Each Percentage Line")
    ax_method.grid(axis="y", linestyle="--", alpha=0.25)
    ax_method.legend(loc="best", frameon=True)

    heatmap_axes = gs[2, 1].subgridspec(1, len(BURDEN_LEVELS), wspace=0.28)
    for idx, burden in enumerate(BURDEN_LEVELS):
        ax = fig.add_subplot(heatmap_axes[0, idx])
        label = f">= {int(burden * 100)}%"
        subset = country_summary[country_summary["burden_label"] == label]
        matrix = (
            subset.pivot(index="Country_name", columns="method", values="share_pct")
            .reindex(index=COUNTRY_ORDER, columns=method_order)
        )
        image = ax.imshow(matrix.to_numpy(dtype=float), cmap="YlOrRd", aspect="auto")
        ax.set_title(label)
        ax.set_xticks(np.arange(len(method_order)))
        ax.set_xticklabels(method_order, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(COUNTRY_ORDER)))
        ax.set_yticklabels(COUNTRY_ORDER if idx == 0 else [])

        values = matrix.to_numpy(dtype=float)
        vmax = np.nanmax(values) if np.isfinite(values).any() else 0
        for row in range(values.shape[0]):
            for col in range(values.shape[1]):
                value = values[row, col]
                if np.isnan(value):
                    continue
                text_color = "white" if vmax and value >= 0.55 * vmax else "black"
                ax.text(col, row, f"{value:.1f}", ha="center", va="center", fontsize=7, color=text_color)

        fig.colorbar(image, ax=ax, fraction=0.05, pad=0.04)

    fig.suptitle(
        "Energy-Burden Percentage Lines: Traditional LIHC vs HQRTM Quantiles",
        fontsize=20,
        fontweight="bold",
        y=0.98,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        country_summary.to_csv(summary_path, index=False)

    print(f"Saved percentage-line dashboard: {output_path}")
    if summary_path is not None:
        print(f"Saved burden summary: {summary_path}")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    combined_df = load_methods(repo_root)
    output_dir = Path(__file__).resolve().parent
    plot_percentage_lines_dashboard(
        combined_df=combined_df,
        output_path=output_dir / "quantile_percentage_vs_traditional_dashboard.png",
        summary_path=output_dir / "quantile_percentage_vs_traditional_summary.csv",
        dpi=500,
    )


if __name__ == "__main__":
    main()
