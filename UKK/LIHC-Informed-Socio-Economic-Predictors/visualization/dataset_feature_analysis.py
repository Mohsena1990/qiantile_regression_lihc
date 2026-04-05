from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_INPUT_PATH = Path("preprocessed_data_clean.csv")
DEFAULT_OUTPUT_DIR = Path("final_saved_models_catboost/dataset_analysis")
EXPORT_DPI = 900

BASE_STRUCTURAL = [
    "floor_area",
    "house_age",
    "dwelling_type",
    "insulation_count",
    "main_heating_source",
    "heating_control",
    "household_size",
    "children_present",
    "elderly_present",
]

CONTEXT_FEATURES = [
    "Country",
    "SettlementSize",
    "S6",
    "C2",
    "C3",
]

SOCIOECONOMIC_AUX = [
    "equivalized_income",
]

QR_FEATURES = [
    "floor_area",
    "house_age",
    "dwelling_type",
    "insulation_count",
    "main_heating_source",
    "household_size",
]

FEATURE_GROUPS = {
    "Base structural": BASE_STRUCTURAL,
    "Context": CONTEXT_FEATURES,
    "Socioeconomic": SOCIOECONOMIC_AUX,
    "Quantile-regression": QR_FEATURES,
}

NUMERIC_PRIORITY = [
    "floor_area",
    "house_age",
    "insulation_count",
    "household_size",
    "equivalized_income",
    "total_expenditure",
    "SettlementSize",
    "C2",
    "C3",
]
CATEGORICAL_PRIORITY = [
    "main_heating_source",
    "dwelling_type",
    "heating_control",
    "Country_name",
    "Country",
    "children_present",
    "elderly_present",
    "S6",
]


def _apply_plot_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": EXPORT_DPI,
            "savefig.dpi": EXPORT_DPI,
            "font.size": 12,
            "axes.titlesize": 16,
            "axes.labelsize": 13,
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }
    )


def _save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=1.8)
    fig.savefig(path, dpi=EXPORT_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _selected_columns(df: pd.DataFrame) -> list[str]:
    requested = set(["total_expenditure", "Country_name"])
    for cols in FEATURE_GROUPS.values():
        requested.update(cols)
    return [column for column in df.columns if column in requested]


def _feature_group_lookup() -> dict[str, str]:
    lookup: dict[str, str] = {}
    for group, columns in FEATURE_GROUPS.items():
        for column in columns:
            lookup[column] = group
    lookup["total_expenditure"] = "Target-related"
    lookup["Country_name"] = "Context"
    return lookup


def load_selected_dataset(input_path: Path) -> pd.DataFrame:
    preview_cols = pd.read_csv(input_path, nrows=1).columns.tolist()
    usecols = _selected_columns(pd.DataFrame(columns=preview_cols))
    return pd.read_csv(input_path, usecols=usecols)


def build_feature_summary(df: pd.DataFrame, output_dir: Path) -> Path:
    group_lookup = _feature_group_lookup()
    rows = []
    for column in df.columns:
        series = df[column]
        numeric_series = pd.to_numeric(series, errors="coerce")
        is_numeric = pd.api.types.is_numeric_dtype(series) or numeric_series.notna().sum() > 0
        row = {
            "feature": column,
            "group": group_lookup.get(column, "Other"),
            "dtype": str(series.dtype),
            "non_null": int(series.notna().sum()),
            "missing_pct": float(series.isna().mean() * 100),
            "unique_values": int(series.nunique(dropna=True)),
        }
        if is_numeric:
            row.update(
                {
                    "mean": float(numeric_series.mean()),
                    "std": float(numeric_series.std()),
                    "median": float(numeric_series.median()),
                    "min": float(numeric_series.min()),
                    "max": float(numeric_series.max()),
                }
            )
        else:
            top_values = series.astype(str).value_counts(dropna=True).head(3)
            row["top_values"] = " | ".join([f"{idx}:{val}" for idx, val in top_values.items()])
        rows.append(row)

    summary_df = pd.DataFrame(rows).sort_values(["group", "feature"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "selected_feature_summary.csv"
    summary_df.to_csv(output_path, index=False)
    return output_path


def plot_numeric_distributions(df: pd.DataFrame, output_dir: Path) -> Path:
    numeric_features = [column for column in NUMERIC_PRIORITY if column in df.columns]
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    axes = axes.ravel()

    for ax, column in zip(axes, numeric_features):
        series = pd.to_numeric(df[column], errors="coerce").dropna()
        ax.hist(series, bins=30, color="#1E88E5", edgecolor="black", alpha=0.85)
        ax.set_title(column.replace("_", " "))
        ax.set_xlabel(column)
        ax.set_ylabel("Count")

    for ax in axes[len(numeric_features):]:
        ax.axis("off")

    fig.suptitle("Distributions of Selected Numeric Features", fontsize=18, fontweight="bold", y=1.02)
    output_path = output_dir / "numeric_feature_distributions.png"
    _save_figure(fig, output_path)
    return output_path


def plot_categorical_distributions(df: pd.DataFrame, output_dir: Path) -> Path:
    categorical_features = [column for column in CATEGORICAL_PRIORITY if column in df.columns]
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.ravel()

    for ax, column in zip(axes, categorical_features):
        series = df[column].astype(str).replace("nan", pd.NA).dropna()
        counts = series.value_counts().head(8).sort_values()
        ax.barh(counts.index.astype(str), counts.values, color="#43A047", edgecolor="black")
        ax.set_title(column.replace("_", " "))
        ax.set_xlabel("Count")

    for ax in axes[len(categorical_features):]:
        ax.axis("off")

    fig.suptitle("Top Categories for Selected Context and Structural Features", fontsize=18, fontweight="bold", y=1.02)
    output_path = output_dir / "categorical_feature_distributions.png"
    _save_figure(fig, output_path)
    return output_path


def plot_missingness(df: pd.DataFrame, output_dir: Path) -> Path:
    missing_pct = (df.isna().mean() * 100).sort_values()
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(missing_pct.index, missing_pct.values, color="#FB8C00", edgecolor="black")
    ax.set_title("Missingness in Selected Analysis Features")
    ax.set_xlabel("Missing percentage")
    ax.set_ylabel("Feature")
    output_path = output_dir / "selected_feature_missingness.png"
    _save_figure(fig, output_path)
    return output_path


def plot_numeric_correlation(df: pd.DataFrame, output_dir: Path) -> Path:
    numeric_features = [column for column in NUMERIC_PRIORITY if column in df.columns]
    corr = df[numeric_features].apply(pd.to_numeric, errors="coerce").corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    image = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index)
    ax.set_title("Correlation Heatmap for Quantitative Selected Features")

    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.iat[i, j]:.2f}", ha="center", va="center", fontsize=9, color="#102027")

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Correlation", rotation=270, labelpad=18)
    output_path = output_dir / "numeric_feature_correlation.png"
    _save_figure(fig, output_path)
    return output_path


def plot_quantitative_relationships(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    output_paths: list[Path] = []

    relationship_specs = [
        ("equivalized_income", "total_expenditure", "household_size", "income_vs_expenditure_scatter.png"),
        ("floor_area", "total_expenditure", "equivalized_income", "floor_area_vs_expenditure_scatter.png"),
    ]

    for x_col, y_col, color_col, filename in relationship_specs:
        if not all(column in df.columns for column in [x_col, y_col, color_col]):
            continue

        plot_df = df[[x_col, y_col, color_col]].copy()
        for column in [x_col, y_col, color_col]:
            plot_df[column] = pd.to_numeric(plot_df[column], errors="coerce")
        plot_df = plot_df.dropna().sample(min(4000, len(plot_df)), random_state=42)

        fig, ax = plt.subplots(figsize=(12, 8))
        scatter = ax.scatter(
            plot_df[x_col],
            plot_df[y_col],
            c=plot_df[color_col],
            cmap="viridis",
            s=22,
            alpha=0.65,
            edgecolor="none",
        )
        ax.set_title(f"{y_col.replace('_', ' ')} vs {x_col.replace('_', ' ')}")
        ax.set_xlabel(x_col.replace("_", " "))
        ax.set_ylabel(y_col.replace("_", " "))
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label(color_col.replace("_", " "), rotation=270, labelpad=18)

        output_path = output_dir / filename
        _save_figure(fig, output_path)
        output_paths.append(output_path)

    return output_paths


def generate_dataset_analysis(input_path: Path = DEFAULT_INPUT_PATH, output_dir: Path = DEFAULT_OUTPUT_DIR) -> list[Path]:
    _apply_plot_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_selected_dataset(input_path)

    saved_paths = [
        build_feature_summary(df, output_dir),
        plot_numeric_distributions(df, output_dir),
        plot_categorical_distributions(df, output_dir),
        plot_missingness(df, output_dir),
        plot_numeric_correlation(df, output_dir),
    ]
    saved_paths.extend(plot_quantitative_relationships(df, output_dir))
    return saved_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate quantitative dataset analysis plots for selected feature groups.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH, help=f"Dataset CSV path (default: {DEFAULT_INPUT_PATH})")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    saved_paths = generate_dataset_analysis(args.input, args.output_dir)
    print("Saved dataset analysis artifacts:")
    for path in saved_paths:
        print(f" - {path}")


if __name__ == "__main__":
    main()
