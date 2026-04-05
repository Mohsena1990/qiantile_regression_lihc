from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_SUMMARY_PATH = Path("final_saved_models_catboost/all_results_summary.csv")
DEFAULT_OUTPUT_DIR = Path("final_saved_models_catboost/fit_diagnostics")
EXPORT_DPI = 900
REQUIRED_TRAIN_COLUMNS = {
    "train_accuracy_mean",
    "train_macro_f1_mean",
    "train_weighted_f1_mean",
    "validation_accuracy_mean",
    "validation_macro_f1_mean",
    "validation_weighted_f1_mean",
    "final_train_accuracy",
    "final_train_macro_f1",
    "final_train_weighted_f1",
}

DATASET_LABELS = {
    "traditional_lihc": "Traditional LIHC",
    "hqrtm_q60": "HQRTM q=0.60",
    "hqrtm_q65": "HQRTM q=0.65",
    "hqrtm_q70": "HQRTM q=0.70",
}


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
            "legend.fontsize": 11,
        }
    )


def _save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=1.8)
    fig.savefig(path, dpi=EXPORT_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def load_summary(summary_path: Path) -> pd.DataFrame:
    summary = pd.read_csv(summary_path)
    missing = REQUIRED_TRAIN_COLUMNS.difference(summary.columns)
    if missing:
        raise ValueError(
            "Training diagnostics require the updated training pipeline outputs. "
            f"Missing columns in {summary_path}: {sorted(missing)}. "
            "Please rerun catboost_run_preprocessed.py to regenerate all_results_summary.csv."
        )
    return summary


def best_rows(summary: pd.DataFrame) -> pd.DataFrame:
    return (
        summary.sort_values(["test_macro_f1", "test_accuracy"], ascending=[False, False])
        .groupby("dataset", as_index=False)
        .first()
        .sort_values("dataset")
        .reset_index(drop=True)
    )


def _run_dir(row: pd.Series) -> Path:
    return Path("final_saved_models_catboost") / str(row["dataset"]) / str(row["model_name"]) / f"k{int(row['k'])}"


def plot_train_validation_test_bars(summary: pd.DataFrame, output_dir: Path) -> Path:
    selected = best_rows(summary)
    metric_specs = [
        ("Accuracy", "final_train_accuracy", "validation_accuracy_mean", "test_accuracy"),
        ("Macro F1", "final_train_macro_f1", "validation_macro_f1_mean", "test_macro_f1"),
        ("Weighted F1", "final_train_weighted_f1", "validation_weighted_f1_mean", "test_weighted_f1"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(18, 12), sharey=True)
    axes = axes.ravel()

    for idx, (_, row) in enumerate(selected.iterrows()):
        ax = axes[idx]
        x = list(range(len(metric_specs)))
        train_values = [row[train_col] for _, train_col, _, _ in metric_specs]
        val_values = [row[val_col] for _, _, val_col, _ in metric_specs]
        test_values = [row[test_col] for _, _, _, test_col in metric_specs]
        width = 0.24

        ax.bar([v - width for v in x], train_values, width=width, color="#546E7A", edgecolor="black", label="Train")
        ax.bar(x, val_values, width=width, color="#90A4AE", edgecolor="black", label="Validation")
        ax.bar([v + width for v in x], test_values, width=width, color="#C62828", edgecolor="black", label="Test")
        ax.set_xticks(x)
        ax.set_xticklabels([name for name, *_ in metric_specs])
        ax.set_ylim(0, 1.0)
        ax.set_title(f"{DATASET_LABELS.get(str(row['dataset']), str(row['dataset']))}\n{row['model_name']} | k={int(row['k'])}")

    for ax in axes[len(selected):]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.01), ncol=3, frameon=True)
    fig.suptitle("Train vs Validation vs Test Metrics for Best Configurations", fontsize=18, fontweight="bold", y=1.02)
    fig.subplots_adjust(bottom=0.13)

    output_path = output_dir / "train_validation_test_best_configs.png"
    _save_figure(fig, output_path)
    return output_path


def plot_learning_curves(summary: pd.DataFrame, output_dir: Path) -> Path:
    selected = best_rows(summary)
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), sharey=False)
    axes = axes.ravel()

    for idx, (_, row) in enumerate(selected.iterrows()):
        ax = axes[idx]
        curve_path = _run_dir(row) / "learning_curve_summary.csv"
        curve_df = pd.read_csv(curve_path)
        metric_names = curve_df["metric"].dropna().unique().tolist()
        chosen_metric = next((metric for metric in metric_names if "Accuracy" in metric), metric_names[0])
        plot_df = curve_df[curve_df["metric"] == chosen_metric].copy()

        for split_name, color in [("learn", "#546E7A"), ("validation", "#C62828")]:
            split_df = plot_df[plot_df["split"] == split_name]
            if split_df.empty:
                continue
            ax.plot(split_df["iteration"], split_df["mean"], color=color, label=split_name.title())
            if "std" in split_df.columns:
                lower = split_df["mean"] - split_df["std"].fillna(0)
                upper = split_df["mean"] + split_df["std"].fillna(0)
                ax.fill_between(split_df["iteration"], lower, upper, color=color, alpha=0.15)

        ax.axhline(row["test_accuracy"], color="#2E7D32", linestyle="--", linewidth=2, label="Test accuracy")
        ax.set_title(f"{DATASET_LABELS.get(str(row['dataset']), str(row['dataset']))}\n{row['model_name']} | k={int(row['k'])}")
        ax.set_xlabel("Boosting iteration")
        ax.set_ylabel(chosen_metric)

    for ax in axes[len(selected):]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.01), ncol=3, frameon=True)
    fig.suptitle("Learning Curves with Test Reference Line for Best Configurations", fontsize=18, fontweight="bold", y=1.02)
    fig.subplots_adjust(bottom=0.13)

    output_path = output_dir / "learning_curves_best_configs.png"
    _save_figure(fig, output_path)
    return output_path


def generate_fit_diagnostics(summary_path: Path = DEFAULT_SUMMARY_PATH, output_dir: Path = DEFAULT_OUTPUT_DIR) -> list[Path]:
    _apply_plot_style()
    summary = load_summary(summary_path)
    return [
        plot_train_validation_test_bars(summary, output_dir),
        plot_learning_curves(summary, output_dir),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate train/validation/test diagnostics from updated CatBoost outputs.")
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY_PATH, help=f"Summary CSV path (default: {DEFAULT_SUMMARY_PATH})")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    saved_paths = generate_fit_diagnostics(args.summary, args.output_dir)
    print("Saved fit-diagnostic plots:")
    for path in saved_paths:
        print(f" - {path}")


if __name__ == "__main__":
    main()
