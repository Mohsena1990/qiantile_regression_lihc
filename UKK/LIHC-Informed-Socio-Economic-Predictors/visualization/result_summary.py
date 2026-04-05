from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


DEFAULT_INPUT_PATH = Path("final_saved_models_catboost/all_results_summary.csv")
DEFAULT_OUTPUT_DIR = Path("final_saved_models_catboost/result_summary_plots")
EXPORT_DPI = 1000

REQUIRED_COLUMNS = {
    "dataset",
    "model_name",
    "k",
    "cv_accuracy_mean",
    "cv_accuracy_std",
    "cv_macro_f1_mean",
    "cv_macro_f1_std",
    "cv_weighted_f1_mean",
    "cv_weighted_f1_std",
    "test_accuracy",
    "test_macro_f1",
    "test_weighted_f1",
    "test_macro_precision",
    "test_macro_recall",
}

METRIC_LABELS = {
    "test_accuracy": "Test Accuracy",
    "test_macro_f1": "Test Macro F1",
    "test_weighted_f1": "Test Weighted F1",
    "test_macro_precision": "Test Macro Precision",
    "test_macro_recall": "Test Macro Recall",
    "cv_accuracy_mean": "CV Accuracy Mean",
    "cv_macro_f1_mean": "CV Macro F1 Mean",
    "cv_weighted_f1_mean": "CV Weighted F1 Mean",
    "accuracy": "Train Proxy Accuracy",
    "macro_f1": "Train Proxy Macro F1",
    "weighted_f1": "Train Proxy Weighted F1",
}

DATASET_ORDER = [
    "traditional_lihc",
    "hqrtm_q60",
    "hqrtm_q65",
    "hqrtm_q70",
]

MODEL_ORDER = [
    "B1_structural",
    "B2_structural_socioeconomic",
    "B3_structural_context",
    "B4_structural_context_socioeconomic",
]

DATASET_LABELS = {
    "traditional_lihc": "Traditional LIHC",
    "hqrtm_q60": "HQRTM q=0.60",
    "hqrtm_q65": "HQRTM q=0.65",
    "hqrtm_q70": "HQRTM q=0.70",
}

MODEL_LABELS = {
    "B1_structural": "B1 Structural",
    "B2_structural_socioeconomic": "B2 Structural + Socioeconomic",
    "B3_structural_context": "B3 Structural + Context",
    "B4_structural_context_socioeconomic": "B4 Structural + Context + Socioeconomic",
}

SHORT_MODEL_LABELS = {
    "B1_structural": "B1",
    "B2_structural_socioeconomic": "B2",
    "B3_structural_context": "B3",
    "B4_structural_context_socioeconomic": "B4",
}

DATASET_COLORS = {
    "traditional_lihc": "#455A64",
    "hqrtm_q60": "#1E88E5",
    "hqrtm_q65": "#43A047",
    "hqrtm_q70": "#FB8C00",
}

MODEL_COLORS = {
    "B1_structural": "#8E6C8A",
    "B2_structural_socioeconomic": "#4E79A7",
    "B3_structural_context": "#F28E2B",
    "B4_structural_context_socioeconomic": "#59A14F",
}

MODEL_MARKERS = {
    "B1_structural": "o",
    "B2_structural_socioeconomic": "s",
    "B3_structural_context": "^",
    "B4_structural_context_socioeconomic": "D",
}

PHASE_COLORS = {
    "train_proxy": "#546E7A",
    "test": "#C62828",
}

CLASS_LABELS = ["Double risk", "Expenditure risk", "Income risk", "No risk"]


def load_results(csv_path: Path) -> pd.DataFrame:
    results = pd.read_csv(csv_path)
    missing_columns = REQUIRED_COLUMNS.difference(results.columns)
    if missing_columns:
        raise KeyError(
            f"Missing required columns in {csv_path}: {sorted(missing_columns)}"
        )

    numeric_cols = [column for column in REQUIRED_COLUMNS if column not in {"dataset", "model_name"}]
    for column in numeric_cols:
        results[column] = pd.to_numeric(results[column], errors="coerce")

    results = results.dropna(subset=["dataset", "model_name", "k"])
    results["k"] = results["k"].astype(int)

    dataset_order = [name for name in DATASET_ORDER if name in set(results["dataset"])]
    dataset_order += sorted(set(results["dataset"]) - set(dataset_order))
    model_order = [name for name in MODEL_ORDER if name in set(results["model_name"])]
    model_order += sorted(set(results["model_name"]) - set(model_order))

    results["dataset"] = pd.Categorical(results["dataset"], categories=dataset_order, ordered=True)
    results["model_name"] = pd.Categorical(results["model_name"], categories=model_order, ordered=True)
    return results.sort_values(["dataset", "model_name", "k"]).reset_index(drop=True)


def _apply_plot_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": EXPORT_DPI,
            "savefig.dpi": EXPORT_DPI,
            "font.size": 15,
            "font.weight": "bold",
            "axes.titlesize": 22,
            "axes.labelsize": 17,
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "legend.fontsize": 13,
            "legend.title_fontsize": 14,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "lines.linewidth": 2.8,
            "axes.linewidth": 1.4,
        }
    )


def _save_figure(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=2.0)
    fig.savefig(output_path, dpi=EXPORT_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _best_rows_by_dataset(results: pd.DataFrame) -> pd.DataFrame:
    return (
        results.sort_values(["test_macro_f1", "test_accuracy"], ascending=[False, False])
        .groupby("dataset", observed=True, as_index=False)
        .first()
        .sort_values("dataset")
        .reset_index(drop=True)
    )


def _artifact_path(row: pd.Series, filename: str) -> Path:
    return (
        Path("final_saved_models_catboost")
        / str(row["dataset"])
        / str(row["model_name"])
        / f"k{int(row['k'])}"
        / filename
    )


def _load_confusion_matrix(row: pd.Series) -> pd.DataFrame:
    matrix_path = _artifact_path(row, "confusion_matrix.csv")
    cm = pd.read_csv(matrix_path)
    row_label_col = cm.columns[0]
    cm = cm.rename(columns={row_label_col: "true_label"})
    cm["true_label"] = cm["true_label"].astype(str).str.replace("true_", "", regex=False)
    cm.columns = ["true_label"] + [str(col).replace("pred_", "") for col in cm.columns[1:]]
    cm = cm.set_index("true_label")
    cm = cm.reindex(index=CLASS_LABELS, columns=CLASS_LABELS)
    return cm.fillna(0)


def _load_cv_results(row: pd.Series) -> pd.DataFrame:
    cv_path = _artifact_path(row, "cv_results.csv")
    cv_df = pd.read_csv(cv_path)
    cv_df["fold"] = pd.to_numeric(cv_df["fold"], errors="coerce").astype(int)
    return cv_df.sort_values("fold").reset_index(drop=True)


def plot_metric_trends_by_model(results: pd.DataFrame, output_dir: Path) -> Path:
    metric = "test_macro_f1"
    datasets = list(results["dataset"].cat.categories)
    models = list(results["model_name"].cat.categories)

    fig, axes = plt.subplots(2, 2, figsize=(20, 14), sharex=True, sharey=True)
    axes = axes.ravel()

    for idx, model_name in enumerate(models):
        ax = axes[idx]
        model_df = results[results["model_name"] == model_name]

        for dataset in datasets:
            dataset_df = model_df[model_df["dataset"] == dataset].sort_values("k")
            if dataset_df.empty:
                continue

            ax.plot(
                dataset_df["k"],
                dataset_df[metric],
                marker="o",
                markersize=8,
                color=DATASET_COLORS.get(str(dataset), "#546E7A"),
                label=DATASET_LABELS.get(str(dataset), str(dataset)),
            )

        ax.set_title(MODEL_LABELS.get(str(model_name), str(model_name)), pad=14)
        ax.set_xlabel("k folds")
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.set_xticks(sorted(model_df["k"].dropna().unique()))
        ax.set_ylim(0.30, min(1.0, results[metric].max() + 0.08))
        ax.grid(axis="y", linestyle="--", alpha=0.35)

    for ax in axes[len(models):]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=2,
            frameon=True,
            title="Dataset",
        )
    fig.suptitle("Test Macro F1 Across k for Each Feature Set", fontsize=25, fontweight="bold", y=1.02)
    fig.subplots_adjust(bottom=0.16)

    output_path = output_dir / "metric_trends_test_macro_f1.png"
    _save_figure(fig, output_path)
    return output_path


def plot_best_model_per_dataset(results: pd.DataFrame, output_dir: Path) -> Path:
    score_col = "test_macro_f1"
    best_rows = _best_rows_by_dataset(results)

    fig, ax = plt.subplots(figsize=(15, 9))
    x_labels = [DATASET_LABELS.get(str(value), str(value)) for value in best_rows["dataset"]]
    bars = ax.bar(
        x_labels,
        best_rows[score_col],
        color=[MODEL_COLORS.get(str(model), "#546E7A") for model in best_rows["model_name"]],
        edgecolor="black",
        linewidth=1.3,
    )

    for bar, (_, row) in zip(bars, best_rows.iterrows()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.012,
            f"{SHORT_MODEL_LABELS.get(str(row['model_name']), str(row['model_name']))} | k={int(row['k'])}\nF1={row[score_col]:.3f}",
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
        )

    model_handles = [
        Line2D([0], [0], marker="s", linestyle="", markersize=12, markerfacecolor=color, markeredgecolor="black", label=SHORT_MODEL_LABELS.get(model, model))
        for model, color in MODEL_COLORS.items()
        if model in set(best_rows["model_name"].astype(str))
    ]
    if model_handles:
        ax.legend(handles=model_handles, loc="upper left", frameon=True, title="Winning model")

    ax.set_title("Best Configuration per Dataset by Test Macro F1", pad=16)
    ax.set_xlabel("Dataset")
    ax.set_ylabel(METRIC_LABELS[score_col])
    ax.set_ylim(0, min(1.0, best_rows[score_col].max() + 0.20))
    ax.tick_params(axis="x", rotation=8)

    output_path = output_dir / "best_configuration_by_dataset.png"
    _save_figure(fig, output_path)
    return output_path


def plot_metric_heatmaps(results: pd.DataFrame, output_dir: Path) -> list[Path]:
    output_paths: list[Path] = []
    selected_metrics = ["test_accuracy", "test_macro_f1", "test_weighted_f1"]

    for metric in selected_metrics:
        heatmap_df = results.copy()
        heatmap_df["row_label"] = heatmap_df.apply(
            lambda row: f"{SHORT_MODEL_LABELS.get(str(row['model_name']), str(row['model_name']))} | k={int(row['k'])}",
            axis=1,
        )
        pivot = (
            heatmap_df.pivot(index="row_label", columns="dataset", values=metric)
            .reindex(sorted(heatmap_df["row_label"].unique()))
        )

        fig_height = max(8.5, 0.55 * len(pivot.index) + 2.5)
        fig, ax = plt.subplots(figsize=(12, fig_height))
        image = ax.imshow(pivot.values, cmap="YlGnBu", aspect="auto")

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([DATASET_LABELS.get(str(col), str(col)) for col in pivot.columns], rotation=12, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_title(f"{METRIC_LABELS[metric]} Heatmap", pad=16)

        for row_idx in range(len(pivot.index)):
            for col_idx in range(len(pivot.columns)):
                value = pivot.iat[row_idx, col_idx]
                if pd.notna(value):
                    ax.text(
                        col_idx,
                        row_idx,
                        f"{value:.3f}",
                        ha="center",
                        va="center",
                        color="#102027",
                        fontsize=11,
                        fontweight="bold",
                    )

        cbar = fig.colorbar(image, ax=ax)
        cbar.set_label(METRIC_LABELS[metric], rotation=270, labelpad=24, fontweight="bold")

        output_path = output_dir / f"heatmap_{metric}.png"
        _save_figure(fig, output_path)
        output_paths.append(output_path)

    return output_paths


def plot_cv_test_gap(results: pd.DataFrame, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(15, 10))

    for dataset in results["dataset"].cat.categories:
        dataset_df = results[results["dataset"] == dataset]
        if dataset_df.empty:
            continue

        for model_name in results["model_name"].cat.categories:
            subset = dataset_df[dataset_df["model_name"] == model_name]
            if subset.empty:
                continue

            ax.scatter(
                subset["cv_macro_f1_mean"],
                subset["test_macro_f1"],
                s=140 + 18 * subset["k"],
                color=DATASET_COLORS.get(str(dataset), "#546E7A"),
                marker=MODEL_MARKERS.get(str(model_name), "o"),
                edgecolor="black",
                linewidth=1.0,
                alpha=0.88,
            )

    line_min = min(results["cv_macro_f1_mean"].min(), results["test_macro_f1"].min()) - 0.02
    line_max = max(results["cv_macro_f1_mean"].max(), results["test_macro_f1"].max()) + 0.02
    ax.plot([line_min, line_max], [line_min, line_max], linestyle="--", color="#C62828", linewidth=2.4)

    dataset_handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=10, markerfacecolor=color, markeredgecolor="black", label=DATASET_LABELS.get(dataset, dataset))
        for dataset, color in DATASET_COLORS.items()
        if dataset in set(results["dataset"].astype(str))
    ]
    model_handles = [
        Line2D([0], [0], marker=marker, linestyle="", markersize=10, color="#37474F", label=SHORT_MODEL_LABELS.get(model, model))
        for model, marker in MODEL_MARKERS.items()
        if model in set(results["model_name"].astype(str))
    ]

    if dataset_handles:
        legend1 = ax.legend(handles=dataset_handles, loc="upper left", frameon=True, title="Dataset")
        ax.add_artist(legend1)
    if model_handles:
        ax.legend(handles=model_handles, loc="lower right", frameon=True, title="Model")

    ax.set_title("CV Macro F1 vs Test Macro F1", pad=16)
    ax.set_xlabel(METRIC_LABELS["cv_macro_f1_mean"])
    ax.set_ylabel(METRIC_LABELS["test_macro_f1"])
    ax.grid(True, linestyle="--", alpha=0.35)

    output_path = output_dir / "cv_vs_test_macro_f1.png"
    _save_figure(fig, output_path)
    return output_path


def plot_cv_test_metric_comparison(results: pd.DataFrame, output_dir: Path) -> Path:
    best_rows = _best_rows_by_dataset(results)
    metric_pairs = [
        ("Accuracy", "cv_accuracy_mean", "test_accuracy"),
        ("Macro F1", "cv_macro_f1_mean", "test_macro_f1"),
        ("Weighted F1", "cv_weighted_f1_mean", "test_weighted_f1"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(18, 12), sharey=True)
    axes = axes.ravel()

    for idx, (_, row) in enumerate(best_rows.iterrows()):
        ax = axes[idx]
        x = list(range(len(metric_pairs)))
        cv_values = [row[cv_metric] for _, cv_metric, _ in metric_pairs]
        test_values = [row[test_metric] for _, _, test_metric in metric_pairs]
        width = 0.38

        ax.bar(
            [value - width / 2 for value in x],
            cv_values,
            width=width,
            color="#90A4AE",
            edgecolor="black",
            linewidth=1.1,
            label="CV mean",
        )
        ax.bar(
            [value + width / 2 for value in x],
            test_values,
            width=width,
            color=DATASET_COLORS.get(str(row["dataset"]), "#546E7A"),
            edgecolor="black",
            linewidth=1.1,
            label="Test",
        )

        ax.set_title(
            f"{DATASET_LABELS.get(str(row['dataset']), str(row['dataset']))}\n{SHORT_MODEL_LABELS.get(str(row['model_name']), str(row['model_name']))} | k={int(row['k'])}",
            pad=14,
        )
        ax.set_xticks(x)
        ax.set_xticklabels([label for label, _, _ in metric_pairs])
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", linestyle="--", alpha=0.35)

    for ax in axes[len(best_rows):]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.01), ncol=2, frameon=True)
    fig.suptitle("CV Mean vs Test Metrics for the Best Configuration in Each Dataset", fontsize=24, fontweight="bold", y=1.02)
    fig.subplots_adjust(bottom=0.14)

    output_path = output_dir / "cv_vs_test_metric_comparison.png"
    _save_figure(fig, output_path)
    return output_path


def plot_confusion_matrix_comparison(results: pd.DataFrame, output_dir: Path) -> Path:
    best_rows = _best_rows_by_dataset(results)
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.ravel()

    for idx, (_, row) in enumerate(best_rows.iterrows()):
        ax = axes[idx]
        cm = _load_confusion_matrix(row)
        cm_pct = cm.div(cm.sum(axis=1), axis=0).fillna(0) * 100
        image = ax.imshow(cm_pct.values, cmap="Blues", vmin=0, vmax=100)

        ax.set_xticks(range(len(CLASS_LABELS)))
        ax.set_xticklabels(CLASS_LABELS, rotation=25, ha="right")
        ax.set_yticks(range(len(CLASS_LABELS)))
        ax.set_yticklabels(CLASS_LABELS)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.set_title(
            f"{DATASET_LABELS.get(str(row['dataset']), str(row['dataset']))}\n{SHORT_MODEL_LABELS.get(str(row['model_name']), str(row['model_name']))} | k={int(row['k'])}",
            pad=14,
        )

        for row_idx in range(len(CLASS_LABELS)):
            for col_idx in range(len(CLASS_LABELS)):
                value = cm_pct.iat[row_idx, col_idx]
                color = "white" if value >= 55 else "#102027"
                ax.text(
                    col_idx,
                    row_idx,
                    f"{value:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                    color=color,
                )

    for ax in axes[len(best_rows):]:
        ax.axis("off")

    # cbar = fig.colorbar(image, ax=axes.tolist(), shrink=0.8)
    # cbar.set_label("Row-normalized percentage", rotation=270, labelpad=28, fontweight="bold")
    fig.suptitle("Comparison Confusion Matrices for the Best Configuration in Each Dataset", fontsize=25, fontweight="bold", y=1.01)

    output_path = output_dir / "confusion_matrix_comparison_best_configs.png"
    _save_figure(fig, output_path)
    return output_path


def plot_train_test_diagnostics(results: pd.DataFrame, output_dir: Path) -> Path:
    best_rows = _best_rows_by_dataset(results)
    metric_pairs = [
        ("accuracy", "test_accuracy", "Accuracy"),
        ("macro_f1", "test_macro_f1", "Macro F1"),
        ("weighted_f1", "test_weighted_f1", "Weighted F1"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(20, 14), sharey=True)
    axes = axes.ravel()

    for idx, (_, row) in enumerate(best_rows.iterrows()):
        ax = axes[idx]
        cv_df = _load_cv_results(row)
        x = list(range(len(metric_pairs)))

        for fold_idx, (_, fold_row) in enumerate(cv_df.iterrows(), start=1):
            validation_values = [fold_row[cv_metric] for cv_metric, _, _ in metric_pairs]
            ax.plot(
                x,
                validation_values,
                marker="o",
                markersize=7,
                color="#90A4AE",
                alpha=0.35 + 0.1 * min(fold_idx, 4),
            )

        mean_values = [cv_df[cv_metric].mean() for cv_metric, _, _ in metric_pairs]
        test_values = [row[test_metric] for _, test_metric, _ in metric_pairs]
        ax.plot(x, mean_values, marker="o", markersize=9, color=PHASE_COLORS["train_proxy"], label="Train proxy")
        ax.plot(x, test_values, marker="D", markersize=9, color=PHASE_COLORS["test"], label="Test")

        for pos, value in enumerate(test_values):
            ax.text(pos, value + 0.018, f"{value:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

        ax.set_title(
            f"{DATASET_LABELS.get(str(row['dataset']), str(row['dataset']))}\n{SHORT_MODEL_LABELS.get(str(row['model_name']), str(row['model_name']))} | k={int(row['k'])}",
            pad=14,
        )
        ax.set_xticks(x)
        ax.set_xticklabels([label for _, _, label in metric_pairs])
        ax.set_ylim(0.25, 1.0)
        ax.grid(axis="y", linestyle="--", alpha=0.35)

    for ax in axes[len(best_rows):]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.01), ncol=2, frameon=True)
    fig.suptitle("Train Proxy vs Test Results for the Best Configuration", fontsize=24, fontweight="bold", y=1.02)
    fig.subplots_adjust(bottom=0.14)

    output_path = output_dir / "train_vs_test_diagnostics.png"
    _save_figure(fig, output_path)
    return output_path


def generate_result_summary_plots(
    csv_path: Path = DEFAULT_INPUT_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> list[Path]:
    _apply_plot_style()
    results = load_results(csv_path)

    saved_paths = [
        plot_metric_trends_by_model(results, output_dir),
        plot_best_model_per_dataset(results, output_dir),
        plot_cv_test_gap(results, output_dir),
        plot_cv_test_metric_comparison(results, output_dir),
        plot_confusion_matrix_comparison(results, output_dir),
        plot_train_test_diagnostics(results, output_dir),
    ]
    saved_paths.extend(plot_metric_heatmaps(results, output_dir))
    return saved_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create summary visualizations for CatBoost all_results_summary.csv."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Path to the summary CSV (default: {DEFAULT_INPUT_PATH})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory where plots will be saved (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    saved_paths = generate_result_summary_plots(args.input, args.output_dir)
    print("Saved result-summary plots:")
    for path in saved_paths:
        print(f" - {path}")


if __name__ == "__main__":
    main()
