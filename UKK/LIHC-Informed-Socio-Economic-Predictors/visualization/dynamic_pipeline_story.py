from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_SUMMARY_PATH = Path("final_saved_models_catboost/all_results_summary.csv")
DEFAULT_FEATURE_SUMMARY_PATH = Path("final_saved_models_catboost/dataset_analysis/selected_feature_summary.csv")
DEFAULT_OUTPUT_DIR = Path("final_saved_models_catboost/dynamic_visualizations")

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
CONTEXT_FEATURES = ["Country", "SettlementSize", "S6", "C2", "C3"]
SOCIOECONOMIC_AUX = ["equivalized_income"]
QR_FEATURES = [
    "floor_area",
    "house_age",
    "dwelling_type",
    "insulation_count",
    "main_heating_source",
    "household_size",
]

PIPELINE_STEPS = [
    ("Raw Input", "Start from the cleaned household-level survey table and keep the core expenditure and income fields."),
    ("Leakage-safe Split", "Create train and test splits before target construction so threshold fitting and quantile labeling stay honest."),
    ("Dataset Labels", "Build traditional LIHC and HQRTM labels with train-only fits, then apply them to both train and test."),
    ("Feature Blocks", "Assemble structural, context, socioeconomic, and quantile-regression feature sets for the baseline families."),
    ("Grouped CV", "Run country-aware grouped folds so entire countries stay inside one side of each split."),
    ("Tuning", "Tune CatBoost on fold-level validation performance while keeping the imbalance strategy consistent."),
    ("Fold Diagnostics", "Save train metrics, validation metrics, confusion matrices, and learning curves for every fold."),
    ("Final Model", "Refit on the full training split, save final train-versus-test metrics, and export summary tables."),
    ("Storytelling", "Turn the saved artifacts into static and dynamic visual diagnostics for reporting and README integration."),
]


def load_best_metric_rows(summary_path: Path) -> list[dict]:
    summary = pd.read_csv(summary_path)
    metric_rows = (
        summary.sort_values(["dataset", "test_macro_f1"], ascending=[True, False])
        .groupby(["dataset", "model_name"], as_index=False)
        .first()
        .sort_values(["dataset", "test_macro_f1"], ascending=[True, False])
    )
    return metric_rows[["dataset", "model_name", "k", "test_macro_f1", "test_accuracy"]].to_dict(orient="records")


def load_feature_group_story(feature_summary_path: Path) -> list[dict]:
    groups = {
        "Base structural": BASE_STRUCTURAL,
        "Context": CONTEXT_FEATURES,
        "Socioeconomic": SOCIOECONOMIC_AUX,
        "Quantile-regression": QR_FEATURES,
    }
    summary_lookup = {}
    if feature_summary_path.exists():
        feature_summary = pd.read_csv(feature_summary_path)
        for _, row in feature_summary.iterrows():
            summary_lookup[row["feature"]] = row.to_dict()

    story = []
    for group_name, features in groups.items():
        cards = []
        for feature in features:
            info = summary_lookup.get(feature, {})
            cards.append(
                {
                    "feature": feature,
                    "missing_pct": round(float(info.get("missing_pct", 0.0)), 2),
                    "unique_values": int(info.get("unique_values", 0) or 0),
                    "mean": None if pd.isna(info.get("mean")) else info.get("mean"),
                    "std": None if pd.isna(info.get("std")) else info.get("std"),
                }
            )
        story.append({"group": group_name, "features": cards})
    return story


def write_pipeline_flow_html(output_path: Path) -> Path:
    steps_json = json.dumps([{"title": title, "description": description} for title, description in PIPELINE_STEPS])
    html = f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <title>Pipeline Flow</title>
  <style>
    body {{ font-family: Georgia, serif; margin: 0; background: linear-gradient(135deg, #f4efe6, #dbe8f2); color: #102027; }}
    .wrap {{ max-width: 1100px; margin: 0 auto; padding: 32px 24px 56px; }}
    h1 {{ margin-bottom: 8px; }}
    p.lead {{ font-size: 18px; max-width: 820px; }}
    .timeline {{ display: grid; gap: 18px; margin-top: 28px; }}
    .step {{ background: rgba(255,255,255,0.82); border: 2px solid transparent; border-radius: 18px; padding: 18px 20px; box-shadow: 0 12px 30px rgba(16,32,39,0.08); opacity: 0.45; transform: translateX(16px); transition: 0.45s ease; }}
    .step.active {{ opacity: 1; transform: translateX(0); border-color: #1e88e5; }}
    .step h2 {{ margin: 0 0 8px; font-size: 22px; }}
    .controls {{ margin-top: 20px; display: flex; gap: 12px; }}
    button {{ border: 0; background: #102027; color: white; padding: 10px 14px; border-radius: 999px; cursor: pointer; }}
  </style>
</head>
<body>
  <div class=\"wrap\">
    <h1>Dynamic Pipeline Flow</h1>
    <p class=\"lead\">This animated flowchart walks through the leakage-safe labeling, grouped validation, CatBoost tuning, fit diagnostics, and reporting steps used in the LIHC/HQRTM pipeline.</p>
    <div id=\"timeline\" class=\"timeline\"></div>
    <div class=\"controls\">
      <button onclick=\"previousStep()\">Previous</button>
      <button onclick=\"nextStep()\">Next</button>
    </div>
  </div>
  <script>
    const steps = {steps_json};
    const timeline = document.getElementById('timeline');
    let activeIndex = 0;

    function render() {{
      timeline.innerHTML = '';
      steps.forEach((step, idx) => {{
        const card = document.createElement('div');
        card.className = 'step' + (idx === activeIndex ? ' active' : '');
        card.innerHTML = `<h2>${{idx + 1}}. ${{step.title}}</h2><p>${{step.description}}</p>`;
        timeline.appendChild(card);
      }});
    }}

    function nextStep() {{ activeIndex = (activeIndex + 1) % steps.length; render(); }}
    function previousStep() {{ activeIndex = (activeIndex - 1 + steps.length) % steps.length; render(); }}
    render();
    setInterval(nextStep, 2600);
  </script>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)
    return output_path


def write_model_metric_story_html(output_path: Path, metric_rows: list[dict]) -> Path:
    rows_json = json.dumps(metric_rows)
    html = f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <title>Model Metric Story</title>
  <style>
    body {{ font-family: 'Trebuchet MS', sans-serif; margin: 0; background: #fffaf0; color: #1b1b1b; }}
    .wrap {{ max-width: 1100px; margin: 0 auto; padding: 30px 24px 56px; }}
    h1 {{ margin-bottom: 8px; }}
    .dataset {{ font-size: 20px; font-weight: bold; color: #0d47a1; margin: 18px 0; }}
    .chart {{ display: grid; gap: 12px; }}
    .bar-row {{ display: grid; grid-template-columns: 320px 1fr 100px; align-items: center; gap: 12px; }}
    .bar-label {{ font-weight: 600; }}
    .bar-track {{ background: #dfe7eb; border-radius: 999px; overflow: hidden; height: 26px; }}
    .bar-fill {{ background: linear-gradient(90deg, #43a047, #1e88e5); height: 100%; width: 0%; transition: width 0.8s ease; }}
    .bar-value {{ text-align: right; font-variant-numeric: tabular-nums; }}
  </style>
</head>
<body>
  <div class=\"wrap\">
    <h1>Animated Best-Model Metric Story</h1>
    <p>The dashboard cycles through datasets and shows the best test macro-F1 ranking across model families, together with the selected <code>k</code>.</p>
    <div id=\"dataset\" class=\"dataset\"></div>
    <div id=\"chart\" class=\"chart\"></div>
  </div>
  <script>
    const rows = {rows_json};
    const datasets = [...new Set(rows.map(r => r.dataset))];
    let datasetIndex = 0;

    function render() {{
      const dataset = datasets[datasetIndex];
      const datasetRows = rows.filter(r => r.dataset === dataset).sort((a, b) => b.test_macro_f1 - a.test_macro_f1);
      document.getElementById('dataset').textContent = dataset;
      const chart = document.getElementById('chart');
      chart.innerHTML = '';
      datasetRows.forEach((row) => {{
        const wrapper = document.createElement('div');
        wrapper.className = 'bar-row';
        wrapper.innerHTML = `
          <div class=\"bar-label\">${{row.model_name}} | k=${{row.k}}</div>
          <div class=\"bar-track\"><div class=\"bar-fill\"></div></div>
          <div class=\"bar-value\">${{row.test_macro_f1.toFixed(3)}}</div>
        `;
        chart.appendChild(wrapper);
        requestAnimationFrame(() => {{
          wrapper.querySelector('.bar-fill').style.width = `${{Math.max(4, row.test_macro_f1 * 100)}}%`;
        }});
      }});
    }}

    function nextDataset() {{ datasetIndex = (datasetIndex + 1) % datasets.length; render(); }}
    render();
    setInterval(nextDataset, 2800);
  </script>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)
    return output_path


def write_feature_group_story_html(output_path: Path, feature_story: list[dict]) -> Path:
    story_json = json.dumps(feature_story)
    html = f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <title>Feature Group Story</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 0; background: linear-gradient(180deg, #eef5db, #dfe7fd); color: #102027; }}
    .wrap {{ max-width: 1150px; margin: 0 auto; padding: 32px 24px 56px; }}
    h1 {{ margin-bottom: 10px; }}
    .tabs {{ display: flex; flex-wrap: wrap; gap: 10px; margin: 20px 0; }}
    .tab {{ background: white; border: 2px solid #c5d6e8; border-radius: 999px; padding: 10px 14px; cursor: pointer; }}
    .tab.active {{ border-color: #1e88e5; color: #1e88e5; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px; }}
    .card {{ background: rgba(255,255,255,0.88); border-radius: 16px; padding: 16px; box-shadow: 0 10px 24px rgba(16,32,39,0.08); }}
    .meta {{ margin-top: 10px; color: #455a64; font-size: 14px; line-height: 1.45; }}
  </style>
</head>
<body>
  <div class=\"wrap\">
    <h1>Dynamic Feature-Group Story</h1>
    <p>This lightweight HTML view helps explain how the structural, contextual, socioeconomic, and quantile-regression feature sets are composed and how complete they are in the processed dataset.</p>
    <div id=\"tabs\" class=\"tabs\"></div>
    <div id=\"grid\" class=\"grid\"></div>
  </div>
  <script>
    const story = {story_json};
    let active = 0;
    const tabs = document.getElementById('tabs');
    const grid = document.getElementById('grid');

    function render() {{
      tabs.innerHTML = '';
      grid.innerHTML = '';
      story.forEach((group, idx) => {{
        const button = document.createElement('button');
        button.className = 'tab' + (idx === active ? ' active' : '');
        button.textContent = group.group;
        button.onclick = () => {{ active = idx; render(); }};
        tabs.appendChild(button);
      }});

      story[active].features.forEach((feature) => {{
        const card = document.createElement('div');
        card.className = 'card';
        card.innerHTML = `
          <h3>${{feature.feature}}</h3>
          <div class=\"meta\">Missing: ${{feature.missing_pct}}%</div>
          <div class=\"meta\">Unique values: ${{feature.unique_values}}</div>
          <div class=\"meta\">Mean: ${{feature.mean === null ? 'n/a' : Number(feature.mean).toFixed(3)}}</div>
          <div class=\"meta\">Std: ${{feature.std === null ? 'n/a' : Number(feature.std).toFixed(3)}}</div>
        `;
        grid.appendChild(card);
      }});
    }}

    render();
    setInterval(() => {{ active = (active + 1) % story.length; render(); }}, 3200);
  </script>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)
    return output_path


def generate_dynamic_assets(
    summary_path: Path = DEFAULT_SUMMARY_PATH,
    feature_summary_path: Path = DEFAULT_FEATURE_SUMMARY_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> list[Path]:
    metric_rows = load_best_metric_rows(summary_path)
    feature_story = load_feature_group_story(feature_summary_path)
    return [
        write_pipeline_flow_html(output_dir / "pipeline_flow.html"),
        write_model_metric_story_html(output_dir / "model_metric_story.html", metric_rows),
        write_feature_group_story_html(output_dir / "feature_group_story.html", feature_story),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate dynamic HTML storytelling assets for the LIHC/HQRTM pipeline.")
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY_PATH, help=f"Summary CSV path (default: {DEFAULT_SUMMARY_PATH})")
    parser.add_argument("--feature-summary", type=Path, default=DEFAULT_FEATURE_SUMMARY_PATH, help=f"Feature summary CSV path (default: {DEFAULT_FEATURE_SUMMARY_PATH})")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    saved_paths = generate_dynamic_assets(args.summary, args.feature_summary, args.output_dir)
    print("Saved dynamic HTML assets:")
    for path in saved_paths:
        print(f" - {path}")


if __name__ == "__main__":
    main()
