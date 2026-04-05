LIHC / HQRTM CatBoost Pipeline
==============================

Overview
--------
This repository trains CatBoost classifiers on leakage-safe LIHC and HQRTM label variants, evaluates them with country-aware grouped cross-validation, and exports static plus dynamic visual diagnostics for model comparison, fit checking, and dataset description.

Core Feature Blocks
-------------------
BASE_STRUCTURAL
- floor_area
- house_age
- dwelling_type
- insulation_count
- main_heating_source
- heating_control
- household_size
- children_present
- elderly_present

CONTEXT_FEATURES
- Country
- SettlementSize
- S6
- C2
- C3

SOCIOECONOMIC_AUX
- equivalized_income

QR_FEATURES
- floor_area
- house_age
- dwelling_type
- insulation_count
- main_heating_source
- household_size

Training Pipeline
-----------------
Main runner:
- `UKK/LIHC-Informed-Socio-Economic-Predictors/model/baseline1(CatBoost)/catboost_run_preprocessed.py`

The updated runner now saves, for each dataset / model / k combination:
- `cv_results.csv`: fold-level train and validation metrics
- `learning_curve_fold_<fold>.csv`: per-fold CatBoost evaluation history
- `learning_curves_long.csv`: stacked learning-curve history across folds
- `learning_curve_summary.csv`: mean and std by split / metric / iteration
- `final_split_metrics.csv`: final-train versus test metrics for the saved final model
- `train_confusion_matrix.csv`: confusion matrix on the final training split
- `confusion_matrix.csv`: confusion matrix on the holdout test split
- `summary_metrics.csv`: rich summary row including train, validation, final-train, and test statistics
- `all_results_summary.csv`: project-level summary table

Run the training pipeline with:
- `python3 UKK/LIHC-Informed-Socio-Economic-Predictors/model/baseline1(CatBoost)/catboost_run_preprocessed.py`

Important note:
- The new fit-diagnostic plots require rerunning the training pipeline once, because the old saved summaries do not contain the new train and learning-curve artifacts.

Visualization Modules
---------------------
Result summary module:
- `UKK/LIHC-Informed-Socio-Economic-Predictors/visualization/result_summary.py`
- Creates high-resolution comparison plots from `all_results_summary.csv`.

Training fit diagnostics:
- `UKK/LIHC-Informed-Socio-Economic-Predictors/visualization/training_fit_diagnostics.py`
- Uses the updated saved train, validation, and learning-curve artifacts to assess whether models look underfit or overfit.
- Outputs include:
  - `train_validation_test_best_configs.png`
  - `learning_curves_best_configs.png`

Dataset statistical analysis:
- `UKK/LIHC-Informed-Socio-Economic-Predictors/visualization/dataset_feature_analysis.py`
- Produces quantitative analysis for the selected structural, context, socioeconomic, and QR feature groups.

Dynamic storytelling assets:
- `UKK/LIHC-Informed-Socio-Economic-Predictors/visualization/dynamic_pipeline_story.py`
- Produces self-contained animated HTML files for communication and README-linked storytelling.

Generated Dataset Analysis Outputs
----------------------------------
Generated in:
- `final_saved_models_catboost/dataset_analysis`

Artifacts:
- `selected_feature_summary.csv`
  Brief description: statistical summary table with dtype, missingness, unique count, and basic descriptive statistics.
- `numeric_feature_distributions.png`
  Brief description: histogram grid for the main quantitative structural and socioeconomic variables.
- `categorical_feature_distributions.png`
  Brief description: top-category distributions for key structural and context variables.
- `selected_feature_missingness.png`
  Brief description: missing-data percentages across the selected analysis features.
- `numeric_feature_correlation.png`
  Brief description: correlation heatmap for the main quantitative selected features.
- `income_vs_expenditure_scatter.png`
  Brief description: quantitative relationship between equivalized income and total expenditure, colored by household size.
- `floor_area_vs_expenditure_scatter.png`
  Brief description: quantitative relationship between floor area and total expenditure, colored by equivalized income.

Run with:
- `python3 UKK/LIHC-Informed-Socio-Economic-Predictors/visualization/dataset_feature_analysis.py`

Generated Dynamic HTML Outputs
------------------------------
Generated in:
- `final_saved_models_catboost/dynamic_visualizations`

Artifacts:
- `pipeline_flow.html`
  Brief description: animated step-by-step flowchart of the full modeling and reporting pipeline.
- `model_metric_story.html`
  Brief description: animated model-ranking story that cycles through datasets and compares test macro-F1 across model families.
- `feature_group_story.html`
  Brief description: dynamic feature-group explainer for the structural, context, socioeconomic, and quantile-regression blocks.

Run with:
- `python3 UKK/LIHC-Informed-Socio-Economic-Predictors/visualization/dynamic_pipeline_story.py`

Existing High-Resolution Result Summary Outputs
-----------------------------------------------
Generated in:
- `final_saved_models_catboost/result_summary_plots`

Key artifacts:
- `metric_trends_test_macro_f1.png`
- `best_configuration_by_dataset.png`
- `cv_vs_test_macro_f1.png`
- `cv_vs_test_metric_comparison.png`
- `confusion_matrix_comparison_best_configs.png`
- `train_vs_test_diagnostics.png`
- `heatmap_test_accuracy.png`
- `heatmap_test_macro_f1.png`
- `heatmap_test_weighted_f1.png`

Recommended Workflow
--------------------
1. Run preprocessing if the clean dataset needs to be regenerated.
2. Run `catboost_run_preprocessed.py` to regenerate the saved model artifacts with the richer train and validation logging.
3. Run `training_fit_diagnostics.py` to inspect real train, validation, and test behavior.
4. Run `result_summary.py` for high-resolution model-comparison figures.
5. Run `dataset_feature_analysis.py` for the quantitative dataset-description figures.
6. Run `dynamic_pipeline_story.py` for animated HTML storytelling assets.

Interpretation Guidance
-----------------------
Use the updated artifacts together:
- Compare train versus validation curves to check optimization stability.
- Compare final train versus test metrics to inspect generalization gaps.
- Use confusion-matrix comparisons to identify class-specific weaknesses.
- Use the dataset-analysis figures to explain how the selected feature blocks behave statistically before modeling.
