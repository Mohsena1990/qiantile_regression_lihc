# ======================================================
# FINAL RUNNING MODULE
# 4 datasets × 4 model baselines × k = {2, 3, 4, 5}
# SMOTE/SMOTENC-ready, leakage-aware, CPU/GPU-compatible
# ======================================================

import os
import sys
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from imblearn.over_sampling import SMOTE, SMOTENC

from CatBoost import CatBoostML
from Stratify_train_test_split_by_country import country_stratified_group_split
from fine_tuning import tune_catboost


# ======================================================
# 1 Paths and configuration
# ======================================================

BASE_DIR = r"/home/mohsen/project/qiantile_regression_lihc"
MODEL_DIR = os.path.join(BASE_DIR, "final_saved_models_catboost")
os.makedirs(MODEL_DIR, exist_ok=True)

PREPROCESS_DIR = os.path.join(
    BASE_DIR,
    "UKK",
    "LIHC-Informed-Socio-Economic-Predictors",
    "preprocessing",
)
if PREPROCESS_DIR not in sys.path:
    sys.path.append(PREPROCESS_DIR)

from risk_category import assign_traditional_lihc, assign_hqrtm

BASE_DATA_PATH = os.path.join(BASE_DIR, "preprocessed_data_clean.csv")

DATASET_CONFIGS = {
    "traditional_lihc": {
        "label_type": "traditional",
        "income_rule": "country_median_60",
        "exp_quantile": 0.80,
    },
    "hqrtm_q60": {
        "label_type": "hqrtm",
        "income_rule": "country_median_60",
        "quantile": 0.60,
        "margin_scale": 0.10,
    },
    "hqrtm_q65": {
        "label_type": "hqrtm",
        "income_rule": "country_median_60",
        "quantile": 0.65,
        "margin_scale": 0.10,
    },
    "hqrtm_q70": {
        "label_type": "hqrtm",
        "income_rule": "country_median_60",
        "quantile": 0.70,
        "margin_scale": 0.10,
    },
}

TARGET = "risk_category"
K_VALUES = [2, 3, 4]
TEST_SIZE = 0.25
RANDOM_STATE = 42

TASK_TYPE = "CPU"   # change to "GPU" if needed
DEVICES = "0"       # used only if TASK_TYPE == "GPU"


# ======================================================
# 2 Feature blocks
# ======================================================
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
    "C3"
]

SOCIOECONOMIC_AUX = [
    "equivalized_income",
]

MODEL_SPECS = {
    "B1_structural": BASE_STRUCTURAL,
    "B2_structural_socioeconomic": BASE_STRUCTURAL + SOCIOECONOMIC_AUX,
    "B3_structural_context": BASE_STRUCTURAL + CONTEXT_FEATURES,
    "B4_structural_context_socioeconomic": BASE_STRUCTURAL + CONTEXT_FEATURES + SOCIOECONOMIC_AUX,
}

# Columns that should be handled as categorical
CATEGORICAL_LIKE = {
    "Country",
    "dwelling_type",
    "main_heating_source",
    "ownership",
    "heating_control",
    "SettlementSize"
}

QR_FEATURES = [
    "floor_area",
    "house_age",
    "dwelling_type",
    "insulation_count",
    "main_heating_source",
    "household_size",
]


# ======================================================
# 3 Helper functions
# ======================================================
def ensure_required_columns(df: pd.DataFrame, cols: list, dataset_name: str, model_name: str = "") -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        prefix = f"[{dataset_name}"
        if model_name:
            prefix += f" - {model_name}"
        prefix += "]"
        raise KeyError(f"{prefix} Missing required columns: {missing}")


def create_labels_for_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    dataset_name: str,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build labels using train-only fit and then apply to train/test.
    This removes label-construction leakage.
    """
    if config["label_type"] == "traditional":
        train_labeled = assign_traditional_lihc(
            train_df,
            income_col="equivalized_income",
            exp_col="total_expenditure",
            country_col="Country",
            income_rule=config.get("income_rule", "country_median_60"),
            exp_quantile=config.get("exp_quantile", 0.80),
            fit_df=train_df,
        )
        test_labeled = assign_traditional_lihc(
            test_df,
            income_col="equivalized_income",
            exp_col="total_expenditure",
            country_col="Country",
            income_rule=config.get("income_rule", "country_median_60"),
            exp_quantile=config.get("exp_quantile", 0.80),
            fit_df=train_df,
        )
    elif config["label_type"] == "hqrtm":
        train_labeled = assign_hqrtm(
            train_df,
            qr_features=QR_FEATURES,
            income_col="equivalized_income",
            exp_col="total_expenditure",
            country_col="Country",
            income_rule=config.get("income_rule", "country_median_60"),
            quantile=config.get("quantile", 0.65),
            add_country_effects=True,
            margin_scale=config.get("margin_scale", 0.10),
            fit_df=train_df,
        )
        test_labeled = assign_hqrtm(
            test_df,
            qr_features=QR_FEATURES,
            income_col="equivalized_income",
            exp_col="total_expenditure",
            country_col="Country",
            income_rule=config.get("income_rule", "country_median_60"),
            quantile=config.get("quantile", 0.65),
            add_country_effects=True,
            margin_scale=config.get("margin_scale", 0.10),
            fit_df=train_df,
        )
    else:
        raise ValueError(f"Unknown label_type for dataset {dataset_name}: {config['label_type']}")

    train_labeled = train_labeled.dropna(subset=[TARGET]).copy()
    test_labeled = test_labeled.dropna(subset=[TARGET]).copy()
    return train_labeled, test_labeled


def fit_feature_preprocessor(
    train_df: pd.DataFrame,
    features: list,
    categorical_cols: list | None = None,
) -> dict:
    """
    Fit preprocessing statistics on train only.
    """
    categorical_cols = set(categorical_cols or [])
    numeric_medians = {}

    for col in features:
        if col in categorical_cols:
            continue
        series = pd.to_numeric(train_df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        med = series.median()
        numeric_medians[col] = 0.0 if pd.isna(med) else float(med)

    return {
        "categorical_cols": categorical_cols,
        "numeric_medians": numeric_medians,
    }


def transform_features(
    df: pd.DataFrame,
    features: list,
    processor: dict,
) -> pd.DataFrame:
    """
    Apply train-fitted preprocessing to any split.
    """
    out = df.copy()
    categorical_cols = processor["categorical_cols"]
    numeric_medians = processor["numeric_medians"]

    for col in features:
        if col not in out.columns:
            continue

        if col in categorical_cols:
            out[col] = (
                out[col]
                .astype(str)
                .replace(["nan", "None", "NaN"], np.nan)
                .fillna("missing")
            )
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce")
            out[col] = out[col].replace([np.inf, -np.inf], np.nan)
            out[col] = out[col].fillna(numeric_medians.get(col, 0.0))

    return out


def assert_no_nan(df: pd.DataFrame, cols: list, dataset_name: str, model_name: str, where: str) -> None:
    nan_counts = df[cols].isna().sum()
    nan_counts = nan_counts[nan_counts > 0]
    if not nan_counts.empty:
        print(f"\nNaN counts in {where}:")
        print(nan_counts)
        raise ValueError(f"[{dataset_name} - {model_name}] NaN values remain in {where}.")


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }


def save_confusion_matrix(cm: pd.DataFrame, path: str) -> None:
    cm.to_csv(path, index=True)


def oversample_training_data(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cat_features: list,
    random_state: int = 42,
    sampling_strategy: str = "not majority"
):
    """
    Apply SMOTENC if categorical columns exist, else fall back to SMOTE.
    Used only for final model training if oversampling is enabled.
    """
    X_res = X_train.copy()
    y_res = y_train.copy()

    if len(cat_features) > 0:
        cat_indices = [X_res.columns.get_loc(c) for c in cat_features if c in X_res.columns]
        sampler = SMOTENC(
            categorical_features=cat_indices,
            sampling_strategy=sampling_strategy,
            random_state=random_state
        )
    else:
        sampler = SMOTE(
            sampling_strategy=sampling_strategy,
            random_state=random_state
        )

    X_res, y_res = sampler.fit_resample(X_res, y_res)
    return X_res, y_res


# ======================================================
# 4 Main experiment loop
# ======================================================
all_results = []

base_df = pd.read_csv(BASE_DATA_PATH, low_memory=False)
ensure_required_columns(base_df, ["Country", "equivalized_income", "total_expenditure"], "base_data")
base_df = base_df.dropna(subset=["Country", "equivalized_income", "total_expenditure"]).copy()

for dataset_name, dataset_config in DATASET_CONFIGS.items():
    print(f"\n{'=' * 80}")
    print(f"Dataset: {dataset_name}")
    print(f"Source: {BASE_DATA_PATH}")
    print(f"Label config: {dataset_config}")
    print(f"{'=' * 80}")

    # Split first, then build labels from train-only fit
    train_df_raw, test_df_raw = train_test_split(
        base_df,
        test_size=TEST_SIZE,
        stratify=base_df["Country"],
        random_state=RANDOM_STATE
    )

    train_df, test_df = create_labels_for_split(
        train_df=train_df_raw,
        test_df=test_df_raw,
        dataset_name=dataset_name,
        config=dataset_config,
    )

    print("Train size:", len(train_df), "| Test size:", len(test_df))
    print("Train class counts:\n", train_df[TARGET].value_counts())
    print("Test class counts:\n", test_df[TARGET].value_counts())

    for model_name, features in MODEL_SPECS.items():
        print(f"\n--- Model: {model_name} ---")
        print("Features:", features)

        ensure_required_columns(train_df, features + [TARGET], dataset_name, model_name)
        ensure_required_columns(test_df, features + [TARGET], dataset_name, model_name)

        cat_features = [f for f in features if f in CATEGORICAL_LIKE]

        # Train-fitted preprocessing only
        processor = fit_feature_preprocessor(train_df, features, categorical_cols=cat_features)
        train_df_model = transform_features(train_df, features, processor)
        test_df_model = transform_features(test_df, features, processor)

        assert_no_nan(train_df_model, features, dataset_name, model_name, "train_df_model")
        assert_no_nan(test_df_model, features, dataset_name, model_name, "test_df_model")

        for k in K_VALUES:
            print(f"\n>>> Running: dataset={dataset_name} | model={model_name} | k={k}")

            run_dir = os.path.join(MODEL_DIR, dataset_name, model_name, f"k{k}")
            os.makedirs(run_dir, exist_ok=True)

            # --------------------------------------------------
            # Country-aware grouped CV folds
            # --------------------------------------------------
            split_df = train_df_model[features + [TARGET]].copy()
            if "Country" not in split_df.columns:
                split_df["Country"] = train_df_model["Country"].values

            folds = country_stratified_group_split(
                df=split_df,
                inputs=features,
                target=TARGET,
                n_splits=k,
                random_state=RANDOM_STATE,
                use_smote=True,
                categorical_cols=cat_features,
                sampling_strategy="not majority"
            )

            # --------------------------------------------------
            # Fine tuning
            # Keep imbalance strategy consistent: SMOTE + no class weights
            # --------------------------------------------------
            best_params = tune_catboost(
                folds=folds,
                features=features,
                cat_features=cat_features,
                score_metric="macro_f1",
                task_type=TASK_TYPE,
                devices=DEVICES if TASK_TYPE.upper() == "GPU" else None,
                n_iter=20,
                random_state=RANDOM_STATE,
                output_file=os.path.join(run_dir, "tuning_results.csv")
            )

            best_params = {
                k_: (v.item() if isinstance(v, np.generic) else v)
                for k_, v in best_params.items()
            }

            with open(os.path.join(run_dir, "best_params.json"), "w") as f:
                json.dump(best_params, f, indent=4)

            with open(os.path.join(run_dir, "features.json"), "w") as f:
                json.dump(
                    {
                        "dataset": dataset_name,
                        "model_name": model_name,
                        "features": features,
                        "cat_features": cat_features,
                        "k": k,
                    },
                    f,
                    indent=4,
                )

            # --------------------------------------------------
            # Cross-validation
            # --------------------------------------------------
            cv_rows = []

            for fold_id, (X_tr, X_val, y_tr, y_val) in enumerate(folds, start=1):
                model = CatBoostML(params=best_params)

                model.train(
                    X_train=X_tr[features],
                    y_train=y_tr,
                    X_val=X_val[features],
                    y_val=y_val,
                    cat_features=cat_features,
                    use_class_weights=False
                )

                y_val_pred = model.predict(X_val[features], cat_features=cat_features)
                fold_metrics = compute_metrics(y_val, y_val_pred)
                fold_metrics["fold"] = fold_id
                cv_rows.append(fold_metrics)

                model.model.save_model(os.path.join(run_dir, f"catboost_fold_{fold_id}.cbm"))

            cv_df = pd.DataFrame(cv_rows)
            cv_df.to_csv(os.path.join(run_dir, "cv_results.csv"), index=False)

            cv_summary = {
                "cv_accuracy_mean": cv_df["accuracy"].mean(),
                "cv_accuracy_std": cv_df["accuracy"].std(),
                "cv_macro_f1_mean": cv_df["macro_f1"].mean(),
                "cv_macro_f1_std": cv_df["macro_f1"].std(),
                "cv_weighted_f1_mean": cv_df["weighted_f1"].mean(),
                "cv_weighted_f1_std": cv_df["weighted_f1"].std(),
            }

            # --------------------------------------------------
            # Final model on full training split
            # Use the same imbalance strategy as CV: oversampling, no class weights
            # --------------------------------------------------
            X_train_final = train_df_model[features].copy()
            y_train_final = train_df_model[TARGET].copy()

            X_train_final, y_train_final = oversample_training_data(
                X_train_final,
                y_train_final,
                cat_features=cat_features,
                random_state=RANDOM_STATE,
                sampling_strategy="not majority"
            )

            final_model = CatBoostML(params=best_params)
            final_model.train(
                X_train=X_train_final,
                y_train=y_train_final,
                X_val=None,
                y_val=None,
                cat_features=cat_features,
                use_class_weights=False
            )

            final_model.model.save_model(os.path.join(run_dir, "catboost_final.cbm"))

            # --------------------------------------------------
            # Holdout test evaluation
            # --------------------------------------------------
            X_test = test_df_model[features].copy()
            y_test = test_df_model[TARGET].copy()

            y_test_pred = final_model.predict(X_test, cat_features=cat_features)
            test_metrics = compute_metrics(y_test, y_test_pred)

            cm = confusion_matrix(y_test, y_test_pred, labels=final_model.model.classes_)
            cm_df = pd.DataFrame(
                cm,
                index=[f"true_{c}" for c in final_model.model.classes_],
                columns=[f"pred_{c}" for c in final_model.model.classes_]
            )
            save_confusion_matrix(cm_df, os.path.join(run_dir, "confusion_matrix.csv"))

            result_row = {
                "dataset": dataset_name,
                "model_name": model_name,
                "k": k,
                **cv_summary,
                "test_accuracy": test_metrics["accuracy"],
                "test_macro_f1": test_metrics["macro_f1"],
                "test_weighted_f1": test_metrics["weighted_f1"],
                "test_macro_precision": test_metrics["macro_precision"],
                "test_macro_recall": test_metrics["macro_recall"],
            }

            all_results.append(result_row)

            pd.DataFrame([result_row]).to_csv(
                os.path.join(run_dir, "summary_metrics.csv"),
                index=False
            )

            print(pd.DataFrame([result_row]))


# ======================================================
# 5 Save master summary
# ======================================================
all_results_df = pd.DataFrame(all_results)
all_results_df = all_results_df.sort_values(
    by=["dataset", "model_name", "k", "test_macro_f1"],
    ascending=[True, True, True, False]
)

master_path = os.path.join(MODEL_DIR, "all_results_summary.csv")
all_results_df.to_csv(master_path, index=False)

print(f"\nFinished. Master summary saved to: {master_path}")
print(all_results_df)
