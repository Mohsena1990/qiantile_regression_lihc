import os
import random
import itertools
import numpy as np
import pandas as pd

from sklearn.metrics import f1_score, accuracy_score
from CatBoost import CatBoostML


def evaluate_catboost_params(
    params,
    folds,
    features,
    cat_features=None,
    score_metric="macro_f1"
):
    """
    Evaluate one CatBoost parameter set across precomputed folds.

    Parameters
    ----------
    params : dict
        CatBoost parameter dictionary.
    folds : list
        List of (X_tr, X_val, y_tr, y_val).
    features : list
        Feature names to use.
    cat_features : list or None
        Categorical feature names.
    score_metric : str
        'macro_f1', 'weighted_f1', or 'accuracy'.

    Returns
    -------
    dict
        Mean score, std score, and fold-level scores.
    """
    scores = []

    fold_cat_features = [c for c in (cat_features or []) if c in features]

    for fold_id, (X_tr, X_val, y_tr, y_val) in enumerate(folds, start=1):
        model = CatBoostML(params=params)

        model.train(
            X_train=X_tr[features],
            y_train=y_tr,
            X_val=X_val[features],
            y_val=y_val,
            cat_features=fold_cat_features,
            use_class_weights=False
        )

        preds = model.predict(X_val[features], cat_features=fold_cat_features)

        if score_metric == "macro_f1":
            score = f1_score(y_val, preds, average="macro")
        elif score_metric == "weighted_f1":
            score = f1_score(y_val, preds, average="weighted")
        elif score_metric == "accuracy":
            score = accuracy_score(y_val, preds)
        else:
            raise ValueError("score_metric must be 'macro_f1', 'weighted_f1', or 'accuracy'")

        scores.append(score)

    return {
        "mean_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)),
        "fold_scores": scores
    }


def build_param_grid(task_type="CPU", devices=None):
    """
    Small, transparent CatBoost parameter grid aligned with the paper strategy.
    """
    base_grid = {
        "iterations": [300, 500],
        "learning_rate": [0.03, 0.05, 0.1],
        "depth": [4, 6, 8],
        "l2_leaf_reg": [3, 10, 20],
        "bootstrap_type": ["Bernoulli"],
        "subsample": [0.7, 0.9],
        "verbose": [0],
        "loss_function": ["MultiClass"],
        "eval_metric": ["Accuracy"],
        "task_type": [task_type],
    }

    if task_type.upper() == "GPU" and devices is not None:
        base_grid["devices"] = [devices]

    keys = list(base_grid.keys())
    values = list(base_grid.values())

    param_list = []
    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        param_list.append(params)

    return param_list


def sample_param_grid(param_grid, n_iter=20, random_state=42):
    """
    Randomly sample a subset from the parameter grid.
    """
    rng = random.Random(random_state)

    if n_iter >= len(param_grid):
        return param_grid

    return rng.sample(param_grid, n_iter)


def tune_catboost(
    folds,
    features,
    cat_features=None,
    score_metric="macro_f1",
    task_type="CPU",
    devices=None,
    n_iter=20,
    random_state=42,
    output_file="catboost_tuning_results.csv"
):
    """
    Simple, transparent parameter tuning for CatBoost.

    Parameters
    ----------
    folds : list
        Precomputed CV folds.
    features : list
        Selected features for the model.
    cat_features : list or None
        Categorical feature names.
    score_metric : str
        'macro_f1', 'weighted_f1', or 'accuracy'.
    task_type : str
        'CPU' or 'GPU'.
    devices : str or None
        GPU device string if task_type='GPU'.
    n_iter : int
        Number of parameter combinations to evaluate.
    random_state : int
        Random seed.
    output_file : str
        CSV file to save tuning results.

    Returns
    -------
    dict
        Best parameter dictionary.
    """
    param_grid = build_param_grid(task_type=task_type, devices=devices)
    param_grid = sample_param_grid(param_grid, n_iter=n_iter, random_state=random_state)

    results = []
    best_score = -np.inf
    best_params = None

    for i, params in enumerate(param_grid, start=1):
        metrics = evaluate_catboost_params(
            params=params,
            folds=folds,
            features=features,
            cat_features=cat_features,
            score_metric=score_metric
        )

        row = {
            "trial": i,
            "mean_score": metrics["mean_score"],
            "std_score": metrics["std_score"],
            **params
        }
        results.append(row)

        print(
            f"Trial {i}/{len(param_grid)} | "
            f"{score_metric}={metrics['mean_score']:.4f} ± {metrics['std_score']:.4f}"
        )

        if metrics["mean_score"] > best_score:
            best_score = metrics["mean_score"]
            best_params = params.copy()

    results_df = pd.DataFrame(results).sort_values("mean_score", ascending=False)

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    results_df.to_csv(output_file, index=False)

    print(f"\nSaved tuning results to: {output_file}")
    print(f"Best params: {best_params}")
    print(f"Best {score_metric}: {best_score:.4f}")

    return best_params