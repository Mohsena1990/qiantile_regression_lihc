import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score, classification_report
from typing import Optional, List, Union, Tuple


class CatBoostML:
    """
    CatBoost wrapper compatible with both CPU and GPU.

    Notes
    ----
    - Set params={"task_type": "CPU"} for CPU
    - Set params={"task_type": "GPU", "devices": "0"} for GPU
    - Removes incompatible parameter combinations automatically
    """

    def __init__(self, params: Optional[dict] = None):
        default_params = {
            "iterations": 700,
            "learning_rate": 0.01,
            "depth": 6,
            "eval_metric": "TotalF1:average=Macro",
            "loss_function": "MultiClass",
            "random_seed": 42,
            "verbose": 100,
            "l2_leaf_reg": 10,
            "subsample": 0.8,
            "bootstrap_type": "Bernoulli",
            "task_type": "CPU",
        }

        self.params = default_params.copy()
        if params:
            self.params.update(params)

        self._sanitize_params()
        self.model = CatBoostClassifier(**self.params)
        self.class_weights = None
        self.evals_result = {}

    def _sanitize_params(self) -> None:
        """Remove incompatible parameter combinations for CPU/GPU."""
        task_type = str(self.params.get("task_type", "CPU")).upper()

        if task_type not in {"CPU", "GPU"}:
            self.params["task_type"] = "CPU"
            task_type = "CPU"

        if task_type == "CPU" and "devices" in self.params:
            self.params.pop("devices", None)

        if task_type == "GPU":
            if "devices" in self.params and self.params["devices"] in [None, "", -1]:
                self.params.pop("devices", None)

            if "rsm" in self.params:
                print("Removing 'rsm' for broader GPU compatibility.")
                self.params.pop("rsm", None)

        if self.params.get("bootstrap_type") == "Bayesian" and "subsample" in self.params:
            print("Removing 'subsample' (not used with Bayesian bootstrap).")
            self.params.pop("subsample", None)

        if self.params.get("bootstrap_type") != "Bayesian" and "bagging_temperature" in self.params:
            self.params.pop("bagging_temperature", None)

    def _rebuild_model(self) -> None:
        self._sanitize_params()
        self.model = CatBoostClassifier(**self.params)
        self.evals_result = {}

    def set_class_weights(self, y_train: pd.Series, scale: float = 1.0):
        """Compute inverse-frequency class weights."""
        classes, counts = np.unique(y_train, return_counts=True)
        total = len(y_train)

        weights = total / (len(classes) * counts)
        weights = weights ** scale

        self.class_weights = dict(zip(classes, weights))
        self.params["class_weights"] = self.class_weights
        self._rebuild_model()

        print("Class Weights:", self.class_weights)

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        cat_features: Optional[Union[List[int], List[str]]] = None,
        early_stopping_rounds: int = 150,
        use_best_model: bool = True,
        use_class_weights: bool = False,
    ):
        if use_class_weights:
            self.set_class_weights(y_train)
        else:
            self._rebuild_model()

        train_pool = Pool(X_train, y_train, cat_features=cat_features)
        eval_set = Pool(X_val, y_val, cat_features=cat_features) if X_val is not None else None

        self.model.fit(
            train_pool,
            eval_set=eval_set,
            use_best_model=use_best_model if eval_set is not None else False,
            early_stopping_rounds=early_stopping_rounds if eval_set is not None else None,
        )
        self.evals_result = self.model.get_evals_result()

    def evaluate(
        self,
        X_eval: pd.DataFrame,
        y_eval: pd.Series,
        cat_features: Optional[Union[List[int], List[str]]] = None,
        split_name: str = "Evaluation",
    ) -> Tuple[float, pd.Series]:
        eval_pool = Pool(X_eval, cat_features=cat_features)
        preds = self.model.predict(eval_pool)
        preds = np.array(preds).ravel()

        acc = accuracy_score(y_eval, preds)
        print(f"{split_name} Accuracy:", acc)
        print(classification_report(y_eval, preds))

        return acc, pd.Series(preds, index=y_eval.index)

    def get_evals_result(self) -> dict:
        return self.evals_result or {}

    def load_model(self, model_path: str):
        self.model = CatBoostClassifier()
        self.model.load_model(model_path)
        self.evals_result = {}

    def predict_proba(
        self,
        X_new: pd.DataFrame,
        cat_features: Optional[Union[List[int], List[str]]] = None,
    ) -> pd.DataFrame:
        X_new = X_new.copy()

        if cat_features is not None:
            for col in cat_features:
                if col in X_new.columns:
                    X_new[col] = X_new[col].astype(str)

        pool = Pool(X_new, cat_features=cat_features)
        return pd.DataFrame(self.model.predict_proba(pool), columns=self.model.classes_)

    def predict(
        self,
        X_new: pd.DataFrame,
        cat_features: Optional[Union[List[int], List[str]]] = None,
    ) -> pd.Series:
        X_new = X_new.copy()

        if cat_features is not None:
            for col in cat_features:
                if col in X_new.columns:
                    X_new[col] = X_new[col].astype(str)

        pool = Pool(X_new, cat_features=cat_features)
        preds = self.model.predict(pool)
        return pd.Series(np.array(preds).ravel())

    def get_feature_importance(self, feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        importances = self.model.feature_importances_

        if feature_names is None:
            feature_names = [f"f{i}" for i in range(len(importances))]

        if len(feature_names) != len(importances):
            min_len = min(len(feature_names), len(importances))
            feature_names = feature_names[:min_len]
            importances = importances[:min_len]

        return pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances,
        }).sort_values(by="Importance", ascending=False)
