import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from imblearn.over_sampling import SMOTENC, SMOTE
import numpy as np


def train_and_test_splitting(
    df,
    inputs,
    target,
    test_size=0.25,
    random_state=42
):
    """
    Standard stratified train/test split.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    X = df[inputs].copy()
    y = df[target].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print("X Train size:", X_train.shape)
    print("X Test size:", X_test.shape)
    print("Y Train size:", y_train.shape)
    print("Y Test size:", y_test.shape)

    return X_train, X_test, y_train, y_test


def country_stratified_group_split(
    df,
    inputs,
    target,
    n_splits=5,
    random_state=42,
    use_smote=False,
    categorical_cols=None,
    sampling_strategy="not majority",
    group_col="Country"
):
    """
    Country-aware stratified grouped CV split.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    inputs : list
        Feature column names.
    target : str
        Target column name.
    n_splits : int
        Number of grouped CV folds.
    random_state : int
        Reproducibility seed.
    use_smote : bool
        Whether to apply SMOTENC inside each training fold only.
    categorical_cols : list or None
        Categorical feature names for SMOTENC.
    sampling_strategy : str or dict
        SMOTENC strategy.
    group_col : str
        Grouping variable, default = Country.

    Returns
    -------
    list
        List of tuples: (X_tr, X_val, y_tr, y_val)
    """
    X = df[inputs].copy()
    y = df[target].copy()
    groups = df[group_col].copy()

    sgkf = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    folds = []

    for fold_id, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups), start=1):
        X_tr = X.iloc[train_idx].copy()
        X_val = X.iloc[val_idx].copy()
        y_tr = y.iloc[train_idx].copy()
        y_val = y.iloc[val_idx].copy()

        if use_smote:
            categorical_cols = categorical_cols or []
            categorical_cols = [c for c in categorical_cols if c in X_tr.columns]

            # Fill values before oversampling
            for col in X_tr.columns:
                if col in categorical_cols:
                    X_tr[col] = X_tr[col].astype(str).fillna("missing")
                    if col in X_val.columns:
                        X_val[col] = X_val[col].astype(str).fillna("missing")
                else:
                    X_tr[col] = pd.to_numeric(X_tr[col], errors="coerce")
                    X_tr[col] = X_tr[col].replace([np.inf, -np.inf], np.nan)
                    X_tr[col] = X_tr[col].fillna(X_tr[col].median())

                    if col in X_val.columns:
                        X_val[col] = pd.to_numeric(X_val[col], errors="coerce")
                        X_val[col] = X_val[col].replace([np.inf, -np.inf], np.nan)
                        X_val[col] = X_val[col].fillna(X_tr[col].median())

            # If categorical columns exist -> use SMOTENC
            if len(categorical_cols) > 0:
                cat_indices = [X_tr.columns.get_loc(c) for c in categorical_cols]

                sampler = SMOTENC(
                    categorical_features=cat_indices,
                    sampling_strategy=sampling_strategy,
                    random_state=random_state
                )
            else:
                # No categorical features -> fall back to plain SMOTE
                sampler = SMOTE(
                    sampling_strategy=sampling_strategy,
                    random_state=random_state
                )

            X_tr, y_tr = sampler.fit_resample(X_tr, y_tr)
        

        # if use_smote:
        #     if categorical_cols is None:
        #         raise ValueError("categorical_cols must be provided when use_smote=True")

        #     cat_indices = [X_tr.columns.get_loc(c) for c in categorical_cols if c in X_tr.columns]

        #     smote_nc = SMOTENC(
        #         categorical_features=cat_indices,
        #         sampling_strategy=sampling_strategy,
        #         random_state=random_state
        #     )
        #     X_tr, y_tr = smote_nc.fit_resample(X_tr, y_tr)

        print(f"\nFold {fold_id}")
        print("Validation countries:", sorted(pd.Series(groups.iloc[val_idx]).unique().tolist()))
        print("Validation class counts:")
        print(y_val.value_counts())

        folds.append((X_tr, X_val, y_tr, y_val))

    return folds