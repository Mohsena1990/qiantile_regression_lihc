import numpy as np




def missing_values(df):

    df = df.copy()
    features = df.columns
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        print(f"Warning: The following features are missing in the dataset: {missing_features}")
        features = [f for f in features if f in df.columns]

    return features























def cleaning(df):
    df['S9MONTH'] = df['S9MONTH'].replace([98, 99], np.nan).fillna(df['S9MONTH'].mode()[0])
    df['T5'] = df['T5'].fillna(df['T5'].median())
    df['T1cluster'] = df['T1cluster'].fillna(df['T1cluster'].median())
    df['NUTS1'] = df['NUTS1'].fillna(df['NUTS1'].mode()[0])
    df['NUTS2'] = df['NUTS2'].fillna(df['NUTS2'].mode()[0])
    df['NUTS3'] = df['NUTS3'].fillna(df['NUTS3'].mode()[0])
    df['SettlementSize'] = df['SettlementSize'].fillna(df['SettlementSize'].median())
    df['H7A1'] = df['H7A1'].fillna(df['H7A1'].median())
    df['H7A2'] = df['H7A2'].fillna(df['H7A2'].median())
    df['H9'] = df['H9'].fillna(df['H9'].mode()[0])
    df['H8A'] = df['H8A'].replace('#Null', np.nan).astype(float).fillna(df['H8A'].median())
    df['H8B'] = df['H8B'].replace('#Null', np.nan).astype(float).fillna(df['H8B'].median())