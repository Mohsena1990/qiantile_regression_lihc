





def stratify_by_country(df, country_col='Country', test_size=0.2, random_state=42):
    """Stratify train-test split by country to preserve distribution"""
    from sklearn.model_selection import train_test_split
    return train_test_split(df, test_size=test_size, stratify=df[country_col], random_state=random_state)