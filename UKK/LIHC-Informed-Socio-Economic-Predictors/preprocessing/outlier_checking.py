from outlier import plot_outliers_per_feature, report_outliers
import pandas as pd
from missing_values import fill_with_mean

from feature_engineering import get_categorical_features






path = r'.\ENABLE.EU_dataset_survey of households.xlsx'


datafram = pd.read_excel(path)

df = datafram.copy()
print(df.info())
df = df.drop(columns=['UID', 'T1', 'T1cluster', 'T3', 'T4', 'T5', 'NUTS1', 'NUTS2', 'NUTS3'])
df = fill_with_mean(df, df.columns)

categorical_features = get_categorical_features(df)
print("length of categorical features: ", len(categorical_features))
df = df.drop(columns=categorical_features)


print("**********************", df.isnull().sum())
print(df.info())
print(report_outliers(df))
plot_outliers_per_feature(df)