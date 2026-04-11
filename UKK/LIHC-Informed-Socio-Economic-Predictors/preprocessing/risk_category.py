

# # In risk_category.py
# from sklearn.cluster import DBSCAN
# from sklearn.metrics import silhouette_score
# from sklearn.preprocessing import StandardScaler
# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np

# def create_risk_category_40_70(df, income_bracket_col='income_bracket', exp_col='total_expenditure',  # Updated to match main
#                         efficiency_col='efficiency_score', demand_col='adjusted_energy_demand', 
#                         housing_cost_col='housing_cost_burden', eps=0.5, min_samples=5, random_state=42):
#     """DBSCAN clustering for LIHC-based risk categories using categorical income thresholds.
    
#     Args:
#         df (pd.DataFrame): Input dataframe with LIHC-aligned features.
#         income_bracket_col (str): Categorical income decile (default: 'income_bracket', 1-10).
#         exp_col (str): Total energy expenditure (default: 'total_expenditure').
#         ... (other args same)
#     """
#     # Input validation
#     required_cols = [income_bracket_col, exp_col, efficiency_col, demand_col, housing_cost_col]
#     if df.empty or df[required_cols].isna().all().all():
#         raise ValueError("Clustering data is empty or contains all NaN values. Check preprocessing.")
#     for col in required_cols:
#         if col not in df.columns:
#             raise KeyError(f"Column '{col}' not found. Ensure preprocessing includes it.")

#     # Prepare data (exclude categorical income_bracket)
#     X_data = df[[exp_col, efficiency_col, demand_col, housing_cost_col]].copy()

#     # Handle NaN/inf values
#     X_data = X_data.replace([np.inf, -np.inf], np.nan)
#     X_data = X_data.fillna(X_data.median())

#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X_data)

#     # DBSCAN clustering with tunable parameters
#     dbscan = DBSCAN(eps=eps, min_samples=min_samples)
#     df = df.assign(cluster=dbscan.fit_predict(X_scaled))  # -1 = noise

#     # Validate clustering
#     core_samples_mask = np.zeros_like(df['cluster'], dtype=bool)
#     core_samples_mask[df['cluster'] != -1] = True
#     if np.sum(core_samples_mask) > 1:
#         silhouette = silhouette_score(X_scaled[core_samples_mask], df['cluster'][core_samples_mask])
#         print(f"Silhouette Score (core samples): {silhouette:.2f} (>0.5 ideal)")
#     else:
#         print("Insufficient core samples for silhouette; try adjusting eps or min_samples.")

#     # Compute country-specific thresholds
#     # Change to income threshold at 40th percentile (low income if below)
#     income_threshold = df.groupby('Country')[income_bracket_col].transform(lambda x: x.quantile(0.4))
#     df['low_income'] = df[income_bracket_col].apply(lambda x: float(x) <= income_threshold.iloc[0] if pd.notna(x) else False)
#     exp_threshold = df.groupby('Country')[exp_col].transform(lambda x: x.quantile(0.7))  # 70th percentile

#     # Assign risk categories
#     df['risk_category'] = np.where(
#         (df['low_income']) & (df[exp_col] > exp_threshold), 'Double risk',
#         np.where(df['low_income'], 'Income risk',
#                  np.where(df[exp_col] > exp_threshold, 'Expenditure risk', 'No risk'))
#     )

#     # Plot with categorical adjustment
#     plt.figure(figsize=(8, 6))
#     scatter = plt.scatter(df[exp_col], df[income_bracket_col].astype(float), c=df['cluster'], cmap='viridis', alpha=0.6)
#     plt.axhline(y=income_threshold.median(), color='black', linestyle='--', label='Income Threshold (40th percentile)')  # Updated
#     plt.axvline(x=exp_threshold.median(), color='red', linestyle='--', label='Energy Threshold (70th percentile)')
#     plt.xlabel('Total Energy Expenditure (€)')
#     plt.ylabel('Income Decile (1-10)')
#     plt.title('DBSCAN Clusters with LIHC Risk Categories')
#     plt.colorbar(scatter, label='Cluster (noise=-1)')
#     plt.legend()
#     plt.savefig('LIHC_DBSCAN_Clusters.png', dpi=300, bbox_inches='tight')
#     plt.close()

#     # Validation
#     print("Cluster Distribution (incl. noise=-1):\n", df['cluster'].value_counts())
#     print("Risk Category Distribution:\n", df['risk_category'].value_counts())
#     print("Risk Category Crosstab:\n", pd.crosstab(df['cluster'], df['risk_category']))

#     return df

# def create_risk_category_30_70(df, income_bracket_col='income_bracket', exp_col='total_expenditure',  # Updated to match main
#                         efficiency_col='efficiency_score', demand_col='adjusted_energy_demand', 
#                         housing_cost_col='housing_cost_burden', eps=0.5, min_samples=5, random_state=42):
#     """DBSCAN clustering for LIHC-based risk categories using categorical income thresholds.
    
#     Args:
#         df (pd.DataFrame): Input dataframe with LIHC-aligned features.
#         income_bracket_col (str): Categorical income decile (default: 'income_bracket', 1-10).
#         exp_col (str): Total energy expenditure (default: 'total_expenditure').
#         ... (other args same)
#     """
#     # Input validation
#     required_cols = [income_bracket_col, exp_col, efficiency_col, demand_col, housing_cost_col]
#     if df.empty or df[required_cols].isna().all().all():
#         raise ValueError("Clustering data is empty or contains all NaN values. Check preprocessing.")
#     for col in required_cols:
#         if col not in df.columns:
#             raise KeyError(f"Column '{col}' not found. Ensure preprocessing includes it.")

#     # Prepare data (exclude categorical income_bracket)
#     X_data = df[[exp_col, efficiency_col, demand_col, housing_cost_col]].copy()

#     # Handle NaN/inf values
#     X_data = X_data.replace([np.inf, -np.inf], np.nan)
#     X_data = X_data.fillna(X_data.median())

#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X_data)

#     # DBSCAN clustering with tunable parameters
#     dbscan = DBSCAN(eps=eps, min_samples=min_samples)
#     df = df.assign(cluster=dbscan.fit_predict(X_scaled))  # -1 = noise

#     # Validate clustering
#     core_samples_mask = np.zeros_like(df['cluster'], dtype=bool)
#     core_samples_mask[df['cluster'] != -1] = True
#     if np.sum(core_samples_mask) > 1:
#         silhouette = silhouette_score(X_scaled[core_samples_mask], df['cluster'][core_samples_mask])
#         print(f"Silhouette Score (core samples): {silhouette:.2f} (>0.5 ideal)")
#     else:
#         print("Insufficient core samples for silhouette; try adjusting eps or min_samples.")

#     # Compute country-specific thresholds
#     # Change to income threshold at 30th percentile (low income if below)
#     income_threshold = df.groupby('Country')[income_bracket_col].transform(lambda x: x.quantile(0.3))
#     df['low_income'] = df[income_bracket_col].apply(lambda x: float(x) <= income_threshold.iloc[0] if pd.notna(x) else False)
#     exp_threshold = df.groupby('Country')[exp_col].transform(lambda x: x.quantile(0.7))  # 70th percentile

#     # Assign risk categories
#     df['risk_category'] = np.where(
#         (df['low_income']) & (df[exp_col] > exp_threshold), 'Double risk',
#         np.where(df['low_income'], 'Income risk',
#                  np.where(df[exp_col] > exp_threshold, 'Expenditure risk', 'No risk'))
#     )

#     # Plot with categorical adjustment
#     plt.figure(figsize=(8, 6))
#     scatter = plt.scatter(df[exp_col], df[income_bracket_col].astype(float), c=df['cluster'], cmap='viridis', alpha=0.6)
#     plt.axhline(y=income_threshold.median(), color='black', linestyle='--', label='Income Threshold (30th percentile)')  # Updated
#     plt.axvline(x=exp_threshold.median(), color='red', linestyle='--', label='Energy Threshold (70th percentile)')
#     plt.xlabel('Total Energy Expenditure (€)')
#     plt.ylabel('Income Decile (1-10)')
#     plt.title('DBSCAN Clusters with LIHC Risk Categories')
#     plt.colorbar(scatter, label='Cluster (noise=-1)')
#     plt.legend()
#     plt.savefig('LIHC_DBSCAN_Clusters.png', dpi=300, bbox_inches='tight')
#     plt.close()

#     # Validation
#     print("Cluster Distribution (incl. noise=-1):\n", df['cluster'].value_counts())
#     print("Risk Category Distribution:\n", df['risk_category'].value_counts())
#     print("Risk Category Crosstab:\n", pd.crosstab(df['cluster'], df['risk_category']))

#     return df



# def create_risk_category_45_70(df, income_bracket_col='income_bracket', exp_col='total_expenditure',  # Updated to match main
#                         efficiency_col='efficiency_score', demand_col='adjusted_energy_demand', 
#                         housing_cost_col='housing_cost_burden', eps=0.5, min_samples=5, random_state=42):
#     """DBSCAN clustering for LIHC-based risk categories using categorical income thresholds.
    
#     Args:
#         df (pd.DataFrame): Input dataframe with LIHC-aligned features.
#         income_bracket_col (str): Categorical income decile (default: 'income_bracket', 1-10).
#         exp_col (str): Total energy expenditure (default: 'total_expenditure').
#         ... (other args same)
#     """
#     # Input validation
#     required_cols = [income_bracket_col, exp_col, efficiency_col, demand_col, housing_cost_col]
#     if df.empty or df[required_cols].isna().all().all():
#         raise ValueError("Clustering data is empty or contains all NaN values. Check preprocessing.")
#     for col in required_cols:
#         if col not in df.columns:
#             raise KeyError(f"Column '{col}' not found. Ensure preprocessing includes it.")

#     # Prepare data (exclude categorical income_bracket)
#     X_data = df[[exp_col, efficiency_col, demand_col, housing_cost_col]].copy()

#     # Handle NaN/inf values
#     X_data = X_data.replace([np.inf, -np.inf], np.nan)
#     X_data = X_data.fillna(X_data.median())

#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X_data)

#     # DBSCAN clustering with tunable parameters
#     dbscan = DBSCAN(eps=eps, min_samples=min_samples)
#     df = df.assign(cluster=dbscan.fit_predict(X_scaled))  # -1 = noise

#     # Validate clustering
#     core_samples_mask = np.zeros_like(df['cluster'], dtype=bool)
#     core_samples_mask[df['cluster'] != -1] = True
#     if np.sum(core_samples_mask) > 1:
#         silhouette = silhouette_score(X_scaled[core_samples_mask], df['cluster'][core_samples_mask])
#         print(f"Silhouette Score (core samples): {silhouette:.2f} (>0.5 ideal)")
#     else:
#         print("Insufficient core samples for silhouette; try adjusting eps or min_samples.")

#     # Compute country-specific thresholds
#     # Change to decile <= 4.5 (top 45% as low income)
#     df['low_income'] = df[income_bracket_col].apply(lambda x: int(x) <= 4.5 if pd.notna(x) else False)
#     exp_threshold = df.groupby('Country')[exp_col].transform(lambda x: x.quantile(0.7))  # 70th percentile

#     # Assign risk categories
#     df['risk_category'] = np.where(
#         (df['low_income']) & (df[exp_col] > exp_threshold), 'Double risk',
#         np.where(df['low_income'], 'Income risk',
#                  np.where(df[exp_col] > exp_threshold, 'Expenditure risk', 'No risk'))
#     )

#     # Plot with categorical adjustment
#     plt.figure(figsize=(8, 6))
#     scatter = plt.scatter(df[exp_col], df[income_bracket_col].astype(float), c=df['cluster'], cmap='viridis', alpha=0.6)
#     plt.axhline(y=4.5, color='black', linestyle='--', label='Income Threshold (decile <= 4.5)')  # Updated label
#     plt.axvline(x=exp_threshold.median(), color='red', linestyle='--', label='Energy Threshold (70th percentile)')  # Updated label
#     plt.xlabel('Total Energy Expenditure (€)')
#     plt.ylabel('Income Decile (1-10)')
#     plt.title('DBSCAN Clusters with LIHC Risk Categories')
#     plt.colorbar(scatter, label='Cluster (noise=-1)')
#     plt.legend()
#     plt.savefig('LIHC_DBSCAN_Clusters.png', dpi=300, bbox_inches='tight')
#     plt.close()

#     # Validation
#     print("Cluster Distribution (incl. noise=-1):\n", df['cluster'].value_counts())
#     print("Risk Category Distribution:\n", df['risk_category'].value_counts())
#     print("Risk Category Crosstab:\n", pd.crosstab(df['cluster'], df['risk_category']))

#     return df



# def create_risk_category_50_60(df, income_bracket_col='income_bracket', exp_col='total_expenditure',  # Updated to match main
#                         efficiency_col='efficiency_score', demand_col='adjusted_energy_demand', 
#                         housing_cost_col='housing_cost_burden', eps=0.5, min_samples=5, random_state=42):
#     """DBSCAN clustering for LIHC-based risk categories using categorical income thresholds.
    
#     Args:
#         df (pd.DataFrame): Input dataframe with LIHC-aligned features.
#         income_bracket_col (str): Categorical income decile (default: 'income_bracket', 1-10).
#         exp_col (str): Total energy expenditure (default: 'total_expenditure').
#         ... (other args same)
#     """
#     # Input validation
#     required_cols = [income_bracket_col, exp_col, efficiency_col, demand_col, housing_cost_col]
#     if df.empty or df[required_cols].isna().all().all():
#         raise ValueError("Clustering data is empty or contains all NaN values. Check preprocessing.")
#     for col in required_cols:
#         if col not in df.columns:
#             raise KeyError(f"Column '{col}' not found. Ensure preprocessing includes it.")

#     # Prepare data (exclude categorical income_bracket)
#     X_data = df[[exp_col, efficiency_col, demand_col, housing_cost_col]].copy()

#     # Handle NaN/inf values
#     X_data = X_data.replace([np.inf, -np.inf], np.nan)
#     X_data = X_data.fillna(X_data.median())

#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X_data)

#     # DBSCAN clustering with tunable parameters
#     dbscan = DBSCAN(eps=eps, min_samples=min_samples)
#     df = df.assign(cluster=dbscan.fit_predict(X_scaled))  # -1 = noise

#     # Validate clustering
#     core_samples_mask = np.zeros_like(df['cluster'], dtype=bool)
#     core_samples_mask[df['cluster'] != -1] = True
#     if np.sum(core_samples_mask) > 1:
#         silhouette = silhouette_score(X_scaled[core_samples_mask], df['cluster'][core_samples_mask])
#         print(f"Silhouette Score (core samples): {silhouette:.2f} (>0.5 ideal)")
#     else:
#         print("Insufficient core samples for silhouette; try adjusting eps or min_samples.")

#     # Compute country-specific thresholds
#     # Change to decile < 5 out of 10 (top 50%)
#     df['low_income'] = df[income_bracket_col].apply(lambda x: int(x) < 5 if pd.notna(x) else False)
#     exp_threshold = df.groupby('Country')[exp_col].transform(lambda x: x.quantile(0.6))  # 60th percentile

#     # Assign risk categories
#     df['risk_category'] = np.where(
#         (df['low_income']) & (df[exp_col] > exp_threshold), 'Double risk',
#         np.where(df['low_income'], 'Income risk',
#                  np.where(df[exp_col] > exp_threshold, 'Expenditure risk', 'No risk'))
#     )

#     # Plot with categorical adjustment
#     plt.figure(figsize=(8, 6))
#     scatter = plt.scatter(df[exp_col], df[income_bracket_col].astype(float), c=df['cluster'], cmap='viridis', alpha=0.6)
#     plt.axhline(y=4.5, color='black', linestyle='--', label='Income Threshold (decile < 5)')  # Between 4 and 5
#     plt.axvline(x=exp_threshold.median(), color='red', linestyle='--', label='Energy Threshold (60th percentile)')
#     plt.xlabel('Total Energy Expenditure (€)')
#     plt.ylabel('Income Decile (1-10)')
#     plt.title('DBSCAN Clusters with LIHC Risk Categories')
#     plt.colorbar(scatter, label='Cluster (noise=-1)')
#     plt.legend()
#     plt.savefig('LIHC_DBSCAN_Clusters.png', dpi=300, bbox_inches='tight')
#     plt.close()

#     # Validation
#     print("Cluster Distribution (incl. noise=-1):\n", df['cluster'].value_counts())
#     print("Risk Category Distribution:\n", df['risk_category'].value_counts())
#     print("Risk Category Crosstab:\n", pd.crosstab(df['cluster'], df['risk_category']))

#     return df

# def create_risk_category(df, income_bracket_col='income_bracket', exp_col='total_expenditure',  # Updated to match main
#                         efficiency_col='efficiency_score', demand_col='adjusted_energy_demand', 
#                         housing_cost_col='housing_cost_burden', eps=0.5, min_samples=5, random_state=42):
#     """DBSCAN clustering for LIHC-based risk categories using categorical income thresholds.
    
#     Args:
#         df (pd.DataFrame): Input dataframe with LIHC-aligned features.
#         income_bracket_col (str): Categorical income decile (default: 'income_bracket', 1-10).
#         exp_col (str): Total energy expenditure (default: 'total_expenditure').
#         ... (other args same)
#     """
#     # Input validation
#     required_cols = [income_bracket_col, exp_col, efficiency_col, demand_col, housing_cost_col]
#     if df.empty or df[required_cols].isna().all().all():
#         raise ValueError("Clustering data is empty or contains all NaN values. Check preprocessing.")
#     for col in required_cols:
#         if col not in df.columns:
#             raise KeyError(f"Column '{col}' not found. Ensure preprocessing includes it.")

#     # Prepare data (exclude categorical income_bracket)

#     # Prepare data (exclude categorical income_bracket)
#     X_data = df[[exp_col, efficiency_col, demand_col, housing_cost_col]].copy()

#     # Handle NaN/inf values
#     X_data = X_data.replace([np.inf, -np.inf], np.nan)
#     X_data = X_data.fillna(X_data.median())

#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X_data)

#     # X_data = df[[exp_col, efficiency_col, demand_col, housing_cost_col]].copy()
#     # scaler = StandardScaler()
#     # X_scaled = scaler.fit_transform(X_data)

#     # DBSCAN clustering with tunable parameters
#     dbscan = DBSCAN(eps=eps, min_samples=min_samples)
#     df = df.assign(cluster=dbscan.fit_predict(X_scaled))  # -1 = noise

#     # Validate clustering
#     core_samples_mask = np.zeros_like(df['cluster'], dtype=bool)
#     core_samples_mask[df['cluster'] != -1] = True
#     if np.sum(core_samples_mask) > 1:
#         silhouette = silhouette_score(X_scaled[core_samples_mask], df['cluster'][core_samples_mask])
#         print(f"Silhouette Score (core samples): {silhouette:.2f} (>0.5 ideal)")
#     else:
#         print("Insufficient core samples for silhouette; try adjusting eps or min_samples.")

#     # Compute country-specific thresholds
#     df['low_income'] = df[income_bracket_col].apply(lambda x: int(x) < 4 if pd.notna(x) else False)
#     exp_threshold = df.groupby('Country')[exp_col].transform(lambda x: x.quantile(0.8))

#     # Assign risk categories
#     df['risk_category'] = np.where(
#         (df['low_income']) & (df[exp_col] > exp_threshold), 'Double risk',
#         np.where(df['low_income'], 'Income risk',
#                  np.where(df[exp_col] > exp_threshold, 'Expenditure risk', 'No risk'))
#     )

#     # Plot with categorical adjustment
#     plt.figure(figsize=(8, 6))
#     scatter = plt.scatter(df[exp_col], df[income_bracket_col].astype(float), c=df['cluster'], cmap='viridis', alpha=0.6)
#     plt.axhline(y=3.5, color='black', linestyle='--', label='Income Threshold (decile < 4)')  # Between 3 and 4
#     plt.axvline(x=exp_threshold.median(), color='red', linestyle='--', label='Energy Threshold (80th percentile)')
#     plt.xlabel('Total Energy Expenditure (€)')
#     plt.ylabel('Income Decile (1-10)')
#     plt.title('DBSCAN Clusters with LIHC Risk Categories')
#     plt.colorbar(scatter, label='Cluster (noise=-1)')
#     plt.legend()
#     plt.savefig('LIHC_DBSCAN_Clusters.png', dpi=300, bbox_inches='tight')
#     plt.close()

#     # Validation
#     print("Cluster Distribution (incl. noise=-1):\n", df['cluster'].value_counts())
#     print("Risk Category Distribution:\n", df['risk_category'].value_counts())
#     print("Risk Category Crosstab:\n", pd.crosstab(df['cluster'], df['risk_category']))

#     return df



# # In risk_category.py
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# def create_risk_category_paper_like(df, income_bracket_col='income_bracket', exp_col='total_expenditure'):
#     """Assign LIHC-based risk categories using only income and expenditure thresholds.
    
#     Args:
#         df (pd.DataFrame): Input dataframe with income and expenditure data.
#         income_bracket_col (str): Categorical income decile (default: 'income_bracket', 1-10).
#         exp_col (str): Total energy expenditure (default: 'total_expenditure').
    
#     Returns:
#         pd.DataFrame: Dataframe with added 'risk_category' column.
#     """
#     # Input validation
#     required_cols = [income_bracket_col, exp_col]
#     if df.empty or df[required_cols].isna().all().all():
#         raise ValueError("Categorization data is empty or contains all NaN values. Check preprocessing.")
#     for col in required_cols:
#         if col not in df.columns:
#             raise KeyError(f"Column '{col}' not found. Ensure preprocessing includes it.")

#     # Compute country-specific thresholds
#     df['low_income'] = df[income_bracket_col].apply(lambda x: int(x) < 4 if pd.notna(x) else False)
#     exp_threshold = df.groupby('Country')[exp_col].transform(lambda x: x.quantile(0.8))

#     # Assign risk categories
#     df['risk_category'] = np.where(
#         (df['low_income']) & (df[exp_col] > exp_threshold), 'Double risk',
#         np.where(df['low_income'], 'Income risk',
#                  np.where(df[exp_col] > exp_threshold, 'Expenditure risk', 'No risk'))
#     )

#     # Plot distribution
#     plt.figure(figsize=(8, 6))
#     scatter = plt.scatter(df[exp_col], df[income_bracket_col].astype(float), c=df['risk_category'].map({'No risk': 0, 'Income risk': 1, 'Expenditure risk': 2, 'Double risk': 3}), cmap='viridis', alpha=0.6)
#     plt.axhline(y=3.5, color='black', linestyle='--', label='Income Threshold (decile < 4)')
#     plt.axvline(x=exp_threshold.median(), color='red', linestyle='--', label='Energy Threshold (80th percentile)')
#     plt.xlabel('Total Energy Expenditure (€)')
#     plt.ylabel('Income Decile (1-10)')
#     plt.title('LIHC Risk Categories')
#     plt.colorbar(scatter, label='Risk Category (0=No, 1=Income, 2=Expenditure, 3=Double)')
#     plt.legend()
#     plt.savefig('LIHC_Risk_Categories.png', dpi=300, bbox_inches='tight')
#     plt.close()

#     # Validation
#     print("Risk Category Distribution:\n", df['risk_category'].value_counts())

#     return df



# # In risk_category.py
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import silhouette_score

# def create_risk_category_kmeans(df, income_bracket_col='income_bracket', exp_col='total_expenditure', n_clusters=11, random_state=42):
#     """KMeans clustering followed by LIHC-based risk categorization using income and expenditure.
    
#     Args:
#         df (pd.DataFrame): Input dataframe with income and expenditure data.
#         income_bracket_col (str): Categorical income decile (default: 'income_bracket', 1-10).
#         exp_col (str): Total energy expenditure (default: 'total_expenditure').
#         n_clusters (int): Number of clusters for KMeans (default: 4).
#         random_state (int): Random seed for reproducibility (default: 42).
    
#     Returns:
#         pd.DataFrame: Dataframe with added 'cluster' and 'risk_category' columns.
#     """
#     # Input validation
#     required_cols = [income_bracket_col, exp_col]
#     if df.empty or df[required_cols].isna().all().all():
#         raise ValueError("Categorization data is empty or contains all NaN values. Check preprocessing.")
#     for col in required_cols:
#         if col not in df.columns:
#             raise KeyError(f"Column '{col}' not found. Ensure preprocessing includes it.")

#     # Prepare data
#     X_data = df[[income_bracket_col, exp_col]].copy()
#     X_data[income_bracket_col] = X_data[income_bracket_col].astype(float)  # Ensure numeric
#     X_data = X_data.replace([np.inf, -np.inf], np.nan).fillna(X_data.median())  # Handle NaN/inf

#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X_data)

#     # KMeans clustering
#     kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
#     df = df.assign(cluster=kmeans.fit_predict(X_scaled))

#     # Validate clustering
#     silhouette = silhouette_score(X_scaled, df['cluster'])
#     print(f"Silhouette Score: {silhouette:.2f} (>0.5 ideal for good separation)")

#     # Compute country-specific thresholds
#     df['low_income'] = df[income_bracket_col].apply(lambda x: int(x) < 4 if pd.notna(x) else False)
#     exp_threshold = df.groupby('Country')[exp_col].transform(lambda x: x.quantile(0.8))

#     # Assign risk categories
#     df['risk_category'] = np.where(
#         (df['low_income']) & (df[exp_col] > exp_threshold), 'Double risk',
#         np.where(df['low_income'], 'Income risk',
#                  np.where(df[exp_col] > exp_threshold, 'Expenditure risk', 'No risk'))
#     )

#     # Plot with categorical adjustment
#     plt.figure(figsize=(8, 6))
#     scatter = plt.scatter(df[exp_col], df[income_bracket_col].astype(float), c=df['cluster'], cmap='viridis', alpha=0.6)
#     plt.axhline(y=3.5, color='black', linestyle='--', label='Income Threshold (decile < 4)')
#     plt.axvline(x=exp_threshold.median(), color='red', linestyle='--', label='Energy Threshold (80th percentile)')
#     plt.xlabel('Total Energy Expenditure (€)')
#     plt.ylabel('Income Decile (1-10)')
#     plt.title('KMeans Clusters with LIHC Risk Categories')
#     plt.colorbar(scatter, label='Cluster')
#     plt.legend()
#     plt.savefig('LIHC_KMeans_Clusters.png', dpi=300, bbox_inches='tight')
#     plt.close()

#     # Validation
#     print("Cluster Distribution:\n", df['cluster'].value_counts())
#     print("Risk Category Distribution:\n", df['risk_category'].value_counts())
#     print("Risk Category Crosstab:\n", pd.crosstab(df['cluster'], df['risk_category']))

#     return df


# def create_risk_category_per_country_new(df, income_bracket_col='income_bracket', exp_col='total_expenditure',
#                          efficiency_col='efficiency_score', demand_col='adjusted_energy_demand',
#                          housing_cost_col='housing_cost_burden', eps=0.5, min_samples=5, random_state=42):
#     # ... (input validation remains)

#     # Prepare data
#     X_data = df[[exp_col, efficiency_col, demand_col, housing_cost_col]].copy()
#     X_data = X_data.replace([np.inf, -np.inf], np.nan).fillna(X_data.median())
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X_data)

#     # Country-specific clustering
#     df['cluster'] = -1  # Initialize as noise
#     for country, group in df.groupby('Country'):
#         if len(group) < min_samples * 2:  # Skip small groups
#             print(f"Skipping clustering for {country}: too few samples ({len(group)})")
#             continue
#         idx = group.index
#         X_group = X_scaled[df.index.get_indexer(idx)]  # Adjust index for slicing

#         # X_group = X_scaled[idx - df.index.min()]  # Adjust index for slicing
#         dbscan = DBSCAN(eps=eps, min_samples=min_samples)
#         df.loc[idx, 'cluster'] = dbscan.fit_predict(X_group)

#     # Validate clustering overall
#     core_samples_mask = df['cluster'] != -1
#     if core_samples_mask.sum() > 1:
#         silhouette = silhouette_score(X_scaled[core_samples_mask], df.loc[core_samples_mask, 'cluster'])
#         print(f"Overall Silhouette Score (core samples): {silhouette:.2f} (>0.5 ideal)")
#     else:
#         print("Insufficient core samples overall; consider adjusting parameters or merging countries.")

#     # Compute country-specific thresholds
#     df['low_income'] = df[income_bracket_col].apply(lambda x: int(x) < 4 if pd.notna(x) else False)
#     exp_threshold = df.groupby('Country')[exp_col].transform(lambda x: x.quantile(0.8))

#     # Assign risk categories
#     df['risk_category'] = np.where(
#         (df['low_income']) & (df[exp_col] > exp_threshold), 'Double risk',
#         np.where(df['low_income'], 'Income risk',
#                  np.where(df[exp_col] > exp_threshold, 'Expenditure risk', 'No risk'))
#     )

#     # Handle missing classes per country
#     for country, group in df.groupby('Country'):
#         class_counts = group['risk_category'].value_counts()
#         missing_classes = set(['No risk', 'Income risk', 'Expenditure risk', 'Double risk']) - set(class_counts.index)
#         if missing_classes:
#             print(f"{country} missing classes: {missing_classes}. Consider merging rare classes or oversampling.")
#             # Example: Merge 'Double risk' with 'Income risk' if negligible
#             if 'Double risk' not in class_counts or class_counts['Double risk'] < 5:
#                 df.loc[group.index, 'risk_category'] = df.loc[group.index, 'risk_category'].replace('Double risk', 'Income risk')

#     # Plot (global for simplicity)
#     plt.figure(figsize=(8, 6))
#     scatter = plt.scatter(df[exp_col], df[income_bracket_col].astype(float), c=df['cluster'], cmap='viridis', alpha=0.6)
#     plt.axhline(y=3.5, color='black', linestyle='--', label='Income Threshold (decile < 4)')
#     plt.axvline(x=exp_threshold.median(), color='red', linestyle='--', label='Energy Threshold (80th percentile)')
#     plt.xlabel('Total Energy Expenditure (€)')
#     plt.ylabel('Income Decile (1-10)')
#     plt.title('DBSCAN Clusters with LIHC Risk Categories')
#     plt.colorbar(scatter, label='Cluster (noise=-1)')
#     plt.legend()
#     plt.savefig('LIHC_DBSCAN_Clusters.png', dpi=300, bbox_inches='tight')
#     plt.close()

#     # Validation
#     print("Cluster Distribution (incl. noise=-1):\n", df['cluster'].value_counts())
#     print("Risk Category Distribution:\n", df['risk_category'].value_counts())
#     print("Risk Category Crosstab:\n", pd.crosstab(df['cluster'], df['risk_category']))

#     return df



# def new_create_risk_category(df, income_bracket_col='income_bracket', exp_col='total_expenditure',
#                              efficiency_col='efficiency_score', demand_col='adjusted_energy_demand',
#                              housing_cost_col='housing_cost_burden', eps=0.5, min_samples=5, random_state=42):
#     """
#     DBSCAN clustering for LIHC-based risk categories with country-specific scaling and thresholds.
#     This version avoids global pooling and ensures clustering is relative to each country's context.

#     Args:
#         df (pd.DataFrame): Input dataframe with LIHC-aligned features and a 'Country' column.
#         income_bracket_col (str): Income decile indicator (default: 'income_bracket', 1-10).
#         exp_col (str): Total energy expenditure (default: 'total_expenditure').
#         efficiency_col (str): Dwelling energy efficiency metric.
#         demand_col (str): Adjusted energy demand variable.
#         housing_cost_col (str): Housing cost burden variable.
#         eps (float): DBSCAN epsilon parameter (default: 0.5).
#         min_samples (int): Minimum samples per cluster (default: 5).
#         random_state (int): Random seed for reproducibility.
#     """
    

#     required_cols = ['Country', income_bracket_col, exp_col, efficiency_col, demand_col, housing_cost_col]
#     for col in required_cols:
#         if col not in df.columns:
#             raise KeyError(f"Column '{col}' not found in dataframe.")
#     if df.empty:
#         raise ValueError("Input dataframe is empty.")
    
#     results = []
#     silhouette_scores = []

#     # Process each country separately
#     for country, group in df.groupby('Country', group_keys=False):
#         print(f"\nProcessing country: {country} ({len(group)} records)")

#         # Prepare and clean data
#         X_data = group[[exp_col, efficiency_col, demand_col, housing_cost_col]].copy()
#         X_data = X_data.replace([np.inf, -np.inf], np.nan).fillna(X_data.median())

#         # Standardize within-country to preserve local scale
#         scaler = StandardScaler()
#         X_scaled = scaler.fit_transform(X_data)

#         # Run DBSCAN clustering
#         dbscan = DBSCAN(eps=eps, min_samples=min_samples)
#         group['cluster'] = dbscan.fit_predict(X_scaled)

#         # Compute silhouette score for diagnostic (if clusters exist)
#         core_mask = group['cluster'] != -1
#         if core_mask.sum() > 1 and len(group['cluster'].unique()) > 1:
#             score = silhouette_score(X_scaled[core_mask], group['cluster'][core_mask])
#             silhouette_scores.append((country, score))
#             print(f"Silhouette score ({country}): {score:.2f}")
#         else:
#             print(f"No valid clusters for {country}. Consider adjusting eps/min_samples.")

#         # Compute country-specific thresholds (LIHC logic)
#         group['low_income'] = group[income_bracket_col].apply(lambda x: int(x) < 4 if pd.notna(x) else False)
#         exp_threshold = group[exp_col].quantile(0.65)  # 70th percentile per country

#         group['risk_category'] = np.select(
#             [
#                 (group['low_income']) & (group[exp_col] > exp_threshold),
#                 (group['low_income']),
#                 (group[exp_col] > exp_threshold)
#             ],
#             ['Double risk', 'Income risk', 'Expenditure risk'],
#             default='No risk'
#         )

#         # Add the country results
#         results.append(group)

#         # Optional: plot per-country clusters
#         plt.figure(figsize=(8, 6))
#         scatter = plt.scatter(group[exp_col], group[income_bracket_col].astype(float),
#                               c=group['cluster'], cmap='viridis', alpha=0.6)
#         plt.axhline(y=3.5, color='black', linestyle='--', label='Income Threshold (<4)')
#         plt.axvline(x=exp_threshold, color='red', linestyle='--', label=f'Expenditure Threshold (70th pct)')
#         plt.xlabel('Total Energy Expenditure (€)')
#         plt.ylabel('Income Decile (1-10)')
#         plt.title(f'DBSCAN Clusters and LIHC Risk Categories — {country}')
#         plt.colorbar(scatter, label='Cluster ID (noise=-1)')
#         plt.legend()
#         plt.savefig(f'LIHC_DBSCAN_{country}.png', dpi=300, bbox_inches='tight')
#         plt.close()

#         # Country summary
#         print(f"Risk Category Distribution for {country}:\n{group['risk_category'].value_counts()}\n")

#     # Combine all countries back together
#     df_out = pd.concat(results, ignore_index=True)

#     # Print global summary
#     if silhouette_scores:
#         mean_silhouette = np.mean([s for _, s in silhouette_scores])
#         print(f"\nAverage silhouette score across countries: {mean_silhouette:.2f}")
#     print("\nGlobal Risk Category Distribution:\n", df_out['risk_category'].value_counts())

#     return df_out



# def log_new_create_risk_category(df, income_bracket_col='income_bracket', exp_col='log_expenditure',
#                              efficiency_col='efficiency_score', demand_col='adjusted_energy_demand',
#                              housing_cost_col='housing_cost_burden', eps=0.5, min_samples=5, random_state=42):
#     """
#     DBSCAN clustering for LIHC-based risk categories with country-specific scaling and thresholds.
#     This version avoids global pooling and ensures clustering is relative to each country's context.

#     Args:
#         df (pd.DataFrame): Input dataframe with LIHC-aligned features and a 'Country' column.
#         income_bracket_col (str): Income decile indicator (default: 'income_bracket', 1-10).
#         exp_col (str): log transferred Total energy expenditure (default: 'log_expenditure').
#         efficiency_col (str): Dwelling energy efficiency metric.
#         demand_col (str): Adjusted energy demand variable.
#         housing_cost_col (str): Housing cost burden variable.
#         eps (float): DBSCAN epsilon parameter (default: 0.5).
#         min_samples (int): Minimum samples per cluster (default: 5).
#         random_state (int): Random seed for reproducibility.
#     """
    

#     required_cols = ['Country', income_bracket_col, exp_col, efficiency_col, demand_col, housing_cost_col]
#     for col in required_cols:
#         if col not in df.columns:
#             raise KeyError(f"Column '{col}' not found in dataframe.")
#     if df.empty:
#         raise ValueError("Input dataframe is empty.")
    
#     results = []
#     silhouette_scores = []

#     # Process each country separately
#     for country, group in df.groupby('Country', group_keys=False):
#         print(f"\nProcessing country: {country} ({len(group)} records)")

#         # Prepare and clean data
#         X_data = group[[exp_col, efficiency_col, demand_col, housing_cost_col]].copy()
#         X_data = X_data.replace([np.inf, -np.inf], np.nan).fillna(X_data.median())

#         # Standardize within-country to preserve local scale
#         scaler = StandardScaler()
#         X_scaled = scaler.fit_transform(X_data)

#         # Run DBSCAN clustering
#         dbscan = DBSCAN(eps=eps, min_samples=min_samples)
#         group['cluster'] = dbscan.fit_predict(X_scaled)

#         # Compute silhouette score for diagnostic (if clusters exist)
#         core_mask = group['cluster'] != -1
#         if core_mask.sum() > 1 and len(group['cluster'].unique()) > 1:
#             score = silhouette_score(X_scaled[core_mask], group['cluster'][core_mask])
#             silhouette_scores.append((country, score))
#             print(f"Silhouette score ({country}): {score:.2f}")
#         else:
#             print(f"No valid clusters for {country}. Consider adjusting eps/min_samples.")

#         # Compute country-specific thresholds (LIHC logic)
#         group['low_income'] = group[income_bracket_col].apply(lambda x: int(x) < 4 if pd.notna(x) else False)
#         exp_threshold = group[exp_col].quantile(0.7)  # 70th percentile per country

#         group['risk_category'] = np.select(
#             [
#                 (group['low_income']) & (group[exp_col] > exp_threshold),
#                 (group['low_income']),
#                 (group[exp_col] > exp_threshold)
#             ],
#             ['Double risk', 'Income risk', 'Expenditure risk'],
#             default='No risk'
#         )

#         # Add the country results
#         results.append(group)

#         # Optional: plot per-country clusters
#         plt.figure(figsize=(8, 6))
#         scatter = plt.scatter(group[exp_col], group[income_bracket_col].astype(float),
#                               c=group['cluster'], cmap='viridis', alpha=0.6)
#         plt.axhline(y=3.5, color='black', linestyle='--', label='Income Threshold (<4)')
#         plt.axvline(x=exp_threshold, color='red', linestyle='--', label=f'Expenditure Threshold (70th pct)')
#         plt.xlabel('Total Energy Expenditure (€)')
#         plt.ylabel('Income Decile (1-10)')
#         plt.title(f'DBSCAN Clusters and LIHC Risk Categories — {country}')
#         plt.colorbar(scatter, label='Cluster ID (noise=-1)')
#         plt.legend()
#         plt.savefig(f'LIHC_DBSCAN_{country}.png', dpi=300, bbox_inches='tight')
#         plt.close()

#         # Country summary
#         print(f"Risk Category Distribution for {country}:\n{group['risk_category'].value_counts()}\n")

#     # Combine all countries back together
#     df_out = pd.concat(results, ignore_index=True)

#     # Print global summary
#     if silhouette_scores:
#         mean_silhouette = np.mean([s for _, s in silhouette_scores])
#         print(f"\nAverage silhouette score across countries: {mean_silhouette:.2f}")
#     print("\nGlobal Risk Category Distribution:\n", df_out['risk_category'].value_counts())

#     return df_out


# import numpy as np
# import pandas as pd
# import statsmodels.formula.api as smf


# def quantile_expenditure_threshold(
#     df: pd.DataFrame,
#     country_col: str = "Country",
#     exp_col: str = "total_expenditure",
#     features: list = None,
#     quantile: float = 0.7,
#     min_rows: int = 30
# ) -> pd.DataFrame:
#     import statsmodels.formula.api as smf

#     if features is None or len(features) == 0:
#         raise ValueError("`features` must be a non-empty list of predictor column names.")

#     missing_cols = [c for c in [country_col, exp_col] + features if c not in df.columns]
#     if missing_cols:
#         raise KeyError(f"Missing required columns: {missing_cols}")

#     df = df.copy()
#     df["expected_exp"] = np.nan
#     df["relative_exp"] = np.nan
#     df["high_exp_flag"] = False
#     df["qr_valid_flag"] = False

#     for country, group in df.groupby(country_col):
#         try:
#             group = group.copy().replace([np.inf, -np.inf], np.nan)

#             valid_fit = group.dropna(subset=[exp_col] + features)
#             if len(valid_fit) < min_rows:
#                 print(f"⚠️ {country}: skipped (too few valid rows: {len(valid_fit)})")
#                 continue

#             predictors = " + ".join(features)
#             formula = f"{exp_col} ~ {predictors}"

#             model = smf.quantreg(formula, data=valid_fit)
#             res = model.fit(q=quantile, max_iter=5000, disp=False)

#             valid_pred = group.dropna(subset=features)
#             if len(valid_pred) == 0:
#                 print(f"⚠️ {country}: no rows available for prediction")
#                 continue

#             expected = pd.Series(index=group.index, dtype=float)
#             expected.loc[valid_pred.index] = res.predict(valid_pred)

#             df.loc[group.index, "expected_exp"] = expected
#             df.loc[group.index, "relative_exp"] = group[exp_col] - expected

#             high_mask = (group[exp_col] > expected).fillna(False)
#             df.loc[group.index, "high_exp_flag"] = high_mask
#             df.loc[group.index, "qr_valid_flag"] = True

#             print(
#                 f"[{country}] fitted q={quantile:.2f}, "
#                 f"pseudo_R2={getattr(res, 'prsquared', np.nan):.3f}, "
#                 f"high_exp={high_mask.mean():.2%}, n_fit={len(valid_fit)}"
#             )

#         except Exception as e:
#             print(f"⚠️ Quantile regression failed for {country}: {e}")

#     return df



# def hqrtm_lihc_categorize(
#     df: pd.DataFrame,
#     income_col: str = "equivalized_income",
#     exp_col: str = "total_expenditure",
#     country_col: str = "Country",
#     features: list = None,
#     quantile: float = 0.7,
#     min_rows: int = 30
# ) -> pd.DataFrame:
#     """
#     LIHC-style categorization using:
#       - low income: income below 60% of country median
#       - high expenditure: above country-specific quantile-regression threshold
#     """

#     if features is None or len(features) == 0:
#         raise ValueError("`features` must be provided to hqrtm_lihc_categorize().")

#     required_cols = [income_col, exp_col, country_col] + features
#     missing_cols = [c for c in required_cols if c not in df.columns]
#     if missing_cols:
#         raise KeyError(f"Missing required columns: {missing_cols}")

#     df = df.copy()
#     df = df.replace([np.inf, -np.inf], np.nan)

#     # 1. Country median income
#     df["national_median_income"] = df.groupby(country_col)[income_col].transform("median")

#     # 2. LIHC-style income threshold
#     df["lihc_income_threshold"] = df["national_median_income"] * 0.60
#     df["is_low_income"] = df[income_col] < df["lihc_income_threshold"]

#     # 3. Quantile regression expenditure threshold
#     df = quantile_expenditure_threshold(
#         df=df,
#         country_col=country_col,
#         exp_col=exp_col,
#         features=features,
#         quantile=quantile,
#         min_rows=min_rows
#     )

#     # 4. Four classes
#     conditions = [
#         (df["high_exp_flag"] & df["is_low_income"]),
#         (~df["high_exp_flag"] & df["is_low_income"]),
#         (df["high_exp_flag"] & ~df["is_low_income"]),
#         (~df["high_exp_flag"] & ~df["is_low_income"])
#     ]

#     choices = [
#         "LIHC_Extreme",
#         "Hidden_Poor",
#         "High_Expenditure_Secure",
#         "Energy_Secure"
#     ]

#     df["vulnerability_class"] = np.select(conditions, choices, default="Unknown")

#     return df



# risk_category.py

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def assign_traditional_lihc(
    df: pd.DataFrame,
    income_col: str = "equivalized_income",
    exp_col: str = "total_expenditure",
    country_col: str = "Country",
    income_rule: str = "country_median_60",
    exp_quantile: float = 0.80,
    income_bracket_col: str = "income_bracket",
    fit_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Traditional LIHC-style categorization.

    If fit_df is provided, thresholds are fit on fit_df and applied to df.
    """
    out = df.copy()
    fit_source = out if fit_df is None else fit_df.copy()

    required_cols = [country_col, exp_col]
    if income_rule == "country_median_60":
        required_cols.append(income_col)
    elif income_rule == "bracket_lt4":
        required_cols.append(income_bracket_col)
    else:
        raise ValueError("income_rule must be 'country_median_60' or 'bracket_lt4'")

    missing_cols = [c for c in required_cols if c not in out.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns: {missing_cols}")
    missing_cols_fit = [c for c in required_cols if c not in fit_source.columns]
    if missing_cols_fit:
        raise KeyError(f"Missing required columns in fit_df: {missing_cols_fit}")

    # ---------- low-income side ----------
    out["national_median_income"] = np.nan
    out["lihc_income_threshold"] = np.nan

    if income_rule == "country_median_60":
        medians = fit_source.groupby(country_col)[income_col].median()
        fallback_income_median = fit_source[income_col].median()
        out["national_median_income"] = out[country_col].map(medians).fillna(fallback_income_median)
        out["lihc_income_threshold"] = 0.60 * out["national_median_income"]
        out["low_income"] = out[income_col] < out["lihc_income_threshold"]
    else:
        out["low_income"] = out[income_bracket_col] < 4

    # ---------- expenditure side ----------
    exp_thresholds = fit_source.groupby(country_col)[exp_col].quantile(exp_quantile)
    fallback_exp_threshold = fit_source[exp_col].quantile(exp_quantile)
    out["exp_threshold"] = out[country_col].map(exp_thresholds).fillna(fallback_exp_threshold)
    out["high_exp_flag"] = out[exp_col] > out["exp_threshold"]

    # ---------- four classes ----------
    out["risk_category"] = np.select(
        [
            out["low_income"] & out["high_exp_flag"],
            out["low_income"] & ~out["high_exp_flag"],
            ~out["low_income"] & out["high_exp_flag"],
        ],
        [
            "Double risk",
            "Income risk",
            "Expenditure risk",
        ],
        default="No risk",
    )

    return out



# def assign_hqrtm(
#     df: pd.DataFrame,
#     qr_features: list,
#     income_col: str = "equivalized_income",
#     exp_col: str = "total_expenditure",
#     country_col: str = "Country",
#     income_rule: str = "country_median_60",
#     quantile: float = 0.65,
#     min_rows: int = 30,
#     income_bracket_col: str = "income_bracket",
# ) -> pd.DataFrame:
#     """
#     HQRTM categorization:
#     low income + conditional high expenditure via country-specific quantile regression.

#     Parameters
#     ----------
#     df : pd.DataFrame
#         Input dataframe.
#     qr_features : list
#         Structural predictors used in country-specific quantile regression.
#         Example:
#         [
#             "floor_area",
#             "house_age",
#             "household_size",
#             "house_detachment",
#             "has_insulation",
#             "heating_strategy",
#             "birth_year_respondent",
#         ]
#     income_col : str
#         Continuous income variable, preferably equivalized income.
#     exp_col : str
#         Annual energy expenditure variable.
#     country_col : str
#         Country/group variable.
#     income_rule : str
#         'country_median_60' -> low income if income < 60% of country median.
#         'bracket_lt4' -> low income if income_bracket < 4.
#     quantile : float
#         Quantile frontier for HQRTM.
#         Recommended sensitivity values: 0.60, 0.65, 0.70
#     min_rows : int
#         Minimum valid rows per country to fit QR.
#     income_bracket_col : str
#         Income bracket column, used only if income_rule='bracket_lt4'.

#     Returns
#     -------
#     pd.DataFrame
#         Copy of df with:
#         - national_median_income
#         - lihc_income_threshold
#         - low_income
#         - expected_exp
#         - relative_exp
#         - high_exp_flag
#         - qr_valid_flag
#         - risk_category
#     """
#     if qr_features is None or len(qr_features) == 0:
#         raise ValueError("qr_features must be a non-empty list")

#     if quantile not in [0.60, 0.65, 0.70]:
#         raise ValueError("For this study, quantile should be one of [0.60, 0.65, 0.70]")

#     out = df.copy()

#     required_cols = [country_col, exp_col] + qr_features
#     if income_rule == "country_median_60":
#         required_cols.append(income_col)
#     elif income_rule == "bracket_lt4":
#         required_cols.append(income_bracket_col)
#     else:
#         raise ValueError("income_rule must be 'country_median_60' or 'bracket_lt4'")

#     missing_cols = [c for c in required_cols if c not in out.columns]
#     if missing_cols:
#         raise KeyError(f"Missing required columns: {missing_cols}")

#     # ---------- low-income side ----------
#     out["national_median_income"] = np.nan
#     out["lihc_income_threshold"] = np.nan

#     if income_rule == "country_median_60":
#         out["national_median_income"] = out.groupby(country_col)[income_col].transform("median")
#         out["lihc_income_threshold"] = 0.60 * out["national_median_income"]
#         out["low_income"] = out[income_col] < out["lihc_income_threshold"]

#     elif income_rule == "bracket_lt4":
#         out["low_income"] = out[income_bracket_col] < 4

#     # ---------- conditional expenditure frontier ----------
#     out["expected_exp"] = np.nan
#     out["relative_exp"] = np.nan
#     out["high_exp_flag"] = False
#     out["qr_valid_flag"] = False

#     for country, group in out.groupby(country_col):
#         try:
#             group = group.copy()
#             group = group.replace([np.inf, -np.inf], np.nan)

#             valid_fit = group.dropna(subset=[exp_col] + qr_features)
#             if len(valid_fit) < min_rows:
#                 print(f"⚠️ {country}: skipped (too few valid rows: {len(valid_fit)})")
#                 continue

#             formula = f"{exp_col} ~ " + " + ".join(qr_features)
#             model = smf.quantreg(formula, data=valid_fit)
#             res = model.fit(q=quantile, max_iter=5000, disp=False)

#             valid_pred = group.dropna(subset=qr_features)
#             if len(valid_pred) == 0:
#                 print(f"⚠️ {country}: no rows available for prediction")
#                 continue

#             expected = pd.Series(index=group.index, dtype=float)
#             expected.loc[valid_pred.index] = res.predict(valid_pred)

#             out.loc[group.index, "expected_exp"] = expected
#             out.loc[group.index, "relative_exp"] = group[exp_col] - expected
#             margin = np.nanstd(group[exp_col]) * 0.1
#             out.loc[group.index, "high_exp_flag"] = (group[exp_col] > (expected + margin))
#             # out.loc[group.index, "high_exp_flag"] = (group[exp_col] > expected).fillna(False)
#             out.loc[group.index, "qr_valid_flag"] = True

#             print(
#                 f"[{country}] HQRTM q={quantile:.2f}, "
#                 f"pseudo_R2={getattr(res, 'prsquared', np.nan):.3f}, "
#                 f"high_exp={(group[exp_col] > expected).mean():.2%}, "
#                 f"n_fit={len(valid_fit)}"
#             )

#         except Exception as e:
#             print(f"⚠️ Quantile regression failed for {country}: {e}")

#     # ---------- four classes ----------
#     out["risk_category"] = np.select(
#         [
#             out["low_income"] & out["high_exp_flag"],
#             out["low_income"] & ~out["high_exp_flag"],
#             ~out["low_income"] & out["high_exp_flag"],
#         ],
#         [
#             "Double risk",
#             "Income risk",
#             "Expenditure risk",
#         ],
#         default="No risk",
#     )

#     return out

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def assign_hqrtm(
    df: pd.DataFrame,
    qr_features: list,
    income_col: str = "equivalized_income",
    exp_col: str = "total_expenditure",
    country_col: str = "Country",
    income_rule: str = "country_median_60",
    quantile: float = 0.65,
    min_rows: int = 100,
    income_bracket_col: str = "income_bracket",
    add_country_effects: bool = True,
    margin_scale: float = 0.10,
    fit_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    HQRTM categorization with pooled quantile regression.

    If fit_df is provided, quantile model and margins are fit on fit_df and
    then applied to df.
    """
    if not qr_features:
        raise ValueError("qr_features must be a non-empty list")

    if quantile not in [0.60, 0.65, 0.70]:
        raise ValueError("For this study, quantile should be one of [0.60, 0.65, 0.70]")

    out = df.copy()
    fit_source = out if fit_df is None else fit_df.copy()

    required_cols = [country_col, exp_col] + qr_features
    if income_rule == "country_median_60":
        required_cols.append(income_col)
    elif income_rule == "bracket_lt4":
        required_cols.append(income_bracket_col)
    else:
        raise ValueError("income_rule must be 'country_median_60' or 'bracket_lt4'")

    missing_cols = [c for c in required_cols if c not in out.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns: {missing_cols}")
    missing_cols_fit = [c for c in required_cols if c not in fit_source.columns]
    if missing_cols_fit:
        raise KeyError(f"Missing required columns in fit_df: {missing_cols_fit}")

    # ---------- low-income side ----------
    out["national_median_income"] = np.nan
    out["lihc_income_threshold"] = np.nan

    if income_rule == "country_median_60":
        medians = fit_source.groupby(country_col)[income_col].median()
        fallback_income_median = fit_source[income_col].median()
        out["national_median_income"] = out[country_col].map(medians).fillna(fallback_income_median)
        out["lihc_income_threshold"] = 0.60 * out["national_median_income"]
        out["low_income"] = out[income_col] < out["lihc_income_threshold"]
    else:
        out["low_income"] = out[income_bracket_col] < 4

    # ---------- conditional expenditure side ----------
    out["expected_exp"] = np.nan
    out["relative_exp"] = np.nan
    out["high_exp_flag"] = False
    out["qr_valid_flag"] = False

    model_cols = [exp_col] + qr_features
    if add_country_effects:
        model_cols.append(country_col)

    pooled_fit = fit_source[model_cols].copy().replace([np.inf, -np.inf], np.nan)
    valid_fit = pooled_fit.dropna()

    if len(valid_fit) < min_rows:
        raise ValueError(
            f"Too few valid rows for pooled quantile regression: {len(valid_fit)} < {min_rows}"
        )

    rhs_terms = qr_features.copy()
    if add_country_effects:
        rhs_terms.append(f"C({country_col})")
    formula = f"{exp_col} ~ " + " + ".join(rhs_terms)

    try:
        model = smf.quantreg(formula, data=valid_fit)
        res = model.fit(q=quantile, max_iter=5000, disp=False)

        pred_source = out[model_cols].copy().replace([np.inf, -np.inf], np.nan)
        pred_subset = qr_features + ([country_col] if add_country_effects else [])
        valid_pred = pred_source.dropna(subset=pred_subset)

        if len(valid_pred) == 0:
            raise ValueError("No rows available for prediction after filtering qr features.")

        expected = pd.Series(index=out.index, dtype=float)
        expected.loc[valid_pred.index] = res.predict(valid_pred)

        out["expected_exp"] = expected
        out["relative_exp"] = out[exp_col] - out["expected_exp"]

        margin = np.nanstd(valid_fit[exp_col]) * margin_scale
        out["high_exp_flag"] = (out[exp_col] > (out["expected_exp"] + margin)).fillna(False)
        out.loc[valid_pred.index, "qr_valid_flag"] = True

        print(
            f"[Pooled HQRTM] q={quantile:.2f}, "
            f"pseudo_R2={getattr(res, 'prsquared', np.nan):.3f}, "
            f"high_exp={out['high_exp_flag'].mean():.2%}, "
            f"n_fit={len(valid_fit)}"
        )

        print("\nHigh expenditure rate by country:")
        print(
            out.groupby(country_col)["high_exp_flag"]
               .mean()
               .sort_index()
               .round(4)
        )

    except Exception as e:
        raise RuntimeError(f"Pooled quantile regression failed: {e}")

    out["risk_category"] = np.select(
        [
            out["low_income"] & out["high_exp_flag"],
            out["low_income"] & ~out["high_exp_flag"],
            ~out["low_income"] & out["high_exp_flag"],
        ],
        [
            "Double risk",
            "Income risk",
            "Expenditure risk",
        ],
        default="No risk",
    )

    return out



import numpy as np
import pandas as pd


def assign_paper_lihc(
    df: pd.DataFrame,
    income_bracket_col: str = "income_bracket",
    exp_col: str = "total_expenditure",
    country_col: str = "Country",
    exp_quantile: float = 0.80,
    fit_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Paper-style LIHC categorization based on:
      - low income: income bracket < 4
      - high expenditure: above country-specific 80th percentile
        of annual energy expenditure

    This matches the framework used in:
    van Hove, Dalla Longa, van der Zwaan (2022),
    where the income threshold is set between deciles 3 and 4,
    and the expenditure threshold is the 80th percentile within country.

    Parameters
    ----------
    df : pd.DataFrame
        Data to classify.
    income_bracket_col : str
        Survey income bracket / decile column (1-10).
    exp_col : str
        Annual energy expenditure column.
    country_col : str
        Country identifier column.
    exp_quantile : float
        Country-specific expenditure quantile. Default is 0.80.
    fit_df : pd.DataFrame | None
        Optional reference dataset on which thresholds are fitted and then
        applied to df.

    Returns
    -------
    pd.DataFrame
        Copy of df with:
        - low_income
        - exp_threshold
        - high_exp_flag
        - risk_category
    """
    out = df.copy()
    fit_source = out if fit_df is None else fit_df.copy()

    required_cols = [country_col, income_bracket_col, exp_col]

    missing_cols = [c for c in required_cols if c not in out.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns: {missing_cols}")

    missing_cols_fit = [c for c in required_cols if c not in fit_source.columns]
    if missing_cols_fit:
        raise KeyError(f"Missing required columns in fit_df: {missing_cols_fit}")

    # low income: threshold between brackets 3 and 4
    out["low_income"] = out[income_bracket_col] < 4

    # high expenditure: country-specific expenditure quantile
    exp_thresholds = fit_source.groupby(country_col)[exp_col].quantile(exp_quantile)
    out["exp_threshold"] = out[country_col].map(exp_thresholds)

    if out["exp_threshold"].isna().any():
        missing_countries = sorted(
            out.loc[out["exp_threshold"].isna(), country_col].dropna().unique().tolist()
        )
        raise ValueError(
            f"Missing expenditure thresholds for countries: {missing_countries}"
        )

    out["high_exp_flag"] = out[exp_col] > out["exp_threshold"]

    # four classes
    out["risk_category"] = np.select(
        [
            out["low_income"] & out["high_exp_flag"],
            out["low_income"] & ~out["high_exp_flag"],
            ~out["low_income"] & out["high_exp_flag"],
        ],
        [
            "Double risk",
            "Income risk",
            "Expenditure risk",
        ],
        default="No risk",
    )

    return out