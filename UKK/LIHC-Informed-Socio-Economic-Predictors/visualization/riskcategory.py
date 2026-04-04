import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np

# Load preprocessed data
path = r"C:\Users\Ilani\OneDrive\Desktop\EP\LIHC-Informed-Socio-Economic-Predictors\UKK\preprocessed_data_LIHC_proxy_cleaned.csv"
df = pd.read_csv(path)

# Use income_continuous instead of S9MONTH for threshold-based analysis
# income_col = 'income_continuous'

income_col = 'income_bracket'
exp_col = 'total_expenditure'

# Compute thresholds
# Option 1: Global threshold for income
income_threshold = df[income_col].quantile(0.4)  # Single 40th percentile across all data
# Option 2: Country-specific threshold (uncomment and comment the line above if needed)
# income_threshold = df.groupby('Country')[income_col].transform(lambda x: x.quantile(0.4))

exp_percentiles = [0.6]  # 70%, 80%, 90% percentiles

# Compute country-specific percentile thresholds
percentile_dict = df.groupby('Country')[exp_col].quantile(exp_percentiles).unstack()
# Merge with original DataFrame to get country-specific thresholds for each row
df = df.merge(percentile_dict.reset_index(), on='Country', how='left')
# Create percentile_values dictionary using actual column names (0.7, 0.8, 0.9)
percentile_values = {pct: df[pct] for pct in exp_percentiles}  # Access columns by percentile value

# Option 1: Global thresholds for expenditure (comment out the line below for country-specific median approach)
global_thresholds = {pct: df[exp_col].quantile(pct) for pct in exp_percentiles}
# Option 2: Median of country-specific thresholds (uncomment and comment the line above if needed)
# global_thresholds = {pct: percentile_values[pct].median() for pct in exp_percentiles}

# Colors for quadrants (aligned with previous definitions)
quadrant_colors = {
    'No risk': 'lightgreen',
    'Income risk': 'yellow',
    'Expenditure risk': 'orange',
    'Double risk': 'red'
}

# Plotting 2D grids for each percentile
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

for ax, pct in zip(axes, exp_percentiles):
    threshold = global_thresholds[pct]  # Use global threshold for the current percentile
    
    # Scatter points (color by risk_category if available)
    if 'risk_category' in df.columns:
        scatter = ax.scatter(df[exp_col], df[income_col], c=df['risk_category'].map(quadrant_colors), alpha=0.6)
    else:
        scatter = ax.scatter(df[exp_col], df[income_col], c='grey', alpha=0.8)
    
    # Threshold lines
    ax.axhline(y=income_threshold, color='black', linestyle='--', label='Income threshold (40th percentile)')
    ax.axvline(x=threshold, color='red', linestyle='--', label=f'Energy {int(pct*100)}th percentile')
    
    # Quadrants (adjusted coordinates)
    ax.add_patch(patches.Rectangle((0, income_threshold), threshold, df[income_col].max() - income_threshold,
                                   facecolor=quadrant_colors['No risk'], alpha=0.3, label='No risk'))
    ax.add_patch(patches.Rectangle((0, 0), threshold, income_threshold,
                                   facecolor=quadrant_colors['Income risk'], alpha=0.3, label='Income risk'))
    ax.add_patch(patches.Rectangle((threshold, income_threshold), df[exp_col].max() - threshold, 
                                   df[income_col].max() - income_threshold, facecolor=quadrant_colors['Expenditure risk'], 
                                   alpha=0.3, label='Expenditure risk'))
    ax.add_patch(patches.Rectangle((threshold, 0), df[exp_col].max() - threshold, income_threshold,
                                   facecolor=quadrant_colors['Double risk'], alpha=0.3, label='Double risk'))
    
    # Labels for quadrants
    ax.text(threshold / 2, (income_threshold + df[income_col].max()) / 2, 'No risk', ha='center', va='center', alpha=0.8)
    ax.text(threshold / 2, income_threshold / 2, 'Income risk', ha='center', va='center', alpha=0.8)
    ax.text((threshold + df[exp_col].max()) / 2, (income_threshold + df[income_col].max()) / 2, 'Expenditure risk', 
            ha='center', va='center', alpha=0.8)
    ax.text((threshold + df[exp_col].max()) / 2, income_threshold / 2, 'Double risk', ha='center', va='center', alpha=0.8)
    
    # Axes labels & title
    ax.set_xlabel('Energy Expenditure (€)')
    ax.set_title(f'Energy Threshold: {int(pct*100)}th Percentile')
    # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

axes[0].set_ylabel('Income')
plt.suptitle('2D Grid of Income vs Energy Expenditure (Energy Poverty Quadrants)', fontsize=14)
plt.tight_layout(rect=[0, 0, 0.85, 0.95])  # Adjust layout to accommodate legend
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('ep_lihc_dual_thresholds.png', dpi=300, bbox_inches='tight')
plt.close()  # Avoid display clutter

print("Plot saved as 'ep_lihc_dual_thresholds.png'")