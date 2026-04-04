### cleaning, imputation, feature engineering, outlier removal, and target creation



# new features

Equivalized Income After Housing Costs (AHC)

Description: Adjusts household income for family composition (equivalization) and subtracts housing costs (rent/mortgage) to reflect disposable income for energy bills, aligning with LIHC's low-income threshold.
LIHC Alignment: Core metric; defines the 40th percentile income threshold.
Derivation: Use S1Ac1-S1Bc3 (household members by age/gender) for equivalization (e.g., OECD scale: 1.0 for first adult, 0.5 for others, 0.3 for children). Proxy housing costs from S8 (income feeling) or assume a flat rate (e.g., 20% of income if unavailable).
Implementation: df['equivalized_income_ahc'] = df['income_continuous'] / equivalization_factor - (0.2 * df['income_continuous'])


Modeled Energy Expenditure (Standardized Cost)

Description: Estimates energy costs based on dwelling characteristics (e.g., floor area, insulation, heating type) rather than actual bills, as per BREDEM in the Handbook.
LIHC Alignment: Defines the high-cost threshold (80th percentile); accounts for efficiency.
Derivation: Combine H3 (floor area group, midpoint), H5A1-H5A4 (insulation types, binary 'has_insulation'), H6A1-H6A11 (heating sources, primary type), and H7AA (heating months). Use a simple model: cost = floor_area * heating_months * (1 - insulation_factor) * heating_type_factor.
Implementation: df['modeled_energy_cost'] = df['floor_area'] * df['H7AA'] * (1 - df['has_insulation'].astype(int) * 0.2) * heating_type_weight


Dwelling Energy Efficiency Score

Description: Aggregates efficiency indicators (e.g., insulation, house age, renewable installations) into a single score, reflecting potential energy cost variations.
LIHC Alignment: Supports high-cost threshold by quantifying efficiency impact.
Derivation: Score from H2 (house age, reverse-scaled), H5A1-H5A4 (insulation, sum), H10A1-H10A5 (renewables, binary). Normalize to 0-1.
Implementation: df['efficiency_score'] = (1 - (df['house_age'] / max_age)) + df['has_insulation'].sum(axis=1) + df['has_renewables'].astype(int)


Household Size-Adjusted Energy Demand

Description: Adjusts energy expenditure by household size to account for per-capita needs, refining the high-cost threshold.
LIHC Alignment: Ensures fair comparison across households, aligning with equivalization.
Derivation: Use S1Ac1-S1Bc3 (sum for household_size) to normalize H7A1 (heating cost) or H8A (electricity bill): demand = expenditure / household_size.
Implementation: df['adjusted_energy_demand'] = df['total_expenditure'] / df['household_size']


Housing Cost Burden Ratio

Description: Ratio of housing costs (rent/mortgage) to income, indicating affordability pressure that affects energy spending capacity.
LIHC Alignment: Indirectly influences low-income threshold by reducing disposable income.
Derivation: Proxy from S8 (income feeling, e.g., 1=Comfortable as low burden, 4=Very difficult as high) or assume 20-30% of income as housing cost (Handbook standard).
Implementation: df['housing_cost_burden'] = np.where(df['S8'] > 2, 0.3, 0.2) * df['income_continuous']


Seasonal Energy Cost Variation

Description: Captures variability in energy costs (e.g., heating months, air conditioning use) to refine the 80th percentile threshold.
LIHC Alignment: High-cost threshold depends on peak expenditure periods.
Derivation: Use H7AA (heating months) and C2-C2A (AC use) to adjust H7A1/H8A: seasonal_cost = monthly_cost * (heating_months / 12 + ac_use_factor).
Implementation: df['seasonal_energy_cost'] = df['H7A1'] * (df['H7AA'] / 12 + df['C2'].astype(int) * 0.1)