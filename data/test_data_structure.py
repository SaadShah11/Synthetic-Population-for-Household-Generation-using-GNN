import pandas as pd
import numpy as np

# Load the data
print("Loading data...")
hh_comp_df = pd.read_csv('./preprocessed-data/crosstables/HH_composition_by_age_by_sex_Main.csv')
age_df = pd.read_csv('./preprocessed-data/individuals/Age_Perfect_5yrs.csv')

# Check area E02005949
print("\nChecking area E02005949...")
area_hh = hh_comp_df[hh_comp_df['geography code'] == 'E02005949']
area_age = age_df[age_df['geography code'] == 'E02005949']

if not area_hh.empty and not area_age.empty:
    print(f"HH total: {area_hh['total'].iloc[0]}")
    print(f"Age total: {area_age['total'].iloc[0]}")
    
    # Get all data columns (excluding geography code and total)
    data_cols = [col for col in hh_comp_df.columns if col not in ['geography code', 'total']]
    print(f"Number of data columns: {len(data_cols)}")
    
    # Calculate sum of all data columns
    col_sum = area_hh[data_cols].sum(axis=1).iloc[0]
    print(f"Sum of all columns: {col_sum}")
    
    # Check if there are any negative values
    negative_vals = area_hh[data_cols].iloc[0][area_hh[data_cols].iloc[0] < 0]
    if len(negative_vals) > 0:
        print(f"Negative values found: {negative_vals}")
    
    # Show column structure
    print(f"\nFirst 10 columns: {data_cols[:10]}")
    print(f"Last 10 columns: {data_cols[-10:]}")
    
    # Check if columns follow expected pattern
    expected_pattern = []
    for age_group in ['0_15', '16_24', '25_34', '35_49', '50+']:
        for sex in ['M', 'F']:
            for hh_comp in ['1PE', '1PA', '1FE', '1FM-0C', '1FM-2C', '1FM-nA', '1FC-0C', '1FC-2C', '1FC-nA', '1FL-2C', '1FL-nA', '1H-2C', '1H-nS', '1H-nE', '1H-nA']:
                expected_pattern.append(f"{sex} {age_group} {hh_comp}")
    
    print(f"\nExpected columns: {len(expected_pattern)}")
    print(f"Actual columns: {len(data_cols)}")
    
    # Check if all expected columns exist
    missing_cols = [col for col in expected_pattern if col not in data_cols]
    extra_cols = [col for col in data_cols if col not in expected_pattern]
    
    if missing_cols:
        print(f"Missing columns: {missing_cols[:5]}...")
    if extra_cols:
        print(f"Extra columns: {extra_cols[:5]}...")
    
    # Show some sample values
    print(f"\nSample values for first 5 columns:")
    for col in data_cols[:5]:
        print(f"{col}: {area_hh[col].iloc[0]}")
        
else:
    print("Area E02005949 not found in one or both datasets") 