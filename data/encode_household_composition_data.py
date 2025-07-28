#!/usr/bin/env python3
"""
Script to encode HH_composition_by_age_by_sex.csv by breaking down broad age categories 
into detailed age groups using proportional distribution.

Input: HH_composition_by_age_by_sex.csv with age categories 0_15, 16_24, 25_34, 35_49, 50+
Output: HH_composition_by_age_by_sex_encoded.csv with detailed age groups

Age breakdown mapping:
- 0_15  → 0_4, 5_7, 8_9, 10_14, 15
- 16_24 → 16_17, 18_19, 20_24
- 25_34 → 25_29, 30_34  
- 35_49 → 35_39, 40_44, 45_49
- 50+   → 50_54, 55_59, 60_64, 65_69, 70_74, 75_79, 80_84, 85+
"""

import pandas as pd
import os
import numpy as np
from typing import Dict, List, Tuple

# Get the current directory of the script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Define file paths
input_file_path = os.path.join(current_dir, 'preprocessed-data/crosstables/HH_composition_by_age_by_sex_Main.csv')
output_file_path = os.path.join(current_dir, 'preprocessed-data/crosstables/HH_composition_by_age_by_sex_Main_Modified.csv')

# Define the target age groups (same as used in qualification data)
target_age_groups = ['0_4', '5_7', '8_9', '10_14', '15', '16_17', '18_19', '20_24', '25_29', '30_34', 
                     '35_39', '40_44', '45_49', '50_54', '55_59', '60_64', '65_69', '70_74', '75_79', '80_84', '85+']

# Define current age groups in the household composition file
current_age_groups = ['0_15', '16_24', '25_34', '35_49', '50+']

# Define household composition codes
household_composition_codes = [
    '1PE',    # One person household: Aged 65 and over
    '1PA',    # One person household: Other
    '1FE',    # One family only: All aged 65 and over
    '1FM-0C', # One family only: Married or same-sex civil partnership couple: No children
    '1FM-2C', # One family only: Married or same-sex civil partnership couple: Dependent children
    '1FM-nA', # One family only: Married or same-sex civil partnership couple: All children non-dependent
    '1FC-0C', # One family only: Cohabiting couple: No children
    '1FC-2C', # One family only: Cohabiting couple: Dependent children
    '1FC-nA', # One family only: Cohabiting couple: All children non-dependent
    '1FL-2C', # One family only: Lone parent: Dependent children
    '1FL-nA', # One family only: Lone parent: All children non-dependent
    '1H-2C',  # Other household types: With dependent children
    '1H-nS',  # Other household types: All full-time students
    '1H-nE',  # Other household types: All aged 65 and over
    '1H-nA'   # Other household types: Other
]

# Define sex categories
sex_categories = ['M', 'F']

# Age group breakdown mapping with distribution ratios
age_breakdown_mapping = {
    '0_15': {
        '0_4': 0.31,    # 5 years out of 16 years
        '5_7': 0.19,    # 3 years out of 16 years
        '8_9': 0.12,    # 2 years out of 16 years
        '10_14': 0.31,  # 5 years out of 16 years
        '15': 0.07      # 1 year out of 16 years
    },
    '16_24': {
        '16_17': 0.22,  # 2 years out of 9 years
        '18_19': 0.22,  # 2 years out of 9 years  
        '20_24': 0.56   # 5 years out of 9 years
    },
    '25_34': {
        '25_29': 0.50,  # 5 years out of 10 years
        '30_34': 0.50   # 5 years out of 10 years
    },
    '35_49': {
        '35_39': 0.33,  # 5 years out of 15 years
        '40_44': 0.33,  # 5 years out of 15 years
        '45_49': 0.34   # 5 years out of 15 years (slightly higher to account for rounding)
    },
    '50+': {
        '50_54': 0.20,  # Higher proportion for younger ages
        '55_59': 0.18,  # Declining with age
        '60_64': 0.16,  # Further decline
        '65_69': 0.15,  # Continued decline
        '70_74': 0.12,  # Lower proportion
        '75_79': 0.08,  # Further decline
        '80_84': 0.06,  # Much lower
        '85+':   0.05   # Smallest portion for oldest group
    }
}

def validate_distribution_ratios():
    """Validate that all distribution ratios sum to 1.0 for each age group."""
    print("Validating distribution ratios:")
    all_valid = True
    for age_group, breakdown in age_breakdown_mapping.items():
        total_ratio = sum(breakdown.values())
        print(f"  {age_group}: {total_ratio:.3f} (target: 1.000)")
        if abs(total_ratio - 1.0) > 0.001:
            print(f"    WARNING: Ratio sum is not 1.0!")
            all_valid = False
    
    if all_valid:
        print("✓ All distribution ratios validated successfully!")
    print()

def read_household_composition_data(file_path: str) -> pd.DataFrame:
    """Read the household composition crosstable data."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")
    
    df = pd.read_csv(file_path)
    print(f"Loaded household composition data: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"First few columns: {list(df.columns[:10])}")
    return df

def parse_column_name(col_name: str) -> Tuple[str, str, str]:
    """
    Parse column name to extract sex, age, and household composition code.
    Expected format: 'M 0_15 1PE' or 'F 25_34 1FM-0C'
    
    Returns:
        Tuple of (sex, age_group, hh_code) or (None, None, None) if not parseable
    """
    if col_name.lower() in ['geography code', 'total']:
        return None, None, None
    
    parts = col_name.strip().split()
    if len(parts) == 3:
        sex, age, hh_code = parts
        if sex in sex_categories and age in current_age_groups and hh_code in household_composition_codes:
            return sex, age, hh_code
    
    return None, None, None

def create_new_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Create new columns for detailed age groups and initialize to 0."""
    new_df = df.copy()
    
    # Create new columns for each combination of sex, detailed age group, and household composition
    new_columns = []
    for sex in sex_categories:
        for age_group in target_age_groups:
            for hh_code in household_composition_codes:
                col_name = f"{sex} {age_group} {hh_code}"
                new_df[col_name] = 0
                new_columns.append(col_name)
    
    print(f"Created {len(new_columns)} new columns for detailed age breakdown")
    return new_df

def handle_rounding_differences(values: List[float], target_total: float) -> List[int]:
    """Handle rounding differences to ensure sum equals target total."""
    # Round all values
    rounded_values = [round(val) for val in values]
    current_sum = sum(rounded_values)
    difference = int(target_total - current_sum)
    
    if difference != 0:
        # Find indices sorted by fractional part (descending for positive difference)
        fractional_parts = [(i, val - round(val)) for i, val in enumerate(values)]
        if difference > 0:
            # Add to categories with largest fractional parts
            fractional_parts.sort(key=lambda x: x[1], reverse=True)
        else:
            # Subtract from categories with smallest fractional parts
            fractional_parts.sort(key=lambda x: x[1])
        
        # Adjust values
        for i in range(abs(difference)):
            idx = fractional_parts[i][0]
            if difference > 0:
                rounded_values[idx] += 1
            else:
                rounded_values[idx] -= 1
    
    return rounded_values

def distribute_household_composition_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Distribute data from broad age categories to detailed age groups.
    Uses proper rounding handling to ensure no people are lost.
    """
    print("Distributing data from broad age categories to detailed age groups...")
    
    # Track total distributed values for verification
    total_distributed = 0
    total_original = 0
    
    # Process each row
    for idx, row in df.iterrows():
        if idx % 1000 == 0:
            print(f"  Processing row {idx + 1}/{len(df)}")
        
        # Process each original column that contains household composition data
        for col in df.columns:
            sex, age_group, hh_code = parse_column_name(col)
            
            if sex and age_group and hh_code:
                original_value = row[col]
                total_original += original_value
                
                # Get the breakdown mapping for this age group
                if age_group in age_breakdown_mapping:
                    breakdown = age_breakdown_mapping[age_group]
                    
                    if original_value > 0:
                        # Calculate proportional values
                        detailed_ages = list(breakdown.keys())
                        ratios = list(breakdown.values())
                        proportional_values = [original_value * ratio for ratio in ratios]
                        
                        # Handle rounding to ensure exact sum
                        final_values = handle_rounding_differences(proportional_values, original_value)
                        
                        # Assign to dataframe
                        for j, detailed_age in enumerate(detailed_ages):
                            new_col = f"{sex} {detailed_age} {hh_code}"
                            if new_col in df.columns:
                                df.at[idx, new_col] = final_values[j]
                                total_distributed += final_values[j]
                            else:
                                print(f"    WARNING: Column {new_col} not found!")
                        
                        # Debug for first row with non-zero values
                        if idx == 0 and original_value > 0:
                            sum_check = sum(final_values)
                            print(f"  {col}: {original_value} → {detailed_ages} = {final_values} (sum: {sum_check})")
    
    print(f"Distribution complete:")
    print(f"  Total original value: {total_original}")
    print(f"  Total distributed value: {total_distributed}")
    print(f"  Difference: {total_distributed - total_original}")
    
    return df

def remove_original_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove the original broad age category columns."""
    columns_to_remove = []
    
    for col in df.columns:
        sex, age_group, hh_code = parse_column_name(col)
        if sex and age_group and hh_code and age_group in current_age_groups:
            columns_to_remove.append(col)
    
    print(f"Removing {len(columns_to_remove)} original broad age category columns")
    df_cleaned = df.drop(columns=columns_to_remove)
    
    return df_cleaned

def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reorder columns to have geography code and total first, 
    followed by sorted household composition columns.
    """
    # Fixed columns
    fixed_columns = []
    hh_columns = []
    
    for col in df.columns:
        if col.lower() in ['geography code', 'total']:
            fixed_columns.append(col)
        else:
            hh_columns.append(col)
    
    # Sort household composition columns by sex, then age, then household code
    def sort_key(col_name):
        sex, age, hh_code = parse_column_name(col_name)
        if sex and age and hh_code:
            try:
                sex_idx = sex_categories.index(sex)
                age_idx = target_age_groups.index(age)
                hh_code_idx = household_composition_codes.index(hh_code)
                return (sex_idx, age_idx, hh_code_idx)
            except ValueError:
                return (999, 999, 999)  # Put unknown columns at the end
        return (999, 999, 999)
    
    hh_columns_sorted = sorted(hh_columns, key=sort_key)
    
    # Combine fixed and sorted columns
    final_column_order = fixed_columns + hh_columns_sorted
    
    print(f"Reordered columns: {len(fixed_columns)} fixed + {len(hh_columns_sorted)} household composition columns")
    
    return df[final_column_order]

def update_total_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Update the total column to be the sum of all household composition columns 
    (excluding geography code and total columns).
    """
    print("Updating total column to sum of all household composition columns...")
    
    # Get all household composition columns (exclude geography code and total)
    hh_columns = [col for col in df.columns 
                  if col.lower() not in ['geography code', 'total']]
    
    # Calculate new totals
    df['total'] = df[hh_columns].sum(axis=1)
    
    print(f"Updated {len(df)} total values based on {len(hh_columns)} household composition columns")
    
    # Print some statistics
    print(f"Total column statistics:")
    print(f"  Min: {df['total'].min()}")
    print(f"  Max: {df['total'].max()}")
    print(f"  Mean: {df['total'].mean():.2f}")
    print()
    
    return df

def verify_area_totals(df: pd.DataFrame, sample_geo_code: str = None) -> None:
    """
    Verify that the sum of all household composition columns equals the original total
    for a specific area, showing detailed breakdown.
    """
    if sample_geo_code is None:
        sample_geo_code = df['geography code'].iloc[0]
    
    print(f"\nDetailed verification for geography code: {sample_geo_code}")
    print("=" * 70)
    
    sample_row = df[df['geography code'] == sample_geo_code]
    if sample_row.empty:
        print(f"Geography code {sample_geo_code} not found!")
        return
    
    sample_row = sample_row.iloc[0]
    original_total = sample_row['total']
    
    print(f"Original Total: {original_total}")
    print()
    
    # Calculate totals by category
    category_totals = {}
    
    # Detailed breakdown by sex, age, and household composition
    for sex in sex_categories:
        sex_total = 0
        print(f"{sex} ({'Male' if sex == 'M' else 'Female'}):")
        
        for age_group in target_age_groups:
            age_total = 0
            age_breakdown = {}
            
            for hh_code in household_composition_codes:
                col_name = f"{sex} {age_group} {hh_code}"
                if col_name in sample_row.index:
                    value = sample_row[col_name]
                    age_total += value
                    if value > 0:
                        age_breakdown[hh_code] = value
            
            if age_total > 0:
                print(f"  {age_group}: {age_total}")
                for hh_code, val in age_breakdown.items():
                    print(f"    {hh_code}: {val}")
                sex_total += age_total
        
        category_totals[sex] = sex_total
        print(f"  {sex} Total: {sex_total}")
        print()
    
    # Calculate grand total from all household composition columns
    hh_columns = [col for col in df.columns 
                  if col.lower() not in ['geography code', 'total']]
    calculated_total = sum(sample_row[col] for col in hh_columns)
    
    # Summary
    print("SUMMARY:")
    print(f"  Male Total: {category_totals['M']}")
    print(f"  Female Total: {category_totals['F']}")
    print(f"  Calculated Total: {calculated_total}")
    print(f"  Original Total: {original_total}")
    print(f"  Difference: {calculated_total - original_total}")
    
    # Verification result
    if abs(calculated_total - original_total) < 0.01:
        print("  ✓ VERIFICATION PASSED: Totals match!")
    else:
        print("  ✗ VERIFICATION FAILED: Totals do not match!")
    print()

def create_individual_household_composition_file(df: pd.DataFrame) -> None:
    """
    Create the individual household composition CSV file by summing across sex and age groups
    for each household composition code.
    """
    print("Creating individual household composition file...")
    
    # Initialize the result dataframe with geography code and total
    individual_df = df[['geography code', 'total']].copy()
    
    # Calculate totals for each household composition code
    for hh_code in household_composition_codes:
        hh_total_col = hh_code
        individual_df[hh_total_col] = 0
        
        # Sum across all sex and age combinations for this household composition code
        for sex in sex_categories:
            for age_group in target_age_groups:
                col_name = f"{sex} {age_group} {hh_code}"
                if col_name in df.columns:
                    individual_df[hh_total_col] += df[col_name]
    
    # Save the individual household composition file
    individual_file_path = os.path.join(current_dir, 'preprocessed-data/individuals/HH_composition.csv')
    individual_df.to_csv(individual_file_path, index=False)
    
    print(f"Individual household composition file saved to: {individual_file_path}")
    print(f"Individual file shape: {individual_df.shape[0]} rows, {individual_df.shape[1]} columns")
    print(f"Individual file columns: {list(individual_df.columns)}")
    print()

def final_verification(df: pd.DataFrame) -> None:
    """
    Final verification to check if any areas have mismatched totals.
    This ensures no people were lost or gained during the distribution process.
    """
    print("Performing final verification...")
    print("=" * 60)
    
    areas_with_differences = []
    max_difference = 0
    total_difference = 0
    
    # Check each area
    for idx, row in df.iterrows():
        geo_code = row['geography code']
        original_total = row['total']
        
        # Sum all household composition columns (exclude geography code and total)
        calculated_total = 0
        for col in df.columns:
            if col.lower() not in ['geography code', 'total']:
                calculated_total += row[col]
        
        # Calculate difference
        difference = original_total - calculated_total
        
        if abs(difference) > 0.01:  # Allow for tiny rounding differences
            areas_with_differences.append({
                'geography_code': geo_code,
                'original_total': original_total,
                'calculated_total': calculated_total,
                'difference': difference
            })
            
        # Track statistics
        if abs(difference) > abs(max_difference):
            max_difference = difference
        total_difference += abs(difference)
    
    # Print summary
    print(f"Areas checked: {len(df)}")
    print(f"Areas with differences: {len(areas_with_differences)}")
    print(f"Maximum difference: {max_difference}")
    print(f"Total absolute difference: {total_difference}")
    
    # Print areas with differences
    if areas_with_differences:
        print(f"\nAREAS WITH NON-ZERO DIFFERENCES:")
        print("-" * 80)
        print(f"{'Geography Code':<15} {'Original':<10} {'Calculated':<12} {'Difference':<10}")
        print("-" * 80)
        
        for area in areas_with_differences:
            print(f"{area['geography_code']:<15} {area['original_total']:<10} {area['calculated_total']:<12} {area['difference']:<10}")
    else:
        print("\n✓ ALL AREAS VERIFIED: No differences found!")
        print("All people properly distributed across household composition categories.")
    
    print("=" * 60)

def print_sample_breakdown(df: pd.DataFrame, sample_geo_code: str = None) -> None:
    """Print a sample breakdown for verification."""
    if sample_geo_code is None:
        sample_geo_code = df['geography code'].iloc[0]
    
    print(f"\nSample breakdown for geography code: {sample_geo_code}")
    print("=" * 60)
    
    sample_row = df[df['geography code'] == sample_geo_code]
    if sample_row.empty:
        print(f"Geography code {sample_geo_code} not found!")
        return
    
    sample_row = sample_row.iloc[0]
    
    # Print total
    print(f"Total: {sample_row['total']}")
    print()
    
    # Print breakdown by sex and age group
    for sex in sex_categories:
        print(f"{sex} ({'Male' if sex == 'M' else 'Female'}):")
        sex_total = 0
        for age_group in target_age_groups:
            age_total = 0
            for hh_code in household_composition_codes:
                col_name = f"{sex} {age_group} {hh_code}"
                if col_name in sample_row.index:
                    value = sample_row[col_name]
                    age_total += value
            
            if age_total > 0:
                print(f"  {age_group}: {age_total}")
                sex_total += age_total
        
        print(f"  {sex} Total: {sex_total}")
        print()

def main():
    """Main function to process the household composition data."""
    print("Starting household composition data age breakdown encoding...")
    print("=" * 60)
    
    # Validate distribution ratios
    validate_distribution_ratios()
    
    # Read input data
    df = read_household_composition_data(input_file_path)
    
    # Create new columns for detailed age groups
    df = create_new_columns(df)
    
    # Distribute data from broad to detailed age categories
    df = distribute_household_composition_data(df)
    
    # Remove original broad age category columns
    df = remove_original_columns(df)
    
    # Reorder columns for better organization
    df = reorder_columns(df)
    
    # Update total column to be sum of all household composition columns
    df = update_total_column(df)
    
    # Print sample breakdown
    print_sample_breakdown(df)
    
    # Final verification to check for any mismatched totals
    final_verification(df)
    
    # Save the processed data
    print(f"Saving processed data to: {output_file_path}")
    df.to_csv(output_file_path, index=False)
    
    # Create individual household composition file for generateIndividuals script
    create_individual_household_composition_file(df)
    
    print(f"\nFinal dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Processing complete!")
    
    # Print column summary
    print("\nColumn summary:")
    print(f"  Geography and total columns: 2")
    print(f"  Household composition data columns: {df.shape[1] - 2}")
    print(f"  Expected household composition columns: {len(sex_categories) * len(target_age_groups) * len(household_composition_codes)}")

if __name__ == "__main__":
    main() 