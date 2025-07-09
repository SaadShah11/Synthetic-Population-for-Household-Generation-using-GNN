#!/usr/bin/env python3
"""
Script to encode qualification_by_sex_by_age.csv by breaking down broad age categories 
into detailed age groups using proportional distribution.

Input: qualification_by_sex_by_age.csv with age categories 16_24, 25_34, 35_49, 50_64, 65+
Output: qualification_by_sex_by_age_encoded.csv with detailed age groups

Age breakdown mapping:
- 16_24 → 16_17, 18_19, 20_24
- 25_34 → 25_29, 30_34  
- 35_49 → 35_39, 40_44, 45_49
- 50_64 → 50_54, 55_59, 60_64
- 65+   → 65_69, 70_74, 75_79, 80_84, 85+
"""

import pandas as pd
import os
import numpy as np
from typing import Dict, List, Tuple

# Get the current directory of the script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Define file paths
input_file_path = os.path.join(current_dir, 'preprocessed-data/crosstables/QualificationBySexByAge.csv')
religion_file_path = os.path.join(current_dir, 'preprocessed-data/crosstables/ReligionbySexbyAge.csv')
output_file_path = os.path.join(current_dir, 'preprocessed-data/crosstables/QualificationBySexByAgeModified.csv')

# Define the target age groups from user specification
target_age_groups = ['0_4', '5_7', '8_9', '10_14', '15', '16_17', '18_19', '20_24', '25_29', '30_34', 
                     '35_39', '40_44', '45_49', '50_54', '55_59', '60_64', '65_69', '70_74', '75_79', '80_84', '85+']

# Define current age groups in the qualification file
current_age_groups = ['16_24', '25_34', '35_49', '50_64', '65+']

# Define qualification levels
qualification_levels = ['L0', 'L1', 'L2', 'LA', 'L3', 'L4', 'LO']

# Define sex categories
sex_categories = ['M', 'F']

# Age group breakdown mapping with distribution ratios
age_breakdown_mapping = {
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
    '50_64': {
        '50_54': 0.33,  # 5 years out of 15 years
        '55_59': 0.33,  # 5 years out of 15 years
        '60_64': 0.34   # 5 years out of 15 years (slightly higher to account for rounding)
    },
    '65+': {
        '65_69': 0.30,  # Highest portion for younger elderly
        '70_74': 0.25,  # Declining with age
        '75_79': 0.20,  # Further decline
        '80_84': 0.15,  # Significant decline
        '85+':   0.10   # Smallest portion for oldest group
    }
}

# Young age groups (0-15) distribution ratios for 'LO' qualifications
# These represent children with no qualifications (ages 0-15)
young_age_ratios = {
    '0_4': 0.33,   # 5 years out of 16 years (0-15)
    '5_7': 0.19,   # 3 years out of 16 years
    '8_9': 0.12,   # 2 years out of 16 years
    '10_14': 0.31, # 5 years out of 16 years
    '15': 0.05     # 1 year out of 16 years
}

def validate_distribution_ratios():
    """Validate that all distribution ratios sum to 1.0 for each age group."""
    print("Validating distribution ratios:")
    for age_group, breakdown in age_breakdown_mapping.items():
        total_ratio = sum(breakdown.values())
        print(f"  {age_group}: {total_ratio:.3f} (target: 1.000)")
        if abs(total_ratio - 1.0) > 0.001:
            print(f"    WARNING: Ratio sum is not 1.0!")
    
    print("Validating young age distribution ratios:")
    young_total = sum(young_age_ratios.values())
    print(f"  Young ages (0-15): {young_total:.3f} (target: 1.000)")
    if abs(young_total - 1.0) > 0.001:
        print(f"    WARNING: Young age ratio sum is not 1.0!")
    print()

def read_qualification_data(file_path: str) -> pd.DataFrame:
    """Read the qualification crosstable data."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")
    
    df = pd.read_csv(file_path)
    print(f"Loaded qualification data: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"First few columns: {list(df.columns[:10])}")
    return df

def read_religion_data(file_path: str) -> pd.DataFrame:
    """Read the religion crosstable data for total values."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Religion file not found: {file_path}")
    
    df = pd.read_csv(file_path)
    print(f"Loaded religion data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def parse_column_name(col_name: str) -> Tuple[str, str, str]:
    """
    Parse column name to extract sex, age, and qualification level.
    Expected format: 'M 16_24 L1' or 'F 25_34 LA'
    
    Returns:
        Tuple of (sex, age_group, qualification_level) or (None, None, None) if not parseable
    """
    if col_name.lower() in ['geography code', 'total']:
        return None, None, None
    
    parts = col_name.strip().split()
    if len(parts) == 3:
        sex, age, qual = parts
        if sex in sex_categories and age in current_age_groups and qual in qualification_levels:
            return sex, age, qual
    
    return None, None, None

def create_new_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Create new columns for detailed age groups and initialize to 0."""
    new_df = df.copy()
    
    # Create new columns for each combination of sex, detailed age group, and qualification level
    new_columns = []
    for sex in sex_categories:
        for age_group in target_age_groups:
            for qual_level in qualification_levels:
                col_name = f"{sex} {age_group} {qual_level}"
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

def distribute_qualification_data(df: pd.DataFrame) -> pd.DataFrame:
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
        
        # Process each original column that contains qualification data
        for col in df.columns:
            sex, age_group, qual_level = parse_column_name(col)
            
            if sex and age_group and qual_level:
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
                            new_col = f"{sex} {detailed_age} {qual_level}"
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

def calculate_and_distribute_young_age_data(df: pd.DataFrame, religion_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate missing values using religion totals and distribute them across 
    young age groups (0-15) with 'LO' qualification.
    This should be called BEFORE distributing the 16+ qualification data.
    """
    print("Calculating and distributing values for young age groups (0-15) using religion totals...")
    
    total_missing_distributed = 0
    
    # Process each row
    for idx, row in df.iterrows():
        if idx % 1000 == 0:
            print(f"  Processing row {idx + 1}/{len(df)}")
        
        # Get geography code to match with religion data
        geo_code = row['geography code']
        
        # Find corresponding religion total
        religion_row = religion_df[religion_df['geography code'] == geo_code]
        if religion_row.empty:
            print(f"  WARNING: Geography code {geo_code} not found in religion data, skipping...")
            continue
        
        religion_total = religion_row['total'].iloc[0]
        
        # Calculate sum of ORIGINAL qualification data (all columns except geography code and total)
        original_qualifications_sum = 0
        original_breakdown = {}
        
        for col in df.columns:
            if col.lower() not in ['geography code', 'total']:
                value = row[col]
                original_qualifications_sum += value
                if idx == 0:  # Detailed logging for first row
                    if value > 0:
                        original_breakdown[col] = value
        
        # Calculate missing value (should be young people with no qualifications)
        # Use religion total instead of qualification total
        missing_value = religion_total - original_qualifications_sum
        
        # Detailed logging for first row
        if idx == 0:
            print(f"\nDETAILED LOGGING FOR FIRST ROW - Geography Code: {geo_code}")
            print("=" * 70)
            print(f"Religion Total: {religion_total}")
            print(f"Sum of Original Qualification Columns: {original_qualifications_sum}")
            print(f"Missing Value (0-15 population): {missing_value}")
            print()
            
            print("Original Qualification Columns with Values > 0:")
            for col, val in sorted(original_breakdown.items()):
                print(f"  {col}: {val}")
            print()
        
        if missing_value > 0:
            # Distribute missing value across young age groups for both sexes
            sex_ratio = 0.5  # Equal distribution between male and female
            young_distribution = {}
            young_age_total = 0
            
            # First, distribute based on ratios
            distributed_values = {}
            total_distributed_by_ratio = 0
            
            for sex in sex_categories:
                sex_missing = missing_value * sex_ratio
                
                if idx == 0:  # Detailed logging for first row
                    print(f"{sex} ({'Male' if sex == 'M' else 'Female'}) - Missing Value to Distribute: {sex_missing}")
                
                # Distribute across young age groups with 'LO' qualification
                for young_age, ratio in young_age_ratios.items():
                    col_name = f"{sex} {young_age} LO"
                    distributed_value = round(sex_missing * ratio)
                    
                    if col_name in df.columns:
                        distributed_values[col_name] = distributed_value
                        total_distributed_by_ratio += distributed_value
                        
                        if idx == 0:  # Detailed logging for first row
                            print(f"  {col_name}: {sex_missing} × {ratio} = {distributed_value}")
                    else:
                        print(f"    WARNING: Column {col_name} not found!")
            
            # Check for rounding differences and adjust
            rounding_difference = missing_value - total_distributed_by_ratio
            
            if idx == 0 and rounding_difference != 0:
                print(f"\nRounding difference detected: {rounding_difference}")
                print("Adjusting the largest category to match exactly...")
            
            # If there's a rounding difference, add it to the largest category (0_4 age group)
            if rounding_difference != 0:
                # Find the largest distributed value to adjust
                max_col = max(distributed_values.keys(), key=lambda k: distributed_values[k])
                distributed_values[max_col] += rounding_difference
                
                if idx == 0:
                    print(f"Adjusted {max_col}: {distributed_values[max_col] - rounding_difference} + {rounding_difference} = {distributed_values[max_col]}")
            
            # Now assign the final distributed values
            for col_name, final_value in distributed_values.items():
                df.at[idx, col_name] = final_value
                total_missing_distributed += final_value
                young_age_total += final_value
                young_distribution[col_name] = final_value
            
            if idx == 0:  # Summary for first row
                print()
                print("Young Age Distribution Summary:")
                total_distributed_first_row = sum(young_distribution.values())
                for col, val in sorted(young_distribution.items()):
                    print(f"  {col}: {val}")
                print(f"Total Distributed to Young Ages: {total_distributed_first_row}")
                print("=" * 70)
                print()
                
        elif missing_value < 0:
            # If negative, distribute 0 (no young people to assign)
            pass
    
    print(f"Young age distribution complete:")
    print(f"  Total missing value distributed: {total_missing_distributed}")
    
    return df

def remove_original_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove the original broad age category columns."""
    columns_to_remove = []
    
    for col in df.columns:
        sex, age_group, qual_level = parse_column_name(col)
        if sex and age_group and qual_level and age_group in current_age_groups:
            columns_to_remove.append(col)
    
    print(f"Removing {len(columns_to_remove)} original broad age category columns")
    df_cleaned = df.drop(columns=columns_to_remove)
    
    return df_cleaned

def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reorder columns to have geography code and total first, 
    followed by sorted qualification columns.
    """
    # Fixed columns
    fixed_columns = []
    qualification_columns = []
    
    for col in df.columns:
        if col.lower() in ['geography code', 'total']:
            fixed_columns.append(col)
        else:
            qualification_columns.append(col)
    
    # Sort qualification columns by sex, then age, then qualification level
    def sort_key(col_name):
        sex, age, qual = parse_column_name(col_name)
        if sex and age and qual:
            try:
                sex_idx = sex_categories.index(sex)
                age_idx = target_age_groups.index(age)
                qual_idx = qualification_levels.index(qual)
                return (sex_idx, age_idx, qual_idx)
            except ValueError:
                return (999, 999, 999)  # Put unknown columns at the end
        return (999, 999, 999)
    
    qualification_columns_sorted = sorted(qualification_columns, key=sort_key)
    
    # Combine fixed and sorted columns
    final_column_order = fixed_columns + qualification_columns_sorted
    
    print(f"Reordered columns: {len(fixed_columns)} fixed + {len(qualification_columns_sorted)} qualification columns")
    
    return df[final_column_order]

def verify_totals(df: pd.DataFrame) -> None:
    """Verify that row totals match the 'total' column."""
    print("Verifying row totals...")
    
    # Calculate row totals (excluding geography code and total columns)
    data_columns = [col for col in df.columns if col.lower() not in ['geography code', 'total']]
    df['calculated_total'] = df[data_columns].sum(axis=1)
    
    # Compare with original total column
    df['total_difference'] = df['calculated_total'] - df['total']
    
    # Print summary statistics
    max_diff = df['total_difference'].abs().max()
    mean_diff = df['total_difference'].abs().mean()
    rows_with_diff = (df['total_difference'].abs() > 0.1).sum()
    
    print(f"  Maximum absolute difference: {max_diff}")
    print(f"  Mean absolute difference: {mean_diff:.3f}")
    print(f"  Rows with significant difference (>0.1): {rows_with_diff}/{len(df)}")
    
    # Remove verification columns
    df.drop(['calculated_total', 'total_difference'], axis=1, inplace=True)

def verify_area_totals(df: pd.DataFrame, sample_geo_code: str = None) -> None:
    """
    Verify that the sum of all qualification columns equals the original total
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
    
    # Detailed breakdown by sex, age, and qualification
    for sex in sex_categories:
        sex_total = 0
        print(f"{sex} ({'Male' if sex == 'M' else 'Female'}):")
        
        for age_group in target_age_groups:
            age_total = 0
            age_breakdown = {}
            
            for qual_level in qualification_levels:
                col_name = f"{sex} {age_group} {qual_level}"
                if col_name in sample_row.index:
                    value = sample_row[col_name]
                    age_total += value
                    if value > 0:
                        age_breakdown[qual_level] = value
            
            if age_total > 0:
                print(f"  {age_group}: {age_total}")
                for qual, val in age_breakdown.items():
                    print(f"    {qual}: {val}")
                sex_total += age_total
        
        category_totals[sex] = sex_total
        print(f"  {sex} Total: {sex_total}")
        print()
    
    # Calculate grand total from all qualification columns
    qualification_columns = [col for col in df.columns 
                           if col.lower() not in ['geography code', 'total']]
    calculated_total = sum(sample_row[col] for col in qualification_columns)
    
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
    
    # Show qualification level breakdown
    print("\nBreakdown by Qualification Level:")
    qual_totals = {}
    for qual_level in qualification_levels:
        qual_total = 0
        for sex in sex_categories:
            for age_group in target_age_groups:
                col_name = f"{sex} {age_group} {qual_level}"
                if col_name in sample_row.index:
                    qual_total += sample_row[col_name]
        qual_totals[qual_level] = qual_total
        if qual_total > 0:
            print(f"  {qual_level}: {qual_total}")
    
    # Show age group breakdown
    print("\nBreakdown by Age Group:")
    age_totals = {}
    for age_group in target_age_groups:
        age_total = 0
        for sex in sex_categories:
            for qual_level in qualification_levels:
                col_name = f"{sex} {age_group} {qual_level}"
                if col_name in sample_row.index:
                    age_total += sample_row[col_name]
        age_totals[age_group] = age_total
        if age_total > 0:
            print(f"  {age_group}: {age_total}")
    
    print()

def create_individual_qualification_file(df: pd.DataFrame) -> None:
    """
    Create the individual qualification CSV file by summing across sex and age groups
    for each qualification level.
    """
    print("Creating individual qualification file...")
    
    # Initialize the result dataframe with geography code and total
    individual_df = df[['geography code', 'total']].copy()
    
    # Calculate totals for each qualification level
    for qual_level in qualification_levels:
        qual_total_col = qual_level
        individual_df[qual_total_col] = 0
        
        # Sum across all sex and age combinations for this qualification level
        for sex in sex_categories:
            for age_group in target_age_groups:
                col_name = f"{sex} {age_group} {qual_level}"
                if col_name in df.columns:
                    individual_df[qual_total_col] += df[col_name]
    
    # Save the individual qualification file
    individual_file_path = os.path.join(current_dir, 'preprocessed-data/individuals/Qualification.csv')
    individual_df.to_csv(individual_file_path, index=False)
    
    print(f"Individual qualification file saved to: {individual_file_path}")
    print(f"Individual file shape: {individual_df.shape[0]} rows, {individual_df.shape[1]} columns")
    print(f"Individual file columns: {list(individual_df.columns)}")
    print()

def update_total_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Update the total column to be the sum of all qualification columns 
    (excluding geography code and total columns).
    """
    print("Updating total column to sum of all qualification columns...")
    
    # Get all qualification columns (exclude geography code and total)
    qualification_columns = [col for col in df.columns 
                           if col.lower() not in ['geography code', 'total']]
    
    # Calculate new totals
    df['total'] = df[qualification_columns].sum(axis=1)
    
    print(f"Updated {len(df)} total values based on {len(qualification_columns)} qualification columns")
    
    # Print some statistics
    print(f"Total column statistics:")
    print(f"  Min: {df['total'].min()}")
    print(f"  Max: {df['total'].max()}")
    print(f"  Mean: {df['total'].mean():.2f}")
    print()
    
    return df

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
        print(f"{sex} (Male/Female):")
        sex_total = 0
        for age_group in target_age_groups:
            age_total = 0
            for qual_level in qualification_levels:
                col_name = f"{sex} {age_group} {qual_level}"
                if col_name in sample_row.index:
                    value = sample_row[col_name]
                    age_total += value
            
            if age_total > 0:
                print(f"  {age_group}: {age_total}")
                sex_total += age_total
        
        print(f"  {sex} Total: {sex_total}")
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
        
        # Sum all qualification columns (exclude geography code and total)
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
        print("All people properly distributed across qualification categories.")
    
    print("=" * 60)

def main():
    """Main function to process the qualification data."""
    print("Starting qualification data age breakdown encoding...")
    print("=" * 60)
    
    # Validate distribution ratios
    validate_distribution_ratios()
    
    # Read input data
    df = read_qualification_data(input_file_path)
    religion_df = read_religion_data(religion_file_path)
    
    # Create new columns for detailed age groups
    df = create_new_columns(df)
    
    # FIRST: Calculate and distribute missing values to young age groups (0-15) with 'LO' qualification
    # This must be done BEFORE distributing the 16+ qualification data
    # Use religion totals for calculating missing values
    df = calculate_and_distribute_young_age_data(df, religion_df)
    
    # SECOND: Distribute data from broad to detailed age categories (16+)
    df = distribute_qualification_data(df)
    
    # Remove original broad age category columns
    df = remove_original_columns(df)
    
    # Reorder columns for better organization
    df = reorder_columns(df)
    
    # Update total column to be sum of all qualification columns
    df = update_total_column(df)
    
    # Print sample breakdown
    print_sample_breakdown(df)
    
    # Final verification to check for any mismatched totals
    final_verification(df)
    
    # Save the processed data
    print(f"Saving processed data to: {output_file_path}")
    df.to_csv(output_file_path, index=False)
    
    # Create individual qualification file for generateIndividuals script
    create_individual_qualification_file(df)
    
    print(f"\nFinal dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Processing complete!")
    
    # Print column summary
    print("\nColumn summary:")
    print(f"  Geography and total columns: 2")
    print(f"  Qualification data columns: {df.shape[1] - 2}")
    print(f"  Expected qualification columns: {len(sex_categories) * len(target_age_groups) * len(qualification_levels)}")

if __name__ == "__main__":
    main() 