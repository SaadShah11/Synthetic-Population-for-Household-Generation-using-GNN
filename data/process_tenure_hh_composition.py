#!/usr/bin/env python3
"""
Script to process Tenure_by_HH_composition_Updated.csv by splitting 1H-nSE household composition 
categories into separate 1H-nS, 1H-nE, and 1H-nA categories.

Input: Tenure_by_HH_composition_Updated.csv with columns like:
       OW 1H-nSE, SO 1H-nSE, SR 1H-nSE, PR 1H-nSE

Output: Updated CSV with split columns:
        OW 1H-nS (1%), OW 1H-nE (4%), OW 1H-nA (95%)
        SO 1H-nS (1%), SO 1H-nE (4%), SO 1H-nA (95%)
        SR 1H-nS (1%), SR 1H-nE (4%), SR 1H-nA (95%)
        PR 1H-nS (1%), PR 1H-nE (4%), PR 1H-nA (95%)

Split ratios:
- 1H-nS: 1% of original 1H-nSE value
- 1H-nE: 4% of original 1H-nSE value  
- 1H-nA: 95% of original 1H-nSE value
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, List, Tuple

# Get the current directory of the script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Define file paths
input_file_path = os.path.join(current_dir, 'preprocessed-data/crosstables/Tenure_by_HH_composition_Updated.csv')
output_file_path = os.path.join(current_dir, 'preprocessed-data/crosstables/Tenure_by_HH_composition_Updated_Modified.csv')

# Define tenure categories
tenure_categories = ['OW', 'SO', 'SR', 'PR']

# Define household compositions (updated with the three separate categories)
hh_compositions_old = ['1PE', '1PA', '1FE', '1FM-0C', '1FM-2C', '1FM-nA', '1FC-0C', '1FC-2C', '1FC-nA', '1FL-2C', '1FL-nA', '1H-2C', '1H-nSE']
hh_compositions_new = ['1PE', '1PA', '1FE', '1FM-0C', '1FM-2C', '1FM-nA', '1FC-0C', '1FC-2C', '1FC-nA', '1FL-2C', '1FL-nA', '1H-2C', '1H-nS', '1H-nE', '1H-nA']

# Define split ratios for 1H-nSE
split_ratios = {
    '1H-nS': 0.01,  # 1%
    '1H-nE': 0.04,  # 4%
    '1H-nA': 0.95   # 95%
}

def validate_split_ratios():
    """Validate that split ratios sum to 1.0"""
    total_ratio = sum(split_ratios.values())
    print(f"Split ratios validation:")
    for category, ratio in split_ratios.items():
        print(f"  {category}: {ratio:.1%}")
    print(f"  Total: {total_ratio:.1%}")
    
    if abs(total_ratio - 1.0) > 0.001:
        raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")
    print("✓ Split ratios validated successfully!")
    print()

def read_tenure_data(file_path: str) -> pd.DataFrame:
    """Read the tenure by household composition crosstable data."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")
    
    df = pd.read_csv(file_path)
    print(f"Loaded tenure data: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")
    return df

def get_nse_columns(df: pd.DataFrame) -> List[str]:
    """Identify columns containing 1H-nSE household composition."""
    nse_columns = []
    for col in df.columns:
        if '1H-nSE' in col:
            nse_columns.append(col)
    
    print(f"Found {len(nse_columns)} columns with 1H-nSE:")
    for col in nse_columns:
        print(f"  {col}")
    print()
    return nse_columns

def create_new_columns(df: pd.DataFrame, nse_columns: List[str]) -> pd.DataFrame:
    """Create new columns for split household compositions and initialize to 0."""
    new_df = df.copy()
    
    # For each 1H-nSE column, create three new columns
    new_columns = []
    for nse_col in nse_columns:
        # Extract tenure type from column name (e.g., "OW 1H-nSE" -> "OW")
        tenure_type = nse_col.split(' ')[0]
        
        # Create three new columns for this tenure type
        for new_hh_comp in ['1H-nS', '1H-nE', '1H-nA']:
            new_col_name = f"{tenure_type} {new_hh_comp}"
            new_df[new_col_name] = 0
            new_columns.append(new_col_name)
    
    print(f"Created {len(new_columns)} new columns:")
    for col in new_columns:
        print(f"  {col}")
    print()
    return new_df

def handle_rounding_differences(values: List[float], target_total: float) -> List[int]:
    """
    Handle rounding differences to ensure sum equals target total.
    Uses largest remainder method for fair distribution.
    """
    # Round all values down
    rounded_values = [int(val) for val in values]
    remainders = [val - int(val) for val in values]
    current_sum = sum(rounded_values)
    difference = int(target_total - current_sum)
    
    if difference > 0:
        # Sort by remainder (descending) and add 1 to the largest remainders
        sorted_indices = sorted(range(len(remainders)), key=lambda i: remainders[i], reverse=True)
        for i in range(difference):
            idx = sorted_indices[i]
            rounded_values[idx] += 1
    
    return rounded_values

def split_nse_data(df: pd.DataFrame, nse_columns: List[str]) -> pd.DataFrame:
    """
    Split 1H-nSE data into 1H-nS, 1H-nE, and 1H-nA using specified ratios.
    Uses proper rounding to ensure no households are lost.
    """
    print("Splitting 1H-nSE data into separate categories...")
    
    total_original = 0
    total_distributed = 0
    
    # Process each row
    for idx, row in df.iterrows():
        if idx % 100 == 0 and len(df) > 100:
            print(f"  Processing row {idx + 1}/{len(df)}")
        
        # Process each 1H-nSE column
        for nse_col in nse_columns:
            original_value = float(row[nse_col])
            total_original += original_value
            
            if original_value > 0:
                # Extract tenure type from column name
                tenure_type = nse_col.split(' ')[0]
                
                # Calculate split values
                raw_values = []
                target_columns = []
                for new_hh_comp, ratio in split_ratios.items():
                    split_value = original_value * ratio
                    raw_values.append(split_value)
                    target_columns.append(f"{tenure_type} {new_hh_comp}")
                
                # Handle rounding to ensure total is preserved
                rounded_values = handle_rounding_differences(raw_values, original_value)
                
                # Assign values to new columns
                for target_col, rounded_value in zip(target_columns, rounded_values):
                    df.at[idx, target_col] = rounded_value
                    total_distributed += rounded_value
    
    print(f"✓ Data splitting completed")
    print(f"  Original total: {total_original:,.0f}")
    print(f"  Distributed total: {total_distributed:,.0f}")
    print(f"  Difference: {total_distributed - total_original:,.0f}")
    print()
    
    return df

def remove_original_nse_columns(df: pd.DataFrame, nse_columns: List[str]) -> pd.DataFrame:
    """Remove the original 1H-nSE columns after splitting."""
    print("Removing original 1H-nSE columns...")
    df_final = df.drop(columns=nse_columns)
    
    print(f"Removed {len(nse_columns)} original columns:")
    for col in nse_columns:
        print(f"  {col}")
    print()
    
    return df_final

def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Reorder columns to maintain logical grouping."""
    print("Reordering columns...")
    
    # Start with geography code and total
    ordered_columns = ['geography code', 'total']
    
    # Add tenure x household composition columns in logical order
    for tenure in tenure_categories:
        for hh_comp in hh_compositions_new:
            col_name = f"{tenure} {hh_comp}"
            if col_name in df.columns:
                ordered_columns.append(col_name)
    
    # Add any remaining columns that weren't matched
    for col in df.columns:
        if col not in ordered_columns:
            ordered_columns.append(col)
    
    # Filter to only columns that exist in the dataframe
    final_columns = [col for col in ordered_columns if col in df.columns]
    
    print(f"Reordered {len(final_columns)} columns")
    return df[final_columns]

def validate_data_integrity(original_df: pd.DataFrame, final_df: pd.DataFrame, nse_columns: List[str]):
    """Validate that data integrity is maintained after processing."""
    print("Validating data integrity...")
    
    # Check that total columns match
    if 'total' in original_df.columns and 'total' in final_df.columns:
        original_totals = original_df['total'].sum()
        final_totals = final_df['total'].sum()
        print(f"  Total column sums - Original: {original_totals:,.0f}, Final: {final_totals:,.0f}")
        
        if original_totals != final_totals:
            print(f"  WARNING: Total column sums don't match!")
    
    # Check that the sum of split columns equals original 1H-nSE columns
    for nse_col in nse_columns:
        if nse_col in original_df.columns:
            original_sum = original_df[nse_col].sum()
            
            # Find corresponding split columns
            tenure_type = nse_col.split(' ')[0]
            split_columns = [f"{tenure_type} {comp}" for comp in ['1H-nS', '1H-nE', '1H-nA']]
            split_sum = 0
            
            for split_col in split_columns:
                if split_col in final_df.columns:
                    split_sum += final_df[split_col].sum()
            
            print(f"  {nse_col}: Original: {original_sum:,.0f}, Split total: {split_sum:,.0f}, Difference: {split_sum - original_sum:,.0f}")
            
            if abs(split_sum - original_sum) > 1:  # Allow for minor rounding differences
                print(f"    WARNING: Significant difference detected!")
    
    print("✓ Data integrity validation completed")
    print()

def main():
    """Main processing function."""
    print("="*60)
    print("TENURE BY HOUSEHOLD COMPOSITION PROCESSOR")
    print("="*60)
    print()
    
    # Validate configuration
    validate_split_ratios()
    
    # Read input data
    print("Step 1: Reading input data...")
    df = read_tenure_data(input_file_path)
    original_df = df.copy()
    print()
    
    # Identify 1H-nSE columns
    print("Step 2: Identifying 1H-nSE columns...")
    nse_columns = get_nse_columns(df)
    
    if not nse_columns:
        print("No 1H-nSE columns found. Nothing to process.")
        return
    print()
    
    # Create new columns
    print("Step 3: Creating new columns...")
    df = create_new_columns(df, nse_columns)
    print()
    
    # Split the data
    print("Step 4: Splitting 1H-nSE data...")
    df = split_nse_data(df, nse_columns)
    print()
    
    # Remove original columns
    print("Step 5: Removing original 1H-nSE columns...")
    df = remove_original_nse_columns(df, nse_columns)
    print()
    
    # Reorder columns
    print("Step 6: Reordering columns...")
    df = reorder_columns(df)
    print()
    
    # Validate data integrity
    print("Step 7: Validating data integrity...")
    validate_data_integrity(original_df, df, nse_columns)
    print()
    
    # Save output
    print("Step 8: Saving processed data...")
    df.to_csv(output_file_path, index=False)
    print(f"✓ Processed data saved to: {output_file_path}")
    print(f"  Output shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print()
    
    print("="*60)
    print("PROCESSING COMPLETED SUCCESSFULLY!")
    print("="*60)

if __name__ == "__main__":
    main() 