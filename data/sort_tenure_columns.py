#!/usr/bin/env python3
"""
Script to sort/reorder column names in Tenure_by_HH_composition_Updated_Modified.csv
Changes format from "OW 1PE" to "1PE OW" (household composition first, then tenure)

Input: Tenure_by_HH_composition_Updated_Modified.csv with columns like:
       geography code, total, OW 1PE, OW 1PA, SO 1PE, etc.

Output: Same data but with reordered column names:
        geography code, total, 1PE OW, 1PA OW, 1PE SO, etc.
"""

import pandas as pd
import os
from typing import Dict, List

# Get the current directory of the script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Define file paths
input_file_path = os.path.join(current_dir, 'preprocessed-data/crosstables/Tenure_by_HH_composition_Updated_Modified.csv')
output_file_path = os.path.join(current_dir, 'preprocessed-data/crosstables/Tenure_by_HH_composition_Updated_Modified_Sorted.csv')

# Define tenure and household composition categories
tenure_categories = ['OW', 'SO', 'SR', 'PR']
hh_compositions = ['1PE', '1PA', '1FE', '1FM-0C', '1FM-2C', '1FM-nA', '1FC-0C', '1FC-2C', '1FC-nA', '1FL-2C', '1FL-nA', '1H-2C', '1H-nS', '1H-nE', '1H-nA']

def read_tenure_data(file_path: str) -> pd.DataFrame:
    """Read the tenure by household composition crosstable data."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")
    
    df = pd.read_csv(file_path)
    print(f"Loaded tenure data: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Original columns: {list(df.columns)[:10]}...")  # Show first 10 columns
    return df

def create_column_mapping(df: pd.DataFrame) -> Dict[str, str]:
    """
    Create a mapping from old column names (OW 1PE) to new column names (1PE OW).
    Only maps the tenure×household composition columns, keeps others unchanged.
    """
    column_mapping = {}
    
    # Keep geography code and total columns unchanged
    non_tenure_cols = ['geography code', 'total']
    for col in non_tenure_cols:
        if col in df.columns:
            column_mapping[col] = col
    
    # Create mapping for tenure×household composition columns
    mapped_cols = set()
    for col in df.columns:
        if col not in non_tenure_cols:
            # Try to parse as "tenure household_composition" format
            parts = col.split(' ', 1)  # Split on first space only
            if len(parts) == 2:
                tenure_part = parts[0]
                hh_comp_part = parts[1]
                
                # Verify it's a valid tenure and household composition combination
                if tenure_part in tenure_categories and hh_comp_part in hh_compositions:
                    # Create new column name with household composition first
                    new_col_name = f"{hh_comp_part} {tenure_part}"
                    column_mapping[col] = new_col_name
                    mapped_cols.add(col)
                    print(f"Mapping: '{col}' -> '{new_col_name}'")
                else:
                    # Keep unknown format unchanged
                    column_mapping[col] = col
                    print(f"Warning: Unrecognized format, keeping unchanged: '{col}'")
            else:
                # Keep columns that don't fit expected format unchanged
                column_mapping[col] = col
                print(f"Warning: Unexpected column format, keeping unchanged: '{col}'")
    
    print(f"\nMapped {len(mapped_cols)} tenure×household composition columns")
    return column_mapping

def reorder_columns_logically(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reorder columns in a logical way: geography code, total, then household composition columns
    grouped by household composition (all tenures for each household composition together).
    """
    print("\nReordering columns logically...")
    
    # Start with geography code and total
    ordered_columns = []
    if 'geography code' in df.columns:
        ordered_columns.append('geography code')
    if 'total' in df.columns:
        ordered_columns.append('total')
    
    # Add household composition × tenure columns in logical order
    # Group by household composition first, then all tenure types for each
    for hh_comp in hh_compositions:
        for tenure in tenure_categories:
            expected_col = f"{hh_comp} {tenure}"
            if expected_col in df.columns:
                ordered_columns.append(expected_col)
    
    # Add any remaining columns that weren't matched
    for col in df.columns:
        if col not in ordered_columns:
            ordered_columns.append(col)
            print(f"Added remaining column: {col}")
    
    # Filter to only columns that exist in the dataframe
    final_columns = [col for col in ordered_columns if col in df.columns]
    
    print(f"Reordered to {len(final_columns)} columns")
    print(f"New column order (first 10): {final_columns[:10]}")
    
    return df[final_columns]

def validate_data_integrity(original_df: pd.DataFrame, final_df: pd.DataFrame):
    """Validate that data integrity is maintained after column reordering."""
    print("\nValidating data integrity...")
    
    # Check that DataFrame shapes match
    if original_df.shape != final_df.shape:
        print(f"ERROR: Shape mismatch! Original: {original_df.shape}, Final: {final_df.shape}")
        return False
    
    # Check that total columns match if present
    if 'total' in original_df.columns and 'total' in final_df.columns:
        original_totals = original_df['total'].sum()
        final_totals = final_df['total'].sum()
        print(f"Total column sums - Original: {original_totals:,.0f}, Final: {final_totals:,.0f}")
        
        if abs(original_totals - final_totals) > 1e-6:
            print(f"ERROR: Total column sums don't match!")
            return False
    
    # Check that row sums are preserved (excluding geography code column)
    exclude_cols = ['geography code']
    orig_numeric_cols = [col for col in original_df.columns if col not in exclude_cols]
    final_numeric_cols = [col for col in final_df.columns if col not in exclude_cols]
    
    if len(orig_numeric_cols) == len(final_numeric_cols):
        original_row_sums = original_df[orig_numeric_cols].sum(axis=1)
        final_row_sums = final_df[final_numeric_cols].sum(axis=1)
        
        max_diff = abs(original_row_sums - final_row_sums).max()
        print(f"Maximum row sum difference: {max_diff}")
        
        if max_diff > 1e-6:
            print(f"ERROR: Row sums don't match!")
            return False
    
    print("✓ Data integrity validation passed")
    return True

def main():
    """Main processing function."""
    print("="*70)
    print("TENURE CROSSTABLE COLUMN SORTER")
    print("="*70)
    print()
    
    # Read input data
    print("Step 1: Reading input data...")
    df = read_tenure_data(input_file_path)
    original_df = df.copy()
    print()
    
    # Create column mapping
    print("Step 2: Creating column name mapping...")
    column_mapping = create_column_mapping(df)
    print()
    
    # Rename columns
    print("Step 3: Renaming columns...")
    df = df.rename(columns=column_mapping)
    print(f"Renamed {len(column_mapping)} columns")
    print()
    
    # Reorder columns logically
    print("Step 4: Reordering columns...")
    df = reorder_columns_logically(df)
    print()
    
    # Validate data integrity
    print("Step 5: Validating data integrity...")
    validation_passed = validate_data_integrity(original_df, df)
    if not validation_passed:
        print("ERROR: Data validation failed! Aborting save operation.")
        return
    print()
    
    # Save output
    print("Step 6: Saving reordered data...")
    df.to_csv(output_file_path, index=False)
    print(f"✓ Reordered data saved to: {output_file_path}")
    print(f"  Output shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print()
    
    # Show example of the transformation
    print("Step 7: Showing transformation example...")
    print("Sample of new column names:")
    sample_cols = [col for col in df.columns if ' ' in col and col not in ['geography code']][:10]
    for i, col in enumerate(sample_cols, 1):
        print(f"  {i:2d}. {col}")
    print()
    
    print("="*70)
    print("COLUMN SORTING COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"Input:  {input_file_path}")
    print(f"Output: {output_file_path}")
    print(f"Format changed from 'OW 1PE' to '1PE OW' (household composition first)")

if __name__ == "__main__":
    main() 