#!/usr/bin/env python3
"""
Script to sort columns in crosstable CSV files.
Sorts columns so that Male (M) and Female (F) pairs are beside each other
for the same Age and Category (Religion/Ethnicity/Marital).

Input format: {Sex} {Age} {Category}
Output format: M {Age} {Category}, F {Age} {Category} pairs
"""

import pandas as pd
import os
import re
from typing import List, Tuple, Dict

# Define ordering arrays as specified by user
age_groups = ['0_4', '5_7', '8_9', '10_14', '15', '16_17', '18_19', '20_24', '25_29', '30_34', '35_39', '40_44', '45_49', '50_54', '55_59', '60_64', '65_69', '70_74', '75_79', '80_84', '85+']
sex_categories = ['M', 'F']
ethnicity_categories = ['W1', 'W2', 'W3', 'W4', 'M1', 'M2', 'M3', 'M4', 'A1', 'A2', 'A3', 'A4', 'A5', 'B1', 'B2', 'B3', 'O1', 'O2']
religion_categories = ['C','B','H','J','M','S','O','N','NS']
marital_categories = ['Single','Married','Partner','Separated','Divorced','Widowed']
hh_compositions = ['1PE','1PA','1FE','1FM-0C','1FM-2C', '1FM-nA','1FC-0C','1FC-2C','1FC-nA','1FL-nA','1FL-2C','1H-nS','1H-nE','1H-nA', '1H-2C']

def parse_column_name(col_name: str) -> Tuple[str, str, str]:
    """
    Parse column name to extract components.
    Handles two formats:
    1. Individual data: "M 0_4 C" or "F 25_29 B" (Sex Age Category)
    2. Household data: "1PE W1" or "1FM-0C C" (HH_composition Category)
    
    Args:
        col_name: Column name in one of the supported formats
    
    Returns:
        Tuple of (first_component, second_component, third_component)
        For individual data: (sex, age, category)
        For household data: (hh_composition, category, None)
    """
    parts = col_name.strip().split()
    
    # Individual data format: 3 parts (Sex Age Category)
    if len(parts) >= 3:
        sex = parts[0]  # M or F
        age = parts[1]  # 0_4, 5_7, etc.
        category = parts[2]  # C, B, H, etc. or W1, M1, etc. or Single, Married, etc.
        return sex, age, category
    
    # Household data format: 2 parts (HH_composition Category)
    elif len(parts) == 2:
        hh_comp = parts[0]  # 1PE, 1PA, 1FE, etc.
        category = parts[1]  # W1, C, etc.
        return hh_comp, category, None
    
    return None, None, None

def get_category_order(category: str) -> int:
    """
    Get the order index for a category based on predefined arrays.
    
    Args:
        category: The category string
    
    Returns:
        Order index for sorting
    """
    # Check religion categories first
    if category in religion_categories:
        return religion_categories.index(category)
    
    # Check ethnicity categories
    if category in ethnicity_categories:
        return ethnicity_categories.index(category)
    
    # Check marital categories
    if category in marital_categories:
        return marital_categories.index(category)
    
    # Default to high number if not found
    return 999

def get_age_order(age: str) -> int:
    """
    Get the order index for an age group.
    
    Args:
        age: The age group string
    
    Returns:
        Order index for sorting
    """
    if age in age_groups:
        return age_groups.index(age)
    
    # Default to high number if not found
    return 999

def get_hh_composition_order(hh_comp: str) -> int:
    """
    Get the order index for a household composition.
    
    Args:
        hh_comp: The household composition string
    
    Returns:
        Order index for sorting
    """
    if hh_comp in hh_compositions:
        return hh_compositions.index(hh_comp)
    
    # Default to high number if not found
    return 999

def sort_crosstable_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort dataframe columns so that M and F pairs are beside each other for individual data,
    or household compositions are ordered correctly for household data.
    
    Args:
        df: Input dataframe with unsorted columns
    
    Returns:
        Dataframe with sorted columns
    """
    # Get all columns
    all_columns = list(df.columns)
    
    # Identify fixed columns (geography code, total)
    fixed_columns = []
    sortable_columns = []
    
    for col in all_columns:
        if col.lower() in ['geography code', 'total']:
            fixed_columns.append(col)
        else:
            sortable_columns.append(col)
    
    # Determine data type by examining first sortable column
    data_type = "individual"  # default
    if sortable_columns:
        first_col = sortable_columns[0]
        comp1, comp2, comp3 = parse_column_name(first_col)
        if comp3 is None and comp1 and comp2:
            # Only 2 components means household data
            data_type = "household"
    
    if data_type == "individual":
        # Individual data: group by (age, category), sort by sex within groups
        column_groups = {}
        
        for col in sortable_columns:
            sex, age, category = parse_column_name(col)
            if sex and age and category:
                key = (age, category)
                if key not in column_groups:
                    column_groups[key] = {'M': None, 'F': None}
                column_groups[key][sex] = col
        
        # Sort groups by age order then by category order
        sorted_keys = sorted(column_groups.keys(), key=lambda x: (get_age_order(x[0]), get_category_order(x[1])))
        
        # Build the sorted columns list
        sorted_columns = []
        for age, category in sorted_keys:
            group = column_groups[(age, category)]
            # Add Male column first if it exists
            if group['M']:
                sorted_columns.append(group['M'])
            # Add Female column second if it exists
            if group['F']:
                sorted_columns.append(group['F'])
    
    else:
        # Household data: group by category, sort by household composition within groups
        column_groups = {}
        
        for col in sortable_columns:
            hh_comp, category, _ = parse_column_name(col)
            if hh_comp and category:
                if category not in column_groups:
                    column_groups[category] = []
                column_groups[category].append((hh_comp, col))
        
        # Sort categories first, then household compositions within each category
        sorted_categories = sorted(column_groups.keys(), key=get_category_order)
        
        # Build the sorted columns list
        sorted_columns = []
        for category in sorted_categories:
            # Sort household compositions within this category
            hh_cols = sorted(column_groups[category], key=lambda x: get_hh_composition_order(x[0]))
            for hh_comp, col in hh_cols:
                sorted_columns.append(col)
    
    # Reconstruct column order: fixed columns + sorted columns
    new_column_order = fixed_columns + sorted_columns
    
    # Verify all columns are included
    if len(new_column_order) != len(all_columns):
        print(f"Warning: Column count mismatch. Original: {len(all_columns)}, New: {len(new_column_order)}")
        missing = set(all_columns) - set(new_column_order)
        extra = set(new_column_order) - set(all_columns)
        if missing:
            print(f"Missing columns: {missing}")
        if extra:
            print(f"Extra columns: {extra}")
    
    return df[new_column_order]

def process_csv_file(input_path: str, output_path: str) -> None:
    """
    Process a single CSV file and save the sorted version.
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to output CSV file
    """
    print(f"Processing {input_path}...")
    
    # Read CSV file
    df = pd.read_csv(input_path)
    print(f"Original shape: {df.shape}")
    print(f"Original columns: {len(df.columns)}")
    
    # Print first few column names for verification
    print(f"First 10 columns: {list(df.columns[:10])}")
    
    # Sort columns
    df_sorted = sort_crosstable_columns(df)
    print(f"Sorted shape: {df_sorted.shape}")
    print(f"Sorted columns: {len(df_sorted.columns)}")
    
    # Print first few sorted column names
    print(f"First 10 sorted columns: {list(df_sorted.columns[:10])}")
    
    # Save sorted CSV
    df_sorted.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
    print()

def main():
    """Main function to process all CSV files."""
    
    # Define input and output paths
    input_dir = "./preprocessed-data/crosstables"
    output_dir = "./preprocessed-data/crosstables"
    
    # Define file mappings (input -> output)
    file_mappings = {
        # Individual crosstables
        "ReligionbySexbyAge.csv": "ReligionbySexbyAge_sorted.csv",
        "EthnicityBySexByAge.csv": "EthnicityBySexByAge_sorted.csv", 
        "MaritalbySexbyAgeModified.csv": "MaritalbySexbyAgeModified_sorted.csv",
        
        # Household crosstables
        "HH_composition_by_ethnicity_Updated.csv": "HH_composition_by_ethnicity_Updated_sorted.csv",
        "HH_composition_by_religion_Updated.csv": "HH_composition_by_religion_Updated_sorted.csv"
    }
    
    # Process each file
    for input_file, output_file in file_mappings.items():
        input_path = os.path.join(input_dir, input_file)
        output_path = os.path.join(output_dir, output_file)
        
        if os.path.exists(input_path):
            try:
                process_csv_file(input_path, output_path)
            except Exception as e:
                print(f"Error processing {input_file}: {str(e)}")
        else:
            print(f"File not found: {input_path}")
    
    print("All files processed!")

if __name__ == "__main__":
    main() 