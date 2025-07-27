#!/usr/bin/env python3
"""
Script to print all column names of a selected CSV file.
Usage: python print_csv_columns.py <csv_file_path>
"""

import pandas as pd
import sys
import os
from pathlib import Path

def print_csv_columns(csv_file_path):
    """
    Print all column names of a CSV file.
    
    Args:
        csv_file_path (str): Path to the CSV file
    """
    try:
        # Check if file exists
        if not os.path.exists(csv_file_path):
            print(f"Error: File '{csv_file_path}' does not exist.")
            return
        
        # Read the CSV file
        print(f"Reading CSV file: {csv_file_path}")
        df = pd.read_csv(csv_file_path)
        
        # Print file information
        print(f"\nFile Information:")
        print(f"  Shape: {df.shape[0]} rows x {df.shape[1]} columns")
        print(f"  Total columns: {len(df.columns)}")
        
        # Print all column names
        print(f"\nAll Column Names:")
        print("-" * 50)
        for i, col in enumerate(df.columns, 1):
            print(f"{i:3d}. {col}")
        
        # Print first few rows for context
        print(f"\nFirst 3 rows (first 5 columns):")
        print("-" * 50)
        print(df.iloc[:3, :5].to_string())
        
        # Check for any problematic column names
        print(f"\nColumn Name Analysis:")
        print("-" * 50)
        
        # Check for spaces in column names
        spaces_in_names = [col for col in df.columns if ' ' in col]
        if spaces_in_names:
            print(f"Columns with spaces: {len(spaces_in_names)}")
            for col in spaces_in_names[:5]:  # Show first 5
                print(f"  - '{col}'")
            if len(spaces_in_names) > 5:
                print(f"  ... and {len(spaces_in_names) - 5} more")
        else:
            print("No columns with spaces found.")
        
        # Check for special characters
        special_chars = [col for col in df.columns if any(char in col for char in ['-', '_', '.', '/', '\\'])]
        if special_chars:
            print(f"Columns with special characters: {len(special_chars)}")
            for col in special_chars[:5]:  # Show first 5
                print(f"  - '{col}'")
            if len(special_chars) > 5:
                print(f"  ... and {len(special_chars) - 5} more")
        else:
            print("No columns with special characters found.")
        
        # Check for very long column names
        long_names = [col for col in df.columns if len(col) > 50]
        if long_names:
            print(f"Very long column names (>50 chars): {len(long_names)}")
            for col in long_names[:3]:  # Show first 3
                print(f"  - '{col}'")
            if len(long_names) > 3:
                print(f"  ... and {len(long_names) - 3} more")
        
        # Check for duplicate column names
        duplicates = df.columns[df.columns.duplicated()].tolist()
        if duplicates:
            print(f"Duplicate column names: {len(duplicates)}")
            for col in duplicates:
                print(f"  - '{col}'")
        else:
            print("No duplicate column names found.")
            
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

def main():
    """Main function to handle command line arguments."""
    if len(sys.argv) != 2:
        print("Usage: python print_csv_columns.py <csv_file_path>")
        print("\nExamples:")
        print("  python print_csv_columns.py ./preprocessed-data/crosstables/HH_composition_by_age_by_sex.csv")
        print("  python print_csv_columns.py ./preprocessed-data/crosstables/HH_composition_by_age_by_sex_Updated.csv")
        return
    
    csv_file_path = sys.argv[1]
    print_csv_columns(csv_file_path)

if __name__ == "__main__":
    main() 