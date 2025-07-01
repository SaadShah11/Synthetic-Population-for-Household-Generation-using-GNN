#!/usr/bin/env python3
"""
Master Glossary Creator
Self-sufficient script that creates all crosstable glossaries and combines them into a single comprehensive glossary file.
"""

import os
import pandas as pd
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Create comprehensive master glossary for all crosstables')
    parser.add_argument('--area_code', type=str, required=True,
                       help='Oxford area code to process (e.g., E02005924)')
    return parser.parse_args()

def create_individuals_glossary():
    """
    Creates glossaries for all individual crosstables.
    Returns dictionary of glossary DataFrames.
    """
    # Define categories from generateIndividuals.py
    age_groups = ['0_4', '5_7', '8_9', '10_14', '15', '16_17', '18_19', '20_24', '25_29', '30_34', '35_39', '40_44', '45_49', '50_54', '55_59', '60_64', '65_69', '70_74', '75_79', '80_84', '85+']
    sex_categories = ['M', 'F']
    ethnicity_categories = ['W1', 'W2', 'W3', 'W4', 'M1', 'M2', 'M3', 'M4', 'A1', 'A2', 'A3', 'A4', 'A5', 'B1', 'B2', 'B3', 'O1', 'O2']
    religion_categories = ['C','B','H','J','M','S','O','N','NS']
    marital_categories = ['Single','Married','Partner','Separated','Divorced','Widowed']
    
    # Create age-sex combinations
    age_sex_combinations = [f"{age} {sex}" for age in age_groups for sex in sex_categories]
    
    def create_crosstable_glossary(row_categories, col_categories, crosstable_name):
        """Creates a glossary for a specific crosstable."""
        glossary_data = []
        sequential_index = 1
        
        # Changed order: iterate through col_categories first (age-sex combinations), 
        # then row_categories (ethnicity/religion/marital)
        for col_cat in col_categories:
            for row_cat in row_categories:
                full_label = f"{row_cat} {col_cat}"
                glossary_data.append({
                    'Sequential_Index': sequential_index,
                    'Full_Label': full_label
                })
                sequential_index += 1
        
        return pd.DataFrame(glossary_data)
    
    # Create individual glossaries
    ethnicity_glossary = create_crosstable_glossary(
        ethnicity_categories, age_sex_combinations, 'Ethnicity_by_Sex_by_Age'
    )
    
    religion_glossary = create_crosstable_glossary(
        religion_categories, age_sex_combinations, 'Religion_by_Sex_by_Age'
    )
    
    marital_glossary = create_crosstable_glossary(
        marital_categories, age_sex_combinations, 'Marital_Status_by_Sex_by_Age'
    )
    
    return {
        'Ethnicity_by_Sex_by_Age': ethnicity_glossary,
        'Religion_by_Sex_by_Age': religion_glossary,
        'Marital_Status_by_Sex_by_Age': marital_glossary
    }

def create_households_glossary():
    """
    Creates glossaries for all household crosstables.
    Returns dictionary of glossary DataFrames.
    """
    # Define categories from generateHouseholds.py
    hh_compositions = ['1PE','1PA','1FE','1FM-0C','1FM-2C', '1FM-nA','1FC-0C','1FC-2C','1FC-nA','1FL-nA','1FL-2C','1H-nS','1H-nE','1H-nA', '1H-2C']
    ethnicity_categories = ['W1', 'W2', 'W3', 'W4', 'M1', 'M2', 'M3', 'M4', 'A1', 'A2', 'A3', 'A4', 'A5', 'B1', 'B2', 'B3', 'O1', 'O2']
    religion_categories = ['C','B','H','J','M','S','O','N','NS']
    
    def create_crosstable_glossary(row_categories, col_categories, crosstable_name):
        """Creates a glossary for a specific crosstable."""
        glossary_data = []
        sequential_index = 1
        
        # Changed order: iterate through row_categories first (household compositions), 
        # then col_categories (ethnicity/religion)
        for row_cat in row_categories:
            for col_cat in col_categories:
                full_label = f"{row_cat} {col_cat}"
                glossary_data.append({
                    'Sequential_Index': sequential_index,
                    'Full_Label': full_label
                })
                sequential_index += 1
        
        return pd.DataFrame(glossary_data)
    
    # Create household glossaries
    hh_ethnicity_glossary = create_crosstable_glossary(
        hh_compositions, ethnicity_categories, 'HH_Composition_by_Ethnicity'
    )
    
    hh_religion_glossary = create_crosstable_glossary(
        hh_compositions, religion_categories, 'HH_Composition_by_Religion'
    )
    
    return {
        'HH_Composition_by_Ethnicity': hh_ethnicity_glossary,
        'HH_Composition_by_Religion': hh_religion_glossary
    }

def create_combined_glossary(individuals_glossaries, households_glossaries):
    """
    Combines all individual glossaries into a single DataFrame.
    """
    all_glossaries = {**individuals_glossaries, **households_glossaries}
    
    # Find maximum index needed
    max_indices = max(len(glossary) for glossary in all_glossaries.values())
    
    # Create master glossary DataFrame
    combined_glossary = pd.DataFrame({
        'Index': range(1, max_indices + 1)
    })
    
    # Add columns for each crosstable
    for crosstable_name, glossary_df in all_glossaries.items():
        combined_glossary[crosstable_name] = ''
        
        # Fill in the labels
        for idx, row in glossary_df.iterrows():
            if idx < len(combined_glossary):
                combined_glossary.loc[idx, crosstable_name] = row['Full_Label']
    
    return combined_glossary, all_glossaries

def create_master_glossary(area_code):
    """
    Creates a comprehensive master glossary combining all crosstables.
    
    Parameters:
    area_code - The Oxford area code to process
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    print(f"\n{'='*60}")
    print("CREATING COMPREHENSIVE MASTER GLOSSARY")
    print(f"{'='*60}")
    print(f"Area Code: {area_code}")
    
    # Create individual glossaries
    print("\nCreating individuals glossaries...")
    individuals_glossaries = create_individuals_glossary()
    print(f"  - Ethnicity by Sex by Age: {len(individuals_glossaries['Ethnicity_by_Sex_by_Age'])} combinations")
    print(f"  - Religion by Sex by Age: {len(individuals_glossaries['Religion_by_Sex_by_Age'])} combinations")
    print(f"  - Marital Status by Sex by Age: {len(individuals_glossaries['Marital_Status_by_Sex_by_Age'])} combinations")
    
    # Create household glossaries
    print("\nCreating households glossaries...")
    households_glossaries = create_households_glossary()
    print(f"  - Household Composition by Ethnicity: {len(households_glossaries['HH_Composition_by_Ethnicity'])} combinations")
    print(f"  - Household Composition by Religion: {len(households_glossaries['HH_Composition_by_Religion'])} combinations")
    
    # Create combined glossary
    print("\nCombining all glossaries...")
    combined_glossary, all_glossaries = create_combined_glossary(individuals_glossaries, households_glossaries)
    
    # Create simple output directory
    output_dir = os.path.join(current_dir, '../outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save master glossary
    master_glossary_path = os.path.join(output_dir, f'glossary_{area_code}.csv')
    combined_glossary.to_csv(master_glossary_path, index=False)
    
    # Print summary
    print(f"\n{'='*60}")
    print("MASTER GLOSSARY CREATED")
    print(f"{'='*60}")
    print(f"Total indices: {len(combined_glossary)}")
    print(f"Total crosstables: {len(all_glossaries)}")
    
    print(f"\nCrosstables included:")
    for crosstable_name in all_glossaries.keys():
        non_empty_count = (combined_glossary[crosstable_name] != '').sum()
        print(f"  - {crosstable_name}: {non_empty_count} entries")
    
    print(f"\nSample entries:")
    print(combined_glossary.head(10))
    
    print(f"\nMaster glossary saved to: {master_glossary_path}")
    
    return True

if __name__ == "__main__":
    args = parse_arguments()
    success = create_master_glossary(args.area_code)
    
    if success:
        print(f"\n{'='*60}")
        print(f"✓ Master glossary creation completed successfully for area {args.area_code}")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print(f"✗ Master glossary creation failed for area {args.area_code}")
        print(f"{'='*60}")