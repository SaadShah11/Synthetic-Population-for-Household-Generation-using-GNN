import pandas as pd
import os
import numpy as np
import random

# Get the current directory of the script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Paths to the CSV files
marital_file_path = os.path.join(current_dir, 'crosstables/MaritalbySexbyAge.csv')
religion_file_path = os.path.join(current_dir, 'crosstables/ReligionbySexbyAge.csv')
new_file_path = os.path.join(current_dir, 'crosstables/MaritalbySexbyAgeModified.csv')

# Read the CSV files
marital_df = pd.read_csv(marital_file_path)
religion_df = pd.read_csv(religion_file_path)

# Define the new age groups, sex categories, and marital statuses
new_age_groups = ['0_4', '5_7', '8_9', '10_14', '15']
sex_categories = ['M', 'F']
marital_statuses = ['Single', 'Married', 'Partner', 'Separated', 'Divorced', 'Widowed']

# Debug - check column creation
print("Creating new columns for sex and age groups:")
print("Sex categories:", sex_categories)
print("Age groups:", new_age_groups)

# Create new columns for each combination and initialize to 0
created_columns = []
for sex in sex_categories:
    for age in new_age_groups:
        for marital in marital_statuses:
            column_name = f'{sex} {age} {marital}'
            marital_df[column_name] = 0
            created_columns.append(column_name)

# Verify all expected columns were created
print(f"Created {len(created_columns)} new columns")
for sex in sex_categories:
    sex_cols = [col for col in created_columns if col.startswith(sex)]
    print(f"  {sex}: {len(sex_cols)} columns created")

# Calculate offset for each geography code and update total column
offset_by_geo = {}
for geo_code in marital_df['geography code'].unique():
    marital_total = marital_df.loc[marital_df['geography code'] == geo_code, 'total'].values[0]
    
    # Find corresponding religion total
    if geo_code in religion_df['geography code'].values:
        religion_total = religion_df.loc[religion_df['geography code'] == geo_code, 'total'].values[0]
        offset = religion_total - marital_total
        
        # Update the 'total' column in marital_df to match religion_total
        marital_df.loc[marital_df['geography code'] == geo_code, 'total'] = religion_total
    else:
        # If no match in religion data, set offset to 0
        offset = 0
    
    offset_by_geo[geo_code] = offset

# Define distribution ratios for different age groups (decreasing as age increases)
# Higher values for younger ages for "Single" status
age_ratios = {
    '0_4': 0.40,   # 40% of offset to 0-4 age group
    '5_7': 0.25,   # 25% of offset to 5-7 age group
    '8_9': 0.15,   # 15% of offset to 8-9 age group
    '10_14': 0.15, # 15% of offset to 10-14 age group
    '15': 0.05     # 5% of offset to 15 age group
}

# Sex distribution ratio (roughly equal)
sex_ratios = {'M': 0.5, 'F': 0.5}

# Track assigned values for verification
assigned_values = {age: 0 for age in new_age_groups}

# Apply the offsets to the new columns
for idx, row in marital_df.iterrows():
    geo_code = row['geography code']
    offset = offset_by_geo[geo_code]
    
    if offset > 0:  # Only distribute if there's a positive offset
        for sex in sex_categories:
            for age in new_age_groups:
                # Assign the calculated values to the 'Single' marital status
                single_col = f'{sex} {age} Single'
                
                # Ensure the column exists
                if single_col not in marital_df.columns:
                    print(f"WARNING: Column {single_col} does not exist!")
                    continue
                    
                single_val = round(offset * age_ratios[age] * sex_ratios[sex])
                marital_df.at[idx, single_col] = single_val
                assigned_values[age] += single_val

# Verify values were assigned to each age group
print("\nValues assigned to age groups:")
for age in new_age_groups:
    print(f"  {age}: {assigned_values[age]}")

# Reformat all column headers to have Sex first, then Age, then Marital status
print("\nReformatting all column headers to have Sex first, then Age, then Marital status...")
columns_to_rename = {}

# Define known categories for matching
all_age_groups = new_age_groups + ['16_17', '18_19', '20_24', '25_29', '30_34', '35_39', '40_44', '45_49', 
                                  '50_54', '55_59', '60_64', '65_69', '70_74', '75_79', '80_84', '85+']

# Skip geography code and total columns
for col in marital_df.columns:
    if col in ['geography code', 'total']:
        continue
    
    # Skip already correctly formatted columns
    if any(col.startswith(f"{sex} ") for sex in sex_categories):
        continue
    
    parts = col.split()
    
    # Try to identify Sex, Age, and Marital status in the column name
    identified_sex = None
    identified_age = None
    identified_marital = None
    
    for part in parts:
        if part in sex_categories:
            identified_sex = part
        elif part in all_age_groups:
            identified_age = part
        elif part in marital_statuses:
            identified_marital = part
    
    # If we identified all three components, create a new column name
    if identified_sex and identified_age and identified_marital:
        new_col = f"{identified_sex} {identified_age} {identified_marital}"
        columns_to_rename[col] = new_col

# Rename the columns
if columns_to_rename:
    print(f"Renaming {len(columns_to_rename)} columns to the new format...")
    marital_df = marital_df.rename(columns=columns_to_rename)
    for old_col, new_col in columns_to_rename.items():
        print(f"  {old_col} -> {new_col}")
else:
    print("No columns need to be renamed.")

# Create the final column order: 'geography code', 'total', followed by all other columns in their original order
final_column_order = ['geography code', 'total'] + [col for col in marital_df.columns if col not in ['geography code', 'total']]

# Reorder the DataFrame columns
marital_df = marital_df[final_column_order]

# Verify the column order
print("\nFinal column order verification:")
print(f"First 10 columns: {', '.join(marital_df.columns[:10])}")

# List columns to verify all sex categories are included before saving
print("\nFinal column check by sex category:")
sex_counts = {sex: 0 for sex in sex_categories}
for col in marital_df.columns:
    for sex in sex_categories:
        if col.startswith(f"{sex} "):
            sex_counts[sex] += 1
            
for sex, count in sex_counts.items():
    print(f"  {sex}: {count} columns")

# Save the updated DataFrame to the new CSV file
print(f"\nSaving to {new_file_path}...")
marital_df.to_csv(new_file_path, index=False)
print("File saved.")

# Print information about the target geography code
target_geo_code = 'E02005924'
print(f"\nDetails for geography code {target_geo_code}:")
print("-------------------------------------------")

if target_geo_code in marital_df['geography code'].values:
    # Get the row for this geography code
    target_row = marital_df[marital_df['geography code'] == target_geo_code].iloc[0]
    
    # Extract total and original offset
    target_total = target_row['total']
    target_offset = offset_by_geo.get(target_geo_code, 0)
    
    print(f"Total: {target_total}")
    print(f"Offset: {target_offset}")
    print("\nNew combinations with assigned values:")
    
    # Print values for each sex category to verify all are included
    for sex in sex_categories:
        print(f"\nSex category: {sex}")
        for age in new_age_groups:
            print(f"  Age group: {age}")
            has_values = False
            for marital in marital_statuses:
                col_name = f"{sex} {age} {marital}"
                if col_name in target_row:
                    value = target_row[col_name]
                    if value > 0:
                        print(f"    {col_name}: {value}")
                        has_values = True
                else:
                    print(f"    WARNING: Column {col_name} not found in data!")
            
            if not has_values:
                print(f"    No values > 0 for this combination")
    
    # Calculate and print the total for new combinations
    new_cols = [f"{sex} {age} {marital}" for sex in sex_categories 
                for age in new_age_groups for marital in marital_statuses]
    existing_cols = [col for col in new_cols if col in target_row]
    new_total = sum(target_row[col] for col in existing_cols)
    print(f"\nSum of all new combinations: {new_total}")
    
    # Calculate and print the grand total by summing all data columns
    # (exclude 'geography code' and 'total' columns)
    data_cols = [col for col in target_row.index if col not in ['geography code', 'total']]
    grand_total = sum(target_row[col] for col in data_cols)
    print(f"\nGRAND TOTAL (sum of ALL columns): {grand_total}")
    
    # Verify that the grand total matches the total column
    if abs(grand_total - target_total) < 0.01:  # Allow for small rounding differences
        print(f"Verification: GRAND TOTAL matches 'total' column value ({target_total})")
    else:
        print(f"WARNING: GRAND TOTAL ({grand_total}) does not match 'total' column value ({target_total})")
        print(f"Difference: {grand_total - target_total}")
else:
    print(f"Geography code {target_geo_code} not found in the data.")
