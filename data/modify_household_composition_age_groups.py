import pandas as pd
import numpy as np

def modify_household_composition_age_groups():
    """
    Modify household composition crosstable with new age groups and handle missing values.
    Also process HH_composition_People_Main.csv file.
    """
    
    # Define age group mappings
    current_age_groups = ['0_15', '16_24', '25_34', '35_49', '50+']
    target_age_groups = ['0_4', '5_7', '8_9', '10_14', '15', '16_17', '18_19', '20_24', 
                        '25_29', '30_34', '35_39', '40_44', '45_49', '50_54', '55_59', 
                        '60_64', '65_69', '70_74', '75_79', '80_84', '85+']
    
    # Household composition codes
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
    
    # Load the household composition data
    hh_comp_file = "./preprocessed-data/crosstables/HH_composition_by_age_by_sex_Main.csv"
    hh_comp_df = pd.read_csv(hh_comp_file)
    
    # Load the household composition people data (without age/sex breakdown)
    hh_comp_people_file = "./preprocessed-data/individuals/HH_composition_People_Main.csv"
    hh_comp_people_df = pd.read_csv(hh_comp_people_file)
    
    # Load the age data for comparison
    age_file = "./preprocessed-data/individuals/Age_Perfect_5yrs.csv"
    age_df = pd.read_csv(age_file)
    
    print(f"Loaded household composition data with {len(hh_comp_df)} areas")
    print(f"Loaded household composition people data with {len(hh_comp_people_df)} areas")
    print(f"Loaded age data with {len(age_df)} areas")
    
    # Process the household composition by age and sex data
    print("\n=== Processing HH_composition_by_age_by_sex_Main.csv ===")
    process_household_composition_by_age_sex(hh_comp_df, age_df, household_composition_codes, 
                                           target_age_groups, "HH_composition_by_age_by_sex_Main_Modified.csv")
    
    # Process the household composition people data
    print("\n=== Processing HH_composition_People_Main.csv ===")
    process_household_composition_people(hh_comp_people_df, age_df, household_composition_codes,
                                       "HH_composition_People_Main_Modified.csv")

def process_household_composition_by_age_sex(hh_comp_df, age_df, household_composition_codes, 
                                           target_age_groups, output_filename):
    """
    Process household composition data with age and sex breakdowns.
    """
    
    # Merge the dataframes on geography code for faster processing
    merged_df = hh_comp_df.merge(age_df, on='geography code', suffixes=('_hh', '_age'))
    
    # Calculate missing values for all areas at once
    merged_df['missing_values'] = merged_df['total_age'] - merged_df['total_hh']
    
    print(f"Processing {len(merged_df)} areas...")
    
    # Create new dataframe with geography code and total
    new_df = merged_df[['geography code', 'total_age']].copy()
    new_df.rename(columns={'total_age': 'total'}, inplace=True)
    
    # Define logical rules for age group assignments
    age_group_rules = {
        '0_4': ['1FM-2C', '1FC-2C', '1FL-2C', '1H-2C'],
        '5_7': ['1FM-2C', '1FC-2C', '1FL-2C', '1H-2C'],
        '8_9': ['1FM-2C', '1FC-2C', '1FL-2C', '1H-2C'],
        '10_14': ['1FM-2C', '1FC-2C', '1FL-2C', '1H-2C'],
        '15': ['1FM-2C', '1FC-2C', '1FL-2C', '1H-2C'],
        '16_17': ['1FM-nA', '1FC-nA', '1FL-nA', '1H-nS', '1H-nA'],
        '18_19': ['1FM-nA', '1FC-nA', '1FL-nA', '1H-nS', '1H-nA'],
        '20_24': ['1PE', '1PA', '1FM-0C', '1FM-nA', '1FC-0C', '1FC-nA', '1H-nS', '1H-nA'],
        '25_29': ['1PE', '1PA', '1FM-0C', '1FM-nA', '1FC-0C', '1FC-nA', '1H-nA'],
        '30_34': ['1PE', '1PA', '1FM-0C', '1FM-2C', '1FM-nA', '1FC-0C', '1FC-2C', '1FC-nA', '1FL-2C', '1FL-nA', '1H-2C', '1H-nA'],
        '35_39': ['1PE', '1PA', '1FM-0C', '1FM-2C', '1FM-nA', '1FC-0C', '1FC-2C', '1FC-nA', '1FL-2C', '1FL-nA', '1H-2C', '1H-nA'],
        '40_44': ['1PE', '1PA', '1FM-0C', '1FM-2C', '1FM-nA', '1FC-0C', '1FC-2C', '1FC-nA', '1FL-2C', '1FL-nA', '1H-2C', '1H-nA'],
        '45_49': ['1PE', '1PA', '1FM-0C', '1FM-2C', '1FM-nA', '1FC-0C', '1FC-2C', '1FC-nA', '1FL-2C', '1FL-nA', '1H-2C', '1H-nA'],
        '50_54': ['1PE', '1PA', '1FM-0C', '1FM-nA', '1FC-0C', '1FC-nA', '1FL-nA', '1H-nA'],
        '55_59': ['1PE', '1PA', '1FM-0C', '1FM-nA', '1FC-0C', '1FC-nA', '1FL-nA', '1H-nA'],
        '60_64': ['1PE', '1PA', '1FM-0C', '1FM-nA', '1FC-0C', '1FC-nA', '1FL-nA', '1H-nA'],
        '65_69': ['1PE', '1PA', '1FE', '1FM-0C', '1FM-nA', '1FC-0C', '1FC-nA', '1FL-nA', '1H-nE', '1H-nA'],
        '70_74': ['1PE', '1PA', '1FE', '1FM-0C', '1FM-nA', '1FC-0C', '1FC-nA', '1FL-nA', '1H-nE', '1H-nA'],
        '75_79': ['1PE', '1PA', '1FE', '1FM-0C', '1FM-nA', '1FC-0C', '1FC-nA', '1FL-nA', '1H-nE', '1H-nA'],
        '80_84': ['1PE', '1PA', '1FE', '1FM-0C', '1FM-nA', '1FC-0C', '1FC-nA', '1FL-nA', '1H-nE', '1H-nA'],
        '85+': ['1PE', '1PA', '1FE', '1FM-0C', '1FM-nA', '1FC-0C', '1FC-nA', '1FL-nA', '1H-nE', '1H-nA']
    }
    
    # Define age group mappings for source columns
    age_group_mappings = {
        '0_15': ['0_4', '5_7', '8_9', '10_14', '15'],
        '16_24': ['16_17', '18_19', '20_24'],
        '25_34': ['25_29', '30_34'],
        '35_49': ['35_39', '40_44', '45_49'],
        '50+': ['50_54', '55_59', '60_64', '65_69', '70_74', '75_79', '80_84', '85+']
    }
    
    # Initialize all new columns with zeros - use vectorized approach
    print("Initializing columns...")
    column_names = []
    for age_group in target_age_groups:
        for sex in ['M', 'F']:
            for hh_comp in household_composition_codes:
                col_name = f"{sex} {age_group} {hh_comp}"
                column_names.append(col_name)
    
    # Create all columns at once with zeros
    for col_name in column_names:
        new_df[col_name] = 0
    
    print("Processing age group mappings...")
    
    # Process each source age group mapping using vectorized operations
    for source_age_group, target_age_groups_list in age_group_mappings.items():
        print(f"Processing {source_age_group} -> {target_age_groups_list}")
        
        # Calculate total population for this source age group across all target groups
        total_population = merged_df[target_age_groups_list].sum(axis=1)
        
        # Create a mapping of source columns to target columns for vectorized processing
        source_to_target_mapping = {}
        
        for sex in ['M', 'F']:
            for hh_comp in household_composition_codes:
                source_col = f"{sex} {source_age_group} {hh_comp}"
                
                if source_col in merged_df.columns:
                    source_values = merged_df[source_col]
                    
                    # Find all valid target columns for this source
                    valid_targets = []
                    for target_age_group in target_age_groups_list:
                        if hh_comp in age_group_rules[target_age_group]:
                            target_col = f"{sex} {target_age_group} {hh_comp}"
                            valid_targets.append((target_col, target_age_group))
                    
                    if valid_targets:
                        source_to_target_mapping[source_col] = {
                            'source_values': source_values,
                            'targets': valid_targets
                        }
        
        # Process all mappings at once
        for source_col, mapping_data in source_to_target_mapping.items():
            source_values = mapping_data['source_values']
            
            for target_col, target_age_group in mapping_data['targets']:
                # Calculate proportion based on age group population
                age_group_population = merged_df[target_age_group]
                
                # Avoid division by zero
                mask = total_population > 0
                proportions = np.where(mask, age_group_population / total_population, 0)
                
                # Calculate new values
                new_values = np.where(mask, (source_values * proportions).astype(int), 0)
                
                new_df[target_col] = new_values
    
    print("Distributing missing values with smoothed proportions across feasible HH comps...")
    alpha = 1.0  # smoothing strength; tune 0.5–3.0 if needed

    # Pre-build per-age feasible column lists per sex
    feasible_cols_per_sex_age = {
        (sex, age): [f"{sex} {age} {hh}" for hh in household_composition_codes if hh in age_group_rules[age]]
        for sex in ['M', 'F'] for age in target_age_groups
    }

    # Loop per area to allocate integer missing counts by largest-remainder method
    for idx, row in merged_df.iterrows():
        missing = int(row['missing_values'])
        if missing <= 0:
            continue

        # Build weights from current allocations + alpha over all feasible cells
        cols = []
        weights = []
        for sex in ['M', 'F']:
            for age in target_age_groups:
                c_list = feasible_cols_per_sex_age[(sex, age)]
                for col in c_list:
                    cols.append(col)
                    base = int(new_df.loc[idx, col]) if col in new_df.columns else 0
                    weights.append(base + alpha)

        weights = np.array(weights, dtype=float)
        total_w = weights.sum()
        if total_w <= 0:
            # Uniform allocation over feasible cells if no base mass
            weights = np.ones_like(weights)
            total_w = weights.sum()

        # Compute fractional allocations
        exact = missing * (weights / total_w)
        floors = np.floor(exact).astype(int)
        remainder = missing - int(floors.sum())

        if remainder > 0:
            residuals = exact - floors
            top_idx = np.argpartition(-residuals, remainder - 1)[:remainder]
            floors[top_idx] += 1

        # Apply integer allocations
        for j, col in enumerate(cols):
            if floors[j] == 0:
                continue
            new_df.loc[idx, col] += int(floors[j])
    
    # Verify totals match - vectorized
    print("Verifying totals...")
    data_cols = [col for col in new_df.columns if col not in ['geography code', 'total']]
    new_df['calculated_total'] = new_df[data_cols].sum(axis=1)
    
    # Check for any discrepancies
    discrepancies = new_df[new_df['total'] != new_df['calculated_total']]
    if len(discrepancies) > 0:
        print(f"Found {len(discrepancies)} areas with total discrepancies:")
        for idx, row in discrepancies.head(5).iterrows():
            print(f"Area {row['geography code']}: Expected {row['total']}, Got {row['calculated_total']}")

        # Fix discrepancies by proportional top-up across all feasible cells using smoothed weights
        for r_idx, r in discrepancies.iterrows():
            diff = int(r['total'] - r['calculated_total'])
            if diff == 0:
                continue

            cols = []
            weights = []
            for sex in ['M', 'F']:
                for age in target_age_groups:
                    c_list = feasible_cols_per_sex_age[(sex, age)]
                    for col in c_list:
                        cols.append(col)
                        base = int(new_df.loc[r_idx, col]) if col in new_df.columns else 0
                        weights.append(base + alpha)

            weights = np.array(weights, dtype=float)
            if weights.sum() <= 0:
                weights = np.ones_like(weights)

            exact = diff * (weights / weights.sum())
            floors = np.floor(exact).astype(int)
            remainder = diff - int(floors.sum())
            if remainder > 0:
                residuals = exact - floors
                top_idx = np.argpartition(-residuals, remainder - 1)[:remainder]
                floors[top_idx] += 1

            for j, col in enumerate(cols):
                if floors[j] == 0:
                    continue
                new_df.loc[r_idx, col] += int(floors[j])
    
    # Remove the calculated_total column before saving
    new_df = new_df.drop('calculated_total', axis=1)
    
    # Final verification
    final_data_cols = [col for col in new_df.columns if col not in ['geography code', 'total']]
    new_df['final_check'] = new_df[final_data_cols].sum(axis=1)
    final_discrepancies = new_df[new_df['total'] != new_df['final_check']]
    if len(final_discrepancies) > 0:
        print(f"Warning: {len(final_discrepancies)} areas still have discrepancies after fixing")
    else:
        print("All totals match correctly!")
    
    new_df = new_df.drop('final_check', axis=1)
    
    # Save the modified data
    output_file = f"./preprocessed-data/crosstables/{output_filename}"
    new_df.to_csv(output_file, index=False)
    print(f"Modified data saved to: {output_file}")
    print(f"New data shape: {new_df.shape}")
    
    # Print summary statistics
    print(f"Total areas processed: {len(new_df)}")
    print(f"Missing values distributed: {merged_df['missing_values'].sum()}")
    print(f"Average missing values per area: {merged_df['missing_values'].mean():.1f}")

def process_household_composition_people(hh_comp_people_df, age_df, household_composition_codes, output_filename):
    """
    Process household composition people data (without age/sex breakdown).
    """
    
    # Merge the dataframes on geography code
    merged_df = hh_comp_people_df.merge(age_df, on='geography code', suffixes=('_hh', '_age'))
    
    # Calculate missing values for all areas at once
    merged_df['missing_values'] = merged_df['total_age'] - merged_df['total_hh']
    
    print(f"Processing {len(merged_df)} areas...")
    
    # Create new dataframe with geography code and total
    new_df = merged_df[['geography code', 'total_age']].copy()
    new_df.rename(columns={'total_age': 'total'}, inplace=True)
    
    # Copy all household composition columns
    for col in household_composition_codes:
        new_df[col] = merged_df[col]
    
    print("Distributing missing values to first 5 categories...")
    
    # Distribute missing values equally to the first 5 household composition categories - vectorized
    first_5_categories = ['1PE', '1PA', '1FE', '1FM-0C', '1FM-2C']
    
    # Calculate missing values per category - vectorized
    missing_per_category = merged_df['missing_values'] // len(first_5_categories)
    remainder = merged_df['missing_values'] % len(first_5_categories)
    
    # Add missing values to each of the first 5 categories - vectorized
    for i, hh_comp in enumerate(first_5_categories):
        additional_values = missing_per_category.copy()
        # Add remainder to the first category
        if i == 0:
            additional_values += remainder
        new_df[hh_comp] += additional_values
    
    # Verify totals match - vectorized
    print("Verifying totals...")
    data_cols = [col for col in new_df.columns if col not in ['geography code', 'total']]
    new_df['calculated_total'] = new_df[data_cols].sum(axis=1)
    
    # Check for any discrepancies
    discrepancies = new_df[new_df['total'] != new_df['calculated_total']]
    if len(discrepancies) > 0:
        print(f"Found {len(discrepancies)} areas with total discrepancies:")
        for idx, row in discrepancies.head(5).iterrows():
            print(f"Area {row['geography code']}: Expected {row['total']}, Got {row['calculated_total']}")
        
        # Fix discrepancies by distributing equally to first 5 categories - vectorized
        first_5_categories = ['1PE', '1PA', '1FE', '1FM-0C', '1FM-2C']
        for idx, row in discrepancies.iterrows():
            diff = row['total'] - row['calculated_total']
            if diff != 0:
                # Distribute the difference equally among the first 5 categories
                missing_per_category = diff // len(first_5_categories)
                remainder = diff % len(first_5_categories)
                
                for i, hh_comp in enumerate(first_5_categories):
                    additional = missing_per_category
                    # Add remainder to the first category
                    if i == 0:
                        additional += remainder
                    new_df.loc[idx, hh_comp] += additional
    
    # Remove the calculated_total column before saving
    new_df = new_df.drop('calculated_total', axis=1)
    
    # Final verification
    final_data_cols = [col for col in new_df.columns if col not in ['geography code', 'total']]
    new_df['final_check'] = new_df[final_data_cols].sum(axis=1)
    final_discrepancies = new_df[new_df['total'] != new_df['final_check']]
    if len(final_discrepancies) > 0:
        print(f"Warning: {len(final_discrepancies)} areas still have discrepancies after fixing")
    else:
        print("All totals match correctly!")
    
    new_df = new_df.drop('final_check', axis=1)
    
    # Save the modified data
    output_file = f"./preprocessed-data/individuals/{output_filename}"
    new_df.to_csv(output_file, index=False)
    print(f"Modified data saved to: {output_file}")
    print(f"New data shape: {new_df.shape}")
    
    # Print summary statistics
    print(f"Total areas processed: {len(new_df)}")
    print(f"Missing values distributed: {merged_df['missing_values'].sum()}")
    print(f"Average missing values per area: {merged_df['missing_values'].mean():.1f}")

if __name__ == "__main__":
    modify_household_composition_age_groups() 