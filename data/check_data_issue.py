import pandas as pd

# Load the data
df = pd.read_csv('./preprocessed-data/crosstables/HH_composition_by_age_by_sex_Main.csv')

# Check area E02005949
area = df[df['geography code'] == 'E02005949']
if not area.empty:
    print('Total from file:', area['total'].iloc[0])
    
    # Get all columns except geography code and total
    cols = [col for col in df.columns if col not in ['geography code', 'total']]
    print('Sum of all columns:', area[cols].sum(axis=1).iloc[0])
    print('Number of columns:', len(cols))
    
    # Check if there are any negative values
    negative_cols = area[cols].columns[area[cols].iloc[0] < 0]
    if len(negative_cols) > 0:
        print('Negative values found in columns:', list(negative_cols))
    
    # Check if there are any NaN values
    nan_cols = area[cols].columns[area[cols].iloc[0].isna()]
    if len(nan_cols) > 0:
        print('NaN values found in columns:', list(nan_cols))
        
    # Show first few columns and their values
    print('\nFirst 10 columns and their values:')
    for i, col in enumerate(cols[:10]):
        print(f"{col}: {area[col].iloc[0]}")
        
    # Show last few columns and their values
    print('\nLast 10 columns and their values:')
    for i, col in enumerate(cols[-10:]):
        print(f"{col}: {area[col].iloc[0]}")
else:
    print('Area E02005949 not found in the data') 