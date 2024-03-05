import json
import os

import pandas as pd

from scripts.utils import utils  # Ensure correct imports

# Set the correct directory
os.chdir('..')

# Load data
intra_network_interlocks = pd.read_excel('data/unprocessed/validated_cccm_interlocks.xlsx')
intra_network_interlocks.to_csv('data/unprocessed/intra_network_interlocks.csv', index=False)

# Define entries to remove
ein_to_remove = "13-1623899"
name_to_remove = "AMERICAN SOCIETY OF MECHANICAL ENGINEERS"

# Initialize dictionary for dataframes
dfs = {}

# Process files
for file in [
    "foundation_metadata.csv",
    "cccm_metadata.csv",
    "cccm_officers.csv",
    "eins.csv",
]:
    df = pd.read_csv(f"data/unprocessed/{file}").drop(columns=["Unnamed: 0"], errors="ignore")

    # Remove the ASME entry from any row that contains it in any column
    df = df[~df.apply(lambda x: (x == ein_to_remove) | (x == name_to_remove)).any(axis=1)]

    # Standardize column names
    df.columns = [c.lower().replace(' ', '_') for c in df.columns]

    # Store and save processed dataframe
    dfs[file] = df
    df.to_csv(f"data/processed/{file}", index=False)

# Merge metadata
cccm_metadata = dfs["cccm_metadata.csv"]
foundation_metadata = dfs["foundation_metadata.csv"]

# Add new columns
cccm_metadata['organization_type'] = cccm_metadata.nonprofit_type
foundation_metadata['organization_type'] = 'Foundation'
cccm_metadata['combined_name'] = cccm_metadata.controlling_entity
foundation_metadata['combined_name'] = foundation_metadata.combine_name

# Find common columns
overlap = set(cccm_metadata.columns).intersection(set(foundation_metadata.columns))

# Select relevant columns and standardize column names
cccm_metadata = cccm_metadata[list(
    overlap |
    {'organization_type', 'combined_name'} |
    {'denial', 'delay', 'sole_climate_energy', 'substantive_focus', 'peripheral_focus'})
]
cccm_metadata.columns = cccm_metadata.columns.str.lower()
cccm_metadata = cccm_metadata.rename(columns={'sole_climate_energy': 'sole_focus'})

# Select relevant columns for foundation metadata and standardize column names
foundation_metadata = foundation_metadata[list(overlap)]
foundation_metadata.columns = foundation_metadata.columns.str.lower()

# Merge metadata and remove duplicates
merged_metadata = pd.concat([cccm_metadata, foundation_metadata])
merged_metadata = merged_metadata.drop_duplicates(subset=['ein'])

# Apply transformations
merged_metadata['ein'] = merged_metadata['ein'].apply(utils.int_to_ein)
merged_metadata = merged_metadata.applymap(lambda x: x.strip().replace('\n', '-') if isinstance(x, str) else x)

# Save merged metadata
merged_metadata.to_csv('data/processed/metadata.csv', index=False)

# Load EINs
eins = pd.read_csv('data/processed/eins.csv')
name_to_ein = dict(zip(eins['org'], eins['ein']))

# Load finances
cccm_finances_loc = 'data/unprocessed/finances.json'
with open(cccm_finances_loc) as f:
    cccm_finances = json.load(f)

# Initialize dictionary for finances
finances = []

# Process finances
for org in cccm_finances:

    # Get the data
    fdata = cccm_finances[org]

    # Standardize name, we have to special case this one
    if org == 'JAMES G':
        org = 'JAMES G. MARTIN CENTER FOR ACADEMIC RENEWAL'
    if org == 'AMERICAN SOCIETY OF MECHANICAL ENGINEERS':
        continue
    ein = name_to_ein[org]

    for finance_type in fdata:
        ffdata = fdata[finance_type]
        finances.extend([
            {'year': year, 'report':finance_type, 'amount': amount, 'kind': kind, 'ein': ein}
            for year in ffdata for kind, amount in ffdata[year].items()
        ])

# Save finances
df = pd.DataFrame(finances)
df.to_csv(f'data/processed/finances.csv', index=False)
