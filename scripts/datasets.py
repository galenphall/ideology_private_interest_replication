import pandas as pd
from pathlib import Path
import os
currpath = Path(__file__)

# if we are not running from the "cccm-structure" directory, then we need to
# change the path to the data
if not currpath.name == 'cccm-structure':
    os.chdir(currpath.parent.parent)

eins = pd.read_csv('data/processed/eins.csv')

docs = pd.read_csv('data/processed/docs.csv')

topic_labels = pd.read_parquet('data/processed/topic_labels.parquet')

topic_proportions = pd.read_parquet('data/processed/topic_proportions.parquet')

metadata = pd.read_csv('data/processed/metadata.csv')

grants = pd.read_csv('data/processed/grants.csv')

# Summary statistics
def summary():
    number_unique_cccm_orgs = len(metadata[metadata.organization_type != 'Foundation'].org_level_1.unique())
    number_unique_cccm_orgs_after_combining = len(metadata[metadata.organization_type != 'Foundation'].org_level_2.unique())
    print(f'Number of unique CCCM organizations: {number_unique_cccm_orgs}')
    print(f'Number of unique cccm organizations after combining parent/child orgs: {number_unique_cccm_orgs_after_combining}')
    print("Renamed organizations: ")
    print(metadata[(metadata.org_level_1 != metadata.org_level_2)][['org_level_1', 'org_level_2']])
    print()
    print(f"Count by designation: \n{metadata.organization_type.value_counts()}")
    print(f"Count by designation after combining parent/child orgs: \n{metadata.drop_duplicates('org_level_2').organization_type.value_counts()}")
    print(f"Number of unique second-level orgs: {len(metadata.org_level_2.unique())}")

    # Number of documents overall
    print(metadata.num_docs.sum())

    # Number of documents by designation
    print(metadata.groupby('organization_type').num_docs.sum())

    print(docs['org'].value_counts().reindex(metadata[metadata.organization_type != 'Foundation'].org_level_2.unique()).fillna(0).astype(int).sort_values().head(10))

    print([o for o in docs['org'].unique() if o not in metadata[metadata.organization_type != 'Foundation'].org_level_2.unique()])
    print([o for o in metadata[metadata.organization_type != 'Foundation'].org_level_2.unique() if o not in docs['org'].unique()])

    print(len(docs['org'].unique()))
    print(len(metadata[metadata.organization_type != 'Foundation'].org_level_2.unique()))