import os
from pathlib import Path

import numpy as np
import pandas as pd

# This should execute from the root of the repository
from scripts.utils.utils import normalize_name

currpath = Path(__file__)

# if we are not running from the "cccm-structure" directory, then we need to
# change the path to the data
if not currpath.name == 'cccm-structure':
    os.chdir(currpath.parent.parent)


def load_and_process_employees():
    # load cccm and foundation officers
    # path = 'data/processed/cccm_officers.csv'
    currpath = Path(__file__)

    # if we are not running from the "cccm-structure" directory, then we need to
    # change the path to the data
    if not currpath.name == 'cccm-structure':
        currpath = currpath.parent.parent

    path = currpath / 'data/unprocessed/cccm_employees.csv'
    if not path.exists():
        raise FileNotFoundError(f'Could not find {path}')
    cccm_employees = pd.read_csv(path)

    path = currpath / 'data/unprocessed/foundation_employees.csv'
    foundation_employees = pd.read_csv(path)

    cccm_employees.columns = foundation_employees.columns

    combined_employees = pd.concat([cccm_employees, foundation_employees])
    combined_employees.columns = combined_employees.columns.str.lower()

    eins = pd.read_csv(currpath / 'data/unprocessed/eins.csv')

    combined_employees['ein'] = combined_employees.org.map(eins.set_index('ORG').EIN)

    combined_employees = combined_employees[['ein', 'name', 'position', 'year', 'org']]

    combined_employees.to_csv(currpath / 'data/processed/combined_employees.csv', index=False)

    combined_employees = combined_employees.dropna(axis=0, subset=['name'])

    eins = pd.read_csv(currpath / 'data/unprocessed/eins.csv')
    combined_employees['ein'] = combined_employees.org.map(eins.set_index('ORG').EIN)

    cccm_officers = pd.read_csv(currpath / 'data/unprocessed/cccm_officers.csv')
    cccm_officers['position'] = 'officer'
    cccm_officers = cccm_officers[cccm_officers.name.notna()]
    combined_employees = pd.concat([combined_employees, cccm_officers[['ein', 'name', 'position', 'year', 'org']]])
    return combined_employees


# employees has structure:
# ein, name, position, year, org
employees = load_and_process_employees()
employees['normalized_name'] = employees.name.map(normalize_name)

# We want to map (name, ein, year) to unique_id since this is the unique identifier
# for a person in the database. We could use (name, ein) but this would
# cause problems if two people with the same name worked at the same organization
# in the same year. This is unlikely but possible.

person_ids = employees[['name', 'normalized_name', 'ein', 'year', 'position', 'org']].drop_duplicates().copy()
person_ids['unique_id'] = range(1, len(person_ids) + 1)

# Now we want to identify person_ids using the confirmed interlocks.
# We will use the following strategy:
# 1. Create a dictionary mapping (name, ein, year) to unique_id
# 2. For each confirmed interlock, we will use the smaller of the two unique_ids as the person_id for both
#    people.
# 3. We will then use the person_ids to identify the remaining unique_ids

cccm_interlocks = pd.read_json('data/unprocessed/cccm_interlocks.json')

# First, we need to create a dictionary mapping (name, ein, year) to unique_id
name_ein_year_to_unique_id = person_ids.set_index(['normalized_name', 'ein', 'year']).unique_id.to_dict()

# Now we need to create a dictionary mapping (name, ein, year) to person_id
# We will use the smaller of the two unique_ids as the person_id for both people

verified_interlocks = cccm_interlocks[cccm_interlocks.Verified]

unique_id_to_person_id = {unique_id: unique_id for unique_id in person_ids.unique_id}
for row in verified_interlocks.itertuples():
    person_1 = {
        'names': row.group1Names,
        'ein': row.group1ID.split('_')[0],
        'years': row.group1Years,
        'org': row.group1Org,
    }
    person_2 = {
        'names': row.group2Names,
        'ein': row.group2ID.split('_')[0],
        'years': row.group2Years,
        'org': row.group2Org,
    }
    updated_id = np.inf
    unique_ids = []
    for person in [person_1, person_2]:
        for name in person['names']:
            for year in person['years']:
                normalized_name = normalize_name(name)

                if normalized_name not in person_ids.normalized_name.values:
                    print(f'Could not find {normalized_name} from {person["org"]} in person_ids')

                # Not every combination of (name, ein, year) will be in the dictionary
                if (normalized_name, person['ein'], year) not in name_ein_year_to_unique_id:
                    continue

                unique_id = name_ein_year_to_unique_id[(normalized_name, person['ein'], year)]

                updated_id = min(unique_id, updated_id)
                unique_ids.append(unique_id)

    for unique_id in unique_ids:
        unique_id_to_person_id[unique_id] = updated_id

# Now we further update the unique ids using the foundation interlocks
foundation_interlocks = pd.read_csv('data/unprocessed/foundation_cccm_interlocks.csv')
for name_1, name_2 in foundation_interlocks[['GRANTMAKER_PERSON', 'RECIPIENT_PERSON']].values:
    normalized_name_1 = normalize_name(name_1)
    normalized_name_2 = normalize_name(name_2)
    name_1_records = person_ids[person_ids.normalized_name == normalized_name_1]
    name_2_records = person_ids[person_ids.normalized_name == normalized_name_2]
    updated_id = np.inf
    for ein, year in name_1_records[['ein', 'year']].values:
        if (normalized_name_1, ein, year) not in name_ein_year_to_unique_id:
            continue
        unique_id = name_ein_year_to_unique_id[(normalized_name_1, ein, year)]
        updated_id = min(unique_id, updated_id)
    for ein, year in name_2_records[['ein', 'year']].values:
        if (normalized_name_2, ein, year) not in name_ein_year_to_unique_id:
            continue
        unique_id = name_ein_year_to_unique_id[(normalized_name_2, ein, year)]
        updated_id = min(unique_id, updated_id)

    for ein, year in name_1_records[['ein', 'year']].values:
        if (normalized_name_1, ein, year) not in name_ein_year_to_unique_id:
            continue
        unique_id = name_ein_year_to_unique_id[(normalized_name_1, ein, year)]
        unique_id_to_person_id[unique_id] = updated_id
    for ein, year in name_2_records[['ein', 'year']].values:
        if (normalized_name_2, ein, year) not in name_ein_year_to_unique_id:
            continue
        unique_id = name_ein_year_to_unique_id[(normalized_name_2, ein, year)]
        unique_id_to_person_id[unique_id] = updated_id

# Finally, update using hand-grouped names of people from foundation boards
fdn_namemap = pd.Series({'ADAM  MEYERSON': 1, 'ALAN T RUSSELL': 2, 'ALEJANDRA  CHAFUEN': 3, 'ALEJANDRO  CHAFUEN': 4,
               'ALEJANDRO A CHAFUEN': 4, 'ANTHONY  WOODLIEF': 5, 'ARTHUR  BROOKS': 6, 'BARBARA W KENNEY': 7,
               'BRIAN  HOOKS': 8, 'BRUCE  JACOBS': 9, 'BRUCE  KOVNER': 10, 'BYRON  LAMM': 11, 'BYRON S LAMM': 11,
               'CARL  HELSTROM': 12, 'CARL O HELSTROM': 12, 'CHARLES G KOCH': 13, 'CHRIS  TALLEY': 14,
               'CHRIS L TALLEY': 14, 'CHRISTINA  WILSON': 15, 'CHRISTOPHER  DEMUTH': 16, 'CONSTANCE  LARCHER': 17,
               'CONSTANCE C LARCHER': 17, 'CURTIN  WINDSOR': 18, 'CURTIN  WINSOR': 18, 'D GIDEON SEARLE': 19,
               "DAN  D'ANIELLO": 20, "DANIEL  D'ANIELLO": 20, 'DANIEL C SEARLE': 19, 'DAVID  BROWN': 21,
               'DAVID  HUMPHREYS': 22, 'DAVID  RIGGS': 23, 'DAVID  THEROUX': 24, 'DAVID J THEROUX': 24,
               'DAVID M STOVER': 25, 'DAVID R BROWN': 21, 'DAVID RIGGS DAVID RIGGS': 23, 'DEBORAH  DONNER': 26,
               'DEBRA GAIL HUMPHREYS': 27, 'DERWOOD S CHASE': 28, 'EDWIN J FEULNER': 29, 'ELIZABETH  KEEN': 30,
               'EMILIO J PACHECO': 31, 'ETHELMAE  HUMPHREYS': 32, 'ETHELMAE C HUMPHREYS': 32, 'EUGENE B MEYER': 33,
               'GEORGE B MARTIN': 34, 'GEORGE GH COATES': 35, 'GISELE  HUFF': 36, 'HON CURTIN WINSOR': 18,
               'INGRID A GREGG': 37, 'IRWIN H REISS': 38, 'JACK  ANDERSON': 39, 'JAMES  PIERESON': 40,
               'JAMES  PIERSON': 41, 'JAMES C RODDEY': 42, 'JAMES R LEININGER': 43, 'JEFF  YASS': 44,
               'JEFFREY  YASS': 44, 'JEFFREY S YASS': 44, 'JO ANNE W SHEPLER': 45, 'JOANNE W SHEPLER': 45,
               'JOHN  HOOD': 46, 'JOHN A  VON KANNON': 47, 'JOHN A VON KANNON': 47, 'JOHN C MALONE': 48,
               'JOHN F SNODGRASS': 49, 'JOHN VON KANNON': 47, 'JOHN W POPEO': 50, 'KENNETH T CRIBB': 51,
               'KENT B HERRICK': 52, 'KEVIN  GENTRY': 53, 'KIMBERLY  DENNIS': 54, 'KIMBERLY O DENNIS': 54,
               'KRIS  MAUREN': 55, 'KRIS A MAUREN': 55, 'KRIS ALAN MAUREN': 55, 'LAWSON  BADER': 56,
               'MANUEL F AYAU': 57, 'MARION G WELLS': 58, 'MARY L  G THEROUX': 59, 'MARY L G THEROUX': 59,
               'MICHAEL  KAISER': 60, 'MICHAEL  KEISER': 61, 'MICHAEL  MURRAY': 62, 'MICHAEL LS KEISER': 61,
               'MICHAEL W GLEBA': 63, 'MONTGOMERY B BROWN': 64, 'PETER  COORS': 65, 'PETER  STEPHAICH': 66,
               'PETER  STEPHANICH': 66, 'REBEKAH  MERCER': 67, 'RICHARD  FINK': 68, 'RICHARD  GABY': 69,
               'RICHARD  SCAIFE': 70, 'RICHARD H FINK': 68, 'RICHARD M SCAIFE': 70, 'RICHARD W DUESENBERG': 71,
               'ROBERT  HEATON': 72, 'ROBERT  WELCH': 73, 'ROGER  KIMBALL': 74, 'ROGER  REAM': 75, 'ROGER R REAM': 75,
               'RUSSELL  PENNOYER': 76, 'SANDRA J SCHALLER': 77, 'STEVEN  HAYWARD': 78, 'STEVEN F HAYWARD': 78,
               'T ALAN RUSSELL': 2, 'T KENNETH CRIBB': 51, 'TALAN  RUSSELL': 2, 'TERRY W ANKER': 79,
               'THOMAS  WILLCOX': 80, 'THOMAS E BEACH': 81, 'THOMAS L WILCOX': 80, 'THOMAS L WILLCOX': 80,
               'THOMAS W LYLES': 82, 'TODD W HERRICK': 83, 'TRACIE  SHARP': 84, 'TULLY  FRIEDMAN': 85,
               'TULLY M FRIEDMAN': 85, 'WALTER  WILLIAMS': 86, 'WALTER E WILLIAMS': 86, 'WHITNEY L BALL': 87,
               'WILIAM  DUNN': 88, 'WILLIAM  DUNN': 88, 'WILLIAM A DUN': 88, 'WILLIAM A DUNN': 88,
               'WILLIAM G THURMAN': 89, 'WILLIAM H MELLOR': 90, 'WILLIAM J HUME': 91
               })

fdn_namemap.index = fdn_namemap.index.map(normalize_name)
for number, group in fdn_namemap.groupby(fdn_namemap):
    if len(group) > 1:
        normalized_names = [normalize_name(name) for name in group.index.tolist()]
        records = person_ids[person_ids['name'].isin(normalized_names)]
        unique_ids = records.unique_id.unique()
        min_person_id = min(unique_id_to_person_id[unique_id] for unique_id in unique_ids)
        for unique_id in unique_ids:
            unique_id_to_person_id[unique_id] = min_person_id

person_ids['person_id'] = person_ids['unique_id'].map(unique_id_to_person_id)
person_ids.to_csv('data/processed/merged_employees.csv', index=False)