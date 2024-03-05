import pandas as pd
from pathlib import Path
import os
import re

# This should execute from the root of the repository
from scripts.utils.utils import normalize_name

currpath = Path(__file__)

# if we are not running from the "cccm-structure" directory, then we need to
# change the path to the data
if not currpath.name == 'cccm-structure':
    os.chdir(currpath.parent.parent)

fdn_interlocks = pd.read_csv('data/unprocessed/foundation_cccm_interlocks.csv')

eins = pd.read_csv('data/processed/eins.csv')
name_to_ein = eins.set_index('org').ein.to_dict()

fdn_namemap = {
    'ADAM  MEYERSON': 1,
    'ALAN T RUSSELL': 2,
    'ALEJANDRA  CHAFUEN': 4,
    'ALEJANDRO  CHAFUEN': 4,
    'ALEJANDRO A CHAFUEN': 4,
    'ANTHONY  WOODLIEF': 5,
    'ARTHUR  BROOKS': 6,
    'BARBARA W KENNEY': 7,
    'BRIAN  HOOKS': 8,
    'BRUCE  JACOBS': 9,
    'BRUCE  KOVNER': 10,
    'BYRON  LAMM': 11,
    'BYRON S LAMM': 11,
    'CARL  HELSTROM': 12,
    'CARL O HELSTROM': 12,
    'CHARLES G KOCH': 13,
    'CHRIS  TALLEY': 14,
    'CHRIS L TALLEY': 14,
    'CHRISTINA  WILSON': 15,
    'CHRISTOPHER  DEMUTH': 16,
    'CONSTANCE  LARCHER': 17,
    'CONSTANCE C LARCHER': 17,
    'CURTIN  WINDSOR': 18,
    'CURTIN  WINSOR': 18,
    'D GIDEON SEARLE': 19,
    "DAN  D'ANIELLO": 20,
    "DANIEL  D'ANIELLO": 20,
    'DANIEL C SEARLE': 19,
    'DAVID  BROWN': 21,
    'DAVID  HUMPHREYS': 22,
    'DAVID  RIGGS': 23,
    'DAVID  THEROUX': 24,
    'DAVID J THEROUX': 24,
    'DAVID M STOVER': 25,
    'DAVID R BROWN': 21,
    'DAVID RIGGS DAVID RIGGS': 23,
    'DEBORAH  DONNER': 26,
    'DEBRA GAIL HUMPHREYS': 27,
    'DERWOOD S CHASE': 28,
    'EDWIN J FEULNER': 29,
    'ELIZABETH  KEEN': 30,
    'EMILIO J PACHECO': 31,
    'ETHELMAE  HUMPHREYS': 32,
    'ETHELMAE C HUMPHREYS': 32,
    'EUGENE B MEYER': 33,
    'GEORGE B MARTIN': 34,
    'GEORGE GH COATES': 35,
    'GISELE  HUFF': 36,
    'HON CURTIN WINSOR': 18,
    'INGRID A GREGG': 37,
    'IRWIN H REISS': 38,
    'JACK  ANDERSON': 39,
    'JAMES  PIERESON': 40,
    'JAMES  PIERSON': 41,
    'JAMES C RODDEY': 42,
    'JAMES R LEININGER': 43,
    'JEFF  YASS': 44,
    'JEFFREY  YASS': 44,
    'JEFFREY S YASS': 44,
    'JO ANNE W SHEPLER': 45,
    'JOANNE W SHEPLER': 45,
    'JOHN  HOOD': 46,
    'JOHN A  VON KANNON': 47,
    'JOHN A VON KANNON': 47,
    'JOHN C MALONE': 48,
    'JOHN F SNODGRASS': 49,
    'JOHN VON KANNON': 47,
    'JOHN W POPEO': 50,
    'KENNETH T CRIBB': 51,
    'KENT B HERRICK': 52,
    'KEVIN  GENTRY': 53,
    'KIMBERLY  DENNIS': 54,
    'KIMBERLY O DENNIS': 54,
    'KRIS  MAUREN': 55,
    'KRIS A MAUREN': 55,
    'KRIS ALAN MAUREN': 55,
    'LAWSON  BADER': 56,
    'MANUEL F AYAU': 57,
    'MARION G WELLS': 58,
    'MARY L  G THEROUX': 59,
    'MARY L G THEROUX': 59,
    'MICHAEL  KAISER': 60,
    'MICHAEL  KEISER': 61,
    'MICHAEL  MURRAY': 62,
    'MICHAEL LS KEISER': 61,
    'MICHAEL W GLEBA': 63,
    'MONTGOMERY B BROWN': 64,
    'PETER  COORS': 65,
    'PETER  STEPHAICH': 66,
    'PETER  STEPHANICH': 66,
    'REBEKAH  MERCER': 67,
    'RICHARD  FINK': 68,
    'RICHARD  GABY': 69,
    'RICHARD  SCAIFE': 70,
    'RICHARD H FINK': 68,
    'RICHARD M SCAIFE': 70,
    'RICHARD W DUESENBERG': 71,
    'ROBERT  HEATON': 72,
    'ROBERT  WELCH': 73,
    'ROGER  KIMBALL': 74,
    'ROGER  REAM': 75,
    'ROGER R REAM': 75,
    'RUSSELL  PENNOYER': 76,
    'SANDRA J SCHALLER': 77,
    'STEVEN  HAYWARD': 78,
    'STEVEN F HAYWARD': 78,
    'T ALAN RUSSELL': 2,
    'T KENNETH CRIBB': 51,
    'TALAN  RUSSELL': 2,
    'TERRY W ANKER': 79,
    'THOMAS  WILLCOX': 80,
    'THOMAS E BEACH': 81,
    'THOMAS L WILCOX': 80,
    'THOMAS L WILLCOX': 80,
    'THOMAS W LYLES': 82,
    'TODD W HERRICK': 83,
    'TRACIE  SHARP': 84,
    'TULLY  FRIEDMAN': 85,
    'TULLY M FRIEDMAN': 85,
    'WALTER  WILLIAMS': 86,
    'WALTER E WILLIAMS': 86,
    'WHITNEY L BALL': 87,
    'WILIAM  DUNN': 88,
    'WILLIAM  DUNN': 88,
    'WILLIAM A DUN': 88,
    'WILLIAM A DUNN': 88,
    'WILLIAM G THURMAN': 89,
    'WILLIAM H MELLOR': 90,
    'WILLIAM J HUME': 91
}


for k in [*fdn_namemap.keys()]:
    fdn_namemap[normalize_name(k)] = fdn_namemap[k]

pd.Series(fdn_namemap).to_csv('data/processed/fdn_namemap.csv')

fdn_standard_name = {
    v: k for k, v in fdn_namemap.items()
}

fdn_interlocks['PERSON_ID'] = fdn_interlocks.RECIPIENT_PERSON.map(fdn_namemap)
fdn_interlocks = fdn_interlocks[fdn_interlocks.Correct]
fdn_interlocks['GRANTMAKER_EIN'] = fdn_interlocks.GRANTMAKER.map(name_to_ein)
fdn_interlocks['RECIPIENT_EIN'] = fdn_interlocks.RECIPIENT.map(name_to_ein)

employees = pd.read_csv('data/processed/combined_employees.csv')

employees = employees[employees.name.notnull()]
employees['name_norm'] = employees.name.apply(normalize_name)
employees['name_norm'] = employees.name_norm.apply(lambda x: fdn_standard_name[fdn_namemap[x]] if x in fdn_namemap else x)
