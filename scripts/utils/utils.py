from infomap import Infomap
import numpy as np
import networkx as nx
import re

def int_to_ein(i):
    if isinstance(i, str):
        return i
    else:
        eid_no_dash = str(int(i)).zfill(9)
        return eid_no_dash[:2] + '-' + eid_no_dash[2:]

def normalize_name(name):
    name = name.upper()
    name = name.replace('.', '')
    name = re.sub(" +", " ", name)
    name = name.strip()
    return name