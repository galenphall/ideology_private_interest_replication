import infomap
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

import sys
sys.path.append('code/')
from utils.utils import int_to_ein

import os

if os.getcwd().split('/')[-1] != 'cccm-structure':
    os.chdir('..')

topic_proportions = pd.read_parquet('data/processed/topic_proportions.parquet')
topics = pd.read_parquet('data/processed/topic_labels.parquet')[['topic', 'label']]
metadata = pd.read_csv('data/processed/metadata.csv')
eins = pd.read_csv('data/processed/eins.csv')

topic_proportions['ein'] = topic_proportions['org'].map(eins.set_index('org').ein.to_dict())
topic_proportions = topic_proportions[topic_proportions.ein.notna()]
topic_proportions['ein'] = topic_proportions['ein'].apply(int_to_ein)
ein_topic_proportions = topic_proportions.groupby('ein')[topics.topic].mean()

print(metadata.columns)

# map eins to org types
ein_org_types = metadata.set_index('ein').organization_type.to_dict()

topic_proportions['org_type'] = topic_proportions['ein'].map(ein_org_types)

topics_org_year = topic_proportions.groupby(['org_type', 'year'])[topics.topic].mean()

topics_org_year.to_csv('data/processed/topic_proportions_by_org_type_by_year.csv')

p = topic_proportions[topics.topic].copy()
p = p.div(p.sum(1), 0)

org_topics = p.join(topic_proportions.ein).groupby(['ein']).sum().T
org_topics = org_topics.div(org_topics.sum(), 1)

A_topics = org_topics.corr(method=lambda a, b: 1 - jensenshannon(a, b))

np.fill_diagonal(A_topics.values, np.nan)

G_topics = nx.Graph((A_topics - A_topics.stack().mean()).replace(np.nan, 0).applymap(lambda x: max(x, 0)))
for ein in eins.ein:
    G_topics.add_node(ein)

im = infomap.Infomap(markov_time = 1, silent = True, no_self_links=True,)
im.add_networkx_graph(G_topics)
im.run()

topic_partition = im.get_dataframe(['name','module_id']).set_index('name').module_id
topic_partition = topic_partition[topic_partition.module_id.map(topic_partition.module_id.value_counts()) > 2]# This just eliminates a few single-node modules
