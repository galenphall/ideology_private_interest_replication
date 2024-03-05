import infomap
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
import os

if not os.getcwd().endswith('cccm-structure'):
    print(os.getcwd())
    os.chdir('..')

print(os.getcwd())

# Load data
topic_proportions = pd.read_parquet('data/processed/topic_proportions.parquet')
metadata = pd.read_csv('data/processed/metadata.csv')
topic_labels = pd.read_csv('data/unprocessed/topiclabels.csv').drop('Unnamed: 0', axis=1, errors='ignore')
topic_labels = topic_labels[~topic_labels.remove].copy()
topic_labels['topic'] = 'V' + topic_labels.topic.astype(str)
topics_to_retain = topic_labels.topic.values
topic_to_label = topic_labels.set_index('topic').label.to_dict()
docs = pd.read_parquet('data/processed/docs.parquet')

# map org names to org types
org_types = metadata.set_index('org_level_2').organization_type.to_dict()

# Create topic proportions similarity matrix
p = topic_proportions[topics_to_retain].copy()
p = p.div(p.sum(1), 0)

org_topics = p.join(topic_proportions.org).groupby(['org']).sum().T
org_topics = org_topics.div(org_topics.sum(), 1)

A_topics = org_topics.corr(method=lambda a, b: 1 - jensenshannon(a, b))

np.fill_diagonal(A_topics.values, np.nan)

"""
Create the graph by thresholding the similarity matrix
Note here: G_topics is an org-org graph, where the nodes are organizations and the edges are the
 similarity between their topic distributions.
"""
A_demeaned = A_topics - A_topics.stack().mean()
A_demeaned = A_demeaned.replace(np.nan, 0) # This is because the diagonal is nan
A_thresholded = A_demeaned.applymap(lambda x: max(x, 0))
topic_graph = nx.Graph(A_thresholded)

# Run Infomap
im = infomap.Infomap(markov_time=1, silent=True)
im.add_networkx_graph(topic_graph)
im.run()

topic_partition = im.get_dataframe(['name', 'module_id']).set_index('name').module_id
topic_partition = topic_partition[
    topic_partition.map(topic_partition.value_counts()) > 2]  # This just eliminates two single-node modules