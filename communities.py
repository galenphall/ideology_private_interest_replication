import pandas as pd
import networkx as nx
import numpy as np
import infomap
from matplotlib import pyplot as plt
from matplotlib import colors

df_org = pd.read_csv('data/metadata.csv')

df_intr = pd.read_json('data/cccm_interlocks.json')
df_intr = df_intr[df_intr.Verified]

df_ofc = pd.read_csv('data/officers.csv')

eins = pd.read_csv('data/eins.csv')
ein_to_name = eins.set_index('EIN').ORG
name_to_ein = eins.set_index('ORG').EIN

interlocks = pd.DataFrame(
    [
        {
            'ein1': row.ein1,
            'ein2': row.ein2,
            'year1': y1,
            'year2': y2,
            'strength': np.exp(- abs(y1 - y2))
        }
        for idx, row in df_intr.iterrows()
        for y1 in row.group1Years
        for y2 in row.group2Years
        if not (((y1 == y2) & (row.ein1 == row.ein2)) | y1 < 2003 | y2 < 2003)
    ]
).groupby(['ein1', 'ein2', 'year1', 'year2']).strength.sum().reset_index(drop = False)

edgelist = interlocks.groupby(['ein1', 'ein2']).strength.sum().reset_index(drop = False)
edgelist.columns = ['source', 'target', 'weight']
edgelist['source'] = edgelist.source.map(ein_to_name)
edgelist['target'] = edgelist.target.map(ein_to_name)

G = nx.from_pandas_edgelist(edgelist, edge_attr = 'weight')
V = list(G.nodes)

im = infomap.Infomap('--markov-time 0.85 --tree data')
im.add_networkx_graph(G, weight='weight')
im.run()
communities = {V[k]:v for k,v in im.get_modules().items()}
nx.set_node_attributes(G, communities, 'community')

communities_list = [c - 1 for c in nx.get_node_attributes(G, 'community').values()]
num_communities = max(communities_list) + 1

# color map from http://colorbrewer2.org/
cmap = colors.ListedColormap(
    ['#d53e4f','#fc8d59','#fee08b','#e6f598','#99d594','#3288bd'], 'indexed', num_communities)

pos = nx.spring_layout(G, k = 2.5, iterations = 500, seed = 19012022)
ew = [e[2]['weight']**0.5 for e in G.edges(data = True)] 

fig, ax = plt.subplots(1,1, figsize = (6,6))

nodes = nx.draw_networkx_nodes(G, ax = ax, pos = pos, node_color = communities_list, 
                               with_labels = False, node_size = 100, cmap = cmap)
nodes.set_edgecolor('k')
edges = nx.draw_networkx_edges(G, ax = ax, pos = pos, width = ew, edge_color = (0,0,0,1))

fig.savefig('figures/cccm_communities.png', dpi = 300)