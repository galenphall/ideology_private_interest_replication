# make sure we're in the right directory
import os

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

os.chdir('..')

# run the figurestyle.py script to set the style

# load combined_employees.csv
employees = pd.read_csv('data/processed/combined_employees.csv')
employees = employees[employees.year >= 2003]
employees = employees.dropna(axis=0, subset=['ein'])
# load the yearly interlocks
years = range(2003, 2019)
interlocks = {}
for year in years:
    interlocks[year] = pd.read_csv(f'data/processed/yearly_interlock/{year}.csv')

# get the organization types
metadata = pd.read_csv('data/processed/metadata.csv')
ein_to_type = dict(zip(metadata.ein, metadata.organization_type))

years = range(2003, 2019)

# separate the lower plot into square subplots along the x-axis
n_years = len(years)

fig = plt.figure(figsize=(10, 4))

# create a large plot
gridspec = plt.GridSpec(2, n_years, fig, height_ratios=[3, 1], hspace=0.1, wspace=0.1)

# merge the top row
top_ax = fig.add_subplot(gridspec[0, :])
middle_axes = []
for i in range(n_years):
    new_ax = fig.add_subplot(gridspec[1, i])
    # make it square
    new_ax.set_aspect('equal')
    middle_axes.append(new_ax)

# create a graph for each year
yearly_graphs = {}
for year, ax in zip(years, middle_axes):
    G = nx.from_pandas_edgelist(interlocks[year], source='ein1', target='ein2', edge_attr='weight')

    # remove self-loops
    G.remove_edges_from(nx.selfloop_edges(G))

    # remove singletons
    G.remove_nodes_from(list(nx.isolates(G)))

    orgtype_to_color = dict(zip(
        metadata.organization_type.unique(),
        ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    ))

    # add in the node attributes
    for ein in G.nodes:
        G.nodes[ein]['organization_type'] = ein_to_type[ein]
        G.nodes[ein]['size'] = employees[(employees.ein == ein) & (employees.year == year)].count().values[0]

    # set the distance between nodes
    for ein1, ein2 in G.edges:
        G.edges[ein1, ein2]['distance'] = 1 / G.edges[ein1, ein2]['weight']

    # draw the graph
    pos = nx.kamada_kawai_layout(G, weight='distance',  center=(0, 0), scale=1)

    # scale the positions down slightly
    pos = {ein: [pos[ein][0] * 0.9, pos[ein][1] * 0.9] for ein in pos}

    nodes = nx.draw_networkx_nodes(G, pos,
                           node_size=[G.nodes[ein]['size'] / 10 for ein in G.nodes],
                           node_color=[orgtype_to_color[G.nodes[ein]['organization_type']] for ein in G.nodes], ax=ax)
    nodes.set_edgecolor('black')
    nodes.set_linewidth(0.1)
    nx.draw_networkx_edges(G, pos, width=[G.edges[ein1, ein2]['weight'] / 10 for ein1, ein2 in G.edges], alpha=0.5,
                           ax=ax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    [ax.spines[s].set_visible(False) for s in ax.spines]
    yearly_graphs[year] = G

# create the upper plot: network density over time
clustering = []
density = []
for year in years:
    yearly_graphs[year].add_nodes_from(metadata.ein.unique())
    clustering.append(nx.average_clustering(yearly_graphs[year], weight='weight'))

top_ax.plot(years, clustering, color='black', marker='o', linestyle='dashed', linewidth=1, markersize=5)

# set the x-axis ticks
top_ax.set_xticks(years)

# set the y-axis label
top_ax.set_title('Average clustering coefficient, all nodes')
for spine in ['top', 'right']:
    top_ax.spines[spine].set_visible(False)

top_ax.set_ylim(0, 0.02)

fig.tight_layout()


# save the figure
plt.savefig('figures/yearly_interlocks_clustering.png', dpi=300, bbox_inches='tight')
plt.show()
