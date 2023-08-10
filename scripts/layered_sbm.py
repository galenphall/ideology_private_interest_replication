import pickle

import graph_tool.all as gt
import matplotlib
import pandas as pd
import os
import numpy as np


if not os.getcwd().endswith('cccm-structure'):
    os.chdir('..')

merged_employees = pd.read_csv('data/processed/merged_employees.csv')
adjacency_matrix = np.sign(merged_employees.pivot_table(index='person_id', columns='ein', values="year").fillna(0))

# Get the weighted one-mode projection for the ein-ein network
ein_ein = (adjacency_matrix.T @ adjacency_matrix).values

# Create a graph-tool graph from the ein-ein matrix
g = gt.Graph(directed=False)
g.add_vertex(ein_ein.shape[0])
g.add_edge_list(np.argwhere(ein_ein))

# Add edge weights
g.ep.weight = g.new_edge_property('float')
for edge in g.edges():
    u = int(edge.source())
    v = int(edge.target())
    g.ep.weight[edge] = ein_ein[u, v]

# Keep only the largest component
g.set_vertex_filter(gt.label_largest_component(g, directed=False))

# Get the position
g.vp.pos = gt.sfdp_layout(g)

state = gt.minimize_nested_blockmodel_dl(g, state_args=dict(recs=[g.ep.weight],
                                                            rec_types=["discrete-poisson"]))

# Draw the graph
state.draw(pos=g.vp.pos, output="sbm.svg")

# # We will first equilibrate the Markov chain
# gt.mcmc_equilibrate(state, wait=1000, mcmc_args=dict(niter=10))
#
# bs = [] # collect some partitions
#
# def collect_partitions(s):
#    global bs
#    bs.append(s.b.a.copy())
#
# # Now we collect partitions for exactly 100,000 sweeps, at intervals
# # of 10 sweeps:
# gt.mcmc_equilibrate(state, force_niter=10000, mcmc_args=dict(niter=10),
#                     callback=collect_partitions)
#
# # Disambiguate partitions and obtain marginals
# pmode = gt.PartitionModeState(bs, converge=True)
# pv = pmode.get_marginal(g)
#
# # Now the node marginals are stored in property map pv. We can
# # visualize them as pie charts on the nodes:
# state.draw(pos=g.vp.pos, vertex_shape="pie", vertex_pie_fractions=pv,
#            output="sbm-marginals.svg")
#
# # Save the state
# with open('data/processed/sbm_state.pkl', 'wb') as f:
#     pickle.dump(state, f)
#
# # Save the partition mode
# with open('data/processed/sbm_partition_mode.pkl', 'wb') as f:
#     pickle.dump(pmode, f)





