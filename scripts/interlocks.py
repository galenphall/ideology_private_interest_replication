from pathlib import Path
import os

currpath = Path(__file__)

# if we are not running from the "cccm-structure" directory, then we need to
# change the path to the data
if not currpath.name == 'cccm-structure':
    os.chdir(currpath.parent.parent)

from scripts.utils.utils import normalize_name
import networkx as nx
import ast
import pandas as pd
import infomap


def make_name_graph(employees, cccm_interlocks, fdn_cccm_interlocks):
    """
    Creates a graph of names to capture identity information. Then
    extracts connected components to create a map from name to identifier.
    """
    G = nx.Graph()

    G.add_nodes_from(employees.name.unique())
    for row in cccm_interlocks.itertuples():
        names = [*ast.literal_eval(row.group1Names), *ast.literal_eval(row.group2Names)]
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                G.add_edge(names[i], names[j])

    for row in fdn_cccm_interlocks.itertuples():
        G.add_edge(row.GRANTMAKER_PERSON, row.RECIPIENT_PERSON)

    for name_norm, grp in employees.groupby('name_norm'):
        names = grp.name.unique()
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                G.add_edge(names[i], names[j])

    for name in [*G.nodes]:
        G.add_edge(name, normalize_name(name))

    # Extract connected components
    components = list(nx.connected_components(G))

    # Create a map from name to identifier using the connected components
    name_to_id = {}
    for i, component in enumerate(components):
        for name in component:
            name_to_id[name] = i

    return name_to_id


def extract_interlocks(employees, cccm_interlocks, name_to_id, metadata):
    """
    Extracts interlocks between organizations from the interlock dataframes using the name_to_id map
    """
    # Create a set of identifiers associated with each organization
    org_to_ids = {org: set() for org in metadata.org_level_2.unique()}
    for row in cccm_interlocks.itertuples():
        org1 = row.group1Org
        if org1 not in org_to_ids:
            org1 = metadata[metadata.org_level_1 == org1].org_level_2.values[0]

        org2 = row.group2Org
        if org2 not in org_to_ids:
            org2 = metadata[metadata.org_level_1 == org2].org_level_2.values[0]

        for name in ast.literal_eval(row.group1Names):
            org_to_ids[org1].add(name_to_id[name])
        for name in ast.literal_eval(row.group2Names):
            org_to_ids[org2].add(name_to_id[name])
    for org, grp in employees.groupby('org'):
        if org not in org_to_ids:
            org = metadata[metadata.org_level_1 == org].org_level_2.values[0]
        for name in grp.name.unique():
            org_to_ids[org].add(name_to_id[name])
    # Find the number of overlapping identifiers for each pair of organizations
    org_pairs = []
    for org1, ids1 in org_to_ids.items():
        for org2, ids2 in org_to_ids.items():
            if org1 != org2:
                org_pairs.append([org1, org2, len(ids1.intersection(ids2))])
    org_pairs = pd.DataFrame(org_pairs, columns=['org1', 'org2', 'num_ids'])

    # Find the jaccard similarity between the sets of identifiers for each pair of organizations
    def jaccard(x):
        try:
            return x.num_ids / (len(org_to_ids[x.org1].union(org_to_ids[x.org2])))
        except ZeroDivisionError:
            return 0

    org_pairs['jaccard'] = org_pairs.apply(jaccard, 1)

    return org_pairs


fdn_cccm_interlocks = pd.read_csv('data/unprocessed/foundation_cccm_interlocks.csv')
cccm_interlocks = pd.read_excel('data/processed/validated_cccm_interlocks.xlsx')
cccm_interlocks = cccm_interlocks[cccm_interlocks.Verified & ~cccm_interlocks['Problem'] & ~cccm_interlocks[False]]
cccm_interlocks = cccm_interlocks[cccm_interlocks.apply(lambda row: 'MECHANICAL' not in str(row), 1)]
employees = pd.read_csv('data/processed/combined_employees.csv')
metadata = pd.read_csv('data/processed/metadata.csv')

eins = pd.read_csv('data/processed/eins.csv')
name_to_ein = eins.set_index('org').ein.to_dict()
fdn_namemap = pd.read_csv('data/processed/fdn_namemap.csv', index_col=0, squeeze=True).to_dict()
fdn_standard_name = {v: k for k, v in fdn_namemap.items()}

employees = employees[employees.name.notnull()]
employees['name_norm'] = employees.name.apply(normalize_name)
employees['name_norm'] = employees.name_norm.apply(
    lambda x: fdn_standard_name[fdn_namemap[x]] if x in fdn_namemap else x)

name_to_id = make_name_graph(employees, cccm_interlocks, fdn_cccm_interlocks)
org_pairs = extract_interlocks(employees, cccm_interlocks, name_to_id, metadata)

# Plot the network of organizations with edges weighted by jaccard similarity
interlock_graph = nx.Graph()
interlock_graph.add_nodes_from(org_pairs.org1.unique())
interlock_graph.add_nodes_from(org_pairs.org2.unique())
for row in org_pairs.itertuples():
    if row.num_ids > 0:
        interlock_graph.add_edge(row.org1, row.org2,
                                 weight=row.num_ids,
                                 distance=1 / row.num_ids,
                                 jaccard=row.jaccard,
                                 inverse_jaccard=1 / row.jaccard)

# Add distance as an edge attribute
for u, v in interlock_graph.edges:
    interlock_graph[u][v]['distance'] = 1 / interlock_graph[u][v]['weight']

org_types = metadata.set_index('org_level_2').organization_type.to_dict()
for node in interlock_graph.nodes:
    interlock_graph.nodes[node]['type'] = org_types[node]

# Find the optimal number of communities using the map equation
gc = interlock_graph.subgraph(max(nx.connected_components(interlock_graph), key=len))

infomapWrapper = infomap.Infomap(markov_time=2, silent=True)
infomapWrapper.add_networkx_graph(gc)
infomapWrapper.run()

# Extract the partition
partition = infomapWrapper.get_dataframe(['name', 'module_id']).set_index('name').module_id
partition = partition.reindex(interlock_graph.nodes)
partition[partition.isnull()] = -1
partition = partition.astype(int)

# Find number of shortest paths between pairs from different communities
# that pass through each node.

# First, find the shortest paths between all pairs of nodes in different communities
shortest_paths = []
for i in range(len(partition)):
    for j in range(i + 1, len(partition)):
        ii = list(partition.index)[i]
        jj = list(partition.index)[j]
        if partition[ii] != partition[jj]:
            try:
                shortest_paths.extend(nx.all_shortest_paths(interlock_graph, ii, jj))
            except nx.NetworkXNoPath:
                pass

# Then, find the number of shortest paths between pairs of nodes from different communities
# that pass through each node
paths_per_node = {node: 0 for node in interlock_graph.nodes}
for path in shortest_paths:
    start = path[0]
    end = path[-1]
    for node in path[1:-1]:
        paths_per_node[node] += 1

# Find the proportion of shortest paths between pairs of nodes from different communities
# that pass through each node
prop_paths_per_node = {node: paths_per_node[node] / len(shortest_paths) for node in interlock_graph.nodes}

# Social distance dataframe
social_distance = dict(nx.shortest_path_length(interlock_graph, weight='distance'))

# Convert the dictionary to a pd.DataFrame matrix
social_distance = pd.DataFrame(social_distance).reindex(index=interlock_graph.nodes, columns=interlock_graph.nodes)
