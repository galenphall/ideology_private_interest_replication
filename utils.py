from infomap import Infomap
import igraph as ig
import numpy as np
import networkx as nx

def process_edgelist(edgelist):
    
    edgelist = edgelist.copy()
    
    nodes = list(set(np.array(edgelist)[:,:2].ravel()))
    node_map = dict(zip( nodes, range(len(nodes)) ))
    inv_node_map = dict(zip( range(len(nodes)), nodes ))
    
    for e in edgelist:
        if len(e) == 2:
            np.append(e, 1)
            
    edgelist[:,0] = np.array([node_map[e] for e in edgelist[:,0]]).reshape(edgelist.shape[0])
    edgelist[:,1] = np.array([node_map[e] for e in edgelist[:,1]]).reshape(edgelist.shape[0])
            
    return edgelist, inv_node_map
    

def infomap_cluster(edgelist, options = "", ):
    
    im = Infomap(options)
    
    edgelist, invnodemap = process_edgelist(edgelist)
            
    im.add_links(edgelist)
    
    im.run()
                
    cms = {invnodemap[k]:v for k,v in im.get_modules().items()}
        
    return cms, im

def multilevel_cluster(edgelist, node_map = {}):
    
    edgelist, invnodemap = process_edgelist(edgelist)
    edgelist = edgelist[edgelist[:,2] > 0]
    
    G = ig.Graph([tuple(e) for e in edgelist[:,:2]])
    weights = edgelist[:,2]
    G.es['weight'] = 1/weights
    
    comms_list = G.community_multilevel(weights = 1/weights)
    cms = {}
    for i in range(len(comms_list)):
        cm = comms_list[i]
        for n in cm:
            cms[invnodemap[n]] = i
        
    return cms

def spinglass_cluster(edgelist, node_map = {}):
    
    edgelist, invnodemap = process_edgelist(edgelist)
    
    G = ig.Graph([tuple(e) for e in edgelist[:,:2]])
    weights = edgelist[:,2]
    
    comms_list = G.community_spinglass(weights = weights)
    cms = {}
    for i in range(len(comms_list)):
        cm = comms_list[i]
        for n in cm:
            cms[invnodemap[n]] = i
        
    return cms

def extract_signed_onemode(bipartite_array, sign):
    product = bipartite_array * bipartite_array[:, None]
    product[np.sign(product) != sign] = 0
    onemode_array = np.nansum(product, 2)
    return onemode_array    

def induced_graph(graph, clusters, weight = 'weight'):
    
    V = graph.nodes
    E = graph.edges(data = True)
    
    clustergraph = nx.Graph()
    clustergraph.add_nodes_from(clusters)
    
    for u, v, data in E:
        c1 = clusters[u]
        c2 = clusters[v]
        
        prev_w = clustergraph.get_edge_data(c1, c2, {weight: 0}).get(weight, 1)
        clustergraph.add_edge(c1, c2, **{weight : prev_w + data[weight]})
        
    return clustergraph