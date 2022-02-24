import networkx as nx
import numpy as np

from louvaincpp import metric_louvain

G = nx.random_partition_graph(
    [500, 500, 500, 500, 258], 0.1, 0.001,
    seed=123)
pos = nx.spectral_layout(G)
X = np.array([pos[node] for node in G.nodes])

y = metric_louvain(G, X)
