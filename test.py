import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from cylouvain import best_partition, modularity

from louvaincpp import louvain, metric_louvain

G = nx.random_partition_graph(
    [500, 500, 500, 500, 258], 0.1, 0.001,
    seed=123)
pos = nx.spectral_layout(G)
X = np.array([pos[node] for node in G.nodes])

y = metric_louvain(G, X)

partition = {i: y[i] for i in range(len(y))}
print(modularity(partition, G))
