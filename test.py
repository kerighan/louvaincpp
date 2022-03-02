import networkx as nx
import numpy as np
from louvaincpp import louvain

G = nx.karate_club_graph()
pos = nx.spectral_layout(G)
X = np.array([pos[node] for node in G.nodes])

y = louvain(G)
print(y)

