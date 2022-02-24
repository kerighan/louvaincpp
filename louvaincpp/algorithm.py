import time

import networkx as nx
import numpy as np
from _louvaincpp2 import modularity, one_level
from scipy.sparse import csr_matrix


def get_adj(G):
    connectivity = [[] for _ in range(len(G.nodes))]
    weights = [[] for _ in range(len(G.nodes))]
    for node in G.nodes:
        cnn = connectivity[node]
        w = weights[node]
        for neighbor in G.neighbors(node):
            cnn.append(neighbor)
            w.append(G.edges[node, neighbor].get("weight", 1.))
    return connectivity, weights


def metric_louvain(G, X, metric="silhouette", verbose=False):
    from sklearn.metrics import silhouette_score

    start_time = time.time()

    n_nodes = len(G.nodes)
    connectivity, weights = get_adj(G)
    C = {i: i for i in range(n_nodes)}
    y = np.arange(n_nodes)

    modified = True
    partition = [[i] for i in range(n_nodes)]

    best_score = -float("inf")
    best_y = y
    while modified:
        connectivity, weights, C, cm2nodes, modified = one_level(
            connectivity, weights, C)

        new_partition = [[] for _ in range(len(cm2nodes))]
        for i, nodes in enumerate(cm2nodes):
            for node in nodes:
                new_partition[i].extend(partition[node])
        partition = new_partition

        for i, nodes in enumerate(partition):
            for node in nodes:
                y[node] = i

        if len(partition) > 1:
            score = silhouette_score(X, y)
            if score > best_score:
                best_score = score
                best_y = y.copy()

    if verbose:
        elapsed = time.time() - start_time
        print(f"T={elapsed:.2f}s")
    return best_y


def louvain(G, verbose=False):
    from cylouvain import modularity as md
    from sklearn.metrics import silhouette_score

    start_time = time.time()

    n_nodes = len(G.nodes)
    connectivity, weights = get_adj(G)
    C = {i: i for i in range(n_nodes)}
    y = np.arange(n_nodes)

    conn = connectivity.copy()
    wg = weights.copy()

    modified = True
    partition = [[i] for i in range(n_nodes)]

    best_Q = -float("inf")
    while modified:
        connectivity, weights, C, cm2nodes, modified = one_level(
            connectivity, weights, C)

        new_partition = [[] for _ in range(len(cm2nodes))]
        for i, nodes in enumerate(cm2nodes):
            for node in nodes:
                new_partition[i].extend(partition[node])
        partition = new_partition

        test = {}
        y_old = y.copy()
        for i, nodes in enumerate(partition):
            for node in nodes:
                y[node] = i
                test[node] = i

        Q = modularity(test, conn, wg)
        if Q < best_Q:
            y = y_old
            break
        best_Q = Q

    if verbose:
        elapsed = time.time() - start_time
        print(f"T={elapsed:.2f}s")
    return y
