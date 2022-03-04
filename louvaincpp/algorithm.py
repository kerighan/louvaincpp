import networkx as nx
import numpy as np
from _louvaincpp import generate_dendrogram, generate_full_dendrogram


def generate_partition(dendrogram, level):
    partition = range(len(dendrogram[-level]))
    for i in range(level, len(dendrogram) + 1):
        new_partition = dict()
        for j in range(len(dendrogram[-i])):
            new_partition[j] = partition[dendrogram[-i][j]]
        partition = new_partition
    return partition


def partition_to_vec(partition):
    y = np.zeros(len(partition), dtype=int)
    for i, val in partition.items():
        y[i] = val
    return y


def louvain(G, resolution=1, prune=False, **_):
    A = nx.adjacency_matrix(G)

    dendrogram = generate_dendrogram(
        A.indptr, A.indices, A.data, resolution, prune)

    partition = range(len(dendrogram[-1]))
    for i in range(1, len(dendrogram) + 1):
        new_partition = dict()
        for j in range(len(dendrogram[-i])):
            new_partition[j] = partition[dendrogram[-i][j]]
        partition = new_partition
    return partition


def metric_louvain(
    G, X=None, resolution=1, prune=False, **_
):
    from sklearn.metrics import silhouette_score as scoring

    # from sklearn.metrics import calinski_harabasz_score as scoring

    A = nx.adjacency_matrix(G)

    dendrogram = generate_full_dendrogram(
        A.indptr, A.indices, A.data, resolution, prune)

    best_score = -float("inf")
    best_y = None
    for level in range(1, len(dendrogram) + 1):
        partition = generate_partition(dendrogram, level)
        y = partition_to_vec(partition)
        try:
            score = scoring(X, y)
        except ValueError:
            score = -float("inf")
        if score >= best_score:
            best_y = y.copy()
            best_score = score
    return best_y
