"""Graph construction over dependency parses + featurisation helpers."""
from __future__ import annotations

import random
from collections import deque

import numpy as np

GRAPH_TYPES = ["dep", "seq", "full", "random", "dep2"]


def dep_edges(heads):
    return [(i, h) for i, h in enumerate(heads) if h != i]


def random_tree_edges(n, seed):
    """Random tree on n nodes (same edge count as a parse, no syntax)."""
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    return [(order[k], order[rng.randrange(k)]) for k in range(1, n)]


def adjacency(heads, graph="dep", seed=0, self_loop=True):
    n = len(heads)
    A = np.zeros((n, n), dtype=np.float32)
    if graph == "dep" or graph == "dep2":
        for i, j in dep_edges(heads):
            A[i, j] = A[j, i] = 1
        if graph == "dep2":  # add 2-hop neighbours
            A2 = (A @ A > 0).astype(np.float32)
            A = np.maximum(A, A2)
    elif graph == "seq":
        for i in range(n - 1):
            A[i, i + 1] = A[i + 1, i] = 1
    elif graph == "full":
        A[:] = 1
    elif graph == "random":
        for i, j in random_tree_edges(n, seed):
            A[i, j] = A[j, i] = 1
    else:
        raise ValueError(graph)
    if self_loop:
        np.fill_diagonal(A, 1)
    return A


def tree_distances(heads, sources):
    """BFS distance (in dependency edges) from a set of source tokens."""
    n = len(heads)
    nbrs = [[] for _ in range(n)]
    for i, j in dep_edges(heads):
        nbrs[i].append(j)
        nbrs[j].append(i)
    dist = [10 ** 6] * n
    q = deque()
    for s in sources:
        dist[s] = 0
        q.append(s)
    while q:
        u = q.popleft()
        for v in nbrs[u]:
            if dist[v] > dist[u] + 1:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def position_weights(n, fr, to):
    """ASGCN position weights; 0 on the aspect itself."""
    w = np.zeros(n, dtype=np.float32)
    ctx = n - (to - fr)
    for i in range(n):
        if i < fr:
            w[i] = 1 - (fr - i) / max(ctx, 1)
        elif i >= to:
            w[i] = 1 - (i - to + 1) / max(ctx, 1)
    return w
