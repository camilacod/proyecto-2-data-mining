"""
communities.py -- Parte II: deteccion de comunidades con Louvain.
Louvain a mano: movimiento local + agregacion por niveles. Sin librerias de grafos.
"""
from collections import defaultdict, Counter
import pandas as pd


def _build_weighted(adjacency):
    return {i: (dict(nbrs) if isinstance(nbrs, dict) else {j: 1.0 for j in nbrs})
            for i, nbrs in adjacency.items()}


def induced_subgraph(adj, nodes):
    s = set(nodes)
    return {u: {v for v in adj[u] if v in s} for u in nodes}


def _one_level(adj_w, m, resolution):
    nodes = list(adj_w)
    com = {i: i for i in nodes}
    k = {i: sum(adj_w[i].values()) for i in nodes}
    tot = dict(k)
    improved, any_imp = True, False
    while improved:
        improved = False
        for i in nodes:
            ci, ki = com[i], k[i]
            neigh = defaultdict(float)
            for j, w in adj_w[i].items():
                if j != i:
                    neigh[com[j]] += w
            tot[ci] -= ki
            best_c = ci
            best_gain = neigh.get(ci, 0.0) - resolution * tot[ci] * ki / (2 * m)
            for c, kin in neigh.items():
                gain = kin - resolution * tot[c] * ki / (2 * m)
                if gain > best_gain:
                    best_c, best_gain = c, gain
            tot[best_c] += ki
            com[i] = best_c
            if best_c != ci:
                improved = any_imp = True
    return com, any_imp


def louvain(adjacency, resolution=1.0, max_levels=50):
    adj_w = _build_weighted(adjacency)
    m = sum(sum(d.values()) for d in adj_w.values()) / 2.0
    if m == 0:
        return {i: 0 for i in adj_w}
    partition = {i: i for i in adj_w}
    cur = adj_w
    for _ in range(max_levels):
        com, improved = _one_level(cur, m, resolution)
        relab = {c: idx for idx, c in enumerate(sorted(set(com.values()), key=str))}
        com = {n: relab[c] for n, c in com.items()}
        partition = {orig: com[sup] for orig, sup in partition.items()}
        if not improved:
            break
        new = defaultdict(lambda: defaultdict(float))
        for i in cur:
            ci = com[i]
            for j, w in cur[i].items():
                new[ci][com[j]] += w
        cur = {c: dict(d) for c, d in new.items()}
    order = [c for c, _ in Counter(partition.values()).most_common()]
    remap = {c: idx for idx, c in enumerate(order)}
    return {n: remap[c] for n, c in partition.items()}


def modularity(adjacency, partition):
    adj_w = _build_weighted(adjacency)
    m = sum(sum(d.values()) for d in adj_w.values()) / 2.0
    if m == 0:
        return 0.0
    inc, tot = defaultdict(float), defaultdict(float)
    for i in adj_w:
        ci = partition[i]
        tot[ci] += sum(adj_w[i].values())
        for j, w in adj_w[i].items():
            if partition[j] == ci:
                inc[ci] += w
    return sum(inc[c] / (2 * m) - (tot[c] / (2 * m)) ** 2 for c in tot)


def community_summary(adj, partition, pr=None, users=None, top_k=5, n_key=5):
    members = defaultdict(list)
    for n, c in partition.items():
        members[c].append(n)
    name_map = dict(zip(users['user_id'], users['name'])) if users is not None else {}
    rows = []
    for c, ms_list in sorted(members.items(), key=lambda kv: -len(kv[1]))[:top_k]:
        ms = set(ms_list)
        n = len(ms)
        ein = sum(1 for u in ms for v in adj[u] if v in ms) // 2
        eout = sum(1 for u in ms for v in adj[u] if v not in ms)
        density = 2 * ein / (n * (n - 1)) if n > 1 else 0
        if pr:
            key = sorted(ms, key=lambda x: pr.get(x, 0), reverse=True)[:n_key]
        else:
            key = sorted(ms, key=lambda x: len(adj[x]), reverse=True)[:n_key]
        rows.append({'comunidad': c, 'tamano': n, 'aristas_int': ein,
                     'densidad_int': round(density, 4), 'aristas_ext': eout,
                     'miembros_clave': [name_map.get(u, u) for u in key]})
    return pd.DataFrame(rows)
