import math
from collections import defaultdict
import pandas as pd


# ===================== PageRank (grafo de amistad) =====================
def pagerank(adj, nodes=None, d=0.85, max_iter=100, tol=1e-6):
    if nodes is None:
        nodes = list(adj)
    nodeset = set(nodes)
    N = len(nodes)
    nbrs = {u: [v for v in adj[u] if v in nodeset] for u in nodes}
    deg = {u: len(nbrs[u]) for u in nodes}
    pr = {u: 1.0 / N for u in nodes}
    base = (1 - d) / N
    for it in range(1, max_iter + 1):
        new = {u: base for u in nodes}
        dangling = 0.0
        for u in nodes:
            if deg[u] == 0:
                dangling += pr[u]
            else:
                share = d * pr[u] / deg[u]
                for v in nbrs[u]:
                    new[v] += share
        if dangling:
            add = d * dangling / N
            for u in new:
                new[u] += add
        diff = sum(abs(new[u] - pr[u]) for u in nodes)
        pr = new
        if diff < tol:
            break
    return pr, it


def top_pagerank(pr, users, k=15):
    s = pd.Series(pr, name='pagerank').sort_values(ascending=False).head(k)
    out = s.rename_axis('user_id').reset_index()
    cols = ['user_id', 'name', 'review_count', 'fans', 'n_friends_subset']
    info = users[[c for c in cols if c in users.columns]]
    return out.merge(info, on='user_id', how='left')


# ===================== HITS (bipartito usuario-negocio) =====================
def build_review_graph(reviews):
    user_to_biz = defaultdict(list)
    biz_to_users = defaultdict(list)
    for u, b in zip(reviews['user_id'], reviews['business_id']):
        user_to_biz[u].append(b)
        biz_to_users[b].append(u)
    return dict(user_to_biz), dict(biz_to_users)


def hits(user_to_biz, biz_to_users, max_iter=100, tol=1e-8):
    hub = {u: 1.0 for u in user_to_biz}
    auth = {b: 1.0 for b in biz_to_users}
    for it in range(1, max_iter + 1):
        new_auth = {b: sum(hub[u] for u in us) for b, us in biz_to_users.items()}
        na = math.sqrt(sum(v * v for v in new_auth.values())) or 1.0
        new_auth = {b: v / na for b, v in new_auth.items()}
        new_hub = {u: sum(new_auth[b] for b in bs) for u, bs in user_to_biz.items()}
        nh = math.sqrt(sum(v * v for v in new_hub.values())) or 1.0
        new_hub = {u: v / nh for u, v in new_hub.items()}
        diff = (sum(abs(new_auth[b] - auth[b]) for b in auth)
                + sum(abs(new_hub[u] - hub[u]) for u in hub))
        auth, hub = new_auth, new_hub
        if diff < tol:
            break
    return hub, auth, it


def top_authorities(auth, business, k=15):
    s = pd.Series(auth, name='authority').sort_values(ascending=False).head(k)
    out = s.rename_axis('business_id').reset_index()
    cols = ['business_id', 'name', 'stars', 'review_count', 'categories']
    info = business[[c for c in cols if c in business.columns]]
    return out.merge(info, on='business_id', how='left')


def top_hubs(hub, users, k=15):
    s = pd.Series(hub, name='hub').sort_values(ascending=False).head(k)
    out = s.rename_axis('user_id').reset_index()
    cols = ['user_id', 'name', 'review_count', 'fans', 'n_friends_subset']
    info = users[[c for c in cols if c in users.columns]]
    return out.merge(info, on='user_id', how='left')
