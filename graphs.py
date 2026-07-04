from collections import deque
import config


def build_bipartite(reviews):
    n_users = reviews['user_id'].nunique()
    n_biz = reviews['business_id'].nunique()
    n_edges = len(reviews)
    density = n_edges / (n_users * n_biz) if n_users and n_biz else 0
    return {'n_users': n_users, 'n_business': n_biz,
            'n_edges': n_edges, 'density': density}


def build_friendship(users):
    adj = {}
    for uid, fr in zip(users['user_id'], users['friends_subset']):
        adj.setdefault(uid, set())
        if isinstance(fr, str) and fr:
            for f in fr.split(','):
                if f:
                    adj[uid].add(f)
                    adj.setdefault(f, set()).add(uid)
    return adj


def _components(adj):
    seen, comps = set(), []
    for start in adj:
        if start in seen:
            continue
        comp, dq = [], deque([start])
        seen.add(start)
        while dq:
            u = dq.popleft()
            comp.append(u)
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    dq.append(v)
        comps.append(comp)
    return comps


def _bfs_farthest(adj, src):
    dist = {src: 0}
    dq = deque([src])
    far, fd = src, 0
    while dq:
        u = dq.popleft()
        for v in adj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                if dist[v] > fd:
                    far, fd = v, dist[v]
                dq.append(v)
    return far, fd


def approx_diameter(adj, lcc_set):
    sub = {u: (adj[u] & lcc_set) for u in lcc_set}
    a, _ = _bfs_farthest(sub, next(iter(lcc_set)))
    _, d = _bfs_farthest(sub, a)
    return d


def friendship_metrics(users):
    adj = build_friendship(users)
    n = len(adj)
    deg_sum = sum(len(v) for v in adj.values())
    edges = deg_sum // 2
    density = deg_sum / (n * (n - 1)) if n > 1 else 0
    comps = _components(adj)
    lcc = max(comps, key=len) if comps else []
    diam = approx_diameter(adj, set(lcc)) if lcc else 0
    metrics = {'n_nodes': n, 'n_edges': edges, 'density': density,
               'n_components': len(comps), 'lcc_size': len(lcc),
               'lcc_fraction': len(lcc) / n if n else 0,
               'lcc_diameter_approx': diam}
    return metrics, adj, lcc


def build_graphs(clean=None):
    if clean is None:
        import cleaning
        clean = cleaning.clean_subset()

    bp = build_bipartite(clean['reviews'])
    fm, adj, lcc = friendship_metrics(clean['users'])

    print(f'GRAFOS INICIALES -- {config.CITY_LABEL}\n')
    print('[Bipartito usuario-negocio]')
    print(f"  usuarios:          {bp['n_users']:,}")
    print(f"  negocios:          {bp['n_business']:,}")
    print(f"  aristas (reseñas): {bp['n_edges']:,}")
    print(f"  densidad:          {bp['density']:.2e}")
    print('\n[Amistad usuario-usuario]')
    print(f"  nodos:             {fm['n_nodes']:,}")
    print(f"  aristas:           {fm['n_edges']:,}")
    print(f"  densidad:          {fm['density']:.2e}")
    print(f"  componentes:       {fm['n_components']:,}")
    print(f"  LCC:               {fm['lcc_size']:,} ({fm['lcc_fraction']:.1%})")
    print(f"  diametro aprox LCC: {fm['lcc_diameter_approx']}")

    return {'bipartite': bp, 'friendship': fm, 'friend_adj': adj, 'lcc': lcc}
