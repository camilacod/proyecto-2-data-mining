# Databricks notebook source
# MAGIC %md
# MAGIC # Parte II -- Analisis de grafos: ranking y comunidades
# MAGIC
# MAGIC Corre en la **misma carpeta del Workspace** que la Parte I: reutiliza sus modulos (`config.py`, `preprocessing.py`, `cleaning.py`, `graphs.py`) y los parquet de `artifacts/`.

# COMMAND ----------

# Setup: modulos y artifacts de la Parte I (misma carpeta del Workspace).
# load_mod lee los .py directo desde disco (evita el cache de imports del Workspace)
import sys, os, importlib.util

src = os.getcwd()
sys.path.insert(0, src)

def load_mod(name):
    path = os.path.join(src, name + '.py')
    assert os.path.exists(path), f'No existe {path}: corre primero la Parte I en esta carpeta'
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

config = load_mod('config')
preprocessing = load_mod('preprocessing')
cleaning = load_mod('cleaning')
graphs = load_mod('graphs')
print('Artefactos en:', config.ARTIFACTS)

# build_subset() es idempotente: usa el cache parquet si existe,
# y si no (p.ej. cluster nuevo) re-streamea los JSON (~3 min)
clean = cleaning.clean_subset(preprocessing.build_subset())
G = graphs.build_graphs(clean)

# COMMAND ----------

# MAGIC %%writefile ranking.py
# MAGIC import math
# MAGIC from collections import defaultdict
# MAGIC import pandas as pd
# MAGIC
# MAGIC
# MAGIC # ===================== PageRank (grafo de amistad) =====================
# MAGIC def pagerank(adj, nodes=None, d=0.85, max_iter=100, tol=1e-6):
# MAGIC     if nodes is None:
# MAGIC         nodes = list(adj)
# MAGIC     nodeset = set(nodes)
# MAGIC     N = len(nodes)
# MAGIC     nbrs = {u: [v for v in adj[u] if v in nodeset] for u in nodes}
# MAGIC     deg = {u: len(nbrs[u]) for u in nodes}
# MAGIC     pr = {u: 1.0 / N for u in nodes}
# MAGIC     base = (1 - d) / N
# MAGIC     for it in range(1, max_iter + 1):
# MAGIC         new = {u: base for u in nodes}
# MAGIC         dangling = 0.0
# MAGIC         for u in nodes:
# MAGIC             if deg[u] == 0:
# MAGIC                 dangling += pr[u]
# MAGIC             else:
# MAGIC                 share = d * pr[u] / deg[u]
# MAGIC                 for v in nbrs[u]:
# MAGIC                     new[v] += share
# MAGIC         if dangling:
# MAGIC             add = d * dangling / N
# MAGIC             for u in new:
# MAGIC                 new[u] += add
# MAGIC         diff = sum(abs(new[u] - pr[u]) for u in nodes)
# MAGIC         pr = new
# MAGIC         if diff < tol:
# MAGIC             break
# MAGIC     return pr, it
# MAGIC
# MAGIC
# MAGIC def top_pagerank(pr, users, k=15):
# MAGIC     s = pd.Series(pr, name='pagerank').sort_values(ascending=False).head(k)
# MAGIC     out = s.rename_axis('user_id').reset_index()
# MAGIC     cols = ['user_id', 'name', 'review_count', 'fans', 'n_friends_subset']
# MAGIC     info = users[[c for c in cols if c in users.columns]]
# MAGIC     return out.merge(info, on='user_id', how='left')
# MAGIC
# MAGIC
# MAGIC # ===================== HITS (bipartito usuario-negocio) =====================
# MAGIC def build_review_graph(reviews):
# MAGIC     user_to_biz = defaultdict(list)
# MAGIC     biz_to_users = defaultdict(list)
# MAGIC     for u, b in zip(reviews['user_id'], reviews['business_id']):
# MAGIC         user_to_biz[u].append(b)
# MAGIC         biz_to_users[b].append(u)
# MAGIC     return dict(user_to_biz), dict(biz_to_users)
# MAGIC
# MAGIC
# MAGIC def hits(user_to_biz, biz_to_users, max_iter=100, tol=1e-8):
# MAGIC     hub = {u: 1.0 for u in user_to_biz}
# MAGIC     auth = {b: 1.0 for b in biz_to_users}
# MAGIC     for it in range(1, max_iter + 1):
# MAGIC         new_auth = {b: sum(hub[u] for u in us) for b, us in biz_to_users.items()}
# MAGIC         na = math.sqrt(sum(v * v for v in new_auth.values())) or 1.0
# MAGIC         new_auth = {b: v / na for b, v in new_auth.items()}
# MAGIC         new_hub = {u: sum(new_auth[b] for b in bs) for u, bs in user_to_biz.items()}
# MAGIC         nh = math.sqrt(sum(v * v for v in new_hub.values())) or 1.0
# MAGIC         new_hub = {u: v / nh for u, v in new_hub.items()}
# MAGIC         diff = (sum(abs(new_auth[b] - auth[b]) for b in auth)
# MAGIC                 + sum(abs(new_hub[u] - hub[u]) for u in hub))
# MAGIC         auth, hub = new_auth, new_hub
# MAGIC         if diff < tol:
# MAGIC             break
# MAGIC     return hub, auth, it
# MAGIC
# MAGIC
# MAGIC def top_authorities(auth, business, k=15):
# MAGIC     s = pd.Series(auth, name='authority').sort_values(ascending=False).head(k)
# MAGIC     out = s.rename_axis('business_id').reset_index()
# MAGIC     cols = ['business_id', 'name', 'stars', 'review_count', 'categories']
# MAGIC     info = business[[c for c in cols if c in business.columns]]
# MAGIC     return out.merge(info, on='business_id', how='left')
# MAGIC
# MAGIC
# MAGIC def top_hubs(hub, users, k=15):
# MAGIC     s = pd.Series(hub, name='hub').sort_values(ascending=False).head(k)
# MAGIC     out = s.rename_axis('user_id').reset_index()
# MAGIC     cols = ['user_id', 'name', 'review_count', 'fans', 'n_friends_subset']
# MAGIC     info = users[[c for c in cols if c in users.columns]]
# MAGIC     return out.merge(info, on='user_id', how='left')

# COMMAND ----------

# load_mod lee los .py directo desde disco (evita el cache de imports del Workspace)
import sys, os, importlib.util

def load_mod(name):
    path = os.path.join(os.getcwd(), name + '.py')
    assert os.path.exists(path), f'No existe {path}: corre primero su celda %%writefile'
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

ranking = load_mod('ranking')

pr, iters = ranking.pagerank(G['friend_adj'], nodes=G['lcc'])
print(f'PageRank convergio en {iters} iteraciones sobre el LCC ({len(G["lcc"]):,} usuarios)')

top = ranking.top_pagerank(pr, clean['users'], k=15)
top

# COMMAND ----------

import matplotlib.pyplot as plt
import pandas as pd

# PageRank + atributos para TODOS los usuarios del LCC
prdf = (pd.Series(pr, name='pagerank').rename_axis('user_id').reset_index()
        .merge(clean['users'][['user_id', 'name', 'fans', 'n_friends_subset', 'review_count']],
               on='user_id', how='left'))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(prdf['n_friends_subset'], prdf['pagerank'], s=8, alpha=0.3, color='#378ADD')
axes[0].set_xlabel('Amigos en la muestra (n_friends_subset)'); axes[0].set_ylabel('PageRank')
axes[0].set_title('PageRank vs amistades  ->  centralidad')

axes[1].scatter(prdf['fans'], prdf['pagerank'], s=8, alpha=0.3, color='#D85A30')
axes[1].set_xlabel('Fans (popularidad)'); axes[1].set_ylabel('PageRank')
axes[1].set_title('PageRank vs fans  ->  popularidad')
plt.tight_layout(); plt.show()

print('corr PageRank ~ n_friends_subset:', round(prdf['pagerank'].corr(prdf['n_friends_subset']), 3))
print('corr PageRank ~ fans:           ', round(prdf['pagerank'].corr(prdf['fans']), 3))
print('corr PageRank ~ review_count:   ', round(prdf['pagerank'].corr(prdf['review_count']), 3))

# COMMAND ----------

# MAGIC %md
# MAGIC ### HITS sobre el grafo bipartito

# COMMAND ----------

# load_mod lee los .py directo desde disco (evita el cache de imports del Workspace)
import sys, os, importlib.util

def load_mod(name):
    path = os.path.join(os.getcwd(), name + '.py')
    assert os.path.exists(path), f'No existe {path}: corre primero su celda %%writefile'
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

ranking = load_mod('ranking')

import pandas as pd

u2b, b2u = ranking.build_review_graph(clean['reviews'])
hub, auth, it = ranking.hits(u2b, b2u)
print(f'HITS convergio en {it} iteraciones')
print(f'  {len(u2b):,} hubs (usuarios) | {len(b2u):,} authorities (negocios)\n')

print('TOP 15 AUTHORITIES (negocios mas importantes segun HITS):')
display(ranking.top_authorities(auth, clean['business'], k=15))

print('TOP 15 HUBS (resenadores mas importantes segun HITS):')
display(ranking.top_hubs(hub, clean['users'], k=15))

# authority vs simple popularidad
authdf = (pd.Series(auth, name='authority').rename_axis('business_id').reset_index()
.merge(clean['business'][['business_id', 'review_count']], on='business_id'))
print('corr authority ~ review_count:', round(authdf['authority'].corr(authdf['review_count']), 3))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Louvain -> comunidades

# COMMAND ----------

# MAGIC %%writefile communities.py
# MAGIC """
# MAGIC communities.py -- Parte II: deteccion de comunidades con Louvain.
# MAGIC Louvain a mano: movimiento local + agregacion por niveles. Sin librerias de grafos.
# MAGIC """
# MAGIC from collections import defaultdict, Counter
# MAGIC import pandas as pd
# MAGIC
# MAGIC
# MAGIC def _build_weighted(adjacency):
# MAGIC     return {i: (dict(nbrs) if isinstance(nbrs, dict) else {j: 1.0 for j in nbrs})
# MAGIC             for i, nbrs in adjacency.items()}
# MAGIC
# MAGIC
# MAGIC def induced_subgraph(adj, nodes):
# MAGIC     s = set(nodes)
# MAGIC     return {u: {v for v in adj[u] if v in s} for u in nodes}
# MAGIC
# MAGIC
# MAGIC def _one_level(adj_w, m, resolution):
# MAGIC     nodes = list(adj_w)
# MAGIC     com = {i: i for i in nodes}
# MAGIC     k = {i: sum(adj_w[i].values()) for i in nodes}
# MAGIC     tot = dict(k)
# MAGIC     improved, any_imp = True, False
# MAGIC     while improved:
# MAGIC         improved = False
# MAGIC         for i in nodes:
# MAGIC             ci, ki = com[i], k[i]
# MAGIC             neigh = defaultdict(float)
# MAGIC             for j, w in adj_w[i].items():
# MAGIC                 if j != i:
# MAGIC                     neigh[com[j]] += w
# MAGIC             tot[ci] -= ki
# MAGIC             best_c = ci
# MAGIC             best_gain = neigh.get(ci, 0.0) - resolution * tot[ci] * ki / (2 * m)
# MAGIC             for c, kin in neigh.items():
# MAGIC                 gain = kin - resolution * tot[c] * ki / (2 * m)
# MAGIC                 if gain > best_gain:
# MAGIC                     best_c, best_gain = c, gain
# MAGIC             tot[best_c] += ki
# MAGIC             com[i] = best_c
# MAGIC             if best_c != ci:
# MAGIC                 improved = any_imp = True
# MAGIC     return com, any_imp
# MAGIC
# MAGIC
# MAGIC def louvain(adjacency, resolution=1.0, max_levels=50):
# MAGIC     adj_w = _build_weighted(adjacency)
# MAGIC     m = sum(sum(d.values()) for d in adj_w.values()) / 2.0
# MAGIC     if m == 0:
# MAGIC         return {i: 0 for i in adj_w}
# MAGIC     partition = {i: i for i in adj_w}
# MAGIC     cur = adj_w
# MAGIC     for _ in range(max_levels):
# MAGIC         com, improved = _one_level(cur, m, resolution)
# MAGIC         relab = {c: idx for idx, c in enumerate(sorted(set(com.values()), key=str))}
# MAGIC         com = {n: relab[c] for n, c in com.items()}
# MAGIC         partition = {orig: com[sup] for orig, sup in partition.items()}
# MAGIC         if not improved:
# MAGIC             break
# MAGIC         new = defaultdict(lambda: defaultdict(float))
# MAGIC         for i in cur:
# MAGIC             ci = com[i]
# MAGIC             for j, w in cur[i].items():
# MAGIC                 new[ci][com[j]] += w
# MAGIC         cur = {c: dict(d) for c, d in new.items()}
# MAGIC     order = [c for c, _ in Counter(partition.values()).most_common()]
# MAGIC     remap = {c: idx for idx, c in enumerate(order)}
# MAGIC     return {n: remap[c] for n, c in partition.items()}
# MAGIC
# MAGIC
# MAGIC def modularity(adjacency, partition):
# MAGIC     adj_w = _build_weighted(adjacency)
# MAGIC     m = sum(sum(d.values()) for d in adj_w.values()) / 2.0
# MAGIC     if m == 0:
# MAGIC         return 0.0
# MAGIC     inc, tot = defaultdict(float), defaultdict(float)
# MAGIC     for i in adj_w:
# MAGIC         ci = partition[i]
# MAGIC         tot[ci] += sum(adj_w[i].values())
# MAGIC         for j, w in adj_w[i].items():
# MAGIC             if partition[j] == ci:
# MAGIC                 inc[ci] += w
# MAGIC     return sum(inc[c] / (2 * m) - (tot[c] / (2 * m)) ** 2 for c in tot)
# MAGIC
# MAGIC
# MAGIC def community_summary(adj, partition, pr=None, users=None, top_k=5, n_key=5):
# MAGIC     members = defaultdict(list)
# MAGIC     for n, c in partition.items():
# MAGIC         members[c].append(n)
# MAGIC     name_map = dict(zip(users['user_id'], users['name'])) if users is not None else {}
# MAGIC     rows = []
# MAGIC     for c, ms_list in sorted(members.items(), key=lambda kv: -len(kv[1]))[:top_k]:
# MAGIC         ms = set(ms_list)
# MAGIC         n = len(ms)
# MAGIC         ein = sum(1 for u in ms for v in adj[u] if v in ms) // 2
# MAGIC         eout = sum(1 for u in ms for v in adj[u] if v not in ms)
# MAGIC         density = 2 * ein / (n * (n - 1)) if n > 1 else 0
# MAGIC         if pr:
# MAGIC             key = sorted(ms, key=lambda x: pr.get(x, 0), reverse=True)[:n_key]
# MAGIC         else:
# MAGIC             key = sorted(ms, key=lambda x: len(adj[x]), reverse=True)[:n_key]
# MAGIC         rows.append({'comunidad': c, 'tamano': n, 'aristas_int': ein,
# MAGIC                      'densidad_int': round(density, 4), 'aristas_ext': eout,
# MAGIC                      'miembros_clave': [name_map.get(u, u) for u in key]})
# MAGIC     return pd.DataFrame(rows)

# COMMAND ----------

# load_mod lee los .py directo desde disco (evita el cache de imports del Workspace)
import sys, os, importlib.util

def load_mod(name):
    path = os.path.join(os.getcwd(), name + '.py')
    assert os.path.exists(path), f'No existe {path}: corre primero su celda %%writefile'
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

communities = load_mod('communities')

lcc_adj = communities.induced_subgraph(G['friend_adj'], G['lcc'])
partition = communities.louvain(lcc_adj)
q = communities.modularity(lcc_adj, partition)
print(f'Louvain: {len(set(partition.values())):,} comunidades | modularidad Q = {q:.4f}')

summary = communities.community_summary(lcc_adj, partition, pr=pr, users=clean['users'], top_k=5)
summary

# COMMAND ----------

# MAGIC %md
# MAGIC ### Dibujando el grafo

# COMMAND ----------

# MAGIC %%writefile graphviz.py
# MAGIC import math
# MAGIC import random
# MAGIC from collections import defaultdict
# MAGIC import matplotlib.pyplot as plt
# MAGIC import matplotlib.patches as mpatches
# MAGIC
# MAGIC
# MAGIC def sample_subgraph(adj, partition, n_comms=6, per_comm=50, seed=0):
# MAGIC     rng = random.Random(seed)
# MAGIC     members = defaultdict(list)
# MAGIC     for n, c in partition.items():
# MAGIC         members[c].append(n)
# MAGIC     top = sorted(members, key=lambda c: -len(members[c]))[:n_comms]
# MAGIC     sampled, node_comm = set(), {}
# MAGIC     for c in top:
# MAGIC         ms = members[c]
# MAGIC         pick = ms if len(ms) <= per_comm else rng.sample(ms, per_comm)
# MAGIC         for u in pick:
# MAGIC             sampled.add(u)
# MAGIC             node_comm[u] = c
# MAGIC     sub = {u: [v for v in adj[u] if v in sampled] for u in sampled}
# MAGIC     return sub, node_comm
# MAGIC
# MAGIC
# MAGIC def spring_layout(sub, iterations=100, seed=0):
# MAGIC     rng = random.Random(seed)
# MAGIC     nodes = list(sub)
# MAGIC     n = len(nodes)
# MAGIC     if n == 0:
# MAGIC         return {}
# MAGIC     pos = {u: [rng.uniform(0, 1), rng.uniform(0, 1)] for u in nodes}
# MAGIC     edges = set()
# MAGIC     for u in sub:
# MAGIC         for v in sub[u]:
# MAGIC             edges.add((u, v) if str(u) < str(v) else (v, u))
# MAGIC     k = math.sqrt(1.0 / n)
# MAGIC     t = 0.1
# MAGIC     dt = t / (iterations + 1)
# MAGIC     for _ in range(iterations):
# MAGIC         disp = {u: [0.0, 0.0] for u in nodes}
# MAGIC         for i in range(n):
# MAGIC             ui = nodes[i]; xi, yi = pos[ui]
# MAGIC             for j in range(i + 1, n):
# MAGIC                 uj = nodes[j]
# MAGIC                 dx = xi - pos[uj][0]; dy = yi - pos[uj][1]
# MAGIC                 d = math.hypot(dx, dy) or 1e-6
# MAGIC                 rep = k * k / d
# MAGIC                 ux, uy = dx / d * rep, dy / d * rep
# MAGIC                 disp[ui][0] += ux; disp[ui][1] += uy
# MAGIC                 disp[uj][0] -= ux; disp[uj][1] -= uy
# MAGIC         for u, v in edges:
# MAGIC             dx = pos[u][0] - pos[v][0]; dy = pos[u][1] - pos[v][1]
# MAGIC             d = math.hypot(dx, dy) or 1e-6
# MAGIC             att = d * d / k
# MAGIC             ax_, ay_ = dx / d * att, dy / d * att
# MAGIC             disp[u][0] -= ax_; disp[u][1] -= ay_
# MAGIC             disp[v][0] += ax_; disp[v][1] += ay_
# MAGIC         for u in nodes:
# MAGIC             dx, dy = disp[u]
# MAGIC             d = math.hypot(dx, dy) or 1e-6
# MAGIC             lim = min(d, t)
# MAGIC             pos[u][0] += dx / d * lim
# MAGIC             pos[u][1] += dy / d * lim
# MAGIC         t -= dt
# MAGIC     return {u: tuple(p) for u, p in pos.items()}
# MAGIC
# MAGIC
# MAGIC def plot_communities(sub, node_comm, pos, pr=None, title=None):
# MAGIC     fig, ax = plt.subplots(figsize=(9, 9))
# MAGIC     for u in sub:
# MAGIC         for v in sub[u]:
# MAGIC             if str(u) < str(v):
# MAGIC                 ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
# MAGIC                         color='#cccccc', lw=0.3, alpha=0.5, zorder=1)
# MAGIC     comms = sorted(set(node_comm.values()))
# MAGIC     cmap = plt.cm.tab10
# MAGIC     color = {c: cmap(i % 10) for i, c in enumerate(comms)}
# MAGIC     nodes = list(sub)
# MAGIC     if pr:
# MAGIC         mx = max((pr.get(u, 0) for u in nodes), default=0) or 1.0
# MAGIC         sizes = [15 + 220 * (pr.get(u, 0) / mx) for u in nodes]
# MAGIC     else:
# MAGIC         sizes = [25] * len(nodes)
# MAGIC     ax.scatter([pos[u][0] for u in nodes], [pos[u][1] for u in nodes],
# MAGIC                c=[color[node_comm[u]] for u in nodes], s=sizes,
# MAGIC                edgecolors='white', linewidths=0.3, zorder=2)
# MAGIC     handles = [mpatches.Patch(color=color[c], label=f'Comunidad {c}') for c in comms]
# MAGIC     ax.legend(handles=handles, loc='upper right', fontsize=8, framealpha=0.9)
# MAGIC     ax.set_title(title or 'Comunidades de amistad (muestra del LCC)')
# MAGIC     ax.axis('off')
# MAGIC     plt.tight_layout(); plt.show()

# COMMAND ----------

# load_mod lee los .py directo desde disco (evita el cache de imports del Workspace)
import sys, os, importlib.util

def load_mod(name):
    path = os.path.join(os.getcwd(), name + '.py')
    assert os.path.exists(path), f'No existe {path}: corre primero su celda %%writefile'
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

graphviz = load_mod('graphviz')

sub, node_comm = graphviz.sample_subgraph(lcc_adj, partition, n_comms=6, per_comm=50)
print(f'Muestra: {len(sub):,} nodos de las 6 comunidades mas grandes')

pos = graphviz.spring_layout(sub, iterations=120)
graphviz.plot_communities(sub, node_comm, pos, pr=pr,
title='Comunidades de amistad (muestra del LCC)')

# COMMAND ----------

# MAGIC %md
# MAGIC Las comunidades grandes son enormes (de 2,110 a 9,167 nodos las top-8) pero internamente dispersas (densidad interna entre 0.0013 y 0.0053).
# MAGIC
# MAGIC Cuando muestreo 50 nodos al azar de una comunidad de miles, la probabilidad de que dos de ellos estén conectados directamente es bajísima: quedan apenas un puñado de aristas por comunidad. O sea, la mayoría de los 300 nodos muestreados quedan aislados (sin aristas entre sí). Y en un layout force-directed, los nodos sin aristas solo sienten repulsión, así que se van todos al borde formando ese anillo. Los pocos cumulitos del centro son los que sí quedaron conectados.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Version mejorada

# COMMAND ----------

# MAGIC %%writefile graphviz.py
# MAGIC import math
# MAGIC import random
# MAGIC from collections import defaultdict, deque
# MAGIC import matplotlib.pyplot as plt
# MAGIC import matplotlib.patches as mpatches
# MAGIC
# MAGIC
# MAGIC def sample_subgraph(adj, partition, pr=None, n_comms=6, per_comm=50, seed=0):
# MAGIC     members = defaultdict(list)
# MAGIC     for n, c in partition.items():
# MAGIC         members[c].append(n)
# MAGIC     top = sorted(members, key=lambda c: -len(members[c]))[:n_comms]
# MAGIC     sampled, node_comm = set(), {}
# MAGIC     for c in top:
# MAGIC         ms = set(members[c])
# MAGIC         if pr:
# MAGIC             seed_node = max(ms, key=lambda x: pr.get(x, 0))
# MAGIC         else:
# MAGIC             seed_node = max(ms, key=lambda x: sum(1 for v in adj[x] if v in ms))
# MAGIC         seen, dq, order = {seed_node}, deque([seed_node]), []
# MAGIC         while dq and len(order) < per_comm:
# MAGIC             u = dq.popleft()
# MAGIC             order.append(u)
# MAGIC             for v in adj[u]:
# MAGIC                 if v in ms and v not in seen:
# MAGIC                     seen.add(v)
# MAGIC                     dq.append(v)
# MAGIC         for u in order:
# MAGIC             sampled.add(u)
# MAGIC             node_comm[u] = c
# MAGIC     sub = {u: [v for v in adj[u] if v in sampled] for u in sampled}
# MAGIC     return sub, node_comm
# MAGIC
# MAGIC
# MAGIC def spring_layout(sub, iterations=100, seed=0):
# MAGIC     rng = random.Random(seed)
# MAGIC     nodes = list(sub)
# MAGIC     n = len(nodes)
# MAGIC     if n == 0:
# MAGIC         return {}
# MAGIC     pos = {u: [rng.uniform(0, 1), rng.uniform(0, 1)] for u in nodes}
# MAGIC     edges = set()
# MAGIC     for u in sub:
# MAGIC         for v in sub[u]:
# MAGIC             edges.add((u, v) if str(u) < str(v) else (v, u))
# MAGIC     k = math.sqrt(1.0 / n)
# MAGIC     t = 0.1
# MAGIC     dt = t / (iterations + 1)
# MAGIC     for _ in range(iterations):
# MAGIC         disp = {u: [0.0, 0.0] for u in nodes}
# MAGIC         for i in range(n):
# MAGIC             ui = nodes[i]; xi, yi = pos[ui]
# MAGIC             for j in range(i + 1, n):
# MAGIC                 uj = nodes[j]
# MAGIC                 dx = xi - pos[uj][0]; dy = yi - pos[uj][1]
# MAGIC                 d = math.hypot(dx, dy) or 1e-6
# MAGIC                 rep = k * k / d
# MAGIC                 ux, uy = dx / d * rep, dy / d * rep
# MAGIC                 disp[ui][0] += ux; disp[ui][1] += uy
# MAGIC                 disp[uj][0] -= ux; disp[uj][1] -= uy
# MAGIC         for u, v in edges:
# MAGIC             dx = pos[u][0] - pos[v][0]; dy = pos[u][1] - pos[v][1]
# MAGIC             d = math.hypot(dx, dy) or 1e-6
# MAGIC             att = d * d / k
# MAGIC             ax_, ay_ = dx / d * att, dy / d * att
# MAGIC             disp[u][0] -= ax_; disp[u][1] -= ay_
# MAGIC             disp[v][0] += ax_; disp[v][1] += ay_
# MAGIC         for u in nodes:
# MAGIC             dx, dy = disp[u]
# MAGIC             d = math.hypot(dx, dy) or 1e-6
# MAGIC             lim = min(d, t)
# MAGIC             pos[u][0] += dx / d * lim
# MAGIC             pos[u][1] += dy / d * lim
# MAGIC         t -= dt
# MAGIC     return {u: tuple(p) for u, p in pos.items()}
# MAGIC
# MAGIC
# MAGIC def plot_communities(sub, node_comm, pos, pr=None, title=None):
# MAGIC     fig, ax = plt.subplots(figsize=(9, 9))
# MAGIC     for u in sub:
# MAGIC         for v in sub[u]:
# MAGIC             if str(u) < str(v):
# MAGIC                 ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
# MAGIC                         color='#cccccc', lw=0.3, alpha=0.5, zorder=1)
# MAGIC     comms = sorted(set(node_comm.values()))
# MAGIC     cmap = plt.cm.tab10
# MAGIC     color = {c: cmap(i % 10) for i, c in enumerate(comms)}
# MAGIC     nodes = list(sub)
# MAGIC     if pr:
# MAGIC         mx = max((pr.get(u, 0) for u in nodes), default=0) or 1.0
# MAGIC         sizes = [15 + 220 * (pr.get(u, 0) / mx) for u in nodes]
# MAGIC     else:
# MAGIC         sizes = [25] * len(nodes)
# MAGIC     ax.scatter([pos[u][0] for u in nodes], [pos[u][1] for u in nodes],
# MAGIC                c=[color[node_comm[u]] for u in nodes], s=sizes,
# MAGIC                edgecolors='white', linewidths=0.3, zorder=2)
# MAGIC     handles = [mpatches.Patch(color=color[c], label=f'Comunidad {c}') for c in comms]
# MAGIC     ax.legend(handles=handles, loc='upper right', fontsize=8, framealpha=0.9)
# MAGIC     ax.set_title(title or 'Comunidades de amistad (muestra del LCC)')
# MAGIC     ax.axis('off')
# MAGIC     plt.tight_layout(); plt.show()

# COMMAND ----------

# load_mod lee los .py directo desde disco (evita el cache de imports del Workspace)
import sys, os, importlib.util

def load_mod(name):
    path = os.path.join(os.getcwd(), name + '.py')
    assert os.path.exists(path), f'No existe {path}: corre primero su celda %%writefile'
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

graphviz = load_mod('graphviz')

sub, node_comm = graphviz.sample_subgraph(lcc_adj, partition, pr=pr, n_comms=6, per_comm=50)
print(f'Muestra: {len(sub):,} nodos | aristas: {sum(len(v) for v in sub.values())//2:,}')

pos = graphviz.spring_layout(sub, iterations=120)
graphviz.plot_communities(sub, node_comm, pos, pr=pr,
title='Comunidades de amistad (nucleo conexo)')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Comparador Lentes de influencia

# COMMAND ----------

import pandas as pd

infl = (pd.concat([pd.Series(pr, name='pagerank'),
                   pd.Series(hub, name='hub')], axis=1, join='inner')
        .rename_axis('user_id').reset_index()
        .merge(clean['users'][['user_id', 'name', 'review_count', 'fans', 'n_friends_subset']],
               on='user_id', how='left'))
infl['pr_rank']  = infl['pagerank'].rank(ascending=False, method='min').astype(int)
infl['hub_rank'] = infl['hub'].rank(ascending=False, method='min').astype(int)

print('corr  PageRank ~ hub:', round(infl['pagerank'].corr(infl['hub']), 3))
top_pr  = set(infl.nsmallest(15, 'pr_rank')['user_id'])
top_hub = set(infl.nsmallest(15, 'hub_rank')['user_id'])
overlap = top_pr & top_hub
print(f'Solapamiento top-15 (influencia social vs reseñador): {len(overlap)}/15')

pilares = (infl[infl.user_id.isin(overlap)].sort_values('pagerank', ascending=False)
           [['name', 'pr_rank', 'hub_rank', 'review_count', 'fans', 'n_friends_subset']])
print('\nPilares de la red (top en AMBOS lentes):')
display(pilares)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Lectura de resultados de la Parte II
# MAGIC
# MAGIC **PageRank mide capital social, no actividad.** Sobre el LCC de 41,832 usuarios convergio en 40 iteraciones y correlaciona casi perfecto con las amistades dentro de la muestra (corr = 0.98) pero mucho menos con fans (0.48) y con review_count (0.36): se puede ser muy influyente en la red social escribiendo relativamente poco.
# MAGIC
# MAGIC **HITS mide otra cosa.** Los top authorities son negocios gastronomicos con cientos o miles de resenas (District Donuts Sliders Brew y Bacchanal Fine Wine & Spirits a la cabeza), pero la correlacion authority ~ review_count es solo 0.43: HITS pondera *quien* te resena, no cuantos. Los top hubs son usuarios prolificos (Marielle con 2,272 resenas, Shannon con 2,317) que no necesariamente tienen muchos amigos.
# MAGIC
# MAGIC **Los dos lentes casi no se solapan.** corr PageRank ~ hub = 0.023 y el top-15 de ambos rankings comparte **una sola persona (Morgan**, #12 social y #7 resenadora): ser influyente en la red de amistades y ser un resenador de referencia son roles distintos en Yelp. Morgan es el raro "pilar" que cumple ambos.
# MAGIC
# MAGIC **Louvain** encontro 191 comunidades con modularidad Q = 0.654 (estructura comunitaria fuerte). Las 8 mayores concentran ~39k de los ~42k nodos del LCC (de 2,110 a 9,167 miembros) con densidades internas de 0.0013-0.0053: comunidades enormes y dispersas, tipico de redes sociales grandes, lo que explica el aspecto del grafo muestreado de arriba.