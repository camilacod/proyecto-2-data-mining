# Databricks notebook source
# MAGIC %md
# MAGIC # Parte III -- Clustering de negocios (K-Means++ y DBSCAN)
# MAGIC
# MAGIC Corre en la **misma carpeta del Workspace** que la Parte I: reutiliza sus modulos y los parquet de `artifacts/`.

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
print('Artefactos en:', config.ARTIFACTS)

# build_subset() es idempotente: usa el cache parquet si existe,
# y si no (p.ej. cluster nuevo) re-streamea los JSON (~3 min)
clean = cleaning.clean_subset(preprocessing.build_subset())

# COMMAND ----------

# MAGIC %md
# MAGIC # Matriz de features

# COMMAND ----------

# MAGIC %md
# MAGIC Clusterizamos negocios. Cada negocio se representa con dos bloques. Primero, 10 features continuas/comportamentales: stars (rating), log_reviews y log_checkins (uso logaritmo porque son power-law osea sin log, un negocio con 2000 reseñas aplastaría a todo lo demás), price_range (imputado con la mediana, porque su 43% de faltantes es estructural, no error), n_categories, n_days_open, is_open, rating_std (polarización), business_age y avg_rev_len. Segundo, one-hot de las 18 categorías más comunes, que describen lo que el negocio es (restaurante vs taller vs spa).
# MAGIC
# MAGIC Estandarizo todo a z-score porque K-Means mide distancias y necesita escalas comparables, y excluyo lat/long a propósito, para que los clusters salgan por tipo de negocio y no por barrio (podemos añadir geografía después si queremos). Y recuerda: esta misma matriz es la que el PCA y SVD de la Parte VI van a reutilizar.

# COMMAND ----------

# MAGIC %%writefile features.py
# MAGIC import numpy as np
# MAGIC import pandas as pd
# MAGIC from collections import Counter
# MAGIC
# MAGIC
# MAGIC def _behavioral(reviews):
# MAGIC     ref = reviews['date'].max()
# MAGIC     agg = reviews.groupby('business_id').agg(
# MAGIC         rating_std=('stars', 'std'),
# MAGIC         first_review=('date', 'min'),
# MAGIC         avg_review_length=('text_len', 'mean'),
# MAGIC     ).reset_index()
# MAGIC     agg['rating_std'] = agg['rating_std'].fillna(0.0)
# MAGIC     agg['business_age_years'] = (ref - agg['first_review']).dt.days / 365.25
# MAGIC     return agg[['business_id', 'rating_std', 'business_age_years', 'avg_review_length']]
# MAGIC
# MAGIC
# MAGIC def build_business_features(clean, top_categories=18, cat_weight=1.0):
# MAGIC     biz = clean['business'].copy().reset_index(drop=True)
# MAGIC
# MAGIC     chk = clean['checkins']
# MAGIC     if 'n_checkins' in chk.columns:
# MAGIC         biz = biz.merge(chk[['business_id', 'n_checkins']], on='business_id', how='left')
# MAGIC     if 'n_checkins' not in biz.columns:
# MAGIC         biz['n_checkins'] = 0
# MAGIC     biz['n_checkins'] = biz['n_checkins'].fillna(0)
# MAGIC
# MAGIC     biz = biz.merge(_behavioral(clean['reviews']), on='business_id', how='left')
# MAGIC     for col in ('rating_std', 'business_age_years', 'avg_review_length'):
# MAGIC         biz[col] = biz[col].fillna(0.0)
# MAGIC
# MAGIC     pr_med = biz['price_range'].median()
# MAGIC     biz['price_range_imp'] = biz['price_range'].fillna(pr_med)
# MAGIC
# MAGIC     cont = pd.DataFrame({
# MAGIC         'stars':         biz['stars'].astype(float),
# MAGIC         'log_reviews':   np.log1p(biz['review_count'].astype(float)),
# MAGIC         'price_range':   biz['price_range_imp'].astype(float),
# MAGIC         'n_categories':  biz['n_categories'].astype(float),
# MAGIC         'n_days_open':   biz['n_days_open'].astype(float),
# MAGIC         'log_checkins':  np.log1p(biz['n_checkins'].astype(float)),
# MAGIC         'is_open':       biz['is_open'].astype(float),
# MAGIC         'rating_std':    biz['rating_std'].astype(float),
# MAGIC         'business_age':  biz['business_age_years'].astype(float),
# MAGIC         'avg_rev_len':   biz['avg_review_length'].astype(float),
# MAGIC     })
# MAGIC
# MAGIC     cat_counts = Counter()
# MAGIC     for lst in biz['categories_list']:
# MAGIC         cat_counts.update(lst)
# MAGIC     top = [c for c, _ in cat_counts.most_common(top_categories)]
# MAGIC     cat_mat = pd.DataFrame(
# MAGIC         {f'cat::{c}': biz['categories_list'].apply(lambda l, c=c: 1.0 if c in l else 0.0)
# MAGIC          for c in top}
# MAGIC     )
# MAGIC
# MAGIC     feat = pd.concat([cont, cat_mat], axis=1)
# MAGIC     names = list(feat.columns)
# MAGIC
# MAGIC     X = feat.to_numpy(dtype=float)
# MAGIC     mu = X.mean(axis=0)
# MAGIC     sd = X.std(axis=0)
# MAGIC     sd[sd == 0] = 1.0
# MAGIC     Xz = (X - mu) / sd
# MAGIC
# MAGIC     if cat_weight != 1.0:
# MAGIC         cat_cols = [i for i, n in enumerate(names) if n.startswith('cat::')]
# MAGIC         Xz[:, cat_cols] *= cat_weight
# MAGIC
# MAGIC     return Xz, names, biz['business_id'].to_numpy(), biz

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

features = load_mod('features')

X, feat_names, biz_ids, biz_df = features.build_business_features(clean, top_categories=18)
print(f'Matriz de features: {X.shape[0]} negocios x {X.shape[1]} features\n')
for n in feat_names:
    print('  ', n)

# COMMAND ----------

# MAGIC %md
# MAGIC # El camino de K-means++

# COMMAND ----------

# MAGIC %%writefile kmeans.py
# MAGIC import numpy as np
# MAGIC import pandas as pd
# MAGIC import matplotlib.pyplot as plt
# MAGIC from collections import Counter
# MAGIC
# MAGIC
# MAGIC def _dists_sq(X, C):
# MAGIC     xx = (X * X).sum(1)[:, None]
# MAGIC     cc = (C * C).sum(1)[None, :]
# MAGIC     return np.maximum(xx + cc - 2 * X @ C.T, 0)
# MAGIC
# MAGIC
# MAGIC def _init_pp(X, k, rng):
# MAGIC     n = X.shape[0]
# MAGIC     first = int(rng.integers(n))
# MAGIC     centroids = [X[first]]
# MAGIC     d2 = ((X - X[first]) ** 2).sum(1)
# MAGIC     for _ in range(1, k):
# MAGIC         s = d2.sum()
# MAGIC         probs = d2 / s if s > 0 else np.full(n, 1.0 / n)
# MAGIC         idx = int(rng.choice(n, p=probs))
# MAGIC         centroids.append(X[idx])
# MAGIC         d2 = np.minimum(d2, ((X - X[idx]) ** 2).sum(1))
# MAGIC     return np.array(centroids)
# MAGIC
# MAGIC
# MAGIC def kmeans_pp(X, k, max_iter=100, tol=1e-4, seed=0, n_init=5):
# MAGIC     rng = np.random.default_rng(seed)
# MAGIC     best = None
# MAGIC     for _ in range(n_init):
# MAGIC         C = _init_pp(X, k, rng)
# MAGIC         for _ in range(max_iter):
# MAGIC             labels = _dists_sq(X, C).argmin(1)
# MAGIC             newC = np.array([X[labels == j].mean(0) if np.any(labels == j) else C[j]
# MAGIC                              for j in range(k)])
# MAGIC             if np.sqrt(((newC - C) ** 2).sum()) < tol:
# MAGIC                 C = newC
# MAGIC                 break
# MAGIC             C = newC
# MAGIC         labels = _dists_sq(X, C).argmin(1)
# MAGIC         inertia = _dists_sq(X, C)[np.arange(len(X)), labels].sum()
# MAGIC         if best is None or inertia < best[2]:
# MAGIC             best = (labels, C, float(inertia))
# MAGIC     return best
# MAGIC
# MAGIC
# MAGIC def _pairwise_D(X):
# MAGIC     xx = (X * X).sum(1)
# MAGIC     return np.sqrt(np.maximum(xx[:, None] + xx[None, :] - 2 * X @ X.T, 0))
# MAGIC
# MAGIC
# MAGIC def silhouette_score(X, labels, D=None):
# MAGIC     if D is None:
# MAGIC         D = _pairwise_D(X)
# MAGIC     labels = np.asarray(labels)
# MAGIC     uniq = np.unique(labels)
# MAGIC     if len(uniq) < 2:
# MAGIC         return 0.0
# MAGIC     sil = np.zeros(len(labels))
# MAGIC     for c in uniq:
# MAGIC         idx = np.where(labels == c)[0]
# MAGIC         if len(idx) <= 1:
# MAGIC             continue
# MAGIC         others = [np.where(labels == o)[0] for o in uniq if o != c]
# MAGIC         for i in idx:
# MAGIC             a = D[i, idx[idx != i]].mean()
# MAGIC             b = min(D[i, o].mean() for o in others)
# MAGIC             sil[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
# MAGIC     return float(sil.mean())
# MAGIC
# MAGIC
# MAGIC def choose_k(X, k_min=2, k_max=10, seed=0):
# MAGIC     D = _pairwise_D(X)
# MAGIC     ks, inertias, sils, results = [], [], [], {}
# MAGIC     for k in range(k_min, k_max + 1):
# MAGIC         labels, C, inertia = kmeans_pp(X, k, seed=seed)
# MAGIC         s = silhouette_score(X, labels, D=D)
# MAGIC         ks.append(k); inertias.append(inertia); sils.append(s)
# MAGIC         results[k] = (labels, C, inertia, s)
# MAGIC         print(f'  k={k}: inercia={inertia:10.1f}  silueta={s:.4f}')
# MAGIC     return ks, inertias, sils, results
# MAGIC
# MAGIC
# MAGIC def plot_k_selection(ks, inertias, sils):
# MAGIC     fig, ax = plt.subplots(1, 2, figsize=(12, 4))
# MAGIC     ax[0].plot(ks, inertias, 'o-', color='#378ADD')
# MAGIC     ax[0].set_xlabel('k'); ax[0].set_ylabel('Inercia (SSE intra-cluster)')
# MAGIC     ax[0].set_title('Metodo del codo')
# MAGIC     ax[1].plot(ks, sils, 'o-', color='#D85A30')
# MAGIC     ax[1].set_xlabel('k'); ax[1].set_ylabel('Coeficiente de silueta')
# MAGIC     best = ks[int(np.argmax(sils))]
# MAGIC     ax[1].axvline(best, ls='--', color='gray', alpha=0.6)
# MAGIC     ax[1].set_title(f'Silueta por k (maxima en k={best})')
# MAGIC     plt.tight_layout(); plt.show()
# MAGIC
# MAGIC
# MAGIC def characterize_clusters(labels, biz_df):
# MAGIC     df = biz_df.copy(); df['cluster'] = labels
# MAGIC     rows = []
# MAGIC     for c in sorted(set(labels)):
# MAGIC         sub = df[df['cluster'] == c]
# MAGIC         cats = Counter()
# MAGIC         for lst in sub['categories_list']:
# MAGIC             cats.update(lst)
# MAGIC         rows.append({
# MAGIC             'cluster': c, 'n': len(sub),
# MAGIC             'stars': round(sub['stars'].mean(), 2),
# MAGIC             'reviews_med': int(sub['review_count'].median()),
# MAGIC             'price': round(sub['price_range_imp'].mean(), 2),
# MAGIC             'rating_std': round(sub['rating_std'].mean(), 2),
# MAGIC             'age_anios': round(sub['business_age_years'].mean(), 1),
# MAGIC             'pct_abierto': round(sub['is_open'].mean(), 2),
# MAGIC             'top_categorias': ', '.join(c for c, _ in cats.most_common(4)),
# MAGIC         })
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

features = load_mod('features')
kmeans = load_mod('kmeans')

# clustering 100% comportamental (categorias solo describiran despues)
X, feat_names, biz_ids, biz_df = features.build_business_features(
clean, top_categories=18, cat_weight=0.0)
print(f'Matriz (comportamental): {X.shape[0]} x {X.shape[1]}  '
f'(10 features activas, categorias en peso 0)\n')

ks, inertias, sils, results = kmeans.choose_k(X, k_min=2, k_max=10)
kmeans.plot_k_selection(ks, inertias, sils)

# COMMAND ----------

# k=5: maximo de silueta y coincide con el codo
labels, C, inertia5, sil5 = results[5]
print(f'k=5  |  silueta={sil5:.4f}  |  inercia={inertia5:.1f}  |  '
      f'tamanos={[int((labels==j).sum()) for j in range(5)]}\n')

profile = kmeans.characterize_clusters(labels, biz_df)
profile

# COMMAND ----------

# MAGIC %md
# MAGIC Cluster 0 — "Los veteranos populares" (n=631). Mediana de 71 reseñas — 4 a 7 veces más que cualquier otro cluster, los más antiguos (10.7 años) y casi todos abiertos (97%). Restaurantes, comida, nightlife y bares: los pesos pesados consolidados de la escena gastronómica. Y acá hay una conexión preciosa con la Parte II: este es exactamente el perfil de los authorities de HITS (District Donuts, Bacchanal, Yo Mama's...) — negocios gastronómicos con cientos o miles de reseñas. Dos métodos distintos, mismo grupo de élite.
# MAGIC
# MAGIC Cluster 1 — "Los que luchan" (n=439). Rating bajo (3.26), pocas reseñas (mediana 10), ~9 años de antigüedad y solo 67% abiertos: un tercio ya cerró. Negocios de gama media en aprietos, a mitad de camino entre los veteranos y el cementerio.
# MAGIC
# MAGIC Cluster 2 — "Las joyas de nicho" (n=513). Rating altísimo (4.74★, el mayor), pocas reseñas (mediana 9), polarización bajísima (0.60 — todos los aman por igual), los más jóvenes (6.4 años) y casi todos abiertos (96%). Shopping, spas, salones y servicios a domicilio: negocios boutique que quien los visita adora, pero de bajo volumen. La joya escondida del manual.
# MAGIC
# MAGIC Cluster 3 — "El cementerio" (n=436). Su rasgo definitorio: 0% abiertos — todos cerrados. Restaurantes/comida/nightlife, ~9 años de antigüedad. K-Means aisló a los que no sobrevivieron. Es un segmento melancólico pero analíticamente jugoso (survivorship bias: ¿qué distingue a los que cerraron?).
# MAGIC
# MAGIC Cluster 4 — "Los servicios divisivos" (n=924, el más grande). Rating bajo (3.06, el menor), pocas reseñas, pero la polarización más alta de todas (1.62), y el 100% abiertos. Shopping, servicios a domicilio, salud, retail cotidiano: el grueso de la economía de servicios, donde las experiencias dividen fuerte (el mecánico que para unos es excelente y para otros un desastre). Amor-odio puro.

# COMMAND ----------

# MAGIC %md
# MAGIC # DBSCAN

# COMMAND ----------

# MAGIC %%writefile dbscan.py
# MAGIC import numpy as np
# MAGIC import pandas as pd
# MAGIC import matplotlib.pyplot as plt
# MAGIC
# MAGIC
# MAGIC def _pairwise_D(X):
# MAGIC     xx = (X * X).sum(1)
# MAGIC     return np.sqrt(np.maximum(xx[:, None] + xx[None, :] - 2 * X @ X.T, 0))
# MAGIC
# MAGIC
# MAGIC def k_distance(X, k, D=None):
# MAGIC     if D is None:
# MAGIC         D = _pairwise_D(X)
# MAGIC     kd = np.sort(D, axis=1)[:, k]
# MAGIC     return np.sort(kd)
# MAGIC
# MAGIC
# MAGIC def plot_k_distance(kd, k):
# MAGIC     plt.figure(figsize=(8, 4))
# MAGIC     plt.plot(np.arange(len(kd)), kd, color='#378ADD')
# MAGIC     plt.xlabel('Negocios (ordenados por distancia)')
# MAGIC     plt.ylabel(f'Distancia al vecino #{k}')
# MAGIC     plt.title(f'k-distance plot (k={k}) — el "codo" sugiere eps')
# MAGIC     plt.grid(alpha=0.3)
# MAGIC     plt.tight_layout(); plt.show()
# MAGIC
# MAGIC
# MAGIC def dbscan(X, eps, min_pts, D=None):
# MAGIC     n = len(X)
# MAGIC     if D is None:
# MAGIC         D = _pairwise_D(X)
# MAGIC     neighbors = [np.where(D[i] <= eps)[0] for i in range(n)]
# MAGIC     labels = np.full(n, -1, dtype=int)
# MAGIC     visited = np.zeros(n, dtype=bool)
# MAGIC     cid = 0
# MAGIC     for i in range(n):
# MAGIC         if visited[i]:
# MAGIC             continue
# MAGIC         visited[i] = True
# MAGIC         if len(neighbors[i]) < min_pts:
# MAGIC             continue
# MAGIC         labels[i] = cid
# MAGIC         seeds = list(neighbors[i])
# MAGIC         j = 0
# MAGIC         while j < len(seeds):
# MAGIC             q = seeds[j]; j += 1
# MAGIC             if not visited[q]:
# MAGIC                 visited[q] = True
# MAGIC                 if len(neighbors[q]) >= min_pts:
# MAGIC                     seeds.extend(neighbors[q].tolist())
# MAGIC             if labels[q] == -1:
# MAGIC                 labels[q] = cid
# MAGIC         cid += 1
# MAGIC     return labels
# MAGIC
# MAGIC
# MAGIC def dbscan_summary(labels):
# MAGIC     n_noise = int((labels == -1).sum())
# MAGIC     clusters = sorted(c for c in set(labels.tolist()) if c != -1)
# MAGIC     rows = [{'cluster': c, 'n': int((labels == c).sum())} for c in clusters]
# MAGIC     rows.append({'cluster': 'outliers (-1)', 'n': n_noise})
# MAGIC     print(f'{len(clusters)} clusters + {n_noise} outliers '
# MAGIC           f'({100 * n_noise / len(labels):.1f}% del total)')
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

dbscan = load_mod('dbscan')

import numpy as np

MIN_PTS = 10
D = dbscan._pairwise_D(X)
kd = dbscan.k_distance(X, k=MIN_PTS, D=D)
dbscan.plot_k_distance(kd, k=MIN_PTS)
print(f'Distancia al vecino #{MIN_PTS}:  min={kd.min():.3f}  '
f'mediana={np.median(kd):.3f}  p90={np.quantile(kd,0.9):.3f}  max={kd.max():.3f}')

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

dbscan = load_mod('dbscan')
kmeans = load_mod('kmeans')

EPS, MIN_PTS = 2.0, 10
db_labels = dbscan.dbscan(X, eps=EPS, min_pts=MIN_PTS, D=D)   # reutiliza la D de arriba
print(f'DBSCAN(eps={EPS}, min_pts={MIN_PTS}):')
display(dbscan.dbscan_summary(db_labels))

# caracterizar incluyendo el grupo -1 (outliers) -> storytelling
print('\nPerfil de los grupos DBSCAN (cluster -1 = outliers):')
kmeans.characterize_clusters(db_labels, biz_df)

# COMMAND ----------

# MAGIC %md
# MAGIC DBSCAN encontró un cluster dominante (el 0, con 1,772 negocios = 60% del total) más cinco grupos chicos y 190 outliers (6.5%). Donde K-Means repartió el continuo en 5 segmentos parejos, DBSCAN dice la verdad geométrica: la mayoría de los negocios forman una sola masa densa conectada, con algunos bolsones más densos alrededor y una cola de rarezas. Ni uno está "mal" — responden preguntas distintas (K-Means segmenta para estrategia; DBSCAN revela densidad y caza anomalías).
# MAGIC
# MAGIC
# MAGIC Coincidencia entre métodos.
# MAGIC
# MAGIC Varios clusters chicos de DBSCAN (el 1, 4 y 5) tienen pct_abierto=0.0 — son negocios cerrados, igual que el "cementerio" (cluster 3) de K-Means. Que dos algoritmos independientes aíslen a los cerrados confirma que el estatus abierto/cerrado es una frontera real en los datos.
# MAGIC
# MAGIC
# MAGIC Los outliers son insight nuevo.
# MAGIC
# MAGIC Los 190 outliers tienen el precio promedio más alto (2.48 vs ~2.0) y estatus mixto (52% abiertos) — son los negocios atípicos (caros, inusuales) que K-Means jamás te marca porque está obligado a meter todo en algún cluster. Esos 190 son material para la discusión ética: ¿son spam, errores, o nichos legítimos? Y fíjate en el cluster 3 de DBSCAN (n=69, precio mediano 3.0, el tope): un bolsón de shopping/moda/spas premium.

# COMMAND ----------

# MAGIC %md
# MAGIC # Comparativa K-Means++ vs DBSCAN

# COMMAND ----------

# MAGIC %%writefile compare.py
# MAGIC import numpy as np
# MAGIC import pandas as pd
# MAGIC import matplotlib.pyplot as plt
# MAGIC
# MAGIC
# MAGIC def pca_2d(X):
# MAGIC     Xc = X - X.mean(0)
# MAGIC     U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
# MAGIC     P = Xc @ Vt[:2].T
# MAGIC     var = (S[:2] ** 2) / (S ** 2).sum()
# MAGIC     return P, var
# MAGIC
# MAGIC
# MAGIC def compare_metrics(X, labels_km, labels_db, silhouette_fn):
# MAGIC     rows = []
# MAGIC     s_km = silhouette_fn(X, np.asarray(labels_km))
# MAGIC     rows.append({'metodo': 'K-Means++', 'n_clusters': len(set(np.asarray(labels_km).tolist())),
# MAGIC                  'outliers': 0, 'silueta': round(s_km, 4)})
# MAGIC     db = np.asarray(labels_db)
# MAGIC     mask = db != -1
# MAGIC     n_db = len(set(db.tolist()) - {-1})
# MAGIC     s_db = silhouette_fn(X[mask], db[mask]) if n_db >= 2 else float('nan')
# MAGIC     rows.append({'metodo': 'DBSCAN', 'n_clusters': n_db,
# MAGIC                  'outliers': int((~mask).sum()),
# MAGIC                  'silueta': round(s_db, 4) if n_db >= 2 else None})
# MAGIC     return pd.DataFrame(rows)
# MAGIC
# MAGIC
# MAGIC def plot_clusters_2d(P, labels_km, labels_db, var):
# MAGIC     fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
# MAGIC     km = np.asarray(labels_km)
# MAGIC     ax[0].scatter(P[:, 0], P[:, 1], c=km, cmap='tab10', s=6, alpha=0.6)
# MAGIC     ax[0].set_title(f'K-Means++  (k={len(set(km.tolist()))})')
# MAGIC
# MAGIC     db = np.asarray(labels_db)
# MAGIC     out = db == -1
# MAGIC     ax[1].scatter(P[out, 0], P[out, 1], c='lightgray', s=6, alpha=0.5, label='outliers')
# MAGIC     ax[1].scatter(P[~out, 0], P[~out, 1], c=db[~out], cmap='tab10', s=6, alpha=0.6)
# MAGIC     ax[1].set_title(f'DBSCAN  ({len(set(db.tolist()) - {-1})} clusters + {int(out.sum())} outliers)')
# MAGIC     ax[1].legend(loc='upper right', fontsize=8)
# MAGIC
# MAGIC     for a in ax:
# MAGIC         a.set_xlabel(f'PC1 ({var[0] * 100:.0f}% var)')
# MAGIC         a.set_ylabel(f'PC2 ({var[1] * 100:.0f}% var)')
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

compare = load_mod('compare')
kmeans = load_mod('kmeans')

metrics = compare.compare_metrics(X, labels, db_labels, kmeans.silhouette_score)
print('Comparativa K-Means++ vs DBSCAN:')
display(metrics)

P, var = compare.pca_2d(X)
compare.plot_clusters_2d(P, labels, db_labels, var)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Lectura de resultados de la Parte III
# MAGIC
# MAGIC En numeros: K-Means++ con k=5 logra silueta 0.167 contra 0.082 de DBSCAN (excluyendo sus 190 outliers). Ninguna es alta -- los negocios no forman globos bien separados sino un continuo -- pero la particion de K-Means es mas equilibrada y accionable: cinco arquetipos nitidos (veteranos populares, los que luchan, joyas de nicho, cementerio y servicios divisivos) que ademas reaparecen al pasar de una ciudad a la muestra global, senal de que son estructura real de Yelp y no un artefacto local.
# MAGIC
# MAGIC DBSCAN aporta lo que K-Means no puede: confirma que el grueso (60%) es una sola masa densa, vuelve a aislar a los negocios cerrados como regiones separadas (misma frontera abierto/cerrado que encontro K-Means) y entrega 190 outliers -- negocios caros y atipicos -- que son candidatos naturales a revision manual.
# MAGIC
# MAGIC Conexion con las otras partes: el cluster 0 (veteranos populares, mediana 71 resenas) es el mismo perfil que los authorities de HITS en la Parte II, y la matriz de features construida aqui se reutiliza tal cual en el PCA de la Parte VI.