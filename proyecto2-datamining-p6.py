# Databricks notebook source
# MAGIC %md
# MAGIC # Parte VI -- Reduccion de Dimensionalidad
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
# MAGIC # PARTE 6 — Reduccion de Dimensionalidad
# MAGIC
# MAGIC Trabajamos sobre dos representaciones de los negocios de la muestra:
# MAGIC
# MAGIC 1. **PCA desde cero** sobre la matriz de features de la Parte III (~28 dimensiones: 10 continuas + 18 categorias binarias): estandarizacion -> matriz de covarianza -> eigendescomposicion -> seleccion de componentes que expliquen 90%+ de varianza -> proyeccion 2D/3D e interpretacion de loadings.
# MAGIC 2. **SVD** sobre la matriz TF-IDF de las resenas (negocios x 2,000 terminos): los factores singulares emergen como *temas latentes*, y la version truncada nos da compresion con error de reconstruccion medible.
# MAGIC
# MAGIC La celda siguiente re-escribe `features.py` tal cual se uso en la Parte III, para construir exactamente la misma matriz.

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

# MAGIC %%writefile dimensionality_reduction.py
# MAGIC import numpy as np
# MAGIC import pandas as pd
# MAGIC import matplotlib.pyplot as plt
# MAGIC
# MAGIC
# MAGIC # ===================== PCA desde cero =====================
# MAGIC def pca_fit(Xz):
# MAGIC     """Xz ya estandarizada. Covarianza -> eigendescomposicion (eigh porque la
# MAGIC     matriz es simetrica) -> componentes ordenadas por varianza descendente."""
# MAGIC     n = Xz.shape[0]
# MAGIC     cov = (Xz.T @ Xz) / (n - 1)
# MAGIC     eigvals, eigvecs = np.linalg.eigh(cov)
# MAGIC     order = np.argsort(eigvals)[::-1]
# MAGIC     eigvals = np.maximum(eigvals[order], 0.0)
# MAGIC     eigvecs = eigvecs[:, order]
# MAGIC     evr = eigvals / eigvals.sum()
# MAGIC     return eigvals, eigvecs, evr
# MAGIC
# MAGIC
# MAGIC def n_components_for(evr, target=0.90):
# MAGIC     cum = np.cumsum(evr)
# MAGIC     k = int(np.searchsorted(cum, target) + 1)
# MAGIC     print(f'{k} componentes explican {cum[k-1]:.1%} de la varianza '
# MAGIC           f'(objetivo {target:.0%}, de {len(evr)} dims originales)')
# MAGIC     return k
# MAGIC
# MAGIC
# MAGIC def plot_scree(evr, target=0.90):
# MAGIC     cum = np.cumsum(evr)
# MAGIC     k = int(np.searchsorted(cum, target) + 1)
# MAGIC     fig, ax = plt.subplots(1, 2, figsize=(12, 4))
# MAGIC     ax[0].bar(range(1, len(evr) + 1), evr, color='#378ADD')
# MAGIC     ax[0].set_xlabel('componente'); ax[0].set_ylabel('varianza explicada')
# MAGIC     ax[0].set_title('Scree plot')
# MAGIC     ax[1].plot(range(1, len(evr) + 1), cum, 'o-', color='#D85A30', ms=3)
# MAGIC     ax[1].axhline(target, ls='--', color='gray', alpha=0.7)
# MAGIC     ax[1].axvline(k, ls='--', color='gray', alpha=0.7)
# MAGIC     ax[1].set_xlabel('componentes'); ax[1].set_ylabel('varianza acumulada')
# MAGIC     ax[1].set_title(f'{k} PCs alcanzan {target:.0%}')
# MAGIC     plt.tight_layout(); plt.show()
# MAGIC     return k
# MAGIC
# MAGIC
# MAGIC def pca_project(Xz, eigvecs, k):
# MAGIC     return Xz @ eigvecs[:, :k]
# MAGIC
# MAGIC
# MAGIC def loadings_table(eigvecs, evr, feat_names, n_pc=4, top=6):
# MAGIC     """Variables que mas pesan (|loading|) en cada componente principal."""
# MAGIC     rows = []
# MAGIC     for j in range(n_pc):
# MAGIC         v = eigvecs[:, j]
# MAGIC         idx = np.argsort(-np.abs(v))[:top]
# MAGIC         rows.append({
# MAGIC             'PC': f'PC{j+1}',
# MAGIC             'var_explicada': f'{evr[j]:.1%}',
# MAGIC             'top_variables': ', '.join(f'{feat_names[i]} ({v[i]:+.2f})' for i in idx),
# MAGIC         })
# MAGIC     return pd.DataFrame(rows)
# MAGIC
# MAGIC
# MAGIC def plot_projection(P, color, color_label, evr):
# MAGIC     fig = plt.figure(figsize=(13, 5.5))
# MAGIC     ax0 = fig.add_subplot(1, 2, 1)
# MAGIC     sc = ax0.scatter(P[:, 0], P[:, 1], c=color, cmap='viridis', s=7, alpha=0.6)
# MAGIC     plt.colorbar(sc, ax=ax0, label=color_label)
# MAGIC     ax0.set_xlabel(f'PC1 ({evr[0]:.0%})'); ax0.set_ylabel(f'PC2 ({evr[1]:.0%})')
# MAGIC     ax0.set_title('Proyeccion PCA 2D')
# MAGIC     ax1 = fig.add_subplot(1, 2, 2, projection='3d')
# MAGIC     ax1.scatter(P[:, 0], P[:, 1], P[:, 2], c=color, cmap='viridis', s=5, alpha=0.5)
# MAGIC     ax1.set_xlabel('PC1'); ax1.set_ylabel('PC2'); ax1.set_zlabel('PC3')
# MAGIC     ax1.set_title('Proyeccion PCA 3D')
# MAGIC     plt.tight_layout(); plt.show()
# MAGIC
# MAGIC
# MAGIC # ===================== SVD sobre TF-IDF =====================
# MAGIC def svd_fit(M):
# MAGIC     U, S, Vt = np.linalg.svd(M, full_matrices=False)
# MAGIC     print(f'SVD: M {M.shape[0]:,}x{M.shape[1]:,} -> '
# MAGIC           f'{len(S)} valores singulares (s1={S[0]:.2f}, s2={S[1]:.2f}, ...)')
# MAGIC     return U, S, Vt
# MAGIC
# MAGIC
# MAGIC def latent_factors(Vt, vocab, n_factors=6, top=8):
# MAGIC     """Cada fila de Vt es una direccion latente en el espacio de terminos:
# MAGIC     los terminos con mayor |peso| revelan el 'tema' del factor."""
# MAGIC     rows = []
# MAGIC     for j in range(n_factors):
# MAGIC         v = Vt[j]
# MAGIC         idx = np.argsort(-np.abs(v))[:top]
# MAGIC         rows.append({'factor': j + 1,
# MAGIC                      'terminos_dominantes': ', '.join(vocab[i] for i in idx)})
# MAGIC     return pd.DataFrame(rows)
# MAGIC
# MAGIC
# MAGIC def reconstruction_curve(M, U, S, Vt, ks):
# MAGIC     """Error relativo de Frobenius al truncar a k factores. Por Eckart-Young la
# MAGIC     SVD truncada es la mejor aproximacion de rango k posible."""
# MAGIC     fro2 = float((S ** 2).sum())
# MAGIC     rows = []
# MAGIC     n, m = M.shape
# MAGIC     full_cells = n * m
# MAGIC     for k in ks:
# MAGIC         err = float(np.sqrt((S[k:] ** 2).sum() / fro2))
# MAGIC         stored = k * (n + m + 1)
# MAGIC         rows.append({'k': k,
# MAGIC                      'error_relativo': round(err, 4),
# MAGIC                      'varianza_capturada': round(1 - err ** 2, 4),
# MAGIC                      'celdas_almacenadas': stored,
# MAGIC                      'compresion': f'{full_cells / stored:.0f}x'})
# MAGIC     return pd.DataFrame(rows)
# MAGIC
# MAGIC
# MAGIC def plot_reconstruction(rec):
# MAGIC     fig, ax = plt.subplots(1, 2, figsize=(12, 4))
# MAGIC     ax[0].plot(rec['k'], rec['error_relativo'], 'o-', color='#378ADD')
# MAGIC     ax[0].set_xlabel('k (factores retenidos)'); ax[0].set_ylabel('error relativo ||M-Mk||/||M||')
# MAGIC     ax[0].set_title('Error de reconstruccion vs k')
# MAGIC     ratio = [float(str(c).rstrip('x')) for c in rec['compresion']]
# MAGIC     ax[1].plot(rec['k'], ratio, 'o-', color='#D85A30')
# MAGIC     ax[1].set_yscale('log')
# MAGIC     ax[1].set_xlabel('k'); ax[1].set_ylabel('factor de compresion (log)')
# MAGIC     ax[1].set_title('Compresion lograda vs k')
# MAGIC     plt.tight_layout(); plt.show()
# MAGIC
# MAGIC
# MAGIC from collections import Counter
# MAGIC
# MAGIC _STOP = set('''a al algo ante antes como con contra cual cuando de del desde donde
# MAGIC durante e el ella ellas ellos en entre era eran es esa esas ese eso esos esta estas
# MAGIC este esto estos fue fueron ha han hasta hay la las le les lo los mas me mi mientras
# MAGIC muy nada ni no nos nosotros o os otra otros para pero poco por porque que quien se
# MAGIC ser si sin sobre son su sus te tiene tienen todo todos tu tus un una uno unos y ya
# MAGIC yo the a an and or of to in for on at is are was were be been it its this that with
# MAGIC i we you they he she my your our their not but so if as from had has have do does
# MAGIC did just very really there here all can will would about out up down them his her
# MAGIC '''.split())
# MAGIC
# MAGIC
# MAGIC def _tokens(text):
# MAGIC     out, cur = [], []
# MAGIC     for ch in text.lower():
# MAGIC         if ch.isalpha():
# MAGIC             cur.append(ch)
# MAGIC         else:
# MAGIC             if len(cur) > 2:
# MAGIC                 out.append(''.join(cur))
# MAGIC             cur = []
# MAGIC     if len(cur) > 2:
# MAGIC         out.append(''.join(cur))
# MAGIC     return [t for t in out if t not in _STOP]
# MAGIC
# MAGIC
# MAGIC def build_tfidf(reviews, items, max_reviews_per_item=50, vocab_size=2000):
# MAGIC     """Documento = negocio (concatenacion de hasta 50 resenas).
# MAGIC     TF = frecuencia relativa en el doc, IDF = log(N / (1+df))."""
# MAGIC     by_item = reviews.groupby('business_id')['text']
# MAGIC     docs_tokens, df_counter = {}, Counter()
# MAGIC     for b in items:
# MAGIC         if b in by_item.groups:
# MAGIC             texts = by_item.get_group(b).head(max_reviews_per_item)
# MAGIC             toks = _tokens(' '.join(texts))
# MAGIC         else:
# MAGIC             toks = []
# MAGIC         docs_tokens[b] = Counter(toks)
# MAGIC         df_counter.update(set(toks))
# MAGIC
# MAGIC     vocab = [w for w, _ in df_counter.most_common(vocab_size)]
# MAGIC     widx = {w: j for j, w in enumerate(vocab)}
# MAGIC     N = len(items)
# MAGIC     idf = np.array([np.log(N / (1 + df_counter[w])) for w in vocab], dtype=np.float32)
# MAGIC
# MAGIC     M = np.zeros((N, len(vocab)), dtype=np.float32)
# MAGIC     for i, b in enumerate(items):
# MAGIC         cnt = docs_tokens[b]
# MAGIC         total = sum(cnt.values()) or 1
# MAGIC         for w, c in cnt.items():
# MAGIC             j = widx.get(w)
# MAGIC             if j is not None:
# MAGIC                 M[i, j] = (c / total) * idf[j]
# MAGIC     print(f'TF-IDF: {N:,} negocios x {len(vocab):,} terminos')
# MAGIC     return M, vocab

# COMMAND ----------

# MAGIC %md
# MAGIC ### PCA: varianza explicada y seleccion de componentes
# MAGIC
# MAGIC La matriz ya sale estandarizada de `features.py` (z-score por columna), condicion necesaria para que ninguna feature domine la covarianza solo por su escala. Diagonalizamos la covarianza con `eigh` (la matriz es simetrica) y ordenamos las componentes por varianza descendente.

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
dimred = load_mod('dimensionality_reduction')

X, feat_names, biz_ids, biz_df = features.build_business_features(clean, top_categories=18)
print(f'Matriz de features: {X.shape[0]:,} negocios x {X.shape[1]} dimensiones\n')

eigvals, eigvecs, evr = dimred.pca_fit(X)
k90 = dimred.n_components_for(evr, target=0.90)
dimred.plot_scree(evr, target=0.90)

# COMMAND ----------

# sanity check: nuestro PCA debe coincidir con la SVD de la matriz centrada
import numpy as np
_, S_check, _ = np.linalg.svd(X - X.mean(0), full_matrices=False)
evr_svd = S_check**2 / (S_check**2).sum()
assert np.allclose(np.cumsum(evr)[:10], np.cumsum(evr_svd)[:10], atol=1e-8)
print('PCA propio == SVD de numpy: OK')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Interpretacion: que variables cargan en cada componente
# MAGIC
# MAGIC Los *loadings* (coordenadas de los eigenvectores) dicen que combinacion de variables originales es cada PC. El signo indica direccion: dos variables con signos opuestos en la misma PC se mueven en sentidos contrarios a lo largo de ese eje.

# COMMAND ----------

dimred.loadings_table(eigvecs, evr, feat_names, n_pc=5, top=6)

# COMMAND ----------

# proyeccion 2D/3D coloreada por rating del negocio
P = dimred.pca_project(X, eigvecs, max(k90, 3))
dimred.plot_projection(P, biz_df['stars'].to_numpy(), 'stars', evr)

# COMMAND ----------

# MAGIC %md
# MAGIC En esta corrida se necesitan **20 de 28 componentes para el 90% de la varianza**: las features estan poco correlacionadas entre si (los dummies de categoria son casi ortogonales), asi que no hay una compresion drastica, pero las primeras PCs si son legibles:
# MAGIC
# MAGIC - **PC1 (12.9%)** junta popularidad y gastronomia: log_checkins, cat::Restaurants, log_reviews, cat::Bars y cat::Nightlife cargan todas positivo -- el eje "negocio gastronomico concurrido".
# MAGIC - **PC2 (7.1%)** opone nightlife/bars (negativo) a coffee & tea (positivo): dentro de la gastronomia, bar nocturno vs cafeteria.
# MAGIC - **PC3 (6.5%)** es el eje shopping/fashion/food con stars positivo: comercio minorista bien calificado.
# MAGIC - **PC4 (6.2%)** es el eje **calidad/polarizacion**: stars carga negativo contra business_age, avg_rev_len y rating_std positivos -- negocios viejos, polarizados y con resenas largas vs jovenes bien calificados.
# MAGIC
# MAGIC La proyeccion 2D muestra franjas/manchas por familia de categorias -- coherente con lo que K-Means y DBSCAN encontraron en la Parte III -- y el gradiente de color muestra donde viven los negocios mejor calificados.

# COMMAND ----------

# MAGIC %md
# MAGIC ### SVD sobre la matriz TF-IDF de resenas
# MAGIC
# MAGIC Ahora la matriz es de verdad de alta dimension: cada negocio como documento (concatenacion de hasta 50 resenas) sobre un vocabulario de 2,000 terminos. `M = U S V^T`:
# MAGIC
# MAGIC - las filas de `V^T` son direcciones en el espacio de terminos -> **temas latentes**;
# MAGIC - `U S` da las coordenadas de cada negocio en esos temas;
# MAGIC - truncar a `k` factores da la mejor aproximacion de rango `k` (teorema de Eckart-Young), con error `||M - M_k||_F / ||M||_F = sqrt(sum_{i>k} s_i^2 / sum_i s_i^2)` calculable directo de los valores singulares.

# COMMAND ----------

M, vocab = dimred.build_tfidf(clean['reviews'], list(biz_df['business_id']),
                              max_reviews_per_item=50, vocab_size=2000)
U, S, Vt = dimred.svd_fit(M)

import matplotlib.pyplot as plt
plt.figure(figsize=(7, 3.5))
plt.plot(S[:200], color='#378ADD')
plt.xlabel('indice del valor singular'); plt.ylabel('s_i')
plt.title('Decaimiento de los valores singulares (primeros 200)')
plt.tight_layout(); plt.show()

# COMMAND ----------

# temas latentes: terminos dominantes de los primeros factores
dimred.latent_factors(Vt, vocab, n_factors=8, top=8)

# COMMAND ----------

# compresion vs perdida: truncamos a k y medimos error de reconstruccion
rec = dimred.reconstruction_curve(M, U, S, Vt, ks=[5, 10, 25, 50, 100, 200, 400])
dimred.plot_reconstruction(rec)
rec

# COMMAND ----------

# MAGIC %md
# MAGIC ### Conclusiones de la parte de dimensionalidad
# MAGIC
# MAGIC - **PCA** necesita 20 de 28 componentes para retener el 90% de la varianza (compresion moderada: los dummies de categoria aportan direcciones casi independientes), pero los loadings hacen el resultado interpretable: el primer eje es *popularidad gastronomica*, luego vienen ejes de *tipo de negocio* (bar vs cafeteria, shopping) y el eje de *calidad/polarizacion*. Ademas validamos la implementacion propia contra la SVD de numpy.
# MAGIC - **SVD sobre TF-IDF** encuentra temas latentes legibles: los primeros factores mezclan los dominios dominantes del corpus (pizza/comida, hair/salon, car/automotriz, tacos/mexicana, hotel/pool), es decir, el texto de las resenas recupera solo el tipo de negocio. La curva de reconstruccion cuantifica el tradeoff compresion/informacion: con k=100 factores (compresion 12x) se captura el 56% de la varianza, y con k=400 (3x) el 80%. El decaimiento gradual dice que el corpus es tematicamente diverso: no hay 5 temas que lo expliquen todo, pero unos cientos de factores bastan frente a 2,000 terminos -- la hipotesis detras de LSA y de los recomendadores de factores latentes.
# MAGIC - Diferencia clave entre ambas: PCA opera sobre features **densas y disenadas a mano** (covarianza interpretable variable a variable), mientras la SVD trabaja la matriz **rala y de alta dimension** del texto, donde la nocion util no es "varianza de una variable" sino "estructura de bajo rango".