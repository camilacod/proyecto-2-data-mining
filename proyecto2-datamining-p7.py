# Databricks notebook source
# MAGIC %md
# MAGIC # Parte VII -- Analisis critico y etico
# MAGIC
# MAGIC Cierre del proyecto: primero comparamos **exactitud vs eficiencia** de todos los metodos implementados (complejidad teorica + mediciones empiricas sobre la muestra) y discutimos cuales escalan a datos realmente masivos. Despues analizamos las **implicancias eticas** de la personalizacion: sesgos de subrepresentacion, resenas falsas/spam y equidad (diversidad vs relevancia, pequenos negocios vs cadenas), anclando cada argumento a numeros medidos en las Partes I-VI.
# MAGIC
# MAGIC Corre en la **misma carpeta del Workspace** que la Parte I: reutiliza sus modulos, los parquet de `artifacts/` y los modulos de las Partes III y V (`features.py`, `kmeans.py`, `dbscan.py`, `streaming.py`).

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
# MAGIC ## Exactitud vs eficiencia
# MAGIC
# MAGIC Todos los algoritmos del proyecto se implementaron desde cero, asi que conocemos exactamente que hace cada linea. La tabla siguiente resume la complejidad de tiempo y espacio de cada uno (con `n` = filas, `d` = features, `k` = clusters/vecinos/factores, `V/E` = nodos/aristas del grafo, `N` = largo del stream, `m` = usuarios, `b` = negocios, `t` = terminos del vocabulario), y la ultima columna adelanta el veredicto de escalabilidad.

# COMMAND ----------

# MAGIC %%writefile ethics.py
# MAGIC """ethics.py -- Parte VII: analisis critico y etico.
# MAGIC Evidencia cuantitativa para la discusion: concentracion de actividad,
# MAGIC representacion de grupos y cadenas vs negocios independientes."""
# MAGIC import numpy as np
# MAGIC import pandas as pd
# MAGIC
# MAGIC
# MAGIC def complexity_table():
# MAGIC     """Complejidad teorica (tiempo / espacio) de cada algoritmo implementado,
# MAGIC     con n = filas, d = features, k = clusters/vecinos/factores, V/E = grafo,
# MAGIC     N = largo del stream, m = usuarios, b = negocios, t = terminos."""
# MAGIC     rows = [
# MAGIC         ('I',   'Streaming JSON + muestreo',   'O(N)',                'O(muestra)',      'si (un solo pase)'),
# MAGIC         ('II',  'PageRank',                    'O(iter * (V + E))',   'O(V + E)',        'si (grafo ralo)'),
# MAGIC         ('II',  'HITS',                        'O(iter * E)',         'O(V + E)',        'si (grafo ralo)'),
# MAGIC         ('II',  'Louvain',                     '~O(E log V)',         'O(V + E)',        'si'),
# MAGIC         ('II',  'Layout force-directed',       'O(iter * n^2)',       'O(n^2)',          'no (solo muestras)'),
# MAGIC         ('III', 'K-Means++',                   'O(iter * n k d)',     'O(n d + k d)',    'si (mini-batch)'),
# MAGIC         ('III', 'DBSCAN (D densa)',            'O(n^2 d)',            'O(n^2)',          'no (requiere indice espacial)'),
# MAGIC         ('III', 'Silueta',                     'O(n^2)',              'O(n^2)',          'no (solo muestras)'),
# MAGIC         ('IV',  'Similitud item-item',         'O(b^2 m)',            'O(b^2)',          'parcial (b chico; m no entra)'),
# MAGIC         ('IV',  'TF-IDF + perfiles',           'O(docs + b t)',       'O(b t)',          'si (matrices ralas)'),
# MAGIC         ('V',   'Ventana deslizante',          'O(1) amortizado',     'O(eventos en w)', 'si'),
# MAGIC         ('V',   'Count-Min Sketch',            'O(d_h) por evento',   'O(e/eps * ln(1/delta))', 'si (memoria fija)'),
# MAGIC         ('V',   'FM / LogLog',                 'O(1) por evento',     'O(m buckets)',    'si (memoria fija)'),
# MAGIC         ('VI',  'PCA (eigh de covarianza)',    'O(n d^2 + d^3)',      'O(d^2)',          'si (d chico)'),
# MAGIC         ('VI',  'SVD completa',                'O(min(n t^2, n^2 t))','O(n t)',          'parcial (usar SVD truncada/aleatorizada)'),
# MAGIC     ]
# MAGIC     return pd.DataFrame(rows, columns=['parte', 'algoritmo', 'tiempo', 'espacio', 'escala a masivo'])
# MAGIC
# MAGIC
# MAGIC def _pct(x):
# MAGIC     return f'{x:.1%}'
# MAGIC
# MAGIC
# MAGIC def gini(values):
# MAGIC     v = np.sort(np.asarray(values, dtype=float))
# MAGIC     n = len(v)
# MAGIC     if n == 0 or v.sum() == 0:
# MAGIC         return 0.0
# MAGIC     cum = np.cumsum(v)
# MAGIC     return float((n + 1 - 2 * (cum / cum[-1]).sum()) / n)
# MAGIC
# MAGIC
# MAGIC def concentration_report(clean):
# MAGIC     """Quien produce la senal de la que aprenden todos los modelos."""
# MAGIC     rev, usr, biz = clean['reviews'], clean['users'], clean['business']
# MAGIC     per_user = rev['user_id'].value_counts()
# MAGIC     per_biz = rev['business_id'].value_counts()
# MAGIC
# MAGIC     top1_u = per_user.head(max(1, len(per_user) // 100)).sum() / len(rev)
# MAGIC     top1_b = per_biz.head(max(1, len(per_biz) // 100)).sum() / len(rev)
# MAGIC
# MAGIC     elite_ids = set(usr.loc[usr['is_elite'], 'user_id'])
# MAGIC     share_elite = rev['user_id'].isin(elite_ids).mean()
# MAGIC
# MAGIC     print('CONCENTRACION DE LA SENAL (quien escribe las resenas)')
# MAGIC     print(f'  usuarios con 1 sola resena:      {_pct((per_user == 1).mean())} de los usuarios')
# MAGIC     print(f'  top 1% de usuarios escribe:      {_pct(top1_u)} de las resenas')
# MAGIC     print(f'  usuarios elite ({_pct(len(elite_ids)/len(usr))} del total) escriben: {_pct(share_elite)} de las resenas')
# MAGIC     print(f'  Gini de resenas por usuario:     {gini(per_user):.3f}')
# MAGIC     print()
# MAGIC     print('CONCENTRACION DE LA VISIBILIDAD (que negocios reciben resenas)')
# MAGIC     print(f'  negocios con < 10 resenas:       {_pct((per_biz.reindex(biz["business_id"]).fillna(0) < 10).mean())} de los negocios')
# MAGIC     print(f'  top 1% de negocios recibe:       {_pct(top1_b)} de las resenas')
# MAGIC     print(f'  Gini de resenas por negocio:     {gini(per_biz.reindex(biz["business_id"]).fillna(0)):.3f}')
# MAGIC     return per_user, per_biz
# MAGIC
# MAGIC
# MAGIC def chains_vs_small(clean, min_locales=5):
# MAGIC     """Proxy de cadena: mismo nombre en >= min_locales locales de la muestra."""
# MAGIC     biz = clean['business'].copy()
# MAGIC     rev = clean['reviews']
# MAGIC     counts = biz['name'].value_counts()
# MAGIC     biz['es_cadena'] = biz['name'].isin(counts[counts >= min_locales].index)
# MAGIC
# MAGIC     per_biz = rev['business_id'].value_counts()
# MAGIC     biz['resenas_muestra'] = per_biz.reindex(biz['business_id']).fillna(0).values
# MAGIC
# MAGIC     g = biz.groupby('es_cadena').agg(
# MAGIC         n=('business_id', 'size'),
# MAGIC         stars_prom=('stars', 'mean'),
# MAGIC         resenas_mediana=('review_count', 'median'),
# MAGIC         resenas_muestra_prom=('resenas_muestra', 'mean'),
# MAGIC         pct_abierto=('is_open', 'mean'),
# MAGIC     ).round(2)
# MAGIC     g.index = g.index.map({False: 'independiente', True: f'cadena (>= {min_locales} locales)'})
# MAGIC     ejemplos = counts[counts >= min_locales].head(8)
# MAGIC     print('Cadenas detectadas (top):', ', '.join(f'{n} ({c})' for n, c in ejemplos.items()))
# MAGIC     return g

# COMMAND ----------

ethics = load_mod('ethics')
display(ethics.complexity_table())

# COMMAND ----------

# Benchmark empirico sobre la muestra (2,943 negocios, 125,744 eventos).
# Requiere que las Partes III y V ya hayan corrido en esta carpeta
# (dejan features.py, kmeans.py, dbscan.py y streaming.py junto al notebook).
import time
from collections import Counter

features = load_mod('features')
kmeans = load_mod('kmeans')
dbscan = load_mod('dbscan')
streaming = load_mod('streaming')

X, feat_names, biz_ids, biz_df = features.build_business_features(
    clean, top_categories=18, cat_weight=0.0)

t0 = time.perf_counter()
kmeans.choose_k(X, k_min=5, k_max=5)
print(f'K-Means k=5 (incluye silueta O(n^2)):    {time.perf_counter()-t0:5.2f}s')

t0 = time.perf_counter()
D = dbscan._pairwise_D(X)
dbscan.dbscan(X, eps=2.0, min_pts=10, D=D)
print(f'DBSCAN (matriz D de {D.nbytes/1e6:.0f} MB en RAM):     {time.perf_counter()-t0:5.2f}s')

stream = streaming.make_stream(clean['reviews'])

t0 = time.perf_counter()
exact = Counter(stream['user_id'])
print(f'Conteo exacto (dict, {len(exact):,} claves):  {time.perf_counter()-t0:5.2f}s')

cms = streaming.CountMinSketch(eps=0.001, delta=0.01)
t0 = time.perf_counter()
for x in stream['user_id']:
    cms.add(x)
print(f'Count-Min Sketch ({cms.memory_cells():,} celdas):    {time.perf_counter()-t0:5.2f}s')

fm = streaming.FlajoletMartinLL(m=256)
t0 = time.perf_counter()
for x in stream['user_id']:
    fm.add(x)
print(f'FM-LogLog (256 enteros):                 {time.perf_counter()-t0:5.2f}s '
      f'-> estima {fm.estimate():,.0f} distintos (exacto {len(exact):,})')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Lectura: tradeoffs de exactitud vs eficiencia
# MAGIC
# MAGIC **Donde pagamos exactitud por eficiencia (y cuanto):**
# MAGIC
# MAGIC - **Count-Min Sketch**: 13,595 celdas en vez de 103,449 contadores (~8x menos memoria) a cambio de sobreestimar en promedio +36 por clave (cota teorica eps*N = 126, cumplida en el 100% de las claves en la Parte V). El error es *unilateral y acotado*: se sabe exactamente que se esta comprando.
# MAGIC - **FM-LogLog**: 256 enteros en vez de un set de ~103k IDs, con error teorico ~8% (1.3/sqrt(256)); en nuestras corridas la estimacion quedo dentro de +-13% del exacto (el hash cambia por sesion, asi que la cifra puntual varia). Para "cuantos usuarios activos hay" ese error es irrelevante; para facturar seria inaceptable. El tradeoff correcto depende del uso.
# MAGIC - **SVD truncada** (Parte VI): con k=100 factores se guarda 12x menos y se retiene el 56% de la varianza. La perdida es medible con los valores singulares antes de decidir k.
# MAGIC - **CF item-based vs content** (Parte IV): el CF es mas exacto (NDCG@10 0.074 vs 0.019) pero depende de co-ratings que el 88% de los usuarios no puede aportar; el content es mas barato y cubre cold-start. El hibrido compra cobertura, no precision.
# MAGIC
# MAGIC **Que escala a datos realmente masivos y que no:**
# MAGIC
# MAGIC - **Escalan**: los algoritmos de streaming (ventanas, CMS, FM: memoria fija, un pase), PageRank/HITS/Louvain (lineales en aristas, paralelizables estilo Pregel/Spark), K-Means en variante mini-batch, TF-IDF con matrices ralas y la SVD truncada/aleatorizada.
# MAGIC - **No escalan tal como estan**: DBSCAN y la silueta con matriz de distancias densa -- ya con n=2,943 la matriz D ocupa ~69 MB; con 300k negocios serian ~700 GB. En produccion se reemplaza por indices espaciales (KD-tree, LSH) o variantes aproximadas (HDBSCAN sobre muestras). Lo mismo el layout force-directed O(n^2), que solo sirve para muestras. La similitud item-item O(b^2) sobrevive aqui porque b=2,943, pero con millones de items requiere LSH o factores latentes (la SVD de la Parte VI es justamente el camino).
# MAGIC - El benchmark de arriba tambien muestra el costo de la *implementacion*: el dict exacto en C (Counter) tarda menos que nuestro CMS en Python puro. La ventaja del sketch es de **memoria y de garantias**, no de velocidad de una implementacion particular -- en un stream real distribuido el sketch ademas se puede fusionar entre nodos (es un monoide), cosa que un dict gigante no hace barato.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Implicancias eticas de la personalizacion
# MAGIC
# MAGIC ### Sesgos: quien produce la senal de la que aprenden los modelos
# MAGIC
# MAGIC Todos los metodos del proyecto (ranking, clustering, recomendacion) aprenden de las resenas. La pregunta etica previa a cualquier algoritmo es: *de quien* son esas resenas y *a quien* describen. Medimos la concentracion:

# COMMAND ----------

ethics = load_mod('ethics')
per_user, per_biz = ethics.concentration_report(clean)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Lectura: subrepresentacion y resenas falsas
# MAGIC
# MAGIC **Subrepresentacion.** El 88% de los usuarios escribio una sola resena, mientras la elite de Yelp (16.5% de los usuarios) produce el 25% de todas las resenas. La "opinion agregada" que ven los modelos es en realidad la opinion de una minoria hiperactiva, demograficamente particular (usuarios urbanos, anglofonos, con smartphone y habito de resenar). Del lado de los negocios la desigualdad es mayor: Gini de 0.666 en resenas por negocio, el top 1% de negocios concentra el 18% de las resenas y un 34% de los negocios tiene menos de 10 -- son casi invisibles para cualquier metodo. Efectos concretos en nuestras partes:
# MAGIC
# MAGIC - **Clustering (III)**: log_reviews y log_checkins son features; los negocios poco resenados caen a los clusters de "baja senal" (el cluster 4, servicios divisivos) no porque sean peores sino porque nadie los describe.
# MAGIC - **Ranking (II)**: PageRank correlaciona 0.98 con las amistades dentro de la muestra -- premia a quien ya esta conectado (rich-get-richer), y HITS pondera a los resenadores prolificos: la elite decide que negocios son "authorities".
# MAGIC - **Recomendacion (IV)**: el CF necesita co-ratings, que solo la minoria activa genera; el 88% con una resena queda en cold-start y recibe el top-popular, o sea, mas exposicion para los ya populares.
# MAGIC - El **muestreo estratificado de la Parte I** preserva proporciones por ciudad (un sesgo que si pudimos controlar), pero no puede corregir sesgos que el dataset ya trae: grupos que no resenan simplemente no existen en los datos.
# MAGIC
# MAGIC **Resenas falsas / spam.** No las detectamos explicitamente, pero el pipeline muestra exactamente donde golpearian:
# MAGIC
# MAGIC - **Clustering**: cuentas infladas moverian rating_std y log_reviews; los 190 outliers de DBSCAN (Parte III) son el lugar natural donde buscarlas -- DBSCAN es aqui una herramienta de auditoria, no solo de segmentacion.
# MAGIC - **Ranking**: HITS es vulnerable a *link farms*: un anillo de cuentas falsas resenandose entre si fabrica un authority. PageRank sobre amistades es mas robusto (crear amistades reciprocas cuesta mas que crear resenas) pero no inmune.
# MAGIC - **Recomendacion**: el ataque clasico de *shilling* (muchas cuentas calificando igual un item objetivo) infla directamente la similitud item-item del CF. El filtro de co-ocurrencia minima (>= 3 usuarios comunes) sube el costo del ataque pero no lo elimina.
# MAGIC - **Streaming**: el CMS *nunca subestima*, asi que el spam solo puede inflar frecuencias; y el 88% de usuarios con una sola resena es estadisticamente indistinguible de cuentas descartables -- cualquier defensa basada en historial deja fuera justo a los usuarios legitimos nuevos.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Equidad: pequenos negocios vs cadenas, diversidad vs relevancia
# MAGIC
# MAGIC Como proxy de "cadena" usamos nombres que aparecen en 5 o mas locales de la muestra (Starbucks, McDonald's, Pizza Hut...):

# COMMAND ----------

ethics = load_mod('ethics')
ethics.chains_vs_small(clean)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Lectura: a quien favorecen los rankings y las recomendaciones
# MAGIC
# MAGIC - **Cadenas vs independientes**: las 135 sucursales de cadena (4.6% de la muestra) tienen rating promedio 2.60 contra 3.63 de los independientes -- en la *calidad percibida* los pequenos ganan con claridad. Pero las cadenas sobreviven mas (92% abiertas vs 78%): su ventaja no es la opinion sino la resiliencia estructural. Un ranking por rating no las favorece; un ranking por *volumen* o disponibilidad si lo haria. La eleccion de la metrica de ordenamiento es una decision con consecuencias distributivas.
# MAGIC - **Popularity bias**: en la Parte IV el baseline top-popular ya logra NDCG@10 = 0.029 (10x el aleatorio) y el CF -- que aprende de co-consumo -- tambien favorece a los items con muchos ratings. El resultado es un *feedback loop*: los negocios visibles reciben recomendaciones, luego mas visitas, luego mas resenas; el 34% con menos de 10 resenas practicamente no puede entrar a un top-10. La relevancia medida (NDCG) premia reforzar lo conocido.
# MAGIC - **Diversidad vs relevancia**: maximizar NDCG@10 puro empuja hacia la homogeneidad. Mitigaciones compatibles con nuestro pipeline: mantener el componente content-based (puntua negocios con 1 sola resena, que el CF ignora), re-rankear el top-K con cuotas de diversidad (por categoria o por tramo de popularidad) y reservar posiciones de *exploracion* para negocios nuevos, aceptando una perdida medida de NDCG a cambio de un ecosistema menos concentrado.
# MAGIC - **Equidad entre usuarios**: la personalizacion de calidad es para la minoria con historial; el 88% restante recibe recomendaciones genericas. Si ese 88% se distribuye distinto por grupo demografico (probable), la *calidad del servicio* queda repartida de forma desigual sin que ninguna metrica agregada lo muestre. Auditar por segmento (no solo en promedio) es la unica forma de verlo.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Conclusiones de la Parte VII
# MAGIC
# MAGIC 1. **Exactitud vs eficiencia no es una sola escala**: los sketches (CMS, FM) y la SVD truncada compran ordenes de magnitud de memoria con errores acotados y medibles; DBSCAN denso, la silueta y el layout de grafos no tienen version barata y en masivo se reemplazan por aproximaciones (LSH, indices espaciales, muestras). Lo que escala comparte un patron: un pase, memoria sublineal, y estructuras fusionables/paralelizables.
# MAGIC 2. **El sesgo dominante del dataset es de participacion**: una minoria (elite 16.5% -> 25% de las resenas) escribe la realidad que los modelos aprenden, y un tercio de los negocios es casi invisible. Ningun ajuste algoritmico posterior recupera a los que no estan en los datos.
# MAGIC 3. **Cada metodo tiene su vector de manipulacion** (link farms para HITS, shilling para CF, inflado para CMS) y tambien sus defensas naturales (filtros de co-ocurrencia, PageRank sobre relaciones costosas de falsificar, outliers de DBSCAN como auditoria).
# MAGIC 4. **La equidad se decide en el diseno, no en el algoritmo**: con estos mismos datos, ordenar por rating favorece a los independientes y ordenar por volumen a los populares/cadenas; maximizar relevancia concentra la exposicion. Explicitar esas decisiones -- y auditarlas por segmento -- es la parte etica del trabajo de mineria de datos.
