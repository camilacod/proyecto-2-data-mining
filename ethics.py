"""ethics.py -- Parte VII: analisis critico y etico.
Evidencia cuantitativa para la discusion: concentracion de actividad,
representacion de grupos y cadenas vs negocios independientes."""
import numpy as np
import pandas as pd


def complexity_table():
    """Complejidad teorica (tiempo / espacio) de cada algoritmo implementado,
    con n = filas, d = features, k = clusters/vecinos/factores, V/E = grafo,
    N = largo del stream, m = usuarios, b = negocios, t = terminos."""
    rows = [
        ('I',   'Streaming JSON + muestreo',   'O(N)',                'O(muestra)',      'si (un solo pase)'),
        ('II',  'PageRank',                    'O(iter * (V + E))',   'O(V + E)',        'si (grafo ralo)'),
        ('II',  'HITS',                        'O(iter * E)',         'O(V + E)',        'si (grafo ralo)'),
        ('II',  'Louvain',                     '~O(E log V)',         'O(V + E)',        'si'),
        ('II',  'Layout force-directed',       'O(iter * n^2)',       'O(n^2)',          'no (solo muestras)'),
        ('III', 'K-Means++',                   'O(iter * n k d)',     'O(n d + k d)',    'si (mini-batch)'),
        ('III', 'DBSCAN (D densa)',            'O(n^2 d)',            'O(n^2)',          'no (requiere indice espacial)'),
        ('III', 'Silueta',                     'O(n^2)',              'O(n^2)',          'no (solo muestras)'),
        ('IV',  'Similitud item-item',         'O(b^2 m)',            'O(b^2)',          'parcial (b chico; m no entra)'),
        ('IV',  'TF-IDF + perfiles',           'O(docs + b t)',       'O(b t)',          'si (matrices ralas)'),
        ('V',   'Ventana deslizante',          'O(1) amortizado',     'O(eventos en w)', 'si'),
        ('V',   'Count-Min Sketch',            'O(d_h) por evento',   'O(e/eps * ln(1/delta))', 'si (memoria fija)'),
        ('V',   'FM / LogLog',                 'O(1) por evento',     'O(m buckets)',    'si (memoria fija)'),
        ('VI',  'PCA (eigh de covarianza)',    'O(n d^2 + d^3)',      'O(d^2)',          'si (d chico)'),
        ('VI',  'SVD completa',                'O(min(n t^2, n^2 t))','O(n t)',          'parcial (usar SVD truncada/aleatorizada)'),
    ]
    return pd.DataFrame(rows, columns=['parte', 'algoritmo', 'tiempo', 'espacio', 'escala a masivo'])


def _pct(x):
    return f'{x:.1%}'


def gini(values):
    v = np.sort(np.asarray(values, dtype=float))
    n = len(v)
    if n == 0 or v.sum() == 0:
        return 0.0
    cum = np.cumsum(v)
    return float((n + 1 - 2 * (cum / cum[-1]).sum()) / n)


def concentration_report(clean):
    """Quien produce la senal de la que aprenden todos los modelos."""
    rev, usr, biz = clean['reviews'], clean['users'], clean['business']
    per_user = rev['user_id'].value_counts()
    per_biz = rev['business_id'].value_counts()

    top1_u = per_user.head(max(1, len(per_user) // 100)).sum() / len(rev)
    top1_b = per_biz.head(max(1, len(per_biz) // 100)).sum() / len(rev)

    elite_ids = set(usr.loc[usr['is_elite'], 'user_id'])
    share_elite = rev['user_id'].isin(elite_ids).mean()

    print('CONCENTRACION DE LA SENAL (quien escribe las resenas)')
    print(f'  usuarios con 1 sola resena:      {_pct((per_user == 1).mean())} de los usuarios')
    print(f'  top 1% de usuarios escribe:      {_pct(top1_u)} de las resenas')
    print(f'  usuarios elite ({_pct(len(elite_ids)/len(usr))} del total) escriben: {_pct(share_elite)} de las resenas')
    print(f'  Gini de resenas por usuario:     {gini(per_user):.3f}')
    print()
    print('CONCENTRACION DE LA VISIBILIDAD (que negocios reciben resenas)')
    print(f'  negocios con < 10 resenas:       {_pct((per_biz.reindex(biz["business_id"]).fillna(0) < 10).mean())} de los negocios')
    print(f'  top 1% de negocios recibe:       {_pct(top1_b)} de las resenas')
    print(f'  Gini de resenas por negocio:     {gini(per_biz.reindex(biz["business_id"]).fillna(0)):.3f}')
    return per_user, per_biz


def chains_vs_small(clean, min_locales=5):
    """Proxy de cadena: mismo nombre en >= min_locales locales de la muestra."""
    biz = clean['business'].copy()
    rev = clean['reviews']
    counts = biz['name'].value_counts()
    biz['es_cadena'] = biz['name'].isin(counts[counts >= min_locales].index)

    per_biz = rev['business_id'].value_counts()
    biz['resenas_muestra'] = per_biz.reindex(biz['business_id']).fillna(0).values

    g = biz.groupby('es_cadena').agg(
        n=('business_id', 'size'),
        stars_prom=('stars', 'mean'),
        resenas_mediana=('review_count', 'median'),
        resenas_muestra_prom=('resenas_muestra', 'mean'),
        pct_abierto=('is_open', 'mean'),
    ).round(2)
    g.index = g.index.map({False: 'independiente', True: f'cadena (>= {min_locales} locales)'})
    ejemplos = counts[counts >= min_locales].head(8)
    print('Cadenas detectadas (top):', ', '.join(f'{n} ({c})' for n, c in ejemplos.items()))
    return g
