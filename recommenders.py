import numpy as np
import pandas as pd


# ===================== Split temporal =====================
def temporal_split(reviews, min_reviews=5, test_frac=0.2):
    """Por cada usuario con >= min_reviews resenas, sus ultimas ~20% van a test.
    Los usuarios con pocas resenas quedan enteros en train (masa cold-start)."""
    df = reviews.sort_values('date', kind='mergesort').reset_index(drop=True)
    df['_pos'] = df.groupby('user_id').cumcount()
    df['_n'] = df.groupby('user_id')['user_id'].transform('size')
    n_test = (df['_n'] * test_frac).round().clip(lower=1)
    is_test = (df['_n'] >= min_reviews) & (df['_pos'] >= df['_n'] - n_test)
    train = df[~is_test].drop(columns=['_pos', '_n']).reset_index(drop=True)
    test = df[is_test].drop(columns=['_pos', '_n']).reset_index(drop=True)
    eval_users = sorted(test['user_id'].unique())
    print(f'Split temporal: {len(train):,} train / {len(test):,} test '
          f'({len(eval_users):,} usuarios evaluables con >= {min_reviews} resenas)')
    return train, test, eval_users


# ===================== Matriz usuario-item =====================
def build_matrix(train):
    users = np.sort(train['user_id'].unique())
    items = np.sort(train['business_id'].unique())
    uidx = {u: i for i, u in enumerate(users)}
    iidx = {b: j for j, b in enumerate(items)}
    R = np.zeros((len(users), len(items)), dtype=np.float32)
    rows = train['user_id'].map(uidx).to_numpy()
    cols = train['business_id'].map(iidx).to_numpy()
    R[rows, cols] = train['stars'].to_numpy(dtype=np.float32)
    mask = R > 0
    dens = mask.sum() / (R.shape[0] * R.shape[1])
    print(f'Matriz R: {R.shape[0]:,} usuarios x {R.shape[1]:,} negocios '
          f'({mask.sum():,} ratings, densidad {dens:.2e})')
    return R, mask, users, items, uidx, iidx


# ===================== CF item-based (adjusted cosine) =====================
def item_similarity(R, mask, min_common=3):
    """Coseno ajustado: centramos cada fila por la media del usuario para
    quitar su sesgo de calificacion, luego coseno columna-columna."""
    cnt = mask.sum(1)
    mu_user = np.divide(R.sum(1), np.maximum(cnt, 1))
    Rc = np.where(mask, R - mu_user[:, None], 0.0).astype(np.float32)

    S = Rc.T @ Rc
    norms = np.sqrt(np.diag(S).copy())
    norms[norms == 0] = 1.0
    S /= norms[:, None]
    S /= norms[None, :]

    # solo pares con suficientes usuarios en comun (evita similitudes espurias)
    co = mask.T.astype(np.float32) @ mask.astype(np.float32)
    S[co < min_common] = 0.0
    np.fill_diagonal(S, 0.0)
    return S, Rc, mu_user


def topk_prune(S, k=30):
    """k-NN: por cada item conservamos solo sus k vecinos mas similares."""
    Sk = np.zeros_like(S)
    k = min(k, S.shape[0] - 1)
    idx = np.argpartition(-S, kth=k - 1, axis=0)[:k]
    cols = np.arange(S.shape[1])[None, :].repeat(k, axis=0)
    Sk[idx, cols] = S[idx, cols]
    Sk[Sk < 0] = 0.0
    return Sk


def predict_cf(Rc, mask, mu_user, Sk, rows=None):
    """pred(u,i) = mu_u + sum_j s_ij * (r_uj - mu_u) / sum_j |s_ij|  (j: items de u)"""
    if rows is None:
        rows = np.arange(Rc.shape[0])
    num = Rc[rows] @ Sk
    den = mask[rows].astype(np.float32) @ np.abs(Sk)
    known = den > 1e-8
    P = np.where(known, mu_user[rows, None] + num / np.maximum(den, 1e-8), np.nan)
    return np.clip(P, 1.0, 5.0), known


def rating_metrics(P_eval, eval_rows_pos, test, uidx, iidx):
    """RMSE/MAE sobre pares (u,i) del test que existen en la matriz de train."""
    pos_of_user = {r: p for p, r in enumerate(eval_rows_pos)}
    y_true, y_pred, n_cold_item, n_no_neighbors = [], [], 0, 0
    for u, b, s in zip(test['user_id'], test['business_id'], test['stars']):
        if b not in iidx:
            n_cold_item += 1
            continue
        p = P_eval[pos_of_user[uidx[u]], iidx[b]]
        if np.isnan(p):
            n_no_neighbors += 1
            continue
        y_true.append(float(s))
        y_pred.append(float(p))
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    rmse = float(np.sqrt(((y_true - y_pred) ** 2).mean()))
    mae = float(np.abs(y_true - y_pred).mean())
    print(f'Ratings evaluables: {len(y_true):,}  |  items cold (sin train): {n_cold_item:,}'
          f'  |  sin vecinos: {n_no_neighbors:,}')
    print(f'RMSE = {rmse:.4f}   MAE = {mae:.4f}')
    return rmse, mae, n_cold_item


def show_topk(scores, seen_mask, items, business, k=10):
    """Imprime el top-k de un vector de puntajes, excluyendo lo ya visto."""
    s = np.where(np.isnan(scores), -np.inf, scores).copy()
    s[seen_mask] = -np.inf
    top = np.argsort(-s)[:k]
    out = pd.DataFrame({'business_id': items[top], 'score': np.round(s[top], 3)})
    info = business[['business_id', 'name', 'stars', 'review_count', 'categories']]
    out = out.merge(info, on='business_id', how='left')
    print(out.to_string(index=False))
    return out


from collections import Counter


# ===================== Content-based: TF-IDF =====================
_STOP = set('''a al algo ante antes como con contra cual cuando de del desde donde
durante e el ella ellas ellos en entre era eran es esa esas ese eso esos esta estas
este esto estos fue fueron ha han hasta hay la las le les lo los mas me mi mientras
muy nada ni no nos nosotros o os otra otros para pero poco por porque que quien se
ser si sin sobre son su sus te tiene tienen todo todos tu tus un una uno unos y ya
yo the a an and or of to in for on at is are was were be been it its this that with
i we you they he she my your our their not but so if as from had has have do does
did just very really there here all can will would about out up down them his her
'''.split())


def _tokens(text):
    out, cur = [], []
    for ch in text.lower():
        if ch.isalpha():
            cur.append(ch)
        else:
            if len(cur) > 2:
                out.append(''.join(cur))
            cur = []
    if len(cur) > 2:
        out.append(''.join(cur))
    return [t for t in out if t not in _STOP]


def build_tfidf(train, items, max_reviews_per_item=50, vocab_size=2000):
    """Perfil de contenido por negocio: TF-IDF del texto concatenado de sus
    resenas de TRAIN (max 50 por negocio para acotar costo), filas L2-normalizadas."""
    by_item = train.groupby('business_id')['text']
    docs_tokens, df_counter = {}, Counter()
    for b in items:
        if b in by_item.groups:
            texts = by_item.get_group(b).head(max_reviews_per_item)
            toks = _tokens(' '.join(texts))
        else:
            toks = []
        docs_tokens[b] = Counter(toks)
        df_counter.update(set(toks))

    vocab = [w for w, _ in df_counter.most_common(vocab_size)]
    widx = {w: j for j, w in enumerate(vocab)}
    N = len(items)
    idf = np.array([np.log(N / (1 + df_counter[w])) for w in vocab], dtype=np.float32)

    T = np.zeros((N, len(vocab)), dtype=np.float32)
    for i, b in enumerate(items):
        cnt = docs_tokens[b]
        total = sum(cnt.values()) or 1
        for w, c in cnt.items():
            j = widx.get(w)
            if j is not None:
                T[i, j] = (c / total) * idf[j]
    norms = np.linalg.norm(T, axis=1)
    norms[norms == 0] = 1.0
    T /= norms[:, None]
    n_empty = int((T.sum(1) == 0).sum())
    print(f'TF-IDF: {N:,} negocios x {len(vocab):,} terminos '
          f'(IDF sobre resenas de train; {n_empty} negocios sin texto)')
    return T, vocab


def content_scores(T, R, mask, rows):
    """score(u,i) = coseno(perfil_u, item_i), con perfil_u = promedio de los
    vectores TF-IDF de los items del usuario ponderado por (rating - 3)."""
    W = np.where(mask[rows], R[rows] - 3.0, 0.0).astype(np.float32)
    prof = W @ T
    norms = np.linalg.norm(prof, axis=1)
    norms[norms == 0] = 1.0
    prof /= norms[:, None]
    return prof @ T.T  # items ya estan L2-normalizados


# ===================== Metricas de ranking =====================
def rank_eval(score_rows, eval_rows_pos, mask, test, users, items, iidx, K=10):
    """Precision@K, Recall@K y NDCG@K. Relevante = item del test con >= 4 estrellas.
    Los items ya vistos en train se excluyen del ranking."""
    rel = {}
    for u, b, s in zip(test['user_id'], test['business_id'], test['stars']):
        if s >= 4.0 and b in iidx:
            rel.setdefault(u, set()).add(iidx[b])

    precs, recs, ndcgs, n_eval = [], [], [], 0
    for p, r in enumerate(eval_rows_pos):
        u = users[r]
        relevant = rel.get(u)
        if not relevant:
            continue
        s = score_rows[p].copy()
        s[mask[r]] = -np.inf          # no recomendar lo ya consumido
        topk = np.argsort(-s)[:K]
        hits = np.array([1.0 if j in relevant else 0.0 for j in topk])
        precs.append(hits.mean())
        recs.append(hits.sum() / len(relevant))
        dcg = (hits / np.log2(np.arange(2, K + 2))).sum()
        ideal = min(len(relevant), K)
        idcg = (1.0 / np.log2(np.arange(2, ideal + 2))).sum()
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
        n_eval += 1
    return {'P@K': float(np.mean(precs)), 'R@K': float(np.mean(recs)),
            'NDCG@K': float(np.mean(ndcgs)), 'usuarios': n_eval}


def baseline_scores(kind, n_pos, R, mask, items, seed=0):
    """random: puntajes uniformes; popular: mismo ranking global para todos."""
    rng = np.random.default_rng(seed)
    if kind == 'random':
        return rng.random((n_pos, len(items))).astype(np.float32)
    pop = mask.sum(0).astype(np.float32)
    return np.repeat(pop[None, :], n_pos, axis=0)


# ===================== Hibrido =====================
def normalize_rows(S):
    """min-max por fila para hacer comparables CF y content antes de mezclar."""
    S = np.where(np.isnan(S), -np.inf, S)
    finite = np.isfinite(S)
    lo = np.where(finite, S, np.inf).min(1, keepdims=True)
    hi = np.where(finite, S, -np.inf).max(1, keepdims=True)
    rng_ = np.maximum(hi - lo, 1e-8)
    out = np.where(finite, (S - lo) / rng_, 0.0)
    return out.astype(np.float32)


def hybrid_scores(cf_rows, cb_rows, alpha):
    return alpha * normalize_rows(cf_rows) + (1 - alpha) * normalize_rows(cb_rows)


def cold_start_analysis(train, test, iidx, uidx):
    tr_users = set(train['user_id'])
    cold_u = test.loc[~test['user_id'].isin(tr_users)]
    cold_i = test.loc[~test['business_id'].isin(iidx)]
    cnt = train.groupby('user_id').size()
    print('--- Analisis de cold-start ---')
    print(f'Pares de test con usuario nuevo (0 resenas en train): {len(cold_u):,}')
    print(f'Pares de test con negocio nuevo (0 resenas en train): {len(cold_i):,}')
    print(f'Usuarios de train con 1 sola resena: {(cnt == 1).sum():,} '
          f'({(cnt == 1).mean():.1%}) -> CF casi ciego para ellos')
    return cold_u, cold_i
