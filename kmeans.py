import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter


def _dists_sq(X, C):
    xx = (X * X).sum(1)[:, None]
    cc = (C * C).sum(1)[None, :]
    return np.maximum(xx + cc - 2 * X @ C.T, 0)


def _init_pp(X, k, rng):
    n = X.shape[0]
    first = int(rng.integers(n))
    centroids = [X[first]]
    d2 = ((X - X[first]) ** 2).sum(1)
    for _ in range(1, k):
        s = d2.sum()
        probs = d2 / s if s > 0 else np.full(n, 1.0 / n)
        idx = int(rng.choice(n, p=probs))
        centroids.append(X[idx])
        d2 = np.minimum(d2, ((X - X[idx]) ** 2).sum(1))
    return np.array(centroids)


def kmeans_pp(X, k, max_iter=100, tol=1e-4, seed=0, n_init=5):
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_init):
        C = _init_pp(X, k, rng)
        for _ in range(max_iter):
            labels = _dists_sq(X, C).argmin(1)
            newC = np.array([X[labels == j].mean(0) if np.any(labels == j) else C[j]
                             for j in range(k)])
            if np.sqrt(((newC - C) ** 2).sum()) < tol:
                C = newC
                break
            C = newC
        labels = _dists_sq(X, C).argmin(1)
        inertia = _dists_sq(X, C)[np.arange(len(X)), labels].sum()
        if best is None or inertia < best[2]:
            best = (labels, C, float(inertia))
    return best


def _pairwise_D(X):
    xx = (X * X).sum(1)
    return np.sqrt(np.maximum(xx[:, None] + xx[None, :] - 2 * X @ X.T, 0))


def silhouette_score(X, labels, D=None):
    if D is None:
        D = _pairwise_D(X)
    labels = np.asarray(labels)
    uniq = np.unique(labels)
    if len(uniq) < 2:
        return 0.0
    sil = np.zeros(len(labels))
    for c in uniq:
        idx = np.where(labels == c)[0]
        if len(idx) <= 1:
            continue
        others = [np.where(labels == o)[0] for o in uniq if o != c]
        for i in idx:
            a = D[i, idx[idx != i]].mean()
            b = min(D[i, o].mean() for o in others)
            sil[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
    return float(sil.mean())


def choose_k(X, k_min=2, k_max=10, seed=0):
    D = _pairwise_D(X)
    ks, inertias, sils, results = [], [], [], {}
    for k in range(k_min, k_max + 1):
        labels, C, inertia = kmeans_pp(X, k, seed=seed)
        s = silhouette_score(X, labels, D=D)
        ks.append(k); inertias.append(inertia); sils.append(s)
        results[k] = (labels, C, inertia, s)
        print(f'  k={k}: inercia={inertia:10.1f}  silueta={s:.4f}')
    return ks, inertias, sils, results


def plot_k_selection(ks, inertias, sils):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(ks, inertias, 'o-', color='#378ADD')
    ax[0].set_xlabel('k'); ax[0].set_ylabel('Inercia (SSE intra-cluster)')
    ax[0].set_title('Metodo del codo')
    ax[1].plot(ks, sils, 'o-', color='#D85A30')
    ax[1].set_xlabel('k'); ax[1].set_ylabel('Coeficiente de silueta')
    best = ks[int(np.argmax(sils))]
    ax[1].axvline(best, ls='--', color='gray', alpha=0.6)
    ax[1].set_title(f'Silueta por k (maxima en k={best})')
    plt.tight_layout(); plt.show()


def characterize_clusters(labels, biz_df):
    df = biz_df.copy(); df['cluster'] = labels
    rows = []
    for c in sorted(set(labels)):
        sub = df[df['cluster'] == c]
        cats = Counter()
        for lst in sub['categories_list']:
            cats.update(lst)
        rows.append({
            'cluster': c, 'n': len(sub),
            'stars': round(sub['stars'].mean(), 2),
            'reviews_med': int(sub['review_count'].median()),
            'price': round(sub['price_range_imp'].mean(), 2),
            'rating_std': round(sub['rating_std'].mean(), 2),
            'age_anios': round(sub['business_age_years'].mean(), 1),
            'pct_abierto': round(sub['is_open'].mean(), 2),
            'top_categorias': ', '.join(c for c, _ in cats.most_common(4)),
        })
    return pd.DataFrame(rows)
