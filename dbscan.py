import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _pairwise_D(X):
    xx = (X * X).sum(1)
    return np.sqrt(np.maximum(xx[:, None] + xx[None, :] - 2 * X @ X.T, 0))


def k_distance(X, k, D=None):
    if D is None:
        D = _pairwise_D(X)
    kd = np.sort(D, axis=1)[:, k]
    return np.sort(kd)


def plot_k_distance(kd, k):
    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(len(kd)), kd, color='#378ADD')
    plt.xlabel('Negocios (ordenados por distancia)')
    plt.ylabel(f'Distancia al vecino #{k}')
    plt.title(f'k-distance plot (k={k}) — el "codo" sugiere eps')
    plt.grid(alpha=0.3)
    plt.tight_layout(); plt.show()


def dbscan(X, eps, min_pts, D=None):
    n = len(X)
    if D is None:
        D = _pairwise_D(X)
    neighbors = [np.where(D[i] <= eps)[0] for i in range(n)]
    labels = np.full(n, -1, dtype=int)
    visited = np.zeros(n, dtype=bool)
    cid = 0
    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        if len(neighbors[i]) < min_pts:
            continue
        labels[i] = cid
        seeds = list(neighbors[i])
        j = 0
        while j < len(seeds):
            q = seeds[j]; j += 1
            if not visited[q]:
                visited[q] = True
                if len(neighbors[q]) >= min_pts:
                    seeds.extend(neighbors[q].tolist())
            if labels[q] == -1:
                labels[q] = cid
        cid += 1
    return labels


def dbscan_summary(labels):
    n_noise = int((labels == -1).sum())
    clusters = sorted(c for c in set(labels.tolist()) if c != -1)
    rows = [{'cluster': c, 'n': int((labels == c).sum())} for c in clusters]
    rows.append({'cluster': 'outliers (-1)', 'n': n_noise})
    print(f'{len(clusters)} clusters + {n_noise} outliers '
          f'({100 * n_noise / len(labels):.1f}% del total)')
    return pd.DataFrame(rows)
