import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def pca_2d(X):
    Xc = X - X.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    P = Xc @ Vt[:2].T
    var = (S[:2] ** 2) / (S ** 2).sum()
    return P, var


def compare_metrics(X, labels_km, labels_db, silhouette_fn):
    rows = []
    s_km = silhouette_fn(X, np.asarray(labels_km))
    rows.append({'metodo': 'K-Means++', 'n_clusters': len(set(np.asarray(labels_km).tolist())),
                 'outliers': 0, 'silueta': round(s_km, 4)})
    db = np.asarray(labels_db)
    mask = db != -1
    n_db = len(set(db.tolist()) - {-1})
    s_db = silhouette_fn(X[mask], db[mask]) if n_db >= 2 else float('nan')
    rows.append({'metodo': 'DBSCAN', 'n_clusters': n_db,
                 'outliers': int((~mask).sum()),
                 'silueta': round(s_db, 4) if n_db >= 2 else None})
    return pd.DataFrame(rows)


def plot_clusters_2d(P, labels_km, labels_db, var):
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
    km = np.asarray(labels_km)
    ax[0].scatter(P[:, 0], P[:, 1], c=km, cmap='tab10', s=6, alpha=0.6)
    ax[0].set_title(f'K-Means++  (k={len(set(km.tolist()))})')

    db = np.asarray(labels_db)
    out = db == -1
    ax[1].scatter(P[out, 0], P[out, 1], c='lightgray', s=6, alpha=0.5, label='outliers')
    ax[1].scatter(P[~out, 0], P[~out, 1], c=db[~out], cmap='tab10', s=6, alpha=0.6)
    ax[1].set_title(f'DBSCAN  ({len(set(db.tolist()) - {-1})} clusters + {int(out.sum())} outliers)')
    ax[1].legend(loc='upper right', fontsize=8)

    for a in ax:
        a.set_xlabel(f'PC1 ({var[0] * 100:.0f}% var)')
        a.set_ylabel(f'PC2 ({var[1] * 100:.0f}% var)')
    plt.tight_layout(); plt.show()
