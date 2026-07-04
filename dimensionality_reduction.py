import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ===================== PCA desde cero =====================
def pca_fit(Xz):
    """Xz ya estandarizada. Covarianza -> eigendescomposicion (eigh porque la
    matriz es simetrica) -> componentes ordenadas por varianza descendente."""
    n = Xz.shape[0]
    cov = (Xz.T @ Xz) / (n - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order], 0.0)
    eigvecs = eigvecs[:, order]
    evr = eigvals / eigvals.sum()
    return eigvals, eigvecs, evr


def n_components_for(evr, target=0.90):
    cum = np.cumsum(evr)
    k = int(np.searchsorted(cum, target) + 1)
    print(f'{k} componentes explican {cum[k-1]:.1%} de la varianza '
          f'(objetivo {target:.0%}, de {len(evr)} dims originales)')
    return k


def plot_scree(evr, target=0.90):
    cum = np.cumsum(evr)
    k = int(np.searchsorted(cum, target) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].bar(range(1, len(evr) + 1), evr, color='#378ADD')
    ax[0].set_xlabel('componente'); ax[0].set_ylabel('varianza explicada')
    ax[0].set_title('Scree plot')
    ax[1].plot(range(1, len(evr) + 1), cum, 'o-', color='#D85A30', ms=3)
    ax[1].axhline(target, ls='--', color='gray', alpha=0.7)
    ax[1].axvline(k, ls='--', color='gray', alpha=0.7)
    ax[1].set_xlabel('componentes'); ax[1].set_ylabel('varianza acumulada')
    ax[1].set_title(f'{k} PCs alcanzan {target:.0%}')
    plt.tight_layout(); plt.show()
    return k


def pca_project(Xz, eigvecs, k):
    return Xz @ eigvecs[:, :k]


def loadings_table(eigvecs, evr, feat_names, n_pc=4, top=6):
    """Variables que mas pesan (|loading|) en cada componente principal."""
    rows = []
    for j in range(n_pc):
        v = eigvecs[:, j]
        idx = np.argsort(-np.abs(v))[:top]
        rows.append({
            'PC': f'PC{j+1}',
            'var_explicada': f'{evr[j]:.1%}',
            'top_variables': ', '.join(f'{feat_names[i]} ({v[i]:+.2f})' for i in idx),
        })
    return pd.DataFrame(rows)


def plot_projection(P, color, color_label, evr):
    fig = plt.figure(figsize=(13, 5.5))
    ax0 = fig.add_subplot(1, 2, 1)
    sc = ax0.scatter(P[:, 0], P[:, 1], c=color, cmap='viridis', s=7, alpha=0.6)
    plt.colorbar(sc, ax=ax0, label=color_label)
    ax0.set_xlabel(f'PC1 ({evr[0]:.0%})'); ax0.set_ylabel(f'PC2 ({evr[1]:.0%})')
    ax0.set_title('Proyeccion PCA 2D')
    ax1 = fig.add_subplot(1, 2, 2, projection='3d')
    ax1.scatter(P[:, 0], P[:, 1], P[:, 2], c=color, cmap='viridis', s=5, alpha=0.5)
    ax1.set_xlabel('PC1'); ax1.set_ylabel('PC2'); ax1.set_zlabel('PC3')
    ax1.set_title('Proyeccion PCA 3D')
    plt.tight_layout(); plt.show()


# ===================== SVD sobre TF-IDF =====================
def svd_fit(M):
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    print(f'SVD: M {M.shape[0]:,}x{M.shape[1]:,} -> '
          f'{len(S)} valores singulares (s1={S[0]:.2f}, s2={S[1]:.2f}, ...)')
    return U, S, Vt


def latent_factors(Vt, vocab, n_factors=6, top=8):
    """Cada fila de Vt es una direccion latente en el espacio de terminos:
    los terminos con mayor |peso| revelan el 'tema' del factor."""
    rows = []
    for j in range(n_factors):
        v = Vt[j]
        idx = np.argsort(-np.abs(v))[:top]
        rows.append({'factor': j + 1,
                     'terminos_dominantes': ', '.join(vocab[i] for i in idx)})
    return pd.DataFrame(rows)


def reconstruction_curve(M, U, S, Vt, ks):
    """Error relativo de Frobenius al truncar a k factores. Por Eckart-Young la
    SVD truncada es la mejor aproximacion de rango k posible."""
    fro2 = float((S ** 2).sum())
    rows = []
    n, m = M.shape
    full_cells = n * m
    for k in ks:
        err = float(np.sqrt((S[k:] ** 2).sum() / fro2))
        stored = k * (n + m + 1)
        rows.append({'k': k,
                     'error_relativo': round(err, 4),
                     'varianza_capturada': round(1 - err ** 2, 4),
                     'celdas_almacenadas': stored,
                     'compresion': f'{full_cells / stored:.0f}x'})
    return pd.DataFrame(rows)


def plot_reconstruction(rec):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(rec['k'], rec['error_relativo'], 'o-', color='#378ADD')
    ax[0].set_xlabel('k (factores retenidos)'); ax[0].set_ylabel('error relativo ||M-Mk||/||M||')
    ax[0].set_title('Error de reconstruccion vs k')
    ratio = [float(str(c).rstrip('x')) for c in rec['compresion']]
    ax[1].plot(rec['k'], ratio, 'o-', color='#D85A30')
    ax[1].set_yscale('log')
    ax[1].set_xlabel('k'); ax[1].set_ylabel('factor de compresion (log)')
    ax[1].set_title('Compresion lograda vs k')
    plt.tight_layout(); plt.show()


from collections import Counter

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


def build_tfidf(reviews, items, max_reviews_per_item=50, vocab_size=2000):
    """Documento = negocio (concatenacion de hasta 50 resenas).
    TF = frecuencia relativa en el doc, IDF = log(N / (1+df))."""
    by_item = reviews.groupby('business_id')['text']
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

    M = np.zeros((N, len(vocab)), dtype=np.float32)
    for i, b in enumerate(items):
        cnt = docs_tokens[b]
        total = sum(cnt.values()) or 1
        for w, c in cnt.items():
            j = widx.get(w)
            if j is not None:
                M[i, j] = (c / total) * idf[j]
    print(f'TF-IDF: {N:,} negocios x {len(vocab):,} terminos')
    return M, vocab
