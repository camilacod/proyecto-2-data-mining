from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import config


def plot_rating_distribution(clean):
    rev = clean['reviews']
    counts = rev['stars'].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(counts.index, counts.values, width=0.6, color='#378ADD')
    ax.set_xlabel('Estrellas'); ax.set_ylabel('Nro de resenas')
    ax.set_title(f'Distribucion de ratings -- {config.CITY_LABEL}')
    for x, y in zip(counts.index, counts.values):
        ax.text(x, y, f'{y:,}', ha='center', va='bottom', fontsize=8)
    plt.tight_layout(); plt.show()
    print(f'Rating promedio: {rev["stars"].mean():.2f} (sesgo a 4-5, tipico de Yelp)')


def plot_review_length(clean):
    rev = clean['reviews']
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(rev['text_len'].clip(upper=2000), bins=50, color='#1D9E75')
    ax.set_xlabel('Longitud de resena (caracteres, recortado a 2000)')
    ax.set_ylabel('Nro de resenas')
    ax.set_title('Distribucion de longitud de resenas')
    plt.tight_layout(); plt.show()
    print(f'Mediana de longitud: {rev["text_len"].median():.0f} caracteres')


def _loglog(series, ax, xlabel, color):
    vc = series.value_counts().sort_index()
    vc = vc[vc.index > 0]
    ax.scatter(vc.index, vc.values, s=12, color=color, alpha=0.6)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(xlabel); ax.set_ylabel('Frecuencia (nro de elementos)')


def plot_powerlaws(clean):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    _loglog(clean['business']['review_count'], axes[0],
            'Resenas por negocio', '#D85A30')
    axes[0].set_title('Power-law: actividad de negocios')
    per_user = clean['reviews']['user_id'].value_counts()
    _loglog(per_user, axes[1], 'Resenas por usuario (en la muestra)', '#534AB7')
    axes[1].set_title('Power-law: actividad de usuarios')
    plt.tight_layout(); plt.show()
    print('Ambas siguen ley de potencias: pocos muy activos, larga cola de casuales.')


def plot_top_categories(clean, top=15):
    cats = Counter()
    for lst in clean['business']['categories_list']:
        cats.update(lst)
    common = pd.Series(dict(cats.most_common(top))).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(common.index, common.values, color='#378ADD')
    ax.set_xlabel('Nro de negocios')
    ax.set_title(f'Top {top} categorias -- {config.CITY_LABEL}')
    plt.tight_layout(); plt.show()


def plot_geographic(clean):
    biz = clean['business']
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(biz['longitude'], biz['latitude'], c=biz['stars'],
                    cmap='viridis', s=14, alpha=0.6)
    ax.set_xlabel('Longitud'); ax.set_ylabel('Latitud')
    ax.set_title(f'Negocios de {config.CITY_LABEL} (color = estrellas)')
    fig.colorbar(sc, ax=ax, label='Estrellas')
    plt.tight_layout(); plt.show()


def plot_temporal(clean):
    rev = clean['reviews'].dropna(subset=['date'])
    monthly = rev.set_index('date').resample('MS').size()
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(monthly.index, monthly.values, color='#0F6E56')
    ax.set_xlabel('Fecha'); ax.set_ylabel('Resenas por mes')
    ax.set_title('Evolucion temporal de resenas')
    plt.tight_layout(); plt.show()


def plot_friendship_degree(clean):
    deg = clean['users']['n_friends_subset']
    fig, ax = plt.subplots(figsize=(6, 4))
    _loglog(deg[deg > 0], ax, 'Amigos dentro de la muestra (grado)', '#993556')
    ax.set_title('Distribucion de grados del grafo de amistad')
    plt.tight_layout(); plt.show()
    print(f'{(deg == 0).mean():.1%} de usuarios sin amigos en la muestra; '
          f'el resto forma el nucleo social de la Parte II.')


def plot_segments(clean):
    biz, usr = clean['business'], clean['users']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    by_open = biz.groupby('is_open')['stars'].mean()
    axes[0].bar(['Cerrado', 'Abierto'],
                [by_open.get(0, np.nan), by_open.get(1, np.nan)],
                color=['#888780', '#1D9E75'])
    axes[0].set_ylabel('Estrellas promedio')
    axes[0].set_title('Rating: abierto vs cerrado')
    by_elite = usr.groupby('is_elite')['average_stars'].mean()
    axes[1].bar(['No elite', 'Elite'],
                [by_elite.get(False, np.nan), by_elite.get(True, np.nan)],
                color=['#888780', '#534AB7'])
    axes[1].set_ylabel('Estrellas promedio dadas')
    axes[1].set_title('Comportamiento: elite vs no elite')
    plt.tight_layout(); plt.show()


def plot_all(clean):
    plot_rating_distribution(clean)
    plot_review_length(clean)
    plot_powerlaws(clean)
    plot_top_categories(clean)
    plot_geographic(clean)
    plot_temporal(clean)
    plot_friendship_degree(clean)
    plot_segments(clean)
