import numpy as np
import pandas as pd
from collections import Counter


def _behavioral(reviews):
    ref = reviews['date'].max()
    agg = reviews.groupby('business_id').agg(
        rating_std=('stars', 'std'),
        first_review=('date', 'min'),
        avg_review_length=('text_len', 'mean'),
    ).reset_index()
    agg['rating_std'] = agg['rating_std'].fillna(0.0)
    agg['business_age_years'] = (ref - agg['first_review']).dt.days / 365.25
    return agg[['business_id', 'rating_std', 'business_age_years', 'avg_review_length']]


def build_business_features(clean, top_categories=18, cat_weight=1.0):
    biz = clean['business'].copy().reset_index(drop=True)

    chk = clean['checkins']
    if 'n_checkins' in chk.columns:
        biz = biz.merge(chk[['business_id', 'n_checkins']], on='business_id', how='left')
    if 'n_checkins' not in biz.columns:
        biz['n_checkins'] = 0
    biz['n_checkins'] = biz['n_checkins'].fillna(0)

    biz = biz.merge(_behavioral(clean['reviews']), on='business_id', how='left')
    for col in ('rating_std', 'business_age_years', 'avg_review_length'):
        biz[col] = biz[col].fillna(0.0)

    pr_med = biz['price_range'].median()
    biz['price_range_imp'] = biz['price_range'].fillna(pr_med)

    cont = pd.DataFrame({
        'stars':         biz['stars'].astype(float),
        'log_reviews':   np.log1p(biz['review_count'].astype(float)),
        'price_range':   biz['price_range_imp'].astype(float),
        'n_categories':  biz['n_categories'].astype(float),
        'n_days_open':   biz['n_days_open'].astype(float),
        'log_checkins':  np.log1p(biz['n_checkins'].astype(float)),
        'is_open':       biz['is_open'].astype(float),
        'rating_std':    biz['rating_std'].astype(float),
        'business_age':  biz['business_age_years'].astype(float),
        'avg_rev_len':   biz['avg_review_length'].astype(float),
    })

    cat_counts = Counter()
    for lst in biz['categories_list']:
        cat_counts.update(lst)
    top = [c for c, _ in cat_counts.most_common(top_categories)]
    cat_mat = pd.DataFrame(
        {f'cat::{c}': biz['categories_list'].apply(lambda l, c=c: 1.0 if c in l else 0.0)
         for c in top}
    )

    feat = pd.concat([cont, cat_mat], axis=1)
    names = list(feat.columns)

    X = feat.to_numpy(dtype=float)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    Xz = (X - mu) / sd

    if cat_weight != 1.0:
        cat_cols = [i for i, n in enumerate(names) if n.startswith('cat::')]
        Xz[:, cat_cols] *= cat_weight

    return Xz, names, biz['business_id'].to_numpy(), biz
