import os
import json
import pandas as pd
import config

_BIZ_FIELDS = ['business_id', 'name', 'address', 'city', 'state', 'postal_code',
               'latitude', 'longitude', 'stars', 'review_count', 'is_open',
               'attributes', 'categories', 'hours']
_REV_FIELDS = ['review_id', 'user_id', 'business_id', 'stars', 'useful', 'funny',
               'cool', 'text', 'date']
_USER_FIELDS = ['user_id', 'name', 'review_count', 'yelping_since', 'useful',
                'funny', 'cool', 'fans', 'elite', 'average_stars']
_TIP_FIELDS = ['user_id', 'business_id', 'text', 'date', 'compliment_count']

P_BIZ      = os.path.join(config.ARTIFACTS, 'sample_business.parquet')
P_REVIEWS  = os.path.join(config.ARTIFACTS, 'sample_reviews.parquet')
P_USERS    = os.path.join(config.ARTIFACTS, 'sample_users.parquet')
P_CHECKINS = os.path.join(config.ARTIFACTS, 'sample_checkins.parquet')
P_TIPS     = os.path.join(config.ARTIFACTS, 'sample_tips.parquet')


def _stream(path):
    with open(path) as fh:
        for line in fh:
            yield json.loads(line)


def _stratified_sample(df):
    """Muestreo estratificado proporcional por (ciudad, estado): cada estrato
    aporta negocios segun su peso en la poblacion, con semilla fija. Las
    ciudades muy chicas pueden aportar 0 (asignacion proporcional pura)."""
    frac = min(1.0, config.SAMPLE_TARGET_BUSINESS / len(df))
    strata = (df['city'].fillna('').str.strip().str.title() + ' | '
              + df['state'].fillna(''))
    sample = (df.groupby(strata, group_keys=False)
                .apply(lambda g: g.sample(frac=frac,
                                          random_state=config.SAMPLE_SEED)))
    _sample_report(df, sample, strata)
    return sample.reset_index(drop=True)


def _sample_report(full, sample, strata):
    """Documenta la representatividad: proporciones por ciudad y medias de
    variables clave en poblacion vs muestra."""
    print(f'     Muestreo estratificado: {len(sample):,} de {len(full):,} negocios '
          f'({len(sample) / len(full):.1%}), semilla={config.SAMPLE_SEED}')
    print('     Representatividad -- top ciudades (poblacion vs muestra):')
    s_strata = strata.loc[sample.index]
    for ciudad, n in strata.value_counts().head(8).items():
        pf = n / len(full)
        ps = (s_strata == ciudad).mean()
        print(f'       {ciudad:<30s} {pf:6.2%} vs {ps:6.2%}')
    print('     Representatividad -- medias (poblacion vs muestra):')
    for col in ('stars', 'review_count', 'is_open'):
        print(f'       {col:<13s} {full[col].mean():8.2f} vs {sample[col].mean():8.2f}')


def _business():
    rows = [{k: o.get(k) for k in _BIZ_FIELDS} for o in _stream(config.BUSINESS)]
    df = pd.DataFrame(rows, columns=_BIZ_FIELDS)
    df = _stratified_sample(df)
    for col in ('attributes', 'hours'):
        df[col] = df[col].apply(
            lambda d: json.dumps(d, ensure_ascii=False) if d is not None else None)
    return df


def _reviews(biz_ids):
    rows, n = [], 0
    for o in _stream(config.REVIEW):
        if o.get('business_id') in biz_ids:
            rows.append({k: o.get(k) for k in _REV_FIELDS})
        n += 1
        if n % 1_000_000 == 0:
            print(f'     ...{n:,} resenas escaneadas')
    print(f'     resenas escaneadas en total: {n:,}')
    df = pd.DataFrame(rows, columns=_REV_FIELDS)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    return df


def _users(user_ids):
    rows, n = [], 0
    for o in _stream(config.USER):
        n += 1
        uid = o.get('user_id')
        if uid not in user_ids:
            continue
        rec = {k: o.get(k) for k in _USER_FIELDS}
        raw = (o.get('friends') or '').strip()
        all_friends = [] if raw in ('', 'None') else [f.strip() for f in raw.split(',')]
        subset_friends = [f for f in all_friends if f in user_ids]
        rec['n_friends_total'] = len(all_friends)
        rec['n_friends_subset'] = len(subset_friends)
        rec['friends_subset'] = ','.join(subset_friends)
        rows.append(rec)
        if n % 1_000_000 == 0:
            print(f'     ...{n:,} usuarios escaneados')
    cols = _USER_FIELDS + ['n_friends_total', 'n_friends_subset', 'friends_subset']
    df = pd.DataFrame(rows, columns=cols)
    df['yelping_since'] = pd.to_datetime(df['yelping_since'], errors='coerce')
    return df


def _filter_by_business(path, biz_ids, fields):
    rows = []
    for o in _stream(path):
        if o.get('business_id') in biz_ids:
            rows.append({k: o.get(k) for k in fields})
    return pd.DataFrame(rows, columns=fields)


def build_subset(rebuild=False):
    print(f'Construyendo el subconjunto de {config.CITY_LABEL}'
          f'{"  (rebuild=True)" if rebuild else ""}\n')

    if not rebuild and os.path.exists(P_BIZ):
        biz = pd.read_parquet(P_BIZ)
        print(f'1/5  Negocios: cache ({len(biz):,})')
    else:
        print('1/5  Negocios...')
        biz = _business()
        biz.to_parquet(P_BIZ, index=False)
        print(f'     {len(biz):,} negocios -> {P_BIZ}')
    biz_ids = set(biz['business_id'])

    if not rebuild and os.path.exists(P_REVIEWS):
        rev = pd.read_parquet(P_REVIEWS)
        print(f'2/5  Resenas: cache ({len(rev):,})')
    else:
        print('2/5  Resenas (streaming review.json, ~2 min)...')
        rev = _reviews(biz_ids)
        rev.to_parquet(P_REVIEWS, index=False)
        print(f'     {len(rev):,} resenas -> {P_REVIEWS}')
    user_ids = set(rev['user_id'])

    if not rebuild and os.path.exists(P_USERS):
        usr = pd.read_parquet(P_USERS)
        print(f'3/5  Usuarios: cache ({len(usr):,})')
    else:
        print('3/5  Usuarios (streaming user.json)...')
        usr = _users(user_ids)
        usr.to_parquet(P_USERS, index=False)
        print(f'     {len(usr):,} usuarios -> {P_USERS}')

    if not rebuild and os.path.exists(P_CHECKINS):
        print('4/5  Check-ins: cache')
    else:
        print('4/5  Check-ins...')
        chk = _filter_by_business(config.CHECKIN, biz_ids, ['business_id', 'date'])
        chk.to_parquet(P_CHECKINS, index=False)
        print(f'     {len(chk):,} negocios con check-ins -> {P_CHECKINS}')

    if not rebuild and os.path.exists(P_TIPS):
        print('5/5  Tips: cache')
    else:
        print('5/5  Tips...')
        tips = _filter_by_business(config.TIP, biz_ids, _TIP_FIELDS)
        tips.to_parquet(P_TIPS, index=False)
        print(f'     {len(tips):,} tips -> {P_TIPS}')

    print(f'\nListo. Subconjunto ({config.CITY_LABEL}) en {config.ARTIFACTS}')
    return load_subset()


def load_subset():
    return {
        'business': pd.read_parquet(P_BIZ),
        'reviews':  pd.read_parquet(P_REVIEWS),
        'users':    pd.read_parquet(P_USERS),
        'checkins': pd.read_parquet(P_CHECKINS),
        'tips':     pd.read_parquet(P_TIPS),
    }


if __name__ == '__main__':
    build_subset()
