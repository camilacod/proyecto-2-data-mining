import ast
import json
import pandas as pd
import config
import preprocessing


def _parse_categories(s):
    if not s or (isinstance(s, float) and pd.isna(s)):
        return []
    return [c.strip() for c in str(s).split(',') if c.strip()]


def _coerce_value(v):
    if v is None or isinstance(v, bool):
        return v
    s = str(v).strip()
    if s in ('None', ''):
        return None
    if s == 'True':
        return True
    if s == 'False':
        return False
    if s.startswith('{') and s.endswith('}'):
        try:
            return {k: _coerce_value(val) for k, val in ast.literal_eval(s).items()}
        except (ValueError, SyntaxError):
            return s
    if s.lstrip('-').isdigit():
        return int(s)
    return s


def _parse_attributes(s):
    if not s or (isinstance(s, float) and pd.isna(s)):
        return {}
    try:
        raw = json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return {}
    return {k: _coerce_value(v) for k, v in raw.items()}


def _parse_hours_count(s):
    if not s or (isinstance(s, float) and pd.isna(s)):
        return 0
    try:
        return len(json.loads(s))
    except (json.JSONDecodeError, TypeError):
        return 0


def clean_business(biz):
    rep, df, n0 = {}, biz.copy(), len(biz)
    df = df.drop_duplicates('business_id')
    rep['business_id duplicados eliminados'] = n0 - len(df)

    df['categories_list'] = df['categories'].apply(_parse_categories)
    df['n_categories'] = df['categories_list'].apply(len)
    rep['negocios sin categoria'] = int((df['n_categories'] == 0).sum())

    df['attributes_clean'] = df['attributes'].apply(_parse_attributes)
    df['price_range'] = df['attributes_clean'].apply(
        lambda d: d.get('RestaurantsPriceRange2') if isinstance(d, dict) else None)
    rep['negocios sin atributos'] = int((df['attributes_clean'].apply(len) == 0).sum())
    rep['negocios sin price_range'] = int(df['price_range'].isna().sum())

    df['n_days_open'] = df['hours'].apply(_parse_hours_count)
    rep['negocios sin horario'] = int((df['n_days_open'] == 0).sum())

    thr = df['review_count'].quantile(0.99)
    df['rc_outlier'] = df['review_count'] > thr
    rep[f'outliers review_count (>p99={thr:.0f}), marcados no eliminados'] = int(df['rc_outlier'].sum())
    return df, rep


def clean_reviews(rev):
    rep, df, n0 = {}, rev.copy(), len(rev)
    df = df.drop_duplicates('review_id')
    rep['review_id duplicados eliminados'] = n0 - len(df)

    n1 = len(df)
    df = df.sort_values('date').drop_duplicates(['user_id', 'business_id'], keep='last')
    rep['pares (user,business) repetidos colapsados al mas reciente'] = n1 - len(df)

    df['stars'] = pd.to_numeric(df['stars'], errors='coerce')
    rep['reviews con stars invalido'] = int(df['stars'].isna().sum())

    df['text'] = df['text'].fillna('')
    df['text_len'] = df['text'].str.len()
    rep['reviews con texto vacio'] = int((df['text_len'] == 0).sum())
    return df, rep


def clean_users(usr):
    rep, df, n0 = {}, usr.copy(), len(usr)
    df = df.drop_duplicates('user_id')
    rep['user_id duplicados eliminados'] = n0 - len(df)

    elite = df['elite'].fillna('').astype(str)
    df['n_elite_years'] = elite.apply(
        lambda s: 0 if s.strip() in ('', 'None') else len([y for y in s.split(',') if y.strip()]))
    df['is_elite'] = df['n_elite_years'] > 0
    rep['usuarios elite'] = int(df['is_elite'].sum())

    thr = df['review_count'].quantile(0.99)
    df['rc_outlier'] = df['review_count'] > thr
    rep[f'outliers review_count (>p99={thr:.0f}), marcados no eliminados'] = int(df['rc_outlier'].sum())
    return df, rep


def clean_checkins(chk):
    df = chk.copy()
    df['n_checkins'] = df['date'].fillna('').apply(
        lambda s: 0 if not s else len(str(s).split(',')))
    return df, {'total check-ins contados': int(df['n_checkins'].sum())}


def clean_tips(tips):
    df, n0 = tips.copy(), len(tips)
    df = df.drop_duplicates()
    df['text'] = df['text'].fillna('')
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    return df, {'tips duplicados eliminados': n0 - len(df)}


def clean_subset(data=None):
    if data is None:
        data = preprocessing.load_subset()

    biz, r1 = clean_business(data['business'])
    rev, r2 = clean_reviews(data['reviews'])
    usr, r3 = clean_users(data['users'])
    chk, r4 = clean_checkins(data['checkins'])
    tip, r5 = clean_tips(data['tips'])

    print(f'REPORTE DE LIMPIEZA -- {config.CITY_LABEL}')
    for title, rep in [('Negocios', r1), ('Resenas', r2), ('Usuarios', r3),
                       ('Check-ins', r4), ('Tips', r5)]:
        print(f'\n[{title}]')
        for k, v in rep.items():
            print(f'  - {k}: {v:,}' if isinstance(v, int) else f'  - {k}: {v}')

    print('\nFilas tras limpieza:')
    for k, df in [('business', biz), ('reviews', rev), ('users', usr),
                  ('checkins', chk), ('tips', tip)]:
        print(f'  {k:9s}: {len(df):,}')

    return {'business': biz, 'reviews': rev, 'users': usr,
            'checkins': chk, 'tips': tip}
