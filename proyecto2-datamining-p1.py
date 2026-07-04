# Databricks notebook source
# MAGIC %md
# MAGIC # Proyecto 2 -- Data Mining sobre el Yelp Open Dataset
# MAGIC
# MAGIC **Parte I: Preprocesamiento, limpieza, EDA y construccion de grafos.**
# MAGIC
# MAGIC Se trabaja con una **muestra estratificada por ciudad** del Yelp Open Dataset (~3,000 negocios: cada ciudad aporta en proporcion a su peso en la poblacion, con semilla fija para reproducibilidad). Esta parte descarga el dataset via `kagglehub`, streamea los JSON crudos, limpia las cinco tablas (negocios, resenas, usuarios, check-ins y tips) y deja los parquet en `artifacts/`, de donde se alimentan las Partes II-VI.

# COMMAND ----------

# Descarga del Yelp Open Dataset (~4.35 GB) via kagglehub; queda cacheado en el
# driver. Credenciales: variables de entorno del cluster o secretos de Databricks.
import os

if not (os.environ.get('KAGGLE_USERNAME') and os.environ.get('KAGGLE_KEY')):
    os.environ['KAGGLE_USERNAME'] = dbutils.secrets.get(scope='proyecto-dm', key='kaggle_username')
    os.environ['KAGGLE_KEY'] = dbutils.secrets.get(scope='proyecto-dm', key='kaggle_key')

import importlib.util, subprocess, sys
if importlib.util.find_spec('kagglehub') is None:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'kagglehub'])

import kagglehub
path = kagglehub.dataset_download('yelp-dataset/yelp-dataset')
print('Path to dataset files:', path)
print('Archivos:', sorted(os.listdir(path)))

# COMMAND ----------

# MAGIC %%writefile config.py
# MAGIC
# MAGIC import os
# MAGIC import glob
# MAGIC
# MAGIC # Alcance de los datos: todas las ciudades, muestreo estratificado por ciudad.
# MAGIC SCOPE = "sample"
# MAGIC
# MAGIC # parametros del muestreo estratificado
# MAGIC SAMPLE_TARGET_BUSINESS = 3000   # tamano objetivo de la muestra de negocios
# MAGIC SAMPLE_SEED = 42                # semilla fija -> muestra reproducible
# MAGIC
# MAGIC CITY_LABEL = "Yelp global (muestra estratificada por ciudad)"
# MAGIC
# MAGIC def _kagglehub_root():
# MAGIC     """Ruta local del dataset descargado con kagglehub. Primero busca el cache
# MAGIC     en disco (no requiere red ni credenciales); solo si no existe llama a la
# MAGIC     API de Kaggle, para lo cual deben estar KAGGLE_USERNAME / KAGGLE_KEY
# MAGIC     (ver la celda de descarga del notebook de la Parte I)."""
# MAGIC     cache_base = os.environ.get(
# MAGIC         "KAGGLEHUB_CACHE", os.path.expanduser("~/.cache/kagglehub"))
# MAGIC     versions = os.path.join(
# MAGIC         cache_base, "datasets", "yelp-dataset", "yelp-dataset", "versions")
# MAGIC     if os.path.isdir(versions):
# MAGIC         found = sorted(glob.glob(os.path.join(versions, "*")))
# MAGIC         if found and glob.glob(os.path.join(found[-1], "*.json")):
# MAGIC             return found[-1]
# MAGIC     import kagglehub
# MAGIC     return kagglehub.dataset_download("yelp-dataset/yelp-dataset")
# MAGIC
# MAGIC
# MAGIC # dataset via kagglehub (cache local primero)
# MAGIC DATA_ROOT = _kagglehub_root()
# MAGIC
# MAGIC def _find_raw(keyword):
# MAGIC     m = glob.glob(os.path.join(DATA_ROOT, f"**/*{keyword}*.json"), recursive=True)
# MAGIC     return m[0] if m else None
# MAGIC
# MAGIC BUSINESS = _find_raw("business")
# MAGIC REVIEW   = _find_raw("review")
# MAGIC USER     = _find_raw("user")
# MAGIC CHECKIN  = _find_raw("checkin")
# MAGIC TIP      = _find_raw("tip")
# MAGIC
# MAGIC def _locate_artifacts():
# MAGIC     # Carpeta 'artifacts' junto al notebook. Con DBR 14+ el cwd es la carpeta
# MAGIC     # del Workspace, asi que persiste entre reinicios del cluster y los
# MAGIC     # notebooks de las Partes II-VI la encuentran sin adjuntar nada.
# MAGIC     local = os.path.join(os.getcwd(), "artifacts")
# MAGIC     os.makedirs(local, exist_ok=True)
# MAGIC     return local
# MAGIC
# MAGIC ARTIFACTS = _locate_artifacts()
# MAGIC
# MAGIC MIN_USERS = 3000
# MAGIC MIN_LCC_FRACTION = 0.20
# MAGIC MIN_AVG_DEGREE = 1.0

# COMMAND ----------

# MAGIC %md
# MAGIC # PARTE 1

# COMMAND ----------

# MAGIC %%writefile preprocessing.py
# MAGIC import os
# MAGIC import json
# MAGIC import pandas as pd
# MAGIC import config
# MAGIC
# MAGIC _BIZ_FIELDS = ['business_id', 'name', 'address', 'city', 'state', 'postal_code',
# MAGIC                'latitude', 'longitude', 'stars', 'review_count', 'is_open',
# MAGIC                'attributes', 'categories', 'hours']
# MAGIC _REV_FIELDS = ['review_id', 'user_id', 'business_id', 'stars', 'useful', 'funny',
# MAGIC                'cool', 'text', 'date']
# MAGIC _USER_FIELDS = ['user_id', 'name', 'review_count', 'yelping_since', 'useful',
# MAGIC                 'funny', 'cool', 'fans', 'elite', 'average_stars']
# MAGIC _TIP_FIELDS = ['user_id', 'business_id', 'text', 'date', 'compliment_count']
# MAGIC
# MAGIC P_BIZ      = os.path.join(config.ARTIFACTS, 'sample_business.parquet')
# MAGIC P_REVIEWS  = os.path.join(config.ARTIFACTS, 'sample_reviews.parquet')
# MAGIC P_USERS    = os.path.join(config.ARTIFACTS, 'sample_users.parquet')
# MAGIC P_CHECKINS = os.path.join(config.ARTIFACTS, 'sample_checkins.parquet')
# MAGIC P_TIPS     = os.path.join(config.ARTIFACTS, 'sample_tips.parquet')
# MAGIC
# MAGIC
# MAGIC def _stream(path):
# MAGIC     with open(path) as fh:
# MAGIC         for line in fh:
# MAGIC             yield json.loads(line)
# MAGIC
# MAGIC
# MAGIC def _stratified_sample(df):
# MAGIC     """Muestreo estratificado proporcional por (ciudad, estado): cada estrato
# MAGIC     aporta negocios segun su peso en la poblacion, con semilla fija. Las
# MAGIC     ciudades muy chicas pueden aportar 0 (asignacion proporcional pura)."""
# MAGIC     frac = min(1.0, config.SAMPLE_TARGET_BUSINESS / len(df))
# MAGIC     strata = (df['city'].fillna('').str.strip().str.title() + ' | '
# MAGIC               + df['state'].fillna(''))
# MAGIC     sample = (df.groupby(strata, group_keys=False)
# MAGIC                 .apply(lambda g: g.sample(frac=frac,
# MAGIC                                           random_state=config.SAMPLE_SEED)))
# MAGIC     _sample_report(df, sample, strata)
# MAGIC     return sample.reset_index(drop=True)
# MAGIC
# MAGIC
# MAGIC def _sample_report(full, sample, strata):
# MAGIC     """Documenta la representatividad: proporciones por ciudad y medias de
# MAGIC     variables clave en poblacion vs muestra."""
# MAGIC     print(f'     Muestreo estratificado: {len(sample):,} de {len(full):,} negocios '
# MAGIC           f'({len(sample) / len(full):.1%}), semilla={config.SAMPLE_SEED}')
# MAGIC     print('     Representatividad -- top ciudades (poblacion vs muestra):')
# MAGIC     s_strata = strata.loc[sample.index]
# MAGIC     for ciudad, n in strata.value_counts().head(8).items():
# MAGIC         pf = n / len(full)
# MAGIC         ps = (s_strata == ciudad).mean()
# MAGIC         print(f'       {ciudad:<30s} {pf:6.2%} vs {ps:6.2%}')
# MAGIC     print('     Representatividad -- medias (poblacion vs muestra):')
# MAGIC     for col in ('stars', 'review_count', 'is_open'):
# MAGIC         print(f'       {col:<13s} {full[col].mean():8.2f} vs {sample[col].mean():8.2f}')
# MAGIC
# MAGIC
# MAGIC def _business():
# MAGIC     rows = [{k: o.get(k) for k in _BIZ_FIELDS} for o in _stream(config.BUSINESS)]
# MAGIC     df = pd.DataFrame(rows, columns=_BIZ_FIELDS)
# MAGIC     df = _stratified_sample(df)
# MAGIC     for col in ('attributes', 'hours'):
# MAGIC         df[col] = df[col].apply(
# MAGIC             lambda d: json.dumps(d, ensure_ascii=False) if d is not None else None)
# MAGIC     return df
# MAGIC
# MAGIC
# MAGIC def _reviews(biz_ids):
# MAGIC     rows, n = [], 0
# MAGIC     for o in _stream(config.REVIEW):
# MAGIC         if o.get('business_id') in biz_ids:
# MAGIC             rows.append({k: o.get(k) for k in _REV_FIELDS})
# MAGIC         n += 1
# MAGIC         if n % 1_000_000 == 0:
# MAGIC             print(f'     ...{n:,} resenas escaneadas')
# MAGIC     print(f'     resenas escaneadas en total: {n:,}')
# MAGIC     df = pd.DataFrame(rows, columns=_REV_FIELDS)
# MAGIC     df['date'] = pd.to_datetime(df['date'], errors='coerce')
# MAGIC     return df
# MAGIC
# MAGIC
# MAGIC def _users(user_ids):
# MAGIC     rows, n = [], 0
# MAGIC     for o in _stream(config.USER):
# MAGIC         n += 1
# MAGIC         uid = o.get('user_id')
# MAGIC         if uid not in user_ids:
# MAGIC             continue
# MAGIC         rec = {k: o.get(k) for k in _USER_FIELDS}
# MAGIC         raw = (o.get('friends') or '').strip()
# MAGIC         all_friends = [] if raw in ('', 'None') else [f.strip() for f in raw.split(',')]
# MAGIC         subset_friends = [f for f in all_friends if f in user_ids]
# MAGIC         rec['n_friends_total'] = len(all_friends)
# MAGIC         rec['n_friends_subset'] = len(subset_friends)
# MAGIC         rec['friends_subset'] = ','.join(subset_friends)
# MAGIC         rows.append(rec)
# MAGIC         if n % 1_000_000 == 0:
# MAGIC             print(f'     ...{n:,} usuarios escaneados')
# MAGIC     cols = _USER_FIELDS + ['n_friends_total', 'n_friends_subset', 'friends_subset']
# MAGIC     df = pd.DataFrame(rows, columns=cols)
# MAGIC     df['yelping_since'] = pd.to_datetime(df['yelping_since'], errors='coerce')
# MAGIC     return df
# MAGIC
# MAGIC
# MAGIC def _filter_by_business(path, biz_ids, fields):
# MAGIC     rows = []
# MAGIC     for o in _stream(path):
# MAGIC         if o.get('business_id') in biz_ids:
# MAGIC             rows.append({k: o.get(k) for k in fields})
# MAGIC     return pd.DataFrame(rows, columns=fields)
# MAGIC
# MAGIC
# MAGIC def build_subset(rebuild=False):
# MAGIC     print(f'Construyendo el subconjunto de {config.CITY_LABEL}'
# MAGIC           f'{"  (rebuild=True)" if rebuild else ""}\n')
# MAGIC
# MAGIC     if not rebuild and os.path.exists(P_BIZ):
# MAGIC         biz = pd.read_parquet(P_BIZ)
# MAGIC         print(f'1/5  Negocios: cache ({len(biz):,})')
# MAGIC     else:
# MAGIC         print('1/5  Negocios...')
# MAGIC         biz = _business()
# MAGIC         biz.to_parquet(P_BIZ, index=False)
# MAGIC         print(f'     {len(biz):,} negocios -> {P_BIZ}')
# MAGIC     biz_ids = set(biz['business_id'])
# MAGIC
# MAGIC     if not rebuild and os.path.exists(P_REVIEWS):
# MAGIC         rev = pd.read_parquet(P_REVIEWS)
# MAGIC         print(f'2/5  Resenas: cache ({len(rev):,})')
# MAGIC     else:
# MAGIC         print('2/5  Resenas (streaming review.json, ~2 min)...')
# MAGIC         rev = _reviews(biz_ids)
# MAGIC         rev.to_parquet(P_REVIEWS, index=False)
# MAGIC         print(f'     {len(rev):,} resenas -> {P_REVIEWS}')
# MAGIC     user_ids = set(rev['user_id'])
# MAGIC
# MAGIC     if not rebuild and os.path.exists(P_USERS):
# MAGIC         usr = pd.read_parquet(P_USERS)
# MAGIC         print(f'3/5  Usuarios: cache ({len(usr):,})')
# MAGIC     else:
# MAGIC         print('3/5  Usuarios (streaming user.json)...')
# MAGIC         usr = _users(user_ids)
# MAGIC         usr.to_parquet(P_USERS, index=False)
# MAGIC         print(f'     {len(usr):,} usuarios -> {P_USERS}')
# MAGIC
# MAGIC     if not rebuild and os.path.exists(P_CHECKINS):
# MAGIC         print('4/5  Check-ins: cache')
# MAGIC     else:
# MAGIC         print('4/5  Check-ins...')
# MAGIC         chk = _filter_by_business(config.CHECKIN, biz_ids, ['business_id', 'date'])
# MAGIC         chk.to_parquet(P_CHECKINS, index=False)
# MAGIC         print(f'     {len(chk):,} negocios con check-ins -> {P_CHECKINS}')
# MAGIC
# MAGIC     if not rebuild and os.path.exists(P_TIPS):
# MAGIC         print('5/5  Tips: cache')
# MAGIC     else:
# MAGIC         print('5/5  Tips...')
# MAGIC         tips = _filter_by_business(config.TIP, biz_ids, _TIP_FIELDS)
# MAGIC         tips.to_parquet(P_TIPS, index=False)
# MAGIC         print(f'     {len(tips):,} tips -> {P_TIPS}')
# MAGIC
# MAGIC     print(f'\nListo. Subconjunto ({config.CITY_LABEL}) en {config.ARTIFACTS}')
# MAGIC     return load_subset()
# MAGIC
# MAGIC
# MAGIC def load_subset():
# MAGIC     return {
# MAGIC         'business': pd.read_parquet(P_BIZ),
# MAGIC         'reviews':  pd.read_parquet(P_REVIEWS),
# MAGIC         'users':    pd.read_parquet(P_USERS),
# MAGIC         'checkins': pd.read_parquet(P_CHECKINS),
# MAGIC         'tips':     pd.read_parquet(P_TIPS),
# MAGIC     }
# MAGIC
# MAGIC
# MAGIC if __name__ == '__main__':
# MAGIC     build_subset()

# COMMAND ----------

# load_mod lee los .py directo desde disco (evita el cache de imports del Workspace)
import sys, os, importlib.util

def load_mod(name):
    path = os.path.join(os.getcwd(), name + '.py')
    assert os.path.exists(path), f'No existe {path}: corre primero su celda %%writefile'
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

config = load_mod('config')
preprocessing = load_mod('preprocessing')

assert hasattr(preprocessing, 'load_subset'), 'El archivo se copio incompleto; revisa la Celda 1.'

data = preprocessing.build_subset()

print('\n--- shapes ---')
for k, df in data.items():
    print(f'{k:9s}: {df.shape}')
data['business'].head()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cleaning

# COMMAND ----------

# MAGIC %%writefile cleaning.py
# MAGIC import ast
# MAGIC import json
# MAGIC import pandas as pd
# MAGIC import config
# MAGIC import preprocessing
# MAGIC
# MAGIC
# MAGIC def _parse_categories(s):
# MAGIC     if not s or (isinstance(s, float) and pd.isna(s)):
# MAGIC         return []
# MAGIC     return [c.strip() for c in str(s).split(',') if c.strip()]
# MAGIC
# MAGIC
# MAGIC def _coerce_value(v):
# MAGIC     if v is None or isinstance(v, bool):
# MAGIC         return v
# MAGIC     s = str(v).strip()
# MAGIC     if s in ('None', ''):
# MAGIC         return None
# MAGIC     if s == 'True':
# MAGIC         return True
# MAGIC     if s == 'False':
# MAGIC         return False
# MAGIC     if s.startswith('{') and s.endswith('}'):
# MAGIC         try:
# MAGIC             return {k: _coerce_value(val) for k, val in ast.literal_eval(s).items()}
# MAGIC         except (ValueError, SyntaxError):
# MAGIC             return s
# MAGIC     if s.lstrip('-').isdigit():
# MAGIC         return int(s)
# MAGIC     return s
# MAGIC
# MAGIC
# MAGIC def _parse_attributes(s):
# MAGIC     if not s or (isinstance(s, float) and pd.isna(s)):
# MAGIC         return {}
# MAGIC     try:
# MAGIC         raw = json.loads(s)
# MAGIC     except (json.JSONDecodeError, TypeError):
# MAGIC         return {}
# MAGIC     return {k: _coerce_value(v) for k, v in raw.items()}
# MAGIC
# MAGIC
# MAGIC def _parse_hours_count(s):
# MAGIC     if not s or (isinstance(s, float) and pd.isna(s)):
# MAGIC         return 0
# MAGIC     try:
# MAGIC         return len(json.loads(s))
# MAGIC     except (json.JSONDecodeError, TypeError):
# MAGIC         return 0
# MAGIC
# MAGIC
# MAGIC def clean_business(biz):
# MAGIC     rep, df, n0 = {}, biz.copy(), len(biz)
# MAGIC     df = df.drop_duplicates('business_id')
# MAGIC     rep['business_id duplicados eliminados'] = n0 - len(df)
# MAGIC
# MAGIC     df['categories_list'] = df['categories'].apply(_parse_categories)
# MAGIC     df['n_categories'] = df['categories_list'].apply(len)
# MAGIC     rep['negocios sin categoria'] = int((df['n_categories'] == 0).sum())
# MAGIC
# MAGIC     df['attributes_clean'] = df['attributes'].apply(_parse_attributes)
# MAGIC     df['price_range'] = df['attributes_clean'].apply(
# MAGIC         lambda d: d.get('RestaurantsPriceRange2') if isinstance(d, dict) else None)
# MAGIC     rep['negocios sin atributos'] = int((df['attributes_clean'].apply(len) == 0).sum())
# MAGIC     rep['negocios sin price_range'] = int(df['price_range'].isna().sum())
# MAGIC
# MAGIC     df['n_days_open'] = df['hours'].apply(_parse_hours_count)
# MAGIC     rep['negocios sin horario'] = int((df['n_days_open'] == 0).sum())
# MAGIC
# MAGIC     thr = df['review_count'].quantile(0.99)
# MAGIC     df['rc_outlier'] = df['review_count'] > thr
# MAGIC     rep[f'outliers review_count (>p99={thr:.0f}), marcados no eliminados'] = int(df['rc_outlier'].sum())
# MAGIC     return df, rep
# MAGIC
# MAGIC
# MAGIC def clean_reviews(rev):
# MAGIC     rep, df, n0 = {}, rev.copy(), len(rev)
# MAGIC     df = df.drop_duplicates('review_id')
# MAGIC     rep['review_id duplicados eliminados'] = n0 - len(df)
# MAGIC
# MAGIC     n1 = len(df)
# MAGIC     df = df.sort_values('date').drop_duplicates(['user_id', 'business_id'], keep='last')
# MAGIC     rep['pares (user,business) repetidos colapsados al mas reciente'] = n1 - len(df)
# MAGIC
# MAGIC     df['stars'] = pd.to_numeric(df['stars'], errors='coerce')
# MAGIC     rep['reviews con stars invalido'] = int(df['stars'].isna().sum())
# MAGIC
# MAGIC     df['text'] = df['text'].fillna('')
# MAGIC     df['text_len'] = df['text'].str.len()
# MAGIC     rep['reviews con texto vacio'] = int((df['text_len'] == 0).sum())
# MAGIC     return df, rep
# MAGIC
# MAGIC
# MAGIC def clean_users(usr):
# MAGIC     rep, df, n0 = {}, usr.copy(), len(usr)
# MAGIC     df = df.drop_duplicates('user_id')
# MAGIC     rep['user_id duplicados eliminados'] = n0 - len(df)
# MAGIC
# MAGIC     elite = df['elite'].fillna('').astype(str)
# MAGIC     df['n_elite_years'] = elite.apply(
# MAGIC         lambda s: 0 if s.strip() in ('', 'None') else len([y for y in s.split(',') if y.strip()]))
# MAGIC     df['is_elite'] = df['n_elite_years'] > 0
# MAGIC     rep['usuarios elite'] = int(df['is_elite'].sum())
# MAGIC
# MAGIC     thr = df['review_count'].quantile(0.99)
# MAGIC     df['rc_outlier'] = df['review_count'] > thr
# MAGIC     rep[f'outliers review_count (>p99={thr:.0f}), marcados no eliminados'] = int(df['rc_outlier'].sum())
# MAGIC     return df, rep
# MAGIC
# MAGIC
# MAGIC def clean_checkins(chk):
# MAGIC     df = chk.copy()
# MAGIC     df['n_checkins'] = df['date'].fillna('').apply(
# MAGIC         lambda s: 0 if not s else len(str(s).split(',')))
# MAGIC     return df, {'total check-ins contados': int(df['n_checkins'].sum())}
# MAGIC
# MAGIC
# MAGIC def clean_tips(tips):
# MAGIC     df, n0 = tips.copy(), len(tips)
# MAGIC     df = df.drop_duplicates()
# MAGIC     df['text'] = df['text'].fillna('')
# MAGIC     df['date'] = pd.to_datetime(df['date'], errors='coerce')
# MAGIC     return df, {'tips duplicados eliminados': n0 - len(df)}
# MAGIC
# MAGIC
# MAGIC def clean_subset(data=None):
# MAGIC     if data is None:
# MAGIC         data = preprocessing.load_subset()
# MAGIC
# MAGIC     biz, r1 = clean_business(data['business'])
# MAGIC     rev, r2 = clean_reviews(data['reviews'])
# MAGIC     usr, r3 = clean_users(data['users'])
# MAGIC     chk, r4 = clean_checkins(data['checkins'])
# MAGIC     tip, r5 = clean_tips(data['tips'])
# MAGIC
# MAGIC     print(f'REPORTE DE LIMPIEZA -- {config.CITY_LABEL}')
# MAGIC     for title, rep in [('Negocios', r1), ('Resenas', r2), ('Usuarios', r3),
# MAGIC                        ('Check-ins', r4), ('Tips', r5)]:
# MAGIC         print(f'\n[{title}]')
# MAGIC         for k, v in rep.items():
# MAGIC             print(f'  - {k}: {v:,}' if isinstance(v, int) else f'  - {k}: {v}')
# MAGIC
# MAGIC     print('\nFilas tras limpieza:')
# MAGIC     for k, df in [('business', biz), ('reviews', rev), ('users', usr),
# MAGIC                   ('checkins', chk), ('tips', tip)]:
# MAGIC         print(f'  {k:9s}: {len(df):,}')
# MAGIC
# MAGIC     return {'business': biz, 'reviews': rev, 'users': usr,
# MAGIC             'checkins': chk, 'tips': tip}

# COMMAND ----------

# load_mod lee los .py directo desde disco (evita el cache de imports del Workspace)
import sys, os, importlib.util

def load_mod(name):
    path = os.path.join(os.getcwd(), name + '.py')
    assert os.path.exists(path), f'No existe {path}: corre primero su celda %%writefile'
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

config        = load_mod('config')
preprocessing = load_mod('preprocessing')
cleaning      = load_mod('cleaning')

clean = cleaning.clean_subset()
clean['business'][['name', 'categories_list', 'price_range', 'n_days_open', 'rc_outlier']].head()

# COMMAND ----------

# MAGIC %md
# MAGIC ### EDA

# COMMAND ----------

# MAGIC %%writefile eda.py
# MAGIC from collections import Counter
# MAGIC import numpy as np
# MAGIC import pandas as pd
# MAGIC import matplotlib.pyplot as plt
# MAGIC import config
# MAGIC
# MAGIC
# MAGIC def plot_rating_distribution(clean):
# MAGIC     rev = clean['reviews']
# MAGIC     counts = rev['stars'].value_counts().sort_index()
# MAGIC     fig, ax = plt.subplots(figsize=(6, 4))
# MAGIC     ax.bar(counts.index, counts.values, width=0.6, color='#378ADD')
# MAGIC     ax.set_xlabel('Estrellas'); ax.set_ylabel('Nro de resenas')
# MAGIC     ax.set_title(f'Distribucion de ratings -- {config.CITY_LABEL}')
# MAGIC     for x, y in zip(counts.index, counts.values):
# MAGIC         ax.text(x, y, f'{y:,}', ha='center', va='bottom', fontsize=8)
# MAGIC     plt.tight_layout(); plt.show()
# MAGIC     print(f'Rating promedio: {rev["stars"].mean():.2f} (sesgo a 4-5, tipico de Yelp)')
# MAGIC
# MAGIC
# MAGIC def plot_review_length(clean):
# MAGIC     rev = clean['reviews']
# MAGIC     fig, ax = plt.subplots(figsize=(6, 4))
# MAGIC     ax.hist(rev['text_len'].clip(upper=2000), bins=50, color='#1D9E75')
# MAGIC     ax.set_xlabel('Longitud de resena (caracteres, recortado a 2000)')
# MAGIC     ax.set_ylabel('Nro de resenas')
# MAGIC     ax.set_title('Distribucion de longitud de resenas')
# MAGIC     plt.tight_layout(); plt.show()
# MAGIC     print(f'Mediana de longitud: {rev["text_len"].median():.0f} caracteres')
# MAGIC
# MAGIC
# MAGIC def _loglog(series, ax, xlabel, color):
# MAGIC     vc = series.value_counts().sort_index()
# MAGIC     vc = vc[vc.index > 0]
# MAGIC     ax.scatter(vc.index, vc.values, s=12, color=color, alpha=0.6)
# MAGIC     ax.set_xscale('log'); ax.set_yscale('log')
# MAGIC     ax.set_xlabel(xlabel); ax.set_ylabel('Frecuencia (nro de elementos)')
# MAGIC
# MAGIC
# MAGIC def plot_powerlaws(clean):
# MAGIC     fig, axes = plt.subplots(1, 2, figsize=(11, 4))
# MAGIC     _loglog(clean['business']['review_count'], axes[0],
# MAGIC             'Resenas por negocio', '#D85A30')
# MAGIC     axes[0].set_title('Power-law: actividad de negocios')
# MAGIC     per_user = clean['reviews']['user_id'].value_counts()
# MAGIC     _loglog(per_user, axes[1], 'Resenas por usuario (en la muestra)', '#534AB7')
# MAGIC     axes[1].set_title('Power-law: actividad de usuarios')
# MAGIC     plt.tight_layout(); plt.show()
# MAGIC     print('Ambas siguen ley de potencias: pocos muy activos, larga cola de casuales.')
# MAGIC
# MAGIC
# MAGIC def plot_top_categories(clean, top=15):
# MAGIC     cats = Counter()
# MAGIC     for lst in clean['business']['categories_list']:
# MAGIC         cats.update(lst)
# MAGIC     common = pd.Series(dict(cats.most_common(top))).iloc[::-1]
# MAGIC     fig, ax = plt.subplots(figsize=(7, 5))
# MAGIC     ax.barh(common.index, common.values, color='#378ADD')
# MAGIC     ax.set_xlabel('Nro de negocios')
# MAGIC     ax.set_title(f'Top {top} categorias -- {config.CITY_LABEL}')
# MAGIC     plt.tight_layout(); plt.show()
# MAGIC
# MAGIC
# MAGIC def plot_geographic(clean):
# MAGIC     biz = clean['business']
# MAGIC     fig, ax = plt.subplots(figsize=(7, 6))
# MAGIC     sc = ax.scatter(biz['longitude'], biz['latitude'], c=biz['stars'],
# MAGIC                     cmap='viridis', s=14, alpha=0.6)
# MAGIC     ax.set_xlabel('Longitud'); ax.set_ylabel('Latitud')
# MAGIC     ax.set_title(f'Negocios de {config.CITY_LABEL} (color = estrellas)')
# MAGIC     fig.colorbar(sc, ax=ax, label='Estrellas')
# MAGIC     plt.tight_layout(); plt.show()
# MAGIC
# MAGIC
# MAGIC def plot_temporal(clean):
# MAGIC     rev = clean['reviews'].dropna(subset=['date'])
# MAGIC     monthly = rev.set_index('date').resample('MS').size()
# MAGIC     fig, ax = plt.subplots(figsize=(9, 4))
# MAGIC     ax.plot(monthly.index, monthly.values, color='#0F6E56')
# MAGIC     ax.set_xlabel('Fecha'); ax.set_ylabel('Resenas por mes')
# MAGIC     ax.set_title('Evolucion temporal de resenas')
# MAGIC     plt.tight_layout(); plt.show()
# MAGIC
# MAGIC
# MAGIC def plot_friendship_degree(clean):
# MAGIC     deg = clean['users']['n_friends_subset']
# MAGIC     fig, ax = plt.subplots(figsize=(6, 4))
# MAGIC     _loglog(deg[deg > 0], ax, 'Amigos dentro de la muestra (grado)', '#993556')
# MAGIC     ax.set_title('Distribucion de grados del grafo de amistad')
# MAGIC     plt.tight_layout(); plt.show()
# MAGIC     print(f'{(deg == 0).mean():.1%} de usuarios sin amigos en la muestra; '
# MAGIC           f'el resto forma el nucleo social de la Parte II.')
# MAGIC
# MAGIC
# MAGIC def plot_segments(clean):
# MAGIC     biz, usr = clean['business'], clean['users']
# MAGIC     fig, axes = plt.subplots(1, 2, figsize=(11, 4))
# MAGIC     by_open = biz.groupby('is_open')['stars'].mean()
# MAGIC     axes[0].bar(['Cerrado', 'Abierto'],
# MAGIC                 [by_open.get(0, np.nan), by_open.get(1, np.nan)],
# MAGIC                 color=['#888780', '#1D9E75'])
# MAGIC     axes[0].set_ylabel('Estrellas promedio')
# MAGIC     axes[0].set_title('Rating: abierto vs cerrado')
# MAGIC     by_elite = usr.groupby('is_elite')['average_stars'].mean()
# MAGIC     axes[1].bar(['No elite', 'Elite'],
# MAGIC                 [by_elite.get(False, np.nan), by_elite.get(True, np.nan)],
# MAGIC                 color=['#888780', '#534AB7'])
# MAGIC     axes[1].set_ylabel('Estrellas promedio dadas')
# MAGIC     axes[1].set_title('Comportamiento: elite vs no elite')
# MAGIC     plt.tight_layout(); plt.show()
# MAGIC
# MAGIC
# MAGIC def plot_all(clean):
# MAGIC     plot_rating_distribution(clean)
# MAGIC     plot_review_length(clean)
# MAGIC     plot_powerlaws(clean)
# MAGIC     plot_top_categories(clean)
# MAGIC     plot_geographic(clean)
# MAGIC     plot_temporal(clean)
# MAGIC     plot_friendship_degree(clean)
# MAGIC     plot_segments(clean)

# COMMAND ----------


import sys, os, importlib.util
def load_mod(name):
    path = os.path.join(os.getcwd(), name + '.py')
    assert os.path.exists(path), f'No existe {path}: corre primero su celda %%writefile'
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod
config = load_mod('config')
preprocessing = load_mod('preprocessing')
cleaning = load_mod('cleaning')
eda = load_mod('eda')
clean = cleaning.clean_subset()
eda.plot_all(clean)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Gráficos

# COMMAND ----------

# MAGIC %%writefile graphs.py
# MAGIC from collections import deque
# MAGIC import config
# MAGIC
# MAGIC
# MAGIC def build_bipartite(reviews):
# MAGIC     n_users = reviews['user_id'].nunique()
# MAGIC     n_biz = reviews['business_id'].nunique()
# MAGIC     n_edges = len(reviews)
# MAGIC     density = n_edges / (n_users * n_biz) if n_users and n_biz else 0
# MAGIC     return {'n_users': n_users, 'n_business': n_biz,
# MAGIC             'n_edges': n_edges, 'density': density}
# MAGIC
# MAGIC
# MAGIC def build_friendship(users):
# MAGIC     adj = {}
# MAGIC     for uid, fr in zip(users['user_id'], users['friends_subset']):
# MAGIC         adj.setdefault(uid, set())
# MAGIC         if isinstance(fr, str) and fr:
# MAGIC             for f in fr.split(','):
# MAGIC                 if f:
# MAGIC                     adj[uid].add(f)
# MAGIC                     adj.setdefault(f, set()).add(uid)
# MAGIC     return adj
# MAGIC
# MAGIC
# MAGIC def _components(adj):
# MAGIC     seen, comps = set(), []
# MAGIC     for start in adj:
# MAGIC         if start in seen:
# MAGIC             continue
# MAGIC         comp, dq = [], deque([start])
# MAGIC         seen.add(start)
# MAGIC         while dq:
# MAGIC             u = dq.popleft()
# MAGIC             comp.append(u)
# MAGIC             for v in adj[u]:
# MAGIC                 if v not in seen:
# MAGIC                     seen.add(v)
# MAGIC                     dq.append(v)
# MAGIC         comps.append(comp)
# MAGIC     return comps
# MAGIC
# MAGIC
# MAGIC def _bfs_farthest(adj, src):
# MAGIC     dist = {src: 0}
# MAGIC     dq = deque([src])
# MAGIC     far, fd = src, 0
# MAGIC     while dq:
# MAGIC         u = dq.popleft()
# MAGIC         for v in adj[u]:
# MAGIC             if v not in dist:
# MAGIC                 dist[v] = dist[u] + 1
# MAGIC                 if dist[v] > fd:
# MAGIC                     far, fd = v, dist[v]
# MAGIC                 dq.append(v)
# MAGIC     return far, fd
# MAGIC
# MAGIC
# MAGIC def approx_diameter(adj, lcc_set):
# MAGIC     sub = {u: (adj[u] & lcc_set) for u in lcc_set}
# MAGIC     a, _ = _bfs_farthest(sub, next(iter(lcc_set)))
# MAGIC     _, d = _bfs_farthest(sub, a)
# MAGIC     return d
# MAGIC
# MAGIC
# MAGIC def friendship_metrics(users):
# MAGIC     adj = build_friendship(users)
# MAGIC     n = len(adj)
# MAGIC     deg_sum = sum(len(v) for v in adj.values())
# MAGIC     edges = deg_sum // 2
# MAGIC     density = deg_sum / (n * (n - 1)) if n > 1 else 0
# MAGIC     comps = _components(adj)
# MAGIC     lcc = max(comps, key=len) if comps else []
# MAGIC     diam = approx_diameter(adj, set(lcc)) if lcc else 0
# MAGIC     metrics = {'n_nodes': n, 'n_edges': edges, 'density': density,
# MAGIC                'n_components': len(comps), 'lcc_size': len(lcc),
# MAGIC                'lcc_fraction': len(lcc) / n if n else 0,
# MAGIC                'lcc_diameter_approx': diam}
# MAGIC     return metrics, adj, lcc
# MAGIC
# MAGIC
# MAGIC def build_graphs(clean=None):
# MAGIC     if clean is None:
# MAGIC         import cleaning
# MAGIC         clean = cleaning.clean_subset()
# MAGIC
# MAGIC     bp = build_bipartite(clean['reviews'])
# MAGIC     fm, adj, lcc = friendship_metrics(clean['users'])
# MAGIC
# MAGIC     print(f'GRAFOS INICIALES -- {config.CITY_LABEL}\n')
# MAGIC     print('[Bipartito usuario-negocio]')
# MAGIC     print(f"  usuarios:          {bp['n_users']:,}")
# MAGIC     print(f"  negocios:          {bp['n_business']:,}")
# MAGIC     print(f"  aristas (reseñas): {bp['n_edges']:,}")
# MAGIC     print(f"  densidad:          {bp['density']:.2e}")
# MAGIC     print('\n[Amistad usuario-usuario]')
# MAGIC     print(f"  nodos:             {fm['n_nodes']:,}")
# MAGIC     print(f"  aristas:           {fm['n_edges']:,}")
# MAGIC     print(f"  densidad:          {fm['density']:.2e}")
# MAGIC     print(f"  componentes:       {fm['n_components']:,}")
# MAGIC     print(f"  LCC:               {fm['lcc_size']:,} ({fm['lcc_fraction']:.1%})")
# MAGIC     print(f"  diametro aprox LCC: {fm['lcc_diameter_approx']}")
# MAGIC
# MAGIC     return {'bipartite': bp, 'friendship': fm, 'friend_adj': adj, 'lcc': lcc}

# COMMAND ----------

import sys, os, importlib.util
def load_mod(name):
    path = os.path.join(os.getcwd(), name + '.py')
    assert os.path.exists(path), f'No existe {path}: corre primero su celda %%writefile'
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod
config = load_mod('config')
preprocessing = load_mod('preprocessing')
cleaning = load_mod('cleaning')
graphs = load_mod('graphs')
clean = cleaning.clean_subset()
G = graphs.build_graphs(clean)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Lectura de resultados de la Parte I
# MAGIC
# MAGIC La muestra estratificada quedo en **2,943 negocios, 125,744 resenas y 103,447 usuarios** (2005-2022), con las proporciones por ciudad de la poblacion preservadas (Philadelphia, Tucson y Tampa encabezan, como en el dataset completo) y medias de stars/review_count/is_open practicamente identicas a las poblacionales, segun el reporte de representatividad.
# MAGIC
# MAGIC De la limpieza destaca que los faltantes son **estructurales, no errores**: 43% de los negocios no declara price_range y 16% no publica horario; se imputan o marcan en vez de eliminarse. Los outliers de actividad (>p99) se marcan pero se conservan, porque en las Partes II y IV son justamente la senal (hubs, authorities, power-users).
# MAGIC
# MAGIC Sobre los grafos: el bipartito usuario-negocio tiene densidad 4.1e-04 (cada usuario toca poquisimos negocios: mediana de 1 resena). El grafo de amistad tiene 305,776 aristas pero esta muy fragmentado: 58% de los usuarios no tiene ningun amigo dentro de la muestra y la componente gigante (LCC) cubre 41,832 usuarios (40.4%), con diametro aproximado 14. Las Partes II (ranking y comunidades) trabajan sobre ese LCC.