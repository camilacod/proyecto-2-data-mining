# Databricks notebook source
# MAGIC %md
# MAGIC # Parte V -- Mineria de Flujos de Datos
# MAGIC
# MAGIC Corre en la **misma carpeta del Workspace** que la Parte I: reutiliza sus modulos y los parquet de `artifacts/`.

# COMMAND ----------

# Setup: modulos y artifacts de la Parte I (misma carpeta del Workspace).
# load_mod lee los .py directo desde disco (evita el cache de imports del Workspace)
import sys, os, importlib.util

src = os.getcwd()
sys.path.insert(0, src)

def load_mod(name):
    path = os.path.join(src, name + '.py')
    assert os.path.exists(path), f'No existe {path}: corre primero la Parte I en esta carpeta'
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

config = load_mod('config')
preprocessing = load_mod('preprocessing')
cleaning = load_mod('cleaning')
print('Artefactos en:', config.ARTIFACTS)

# build_subset() es idempotente: usa el cache parquet si existe,
# y si no (p.ej. cluster nuevo) re-streamea los JSON (~3 min)
clean = cleaning.clean_subset(preprocessing.build_subset())

# COMMAND ----------

# MAGIC %md
# MAGIC # PARTE 5 — Mineria de Flujos de Datos
# MAGIC
# MAGIC Las resenas del subconjunto tienen timestamp, asi que podemos **reproducirlas en orden cronologico y tratarlas como un stream**: un evento a la vez, sin mirar el futuro, y con memoria acotada. Sobre ese stream aplicamos tres tecnicas:
# MAGIC
# MAGIC 1. **Ventanas deslizantes** (1h, 4h, 24h): estadisticas exactas pero solo sobre el pasado reciente.
# MAGIC 2. **Count-Min Sketch**: conteo de frecuencias aproximado con memoria sublineal y garantias de error demostrables.
# MAGIC 3. **Flajolet-Martin (variante LogLog)**: conteo de elementos *distintos* con un punado de enteros -- la tecnica adicional del enunciado, justificada mas abajo con evidencia de los datos.

# COMMAND ----------

# MAGIC %%writefile streaming.py
# MAGIC import math
# MAGIC import numpy as np
# MAGIC import pandas as pd
# MAGIC import matplotlib.pyplot as plt
# MAGIC from collections import deque, Counter
# MAGIC
# MAGIC
# MAGIC # ===================== Stream simulado =====================
# MAGIC def make_stream(reviews):
# MAGIC     """Ordenamos las resenas por fecha y las emitimos una a una: eso simula el
# MAGIC     flujo que veria el servidor de Yelp en tiempo real."""
# MAGIC     cols = ['date', 'user_id', 'business_id', 'stars']
# MAGIC     st = reviews[cols].dropna(subset=['date']).sort_values('date', kind='mergesort')
# MAGIC     st = st.reset_index(drop=True)
# MAGIC     print(f'Stream simulado: {len(st):,} eventos, '
# MAGIC           f'de {st["date"].min():%Y-%m-%d} a {st["date"].max():%Y-%m-%d}')
# MAGIC     return st
# MAGIC
# MAGIC
# MAGIC # ===================== Ventanas deslizantes =====================
# MAGIC class SlidingWindow:
# MAGIC     """Mantiene solo los eventos de las ultimas `hours` horas (deque);
# MAGIC     count/sum/avg se actualizan incrementalmente en O(1) amortizado."""
# MAGIC
# MAGIC     def __init__(self, hours):
# MAGIC         self.width = pd.Timedelta(hours=hours)
# MAGIC         self.buf = deque()
# MAGIC         self.count = 0
# MAGIC         self.sum = 0.0
# MAGIC
# MAGIC     def add(self, ts, value):
# MAGIC         self.buf.append((ts, value))
# MAGIC         self.count += 1
# MAGIC         self.sum += value
# MAGIC         limit = ts - self.width
# MAGIC         while self.buf and self.buf[0][0] <= limit:
# MAGIC             _, old = self.buf.popleft()
# MAGIC             self.count -= 1
# MAGIC             self.sum -= old
# MAGIC
# MAGIC     def stats(self):
# MAGIC         avg = self.sum / self.count if self.count else float('nan')
# MAGIC         return self.count, self.sum, avg
# MAGIC
# MAGIC
# MAGIC def run_windows(stream, hours_list=(1, 4, 24), snapshot_every=2000):
# MAGIC     """Procesa el stream y toma una foto de cada ventana cada `snapshot_every`
# MAGIC     eventos. Devuelve un DataFrame de trayectorias (count y avg por ventana)."""
# MAGIC     wins = {h: SlidingWindow(h) for h in hours_list}
# MAGIC     rows = []
# MAGIC     for i, (ts, stars) in enumerate(zip(stream['date'], stream['stars'])):
# MAGIC         for w in wins.values():
# MAGIC             w.add(ts, float(stars))
# MAGIC         if (i + 1) % snapshot_every == 0:
# MAGIC             row = {'ts': ts}
# MAGIC             for h, w in wins.items():
# MAGIC                 c, s, a = w.stats()
# MAGIC                 row[f'count_{h}h'] = c
# MAGIC                 row[f'avg_{h}h'] = a
# MAGIC             rows.append(row)
# MAGIC     snap = pd.DataFrame(rows)
# MAGIC     print(f'{len(stream):,} eventos procesados, {len(snap):,} snapshots '
# MAGIC           f'(cada {snapshot_every:,} eventos)')
# MAGIC     return snap
# MAGIC
# MAGIC
# MAGIC def plot_windows(snap, hours_list=(1, 4, 24)):
# MAGIC     fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
# MAGIC     colors = ['#378ADD', '#D85A30', '#4CAF50']
# MAGIC     for h, c in zip(hours_list, colors):
# MAGIC         ax[0].plot(snap['ts'], snap[f'count_{h}h'], color=c, lw=1, label=f'ventana {h}h')
# MAGIC         ax[1].plot(snap['ts'], snap[f'avg_{h}h'], color=c, lw=1, label=f'ventana {h}h')
# MAGIC     ax[0].set_ylabel('resenas en ventana (count)')
# MAGIC     ax[0].set_yscale('log')
# MAGIC     ax[0].set_title('Actividad dentro de cada ventana deslizante')
# MAGIC     ax[1].set_ylabel('rating promedio en ventana')
# MAGIC     ax[1].set_title('Rating promedio dentro de cada ventana')
# MAGIC     for a in ax:
# MAGIC         a.legend(fontsize=8)
# MAGIC     plt.tight_layout(); plt.show()
# MAGIC
# MAGIC
# MAGIC def temporal_patterns(stream):
# MAGIC     """Agregados clasicos sobre el timestamp: patron horario, semanal y anual."""
# MAGIC     df = stream.copy()
# MAGIC     df['hour'] = df['date'].dt.hour
# MAGIC     df['dow'] = df['date'].dt.dayofweek
# MAGIC     df['month'] = df['date'].dt.month
# MAGIC     fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
# MAGIC     df.groupby('hour').size().plot(kind='bar', ax=ax[0], color='#378ADD')
# MAGIC     ax[0].set_title('Resenas por hora del dia'); ax[0].set_xlabel('hora')
# MAGIC     df.groupby('dow').size().plot(kind='bar', ax=ax[1], color='#D85A30')
# MAGIC     ax[1].set_title('Resenas por dia de semana (0=lun)'); ax[1].set_xlabel('dia')
# MAGIC     df.groupby('month').size().plot(kind='bar', ax=ax[2], color='#4CAF50')
# MAGIC     ax[2].set_title('Resenas por mes (estacionalidad)'); ax[2].set_xlabel('mes')
# MAGIC     for a in ax:
# MAGIC         a.set_ylabel('resenas')
# MAGIC     plt.tight_layout(); plt.show()
# MAGIC
# MAGIC
# MAGIC # ===================== Count-Min Sketch =====================
# MAGIC class CountMinSketch:
# MAGIC     """d filas x w columnas. Con w = ceil(e/eps) y d = ceil(ln(1/delta)):
# MAGIC        est(x) >= true(x)  siempre  (solo sobreestima)
# MAGIC        est(x) <= true(x) + eps*N   con probabilidad >= 1 - delta."""
# MAGIC
# MAGIC     def __init__(self, eps=0.001, delta=0.01, seed=7):
# MAGIC         self.w = int(math.ceil(math.e / eps))
# MAGIC         self.d = int(math.ceil(math.log(1.0 / delta)))
# MAGIC         self.eps, self.delta = eps, delta
# MAGIC         self.table = np.zeros((self.d, self.w), dtype=np.int64)
# MAGIC         rng = np.random.default_rng(seed)
# MAGIC         # familia universal: h_i(x) = ((a_i*x + b_i) mod p) mod w
# MAGIC         self.p = 2_147_483_647
# MAGIC         self.a = rng.integers(1, self.p, size=self.d)
# MAGIC         self.b = rng.integers(0, self.p, size=self.d)
# MAGIC         self.N = 0
# MAGIC
# MAGIC     def _hashes(self, x):
# MAGIC         hx = hash(x) & 0x7FFFFFFF
# MAGIC         return (self.a * hx + self.b) % self.p % self.w
# MAGIC
# MAGIC     def add(self, x):
# MAGIC         self.table[np.arange(self.d), self._hashes(x)] += 1
# MAGIC         self.N += 1
# MAGIC
# MAGIC     def estimate(self, x):
# MAGIC         return int(self.table[np.arange(self.d), self._hashes(x)].min())
# MAGIC
# MAGIC     def memory_cells(self):
# MAGIC         return self.d * self.w
# MAGIC
# MAGIC
# MAGIC def cms_vs_exact(stream, key='business_id', eps=0.001, delta=0.01, top=15):
# MAGIC     cms = CountMinSketch(eps=eps, delta=delta)
# MAGIC     exact = Counter()
# MAGIC     for x in stream[key]:
# MAGIC         cms.add(x)
# MAGIC         exact[x] += 1
# MAGIC     rows = []
# MAGIC     for x, true in exact.most_common(top):
# MAGIC         est = cms.estimate(x)
# MAGIC         rows.append({key: x, 'exacto': true, 'cms': est, 'error': est - true})
# MAGIC     df = pd.DataFrame(rows)
# MAGIC
# MAGIC     errs = np.array([cms.estimate(x) - c for x, c in exact.items()])
# MAGIC     bound = eps * cms.N
# MAGIC     frac_ok = float((errs <= bound).mean())
# MAGIC     print(f'CMS: {cms.d} filas x {cms.w:,} columnas = {cms.memory_cells():,} celdas '
# MAGIC           f'(vs {len(exact):,} claves exactas)')
# MAGIC     print(f'Garantia: error <= eps*N = {bound:.1f} con prob >= {1 - delta:.0%}')
# MAGIC     print(f'Observado: error <= cota en {frac_ok:.2%} de las claves | '
# MAGIC           f'error max = {errs.max()}, error medio = {errs.mean():.2f}')
# MAGIC     print(f'(el error nunca es negativo: min = {errs.min()})')
# MAGIC     return df, cms, exact
# MAGIC
# MAGIC
# MAGIC # ============ Flajolet-Martin / LogLog (tecnica adicional) ============
# MAGIC class FlajoletMartinLL:
# MAGIC     """Conteo aproximado de elementos DISTINTOS (variante LogLog de FM).
# MAGIC     Idea FM: si hasheamos n valores distintos, el maximo numero de ceros al
# MAGIC     final del hash crece como log2(n). Para reducir la varianza, un solo hash
# MAGIC     reparte cada elemento en m buckets (bits bajos) y en cada bucket se guarda
# MAGIC     el maximo rho (posicion del primer 1) de los bits restantes. Estimador:
# MAGIC     alpha * m * 2^(promedio de los m maximos). Memoria: m enteros pequenos.
# MAGIC     Error tipico ~ 1.3/sqrt(m) (~8% con m=256)."""
# MAGIC
# MAGIC     ALPHA = 0.39701
# MAGIC
# MAGIC     def __init__(self, m=256, seed=11):
# MAGIC         rng = np.random.default_rng(seed)
# MAGIC         self.m = m
# MAGIC         self.p = 2_147_483_647
# MAGIC         self.a = int(rng.integers(1, self.p))
# MAGIC         self.b = int(rng.integers(0, self.p))
# MAGIC         self.M = np.zeros(m, dtype=np.int64)
# MAGIC
# MAGIC     @staticmethod
# MAGIC     def _rho(v):
# MAGIC         # posicion (1-indexada) del bit 1 menos significativo
# MAGIC         return (v & -v).bit_length() if v else 32
# MAGIC
# MAGIC     def add(self, x):
# MAGIC         hx = (self.a * (hash(x) & 0x7FFFFFFF) + self.b) % self.p
# MAGIC         j = hx % self.m
# MAGIC         r = self._rho(int(hx // self.m))
# MAGIC         if r > self.M[j]:
# MAGIC             self.M[j] = r
# MAGIC
# MAGIC     def estimate(self):
# MAGIC         return float(self.ALPHA * self.m * 2.0 ** self.M.mean())
# MAGIC
# MAGIC
# MAGIC def fm_distinct_users(stream, m=256):
# MAGIC     """Cuantos usuarios DISTINTOS escriben por anio, sin guardar sus IDs."""
# MAGIC     rows = []
# MAGIC     for period, g in stream.groupby(stream['date'].dt.to_period('Y')):
# MAGIC         fm = FlajoletMartinLL(m=m)
# MAGIC         for u in g['user_id']:
# MAGIC             fm.add(u)
# MAGIC         true = g['user_id'].nunique()
# MAGIC         est = fm.estimate()
# MAGIC         rows.append({'periodo': str(period), 'exacto': true,
# MAGIC                      'FM_LogLog': round(est),
# MAGIC                      'error_rel': round(abs(est - true) / true, 3)})
# MAGIC     df = pd.DataFrame(rows)
# MAGIC     print(f'FM-LogLog usa {m} enteros por periodo (vs un set con miles de '
# MAGIC           f'user_ids); error teorico ~ {1.3 / math.sqrt(m):.1%}')
# MAGIC     return df

# COMMAND ----------

# MAGIC %md
# MAGIC ### Ventanas deslizantes (1h / 4h / 24h)
# MAGIC
# MAGIC Cada ventana mantiene en un `deque` solo los eventos de las ultimas *w* horas; `count` y `sum` se actualizan al entrar/salir eventos, asi que `average = sum/count` sale en O(1) sin re-escanear. Cada 500 eventos tomamos una foto del estado de las tres ventanas para graficar su evolucion a lo largo de los ~17 anos del dataset (2005-2022, ~126k eventos).

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

streaming = load_mod('streaming')

stream = streaming.make_stream(clean['reviews'])
snap = streaming.run_windows(stream, hours_list=(1, 4, 24), snapshot_every=500)
streaming.plot_windows(snap)
snap.tail(3)

# COMMAND ----------

# patrones temporales agregados: ciclo diario, semanal y estacional
streaming.temporal_patterns(stream)

# COMMAND ----------

# MAGIC %md
# MAGIC La ventana de 24h suaviza y muestra la **tendencia de largo plazo** (crecimiento de Yelp, caida en 2020 por la pandemia); las de 1h/4h son ruidosas y sirven para deteccion reactiva (picos de actividad). Los barplots muestran los ciclos: horas de comida, fines de semana y estacionalidad anual.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Count-Min Sketch: conteo aproximado de frecuencias
# MAGIC
# MAGIC Queremos "cuantas resenas lleva cada negocio / cada usuario" sin guardar un contador por clave. El CMS usa una tabla de `d x w` contadores y `d` funciones hash universales; cada elemento incrementa una celda por fila y la estimacion es el **minimo** de sus `d` celdas.
# MAGIC
# MAGIC Garantias teoricas con `w = ceil(e/eps)` y `d = ceil(ln(1/delta))`:
# MAGIC
# MAGIC - `est(x) >= true(x)` **siempre** (las colisiones solo suman, nunca restan);
# MAGIC - `est(x) <= true(x) + eps*N` con probabilidad `>= 1 - delta`, donde `N` es el largo del stream.
# MAGIC
# MAGIC Lo verificamos empiricamente contra el conteo exacto.

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

streaming = load_mod('streaming')

# claves = usuarios (~103k claves distintas): aqui el sketch si comprime de verdad
df_usr, cms_u, exact_u = streaming.cms_vs_exact(stream, key='user_id', eps=0.001, delta=0.01, top=10)
df_usr

# COMMAND ----------

# claves = negocios (~3k claves): el CMS funciona igual de bien,
# pero con tan pocas claves un dict exacto ya es barato (ver tradeoff abajo)
df_biz, cms_b, exact_b = streaming.cms_vs_exact(stream, key='business_id', eps=0.005, delta=0.01, top=10)
df_biz

# COMMAND ----------

# tradeoff memoria vs error: mismo stream (user_id), distintos eps
import numpy as np
import pandas as pd
from collections import Counter

exact = Counter(stream['user_id'])
filas = []
for eps in (0.01, 0.005, 0.001, 0.0005):
    cms = streaming.CountMinSketch(eps=eps, delta=0.01)
    for x in stream['user_id']:
        cms.add(x)
    errs = np.array([cms.estimate(x) - c for x, c in exact.items()])
    filas.append({'eps': eps, 'celdas': cms.memory_cells(),
                  'cota eps*N': round(eps * cms.N, 1),
                  'error medio': round(float(errs.mean()), 2),
                  'error max': int(errs.max()),
                  'dentro de cota': f'{(errs <= eps * cms.N).mean():.1%}'})
pd.DataFrame(filas)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Tecnica adicional: Flajolet-Martin (LogLog) para usuarios distintos
# MAGIC
# MAGIC **Justificacion con evidencia del dataset**: el CMS responde "*cuantas veces* aparecio x", pero una pregunta operativa distinta es "*cuantos usuarios distintos* estuvieron activos este anio". En la muestra hay ~103k usuarios y la gran mayoria escribe 1-2 resenas (el 88% tiene una sola resena, como vimos en el EDA de la Parte I y en el cold-start de la Parte IV), asi que frecuencia y cardinalidad cuentan historias muy diferentes: el volumen de resenas puede crecer porque los mismos power-users escriben mas, o porque llegan usuarios nuevos -- y solo un conteo de distintos separa ambas hipotesis. Guardar el set exacto de IDs cuesta O(usuarios distintos); FM lo estima con memoria constante.
# MAGIC
# MAGIC **Idea del algoritmo**: si hasheamos `n` valores distintos, el maximo numero de bits 0 al final del hash crece como `log2(n)`. La variante LogLog reduce la varianza: un solo hash reparte cada elemento en `m=256` buckets y en cada bucket se guarda el maximo `rho` (posicion del primer bit 1); el estimador es `alpha * m * 2^(promedio de maximos)`, con error tipico `~1.3/sqrt(m) ~ 8%`. Los duplicados no afectan (el mismo elemento cae siempre igual), que es justo lo que lo hace un contador de *distintos*.

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

streaming = load_mod('streaming')

fm_df = streaming.fm_distinct_users(stream, m=256)
fm_df

# COMMAND ----------

# MAGIC %md
# MAGIC ### Conclusiones de la parte de streaming
# MAGIC
# MAGIC - **Ventanas deslizantes**: estadisticas exactas del pasado reciente en O(1) por evento; revelan tendencia (crecimiento sostenido hasta 2019 y caida en 2020 por la pandemia), ciclo horario y estacionalidad.
# MAGIC - **Count-Min Sketch**: los errores observados respetan la teoria (nunca subestima; sobreestima <= eps*N con prob >= 1-delta; con eps=0.001 el error quedo dentro de la cota en el 100% de las claves). El tradeoff quedo explicito en la tabla: con ~103k claves (usuarios) el sketch comprime de verdad (13,595 celdas vs 103,449 contadores exactos, ~8x menos); con ~3k claves (negocios) un contador exacto sigue siendo viable -- el sketch se justifica cuando el dominio de claves explota (p.ej. todos los usuarios de Yelp, no solo esta muestra).
# MAGIC - **FM/LogLog**: estima usuarios distintos por anio con solo 256 enteros y error tipico de un digito porcentual en los anios con volumen (2011-2022: entre 0.4% y 13%); los primeros anios, con menos de ~100 usuarios, son ruidosos porque la varianza del estimador domina. Responde una pregunta (cardinalidad) que ni las ventanas ni el CMS pueden responder con memoria acotada.