import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque, Counter


# ===================== Stream simulado =====================
def make_stream(reviews):
    """Ordenamos las resenas por fecha y las emitimos una a una: eso simula el
    flujo que veria el servidor de Yelp en tiempo real."""
    cols = ['date', 'user_id', 'business_id', 'stars']
    st = reviews[cols].dropna(subset=['date']).sort_values('date', kind='mergesort')
    st = st.reset_index(drop=True)
    print(f'Stream simulado: {len(st):,} eventos, '
          f'de {st["date"].min():%Y-%m-%d} a {st["date"].max():%Y-%m-%d}')
    return st


# ===================== Ventanas deslizantes =====================
class SlidingWindow:
    """Mantiene solo los eventos de las ultimas `hours` horas (deque);
    count/sum/avg se actualizan incrementalmente en O(1) amortizado."""

    def __init__(self, hours):
        self.width = pd.Timedelta(hours=hours)
        self.buf = deque()
        self.count = 0
        self.sum = 0.0

    def add(self, ts, value):
        self.buf.append((ts, value))
        self.count += 1
        self.sum += value
        limit = ts - self.width
        while self.buf and self.buf[0][0] <= limit:
            _, old = self.buf.popleft()
            self.count -= 1
            self.sum -= old

    def stats(self):
        avg = self.sum / self.count if self.count else float('nan')
        return self.count, self.sum, avg


def run_windows(stream, hours_list=(1, 4, 24), snapshot_every=2000):
    """Procesa el stream y toma una foto de cada ventana cada `snapshot_every`
    eventos. Devuelve un DataFrame de trayectorias (count y avg por ventana)."""
    wins = {h: SlidingWindow(h) for h in hours_list}
    rows = []
    for i, (ts, stars) in enumerate(zip(stream['date'], stream['stars'])):
        for w in wins.values():
            w.add(ts, float(stars))
        if (i + 1) % snapshot_every == 0:
            row = {'ts': ts}
            for h, w in wins.items():
                c, s, a = w.stats()
                row[f'count_{h}h'] = c
                row[f'avg_{h}h'] = a
            rows.append(row)
    snap = pd.DataFrame(rows)
    print(f'{len(stream):,} eventos procesados, {len(snap):,} snapshots '
          f'(cada {snapshot_every:,} eventos)')
    return snap


def plot_windows(snap, hours_list=(1, 4, 24)):
    fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    colors = ['#378ADD', '#D85A30', '#4CAF50']
    for h, c in zip(hours_list, colors):
        ax[0].plot(snap['ts'], snap[f'count_{h}h'], color=c, lw=1, label=f'ventana {h}h')
        ax[1].plot(snap['ts'], snap[f'avg_{h}h'], color=c, lw=1, label=f'ventana {h}h')
    ax[0].set_ylabel('resenas en ventana (count)')
    ax[0].set_yscale('log')
    ax[0].set_title('Actividad dentro de cada ventana deslizante')
    ax[1].set_ylabel('rating promedio en ventana')
    ax[1].set_title('Rating promedio dentro de cada ventana')
    for a in ax:
        a.legend(fontsize=8)
    plt.tight_layout(); plt.show()


def temporal_patterns(stream):
    """Agregados clasicos sobre el timestamp: patron horario, semanal y anual."""
    df = stream.copy()
    df['hour'] = df['date'].dt.hour
    df['dow'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
    df.groupby('hour').size().plot(kind='bar', ax=ax[0], color='#378ADD')
    ax[0].set_title('Resenas por hora del dia'); ax[0].set_xlabel('hora')
    df.groupby('dow').size().plot(kind='bar', ax=ax[1], color='#D85A30')
    ax[1].set_title('Resenas por dia de semana (0=lun)'); ax[1].set_xlabel('dia')
    df.groupby('month').size().plot(kind='bar', ax=ax[2], color='#4CAF50')
    ax[2].set_title('Resenas por mes (estacionalidad)'); ax[2].set_xlabel('mes')
    for a in ax:
        a.set_ylabel('resenas')
    plt.tight_layout(); plt.show()


# ===================== Count-Min Sketch =====================
class CountMinSketch:
    """d filas x w columnas. Con w = ceil(e/eps) y d = ceil(ln(1/delta)):
       est(x) >= true(x)  siempre  (solo sobreestima)
       est(x) <= true(x) + eps*N   con probabilidad >= 1 - delta."""

    def __init__(self, eps=0.001, delta=0.01, seed=7):
        self.w = int(math.ceil(math.e / eps))
        self.d = int(math.ceil(math.log(1.0 / delta)))
        self.eps, self.delta = eps, delta
        self.table = np.zeros((self.d, self.w), dtype=np.int64)
        rng = np.random.default_rng(seed)
        # familia universal: h_i(x) = ((a_i*x + b_i) mod p) mod w
        self.p = 2_147_483_647
        self.a = rng.integers(1, self.p, size=self.d)
        self.b = rng.integers(0, self.p, size=self.d)
        self.N = 0

    def _hashes(self, x):
        hx = hash(x) & 0x7FFFFFFF
        return (self.a * hx + self.b) % self.p % self.w

    def add(self, x):
        self.table[np.arange(self.d), self._hashes(x)] += 1
        self.N += 1

    def estimate(self, x):
        return int(self.table[np.arange(self.d), self._hashes(x)].min())

    def memory_cells(self):
        return self.d * self.w


def cms_vs_exact(stream, key='business_id', eps=0.001, delta=0.01, top=15):
    cms = CountMinSketch(eps=eps, delta=delta)
    exact = Counter()
    for x in stream[key]:
        cms.add(x)
        exact[x] += 1
    rows = []
    for x, true in exact.most_common(top):
        est = cms.estimate(x)
        rows.append({key: x, 'exacto': true, 'cms': est, 'error': est - true})
    df = pd.DataFrame(rows)

    errs = np.array([cms.estimate(x) - c for x, c in exact.items()])
    bound = eps * cms.N
    frac_ok = float((errs <= bound).mean())
    print(f'CMS: {cms.d} filas x {cms.w:,} columnas = {cms.memory_cells():,} celdas '
          f'(vs {len(exact):,} claves exactas)')
    print(f'Garantia: error <= eps*N = {bound:.1f} con prob >= {1 - delta:.0%}')
    print(f'Observado: error <= cota en {frac_ok:.2%} de las claves | '
          f'error max = {errs.max()}, error medio = {errs.mean():.2f}')
    print(f'(el error nunca es negativo: min = {errs.min()})')
    return df, cms, exact


# ============ Flajolet-Martin / LogLog (tecnica adicional) ============
class FlajoletMartinLL:
    """Conteo aproximado de elementos DISTINTOS (variante LogLog de FM).
    Idea FM: si hasheamos n valores distintos, el maximo numero de ceros al
    final del hash crece como log2(n). Para reducir la varianza, un solo hash
    reparte cada elemento en m buckets (bits bajos) y en cada bucket se guarda
    el maximo rho (posicion del primer 1) de los bits restantes. Estimador:
    alpha * m * 2^(promedio de los m maximos). Memoria: m enteros pequenos.
    Error tipico ~ 1.3/sqrt(m) (~8% con m=256)."""

    ALPHA = 0.39701

    def __init__(self, m=256, seed=11):
        rng = np.random.default_rng(seed)
        self.m = m
        self.p = 2_147_483_647
        self.a = int(rng.integers(1, self.p))
        self.b = int(rng.integers(0, self.p))
        self.M = np.zeros(m, dtype=np.int64)

    @staticmethod
    def _rho(v):
        # posicion (1-indexada) del bit 1 menos significativo
        return (v & -v).bit_length() if v else 32

    def add(self, x):
        hx = (self.a * (hash(x) & 0x7FFFFFFF) + self.b) % self.p
        j = hx % self.m
        r = self._rho(int(hx // self.m))
        if r > self.M[j]:
            self.M[j] = r

    def estimate(self):
        return float(self.ALPHA * self.m * 2.0 ** self.M.mean())


def fm_distinct_users(stream, m=256):
    """Cuantos usuarios DISTINTOS escriben por anio, sin guardar sus IDs."""
    rows = []
    for period, g in stream.groupby(stream['date'].dt.to_period('Y')):
        fm = FlajoletMartinLL(m=m)
        for u in g['user_id']:
            fm.add(u)
        true = g['user_id'].nunique()
        est = fm.estimate()
        rows.append({'periodo': str(period), 'exacto': true,
                     'FM_LogLog': round(est),
                     'error_rel': round(abs(est - true) / true, 3)})
    df = pd.DataFrame(rows)
    print(f'FM-LogLog usa {m} enteros por periodo (vs un set con miles de '
          f'user_ids); error teorico ~ {1.3 / math.sqrt(m):.1%}')
    return df
