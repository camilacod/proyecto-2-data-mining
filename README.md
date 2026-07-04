# Proyecto 2 — Minería de Datos Masivos

> Grafos · Clustering · Recomendación · Streams · Reducción de dimensionalidad

**Curso:** Data Mining — UTEC (2026-I)
**Profesor:** Heider Sanchez
**Equipo:**

| Alumno |
| --- |
| Christian Frisancho |
| Camila Rodriguez |
| Marcelo Zuloeta |

Pipeline integral de minería de datos sobre el **Yelp Open Dataset** (6.9M reseñas, 150k negocios, 4.35 GB). Todos los algoritmos están implementados **desde cero** — sin scikit-learn, sin networkx, sin surprise — usando únicamente `numpy`, `pandas` y `matplotlib`, de modo que el código refleja directamente los principios de cada técnica.

---

## Arquitectura

```
                        yelp-dataset (kagglehub, 4.07 GB)
                                      |
                                      v
              +---------------------------------------------+
              |  P1: streaming JSON + muestreo estratificado |
              |      limpieza -> artifacts/*.parquet         |
              +---------------------------------------------+
                                      |
        +----------+----------+-------+------+----------+----------+
        v          v          v              v          v          v
      P2         P3          P4             P5         P6         P7
    grafos    clustering  recomendacion  streams    PCA/SVD    analisis
   PageRank   K-Means++   CF item-item   ventanas   eigen-     critico
   HITS       DBSCAN      TF-IDF         CMS        decomp.    y etico
   Louvain    silueta     hibrido        FM-LogLog  SVD trunc.
```

Cada notebook escribe sus módulos Python con `%%writefile` y los carga con un helper `load_mod()` (basado en `importlib.util.spec_from_file_location`), lo que evita los problemas de caché de imports de Databricks. Las Partes II–VII consumen los parquet de `artifacts/` generados por la Parte I: no re-streamean el dataset crudo.

## Estructura del repositorio

```
proyecto-2-data-mining/
├── proyecto2-datamining-p1.ipynb   # Parte I   — preprocesamiento y EDA
├── proyecto2-datamining-p2.ipynb   # Parte II  — grafos, ranking y comunidades
├── proyecto2-datamining-p3.ipynb   # Parte III — clustering
├── proyecto2-datamining-p4.ipynb   # Parte IV  — sistemas de recomendación
├── proyecto2-datamining-p5.ipynb   # Parte V   — minería de flujos de datos
├── proyecto2-datamining-p6.ipynb   # Parte VI  — reducción de dimensionalidad
├── proyecto2-datamining-p7.ipynb   # Parte VII — análisis crítico y ético
│
├── informe.tex / informe.pdf       # reporte final (formato IEEE)
├── imgs/                           # figuras de los notebooks usadas por el informe
│
├── config.py                       # rutas, parámetros de muestreo, umbrales
├── preprocessing.py                # streaming JSON + muestreo estratificado
├── cleaning.py                     # limpieza: duplicados, nulos, outliers
├── eda.py                          # estadísticas descriptivas y gráficas
├── graphs.py                       # construcción de grafos (listas de adyacencia)
├── ranking.py                      # PageRank y HITS (power iteration)
├── communities.py                  # Louvain (optimización de modularidad)
├── graphviz.py                     # layout force-directed para visualización
├── features.py                     # matriz de features + estandarización
├── kmeans.py                       # K-Means++ con selección de k por silueta
├── dbscan.py                       # DBSCAN sobre matriz de distancias
├── compare.py                      # comparativa K-Means vs DBSCAN
├── recommenders.py                 # CF item-item, TF-IDF content-based, híbrido
├── streaming.py                    # ventanas deslizantes, Count-Min Sketch, FM-LogLog
├── dimensionality_reduction.py     # PCA (eigendecomposición) y SVD truncada
├── ethics.py                       # métricas de concentración, Gini, cadenas vs indep.
│
└── artifacts/                      # parquet intermedios (cache del pipeline)
    ├── sample_business.parquet
    ├── sample_reviews.parquet
    ├── sample_users.parquet
    ├── sample_checkins.parquet
    └── sample_tips.parquet
```

Los `.py` de módulos en la raíz son los mismos que cada notebook escribe con `%%writefile`; se versionan para poder leerlos y diffearlos fuera del notebook.

## Dependencias

| Paquete | Uso |
| --- | --- |
| `python >= 3.10` | runtime |
| `numpy` | álgebra lineal, eigendecomposición, hashing vectorizado |
| `pandas` | manipulación tabular, agregaciones |
| `matplotlib` | todas las visualizaciones |
| `pyarrow` | lectura/escritura de parquet |
| `kagglehub` | descarga del Yelp Open Dataset (solo Parte I, primera corrida) |

En Databricks (DBR 14+) todo excepto `kagglehub` viene preinstalado; la Parte I lo instala con `%pip install kagglehub`. Para una corrida local:

```bash
python -m venv .venv && source .venv/bin/activate
pip install numpy pandas matplotlib pyarrow kagglehub
```

Restricción del curso: **ningún algoritmo usa librerías de ML/grafos**. `numpy`/`pandas` se emplean solo como soporte numérico y tabular.

## Entorno de ejecución

El proyecto fue ejecutado íntegramente en **Databricks** sobre el siguiente cluster:

| Componente | Especificación |
| --- | --- |
| Runtime | Databricks Runtime 18 ML · Spark 4.1.0 · Scala 2.13 |
| Driver | `Standard_E8ads_v7` · 64 GB RAM · 8 cores |
| Workers | 2 × `Standard_E8ads_v7` · 128 GB RAM · 16 cores en total |

Nota: el pipeline corre en el **driver** con Python puro (`numpy`/`pandas`); Spark no se usa para el cómputo de los algoritmos. Cualquier cluster con ≥ 32 GB de RAM en el driver es suficiente (el paso más pesado es el streaming del JSON crudo de la Parte I).

## Ejecución

### 1. Credenciales de Kaggle (solo primera corrida)

La Parte I descarga el dataset vía `kagglehub`. Se requiere un API token de Kaggle ([kaggle.com/settings](https://www.kaggle.com/settings) → *Create New Token*) y haber aceptado los términos del [Yelp Dataset](https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset) en la web. Luego, antes de la celda de descarga:

```python
import os
os.environ["KAGGLE_USERNAME"] = "<usuario>"
os.environ["KAGGLE_KEY"] = "<token>"
```

En Databricks se recomienda usar secrets en lugar de valores en claro:

```python
os.environ["KAGGLE_USERNAME"] = dbutils.secrets.get("kaggle", "username")
os.environ["KAGGLE_KEY"] = dbutils.secrets.get("kaggle", "key")
```

Una vez descargado, `config.py` resuelve el dataset desde el cache local de kagglehub (`~/.cache/kagglehub`) sin red ni credenciales.

### 2. Orden de ejecución

Importar los siete notebooks a **una misma carpeta del Workspace** de Databricks (el pipeline usa `os.getcwd()` como raíz, y con DBR 14+ el cwd es la carpeta del notebook, así que `artifacts/` persiste entre reinicios de cluster).

| Orden | Notebook | Contenido | Depende de |
| --- | --- | --- | --- |
| 1 | `p1` | Streaming del JSON crudo, muestreo estratificado por ciudad, limpieza, EDA, grafo usuario-negocio | dataset crudo |
| 2 | `p2` | Grafo de amistades, PageRank, HITS, Louvain, visualización | `artifacts/` |
| 3 | `p3` | Features de negocios, K-Means++ (k por silueta), DBSCAN, comparativa | `artifacts/` |
| 4 | `p4` | CF item-item, content-based TF-IDF, híbrido; RMSE/MAE, P@K, R@K, NDCG@K | `artifacts/` |
| 5 | `p5` | Ventanas deslizantes, Count-Min Sketch, Flajolet-Martin/LogLog, detección de ráfagas | `artifacts/` |
| 6 | `p6` | PCA sobre features de negocios, SVD truncada de la matriz usuario-negocio | `artifacts/` + módulos de p3/p4 |
| 7 | `p7` | Complejidad teórica + benchmarks, sesgos de representación, spam, equidad | `artifacts/` + módulos de p1/p3/p5 |

La Parte I es la única que toca el dataset crudo (~10 GB descomprimido); tarda unos minutos en streamear los cinco JSON. Las demás partes cargan los parquet en segundos. Cada notebook es re-ejecutable de punta a punta: la primera celda regenera los módulos y `config` detecta el cache existente.

### 3. Reproducibilidad

- Muestreo estratificado con `SAMPLE_SEED = 42` y objetivo de `SAMPLE_TARGET_BUSINESS = 3000` (ver `config.py`): la muestra resultante es de **2,943 negocios, 125,744 reseñas y 103,447 usuarios**, con proporciones por ciudad idénticas al dataset completo.
- Los algoritmos con inicialización aleatoria (K-Means++, SVD, FM-LogLog) fijan semilla donde el resultado se reporta; FM-LogLog usa hashes por sesión, por lo que su estimación puntual varía dentro del error teórico (~8%).
- `artifacts/` actúa como cache idempotente: si los parquet existen, la Parte I no re-streamea.

## Resultados principales

| Parte | Resultado |
| --- | --- |
| II | LCC con PageRank ~ amistades r=0.98; Louvain: 140 comunidades, Q=0.656 |
| III | K-Means++ k=5 (silueta 0.167); DBSCAN: 6 clusters + 190 outliers (6.5%) |
| IV | CF item-item NDCG@10=0.070 vs content 0.019 y top-popular 0.029; mejor híbrido α=0.6 |
| V | CMS: error ≤ cota εN en 100% de las claves con ~8x menos memoria; FM-LogLog: 256 enteros, error ~4-8% |
| VI | 20/28 componentes PCA explican 90.1% de varianza; SVD k=100: compresión 12x reteniendo 56% |
| VII | Gini de reseñas por negocio 0.666; 88% de usuarios con 1 reseña; cadenas: rating 2.60 vs 3.63 de independientes |

El análisis completo, con la lectura de cada resultado, está en los notebooks.
