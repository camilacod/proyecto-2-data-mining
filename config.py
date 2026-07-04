
import os
import glob

# Alcance de los datos: todas las ciudades, muestreo estratificado por ciudad.
SCOPE = "sample"

# parametros del muestreo estratificado
SAMPLE_TARGET_BUSINESS = 3000   # tamano objetivo de la muestra de negocios
SAMPLE_SEED = 42                # semilla fija -> muestra reproducible

CITY_LABEL = "Yelp global (muestra estratificada por ciudad)"

def _kagglehub_root():
    """Ruta local del dataset descargado con kagglehub. Primero busca el cache
    en disco (no requiere red ni credenciales); solo si no existe llama a la
    API de Kaggle, para lo cual deben estar KAGGLE_USERNAME / KAGGLE_KEY
    (ver la celda de descarga del notebook de la Parte I)."""
    cache_base = os.environ.get(
        "KAGGLEHUB_CACHE", os.path.expanduser("~/.cache/kagglehub"))
    versions = os.path.join(
        cache_base, "datasets", "yelp-dataset", "yelp-dataset", "versions")
    if os.path.isdir(versions):
        found = sorted(glob.glob(os.path.join(versions, "*")))
        if found and glob.glob(os.path.join(found[-1], "*.json")):
            return found[-1]
    import kagglehub
    return kagglehub.dataset_download("yelp-dataset/yelp-dataset")


# dataset via kagglehub (cache local primero)
DATA_ROOT = _kagglehub_root()

def _find_raw(keyword):
    m = glob.glob(os.path.join(DATA_ROOT, f"**/*{keyword}*.json"), recursive=True)
    return m[0] if m else None

BUSINESS = _find_raw("business")
REVIEW   = _find_raw("review")
USER     = _find_raw("user")
CHECKIN  = _find_raw("checkin")
TIP      = _find_raw("tip")

def _locate_artifacts():
    # Carpeta 'artifacts' junto al notebook. Con DBR 14+ el cwd es la carpeta
    # del Workspace, asi que persiste entre reinicios del cluster y los
    # notebooks de las Partes II-VI la encuentran sin adjuntar nada.
    local = os.path.join(os.getcwd(), "artifacts")
    os.makedirs(local, exist_ok=True)
    return local

ARTIFACTS = _locate_artifacts()

MIN_USERS = 3000
MIN_LCC_FRACTION = 0.20
MIN_AVG_DEGREE = 1.0
