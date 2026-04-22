"""
monitor_deforestacion.py
Detecta cambios de cobertura boscosa en predios registrados usando
Google Earth Engine + Sentinel-2. Escribe alertas en Supabase.

Flujo:
  1. Carga Predios.geojson desde GitHub
  2. Para cada predio, compara NDVI de imagen reciente vs imagen de referencia
  3. Si el cambio supera el umbral, registra una alerta en Supabase
  4. Evita duplicados por predio+fecha

Requisitos:
    pip install earthengine-api requests supabase python-dotenv

Variables de entorno (.env):
    SUPABASE_URL=https://xxxx.supabase.co
    SUPABASE_KEY=sb_publishable_xxx
    GEE_SERVICE_ACCOUNT=monitor-bosques@proyecto.iam.gserviceaccount.com
    GEE_KEY_FILE=gee_key.json   # ruta al archivo JSON de la service account

Alternativa sin service account (para pruebas locales):
    Comentar el bloque de autenticación con service account y usar:
    ee.Authenticate()
    ee.Initialize(project='tu-proyecto-gee')
"""

import os
import json
import math
import requests
from datetime import date, timedelta
from dotenv import load_dotenv
import ee
from supabase import create_client

load_dotenv()

# ── Clientes ──────────────────────────────────────────────────────────────────

supabase = create_client(os.environ['SUPABASE_URL'], os.environ['SUPABASE_KEY'])

# ── Autenticación GEE ─────────────────────────────────────────────────────────
# Opción A: Service Account (recomendada para automatización en servidor)
KEY_FILE = os.environ.get('GEE_KEY_FILE', 'gee_key.json')
SA       = os.environ.get('GEE_SERVICE_ACCOUNT', '')

if SA and os.path.exists(KEY_FILE):
    credentials = ee.ServiceAccountCredentials(SA, KEY_FILE)
    ee.Initialize(credentials)
else:
    # Opción B: autenticación interactiva (para pruebas locales)
    # La primera vez corre ee.Authenticate() manualmente
    ee.Initialize(project=os.environ.get('GEE_PROJECT', 'ee-monitoreo-chameza'))

# ── Configuración ─────────────────────────────────────────────────────────────

GEOJSON_URL   = 'https://raw.githubusercontent.com/vmartinezarias/monitoreo_tfc_cunaguaro/main/Predios.geojson'

# Umbral de caída de NDVI para considerar posible deforestación
# -0.15 = caída del 15% en vigor vegetal → señal moderada
UMBRAL_NDVI   = -0.15

# Días hacia atrás para imagen "reciente"
DIAS_RECIENTE = 30

# Días hacia atrás para imagen de "referencia" (antes del evento)
DIAS_REF_FIN  = 30
DIAS_REF_INI  = 90

# Nubosidad máxima permitida (%)
MAX_NUBES     = 20

# ── Cargar predios ────────────────────────────────────────────────────────────

def cargar_predios() -> dict:
    """Devuelve un dict {id_predio: ee.Geometry} con la unión de polígonos por predio."""
    resp = requests.get(GEOJSON_URL, timeout=30)
    resp.raise_for_status()
    gj = resp.json()

    predios = {}
    for feat in gj.get('features', []):
        pid  = feat['properties'].get('id_predio', 'sin_id')
        geom = ee.Geometry(feat['geometry'])
        if pid not in predios:
            predios[pid] = geom
        else:
            predios[pid] = predios[pid].union(geom)
    return predios

# ── Calcular NDVI medio en un área ───────────────────────────────────────────

def ndvi_medio(geom: ee.Geometry, fecha_ini: date, fecha_fin: date) -> float | None:
    """Calcula el NDVI promedio de Sentinel-2 en el período dado."""
    col = (
        ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(geom)
        .filterDate(str(fecha_ini), str(fecha_fin))
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', MAX_NUBES))
    )
    if col.size().getInfo() == 0:
        return None

    mediana = col.median()
    ndvi    = mediana.normalizedDifference(['B8', 'B4']).rename('ndvi')
    stats   = ndvi.reduceRegion(
        reducer  = ee.Reducer.mean(),
        geometry = geom,
        scale    = 10,
        maxPixels= 1e9
    )
    val = stats.getInfo().get('ndvi')
    return round(val, 4) if val is not None else None

# ── Calcular área afectada ────────────────────────────────────────────────────

def area_afectada_ha(geom: ee.Geometry, fecha_ini: date, fecha_fin: date,
                     fecha_ref_ini: date, fecha_ref_fin: date) -> float:
    """Estima hectáreas con caída de NDVI > umbral."""
    col_rec = (
        ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(geom).filterDate(str(fecha_ini), str(fecha_fin))
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', MAX_NUBES))
    )
    col_ref = (
        ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(geom).filterDate(str(fecha_ref_ini), str(fecha_ref_fin))
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', MAX_NUBES))
    )
    if col_rec.size().getInfo() == 0 or col_ref.size().getInfo() == 0:
        return 0.0

    ndvi_rec = col_rec.median().normalizedDifference(['B8','B4'])
    ndvi_ref = col_ref.median().normalizedDifference(['B8','B4'])
    cambio   = ndvi_rec.subtract(ndvi_ref)
    mascara  = cambio.lt(UMBRAL_NDVI)

    area = mascara.multiply(ee.Image.pixelArea()).reduceRegion(
        reducer  = ee.Reducer.sum(),
        geometry = geom,
        scale    = 10,
        maxPixels= 1e9
    )
    m2 = area.getInfo().get('nd', 0) or 0
    return round(m2 / 10000, 2)

# ── Calcular centroide aproximado ────────────────────────────────────────────

def centroide(geom: ee.Geometry) -> tuple[float, float]:
    c   = geom.centroid(maxError=1).getInfo()
    lon = c['coordinates'][0]
    lat = c['coordinates'][1]
    return round(lat, 6), round(lon, 6)

# ── Clasificar severidad ──────────────────────────────────────────────────────

def clasificar_severidad(cambio: float) -> str:
    if cambio > -0.20:   return 'leve'
    if cambio > -0.35:   return 'moderada'
    return 'severa'

# ── Verificar duplicado ───────────────────────────────────────────────────────

def ya_existe(predio_id: str, fecha: date) -> bool:
    resp = supabase.table('alertas_deforestacion').select('id').eq(
        'predio_id', predio_id).eq('fecha_deteccion', str(fecha)).execute()
    return len(resp.data) > 0

# ── Guardar alerta ────────────────────────────────────────────────────────────

def guardar_alerta(predio_id: str, lat: float, lon: float,
                   ndvi_a: float, ndvi_d: float, cambio: float,
                   area_ha: float, fecha: date):
    supabase.table('alertas_deforestacion').insert({
        'predio_id':        predio_id,
        'fecha_deteccion':  str(fecha),
        'latitud':          lat,
        'longitud':         lon,
        'ndvi_antes':       ndvi_a,
        'ndvi_despues':     ndvi_d,
        'cambio_ndvi':      cambio,
        'area_afectada_ha': area_ha,
        'severidad':        clasificar_severidad(cambio),
        'estado':           'nueva'
    }).execute()

# ── Flujo principal ───────────────────────────────────────────────────────────

def main():
    hoy         = date.today()
    # Ventana reciente: últimos DIAS_RECIENTE días
    rec_ini     = hoy - timedelta(days=DIAS_RECIENTE)
    rec_fin     = hoy
    # Ventana de referencia: período anterior estable
    ref_fin     = hoy - timedelta(days=DIAS_REF_FIN)
    ref_ini     = hoy - timedelta(days=DIAS_REF_INI)

    print(f'Fecha: {hoy}')
    print(f'Reciente:    {rec_ini} → {rec_fin}')
    print(f'Referencia:  {ref_ini} → {ref_fin}')

    predios = cargar_predios()
    print(f'{len(predios)} predios cargados.')

    nuevas = 0
    for pid, geom in predios.items():
        print(f'  Procesando {pid}...', end=' ', flush=True)

        if ya_existe(pid, hoy):
            print('ya procesado hoy.')
            continue

        ndvi_ref = ndvi_medio(geom, ref_ini, ref_fin)
        ndvi_rec = ndvi_medio(geom, rec_ini, rec_fin)

        if ndvi_ref is None or ndvi_rec is None:
            print('sin imágenes disponibles.')
            continue

        cambio = round(ndvi_rec - ndvi_ref, 4)
        print(f'NDVI ref={ndvi_ref:.3f} rec={ndvi_rec:.3f} Δ={cambio:+.3f}', end=' ')

        if cambio >= UMBRAL_NDVI:
            print('→ sin alerta.')
            continue

        # Hay caída significativa
        area_ha = area_afectada_ha(geom, rec_ini, rec_fin, ref_ini, ref_fin)
        lat, lon = centroide(geom)
        guardar_alerta(pid, lat, lon, ndvi_ref, ndvi_rec, cambio, area_ha, hoy)
        nuevas += 1
        print(f'→ ALERTA ({clasificar_severidad(cambio)}, {area_ha} ha).')

    print(f'\nListo. {nuevas} alertas de deforestación nuevas guardadas.')

if __name__ == '__main__':
    main()
