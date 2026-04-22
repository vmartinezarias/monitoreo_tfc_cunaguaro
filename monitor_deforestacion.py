"""
monitor_deforestacion.py
Detecta cambios de cobertura boscosa dentro del area_estudio.geojson
usando Google Earth Engine + Sentinel-2. Escribe alertas en Supabase.

Flujo:
  1. Carga area_estudio.geojson desde GitHub (geometria de analisis)
  2. Carga Predios.geojson para saber a que predio pertenece cada alerta
  3. Compara NDVI reciente vs referencia por predio
  4. Si el cambio supera el umbral, guarda alerta en Supabase
  5. Evita duplicados predio+fecha

Requisitos:
    pip install earthengine-api requests supabase python-dotenv

Variables de entorno (.env):
    SUPABASE_URL=https://xxxx.supabase.co
    SUPABASE_KEY=sb_publishable_xxx
    GEE_SERVICE_ACCOUNT=monitor-bosques@proyecto.iam.gserviceaccount.com
    GEE_KEY_FILE=gee_key.json
    GEE_PROJECT=ee-vmartinezarias

Para pruebas locales sin service account:
    Comentar bloque SA y correr ee.Authenticate() + ee.Initialize() una vez.
"""

import os
import requests
from datetime import date, timedelta
from dotenv import load_dotenv
import ee
from supabase import create_client

load_dotenv()

# ── Clientes ──────────────────────────────────────────────────────────────────

supabase = create_client(os.environ['SUPABASE_URL'], os.environ['SUPABASE_KEY'])

# ── Autenticacion GEE ─────────────────────────────────────────────────────────

KEY_FILE = os.environ.get('GEE_KEY_FILE', 'gee_key.json')
SA       = os.environ.get('GEE_SERVICE_ACCOUNT', '')
PROJECT  = os.environ.get('GEE_PROJECT', 'ee-vmartinezarias')

if SA and os.path.exists(KEY_FILE):
    credentials = ee.ServiceAccountCredentials(SA, KEY_FILE)
    ee.Initialize(credentials, project=PROJECT)
else:
    # Primera vez: correr ee.Authenticate() en terminal antes de este script
    ee.Initialize(project=PROJECT)

# ── Configuracion ─────────────────────────────────────────────────────────────

BASE_URL       = 'https://raw.githubusercontent.com/vmartinezarias/monitoreo_tfc_cunaguaro/main/'
AREA_ESTUDIO_URL = BASE_URL + 'area_estudio.geojson'
PREDIOS_URL      = BASE_URL + 'Predios.geojson'

UMBRAL_NDVI   = -0.15   # caida minima para generar alerta
DIAS_RECIENTE = 30      # ventana de imagen reciente
DIAS_REF_FIN  = 30      # fin de ventana de referencia (dias atras)
DIAS_REF_INI  = 90      # inicio de ventana de referencia (dias atras)
MAX_NUBES     = 20      # % nubosidad maxima permitida

# ── Cargar geometrias ─────────────────────────────────────────────────────────

def cargar_area_estudio() -> ee.Geometry:
    """Carga area_estudio.geojson y devuelve la geometria unida como ee.Geometry."""
    resp = requests.get(AREA_ESTUDIO_URL, timeout=30)
    resp.raise_for_status()
    gj   = resp.json()
    geoms = [ee.Geometry(f['geometry']) for f in gj.get('features', [])]
    if not geoms:
        raise ValueError("area_estudio.geojson vacio o sin features")
    if len(geoms) == 1:
        return geoms[0]
    return ee.Geometry.MultiPolygon([g for g in geoms]).dissolve(maxError=1)


def cargar_predios() -> dict:
    """Devuelve dict {id_predio: ee.Geometry} intersectando con area_estudio."""
    resp = requests.get(PREDIOS_URL, timeout=30)
    resp.raise_for_status()
    gj   = resp.json()
    predios = {}
    for feat in gj.get('features', []):
        pid  = feat['properties'].get('id_predio', 'sin_id')
        geom = ee.Geometry(feat['geometry'])
        if pid not in predios:
            predios[pid] = geom
        else:
            predios[pid] = predios[pid].union(geom, maxError=1)
    return predios

# ── Analisis NDVI ─────────────────────────────────────────────────────────────

def ndvi_medio(geom: ee.Geometry, fecha_ini: date, fecha_fin: date) -> float | None:
    """NDVI promedio Sentinel-2 en el periodo."""
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
        reducer   = ee.Reducer.mean(),
        geometry  = geom,
        scale     = 10,
        maxPixels = 1e9
    )
    val = stats.getInfo().get('ndvi')
    return round(val, 4) if val is not None else None


def area_afectada_ha(geom: ee.Geometry,
                     rec_ini: date, rec_fin: date,
                     ref_ini: date, ref_fin: date) -> float:
    """Hectareas con caida de NDVI mayor al umbral."""
    col_rec = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
               .filterBounds(geom).filterDate(str(rec_ini), str(rec_fin))
               .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', MAX_NUBES)))
    col_ref = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
               .filterBounds(geom).filterDate(str(ref_ini), str(ref_fin))
               .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', MAX_NUBES)))

    if col_rec.size().getInfo() == 0 or col_ref.size().getInfo() == 0:
        return 0.0

    ndvi_rec = col_rec.median().normalizedDifference(['B8', 'B4'])
    ndvi_ref = col_ref.median().normalizedDifference(['B8', 'B4'])
    cambio   = ndvi_rec.subtract(ndvi_ref)
    mascara  = cambio.lt(UMBRAL_NDVI)
    area     = mascara.multiply(ee.Image.pixelArea()).reduceRegion(
        reducer   = ee.Reducer.sum(),
        geometry  = geom,
        scale     = 10,
        maxPixels = 1e9
    )
    m2 = area.getInfo().get('nd', 0) or 0
    return round(m2 / 10000, 2)

# ── Utilidades ────────────────────────────────────────────────────────────────

def centroide(geom: ee.Geometry) -> tuple[float, float]:
    c   = geom.centroid(maxError=1).getInfo()
    lon, lat = c['coordinates']
    return round(lat, 6), round(lon, 6)


def clasificar_severidad(cambio: float) -> str:
    if cambio > -0.20: return 'leve'
    if cambio > -0.35: return 'moderada'
    return 'severa'


def ya_existe(predio_id: str, fecha: date) -> bool:
    resp = supabase.table('alertas_deforestacion').select('id').eq(
        'predio_id', predio_id).eq('fecha_deteccion', str(fecha)).execute()
    return len(resp.data) > 0


def guardar_alerta(predio_id, lat, lon, ndvi_a, ndvi_d, cambio, area_ha, fecha):
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
    hoy     = date.today()
    rec_ini = hoy - timedelta(days=DIAS_RECIENTE)
    rec_fin = hoy
    ref_fin = hoy - timedelta(days=DIAS_REF_FIN)
    ref_ini = hoy - timedelta(days=DIAS_REF_INI)

    print(f"Fecha hoy:    {hoy}")
    print(f"Reciente:     {rec_ini} -> {rec_fin}")
    print(f"Referencia:   {ref_ini} -> {ref_fin}")
    print(f"Proyecto GEE: {PROJECT}")

    # Cargar area de estudio como limite de analisis
    print("\nCargando area_estudio.geojson...", end=" ", flush=True)
    area_estudio = cargar_area_estudio()
    print("OK")

    # Cargar predios dentro del area
    print("Cargando predios...", end=" ", flush=True)
    predios = cargar_predios()
    print(f"{len(predios)} predios")

    nuevas = 0
    for pid, geom_predio in predios.items():
        print(f"\n  [{pid}]", end=" ", flush=True)

        if ya_existe(pid, hoy):
            print("ya procesado hoy, saltando.")
            continue

        # Intersectar predio con area de estudio para limitar el analisis
        geom = geom_predio.intersection(area_estudio, maxError=1)

        ndvi_ref = ndvi_medio(geom, ref_ini, ref_fin)
        ndvi_rec = ndvi_medio(geom, rec_ini, rec_fin)

        if ndvi_ref is None or ndvi_rec is None:
            print("sin imagenes disponibles.")
            continue

        cambio = round(ndvi_rec - ndvi_ref, 4)
        print(f"NDVI ref={ndvi_ref:.3f} rec={ndvi_rec:.3f} D={cambio:+.3f}", end=" ")

        if cambio >= UMBRAL_NDVI:
            print("-> sin alerta.")
            continue

        area_ha  = area_afectada_ha(geom, rec_ini, rec_fin, ref_ini, ref_fin)
        lat, lon = centroide(geom)
        guardar_alerta(pid, lat, lon, ndvi_ref, ndvi_rec, cambio, area_ha, hoy)
        nuevas += 1
        sev = clasificar_severidad(cambio)
        print(f"-> ALERTA ({sev}, {area_ha} ha).")

    print(f"\nListo. {nuevas} alertas de deforestacion nuevas guardadas.")


if __name__ == '__main__':
    main()
