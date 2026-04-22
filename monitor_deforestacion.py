"""
monitor_deforestacion.py  v2.1
Detecta cambios de cobertura boscosa usando Google Earth Engine + Sentinel-2.
Escribe alertas en Supabase tabla `alertas_deforestacion`.

UNIDADES DE MONITOREO:
  1. Predios individuales (Predios.geojson)          -> predio_id = id_predio
  2. Area de estudio completa (area_estudio.geojson)  -> predio_id = "area_estudio_N"
  3. Nucleos boscosos (Nucleos_boscosos.geojson)      -> predio_id = "nucleo_N"
  4. Buffer por predio (radio configurable en km)     -> predio_id = "buffer_<id>_<km>km"

CAMBIOS vs v1:
  - MAX_NUBES subido a 80% + mascara SCL pixel a pixel (limpieza fina)
  - DIAS_RECIENTE ampliado a 45 dias
  - Ventana referencia ampliada (DIAS_REF_INI = 180 dias)
  - Logs detallados: cuenta imagenes, muestra NDVI aunque no haya alerta
  - Captura excepciones por unidad sin detener el proceso completo
  - Variable BUFFER_KM para buffer adicional alrededor de cada predio
  - Variables PROCESAR_* para activar/desactivar cada tipo de unidad

Requisitos:
    pip install earthengine-api requests supabase python-dotenv

Variables de entorno (.env o GitHub Secrets):
    SUPABASE_URL, SUPABASE_KEY
    GEE_SERVICE_ACCOUNT, GEE_KEY_FILE   <- servidor
    GEE_PROJECT                          <- fallback local
"""

import os
import requests
from datetime import date, timedelta
from dotenv import load_dotenv
import ee
from supabase import create_client

load_dotenv()

# =============================================================================
# CONFIGURACION — ajusta aqui sin tocar el resto del script
# =============================================================================

UMBRAL_NDVI          = -0.10   # caida minima para generar alerta
DIAS_RECIENTE        = 45      # ventana imagen reciente (dias atras)
DIAS_REF_FIN         = 45      # la referencia termina hace X dias
DIAS_REF_INI         = 180     # la referencia empieza hace X dias
MAX_NUBES_COL        = 80      # % nubes para filtro de coleccion (SCL hace la limpieza fina)
BUFFER_KM            = 2       # radio buffer alrededor de cada predio

PROCESAR_PREDIOS         = True
PROCESAR_AREA_ESTUDIO    = True
PROCESAR_NUCLEOS         = True
PROCESAR_BUFFER_PREDIOS  = True

# =============================================================================
# URLs repositorio
# =============================================================================
BASE_URL        = 'https://raw.githubusercontent.com/vmartinezarias/monitoreo_tfc_cunaguaro/main/'
GEOJSON_PREDIOS = BASE_URL + 'Predios.geojson'
GEOJSON_ESTUDIO = BASE_URL + 'area_estudio.geojson'
GEOJSON_NUCLEOS = BASE_URL + 'Nucleos_boscosos.geojson'

# =============================================================================
# Clientes
# =============================================================================
supabase = create_client(os.environ['SUPABASE_URL'], os.environ['SUPABASE_KEY'])

KEY_FILE = os.environ.get('GEE_KEY_FILE', 'gee_key.json')
SA       = os.environ.get('GEE_SERVICE_ACCOUNT', '')

if SA and os.path.exists(KEY_FILE):
    print(f'[GEE] Service account: {SA}')
    credentials = ee.ServiceAccountCredentials(SA, KEY_FILE)
    ee.Initialize(credentials)
else:
    print('[GEE] Autenticacion interactiva (modo local)')
    ee.Initialize(project=os.environ.get('GEE_PROJECT', 'ee-vmartinezarias'))

print('[GEE] Inicializado.\n')

# =============================================================================
# Helpers GEE
# =============================================================================

def mask_s2_scl(image):
    """Enmascara nubes/sombras usando banda SCL de Sentinel-2 SR."""
    scl = image.select('SCL')
    # Excluir: 3=sombra, 7=nubes baja prob, 8=nubes alta prob, 9=cirrus, 10=cirrus
    mask = (scl.neq(3).And(scl.neq(7)).And(scl.neq(8))
               .And(scl.neq(9)).And(scl.neq(10)))
    return image.updateMask(mask)


def coleccion_s2(geom, fecha_ini, fecha_fin):
    return (
        ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(geom)
        .filterDate(str(fecha_ini), str(fecha_fin))
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', MAX_NUBES_COL))
        .map(mask_s2_scl)
    )


def ndvi_medio(geom, fecha_ini, fecha_fin):
    """NDVI promedio en el area y periodo. Devuelve None si no hay imagenes."""
    col = coleccion_s2(geom, fecha_ini, fecha_fin)
    n   = col.size().getInfo()
    print(f'[{fecha_ini} -> {fecha_fin}] {n} imagenes', end='  ')
    if n == 0:
        return None
    ndvi  = col.median().normalizedDifference(['B8', 'B4']).rename('ndvi')
    stats = ndvi.reduceRegion(
        reducer=ee.Reducer.mean(), geometry=geom, scale=10, maxPixels=1e9
    )
    val = stats.getInfo().get('ndvi')
    return round(val, 4) if val is not None else None


def area_afectada_ha(geom, rec_ini, rec_fin, ref_ini, ref_fin):
    """Hectareas con caida de NDVI > umbral."""
    col_rec = coleccion_s2(geom, rec_ini, rec_fin)
    col_ref = coleccion_s2(geom, ref_ini, ref_fin)
    if col_rec.size().getInfo() == 0 or col_ref.size().getInfo() == 0:
        return 0.0
    ndvi_rec = col_rec.median().normalizedDifference(['B8', 'B4'])
    ndvi_ref = col_ref.median().normalizedDifference(['B8', 'B4'])
    mascara  = ndvi_rec.subtract(ndvi_ref).lt(UMBRAL_NDVI)
    area = mascara.multiply(ee.Image.pixelArea()).reduceRegion(
        reducer=ee.Reducer.sum(), geometry=geom, scale=10, maxPixels=1e9
    )
    m2 = area.getInfo().get('nd', 0) or 0
    return round(m2 / 10000, 2)


def centroide(geom):
    c = geom.centroid(maxError=1).getInfo()
    lon, lat = c['coordinates']
    return round(lat, 6), round(lon, 6)


def clasificar_severidad(cambio):
    if cambio > -0.20:  return 'leve'
    if cambio > -0.35:  return 'moderada'
    return 'severa'


def ya_existe(predio_id, fecha):
    r = supabase.table('alertas_deforestacion').select('id') \
        .eq('predio_id', predio_id).eq('fecha_deteccion', str(fecha)).execute()
    return len(r.data) > 0


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

# =============================================================================
# Carga de geometrias
# =============================================================================

def cargar_geojson(url):
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()


def unidades_predios(gj):
    """Agrupa poligonos por id_predio -> lista de (id, geom)."""
    grupos = {}
    for f in gj.get('features', []):
        pid  = f['properties'].get('id_predio', 'sin_id')
        geom = ee.Geometry(f['geometry'])
        grupos[pid] = grupos[pid].union(geom, maxError=1) if pid in grupos else geom
    return list(grupos.items())


def unidades_capa(gj, prefijo):
    """Una unidad por feature, con ID derivado de propiedades."""
    result = []
    for i, f in enumerate(gj.get('features', [])):
        props  = f['properties']
        nombre = props.get('nombre') or props.get('id') or props.get('name') or str(i + 1)
        pid    = f'{prefijo}_{nombre}'.replace(' ', '_')[:80]
        result.append((pid, ee.Geometry(f['geometry'])))
    return result


def unidades_buffer(predios, km):
    return [
        (f'buffer_{pid}_{km}km', geom.buffer(km * 1000, maxError=100))
        for pid, geom in predios
    ]

# =============================================================================
# Procesar una unidad
# =============================================================================

def procesar_unidad(pid, geom, rec_ini, rec_fin, ref_ini, ref_fin, hoy):
    print(f'\n  Unidad: {pid}')
    if ya_existe(pid, hoy):
        print('    -> ya procesado hoy.')
        return 'duplicado'
    try:
        print('    ref:  ', end='')
        ndvi_ref = ndvi_medio(geom, ref_ini, ref_fin)
        print(f'| NDVI={ndvi_ref}')

        print('    rec:  ', end='')
        ndvi_rec = ndvi_medio(geom, rec_ini, rec_fin)
        print(f'| NDVI={ndvi_rec}')

        if ndvi_ref is None or ndvi_rec is None:
            print('    -> sin imagenes suficientes.')
            return 'sin_imagenes'

        cambio = round(ndvi_rec - ndvi_ref, 4)
        print(f'    NDVI ref={ndvi_ref:.4f}  rec={ndvi_rec:.4f}  delta={cambio:+.4f}  umbral={UMBRAL_NDVI}')

        if cambio >= UMBRAL_NDVI:
            print('    -> sin alerta (cambio normal).')
            return 'sin_alerta'

        area_ha      = area_afectada_ha(geom, rec_ini, rec_fin, ref_ini, ref_fin)
        lat, lon     = centroide(geom)
        sev          = clasificar_severidad(cambio)
        guardar_alerta(pid, lat, lon, ndvi_ref, ndvi_rec, cambio, area_ha, hoy)
        print(f'    *** ALERTA: {sev} | {area_ha} ha ***')
        return f'alerta_{sev}'

    except Exception as e:
        print(f'    ERROR: {e}')
        return 'error'

# =============================================================================
# Main
# =============================================================================

def main():
    hoy     = date.today()
    rec_ini = hoy - timedelta(days=DIAS_RECIENTE)
    rec_fin = hoy
    ref_fin = hoy - timedelta(days=DIAS_REF_FIN)
    ref_ini = hoy - timedelta(days=DIAS_REF_INI)

    sep = '=' * 65
    print(sep)
    print(f'Monitor Deforestacion v2  —  {hoy}')
    print(f'  Reciente  : {rec_ini} -> {rec_fin}')
    print(f'  Referencia: {ref_ini} -> {ref_fin}')
    print(f'  Umbral NDVI: {UMBRAL_NDVI}  |  Max nubes coleccion: {MAX_NUBES_COL}% + SCL')
    print(sep)

    todas: list[tuple[str, object]] = []

    if PROCESAR_PREDIOS or PROCESAR_BUFFER_PREDIOS:
        print('\nCargando Predios.geojson...')
        predios = unidades_predios(cargar_geojson(GEOJSON_PREDIOS))
        print(f'  {len(predios)} predios.')
        if PROCESAR_PREDIOS:
            todas += predios
        if PROCESAR_BUFFER_PREDIOS:
            buf = unidades_buffer(predios, BUFFER_KM)
            todas += buf
            print(f'  {len(buf)} buffers de {BUFFER_KM} km.')

    if PROCESAR_AREA_ESTUDIO:
        print('\nCargando area_estudio.geojson...')
        est = unidades_capa(cargar_geojson(GEOJSON_ESTUDIO), 'area_estudio')
        todas += est
        print(f'  {len(est)} unidades de area de estudio.')

    if PROCESAR_NUCLEOS:
        print('\nCargando Nucleos_boscosos.geojson...')
        nuc = unidades_capa(cargar_geojson(GEOJSON_NUCLEOS), 'nucleo')
        todas += nuc
        print(f'  {len(nuc)} nucleos boscosos.')

    print(f'\nTotal unidades: {len(todas)}')
    print(sep)

    conteo = {'duplicado': 0, 'sin_imagenes': 0, 'sin_alerta': 0, 'error': 0}
    nuevas = 0

    for pid, geom in todas:
        res = procesar_unidad(pid, geom, rec_ini, rec_fin, ref_ini, ref_fin, hoy)
        if res.startswith('alerta_'):
            nuevas += 1
        elif res in conteo:
            conteo[res] += 1

    print(f'\n{sep}')
    print('RESUMEN FINAL')
    print(f'  Alertas nuevas    : {nuevas}')
    print(f'  Sin imagenes      : {conteo["sin_imagenes"]}')
    print(f'  Sin cambio        : {conteo["sin_alerta"]}')
    print(f'  Ya procesadas hoy : {conteo["duplicado"]}')
    print(f'  Errores           : {conteo["error"]}')
    print(sep)

    if conteo['sin_imagenes'] == len(todas):
        print('\nADVERTENCIA: Todas las unidades retornaron "sin imagenes".')
        print('Ampliar DIAS_RECIENTE o revisar autenticacion GEE.')


if __name__ == '__main__':
    main()
