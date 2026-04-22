"""
monitor_deforestacion.py  v3
"""
import os
import json
import requests
from datetime import date, timedelta
from dotenv import load_dotenv
import ee
from supabase import create_client

load_dotenv()

UMBRAL_NDVI     = -0.10
DIAS_RECIENTE   = 45
DIAS_REF_FIN    = 45
DIAS_REF_INI    = 180
MAX_NUBES_COL   = 80
BUFFER_KM       = 2

PROCESAR_PREDIOS         = True
PROCESAR_AREA_ESTUDIO    = True
PROCESAR_NUCLEOS         = True
PROCESAR_BUFFER_PREDIOS  = True

BASE_URL        = 'https://raw.githubusercontent.com/vmartinezarias/monitoreo_tfc_cunaguaro/main/'
GEOJSON_PREDIOS = BASE_URL + 'Predios.geojson'
GEOJSON_ESTUDIO = BASE_URL + 'area_estudio.geojson'
GEOJSON_NUCLEOS = BASE_URL + 'Nucleos_boscosos.geojson'

# =============================================================================
# AUTENTICACION GEE CON DIAGNOSTICO COMPLETO
# =============================================================================
print('=' * 65)
print('DIAGNOSTICO GEE')

KEY_FILE = os.environ.get('GEE_KEY_FILE', 'gee_key.json')
SA       = os.environ.get('GEE_SERVICE_ACCOUNT', '')
PROJECT  = os.environ.get('GEE_PROJECT', 'ee-vmartinezarias')

print(f'  GEE_SERVICE_ACCOUNT : {SA or "(no definido)"}')
print(f'  GEE_KEY_FILE        : {KEY_FILE}')
print(f'  GEE_PROJECT         : {PROJECT}')
print(f'  Key file existe     : {os.path.exists(KEY_FILE)}')

if os.path.exists(KEY_FILE):
    try:
        with open(KEY_FILE) as f:
            key_data = json.load(f)
        print(f'  Key type            : {key_data.get("type")}')
        print(f'  Key project_id      : {key_data.get("project_id")}')
        print(f'  Key client_email    : {key_data.get("client_email")}')
    except Exception as e:
        print(f'  ERROR leyendo key   : {e}')

try:
    if SA and os.path.exists(KEY_FILE):
        print('\n[GEE] Usando ServiceAccountCredentials...')
        credentials = ee.ServiceAccountCredentials(SA, KEY_FILE)
        ee.Initialize(credentials, project=PROJECT)
    else:
        print('\n[GEE] ADVERTENCIA: Sin service account, usando autenticacion interactiva...')
        ee.Initialize(project=PROJECT)

    print('[GEE] ee.Initialize() completado. Probando conexion...')
    test_val = ee.Number(1).add(1).getInfo()
    print(f'[GEE] Test aritmetico OK: 1+1={test_val}')

    print('[GEE] Probando acceso a Sentinel-2 sobre Chameza...')
    zona_test = ee.Geometry.Point([-72.47, 5.09]).buffer(1000)
    n_test = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
              .filterBounds(zona_test)
              .filterDate('2024-01-01', '2024-12-31')
              .size().getInfo())
    print(f'[GEE] Imagenes S2 en 2024 sobre Chameza: {n_test}')

    if n_test == 0:
        print('[GEE] PROBLEMA: 0 imagenes - revisar permisos o cobertura')
    else:
        print('[GEE] Acceso a datos OK')

except Exception as e:
    print(f'[GEE] ERROR CRITICO: {e}')
    import traceback
    traceback.print_exc()
    raise SystemExit(1)

print('=' * 65 + '\n')

# =============================================================================
# Supabase
# =============================================================================
supabase = create_client(os.environ['SUPABASE_URL'], os.environ['SUPABASE_KEY'])

# =============================================================================
# Helpers GEE
# =============================================================================

def mask_s2_scl(image):
    scl = image.select('SCL')
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
    col = coleccion_s2(geom, fecha_ini, fecha_fin)
    n   = col.size().getInfo()
    print(f'      [{fecha_ini} -> {fecha_fin}] {n} imagenes', end='  ')
    if n == 0:
        return None
    ndvi  = col.median().normalizedDifference(['B8', 'B4']).rename('ndvi')
    stats = ndvi.reduceRegion(
        reducer=ee.Reducer.mean(), geometry=geom, scale=10, maxPixels=1e9
    )
    val = stats.getInfo().get('ndvi')
    return round(val, 4) if val is not None else None


def area_afectada_ha(geom, rec_ini, rec_fin, ref_ini, ref_fin):
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


def cargar_geojson(url):
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()


def unidades_predios(gj):
    grupos = {}
    for f in gj.get('features', []):
        pid  = f['properties'].get('id_predio', 'sin_id')
        geom = ee.Geometry(f['geometry'])
        grupos[pid] = grupos[pid].union(geom, maxError=1) if pid in grupos else geom
    return list(grupos.items())


def unidades_capa(gj, prefijo):
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


def procesar_unidad(pid, geom, rec_ini, rec_fin, ref_ini, ref_fin, hoy):
    print(f'\n  [{pid}]')
    if ya_existe(pid, hoy):
        print('    -> ya procesado hoy.')
        return 'duplicado'
    try:
        print('    ref:', end='')
        ndvi_ref = ndvi_medio(geom, ref_ini, ref_fin)
        print(f'  NDVI={ndvi_ref}')

        print('    rec:', end='')
        ndvi_rec = ndvi_medio(geom, rec_ini, rec_fin)
        print(f'  NDVI={ndvi_rec}')

        if ndvi_ref is None or ndvi_rec is None:
            print('    -> sin imagenes suficientes.')
            return 'sin_imagenes'

        cambio = round(ndvi_rec - ndvi_ref, 4)
        print(f'    delta={cambio:+.4f}  umbral={UMBRAL_NDVI}')

        if cambio >= UMBRAL_NDVI:
            print('    -> sin alerta.')
            return 'sin_alerta'

        area_ha  = area_afectada_ha(geom, rec_ini, rec_fin, ref_ini, ref_fin)
        lat, lon = centroide(geom)
        sev      = clasificar_severidad(cambio)
        guardar_alerta(pid, lat, lon, ndvi_ref, ndvi_rec, cambio, area_ha, hoy)
        print(f'    *** ALERTA: {sev} | {area_ha} ha ***')
        return f'alerta_{sev}'

    except Exception as e:
        print(f'    ERROR: {e}')
        import traceback; traceback.print_exc()
        return 'error'


def main():
    hoy     = date.today()
    rec_ini = hoy - timedelta(days=DIAS_RECIENTE)
    rec_fin = hoy
    ref_fin = hoy - timedelta(days=DIAS_REF_FIN)
    ref_ini = hoy - timedelta(days=DIAS_REF_INI)

    sep = '=' * 65
    print(sep)
    print(f'Monitor Deforestacion v3  —  {hoy}')
    print(f'  Reciente  : {rec_ini} -> {rec_fin}')
    print(f'  Referencia: {ref_ini} -> {ref_fin}')
    print(f'  Umbral NDVI: {UMBRAL_NDVI}  |  Max nubes: {MAX_NUBES_COL}% + SCL')
    print(sep)

    todas = []

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
        print(f'  {len(est)} unidades.')

    if PROCESAR_NUCLEOS:
        print('\nCargando Nucleos_boscosos.geojson...')
        nuc = unidades_capa(cargar_geojson(GEOJSON_NUCLEOS), 'nucleo')
        todas += nuc
        print(f'  {len(nuc)} nucleos.')

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


if __name__ == '__main__':
    main()
