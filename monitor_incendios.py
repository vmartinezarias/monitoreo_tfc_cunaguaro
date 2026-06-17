"""
monitor_incendios.py
Consulta NASA FIRMS (VIIRS + MODIS) para la zona de Chámeza/Casanare
y guarda las alertas de los últimos 30 días en Supabase.

Corre vía GitHub Actions cada 3 horas.
Variables de entorno requeridas:
  SUPABASE_URL  — https://qryktfqnicnwiuwbijvd.supabase.co
  SUPABASE_KEY  — anon/service key de Supabase
  FIRMS_KEY     — cef2517155cca7f902d6bad2eaec8210
"""

import os
import sys
import csv
import io
import math
import uuid
import json
import datetime
import urllib.request
import urllib.parse

# ── Configuración ──────────────────────────────────────────────────────────────
SUPABASE_URL = os.environ["SUPABASE_URL"].rstrip("/")
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
FIRMS_KEY    = os.environ["FIRMS_KEY"]

# Bounding box Chámeza + buffer generoso (cubre todo el área de estudio)
# xmin, ymin, xmax, ymax  (lon_min, lat_min, lon_max, lat_max)
BBOX = "-73.20,4.60,-71.80,5.60"

# Días hacia atrás que queremos cubrir (máximo permitido por FIRMS: 10 por llamada)
# Hacemos 3 llamadas de 10 días para cubrir 30 días en total
DIAS_TOTAL = 30
DIAS_POR_LLAMADA = 10

# Fuentes FIRMS a consultar (ambas dan resolución diferente)
FUENTES = [
    {"producto": "VIIRS_SNPP_NRT",   "nombre": "VIIRS_SNPP_NRT"},
    {"producto": "VIIRS_NOAA20_NRT", "nombre": "VIIRS_NOAA20_NRT"},
    {"producto": "MODIS_NRT",        "nombre": "MODIS_NRT"},
]

FIRMS_BASE = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"

# ── Helpers ────────────────────────────────────────────────────────────────────
def fetch_firms(producto, bbox, days):
    """Descarga CSV de FIRMS y retorna lista de dicts."""
    url = f"{FIRMS_BASE}/{FIRMS_KEY}/{producto}/{bbox}/{days}"
    print(f"  GET {url}")
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            raw = resp.read().decode("utf-8")
    except Exception as e:
        print(f"  ⚠ Error descargando {producto}: {e}")
        return []

    if not raw.strip() or raw.startswith("Error") or "<!DOCTYPE" in raw:
        print(f"  ⚠ Respuesta inválida para {producto}: {raw[:120]}")
        return []

    reader = csv.DictReader(io.StringIO(raw))
    rows = list(reader)
    print(f"  ✓ {len(rows)} registros de {producto}")
    return rows


def normalizar_confianza(row, producto):
    """Normaliza el campo de confianza a: high | nominal | low"""
    if "VIIRS" in producto:
        # VIIRS usa: h | n | l
        conf = str(row.get("confidence", "n")).strip().lower()
        mapa = {"h": "high", "n": "nominal", "l": "low",
                "high": "high", "nominal": "nominal", "low": "low"}
        return mapa.get(conf, "nominal")
    else:
        # MODIS usa valor numérico 0-100
        try:
            v = int(row.get("confidence", 50))
            if v >= 80: return "high"
            if v >= 30: return "nominal"
            return "low"
        except Exception:
            return "nominal"


def row_a_alerta(row, fuente_nombre):
    """Convierte una fila de FIRMS al formato de la tabla alertas."""
    try:
        lat = float(row.get("latitude") or row.get("lat", 0))
        lon = float(row.get("longitude") or row.get("lon", 0))
    except Exception:
        return None

    # Fecha + hora de adquisición
    acq_date = row.get("acq_date", "")
    acq_time = str(row.get("acq_time", "0000")).zfill(4)
    try:
        hora = f"{acq_time[:2]}:{acq_time[2:]}:00"
        fecha_str = f"{acq_date}T{hora}+00:00"
        # Validar parseando
        datetime.datetime.fromisoformat(fecha_str)
    except Exception:
        fecha_str = f"{acq_date}T00:00:00+00:00" if acq_date else None

    if not fecha_str:
        return None

    try:
        frp = float(row.get("frp", 0) or 0)
    except Exception:
        frp = None

    try:
        brightness = float(row.get("brightness") or row.get("bright_ti4") or row.get("bright_t31") or 0)
    except Exception:
        brightness = None

    confianza = normalizar_confianza(row, fuente_nombre)

    # Punto WKT para columna geom
    geom_wkt = f"POINT({lon} {lat})"

    return {
        "tipo":            "incendio",
        "fuente":          fuente_nombre,
        "fecha_deteccion": fecha_str,
        "latitud":         lat,
        "longitud":        lon,
        "firms_brightness": brightness,
        "firms_frp":        frp,
        "firms_confidence": confianza,
        "firms_satellite":  fuente_nombre,
        "estado":           "nueva",
        # geom se inserta con ST_GeomFromText en Supabase vía RPC,
        # pero como usamos la REST API con insert directo necesitamos WKT
        # La columna geom tiene default null, la rellenamos separado si hace falta
    }


def supabase_request(method, path, body=None):
    """Hace una petición HTTP a la API REST de Supabase."""
    url = f"{SUPABASE_URL}/rest/v1/{path}"
    headers = {
        "apikey":        SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type":  "application/json",
        "Prefer":        "return=minimal",
    }
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()
    except Exception as e:
        return 0, str(e)


def obtener_existentes(dias=30):
    """Retorna set de (fuente, fecha_deteccion, lat_redondeada, lon_redondeada)
    para evitar duplicados."""
    fecha_min = (datetime.datetime.utcnow() - datetime.timedelta(days=dias)).isoformat() + "+00:00"
    path = f"alertas?select=fuente,fecha_deteccion,latitud,longitud&fecha_deteccion=gte.{urllib.parse.quote(fecha_min)}&limit=50000"
    status, body = supabase_request("GET", path)
    if status != 200:
        print(f"  ⚠ No se pudo obtener existentes (HTTP {status}): {body[:200]}")
        return set()
    try:
        registros = json.loads(body)
        existentes = set()
        for r in registros:
            clave = (
                r.get("fuente", ""),
                r.get("fecha_deteccion", "")[:16],   # minuto exacto
                round(float(r.get("latitud",  0)), 4),
                round(float(r.get("longitud", 0)), 4),
            )
            existentes.add(clave)
        print(f"  ✓ {len(existentes)} alertas ya en Supabase (últimos {dias} días)")
        return existentes
    except Exception as e:
        print(f"  ⚠ Error parseando existentes: {e}")
        return set()


def insertar_lote(alertas):
    """Inserta lista de alertas en Supabase en lotes de 500."""
    if not alertas:
        return 0
    insertadas = 0
    lote_size = 500
    for i in range(0, len(alertas), lote_size):
        lote = alertas[i:i+lote_size]
        status, body = supabase_request("POST", "alertas", lote)
        if status in (200, 201):
            insertadas += len(lote)
            print(f"  ✓ Lote {i//lote_size + 1}: {len(lote)} alertas insertadas")
        else:
            print(f"  ✗ Error en lote {i//lote_size + 1} (HTTP {status}): {body[:300]}")
    return insertadas


def limpiar_antiguas():
    """Elimina alertas de incendio con más de 32 días para no acumular indefinidamente."""
    fecha_limite = (datetime.datetime.utcnow() - datetime.timedelta(days=32)).isoformat() + "+00:00"
    path = f"alertas?tipo=eq.incendio&fecha_deteccion=lt.{urllib.parse.quote(fecha_limite)}"
    status, body = supabase_request("DELETE", path)
    if status in (200, 204):
        print(f"  ✓ Alertas antiguas eliminadas (anteriores a {fecha_limite[:10]})")
    else:
        print(f"  ⚠ No se pudo limpiar antiguas (HTTP {status}): {body[:200]}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print(f"Monitor de Incendios — {datetime.datetime.utcnow().isoformat()[:19]} UTC")
    print("=" * 60)

    # 1. Obtener registros ya existentes para deduplicar
    print("\n[1] Consultando alertas existentes en Supabase…")
    existentes = obtener_existentes(dias=DIAS_TOTAL + 2)

    # 2. Descargar datos de FIRMS en bloques de DIAS_POR_LLAMADA
    print(f"\n[2] Descargando datos de NASA FIRMS (últimos {DIAS_TOTAL} días)…")
    todas_nuevas = []

    # Bloques de días: [1-10], [11-20], [21-30]
    # FIRMS acepta "days" como entero: trae desde hoy hacia atrás N días
    # Para cubrir 30 días hacemos 3 llamadas con 10 días c/u desplazadas por fecha
    # La API de área acepta: /api/area/csv/{key}/{product}/{bbox}/{days}
    # donde days puede ser 1-10. Para ir hacia atrás necesitamos el parámetro
    # de fecha de inicio — usamos la versión con fecha:
    # /api/area/csv/{key}/{product}/{bbox}/{days}/{date}

    hoy = datetime.date.today()

    for fuente in FUENTES:
        print(f"\n  Fuente: {fuente['producto']}")
        for bloque in range(DIAS_TOTAL // DIAS_POR_LLAMADA):
            # Fecha de fin del bloque (más reciente del bloque)
            dias_offset = bloque * DIAS_POR_LLAMADA
            fecha_fin_bloque = hoy - datetime.timedelta(days=dias_offset)
            fecha_str = fecha_fin_bloque.strftime("%Y-%m-%d")

            url_con_fecha = (
                f"{FIRMS_BASE}/{FIRMS_KEY}/{fuente['producto']}"
                f"/{BBOX}/{DIAS_POR_LLAMADA}/{fecha_str}"
            )
            print(f"    GET {url_con_fecha}")
            try:
                with urllib.request.urlopen(url_con_fecha, timeout=60) as resp:
                    raw = resp.read().decode("utf-8")
            except Exception as e:
                print(f"    ⚠ Error: {e}")
                continue

            if not raw.strip() or raw.startswith("Error") or "<!DOCTYPE" in raw:
                print(f"    ⚠ Sin datos o error: {raw[:80]}")
                continue

            reader = csv.DictReader(io.StringIO(raw))
            filas = list(reader)
            print(f"    ✓ {len(filas)} registros")

            for fila in filas:
                alerta = row_a_alerta(fila, fuente["nombre"])
                if alerta is None:
                    continue

                # Clave de deduplicación
                clave = (
                    alerta["fuente"],
                    alerta["fecha_deteccion"][:16],
                    round(alerta["latitud"],  4),
                    round(alerta["longitud"], 4),
                )
                if clave in existentes:
                    continue

                existentes.add(clave)   # marcar para no duplicar en esta misma corrida
                todas_nuevas.append(alerta)

    print(f"\n[3] Total nuevas alertas a insertar: {len(todas_nuevas)}")

    # 3. Insertar en Supabase
    if todas_nuevas:
        print("\n[4] Insertando en Supabase…")
        insertadas = insertar_lote(todas_nuevas)
        print(f"\n  ✅ {insertadas} alertas insertadas correctamente")
    else:
        print("\n  ℹ No hay alertas nuevas para insertar")

    # 4. Limpiar alertas muy antiguas (>32 días)
    print("\n[5] Limpiando alertas antiguas…")
    limpiar_antiguas()

    print("\n" + "=" * 60)
    print("✅ Monitor de incendios completado")
    print("=" * 60)


if __name__ == "__main__":
    main()
