"""
carga_historica_incendios.py
────────────────────────────
Descarga TODOS los puntos de calor NASA FIRMS desde el 1 enero 2026
hasta hoy y los guarda en la tabla `alertas` de Supabase.

Corre UNA SOLA VEZ como GitHub Actions workflow_dispatch.
Después, monitor_incendios.py se encarga del mantenimiento diario.

Variables de entorno requeridas:
  SUPABASE_URL  — https://qryktfqnicnwiuwbijvd.supabase.co
  SUPABASE_KEY  — anon/service key de Supabase
  FIRMS_KEY     — cef2517155cca7f902d6bad2eaec8210
"""

import os
import sys
import csv
import io
import json
import datetime
import time
import urllib.request
import urllib.parse

# ── Configuración ──────────────────────────────────────────────────────────────
SUPABASE_URL = os.environ["SUPABASE_URL"].rstrip("/")
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
FIRMS_KEY    = os.environ["FIRMS_KEY"]

# Bounding box Chámeza + buffer generoso (lon_min, lat_min, lon_max, lat_max)
BBOX = "-73.20,4.60,-71.80,5.60"

# Fecha de inicio del histórico
FECHA_INICIO = datetime.date(2026, 1, 1)

# Días por llamada (máximo que acepta FIRMS)
DIAS_POR_BLOQUE = 10

# Fuentes a descargar
FUENTES = [
    "VIIRS_SNPP_NRT",
    "VIIRS_NOAA20_NRT",
    "MODIS_NRT",
]

FIRMS_BASE = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"

# Pausa entre llamadas a FIRMS para no saturar la API (segundos)
PAUSA_ENTRE_LLAMADAS = 2

# ── Helpers ────────────────────────────────────────────────────────────────────
def generar_bloques(fecha_inicio, fecha_fin, dias_bloque):
    """
    Genera lista de (fecha_fin_bloque, dias) para cubrir el rango completo.
    FIRMS devuelve datos de los N días ANTERIORES a la fecha dada.
    """
    bloques = []
    cursor = fecha_fin
    while cursor > fecha_inicio:
        dias = min(dias_bloque, (cursor - fecha_inicio).days + 1)
        if dias <= 0:
            break
        bloques.append((cursor.strftime("%Y-%m-%d"), dias))
        cursor -= datetime.timedelta(days=dias)
    return bloques


def fetch_firms(producto, bbox, dias, fecha_str):
    """Descarga CSV de FIRMS para un bloque y retorna lista de dicts."""
    url = f"{FIRMS_BASE}/{FIRMS_KEY}/{producto}/{bbox}/{dias}/{fecha_str}"
    try:
        with urllib.request.urlopen(url, timeout=90) as resp:
            raw = resp.read().decode("utf-8")
    except Exception as e:
        print(f"      ⚠ Error HTTP: {e}")
        return []

    if not raw.strip() or raw.startswith("Error") or "<!DOCTYPE" in raw:
        # Sin datos para este bloque es normal (puede que no haya fuegos)
        return []

    try:
        reader = csv.DictReader(io.StringIO(raw))
        return list(reader)
    except Exception as e:
        print(f"      ⚠ Error parseando CSV: {e}")
        return []


def normalizar_confianza(row, producto):
    if "VIIRS" in producto:
        conf = str(row.get("confidence", "n")).strip().lower()
        return {"h": "high", "n": "nominal", "l": "low",
                "high": "high", "nominal": "nominal", "low": "low"}.get(conf, "nominal")
    else:
        try:
            v = int(row.get("confidence", 50))
            if v >= 80: return "high"
            if v >= 30: return "nominal"
            return "low"
        except Exception:
            return "nominal"


def row_a_alerta(row, fuente_nombre):
    try:
        lat = float(row.get("latitude") or row.get("lat", 0))
        lon = float(row.get("longitude") or row.get("lon", 0))
    except Exception:
        return None

    acq_date = row.get("acq_date", "")
    acq_time = str(row.get("acq_time", "0000")).zfill(4)
    try:
        hora = f"{acq_time[:2]}:{acq_time[2:]}:00"
        fecha_str = f"{acq_date}T{hora}+00:00"
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
        brightness = float(
            row.get("brightness") or
            row.get("bright_ti4") or
            row.get("bright_t31") or 0
        )
    except Exception:
        brightness = None

    return {
        "tipo":             "incendio",
        "fuente":           fuente_nombre,
        "fecha_deteccion":  fecha_str,
        "latitud":          lat,
        "longitud":         lon,
        "firms_brightness": brightness,
        "firms_frp":        frp,
        "firms_confidence": normalizar_confianza(row, fuente_nombre),
        "firms_satellite":  fuente_nombre,
        "estado":           "nueva",
    }


def supabase_request(method, path, body=None):
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
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()
    except Exception as e:
        return 0, str(e)


def obtener_existentes():
    """
    Carga las claves (fuente + fecha_min + lat + lon) de todo lo que
    ya está en Supabase para evitar duplicados.
    Se pagina de 10 000 en 10 000.
    """
    print("  Cargando registros existentes de Supabase para deduplicar…")
    existentes = set()
    offset = 0
    pagina = 10_000
    while True:
        path = (
            f"alertas?select=fuente,fecha_deteccion,latitud,longitud"
            f"&tipo=eq.incendio&limit={pagina}&offset={offset}"
        )
        status, body = supabase_request("GET", path)
        if status != 200:
            print(f"  ⚠ No se pudo paginar existentes (HTTP {status})")
            break
        registros = json.loads(body)
        if not registros:
            break
        for r in registros:
            clave = (
                r.get("fuente", ""),
                r.get("fecha_deteccion", "")[:16],
                round(float(r.get("latitud",  0)), 4),
                round(float(r.get("longitud", 0)), 4),
            )
            existentes.add(clave)
        offset += pagina
        if len(registros) < pagina:
            break
    print(f"  ✓ {len(existentes):,} alertas ya existentes en Supabase")
    return existentes


def insertar_lote(alertas):
    """Inserta en lotes de 500 y retorna total insertado."""
    if not alertas:
        return 0
    insertadas = 0
    lote_size = 500
    for i in range(0, len(alertas), lote_size):
        lote = alertas[i:i + lote_size]
        status, body = supabase_request("POST", "alertas", lote)
        if status in (200, 201):
            insertadas += len(lote)
        else:
            print(f"      ✗ Error lote (HTTP {status}): {body[:250]}")
    return insertadas


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    hoy = datetime.date.today()

    print("=" * 65)
    print("CARGA HISTÓRICA DE INCENDIOS — NASA FIRMS")
    print(f"Rango: {FECHA_INICIO}  →  {hoy}")
    print(f"Fuentes: {', '.join(FUENTES)}")
    print(f"BBOX: {BBOX}")
    print("=" * 65)

    dias_totales = (hoy - FECHA_INICIO).days + 1
    bloques = generar_bloques(FECHA_INICIO, hoy, DIAS_POR_BLOQUE)
    print(f"\nBloques a descargar: {len(bloques)} × {len(FUENTES)} fuentes = "
          f"{len(bloques) * len(FUENTES)} llamadas a FIRMS\n")

    # ── 1. Cargar existentes ──
    existentes = obtener_existentes()

    # ── 2. Descargar por bloque y fuente ──
    total_nuevas = 0
    total_llamadas = len(bloques) * len(FUENTES)
    llamada_n = 0

    for fuente in FUENTES:
        print(f"\n{'─'*65}")
        print(f"FUENTE: {fuente}")
        print(f"{'─'*65}")

        for fecha_bloque, dias in bloques:
            llamada_n += 1
            print(f"  [{llamada_n}/{total_llamadas}] "
                  f"{fuente} · hasta {fecha_bloque} · {dias} días…", end=" ")

            filas = fetch_firms(fuente, BBOX, dias, fecha_bloque)

            if not filas:
                print("sin datos")
                time.sleep(PAUSA_ENTRE_LLAMADAS)
                continue

            nuevas = []
            for fila in filas:
                alerta = row_a_alerta(fila, fuente)
                if alerta is None:
                    continue
                clave = (
                    alerta["fuente"],
                    alerta["fecha_deteccion"][:16],
                    round(alerta["latitud"],  4),
                    round(alerta["longitud"], 4),
                )
                if clave in existentes:
                    continue
                existentes.add(clave)
                nuevas.append(alerta)

            print(f"{len(filas)} filas → {len(nuevas)} nuevas", end="")

            if nuevas:
                ins = insertar_lote(nuevas)
                total_nuevas += ins
                print(f" → ✓ {ins} insertadas")
            else:
                print(" (ya existían)")

            time.sleep(PAUSA_ENTRE_LLAMADAS)

    # ── 3. Resumen final ──
    print("\n" + "=" * 65)
    print("RESUMEN FINAL")
    print("=" * 65)
    print(f"  Período cubierto : {FECHA_INICIO} → {hoy} ({dias_totales} días)")
    print(f"  Llamadas a FIRMS : {llamada_n}")
    print(f"  Alertas insertadas: {total_nuevas:,}")
    print("=" * 65)
    print("✅ Carga histórica completada.")
    print("   A partir de ahora usa monitor_incendios.yml (cada 3 h)")
    print("   para mantener la base actualizada.")


if __name__ == "__main__":
    main()
