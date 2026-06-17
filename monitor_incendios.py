"""
monitor_incendios.py
────────────────────
Consulta NASA FIRMS para la zona de Chámeza/Casanare y guarda las alertas
recientes de incendios en la tabla `alertas` de Supabase.

Este script está pensado para correr regularmente con GitHub Actions,
por ejemplo cada 3 horas.

IMPORTANTE:
- Este monitor NO borra alertas antiguas.
- Sirve para mantener actualizada una base histórica que ya fue cargada
  previamente con carga_historica_incendios.py.

Variables de entorno requeridas:
  SUPABASE_URL  — URL del proyecto Supabase
  SUPABASE_KEY  — preferiblemente service_role key de Supabase
  FIRMS_KEY     — API key de NASA FIRMS
"""

import os
import csv
import io
import json
import datetime
import time
import urllib.request
import urllib.parse
import urllib.error
import socket
import sys


# ── FIX GitHub Actions / NASA FIRMS ────────────────────────────────────────────
# En algunos runners, urllib intenta resolver IPv6 y NASA FIRMS puede responder:
# <urlopen error [Errno 101] Network is unreachable>
# Esto fuerza IPv4 para todas las conexiones.
_original_getaddrinfo = socket.getaddrinfo

def _getaddrinfo_ipv4(host, port, family=0, type=0, proto=0, flags=0):
    return _original_getaddrinfo(
        host,
        port,
        socket.AF_INET,
        type,
        proto,
        flags
    )

socket.getaddrinfo = _getaddrinfo_ipv4


# ── Configuración ──────────────────────────────────────────────────────────────
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
FIRMS_KEY = os.environ.get("FIRMS_KEY", "")

if not SUPABASE_URL:
    raise RuntimeError("Falta la variable de entorno SUPABASE_URL")

if not SUPABASE_KEY:
    raise RuntimeError("Falta la variable de entorno SUPABASE_KEY")

if not FIRMS_KEY:
    raise RuntimeError("Falta la variable de entorno FIRMS_KEY")


# Bounding box Chámeza + buffer generoso:
# lon_min, lat_min, lon_max, lat_max
BBOX = "-73.20,4.60,-71.80,5.60"

# Ventana reciente a revisar en cada corrida.
# Aunque el workflow corra cada 3 horas, usamos 30 días para capturar retrasos
# y evitar perder datos si alguna corrida anterior falló.
DIAS_TOTAL = 30

# FIRMS permite máximo 10 días por llamada
DIAS_POR_BLOQUE = 10

# Fuentes FIRMS
FUENTES = [
    "VIIRS_SNPP_NRT",
    "VIIRS_NOAA20_NRT",
    "MODIS_NRT",
]

FIRMS_BASE = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"

# Pausas y reintentos
PAUSA_ENTRE_LLAMADAS = 2
MAX_REINTENTOS = 3
PAUSA_REINTENTO = 5


# ── Helpers ───────────────────────────────────────────────────────────────────
def ocultar_key_en_url(url):
    """Oculta la FIRMS_KEY en los logs."""
    return url.replace(FIRMS_KEY, "***")


def generar_bloques(fecha_fin, dias_total, dias_bloque):
    """
    Genera bloques correctos para FIRMS.

    OJO:
    En FIRMS, el parámetro DATE es la fecha INICIAL del bloque.
    La API devuelve:
      DATE → DATE + DAY_RANGE - 1

    Para últimos 30 días, si hoy es 2026-06-17:
      2026-05-19 · 10 días
      2026-05-29 · 10 días
      2026-06-08 · 10 días
    """
    fecha_inicio = fecha_fin - datetime.timedelta(days=dias_total - 1)

    bloques = []
    cursor = fecha_inicio

    while cursor <= fecha_fin:
        dias_restantes = (fecha_fin - cursor).days + 1
        dias = min(dias_bloque, dias_restantes)

        bloques.append((cursor.strftime("%Y-%m-%d"), dias))
        cursor += datetime.timedelta(days=dias)

    return bloques


def fetch_firms(producto, bbox, dias, fecha_str):
    """
    Descarga CSV de FIRMS para un bloque.

    Retorna:
      filas, ok

    ok = True  → la petición llegó a FIRMS, aunque no haya datos.
    ok = False → hubo error de red o HTTP.
    """
    url = f"{FIRMS_BASE}/{FIRMS_KEY}/{producto}/{bbox}/{dias}/{fecha_str}"
    print(f"\n      GET {ocultar_key_en_url(url)}")

    headers = {
        "User-Agent": "monitor-incendios-cunaguaro/1.0"
    }

    for intento in range(1, MAX_REINTENTOS + 1):
        try:
            req = urllib.request.Request(url, headers=headers, method="GET")
            with urllib.request.urlopen(req, timeout=90) as resp:
                raw = resp.read().decode("utf-8")

            if not raw.strip():
                return [], True

            if raw.startswith("Error") or "<!DOCTYPE" in raw:
                print(f"      ⚠ Respuesta no CSV de FIRMS: {raw[:160]}")
                return [], True

            reader = csv.DictReader(io.StringIO(raw))
            filas = list(reader)
            return filas, True

        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="ignore")
            print(
                f"      ⚠ HTTPError intento {intento}/{MAX_REINTENTOS}: "
                f"{e.code} · {body[:180]}"
            )

        except Exception as e:
            print(
                f"      ⚠ Error intento {intento}/{MAX_REINTENTOS}: {e}"
            )

        if intento < MAX_REINTENTOS:
            time.sleep(PAUSA_REINTENTO)

    return [], False


def normalizar_confianza(row, producto):
    """Normaliza confianza a high | nominal | low."""
    if "VIIRS" in producto:
        conf = str(row.get("confidence", "n")).strip().lower()
        mapa = {
            "h": "high",
            "n": "nominal",
            "l": "low",
            "high": "high",
            "nominal": "nominal",
            "low": "low",
        }
        return mapa.get(conf, "nominal")

    # MODIS suele usar confianza numérica 0–100
    try:
        v = int(float(row.get("confidence", 50)))
        if v >= 80:
            return "high"
        if v >= 30:
            return "nominal"
        return "low"
    except Exception:
        return "nominal"


def row_a_alerta(row, fuente_nombre):
    """Convierte una fila FIRMS al formato de la tabla public.alertas."""
    try:
        lat = float(row.get("latitude") or row.get("lat"))
        lon = float(row.get("longitude") or row.get("lon"))
    except Exception:
        return None

    acq_date = row.get("acq_date", "")
    acq_time = str(row.get("acq_time", "0000")).zfill(4)

    try:
        hora = f"{acq_time[:2]}:{acq_time[2:]}:00"
        fecha_str = f"{acq_date}T{hora}+00:00"
        datetime.datetime.fromisoformat(fecha_str)
    except Exception:
        if acq_date:
            fecha_str = f"{acq_date}T00:00:00+00:00"
        else:
            return None

    try:
        frp = float(row.get("frp", 0) or 0)
    except Exception:
        frp = None

    try:
        brightness = float(
            row.get("brightness")
            or row.get("bright_ti4")
            or row.get("bright_t31")
            or 0
        )
    except Exception:
        brightness = None

    return {
        "tipo": "incendio",
        "fuente": fuente_nombre,
        "fecha_deteccion": fecha_str,
        "latitud": lat,
        "longitud": lon,
        "firms_brightness": brightness,
        "firms_frp": frp,
        "firms_confidence": normalizar_confianza(row, fuente_nombre),
        "firms_satellite": fuente_nombre,
        "estado": "nueva",
    }


def supabase_request(method, path, body=None):
    """Hace una petición a Supabase REST."""
    url = f"{SUPABASE_URL}/rest/v1/{path}"

    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }

    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        headers=headers,
        method=method
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status, resp.read().decode("utf-8")

    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="ignore")

    except Exception as e:
        return 0, str(e)


def obtener_existentes(dias=32):
    """
    Carga claves existentes recientes para evitar duplicados.

    Como este monitor solo consulta los últimos 30 días, no necesita traer
    toda la base histórica. Solo carga una ventana un poco mayor.

    Clave:
      fuente + fecha hasta minuto + latitud redondeada + longitud redondeada
    """
    print("  Cargando registros existentes recientes de Supabase…")

    fecha_min = (
        datetime.datetime.utcnow() - datetime.timedelta(days=dias)
    ).isoformat() + "+00:00"

    fecha_min_q = urllib.parse.quote(fecha_min)

    path = (
        "alertas?"
        "select=fuente,fecha_deteccion,latitud,longitud"
        "&tipo=eq.incendio"
        f"&fecha_deteccion=gte.{fecha_min_q}"
        "&limit=50000"
    )

    status, body = supabase_request("GET", path)

    if status != 200:
        print(f"  ⚠ No se pudo leer Supabase. HTTP {status}: {body[:250]}")
        return set()

    try:
        registros = json.loads(body)
    except Exception as e:
        print(f"  ⚠ Error leyendo respuesta de Supabase: {e}")
        return set()

    existentes = set()

    for r in registros:
        try:
            clave = (
                r.get("fuente", ""),
                str(r.get("fecha_deteccion", ""))[:16],
                round(float(r.get("latitud", 0)), 4),
                round(float(r.get("longitud", 0)), 4),
            )
            existentes.add(clave)
        except Exception:
            continue

    print(f"  ✓ {len(existentes):,} alertas existentes en ventana de {dias} días")
    return existentes


def insertar_lote(alertas):
    """Inserta alertas en Supabase en lotes."""
    if not alertas:
        return 0

    insertadas = 0
    lote_size = 500

    for i in range(0, len(alertas), lote_size):
        lote = alertas[i:i + lote_size]
        status, body = supabase_request("POST", "alertas", lote)

        if status in (200, 201, 204):
            insertadas += len(lote)
            print(
                f"      ✓ Lote {i // lote_size + 1}: "
                f"{len(lote)} insertadas"
            )
        else:
            print(
                f"      ✗ Error insertando lote {i // lote_size + 1}. "
                f"HTTP {status}: {body[:300]}"
            )

    return insertadas


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    hoy = datetime.datetime.utcnow().date()

    print("=" * 70)
    print(f"MONITOR DE INCENDIOS — NASA FIRMS")
    print(f"Fecha UTC: {datetime.datetime.utcnow().isoformat(timespec='seconds')}")
    print(f"Ventana consultada: últimos {DIAS_TOTAL} días")
    print(f"Fuentes: {', '.join(FUENTES)}")
    print(f"BBOX: {BBOX}")
    print("=" * 70)

    bloques = generar_bloques(hoy, DIAS_TOTAL, DIAS_POR_BLOQUE)

    print(
        f"\nBloques a descargar: {len(bloques)} × {len(FUENTES)} fuentes = "
        f"{len(bloques) * len(FUENTES)} llamadas a FIRMS\n"
    )

    if not bloques:
        print("No hay bloques para descargar. Revisa DIAS_TOTAL.")
        sys.exit(1)

    # 1. Existentes recientes para deduplicar
    print("[1] Consultando alertas existentes en Supabase…")
    existentes = obtener_existentes(dias=DIAS_TOTAL + 2)

    # 2. Descargar datos recientes de FIRMS
    print(f"\n[2] Descargando datos de NASA FIRMS…")

    todas_nuevas = []
    total_filas_firms = 0
    total_llamadas = len(bloques) * len(FUENTES)
    llamadas_ok = 0
    llamadas_error = 0
    llamada_n = 0

    for fuente in FUENTES:
        print("\n" + "─" * 70)
        print(f"FUENTE: {fuente}")
        print("─" * 70)

        for fecha_bloque, dias in bloques:
            llamada_n += 1

            print(
                f"  [{llamada_n}/{total_llamadas}] "
                f"{fuente} · hasta {fecha_bloque} · {dias} días…",
                end=""
            )

            filas, ok = fetch_firms(fuente, BBOX, dias, fecha_bloque)

            if ok:
                llamadas_ok += 1
            else:
                llamadas_error += 1
                print("      ✗ Falló la llamada a FIRMS")
                time.sleep(PAUSA_ENTRE_LLAMADAS)
                continue

            if not filas:
                print("      sin datos")
                time.sleep(PAUSA_ENTRE_LLAMADAS)
                continue

            total_filas_firms += len(filas)

            nuevas_bloque = []

            for fila in filas:
                alerta = row_a_alerta(fila, fuente)

                if alerta is None:
                    continue

                clave = (
                    alerta["fuente"],
                    alerta["fecha_deteccion"][:16],
                    round(alerta["latitud"], 4),
                    round(alerta["longitud"], 4),
                )

                if clave in existentes:
                    continue

                existentes.add(clave)
                nuevas_bloque.append(alerta)

            print(
                f"      {len(filas)} filas FIRMS → "
                f"{len(nuevas_bloque)} nuevas"
            )

            todas_nuevas.extend(nuevas_bloque)

            time.sleep(PAUSA_ENTRE_LLAMADAS)

    # 3. Insertar en Supabase
    print(f"\n[3] Total nuevas alertas a insertar: {len(todas_nuevas):,}")

    if todas_nuevas:
        print("\n[4] Insertando en Supabase…")
        insertadas = insertar_lote(todas_nuevas)
        print(f"\n  ✅ {insertadas:,} alertas insertadas correctamente")
    else:
        print("\n  ℹ No hay alertas nuevas para insertar")

    # 4. No borrar histórico
    print("\n[5] Histórico activo — no se eliminan alertas antiguas")

    # 5. Resumen y validación
    print("\n" + "=" * 70)
    print("RESUMEN FINAL")
    print("=" * 70)
    print(f"  Llamadas FIRMS totales : {llamada_n}")
    print(f"  Llamadas exitosas      : {llamadas_ok}")
    print(f"  Llamadas fallidas      : {llamadas_error}")
    print(f"  Filas recibidas FIRMS  : {total_filas_firms:,}")
    print(f"  Alertas nuevas         : {len(todas_nuevas):,}")
    print("=" * 70)

    if llamadas_ok == 0 and llamadas_error > 0:
        print("❌ Ninguna llamada a FIRMS fue exitosa. Revisa red, FIRMS_KEY o endpoint.")
        sys.exit(1)

    print("✅ Monitor de incendios completado")
    print("   Revisa Supabase → public.alertas.")


if __name__ == "__main__":
    main()
