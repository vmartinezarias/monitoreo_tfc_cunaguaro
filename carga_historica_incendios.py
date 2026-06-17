"""
carga_historica_incendios.py
────────────────────────────
Descarga TODOS los puntos de calor NASA FIRMS desde el 1 enero 2026
hasta hoy y los guarda en la tabla `alertas` de Supabase.

Corre UNA SOLA VEZ como GitHub Actions workflow_dispatch.
Después, monitor_incendios.py se encarga del mantenimiento regular.

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

# Fecha de inicio del histórico
FECHA_INICIO = datetime.date(2026, 1, 1)

# FIRMS permite máximo 10 días por llamada
DIAS_POR_BLOQUE = 10

# Fuentes FIRMS
FUENTES = [
    "VIIRS_SNPP_NRT",
    "VIIRS_NOAA20_NRT",
    "MODIS_NRT",
]

FIRMS_BASE = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"

# Pausa para no saturar la API
PAUSA_ENTRE_LLAMADAS = 2

# Reintentos por llamada
MAX_REINTENTOS = 3
PAUSA_REINTENTO = 5


# ── Helpers ───────────────────────────────────────────────────────────────────
def ocultar_key_en_url(url):
    """Oculta la FIRMS_KEY en los logs."""
    return url.replace(FIRMS_KEY, "***")


def generar_bloques(fecha_inicio, fecha_fin, dias_bloque):
    """
    Genera bloques para cubrir el rango completo.

    La URL de FIRMS usa:
    /api/area/csv/{key}/{producto}/{bbox}/{dias}/{fecha}

    En este flujo usamos `fecha` como fecha final del bloque y pedimos
    los días anteriores hasta cubrir todo el período.
    """
    if fecha_fin < fecha_inicio:
        return []

    bloques = []
    cursor = fecha_fin

    while cursor >= fecha_inicio:
        dias = min(dias_bloque, (cursor - fecha_inicio).days + 1)
        if dias <= 0:
            break

        bloques.append((cursor.strftime("%Y-%m-%d"), dias))
        cursor -= datetime.timedelta(days=dias)

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


def obtener_existentes():
    """
    Carga claves existentes para evitar duplicados.

    Clave usada:
      fuente + fecha hasta minuto + latitud redondeada + longitud redondeada
    """
    print("  Cargando registros existentes de Supabase para deduplicar…")

    existentes = set()
    offset = 0
    pagina = 10_000

    while True:
        path = (
            "alertas?"
            "select=fuente,fecha_deteccion,latitud,longitud"
            f"&tipo=eq.incendio"
            f"&limit={pagina}"
            f"&offset={offset}"
        )

        status, body = supabase_request("GET", path)

        if status != 200:
            print(f"  ⚠ No se pudo leer Supabase. HTTP {status}: {body[:250]}")
            break

        try:
            registros = json.loads(body)
        except Exception as e:
            print(f"  ⚠ Error leyendo respuesta de Supabase: {e}")
            break

        if not registros:
            break

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

        offset += pagina

        if len(registros) < pagina:
            break

    print(f"  ✓ {len(existentes):,} alertas ya existentes en Supabase")
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
    hoy = datetime.date.today()

    print("=" * 70)
    print("CARGA HISTÓRICA DE INCENDIOS — NASA FIRMS")
    print(f"Rango: {FECHA_INICIO} → {hoy}")
    print(f"Fuentes: {', '.join(FUENTES)}")
    print(f"BBOX: {BBOX}")
    print("=" * 70)

    dias_totales = (hoy - FECHA_INICIO).days + 1
    bloques = generar_bloques(FECHA_INICIO, hoy, DIAS_POR_BLOQUE)

    print(
        f"\nBloques a descargar: {len(bloques)} × {len(FUENTES)} fuentes = "
        f"{len(bloques) * len(FUENTES)} llamadas a FIRMS\n"
    )

    if not bloques:
        print("No hay bloques para descargar. Revisa FECHA_INICIO.")
        sys.exit(1)

    # 1. Existentes
    existentes = obtener_existentes()

    # 2. Descargar e insertar
    total_nuevas = 0
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

            nuevas = []

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
                nuevas.append(alerta)

            print(
                f"      {len(filas)} filas FIRMS → "
                f"{len(nuevas)} nuevas"
            )

            if nuevas:
                ins = insertar_lote(nuevas)
                total_nuevas += ins

            time.sleep(PAUSA_ENTRE_LLAMADAS)

    # 3. Resumen
    print("\n" + "=" * 70)
    print("RESUMEN FINAL")
    print("=" * 70)
    print(f"  Período cubierto       : {FECHA_INICIO} → {hoy} ({dias_totales} días)")
    print(f"  Llamadas FIRMS totales : {llamada_n}")
    print(f"  Llamadas exitosas      : {llamadas_ok}")
    print(f"  Llamadas fallidas      : {llamadas_error}")
    print(f"  Filas recibidas FIRMS  : {total_filas_firms:,}")
    print(f"  Alertas insertadas     : {total_nuevas:,}")
    print("=" * 70)

    if llamadas_ok == 0 and llamadas_error > 0:
        print("❌ Ninguna llamada a FIRMS fue exitosa. Revisa red, FIRMS_KEY o endpoint.")
        sys.exit(1)

    print("✅ Carga histórica completada.")
    print("   Revisa Supabase → public.alertas.")
    print("   Luego usa monitor_incendios.yml para mantener la base actualizada.")


if __name__ == "__main__":
    main()
