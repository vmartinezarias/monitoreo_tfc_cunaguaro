"""
carga_historica_incendios.py
────────────────────────────
Descarga TODOS los puntos de calor NASA FIRMS desde el 1 enero 2026
hasta hoy y los guarda en la tabla `alertas` de Supabase.

Corre UNA SOLA VEZ como GitHub Actions workflow_dispatch.
Después, monitor_incendios.py se encarga del mantenimiento regular.

NOTAS TÉCNICAS FIRMS:
  - Con fecha específica: máximo 5 días por llamada (no 10).
  - Se prueban dos endpoints: firms2 (primario) y firms (fallback).
  - Timeout extendido a 120s para runners con latencia alta.

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
import ssl
import sys


# ── Forzar IPv4 (fix GitHub Actions / NASA FIRMS) ─────────────────────────────
_orig_getaddrinfo = socket.getaddrinfo
def _ipv4_only(host, port, family=0, type=0, proto=0, flags=0):
    return _orig_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)
socket.getaddrinfo = _ipv4_only


# ── Configuración ──────────────────────────────────────────────────────────────
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
FIRMS_KEY    = os.environ.get("FIRMS_KEY", "")

if not SUPABASE_URL: raise RuntimeError("Falta SUPABASE_URL")
if not SUPABASE_KEY: raise RuntimeError("Falta SUPABASE_KEY")
if not FIRMS_KEY:    raise RuntimeError("Falta FIRMS_KEY")

BBOX         = "-73.20,4.60,-71.80,5.60"
FECHA_INICIO = datetime.date(2026, 1, 1)

# FIX: FIRMS acepta máximo 5 días cuando se usa fecha específica (no 10)
DIAS_POR_BLOQUE = 5

FUENTES = [
    "VIIRS_SNPP_NRT",
    "VIIRS_NOAA20_NRT",
    "MODIS_NRT",
]

# Endpoints en orden de preferencia
FIRMS_ENDPOINTS = [
    "https://firms2.modaps.eosdis.nasa.gov/api/area/csv",
    "https://firms.modaps.eosdis.nasa.gov/api/area/csv",
]

TIMEOUT              = 120
PAUSA_ENTRE_LLAMADAS = 3
MAX_REINTENTOS       = 2
PAUSA_REINTENTO      = 8


# ── Helpers ────────────────────────────────────────────────────────────────────
def ocultar_key(url):
    return url.replace(FIRMS_KEY, "***")


def generar_bloques(fecha_inicio, fecha_fin, dias_bloque):
    """
    Genera lista de (fecha_inicio_bloque, n_dias) hacia adelante
    desde fecha_inicio hasta fecha_fin.
    FIRMS con fecha: DATE es el INICIO del bloque.
    """
    bloques = []
    cursor = fecha_inicio
    while cursor <= fecha_fin:
        dias = min(dias_bloque, (fecha_fin - cursor).days + 1)
        bloques.append((cursor.strftime("%Y-%m-%d"), dias))
        cursor += datetime.timedelta(days=dias)
    return bloques


def fetch_firms(producto, bbox, dias, fecha_str):
    """
    Descarga CSV de FIRMS probando ambos endpoints.
    Retorna (filas, ok).
    """
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    for base_url in FIRMS_ENDPOINTS:
        url = f"{base_url}/{FIRMS_KEY}/{producto}/{bbox}/{dias}/{fecha_str}"
        print(f"      → {ocultar_key(url)}")

        for intento in range(1, MAX_REINTENTOS + 1):
            try:
                req = urllib.request.Request(
                    url,
                    headers={
                        "User-Agent": "carga-historica-cunaguaro/2.0",
                        "Accept":     "text/csv,*/*",
                    },
                    method="GET"
                )
                with urllib.request.urlopen(req, timeout=TIMEOUT, context=ctx) as resp:
                    raw = resp.read().decode("utf-8", errors="replace")

                if not raw.strip():
                    return [], True

                if raw.lstrip().startswith("Error") or "<!DOCTYPE" in raw:
                    print(f"      ⚠ Respuesta no-CSV ({base_url.split('/')[2]}): {raw[:160]}")
                    break  # probar siguiente endpoint

                filas = list(csv.DictReader(io.StringIO(raw)))
                print(f"      ✓ {len(filas)} registros ({base_url.split('/')[2]})")
                return filas, True

            except urllib.error.HTTPError as e:
                body = e.read().decode(errors="ignore")
                print(f"      ⚠ HTTP {e.code} intento {intento}/{MAX_REINTENTOS} "
                      f"({base_url.split('/')[2]}): {body[:150]}")

            except Exception as e:
                print(f"      ⚠ Error intento {intento}/{MAX_REINTENTOS} "
                      f"({base_url.split('/')[2]}): {type(e).__name__}: {e}")

            if intento < MAX_REINTENTOS:
                time.sleep(PAUSA_REINTENTO)

    print("      ✗ Ambos endpoints fallaron para este bloque")
    return [], False


def normalizar_confianza(row, producto):
    if "VIIRS" in producto:
        conf = str(row.get("confidence", "n")).strip().lower()
        return {"h":"high","n":"nominal","l":"low",
                "high":"high","nominal":"nominal","low":"low"}.get(conf, "nominal")
    try:
        v = int(float(row.get("confidence", 50)))
        if v >= 80: return "high"
        if v >= 30: return "nominal"
        return "low"
    except Exception:
        return "nominal"


def row_a_alerta(row, fuente):
    try:
        lat = float(row.get("latitude") or row.get("lat"))
        lon = float(row.get("longitude") or row.get("lon"))
    except Exception:
        return None

    acq_date = row.get("acq_date", "")
    acq_time = str(row.get("acq_time", "0000")).zfill(4)
    try:
        fecha_str = f"{acq_date}T{acq_time[:2]}:{acq_time[2:]}:00+00:00"
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
            row.get("brightness") or row.get("bright_ti4") or
            row.get("bright_t31") or 0
        )
    except Exception:
        brightness = None

    return {
        "tipo":             "incendio",
        "fuente":           fuente,
        "fecha_deteccion":  fecha_str,
        "latitud":          lat,
        "longitud":         lon,
        "firms_brightness": brightness,
        "firms_frp":        frp,
        "firms_confidence": normalizar_confianza(row, fuente),
        "firms_satellite":  fuente,
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
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status, resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="ignore")
    except Exception as e:
        return 0, str(e)


def obtener_existentes():
    """
    Carga claves de TODO lo existente en Supabase para deduplicar.
    Pagina de 10 000 en 10 000 para no superar límites.
    """
    print("  Cargando registros existentes de Supabase para deduplicar…")
    existentes = set()
    offset = 0
    pagina = 10_000

    while True:
        path = (
            "alertas?select=fuente,fecha_deteccion,latitud,longitud"
            "&tipo=eq.incendio"
            f"&limit={pagina}&offset={offset}"
        )
        status, body = supabase_request("GET", path)
        if status != 200:
            print(f"  ⚠ Error leyendo Supabase (HTTP {status}): {body[:250]}")
            break
        try:
            registros = json.loads(body)
        except Exception as e:
            print(f"  ⚠ Error parseando respuesta: {e}")
            break
        if not registros:
            break
        for r in registros:
            try:
                existentes.add((
                    r.get("fuente", ""),
                    str(r.get("fecha_deteccion", ""))[:16],
                    round(float(r.get("latitud",  0)), 4),
                    round(float(r.get("longitud", 0)), 4),
                ))
            except Exception:
                continue
        offset += pagina
        if len(registros) < pagina:
            break

    print(f"  ✓ {len(existentes):,} alertas ya existentes en Supabase")
    return existentes


def insertar_lote(alertas):
    if not alertas:
        return 0
    insertadas = 0
    for i in range(0, len(alertas), 500):
        lote = alertas[i:i+500]
        status, body = supabase_request("POST", "alertas", lote)
        if status in (200, 201, 204):
            insertadas += len(lote)
            print(f"      ✓ Lote {i//500+1}: {len(lote)} insertadas")
        else:
            print(f"      ✗ Error lote {i//500+1} (HTTP {status}): {body[:300]}")
    return insertadas


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    hoy = datetime.date.today()
    dias_totales = (hoy - FECHA_INICIO).days + 1
    bloques = generar_bloques(FECHA_INICIO, hoy, DIAS_POR_BLOQUE)
    total_llamadas = len(bloques) * len(FUENTES)

    print("=" * 70)
    print("CARGA HISTÓRICA DE INCENDIOS — NASA FIRMS")
    print(f"Rango     : {FECHA_INICIO} → {hoy} ({dias_totales} días)")
    print(f"Bloque    : {DIAS_POR_BLOQUE} días (límite FIRMS con fecha)")
    print(f"Endpoints : {' · '.join(e.split('/')[2] for e in FIRMS_ENDPOINTS)}")
    print(f"Fuentes   : {', '.join(FUENTES)}")
    print(f"BBOX      : {BBOX}")
    print(f"Llamadas  : {len(bloques)} bloques × {len(FUENTES)} fuentes = {total_llamadas}")
    print("=" * 70)

    if not bloques:
        print("No hay bloques. Revisa FECHA_INICIO.")
        sys.exit(1)

    # 1. Existentes
    existentes = obtener_existentes()

    # 2. Descargar por bloque y fuente
    total_nuevas   = 0
    total_filas    = 0
    llamadas_ok    = 0
    llamadas_error = 0
    llamada_n      = 0

    for fuente in FUENTES:
        print(f"\n{'─'*70}")
        print(f"FUENTE: {fuente}")
        print(f"{'─'*70}")

        for fecha_bloque, dias in bloques:
            llamada_n += 1
            print(f"\n  [{llamada_n}/{total_llamadas}] "
                  f"{fuente} · {fecha_bloque} · {dias} días")

            filas, ok = fetch_firms(fuente, BBOX, dias, fecha_bloque)

            if not ok:
                llamadas_error += 1
                time.sleep(PAUSA_ENTRE_LLAMADAS)
                continue

            llamadas_ok += 1

            if not filas:
                print("      sin datos en este bloque")
                time.sleep(PAUSA_ENTRE_LLAMADAS)
                continue

            total_filas += len(filas)
            nuevas = []

            for fila in filas:
                alerta = row_a_alerta(fila, fuente)
                if not alerta:
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

            print(f"      {len(filas)} filas → {len(nuevas)} nuevas")

            if nuevas:
                ins = insertar_lote(nuevas)
                total_nuevas += ins

            time.sleep(PAUSA_ENTRE_LLAMADAS)

    # 3. Resumen
    print("\n" + "=" * 70)
    print("RESUMEN FINAL")
    print("=" * 70)
    print(f"  Período cubierto  : {FECHA_INICIO} → {hoy} ({dias_totales} días)")
    print(f"  Llamadas FIRMS    : {llamada_n} ({llamadas_ok} ok · {llamadas_error} error)")
    print(f"  Filas recibidas   : {total_filas:,}")
    print(f"  Alertas insertadas: {total_nuevas:,}")
    print("=" * 70)

    if llamadas_ok == 0 and llamadas_error > 0:
        print("❌ Ninguna llamada a FIRMS fue exitosa.")
        sys.exit(1)

    print("✅ Carga histórica completada.")
    print("   Revisa Supabase → public.alertas")
    print("   Desde ahora usa monitor_incendios.yml (cada 3 h) para mantenimiento.")


if __name__ == "__main__":
    main()
