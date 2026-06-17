"""
monitor_incendios.py
────────────────────
Consulta NASA FIRMS para la zona de Chámeza/Casanare y guarda las alertas
recientes de incendios en la tabla `alertas` de Supabase.

Corre cada 3 horas via GitHub Actions.
NO borra alertas antiguas — la base histórica crece indefinidamente.

NOTAS TÉCNICAS FIRMS:
  - Con fecha específica: máximo 5 días por llamada.
  - Se prueban dos endpoints: firms2 (primario) y firms (fallback).
  - Timeout extendido a 120s para runners con latencia alta.

Variables de entorno requeridas:
  SUPABASE_URL  — URL del proyecto Supabase
  SUPABASE_KEY  — service_role key de Supabase
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


# ── Forzar IPv4 (fix GitHub Actions) ──────────────────────────────────────────
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

BBOX            = "-73.20,4.60,-71.80,5.60"
DIAS_TOTAL      = 30
DIAS_POR_BLOQUE = 5    # máximo permitido por FIRMS con fecha específica

FUENTES = [
    "VIIRS_SNPP_NRT",
    "VIIRS_NOAA20_NRT",
    "MODIS_NRT",
]

# Endpoints en orden de preferencia — si el primero falla se prueba el segundo
FIRMS_ENDPOINTS = [
    "https://firms2.modaps.eosdis.nasa.gov/api/area/csv",
    "https://firms.modaps.eosdis.nasa.gov/api/area/csv",
]

TIMEOUT              = 120   # segundos — runners CI pueden ser lentos
PAUSA_ENTRE_LLAMADAS = 3
MAX_REINTENTOS       = 2
PAUSA_REINTENTO      = 8


# ── Helpers ────────────────────────────────────────────────────────────────────
def ocultar_key(url):
    return url.replace(FIRMS_KEY, "***")


def generar_bloques(fecha_fin, dias_total, dias_bloque):
    """Genera bloques (fecha_inicio, n_dias) cubriendo los últimos dias_total días."""
    fecha_inicio = fecha_fin - datetime.timedelta(days=dias_total - 1)
    bloques = []
    cursor = fecha_inicio
    while cursor <= fecha_fin:
        dias = min(dias_bloque, (fecha_fin - cursor).days + 1)
        bloques.append((cursor.strftime("%Y-%m-%d"), dias))
        cursor += datetime.timedelta(days=dias)
    return bloques


def fetch_firms(producto, bbox, dias, fecha_str):
    """
    Intenta descargar el CSV de FIRMS probando ambos endpoints.
    Retorna (filas, ok).
    """
    # Contexto SSL que no verifica certificado — necesario en algunos runners
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
                        "User-Agent": "monitor-incendios-cunaguaro/3.0",
                        "Accept":     "text/csv,*/*",
                    },
                    method="GET"
                )
                with urllib.request.urlopen(req, timeout=TIMEOUT, context=ctx) as resp:
                    raw = resp.read().decode("utf-8", errors="replace")

                # Respuesta vacía = sin fuegos en esa zona/período → OK
                if not raw.strip():
                    return [], True

                # Error textual de FIRMS
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


def obtener_existentes(dias=32):
    print("  Cargando alertas existentes de Supabase…")
    fecha_min = (
        datetime.datetime.utcnow() - datetime.timedelta(days=dias)
    ).isoformat() + "+00:00"
    path = (
        f"alertas?select=fuente,fecha_deteccion,latitud,longitud"
        f"&tipo=eq.incendio"
        f"&fecha_deteccion=gte.{urllib.parse.quote(fecha_min)}"
        f"&limit=100000"
    )
    status, body = supabase_request("GET", path)
    if status != 200:
        print(f"  ⚠ Error leyendo Supabase (HTTP {status}): {body[:250]}")
        return set()
    try:
        registros = json.loads(body)
    except Exception as e:
        print(f"  ⚠ Error parseando respuesta: {e}")
        return set()
    existentes = set()
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
    print(f"  ✓ {len(existentes):,} alertas existentes en ventana de {dias} días")
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
    hoy = datetime.datetime.utcnow().date()

    print("=" * 70)
    print("MONITOR DE INCENDIOS — NASA FIRMS")
    print(f"Fecha UTC : {datetime.datetime.utcnow().isoformat(timespec='seconds')}")
    print(f"Ventana   : últimos {DIAS_TOTAL} días  |  Bloque: {DIAS_POR_BLOQUE} días")
    print(f"Endpoints : {' · '.join(e.split('/')[2] for e in FIRMS_ENDPOINTS)}")
    print(f"Fuentes   : {', '.join(FUENTES)}")
    print(f"BBOX      : {BBOX}")
    print("=" * 70)

    bloques = generar_bloques(hoy, DIAS_TOTAL, DIAS_POR_BLOQUE)
    total_llamadas = len(bloques) * len(FUENTES)
    print(f"\n{len(bloques)} bloques × {len(FUENTES)} fuentes = {total_llamadas} llamadas\n")

    print("[1] Alertas existentes en Supabase…")
    existentes = obtener_existentes(dias=DIAS_TOTAL + 2)

    print(f"\n[2] Descargando de NASA FIRMS…")
    todas_nuevas   = []
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
            todas_nuevas.extend(nuevas)
            time.sleep(PAUSA_ENTRE_LLAMADAS)

    print(f"\n[3] Total nuevas alertas: {len(todas_nuevas):,}")

    if todas_nuevas:
        print("\n[4] Insertando en Supabase…")
        insertadas = insertar_lote(todas_nuevas)
        print(f"\n  ✅ {insertadas:,} alertas insertadas")
    else:
        print("\n  ℹ Sin alertas nuevas")

    print("\n[5] Histórico activo — no se eliminan alertas antiguas")

    print("\n" + "=" * 70)
    print("RESUMEN FINAL")
    print("=" * 70)
    print(f"  Llamadas: {llamada_n} total · {llamadas_ok} ok · {llamadas_error} error")
    print(f"  Filas FIRMS recibidas : {total_filas:,}")
    print(f"  Alertas nuevas        : {len(todas_nuevas):,}")
    print("=" * 70)

    if llamadas_ok == 0 and llamadas_error > 0:
        print("❌ Ninguna llamada a FIRMS fue exitosa.")
        sys.exit(1)

    print("✅ Monitor de incendios completado")


if __name__ == "__main__":
    main()
