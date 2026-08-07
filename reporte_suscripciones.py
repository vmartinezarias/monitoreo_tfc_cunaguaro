"""
reporte_suscripciones.py
────────────────────────
Ejecutar vía GitHub Actions.
Lee suscriptores desde suscriptores.csv, genera reportes y los envía por email.
"""
import os
import sys
import json
import csv
import io
import urllib.request
import base64
from datetime import datetime, timedelta, date

# ── Configuración ─────────────────────────────────────────────────────────────
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
RESEND_API_KEY = os.environ.get("RESEND_API_KEY", "")
MODO_PRUEBA = os.environ.get("MODO_PRUEBA", "").lower() in ("true", "1", "yes")
FORZAR_MES = os.environ.get("FORZAR_MES", "")  # ej: "2026-08"

GITHUB_RAW_BASE = os.environ.get(
    "GITHUB_RAW_BASE",
    "https://raw.githubusercontent.com/vmartinezarias/monitoreo_tfc_cunaguaro/main/"
)

CSV_SUSCRIPTORES = "suscriptores.csv"
REPORTES_DIR = "reportes"

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Faltan credenciales de Supabase")

# Crear carpeta de reportes
os.makedirs(REPORTES_DIR, exist_ok=True)

# ── Calcular rango de fechas ─────────────────────────────────────────────────
hoy = date.today()

if FORZAR_MES and len(FORZAR_MES.split("-")) == 2:
    anio, mes = map(int, FORZAR_MES.split("-"))
    primer_dia_mes = date(anio, mes, 1)
    if mes == 12:
        siguiente = date(anio + 1, 1, 1)
    else:
        siguiente = date(anio, mes + 1, 1)
    fecha_fin = siguiente - timedelta(days=1)
    fecha_ini = primer_dia_mes
    mes_anio = primer_dia_mes.strftime("%B %Y")
    print(f"⚠ MODO PRUEBA: forzando rango {fecha_ini} → {fecha_fin}")
else:
    primer_dia_mes = hoy.replace(day=1)
    mes_anio = primer_dia_mes.strftime("%B %Y")
    fecha_fin = primer_dia_mes - timedelta(days=1)
    fecha_ini = fecha_fin.replace(day=1)

print(f"Reporte para: {fecha_ini} → {fecha_fin}")
print(f"Modo prueba: {'SÍ (no se enviarán emails)' if MODO_PRUEBA else 'NO (se intentarán enviar emails)'}")

if not RESEND_API_KEY:
    print("⚠ RESEND_API_KEY no definido — los correos NO se enviarán.")
else:
    print(f"✓ RESEND_API_KEY configurada ({RESEND_API_KEY[:8]}...)")

# ── Helpers HTTP ──────────────────────────────────────────────────────────────
def http_get_json(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "reporte-cunaguaro/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"  ✗ Error descargando {url}: {e}")
        return None


def supabase_request(method, path, body=None):
    url = f"{SUPABASE_URL}/rest/v1/{path}"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }
    data = json.dumps(body).encode("utf-8") if body else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status, resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="ignore")
    except Exception as e:
        return 0, str(e)


# ── Cache de capas GeoJSON ───────────────────────────────────────────────────
_capa_cache = {}

def obtener_capa(nombre_archivo):
    if nombre_archivo in _capa_cache:
        return _capa_cache[nombre_archivo]

    url = nombre_archivo if nombre_archivo.startswith("http") else f"{GITHUB_RAW_BASE}{nombre_archivo}"
    print(f"  ↓ Descargando capa: {url}")

    gj = http_get_json(url)
    if gj is None:
        return None

    _capa_cache[nombre_archivo] = gj
    tipo = gj.get("type", "?")
    n_feat = len(gj.get("features", [])) if tipo == "FeatureCollection" else 1
    print(f"  ✓ Capa lista: {tipo} con {n_feat} feature(s)")
    return gj


def extraer_geometria(gj, columna_match, valor_match):
    if not columna_match or not valor_match:
        if gj.get("type") == "FeatureCollection":
            geoms = [f["geometry"] for f in gj.get("features", []) if f.get("geometry")]
            if not geoms:
                return None
            return {"type": "GeometryCollection", "geometries": geoms}
        elif gj.get("type") == "Feature":
            return gj.get("geometry")
        else:
            return gj

    features = []
    if gj.get("type") == "FeatureCollection":
        features = gj.get("features", [])
    elif gj.get("type") == "Feature":
        features = [gj]

    valor_busqueda = str(valor_match).strip().lower()

    for f in features:
        props = f.get("properties", {})
        for k, v in props.items():
            if k.strip().lower() == columna_match.strip().lower():
                if str(v).strip().lower() == valor_busqueda:
                    print(f"  ✓ Match encontrado: {k} = {v}")
                    return f.get("geometry")
    print(f"  ⚠ No se encontró coincidencia para {columna_match} = {valor_match}")
    return None


# ── Geometría: punto en polígono ──────────────────────────────────────────────
def point_in_polygon(lat, lng, geom):
    if geom is None:
        return False
    if geom.get("type") == "Feature":
        return point_in_polygon(lat, lng, geom.get("geometry", {}))
    if geom.get("type") == "FeatureCollection":
        for f in geom.get("features", []):
            if point_in_polygon(lat, lng, f):
                return True
        return False
    if geom.get("type") == "GeometryCollection":
        for g in geom.get("geometries", []):
            if point_in_polygon(lat, lng, g):
                return True
        return False

    t = geom.get("type")
    coords = geom.get("coordinates", [])

    if t == "Polygon":
        rings = coords
        if not rings:
            return False
        if _ring_contains(lng, lat, rings[0]):
            for hole in rings[1:]:
                if _ring_contains(lng, lat, hole):
                    return False
            return True
        return False

    if t == "MultiPolygon":
        for poly in coords:
            if point_in_polygon(lat, lng, {"type": "Polygon", "coordinates": poly}):
                return True
        return False
    return False


def _ring_contains(x, y, ring):
    inside = False
    j = len(ring) - 1
    for i in range(len(ring)):
        xi, yi = ring[i]
        xj, yj = ring[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi):
            inside = not inside
        j = i
    return inside


# ── Consulta de alertas ─────────────────────────────────────────────────────
def obtener_alertas(tabla, fecha_ini, fecha_fin, geometria):
    fi = f"{fecha_ini}T00:00:00+00:00"
    ff = f"{fecha_fin}T23:59:59+00:00"
    path = (
        f"{tabla}?select=*"
        f"&fecha_deteccion=gte.{fi}"
        f"&fecha_deteccion=lte.{ff}"
        f"&limit=10000"
    )
    status, body = supabase_request("GET", path)
    if status != 200:
        print(f"  ⚠ Error leyendo {tabla}: HTTP {status}")
        return []
    try:
        rows = json.loads(body)
    except Exception as e:
        print(f"  ⚠ Error parseando {tabla}: {e}")
        return []

    filtradas = []
    for r in rows:
        try:
            lat = float(r.get("latitud", r.get("latitude", 0)))
            lng = float(r.get("longitud", r.get("longitude", 0)))
        except Exception:
            continue
        if point_in_polygon(lat, lng, geometria):
            filtradas.append(r)
    return filtradas


# ── Generación de adjuntos ───────────────────────────────────────────────────
def csv_from_rows(rows, headers):
    if not rows:
        return ""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=headers, extrasaction="ignore")
    writer.writeheader()
    for r in rows:
        writer.writerow(r)
    return buf.getvalue()


def geojson_from_rows(rows):
    features = []
    for i, r in enumerate(rows):
        try:
            lat = float(r.get("latitud", r.get("latitude", 0)))
            lng = float(r.get("longitud", r.get("longitude", 0)))
        except Exception:
            continue
        props = {k: v for k, v in r.items() if k not in ("latitud", "longitud", "latitude", "longitude")}
        props["_reporte_id"] = i + 1
        features.append({
            "type": "Feature",
            "properties": props,
            "geometry": {"type": "Point", "coordinates": [lng, lat]}
        })
    return json.dumps({"type": "FeatureCollection", "features": features}, indent=2)


# ── Guardar archivos localmente ──────────────────────────────────────────────
def guardar_reporte(email, html, attachments):
    safe_email = email.replace("@", "_").replace(".", "_")
    carpeta = os.path.join(REPORTES_DIR, safe_email)
    os.makedirs(carpeta, exist_ok=True)

    # Guardar HTML
    ruta_html = os.path.join(carpeta, "reporte.html")
    with open(ruta_html, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"    💾 HTML guardado: {ruta_html}")

    # Guardar adjuntos
    for att in attachments:
        ruta = os.path.join(carpeta, att["filename"])
        with open(ruta, "w", encoding="utf-8") as f:
            f.write(att["content"])
        print(f"    💾 {att['filename']} guardado")

    return carpeta


# ── Envío de correo vía Resend ──────────────────────────────────────────────
def enviar_email(destinatario, nombre, html_body, attachments):
    if MODO_PRUEBA:
        print(f"  [MODO PRUEBA] Email NO enviado a {destinatario}")
        return 200, '{"id":"modo-prueba"}'

    if not RESEND_API_KEY:
        print(f"  [SIN API KEY] Email NO enviado a {destinatario}")
        return 200, '{"id":"sin-api-key"}'

    att_list = []
    for att in attachments:
        att_list.append({
            "filename": att["filename"],
            "content": base64.b64encode(att["content"].encode("utf-8")).decode(),
        })

    payload = {
        "from": "Bosques en Movimiento <reportes@cunaguaro.org>",
        "to": [destinatario],
        "subject": f"Tu reporte mensual - {mes_anio}",
        "html": html_body,
    }
    if att_list:
        payload["attachments"] = att_list

    req = urllib.request.Request(
        "https://api.resend.com/emails",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {RESEND_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


# ── Generador de HTML ─────────────────────────────────────────────────────────
def generar_html(nombre, incendios, defor, periodo):
    total_inc = len(incendios)
    total_def = len(defor)
    high = sum(1 for x in incendios if str(x.get("firms_confidence", "")).lower() == "high")
    frp_max = max((x.get("firms_frp") or 0 for x in incendios), default=0)
    severa = sum(1 for x in defor if x.get("severidad") == "severa")
    ha_total = sum(x.get("area_afectada_ha") or 0 for x in defor)

    filas_inc = "".join([
        f"<tr><td>{i+1}</td><td>{r.get('fecha_deteccion','')[:10]}</td>"
        f"<td>{float(r.get('latitud',0)):.4f}</td><td>{float(r.get('longitud',0)):.4f}</td>"
        f"<td>{r.get('firms_confidence','')}</td><td>{r.get('firms_frp','')}</td></tr>"
        for i, r in enumerate(incendios[:10])
    ]) if incendios else '<tr><td colspan="6" style="text-align:center;color:#94a3b8">Sin alertas este mes</td></tr>'

    filas_def = "".join([
        f"<tr><td>{i+1}</td><td>{r.get('fecha_deteccion','')[:10]}</td>"
        f"<td>{float(r.get('latitud',0)):.4f}</td><td>{float(r.get('longitud',0)):.4f}</td>"
        f"<td>{r.get('severidad','')}</td><td>{r.get('area_afectada_ha','')}</td></tr>"
        for i, r in enumerate(defor[:10])
    ]) if defor else '<tr><td colspan="6" style="text-align:center;color:#94a3b8">Sin alertas este mes</td></tr>'

    return f"""<!DOCTYPE html>
<html lang="es">
<head><meta charset="UTF-8"><style>
body{{font-family:Inter,sans-serif;background:#f0f4f1;color:#1e293b;padding:24px;}}
.card{{background:#fff;border-radius:16px;padding:24px;max-width:640px;margin:0 auto;box-shadow:0 4px 20px rgba(0,0,0,.06);}}
h1{{font-family:Syne,sans-serif;color:#1a4a2e;font-size:22px;margin-bottom:4px;}}
.sub{{color:#64748b;font-size:12px;margin-bottom:20px;}}
.kpi-wrap{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:20px;}}
.kpi{{background:#f8faf9;border-radius:12px;padding:14px;text-align:center;border:1px solid #e2e8f0;}}
.kpi-num{{font-size:24px;font-weight:800;color:#1a4a2e;}}
.kpi-label{{font-size:10px;color:#94a3b8;text-transform:uppercase;letter-spacing:.1em;margin-top:4px;}}
table{{width:100%;border-collapse:collapse;font-size:12px;margin-top:8px;}}
th{{background:#f1f5f9;text-align:left;padding:8px;color:#64748b;font-size:10px;text-transform:uppercase;}}
td{{padding:8px;border-bottom:1px solid #f1f5f9;}}
.footer{{text-align:center;color:#94a3b8;font-size:11px;margin-top:24px;}}
</style></head>
<body>
<div class="card">
  <h1>Hola, {nombre or 'Usuario'} 👋</h1>
  <div class="sub">Reporte mensual · {periodo} · Bosques en Movimiento</div>
  <div class="kpi-wrap">
    <div class="kpi"><div class="kpi-num">{total_inc}</div><div class="kpi-label">Incendios</div></div>
    <div class="kpi"><div class="kpi-num">{high}</div><div class="kpi-label">Alta confianza</div></div>
    <div class="kpi"><div class="kpi-num">{frp_max and f'{frp_max:.1f}' or '—'}</div><div class="kpi-label">FRP máx. (MW)</div></div>
  </div>
  <div class="kpi-wrap" style="margin-top:12px;">
    <div class="kpi"><div class="kpi-num">{total_def}</div><div class="kpi-label">Deforestación</div></div>
    <div class="kpi"><div class="kpi-num">{severa}</div><div class="kpi-label">Severas</div></div>
    <div class="kpi"><div class="kpi-num">{ha_total:.2f}</div><div class="kpi-label">Ha afectadas</div></div>
  </div>
  <h3 style="margin-top:24px;font-size:14px;color:#1a4a2e;">🔥 Últimos incendios</h3>
  <table><thead><tr><th>#</th><th>Fecha</th><th>Lat</th><th>Lon</th><th>Conf.</th><th>FRP</th></tr></thead><tbody>{filas_inc}</tbody></table>
  <h3 style="margin-top:24px;font-size:14px;color:#1a4a2e;">🌿 Últimas deforestaciones</h3>
  <table><thead><tr><th>#</th><th>Fecha</th><th>Lat</th><th>Lon</th><th>Severidad</th><th>Ha</th></tr></thead><tbody>{filas_def}</tbody></table>
  <div class="footer">CBC Cunaguaro · TFCA Colombia</div>
</div>
</body></html>"""


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("REPORTE MENSUAL DE SUSCRIPTORES")
    print("=" * 65)

    # 1. Leer CSV local
    if not os.path.exists(CSV_SUSCRIPTORES):
        print(f"✗ FATAL: No se encontró {CSV_SUSCRIPTORES}")
        print("   Asegúrate de que el archivo esté commiteado en la rama main.")
        sys.exit(1)

    suscriptores = []
    with open(CSV_SUSCRIPTORES, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            suscriptores.append({k.strip(): (v.strip() if v else "") for k, v in row.items()})

    print(f"Suscriptores leídos: {len(suscriptores)}")
    if not suscriptores:
        print("✗ No hay suscriptores en el CSV.")
        sys.exit(1)

    for s in suscriptores:
        email = s.get("email", "")
        nombre = s.get("nombre", "")
        tipo = s.get("tipo_reporte", "completo")
        capa = s.get("capa_geojson", "")
        col_match = s.get("columna_match", "")
        val_match = s.get("valor_match", "")

        if not email or not capa:
            print(f"\n→ Fila incompleta, saltando: {s}")
            continue

        print(f"\n{'─' * 65}")
        print(f"→ {email} | capa: {capa} | match: {col_match or '-'} = {val_match or '(toda)'}")
        print(f"{'─' * 65}")

        # 2. Obtener capa y geometría
        gj = obtener_capa(capa)
        if gj is None:
            print("  ✗ No se pudo cargar la capa, saltando.")
            continue

        geometria = extraer_geometria(gj, col_match, val_match)
        if geometria is None:
            print("  ✗ No se pudo extraer geometría, saltando.")
            continue

        # 3. Consultar alertas
        incendios = []
        defor = []

        if tipo in ("completo", "incendios"):
            incendios = obtener_alertas("alertas", fecha_ini, fecha_fin, geometria)
            print(f"  Incendios encontrados: {len(incendios)}")

        if tipo in ("completo", "deforestacion"):
            defor = obtener_alertas("alertas_deforestacion", fecha_ini, fecha_fin, geometria)
            print(f"  Deforestación encontradas: {len(defor)}")

        # 4. Generar reporte
        html = generar_html(nombre, incendios, defor, mes_anio)

        attachments = []
        if incendios:
            csv_inc = csv_from_rows(incendios, ["fecha_deteccion", "latitud", "longitud", "firms_confidence", "firms_frp", "firms_satellite"])
            attachments.append({"filename": f"incendios.csv", "content": csv_inc, "mime": "text/csv"})
            attachments.append({"filename": f"incendios.geojson", "content": geojson_from_rows(incendios), "mime": "application/geo+json"})
        if defor:
            csv_def = csv_from_rows(defor, ["fecha_deteccion", "latitud", "longitud", "severidad", "area_afectada_ha", "cambio_ndvi"])
            attachments.append({"filename": f"deforestacion.csv", "content": csv_def, "mime": "text/csv"})
            attachments.append({"filename": f"deforestacion.geojson", "content": geojson_from_rows(defor), "mime": "application/geo+json"})

        # 5. Guardar localmente (siempre)
        guardar_reporte(email, html, attachments)

        # 6. Enviar email
        status, resp = enviar_email(email, nombre, html, attachments)
        if status == 200:
            print(f"  ✓ Correo enviado a {email}")
        else:
            print(f"  ✗ Error enviando a {email}: HTTP {status} — {resp[:300]}")

    print("\n" + "=" * 65)
    print("✅ Reportes completados.")
    print(f"📁 Archivos guardados en: {REPORTES_DIR}/")
    print("=" * 65)


if __name__ == "__main__":
    main()
