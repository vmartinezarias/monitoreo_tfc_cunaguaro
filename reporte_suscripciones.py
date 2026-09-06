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
import urllib.parse
import base64
from datetime import datetime, timedelta, date

try:
    import matplotlib
    matplotlib.use("Agg")  # backend sin pantalla, necesario en GitHub Actions
    import matplotlib.pyplot as plt
    MATPLOTLIB_OK = True
except ImportError:
    MATPLOTLIB_OK = False

# ── Configuración ─────────────────────────────────────────────────────────────
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
RESEND_API_KEY = os.environ.get("RESEND_API_KEY", "")
MODO_PRUEBA = os.environ.get("MODO_PRUEBA", "").lower() in ("true", "1", "yes")
FORZAR_MES = os.environ.get("FORZAR_MES", "")

GITHUB_RAW_BASE = os.environ.get(
    "GITHUB_RAW_BASE",
    "https://raw.githubusercontent.com/vmartinezarias/monitoreo_tfc_cunaguaro/main/"
)

CSV_SUSCRIPTORES = os.environ.get("SUSCRIPTORES_FILE", "suscriptores.csv")
REPORTES_DIR = "reportes"

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Faltan credenciales de Supabase")

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
print(f"Modo prueba: {'SÍ' if MODO_PRUEBA else 'NO'}")

if not RESEND_API_KEY:
    print("⚠ RESEND_API_KEY no definido")
else:
    print(f"✓ RESEND_API_KEY configurada")

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


# ── GFW (GLAD / RADD) ────────────────────────────────────────────────────────
# Misma API key y datasets que usa app.js en el visor (GFW_API_KEY / GFW_CFG).
GFW_API_KEY = "6b196681-4bfb-4c71-8757-b745b9290f95"
GFW_BASE = "https://data-api.globalforestwatch.org"
AREA_HA_POR_ALERTA_GFW = 0.09  # resolución Landsat/Sentinel (~30m) de GLAD/RADD


def gfw_geometria_valida(geom):
    """La API de GFW solo acepta Polygon o MultiPolygon (rechaza GeometryCollection,
    que es justo lo que devuelve extraer_geometria() para suscriptores 'area_completa'
    con varias features). Aquí se combinan en un único Polygon/MultiPolygon válido."""
    polys = []

    def _recolectar(g):
        if not g:
            return
        t = g.get("type")
        if t == "Polygon":
            polys.append(g["coordinates"])
        elif t == "MultiPolygon":
            polys.extend(g["coordinates"])
        elif t == "GeometryCollection":
            for gg in g.get("geometries", []):
                _recolectar(gg)
        elif t == "Feature":
            _recolectar(g.get("geometry"))
        elif t == "FeatureCollection":
            for f in g.get("features", []):
                _recolectar(f.get("geometry"))

    _recolectar(geom)
    if not polys:
        return None
    if len(polys) == 1:
        return {"type": "Polygon", "coordinates": polys[0]}
    return {"type": "MultiPolygon", "coordinates": polys}


def gfw_sql_glad(fi, ff):
    return (
        "SELECT latitude,longitude,gfw_integrated_alerts__date AS fecha,"
        "gfw_integrated_alerts__confidence AS confianza "
        f"FROM results WHERE gfw_integrated_alerts__date>='{fi}' "
        f"AND gfw_integrated_alerts__date<='{ff}' LIMIT 2000"
    )


def gfw_sql_radd(fi, ff):
    return (
        "SELECT latitude,longitude,wur_radd_alerts__date AS fecha,"
        "wur_radd_alerts__confidence AS confianza "
        f"FROM results WHERE wur_radd_alerts__date>='{fi}' "
        f"AND wur_radd_alerts__date<='{ff}' LIMIT 2000"
    )


def obtener_alertas_gfw(dataset, sql, geometria):
    geom_valida = gfw_geometria_valida(geometria)
    if geom_valida is None:
        print(f"  ⚠ GFW {dataset}: geometría vacía, se omite")
        return []

    url = f"{GFW_BASE}/dataset/{dataset}/latest/query/json"
    body = json.dumps({"sql": sql, "geometry": geom_valida}).encode("utf-8")
    req = urllib.request.Request(
        url, data=body,
        headers={
            "Content-Type": "application/json",
            "x-api-key": GFW_API_KEY,
            "User-Agent": "reporte-cunaguaro/1.0",
            # La API key de GFW está restringida al dominio del sitio (así funciona
            # desde el navegador). Un request servidor-a-servidor no manda Referer/Origin
            # por defecto, así que hay que simularlo o GFW la rechaza como "missing valid API key".
            "Referer": "https://monitoreo-tfc-cunaguaro.vercel.app/",
            "Origin": "https://monitoreo-tfc-cunaguaro.vercel.app",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("data", [])
    except urllib.error.HTTPError as e:
        print(f"  ✗ Error GFW {dataset}: HTTP {e.code} — {e.read().decode(errors='ignore')[:200]}")
        return []
    except Exception as e:
        print(f"  ✗ Error GFW {dataset}: {e}")
        return []


# ── Consulta de alertas ─────────────────────────────────────────────────────
def obtener_alertas(tabla, fecha_ini, fecha_fin, geometria):
    # FIX: codificar fechas para URL
    fi = urllib.parse.quote(f"{fecha_ini}T00:00:00+00:00")
    ff = urllib.parse.quote(f"{fecha_fin}T23:59:59+00:00")
    
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


# ── Mapa espacial estático (imagen embebida en el correo) ───────────────────
# Un correo no puede correr JavaScript (Leaflet, clusters, calor), así que
# esto genera una imagen PNG con los puntos y el contorno del área del
# suscriptor, para dar la misma idea de "patrón espacial" que el visor web.
def _anillos_de_geometria(geom):
    """Extrae los anillos exteriores (listas de [lng,lat]) de cualquier
    geometría, para dibujar el contorno del área del suscriptor."""
    anillos = []
    g = gfw_geometria_valida(geom)  # normaliza a Polygon/MultiPolygon
    if g is None:
        return anillos
    if g["type"] == "Polygon":
        if g["coordinates"]:
            anillos.append(g["coordinates"][0])
    elif g["type"] == "MultiPolygon":
        for poly in g["coordinates"]:
            if poly:
                anillos.append(poly[0])
    return anillos


def generar_mapa_estatico(geometria, incendios, glad, radd):
    if not MATPLOTLIB_OK:
        print("  ⚠ matplotlib no disponible, se omite el mapa estático")
        return None

    anillos = _anillos_de_geometria(geometria)
    pts_inc = [(float(r.get("longitud", 0)), float(r.get("latitud", 0))) for r in incendios]
    pts_glad = [(float(r.get("longitude", 0)), float(r.get("latitude", 0))) for r in glad]
    pts_radd = [(float(r.get("longitude", 0)), float(r.get("latitude", 0))) for r in radd]

    if not anillos and not (pts_inc or pts_glad or pts_radd):
        return None

    fig, ax = plt.subplots(figsize=(6.4, 5.2), dpi=150)

    for anillo in anillos:
        xs = [p[0] for p in anillo]
        ys = [p[1] for p in anillo]
        ax.plot(xs, ys, color="#1a4a2e", linewidth=1.4, zorder=1)
        ax.fill(xs, ys, color="#1a4a2e", alpha=0.05, zorder=0)

    if pts_glad:
        ax.scatter([p[0] for p in pts_glad], [p[1] for p in pts_glad],
                   s=26, c="#0284c7", marker="o", label=f"GLAD ({len(pts_glad)})", zorder=2)
    if pts_radd:
        ax.scatter([p[0] for p in pts_radd], [p[1] for p in pts_radd],
                   s=26, c="#7c3aed", marker="^", label=f"RADD ({len(pts_radd)})", zorder=2)
    if pts_inc:
        ax.scatter([p[0] for p in pts_inc], [p[1] for p in pts_inc],
                   s=34, c="#e8480a", marker="*", label=f"Incendios ({len(pts_inc)})", zorder=3)

    todos_lats = [p[1] for p in (pts_inc + pts_glad + pts_radd)] + [p[1] for a in anillos for p in a]
    if todos_lats:
        lat_media = sum(todos_lats) / len(todos_lats)
        import math
        ax.set_aspect(1 / max(math.cos(math.radians(lat_media)), 0.15))

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if pts_inc or pts_glad or pts_radd:
        ax.legend(loc="upper right", fontsize=8, frameon=True, facecolor="white", framealpha=.9)
    fig.tight_layout(pad=0.6)

    buf_png = io.BytesIO()
    fig.savefig(buf_png, format="png", facecolor="white")
    buf_png.seek(0)
    png_b64 = base64.b64encode(buf_png.read()).decode()

    buf_pdf = io.BytesIO()
    fig.savefig(buf_pdf, format="pdf", facecolor="white")
    buf_pdf.seek(0)
    pdf_b64 = base64.b64encode(buf_pdf.read()).decode()

    plt.close(fig)
    return {"png_b64": png_b64, "pdf_b64": pdf_b64}


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

    ruta_html = os.path.join(carpeta, "reporte.html")
    with open(ruta_html, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"    💾 HTML guardado")

    for att in attachments:
        ruta = os.path.join(carpeta, att["filename"])
        if "content_b64" in att:
            with open(ruta, "wb") as f:
                f.write(base64.b64decode(att["content_b64"]))
        else:
            with open(ruta, "w", encoding="utf-8") as f:
                f.write(att["content"])
        print(f"    💾 {att['filename']} guardado")

    return carpeta


# ── Envío de correo vía Resend ──────────────────────────────────────────────
def enviar_email(destinatario, nombre, html_body, attachments):
    if MODO_PRUEBA:
        print(f"  [MODO PRUEBA] Email NO enviado")
        return 200, '{"id":"modo-prueba"}'

    if not RESEND_API_KEY:
        print(f"  [SIN API KEY] Email NO enviado")
        return 200, '{"id":"sin-api-key"}'

    # FIX: usar dirección de Resend verificada o de prueba
    # Si tienes dominio verificado en Resend, cámbialo aquí:
    from_email = os.environ.get("RESEND_FROM", "onboarding@resend.dev")

    att_list = []
    for att in attachments:
        # Adjuntos de texto (CSV/GeoJSON) llegan como texto plano y se codifican aquí;
        # los binarios (PNG/PDF del mapa) ya llegan codificados en base64 (content_b64).
        if "content_b64" in att:
            contenido_b64 = att["content_b64"]
        else:
            contenido_b64 = base64.b64encode(att["content"].encode("utf-8")).decode()

        item = {"filename": att["filename"], "content": contenido_b64}
        # content_id: si viene, Resend incrusta la imagen inline y se referencia
        # en el HTML como <img src="cid:...">, en vez de mostrarse solo como adjunto.
        if att.get("content_id"):
            item["content_id"] = att["content_id"]
        att_list.append(item)

    payload = {
        "from": f"Bosques en Movimiento <{from_email}>",
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
            # FIX: Resend/Cloudflare bloquea con 403 (error 1010) las peticiones
            # sin User-Agent explícito, porque el default de Python es "Python-urllib/x.y".
            "User-Agent": "reporte-cunaguaro/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


# ── Generador de HTML ─────────────────────────────────────────────────────────
def generar_html(nombre, incendios, glad, radd, mapa, periodo):
    total_inc = len(incendios)
    high = sum(1 for x in incendios if str(x.get("firms_confidence", "")).lower() == "high")
    frp_max = max((x.get("firms_frp") or 0 for x in incendios), default=0)

    gfw_todas = (
        [{**r, "fuente": "GLAD"} for r in glad] +
        [{**r, "fuente": "RADD"} for r in radd]
    )
    gfw_todas.sort(key=lambda r: r.get("fecha") or "", reverse=True)
    total_gfw = len(gfw_todas)
    ha_gfw = total_gfw * AREA_HA_POR_ALERTA_GFW

    filas_inc = "".join([
        f"<tr><td>{i+1}</td><td>{r.get('fecha_deteccion','')[:10]}</td>"
        f"<td>{float(r.get('latitud',0)):.4f}</td><td>{float(r.get('longitud',0)):.4f}</td>"
        f"<td>{r.get('firms_confidence','')}</td><td>{r.get('firms_frp','')}</td></tr>"
        for i, r in enumerate(incendios[:10])
    ]) if incendios else '<tr><td colspan="6" style="text-align:center;color:#94a3b8">Sin alertas este mes</td></tr>'

    filas_gfw = "".join([
        f"<tr><td>{i+1}</td><td>{str(r.get('fecha',''))[:10]}</td>"
        f"<td>{float(r.get('latitude',0)):.4f}</td><td>{float(r.get('longitude',0)):.4f}</td>"
        f"<td>{r.get('fuente','')}</td><td>{r.get('confianza','')}</td></tr>"
        for i, r in enumerate(gfw_todas[:10])
    ]) if gfw_todas else '<tr><td colspan="6" style="text-align:center;color:#94a3b8">Sin alertas GFW este mes</td></tr>'

    mapa_html = (
        f'<h3 style="margin-top:24px;font-size:14px;color:#1a4a2e;">🗺️ Patrón espacial (incendios + GFW)</h3>'
        f'<img src="cid:mapa-alertas" alt="Mapa de alertas" '
        f'style="width:100%;border-radius:12px;border:1px solid #e2e8f0;margin-top:8px;">'
        f'<p style="font-size:10px;color:#94a3b8;margin-top:4px;">'
        f'¿No ves el mapa? Va también adjunto como imagen y como PDF en este correo.</p>'
    ) if mapa else ""

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
  <h3 style="margin-top:8px;font-size:14px;color:#1a4a2e;">🔥 Incendios (NASA FIRMS)</h3>
  <div class="kpi-wrap">
    <div class="kpi"><div class="kpi-num">{total_inc}</div><div class="kpi-label">Incendios</div></div>
    <div class="kpi"><div class="kpi-num">{high}</div><div class="kpi-label">Alta confianza</div></div>
    <div class="kpi"><div class="kpi-num">{frp_max and f'{frp_max:.1f}' or '—'}</div><div class="kpi-label">FRP máx. (MW)</div></div>
  </div>
  <table><thead><tr><th>#</th><th>Fecha</th><th>Lat</th><th>Lon</th><th>Conf.</th><th>FRP</th></tr></thead><tbody>{filas_inc}</tbody></table>

  <h3 style="margin-top:24px;font-size:14px;color:#1a4a2e;">🛰️ Global Forest Watch (GLAD + RADD)</h3>
  <div class="kpi-wrap">
    <div class="kpi"><div class="kpi-num">{len(glad)}</div><div class="kpi-label">GLAD</div></div>
    <div class="kpi"><div class="kpi-num">{len(radd)}</div><div class="kpi-label">RADD</div></div>
    <div class="kpi"><div class="kpi-num">{ha_gfw:.2f}</div><div class="kpi-label">Ha estimadas</div></div>
  </div>
  <table><thead><tr><th>#</th><th>Fecha</th><th>Lat</th><th>Lon</th><th>Fuente</th><th>Confianza</th></tr></thead><tbody>{filas_gfw}</tbody></table>

  {mapa_html}
  <div class="footer">CBC Cunaguaro · TFCA Colombia</div>
</div>
</body></html>"""


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("REPORTE MENSUAL DE SUSCRIPTORES")
    print("=" * 65)

    if not os.path.exists(CSV_SUSCRIPTORES):
        print(f"✗ FATAL: No se encontró {CSV_SUSCRIPTORES}")
        sys.exit(1)

    suscriptores = []
    with open(CSV_SUSCRIPTORES, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            suscriptores.append({k.strip(): (v.strip() if v else "") for k, v in row.items()})

    print(f"Suscriptores leídos: {len(suscriptores)}")
    if not suscriptores:
        print("✗ No hay suscriptores")
        sys.exit(1)

    for s in suscriptores:
        email = s.get("email", "")
        nombre = s.get("nombre", "")
        tipo = s.get("tipo_reporte", "completo")
        capa = s.get("capa_geojson", "")
        col_match = s.get("columna_match", "")
        val_match = s.get("valor_match", "")
        area_completa = s.get("area_completa", "").strip().lower() in ("si", "sí", "true", "1", "yes")

        if not email or not capa:
            print(f"\n→ Fila incompleta, saltando")
            continue

        print(f"\n{'─' * 65}")
        print(f"→ {email} | capa: {capa}")
        print(f"  match: {col_match or '-'} = {val_match or '(toda la capa)'}")

        # Sin columna_match/valor_match, extraer_geometria() devuelve TODA la capa.
        # Eso es válido solo si el suscriptor lo pidió explícitamente
        # (columna "area_completa" = si en el CSV); si no, es casi seguro
        # un dato faltante del formulario y el reporte saldría idéntico
        # al de cualquier otro suscriptor de la misma capa.
        if (not col_match or not val_match) and not area_completa:
            print(f"  ⚠ SIN FILTRO GEOGRÁFICO y sin 'area_completa=si' — "
                  f"este suscriptor recibiría el área completa de '{capa}'. "
                  f"Saltando para evitar reporte duplicado/incorrecto. "
                  f"Revisa columna_match/valor_match o marca area_completa=si.")
            continue

        gj = obtener_capa(capa)
        if gj is None:
            print("  ✗ No se pudo cargar la capa")
            continue

        geometria = extraer_geometria(gj, col_match, val_match)
        if geometria is None:
            print("  ✗ No se pudo extraer geometría")
            continue

        incendios = []
        glad = []
        radd = []

        if tipo in ("completo", "incendios"):
            incendios = obtener_alertas("alertas", fecha_ini, fecha_fin, geometria)
            print(f"  Incendios: {len(incendios)}")

        if tipo in ("completo", "deforestacion"):
            glad = obtener_alertas_gfw("gfw_integrated_alerts", gfw_sql_glad(fecha_ini, fecha_fin), geometria)
            print(f"  GFW GLAD: {len(glad)}")

            radd = obtener_alertas_gfw("wur_radd_alerts", gfw_sql_radd(fecha_ini, fecha_fin), geometria)
            print(f"  GFW RADD: {len(radd)}")

        mapa = generar_mapa_estatico(geometria, incendios, glad, radd)
        print(f"  Mapa espacial: {'generado' if mapa else 'omitido (sin datos o sin matplotlib)'}")

        html = generar_html(nombre, incendios, glad, radd, mapa, mes_anio)

        attachments = []
        if incendios:
            csv_inc = csv_from_rows(incendios, ["fecha_deteccion", "latitud", "longitud", "firms_confidence", "firms_frp", "firms_satellite"])
            attachments.append({"filename": "incendios.csv", "content": csv_inc, "mime": "text/csv"})
            attachments.append({"filename": "incendios.geojson", "content": geojson_from_rows(incendios), "mime": "application/geo+json"})
        if glad or radd:
            gfw_rows = (
                [{**r, "fuente": "GLAD", "latitud": r.get("latitude"), "longitud": r.get("longitude"), "fecha_deteccion": r.get("fecha")} for r in glad] +
                [{**r, "fuente": "RADD", "latitud": r.get("latitude"), "longitud": r.get("longitude"), "fecha_deteccion": r.get("fecha")} for r in radd]
            )
            csv_gfw = csv_from_rows(gfw_rows, ["fecha_deteccion", "latitud", "longitud", "fuente", "confianza"])
            attachments.append({"filename": "gfw_glad_radd.csv", "content": csv_gfw, "mime": "text/csv"})
            attachments.append({"filename": "gfw_glad_radd.geojson", "content": geojson_from_rows(gfw_rows), "mime": "application/geo+json"})
        if mapa:
            # content_id -> Resend la incrusta inline y el HTML la referencia con cid:mapa-alertas.
            # Igual queda disponible como adjunto normal (PNG) por si el cliente de correo no la muestra inline.
            attachments.append({"filename": "mapa_alertas.png", "content_b64": mapa["png_b64"], "content_id": "mapa-alertas", "mime": "image/png"})
            attachments.append({"filename": "mapa_alertas.pdf", "content_b64": mapa["pdf_b64"], "mime": "application/pdf"})

        guardar_reporte(email, html, attachments)

        status, resp = enviar_email(email, nombre, html, attachments)
        if status == 200:
            print(f"  ✓ Correo enviado a {email}")
        else:
            print(f"  ✗ Error enviando: HTTP {status} — {resp[:300]}")

    print("\n" + "=" * 65)
    print("✅ Reportes completados.")
    print(f"📁 Descarga los archivos en: Actions → este run → Artifacts")
    print("=" * 65)


if __name__ == "__main__":
    main()
