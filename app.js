// ── Configuración ─────────────────────────────────────────────────────────────
const SUPABASE_URL = 'https://qryktfqnicnwiuwbijvd.supabase.co';
const SUPABASE_KEY = 'sb_publishable_tVzVULeeVvP-QRCXviO7cg_EOuOfX-0';
const GEOJSON_URL  = 'https://raw.githubusercontent.com/vmartinezarias/monitoreo_tfc_cunaguaro/main/Predios.geojson';
const BASE_URL     = 'https://raw.githubusercontent.com/vmartinezarias/monitoreo_tfc_cunaguaro/main/';

const GFW_API_KEY  = '6b196681-4bfb-4c71-8757-b745b9290f95';
const GFW_BASE     = 'https://data-api.globalforestwatch.org';
const CHAMEZA_BBOX = { xmin: -72.80, ymin: 4.85, xmax: -72.20, ymax: 5.35 };
const GFW_DATASETS = { glad:'gfw_integrated_alerts', radd:'wur_radd_alerts', hansen:'umd_tree_cover_loss', fires:'nasa_viirs_fire_alerts' };

const URLS = {
  municipio:    BASE_URL + 'Chameza4326.geojson',
  estudio:      BASE_URL + 'area_estudio.geojson',
  bosque:       BASE_URL + 'Nucleos_boscosos.geojson',
  perdidas:     BASE_URL + 'Perdidas2020-2024.geojson',
  conectividad: BASE_URL + 'cum_currmap_deciles3_4326_web.tif'
};

// ── Paleta ────────────────────────────────────────────────────────────────────
const PALETA = ['#4caf7d','#f5a623','#e8480a','#378add','#a855f7','#ec4899','#14b8a6','#f97316','#84cc16','#64748b'];
const mapaColores = {};
function obtenerColor(attr, val) {
  if (!mapaColores[attr]) mapaColores[attr] = {};
  const mc = mapaColores[attr];
  if (!mc[val]) mc[val] = PALETA[Object.keys(mc).length % PALETA.length];
  return mc[val];
}

// ── Mapa ── CAPA BASE: Google Maps (blanco) por defecto ──────────────────────
const baseMaps = {
  'Claro (CartoDB)': L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',{attribution:'© CARTO',subdomains:'abcd',maxZoom:19}),
  'Google Maps': L.tileLayer('https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}',{attribution:'© Google'}),
  'Satélite':    L.tileLayer('https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',{attribution:'© Google'}),
  'Relieve':     L.tileLayer('https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png',{attribution:'© OpenStreetMap'}),
  'Oscuro':      L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',{attribution:'© CARTO',subdomains:'abcd',maxZoom:19})
};
const map = L.map('map',{zoomControl:false,layers:[baseMaps['Claro (CartoDB)']]}).setView([5.09,-72.47],11);
L.control.zoom({position:'bottomright'}).addTo(map);
L.control.layers(baseMaps,null,{position:'bottomright'}).addTo(map);

// ── Estado ────────────────────────────────────────────────────────────────────
let todasAlertas = [], marcadores = [];
let filtroActual = 'all', diasActual = 30;
let modoPeriodo = 'rango';
let fechaInicio = null;
let fechaFin    = null;
const FECHA_MIN_DATA = '2026-01-01';
let areaFiltro = null, predioSelId = null;
let prediosAgrupados = {}, atributoActual = 'cobertura';
let opacidadActual = 0.4, radioKm = 10;
let capaPredioa = null, capaRadio = null;
let tabActual = 'incendios';
let tipoActual = 'incendios';

// Geometrías cargadas para áreas de análisis
let geomAreaEstudio  = null;
let geomNucleos      = null;

// Datos veredas
let municipiosGJ     = null;
let veredasGJ        = null;
let municipioActual  = '';
let capaMunicipioViz = null;
let capaVeredasViz   = null;

// Área de análisis activa: 'estudio' | 'nucleos' | 'municipio' | 'dibujo'
let areaAnalisisActiva = 'estudio';

// ── DIBUJO POLÍGONO PERSONALIZADO ─────────────────────────────────────────────
let dibujoPoligonoLayer  = null;
let dibujoPoligonoCoords = null;
let dibujoHandler        = null;
let dibujoArea_ha        = 0;

const RECORTAR_DIBUJO_A_AREA_ESTUDIO = false;

const drawnItems   = new L.FeatureGroup().addTo(map);
const capaTodosGeo = new L.FeatureGroup().addTo(map);

const capasInfo = {
  municipio:    L.layerGroup().addTo(map),
  estudio:      L.layerGroup().addTo(map),
  bosque:       L.layerGroup().addTo(map),
  perdidas:     L.layerGroup().addTo(map),
  conectividad: L.layerGroup().addTo(map)
};
let datosPerdidas = null, georasterConectividad = null, cuantilMinimo = 1;
const turboColors = [null,'#30123b','#4145ab','#4675ed','#39a2ff','#1ae4b6','#62fc6b','#addc30','#e49c1b','#f25d0a','#aa1016'];

let marcadoresDefor = [], alertasDefor = [], deforVisible = false;
const COLORES_SEV = { leve:'#eab308', moderada:'#f97316', severa:'#dc2626' };

const gfwCapas = {
  glad:   { visible:false, marcadores:[], datos:[], tileLayer:null },
  radd:   { visible:false, marcadores:[], datos:[], tileLayer:null },
  hansen: { visible:false, marcadores:[], datos:[], tileLayer:null },
  fires:  { visible:false, marcadores:[], datos:[], tileLayer:null }
};
let hansenOpacity = 0.7;

// ── GFW geometría área de estudio ─────────────────────────────────────────────
let areaEstudioGeom = null, areaEstudioCargando = false;

// Helpers to safely typecast and check coordinates
function _numCoord(v) {
  const n = (typeof v === 'number') ? v : parseFloat(String(v).replace(',', '.'));
  return Number.isFinite(n) ? n : null;
}

function _alertaLatLng(a) {
  const lat = _numCoord(a?.latitud ?? a?.latitude ?? a?.lat);
  const lng = _numCoord(a?.longitud ?? a?.longitude ?? a?.lng ?? a?.lon);
  if (lat === null || lng === null) return null;
  return { lat, lng };
}

// Highly robust Point-in-Polygon algorithm with automatic coordinate order detection
function isPointInPolygon(lat, lng, polygonCoords) {
  let x = _numCoord(lng);
  let y = _numCoord(lat);
  if (x === null || y === null || !Array.isArray(polygonCoords) || polygonCoords.length < 3) return false;

  let pts = [];
  for (let i = 0; i < polygonCoords.length; i++) {
    let p = polygonCoords[i];
    if (!p) continue;
    let plat = null, plng = null;
    if (typeof p.lat === 'number' && typeof p.lng === 'number') {
      plat = p.lat;
      plng = p.lng;
    } else if (Array.isArray(p) && p.length >= 2) {
      if (Math.abs(p[0]) > Math.abs(p[1])) {
        plng = p[0];
        plat = p[1];
      } else {
        plat = p[0];
        plng = p[1];
      }
    }
    if (plat !== null && plng !== null) {
      pts.push({ lat: plat, lng: plng });
    }
  }

  let inside = false;
  for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
    let xi = pts[i].lng, yi = pts[i].lat;
    let xj = pts[j].lng, yj = pts[j].lat;
    
    let intersect = ((yi > y) !== (yj > y))
        && (x < (xj - xi) * (y - yi) / ((yj - yi) || 1e-12) + xi);
    if (intersect) inside = !inside;
  }
  return inside;
}

function puntoEnRingLngLat(lat, lng, ring) {
  return isPointInPolygon(lat, lng, ring);
}

function puntoEnGeoJSON(lat, lng, geom) {
  if (!geom) return false;
  if (geom.type === 'Feature') {
    return puntoEnGeoJSON(lat, lng, geom.geometry);
  }
  if (geom.type === 'FeatureCollection') {
    if (!Array.isArray(geom.features)) return false;
    for (const f of geom.features) {
      if (puntoEnGeoJSON(lat, lng, f)) return true;
    }
    return false;
  }
  if (geom.type === 'GeometryCollection') {
    if (!Array.isArray(geom.geometries)) return false;
    for (const g of geom.geometries) {
      if (puntoEnGeoJSON(lat, lng, g)) return true;
    }
    return false;
  }
  if (geom.type === 'Polygon') {
    const rings = geom.coordinates;
    if (!Array.isArray(rings) || !rings.length) return false;
    const insideOuter = isPointInPolygon(lat, lng, rings[0]);
    if (!insideOuter) return false;
    for (let i = 1; i < rings.length; i++) {
      if (isPointInPolygon(lat, lng, rings[i])) return false;
    }
    return true;
  }
  if (geom.type === 'MultiPolygon') {
    const polys = geom.coordinates;
    if (!Array.isArray(polys)) return false;
    for (const poly of polys) {
      if (!Array.isArray(poly) || !poly.length) continue;
      const insideOuter = isPointInPolygon(lat, lng, poly[0]);
      if (insideOuter) {
        let insideHole = false;
        for (let i = 1; i < poly.length; i++) {
          if (isPointInPolygon(lat, lng, poly[i])) {
            insideHole = true;
            break;
          }
        }
        if (!insideHole) return true;
      }
    }
    return false;
  }
  return false;
}

function puntoEnPoligono(lat, lng, feature) {
  return puntoEnGeoJSON(lat, lng, feature);
}

function puntoEnArea(lat, lng, area) {
  lat = _numCoord(lat);
  lng = _numCoord(lng);
  if (lat === null || lng === null) return false;
  if (!area) return true;

  if (area.tipo === 'circulo') {
    const alat = _numCoord(area.lat), alng = _numCoord(area.lng), radio = _numCoord(area.radio);
    if (alat === null || alng === null || radio === null) return true;
    const d = Math.sqrt(
      Math.pow((lat - alat) * 111, 2) +
      Math.pow((lng - alng) * 111 * Math.cos(alat * Math.PI / 180), 2)
    );
    return d <= radio;
  }

  if (area.tipo === 'poligono' && Array.isArray(area.coords)) {
    return isPointInPolygon(lat, lng, area.coords);
  }

  return true;
}

async function obtenerGeomAreaEstudio() {
  if (areaEstudioGeom) return areaEstudioGeom;
  if (areaEstudioCargando) {
    await new Promise(r => { const t = setInterval(()=>{ if(!areaEstudioCargando){clearInterval(t);r();} },100); });
    return areaEstudioGeom;
  }
  areaEstudioCargando = true;
  try {
    const resp = await fetch(URLS.estudio);
    const gj   = await resp.json();
    let geom = null;
    if (gj.type === 'FeatureCollection' && gj.features && gj.features.length > 0) {
      geom = gj.features[0].geometry;
    } else if (gj.type === 'Feature') {
      geom = gj.geometry;
    } else if (gj.type === 'Polygon' || gj.type === 'MultiPolygon' || gj.type === 'GeometryCollection') {
      geom = gj;
    } else if (gj.geometry) {
      geom = gj.geometry;
    }
    areaEstudioGeom = geom || { type:'Polygon', coordinates:[[[-72.80,4.85],[-72.20,4.85],[-72.20,5.35],[-72.80,5.35],[-72.80,4.85]]] };
  } catch(e) {
    areaEstudioGeom = { type:'Polygon', coordinates:[[[-72.85,4.85],[-72.20,4.85],[-72.20,5.35],[-72.85,5.35],[-72.85,4.85]]] };
  }
  areaEstudioCargando = false;
  return areaEstudioGeom;
}

// ── SVG hatch ─────────────────────────────────────────────────────────────────
const svgNS = 'http://www.w3.org/2000/svg';
const pat = document.createElementNS(svgNS,'pattern');
pat.setAttribute('id','hatch'); pat.setAttribute('patternUnits','userSpaceOnUse');
pat.setAttribute('width','8'); pat.setAttribute('height','8'); pat.setAttribute('patternTransform','rotate(45)');
const ln = document.createElementNS(svgNS,'line');
ln.setAttribute('x1','0'); ln.setAttribute('y1','0'); ln.setAttribute('x2','0'); ln.setAttribute('y2','8');
ln.setAttribute('stroke','#4caf7d'); ln.setAttribute('stroke-width','2');
pat.appendChild(ln);
map.on('layeradd',()=>{
  const svg=document.querySelector('#map svg');
  if(svg&&!svg.querySelector('#hatch')){let d=svg.querySelector('defs');if(!d){d=document.createElementNS(svgNS,'defs');svg.insertBefore(d,svg.firstChild);}d.appendChild(pat);}
});

// ── UI: Cambiar tipo de alerta ────────────────────────────────────────────────
function cambiarTipo(tipo) {
  tipoActual = tipo;
  const bi=document.getElementById('btn-tipo-incendios'), bg=document.getElementById('btn-tipo-gfw');
  bi.classList.toggle('activo', tipo==='incendios');
  bg.classList.toggle('activo', tipo==='gfw');
  const pi=document.getElementById('panel-incendios'), pg=document.getElementById('panel-gfw');
  if(tipo==='incendios'){ pi.classList.remove('hidden'); pg.classList.add('hidden'); }
  else { pg.classList.remove('hidden'); pi.classList.add('hidden'); }
  if(tipo==='incendios') cambiarTab('incendios', document.getElementById('tab-incendios-btn'));
  else cambiarTab('deforestacion', document.getElementById('tab-defor-btn'));
  const gd=document.getElementById('gfw-dl-btns');
  if(gd){ if(tipo==='gfw')gd.classList.remove('hidden'); else gd.classList.add('hidden'); }
}

function toggleCapasExtra(btn) {
  const body = document.getElementById('capas-extra-body');
  const ico  = document.getElementById('ico-capas');
  const open = body.classList.contains('hidden');
  if(open){ body.classList.remove('hidden'); ico.textContent='▾'; }
  else { body.classList.add('hidden'); ico.textContent='▸'; }
}

// ── UI: Área de análisis ──────────────────────────────────────────────────────
let _cambiandoArea = false;

function cambiarAreaAnalisis(origenId) {
  if (_cambiandoArea) return;
  _cambiandoArea = true;

  const chkE = document.getElementById('chk-area-estudio');
  const chkN = document.getElementById('chk-area-nucleos');
  const chkM = document.getElementById('chk-area-municipio');
  const chkD = document.getElementById('chk-area-dibujo');

  const previo = areaAnalisisActiva;
  const activos = [chkE, chkN, chkM, chkD].filter(c => c.checked);

  if (activos.length > 1) {
    [chkE, chkN, chkM, chkD].forEach(c => {
      if (c.checked && (
        (c.id === 'chk-area-estudio'   && previo === 'estudio') ||
        (c.id === 'chk-area-nucleos'   && previo === 'nucleos') ||
        (c.id === 'chk-area-municipio' && previo === 'municipio') ||
        (c.id === 'chk-area-dibujo'    && previo === 'dibujo')
      )) {
        c.checked = false;
      }
    });
  }

  const eE = chkE.checked, eN = chkN.checked, eM = chkM.checked, eD = chkD.checked;
  let nueva = null;
  if (eE)      nueva = 'estudio';
  else if (eN) nueva = 'nucleos';
  else if (eM) nueva = 'municipio';
  else if (eD) nueva = 'dibujo';

  if (!nueva) {
    chkE.checked = true;
    nueva = 'estudio';
  }

  areaAnalisisActiva = nueva;

  const wrapM = document.getElementById('sel-municipio-wrap');
  if (wrapM) { if (chkM.checked) wrapM.classList.remove('hidden'); else wrapM.classList.add('hidden'); }

  const subD = document.getElementById('sub-dibujo');
  if (subD)  { if (chkD.checked) subD.classList.remove('hidden'); else subD.classList.add('hidden'); }

  if (!chkM.checked) { limpiarCapaMunicipio(); municipioActual = ''; }
  
  if (!chkD.checked && dibujoPoligonoLayer) {
    if (map.hasLayer(dibujoPoligonoLayer)) map.removeLayer(dibujoPoligonoLayer);
    if (dibujoHandler) { dibujoHandler.disable(); dibujoHandler = null; }
    document.getElementById('dibujo-hint').style.display = 'none';
  }
  if (chkD.checked && dibujoPoligonoLayer && !map.hasLayer(dibujoPoligonoLayer)) {
    map.addLayer(dibujoPoligonoLayer);
  }

  if (chkM.checked && !municipiosGJ) cargarMunicipios();

  document.getElementById('opt-estudio').classList.toggle('activo', chkE.checked);
  document.getElementById('opt-nucleos').classList.toggle('activo', chkN.checked);
  document.getElementById('opt-municipio').classList.toggle('activo', chkM.checked);
  document.getElementById('opt-dibujo').classList.toggle('activo', chkD.checked);

  const txt = document.getElementById('area-activa-txt');
  const mLabel = municipioActual ? `Municipio: ${municipioActual}` : 'Todos los municipios';
  let badge = '';
  if (nueva === 'estudio')   badge = 'Área de estudio activa';
  else if (nueva === 'nucleos')   badge = 'Núcleos boscosos activos';
  else if (nueva === 'municipio') badge = mLabel;
  else if (nueva === 'dibujo')    badge = dibujoPoligonoLayer ? `Polígono dibujado · ${dibujoArea_ha.toFixed(1)} ha` : 'Polígono pendiente';
  if (txt) txt.textContent = badge;

  if (chkE.checked) toggleCapa('estudio', true); else capasInfo.estudio.clearLayers();
  if (chkN.checked) toggleCapa('bosque', true);  else capasInfo.bosque.clearLayers();
  capaTodosGeo.clearLayers();

  _cambiandoArea = false;

  aplicarFiltros();
  Object.keys(gfwCapas).forEach(k => {
    if (gfwCapas[k].visible && k!=='hansen') cargarGFWAlertas(k);
  });
}

// Obtener geometría según el área activa
async function obtenerGeomActiva() {
  if (!areaAnalisisActiva) return await obtenerGeomAreaEstudio();
  if (areaAnalisisActiva==='estudio') return await obtenerGeomAreaEstudio();
  if (areaAnalisisActiva==='nucleos') {
    if (!geomNucleos) {
      try {
        const gj = await (await fetch(URLS.bosque)).json();
        geomNucleos = gj.features[0].geometry;
      } catch(e) { return await obtenerGeomAreaEstudio(); }
    }
    return geomNucleos;
  }
  if (areaAnalisisActiva==='municipio') {
    if (!municipioActual) return await obtenerGeomAreaEstudio();
    try {
      if (!veredasGJ) await cargarVeredas();
      const vMun = veredasGJ.features.filter(
        f => (f.properties.NOMB_MPIO||'').toUpperCase() === municipioActual.toUpperCase()
      );
      if (vMun.length === 1) return vMun[0].geometry;
      if (vMun.length > 1) {
        return {
          type: 'GeometryCollection',
          geometries: vMun.map(f => f.geometry)
        };
      }
    } catch(e) { console.warn('geom municipio:', e); }
    return await obtenerGeomAreaEstudio();
  }
  if (areaAnalisisActiva==='dibujo') {
    if (dibujoPoligonoCoords && dibujoPoligonoCoords.length >= 3) {
      const ring = [];
      for (const p of dibujoPoligonoCoords) {
        let plat = null, plng = null;
        if (Array.isArray(p)) {
          if (Math.abs(p[0]) > Math.abs(p[1])) {
            plng = p[0]; plat = p[1];
          } else {
            plat = p[0]; plng = p[1];
          }
        }
        if (plat !== null && plng !== null) {
          ring.push([plng, plat]);
        }
      }
      if (ring.length >= 3) {
        const first = ring[0], last = ring[ring.length-1];
        if (first[0] !== last[0] || first[1] !== last[1]) {
          ring.push([first[0], first[1]]);
        }
        return { type: 'Polygon', coordinates: [ring] };
      }
    }
    return await obtenerGeomAreaEstudio();
  }
  return await obtenerGeomAreaEstudio();
}

function bboxAreaActiva() {
  return CHAMEZA_BBOX;
}

// ── DIBUJO POLÍGONO PERSONALIZADO ─────────────────────────────────────────────
function iniciarDibujoPoligono() {
  if (dibujoPoligonoLayer) {
    if (map.hasLayer(dibujoPoligonoLayer)) map.removeLayer(dibujoPoligonoLayer);
    dibujoPoligonoLayer = null;
    dibujoPoligonoCoords = null;
    dibujoArea_ha = 0;
  }
  if (dibujoHandler) { dibujoHandler.disable(); dibujoHandler = null; }

  dibujoHandler = new L.Draw.Polygon(map, {
    shapeOptions: { color:'#7c3aed', weight:2.5, fillColor:'#7c3aed', fillOpacity:0.15, dashArray:'4 4' },
    allowIntersection: false,
    showArea: false
  });
  dibujoHandler.enable();

  document.getElementById('dibujo-hint').style.display = 'flex';
  const btn = document.getElementById('btn-iniciar-dibujo');
  if (btn) btn.textContent = '⏸ Dibujando… (doble-clic para cerrar)';
  document.getElementById('dibujo-info').textContent = 'Haz clic en el mapa para añadir vértices';
}

function limpiarDibujoPoligono() {
  if (dibujoPoligonoLayer && map.hasLayer(dibujoPoligonoLayer)) {
    map.removeLayer(dibujoPoligonoLayer);
  }
  dibujoPoligonoLayer = null;
  dibujoPoligonoCoords = null;
  dibujoArea_ha = 0;
  document.getElementById('dibujo-info').textContent = 'Sin polígono dibujado';
  document.getElementById('btn-limpiar-dibujo').style.display = 'none';
  document.getElementById('btn-iniciar-dibujo').textContent = '✏️ Empezar a dIbujar';
  
  aplicarFiltros();
  Object.keys(gfwCapas).forEach(k => {
    if (gfwCapas[k].visible && k!=='hansen') cargarGFWAlertas(k);
  });
  const txt = document.getElementById('area-activa-txt');
  if (txt && areaAnalisisActiva === 'dibujo') txt.textContent = 'Polígono pendiente';
}

function calcularAreaPoligonoHa(coords) {
  if (!coords || coords.length < 3) return 0;
  const R = 6378137;
  let area = 0;
  const n = coords.length;
  for (let i = 0; i < n; i++) {
    const [lat1, lng1] = coords[i];
    const [lat2, lng2] = coords[(i+1) % n];
    area += (lng2 - lng1) * Math.PI/180 * (2 + Math.sin(lat1*Math.PI/180) + Math.sin(lat2*Math.PI/180));
  }
  area = Math.abs(area * R * R / 2);
  return area / 10000;
}

map.on(L.Draw.Event.CREATED, function(e) {
  if (!e || e.layerType !== 'polygon') return;
  if (areaAnalisisActiva !== 'dibujo') return;

  const layer = e.layer;
  let latlngs = layer.getLatLngs();
  
  // Unify multidimensional arrays representing polygon bounds to flat LatLng array
  while (latlngs.length > 0 && Array.isArray(latlngs[0])) {
    latlngs = latlngs[0];
  }

  if (!latlngs || latlngs.length < 3) {
    alert('El polígono debe tener al menos 3 vértices.');
    return;
  }
  
  dibujoPoligonoCoords = latlngs.map(p => [p.lat, p.lng]);
  const first = dibujoPoligonoCoords[0], last = dibujoPoligonoCoords[dibujoPoligonoCoords.length-1];
  if (first[0] !== last[0] || first[1] !== last[1]) dibujoPoligonoCoords.push([first[0], first[1]]);
  dibujoArea_ha = calcularAreaPoligonoHa(dibujoPoligonoCoords);

  dibujoPoligonoLayer = L.polygon(latlngs, {
    color: '#7c3aed', weight: 2.5,
    fillColor: '#7c3aed', fillOpacity: 0.12,
    dashArray: '4 4'
  }).bindTooltip(`Área personalizada · ${dibujoArea_ha.toFixed(1)} ha`, { sticky: true });
  map.addLayer(dibujoPoligonoLayer);

  if (dibujoHandler) { dibujoHandler.disable(); dibujoHandler = null; }
  document.getElementById('dibujo-hint').style.display = 'none';

  document.getElementById('btn-iniciar-dibujo').textContent = '✏️ Empezar a dibujar';
  document.getElementById('btn-limpiar-dibujo').style.display = 'block';

  obtenerGeomAreaEstudio().then(geomEst => {
    let advertencia = '';
    if (geomEst) {
      const dentro = dibujoPoligonoCoords.every(([lat,lng]) => puntoEnGeoJSON(lat, lng, geomEst));
      if (!dentro) advertencia = ' (parte fuera del Área de Estudio · resultados se filtrarán a la intersección)';
    }
    document.getElementById('dibujo-info').innerHTML =
      `✓ Polígono · <span class="mono text-violet-600">${dibujoArea_ha.toFixed(1)} ha</span>${advertencia}`;
    const txt = document.getElementById('area-activa-txt');
    if (txt) txt.textContent = `Polígono dibujado · ${dibujoArea_ha.toFixed(1)} ha`;
  });

  aplicarFiltros();
  Object.keys(gfwCapas).forEach(k => {
    if (gfwCapas[k].visible && k!=='hansen') cargarGFWAlertas(k);
  });
});

function pasaFiltroAreaDibujo(lat, lng) {
  if (areaAnalisisActiva !== 'dibujo') return true;
  if (!dibujoPoligonoCoords || dibujoPoligonoCoords.length < 3) return true;

  lat = _numCoord(lat);
  lng = _numCoord(lng);
  if (lat === null || lng === null) return false;

  const insideDrawn = isPointInPolygon(lat, lng, dibujoPoligonoCoords);
  if (!insideDrawn) return false;

  if (RECORTAR_DIBUJO_A_AREA_ESTUDIO && areaEstudioGeom && typeof areaEstudioGeom === 'object') {
    const dentroEstudio = puntoEnGeoJSON(lat, lng, areaEstudioGeom);
    if (!dentroEstudio) return false;
  }

  return true;
}

function alertasEnPeriodo(arr) {
  if (modoPeriodo === 'rango' && fechaInicio && fechaFin) {
    return arr.filter(a => {
      const f = new Date(a.fecha_deteccion);
      return f >= fechaInicio && f <= fechaFin;
    });
  }
  const ini=new Date(Date.now()-diasActual*864e5);
  return arr.filter(a=>new Date(a.fecha_deteccion)>=ini);
}

function aplicarFiltros() {
  let firms = alertasEnPeriodo(todasAlertas);
  if (filtroActual !== 'all') {
    firms = firms.filter(a => (a.firms_confidence || '').toLowerCase() === filtroActual);
  }

  const aplicaFiltroDibujo = (areaAnalisisActiva === 'dibujo' && dibujoPoligonoCoords && dibujoPoligonoCoords.length >= 3);
  const firmsArea = aplicaFiltroDibujo
    ? firms.filter(a => {
        const p = _alertaLatLng(a);
        return p ? pasaFiltroAreaDibujo(p.lat, p.lng) : false;
      })
    : firms;

  const deforPeriodo = alertasEnPeriodo(alertasDefor);
  const deforArea = aplicaFiltroDibujo
    ? deforPeriodo.filter(a => {
        const p = _alertaLatLng(a);
        return p ? pasaFiltroAreaDibujo(p.lat, p.lng) : false;
      })
    : deforPeriodo;

  actualizarStats(firms, firmsArea, deforArea);
  renderMarcadores(firmsArea);
  if (tabActual === 'incendios') renderLista(firmsArea, 'incendios');
  else renderLista(deforArea, 'deforestacion');
  
  setTimeout(filtrarDeforestacion, 0);
  Object.keys(gfwCapas).forEach(k => {
    if (gfwCapas[k].visible && k !== 'hansen') filtrarGFW(k);
  });
}

function actualizarStats(firms,firmsArea,deforArea) {
  const aplicaFiltroDibujo = (areaAnalisisActiva === 'dibujo' && dibujoPoligonoCoords);
  document.getElementById('total-alertas').textContent = aplicaFiltroDibujo ? firmsArea.length : firms.length;
  document.getElementById('total-defor').textContent   = deforArea.length;
  document.getElementById('total-high').textContent    = firms.filter(a=>(a.firms_confidence||'').toLowerCase()==='high').length;
  const frps=firms.map(a=>a.firms_frp||0).filter(Boolean);
  document.getElementById('total-frp').textContent = frps.length?Math.max(...frps).toFixed(0):'0';
}

function colorConfianza(c) {
  if(!c)return'#f5a623';const cl=c.toLowerCase();
  if(cl==='high')return'#e8480a';if(cl==='nominal')return'#f5a623';return'#4caf7d';
}

function renderMarcadores(alertas) {
  marcadores.forEach(m=>map.removeLayer(m)); marcadores=[];
  alertas.forEach(a=>{
    const color=colorConfianza(a.firms_confidence);
    const m=L.circleMarker([a.latitud,a.longitud],{radius:Math.max(5,Math.min(16,(a.firms_frp||5)/8)),fillColor:color,color:'#fff',weight:1.5,opacity:.9,fillOpacity:.8}).addTo(map);
    const fecha=new Date(a.fecha_deteccion).toLocaleString('es-CO',{timeZone:'America/Bogota'});
    m.bindPopup(`<div class="popup-title">Incendio activo</div><div class="popup-row">Coords <span>${a.latitud.toFixed(4)}, ${a.longitud.toFixed(4)}</span></div><div class="popup-row">Confianza <span>${a.firms_confidence||'N/A'}</span></div><div class="popup-row">FRP <span>${a.firms_frp?a.firms_frp.toFixed(1)+' MW':'N/A'}</span></div><div class="popup-row">Satélite <span>${a.firms_satellite||'VIIRS'}</span></div><div class="popup-row">Detectado <span>${fecha}</span></div>`);
    marcadores.push(m);
  });
}

function cambiarTab(tab,btn) {
  tabActual=tab;
  ['tab-incendios-btn','tab-defor-btn'].forEach(id=>{
    const el=document.getElementById(id);
    if(el){ el.classList.remove('tab-act'); el.classList.add('tab-inact'); }
  });
  if(btn){ btn.classList.remove('tab-inact'); btn.classList.add('tab-act'); }
  aplicarFiltros();
}

function renderLista(alertas,tipo) {
  const el=document.getElementById('lista-alertas');
  if (!alertas.length){el.innerHTML='<div class="flex items-center justify-center py-10 text-slate-400 text-xs italic">Sin alertas en este período</div>';return;}
  if (tipo==='incendios') {
    el.innerHTML='<div class="divide-y divide-slate-50">'+alertas.map((a,i)=>{
      const conf=(a.firms_confidence||'low').toLowerCase();
      const fecha=new Date(a.fecha_deteccion).toLocaleString('es-CO',{month:'short',day:'numeric',hour:'2-digit',minute:'2-digit',timeZone:'America/Bogota'});
      const dot=conf==='high'?'bg-red-500 shadow-red-200':conf==='nominal'?'bg-amber-400':'bg-emerald-400';
      return `<div class="flex items-center gap-3 px-4 py-3 hover:bg-emerald-50/50 cursor-pointer transition-colors" onclick="centrarEn(${i})"><div class="w-2 h-2 rounded-full ${dot} shadow-sm shrink-0"></div><div class="flex-1 min-w-0"><div class="mono text-[10px] text-slate-700 font-bold">${a.latitud.toFixed(4)}, ${a.longitud.toFixed(4)}</div><div class="text-[9px] text-slate-400 mt-0.5">${fecha} · ${a.firms_satellite||'VIIRS'}</div></div><div class="mono text-[10px] text-red-500 font-bold shrink-0">${a.firms_frp?a.firms_frp.toFixed(0)+' MW':'—'}</div></div>`;
    }).join('')+'</div>';
  } else {
    el.innerHTML='<div class="divide-y divide-slate-50">'+alertas.map((a,i)=>{
      const fecha=new Date(a.fecha_deteccion+'T12:00:00').toLocaleDateString('es-CO');
      const dot=a.severidad==='severa'?'bg-red-500':a.severidad==='moderada'?'bg-orange-400':'bg-yellow-400';
      return `<div class="flex items-center gap-3 px-4 py-3 hover:bg-emerald-50/50 cursor-pointer transition-colors" onclick="centrarEnDefor(${i})"><div class="w-2 h-2 rounded-sm ${dot} shrink-0"></div><div class="flex-1 min-w-0"><div class="mono text-[10px] text-slate-700 font-bold">${a.latitud.toFixed(4)}, ${a.longitud.toFixed(4)}</div><div class="text-[9px] text-slate-400 mt-0.5">${fecha} · ${a.predio_id}</div></div><div class="mono text-[10px] text-amber-500 font-bold shrink-0">${a.area_afectada_ha||'?'} ha</div></div>`;
    }).join('')+'</div>';
  }
}

function centrarEn(i)      { if(marcadores[i]){map.setView(marcadores[i].getLatLng(),13);marcadores[i].openPopup();} }
function centrarEnDefor(i) {
  const deforFiltrada=alertasDefor;
  const a=deforFiltrada[i];
  if(a){map.setView([a.latitud,a.longitud],13);}
}

function cambiarPeriodo(btn) {
  document.querySelectorAll('.btn-per').forEach(b=>b.classList.remove('activo'));
  btn.classList.add('activo'); diasActual=parseInt(btn.dataset.dias);
  modoPeriodo = 'preset';
  const blq = document.getElementById('modo-rango');
  if (blq) { blq.classList.add('rango-inactivo'); blq.classList.remove('rango-activo'); }
  cargarAlertas();
  if(deforVisible)cargarDeforestacion();else aplicarFiltros();
  Object.keys(gfwCapas).forEach(k=>{ if(gfwCapas[k].visible && k!=='hansen') cargarGFWAlertas(k); });
}

// ── Rango personalizado ───────────────────────────────────────────────────────
function cambiarRangoFechas() {
  const inpI = document.getElementById('fecha-inicio');
  const inpF = document.getElementById('fecha-fin');
  if (!inpI || !inpF) return;

  const sI = inpI.value, sF = inpF.value;
  if (!sI || !sF) return;

  if (sI < FECHA_MIN_DATA) { inpI.value = FECHA_MIN_DATA; return cambiarRangoFechas(); }
  if (sI > sF) { inpF.value = sI; return cambiarRangoFechas(); }

  fechaInicio = new Date(sI + 'T00:00:00');
  fechaFin    = new Date(sF + 'T23:59:59');

  const ahora = new Date();
  const diffMs = ahora - fechaInicio;
  diasActual = Math.max(1, Math.ceil(diffMs / 864e5));

  modoPeriodo = 'rango';
  document.querySelectorAll('.btn-per').forEach(b=>b.classList.remove('activo'));
  const blq = document.getElementById('modo-rango');
  if (blq) { blq.classList.add('rango-activo'); blq.classList.remove('rango-inactivo'); }

  const dias = Math.ceil((fechaFin - fechaInicio) / 864e5);
  const info = document.getElementById('rango-info');
  if (info) info.textContent = `${dias} día${dias!==1?'s':''} seleccionado${dias!==1?'s':''}`;

  cargarAlertas();
  if(deforVisible) cargarDeforestacion(); else aplicarFiltros();
  Object.keys(gfwCapas).forEach(k=>{ if(gfwCapas[k].visible && k!=='hansen') cargarGFWAlertas(k); });
}

function inicializarRangoFechas() {
  const inpI = document.getElementById('fecha-inicio');
  const inpF = document.getElementById('fecha-fin');
  if (!inpI || !inpF) return;

  const hoy = new Date();
  const hoyStr = hoy.toISOString().slice(0,10);
  let inicio = new Date(Date.now() - 30*864e5);
  let inicioStr = inicio.toISOString().slice(0,10);
  if (inicioStr < FECHA_MIN_DATA) inicioStr = FECHA_MIN_DATA;

  inpI.max = hoyStr;
  inpF.max = hoyStr;
  inpI.value = inicioStr;
  inpF.value = hoyStr;

  cambiarRangoFechas();
}

function filtrar(btn) {
  document.querySelectorAll('.filtro-btn').forEach(b=>{
    b.classList.remove('activo','bg-red-500','text-white','border-red-400');
    b.classList.add('bg-white','text-slate-600','border-slate-200');
  });
  btn.classList.add('activo','bg-red-500','text-white','border-red-400');
  btn.classList.remove('bg-white','text-slate-600','border-slate-200');
  filtroActual=btn.dataset.filtro; aplicarFiltros();
}

// ── Descarga unificada ────────────────────────────────────────────────────────
async function descargarActual(formato) {
  if (tipoActual==='incendios') {
    if (formato==='csv') descargarCSV('incendios');
    else descargarPDF('incendios');
  } else {
    const activasGFW = Object.keys(gfwCapas).filter(k=>gfwCapas[k].visible && k!=='hansen' && gfwCapas[k].datos.length);
    if (activasGFW.length) {
      activasGFW.forEach(k => descargarGFW(k));
    } else {
      if (formato==='csv') descargarCSV('deforestacion');
      else descargarPDF('deforestacion');
    }
  }
}

function descargarGFWActivo() {
  const activas = Object.keys(gfwCapas).filter(k=>gfwCapas[k].visible && k!=='hansen' && gfwCapas[k].datos.length);
  if (!activas.length) { alert('Activa primero una capa GFW (GLAD o RADD) y espera que cargue.'); return; }
  activas.forEach(k => descargarGFW(k));
}

// ── Descarga CSV ──────────────────────────────────────────────────────────────
function alertasActualesIncendios() {
  let r = alertasEnPeriodo(todasAlertas);
  if (filtroActual !== 'all') r = r.filter(a => (a.firms_confidence||'').toLowerCase() === filtroActual);
  if (areaAnalisisActiva === 'dibujo' && dibujoPoligonoCoords && dibujoPoligonoCoords.length >= 3) {
    r = r.filter(a => {
      const p = _alertaLatLng(a);
      return p ? pasaFiltroAreaDibujo(p.lat, p.lng) : false;
    });
  }
  return r;
}

function alertasActualesDefor() {
  let r = alertasEnPeriodo(alertasDefor);
  if (areaAnalisisActiva === 'dibujo' && dibujoPoligonoCoords && dibujoPoligonoCoords.length >= 3) {
    r = r.filter(a => {
      const p = _alertaLatLng(a);
      return p ? pasaFiltroAreaDibujo(p.lat, p.lng) : false;
    });
  }
  return r;
}

async function agruparAlertasPorVereda(alertas) {
  if (!municipioActual) return null;
  await cargarVeredas();
  if (!veredasGJ) return null;

  const veredasMunicipio = veredasGJ.features.filter(
    f => (f.properties.NOMB_MPIO||'').toUpperCase() === municipioActual.toUpperCase()
  );
  if (!veredasMunicipio.length) return null;

  const grupos = {};
  for (const alerta of alertas) {
    for (const vFeat of veredasMunicipio) {
      if (puntoEnPoligono(alerta.latitud, alerta.longitud, vFeat)) {
        const nom = vFeat.properties.NOMBRE_VER || 'Sin nombre';
        if (!grupos[nom]) grupos[nom] = [];
        grupos[nom].push(alerta);
        break;
      }
    }
  }
  return Object.keys(grupos).length ? grupos : null;
}

async function descargarCSV(modo) {
  const hoy = new Date().toISOString().slice(0,10);
  const esc = v => {
    const s = String(v ?? '');
    return s.includes(',') || s.includes('"') || s.includes('\n')
      ? '"' + s.replace(/"/g, '""') + '"'
      : s;
  };

  function veredaDeAlerta(lat, lng, veredasMpio) {
    for (const vf of veredasMpio) {
      if (puntoEnPoligono(lat, lng, vf))
        return vf.properties.NOMBRE_VER || 'Sin nombre';
    }
    return 'Sin vereda';
  }

  const fi = alertasActualesIncendios();
  const fd = alertasActualesDefor();
  if (!veredasGJ) await cargarVeredas();

  if (modo === 'incendios') {
    if (!fi.length) { alert('No hay alertas de incendio para exportar.'); return; }

    if (municipioActual && veredasGJ) {
      const veredasMpio = veredasGJ.features.filter(
        f => (f.properties.NOMB_MPIO||'').toUpperCase() === municipioActual.toUpperCase()
      );
      const cab = ['vereda','municipio','fecha_deteccion','latitud','longitud',
                   'confianza','frp_mw','satelite','estado'].join(',');
      const rows = fi.map(a => [
        esc(veredaDeAlerta(a.latitud, a.longitud, veredasMpio)),
        esc(municipioActual),
        esc(a.fecha_deteccion), a.latitud, a.longitud,
        esc(a.firms_confidence||''), a.firms_frp||'',
        esc(a.firms_satellite||''), esc(a.estado)
      ].join(','));
      _dl([cab, ...rows].join('\n'),
          `incendios_${municipioActual.replace(/ /g,'_')}_${hoy}.csv`);

    } else {
      if (!municipiosGJ) await cargarMunicipios();

      function mpioDeAlerta(lat, lng) {
        if (!veredasGJ) return 'Sin datos';
        const vf = veredasGJ.features.find(f => puntoEnPoligono(lat, lng, f));
        return vf ? (vf.properties.NOMB_MPIO || 'Sin nombre') : 'Fuera de límites';
      }

      const cabDet = ['municipio','fecha_deteccion','latitud','longitud',
                      'confianza','frp_mw','satelite','estado'].join(',');
      const rowsDet = fi.map(a => [
        esc(mpioDeAlerta(a.latitud, a.longitud)),
        esc(a.fecha_deteccion), a.latitud, a.longitud,
        esc(a.firms_confidence||''), a.firms_frp||'',
        esc(a.firms_satellite||''), esc(a.estado)
      ].join(','));

      const conteos = {};
      fi.forEach(a => {
        const m = mpioDeAlerta(a.latitud, a.longitud);
        conteos[m] = (conteos[m] || 0) + 1;
      });
      const cabRes = ['municipio','n_incendios'].join(',');
      const rowsRes = Object.entries(conteos)
        .sort((a,b) => b[1]-a[1])
        .map(([m,n]) => [esc(m), n].join(','));

      const lines = [
        '## DETALLE POR ALERTA',
        cabDet, ...rowsDet,
        '',
        '## RESUMEN POR MUNICIPIO',
        cabRes, ...rowsRes
      ];
      _dl(lines.join('\n'), `incendios_area_estudio_${hoy}.csv`);
    }

  } else if (modo === 'deforestacion') {
    if (!fd.length) { alert('No hay alertas de deforestación para exportar.'); return; }

    if (municipioActual && veredasGJ) {
      const veredasMpio = veredasGJ.features.filter(
        f => (f.properties.NOMB_MPIO||'').toUpperCase() === municipioActual.toUpperCase()
      );
      const cab = ['vereda','municipio','fecha_deteccion','predio_id',
                   'latitud','longitud','severidad','area_ha','cambio_ndvi','estado'].join(',');
      const rows = fd.map(a => [
        esc(veredaDeAlerta(a.latitud, a.longitud, veredasMpio)),
        esc(municipioActual),
        esc(a.fecha_deteccion), esc(a.predio_id||''),
        a.latitud, a.longitud,
        esc(a.severidad||''), a.area_afectada_ha||'',
        a.cambio_ndvi!=null ? Number(a.cambio_ndvi).toFixed(3) : '',
        esc(a.estado)
      ].join(','));
      _dl([cab, ...rows].join('\n'),
          `deforestacion_${municipioActual.replace(/ /g,'_')}_${hoy}.csv`);

    } else {
      if (!municipiosGJ) await cargarMunicipios();

      function mpioDeAlertaDef(lat, lng) {
        if (!veredasGJ) return 'Sin datos';
        const vf = veredasGJ.features.find(f => puntoEnPoligono(lat, lng, f));
        return vf ? (vf.properties.NOMB_MPIO || 'Sin nombre') : 'Fuera de límites';
      }

      const cabDet = ['municipio','fecha_deteccion','predio_id','latitud','longitud',
                      'severidad','area_ha','cambio_ndvi','estado'].join(',');
      const rowsDet = fd.map(a => [
        esc(mpioDeAlertaDef(a.latitud, a.longitud)),
        esc(a.fecha_deteccion), esc(a.predio_id||''),
        a.latitud, a.longitud,
        esc(a.severidad||''), a.area_afectada_ha||'',
        a.cambio_ndvi!=null ? Number(a.cambio_ndvi).toFixed(3) : '',
        esc(a.estado)
      ].join(','));

      const conteosD = {};
      fd.forEach(a => {
        const m = mpioDeAlertaDef(a.latitud, a.longitud);
        conteosD[m] = (conteosD[m] || 0) + 1;
      });
      const cabRes = ['municipio','n_deforestacion'].join(',');
      const rowsRes = Object.entries(conteosD)
        .sort((a,b) => b[1]-a[1])
        .map(([m,n]) => [esc(m), n].join(','));

      const lines = [
        '## DETALLE POR ALERTA',
        cabDet, ...rowsDet,
        '',
        '## RESUMEN POR MUNICIPIO',
        cabRes, ...rowsRes
      ];
      _dl(lines.join('\n'), `deforestacion_area_estudio_${hoy}.csv`);
    }

  } else if (modo === 'combinado') {
    if (!fi.length && !fd.length) { alert('No hay datos para exportar.'); return; }
    if (fi.length) { descargarCSV('incendios'); await new Promise(r => setTimeout(r, 400)); }
    if (fd.length) { descargarCSV('deforestacion'); }
  }
}

function _dl(content,filename){
  const blob=new Blob([content],{type:'text/csv;charset=utf-8;'});
  const url=URL.createObjectURL(blob);
  const a=document.createElement('a');
  a.href=url; a.download=filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

// ── Descarga PDF ──────────────────────────────────────────────────────────────
function descargarPDF(modo) {
  const {jsPDF}=window.jspdf;
  const doc=new jsPDF();
  const hoy=new Date().toLocaleDateString('es-CO',{timeZone:'America/Bogota'});
  const periodo=diasActual===1?'Últimas 24 h':diasActual===7?'Últimos 7 días':'Últimos 30 días';
  const areaLbl=areaAnalisisActiva==='estudio'?'Área de estudio':areaAnalisisActiva==='nucleos'?'Núcleos boscosos':areaAnalisisActiva==='municipio'?(municipioActual?`Municipio: ${municipioActual}`:'Todos los municipios'):areaAnalisisActiva==='dibujo'?`Polígono personalizado (${dibujoArea_ha.toFixed(1)} ha)`:'Área completa';
  doc.setFillColor(26,74,46);doc.rect(0,0,210,36,'F');
  doc.setTextColor(76,175,125);doc.setFontSize(15);doc.setFont('helvetica','bold');
  const titulo=modo==='incendios'?'Reporte de Incendios':modo==='deforestacion'?'Reporte de Deforestación':'Reporte Combinado · Incendios & Deforestación';
  doc.text(titulo,14,14);
  doc.setFontSize(9);doc.setFont('helvetica','normal');doc.setTextColor(122,171,138);
  doc.text(`Chámeza, Casanare · ${hoy} · ${periodo} · ${areaLbl}`,14,24);
  doc.setTextColor(180,220,195);
  doc.text('Fuente: NASA FIRMS + GEE · CBC Cunaguaro / TFCA Colombia',14,32);
  let y=44;
  if(modo!=='deforestacion'){
    const fi=alertasActualesIncendios();
    doc.setFillColor(30,53,40);doc.rect(0,y,210,9,'F');
    doc.setTextColor(255,112,67);doc.setFontSize(11);doc.setFont('helvetica','bold');
    doc.text(`INCENDIOS (${fi.length})`,14,y+6);y+=13;
    if(!fi.length){doc.setFont('helvetica','italic');doc.setTextColor(122,171,138);doc.setFontSize(9);doc.text('Sin alertas en este período.',14,y);y+=10;}
    else{
      const conMunicipio = municipioActual && veredasGJ;
      doc.setFontSize(8);doc.setFont('helvetica','bold');doc.setFillColor(220,240,228);
      doc.rect(14,y,182,7,'F');doc.setTextColor(40,40,40);
      if(conMunicipio){
        doc.text('Vereda',16,y+5);doc.text('Fecha',56,y+5);doc.text('Lat',96,y+5);doc.text('Lon',114,y+5);doc.text('Conf.',134,y+5);doc.text('FRP',158,y+5);
      } else {
        doc.text('Fecha',16,y+5);doc.text('Lat',68,y+5);doc.text('Lon',92,y+5);
        doc.text('Confianza',116,y+5);doc.text('FRP (MW)',146,y+5);doc.text('Satélite',172,y+5);
      }
      y+=9;doc.setFont('helvetica','normal');
      fi.slice(0,40).forEach((a,i)=>{
        if(y>272){doc.addPage();y=20;}
        if(i%2===0){doc.setFillColor(248,252,249);doc.rect(14,y-4,182,7,'F');}
        const f=new Date(a.fecha_deteccion).toLocaleString('es-CO',{timeZone:'America/Bogota',month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'});
        doc.setTextColor(40,40,40);
        if(conMunicipio){
          let verNom='—';
          const vMun=veredasGJ.features.filter(v=>(v.properties.NOMB_MPIO||'').toUpperCase()===municipioActual.toUpperCase());
          for(const vf of vMun){if(puntoEnPoligono(a.latitud,a.longitud,vf)){verNom=(vf.properties.NOMBRE_VER||'').slice(0,14);break;}}
          doc.text(verNom,16,y);doc.text(f,56,y);doc.text(a.latitud.toFixed(3),96,y);doc.text(a.longitud.toFixed(3),114,y);
          doc.text(a.firms_confidence||'—',134,y);doc.text(a.firms_frp?a.firms_frp.toFixed(1):'—',158,y);
        } else {
          doc.text(f,16,y);doc.text(a.latitud.toFixed(4),68,y);doc.text(a.longitud.toFixed(4),92,y);
          doc.text(a.firms_confidence||'—',116,y);doc.text(a.firms_frp?a.firms_frp.toFixed(1):'—',146,y);
          doc.text(a.firms_satellite||'—',172,y);
        }
        y+=7;
      });
    }
    y+=6;
  }
  if(modo!=='incendios'){
    const fd=alertasActualesDefor();
    if(y>240){doc.addPage();y=20;}
    doc.setFillColor(30,53,40);doc.rect(0,y,210,9,'F');
    doc.setTextColor(234,179,8);doc.setFontSize(11);doc.setFont('helvetica','bold');
    doc.text(`DEFORESTACION (${fd.length})`,14,y+6);y+=13;
    if(!fd.length){doc.setFont('helvetica','italic');doc.setTextColor(122,171,138);doc.setFontSize(9);doc.text('Sin alertas en este período.',14,y);y+=10;}
    else{
      const sev={leve:0,moderada:0,severa:0};fd.forEach(a=>{if(sev[a.severidad]!==undefined)sev[a.severidad]++;});
      const haTotal=fd.reduce((s,a)=>s+(a.area_afectada_ha||0),0).toFixed(1);
      doc.setFontSize(9);doc.setFont('helvetica','normal');doc.setTextColor(40,40,40);
      doc.text(`Área total: ${haTotal} ha · Severa: ${sev.severa} · Moderada: ${sev.moderada} · Leve: ${sev.leve}`,14,y);y+=8;
      doc.setFontSize(8);doc.setFont('helvetica','bold');doc.setFillColor(220,240,228);
      doc.rect(14,y,182,7,'F');doc.setTextColor(40,40,40);
      doc.text('Fecha',16,y+5);doc.text('Predio',54,y+5);doc.text('Lat',110,y+5);
      doc.text('ΔNDVI',128,y+5);doc.text('Ha',148,y+5);doc.text('Severidad',172,y+5);
      y+=9;doc.setFont('helvetica','normal');
      fd.slice(0,40).forEach((a,i)=>{
        if(y>272){doc.addPage();y=20;}
        if(i%2===0){doc.setFillColor(248,252,249);doc.rect(14,y-4,182,7,'F');}
        const f=new Date(a.fecha_deteccion+'T12:00:00').toLocaleDateString('es-CO');
        const sc={'severa':[220,38,38],'moderada':[249,115,22],'leve':[234,179,8]}[a.severidad]||[40,40,40];
        doc.setTextColor(40,40,40);
        doc.text(f,16,y);doc.text((a.predio_id||'').slice(0,22),54,y);
        doc.text(a.latitud?.toFixed(4)||'—',110,y);
        doc.text(a.cambio_ndvi!=null?Number(a.cambio_ndvi).toFixed(3):'—',128,y);
        doc.text(a.area_afectada_ha!=null?String(a.area_afectada_ha):'—',148,y);
        doc.setTextColor(...sc);doc.text(a.severidad||'—',172,y);y+=7;
      });
    }
  }
  doc.setFontSize(7);doc.setTextColor(120,120,120);doc.setFont('helvetica','normal');
  doc.text('CBC Cunaguaro · TFCA Colombia · datos: NASA FIRMS + GEE Sentinel-2',14,290);
  doc.save(`reporte_${modo}_chameza_${new Date().toISOString().slice(0,10)}.pdf`);
}

// ── Capas informativas ────────────────────────────────────────────────────────
async function toggleCapa(key,visible) {
  const group=capasInfo[key];
  if(!visible){
    group.clearLayers();
    const sub=document.getElementById(`sub-${key}`);
    if(sub) sub.classList.remove('visible');
    return;
  }
  const sub=document.getElementById(`sub-${key}`);
  if(sub) sub.classList.add('visible');
  try {
    if(key==='conectividad'){
      if(!georasterConectividad){const r=await fetch(URLS.conectividad);georasterConectividad=await parseGeoraster(await r.arrayBuffer());}
      renderRasterConectividad();return;
    }
    const data=await(await fetch(URLS[key])).json();
    if(key==='perdidas'){datosPerdidas=data;actualizarPerdidas();return;}
    let style={};
    if(key==='municipio') style={color:'#333',weight:2,fillOpacity:0};
    if(key==='estudio')   style={color:'#2563eb',weight:2.5,fillOpacity:0.04,dashArray:'6 4'};
    if(key==='bosque')    style={color:'#2d7a4f',weight:1,fillOpacity:0.5};
    const layer=L.geoJSON(data,{style,onEachFeature:(f,l)=>{if(key==='bosque')l.on('add',()=>{if(l._path)l._path.setAttribute('fill','url(#hatch)');});}});
    group.addLayer(layer);
  } catch(e){console.error(`Error capa ${key}:`,e);}
}

function actualizarPerdidas(){
  if(!datosPerdidas)return;
  const group=capasInfo.perdidas;group.clearLayers();
  const ys=document.getElementById('sel-year').value;
  const cy={'20':'#eab308','21':'#f97316','22':'#ef4444','23':'#ec4899','24':'#a855f7'};
  L.geoJSON(datosPerdidas,{
    filter:f=>ys==='all'||String(f.properties.year)===ys,
    pointToLayer:(f,ll)=>L.circleMarker(ll,{radius:4,fillColor:cy[String(f.properties.year)]||'#fff',color:'#000',weight:1,fillOpacity:0.8}).bindPopup(`Pérdida · Año: 20${f.properties.year}`)
  }).addTo(group);
}

function actualizarConectividad(val){cuantilMinimo=parseInt(val);document.getElementById('val-cuantil').textContent=val;if(capasInfo.conectividad.getLayers().length>0)renderRasterConectividad();}

function renderRasterConectividad(){
  if(!georasterConectividad)return;
  const g=capasInfo.conectividad;g.clearLayers();
  g.addLayer(new GeoRasterLayer({georaster:georasterConectividad,opacity:0.7,pixelValuesToColorFn:v=>{const n=Math.round(v[0]);if(n<cuantilMinimo||n>10||isNaN(n))return null;return turboColors[n];},resolution:256}));
}

// ── Municipios y Veredas ─────────────────────────────────────────────────────
async function cargarMunicipios() {
  if (veredasGJ) { municipiosGJ = veredasGJ; return veredasGJ; }
  await cargarVeredas();
  municipiosGJ = veredasGJ;

  const sel = document.getElementById('sel-municipio');
  if (sel && veredasGJ) {
    sel.innerHTML = '<option value="">— Todos los municipios —</option>';
    const nombres = [...new Set(
      (veredasGJ.features||[])
        .map(f => (f.properties.NOMB_MPIO||'').trim())
        .filter(Boolean)
    )].sort();
    nombres.forEach(nom => {
      const o = document.createElement('option');
      o.value = nom; o.textContent = nom;
      sel.appendChild(o);
    });
  }
  return veredasGJ;
}

async function cargarVeredas() {
  if (veredasGJ) return veredasGJ;
  try {
    const resp = await fetch(BASE_URL + 'veredas.geojson');
    veredasGJ = await resp.json();
    return veredasGJ;
  } catch(e) { console.error('Error cargando veredas:', e); return null; }
}

function limpiarCapaMunicipio() {
  if (capaMunicipioViz) { map.removeLayer(capaMunicipioViz); capaMunicipioViz = null; }
  if (capaVeredasViz)   { map.removeLayer(capaVeredasViz);   capaVeredasViz   = null; }
}

async function seleccionarMunicipio(nombre) {
  municipioActual = nombre;
  limpiarCapaMunicipio();
  const info = document.getElementById('municipio-veredas-info');

  if (!nombre) {
    if (info) info.textContent = '';
    cambiarAreaAnalisis(); aplicarFiltros();
    return;
  }

  await cargarVeredas();

  if (veredasGJ) {
    const vFeat = veredasGJ.features.filter(
      f => (f.properties.NOMB_MPIO||'').toUpperCase() === nombre.toUpperCase()
    );
    if (vFeat.length > 0) {
      capaVeredasViz = L.geoJSON(
        { type: 'FeatureCollection', features: vFeat },
        {
          style: { color: '#7c3aed', weight: 1, fillColor: '#7c3aed', fillOpacity: 0.04, dashArray: '3 3' },
          onEachFeature: (f, l) => l.bindTooltip(f.properties.NOMBRE_VER || '', { permanent: false, className: 'text-[9px]' })
        }
      ).addTo(map);
      if (info) info.textContent = `${vFeat.length} vereda${vFeat.length!==1?'s':''} en ${nombre}`;
    } else {
      if (info) info.textContent = 'Sin veredas en este municipio';
    }
  }

  cambiarAreaAnalisis();
  aplicarFiltros();
}

function _paramsFechaAlertas() {
  let ini, fin;
  if (modoPeriodo === 'rango' && fechaInicio && fechaFin) {
    ini = fechaInicio.toISOString();
    fin = fechaFin.toISOString();
  } else {
    ini = new Date(Date.now() - diasActual * 864e5).toISOString();
    fin = new Date().toISOString();
  }
  return `fecha_deteccion=gte.${encodeURIComponent(ini)}&fecha_deteccion=lte.${encodeURIComponent(fin)}`;
}

async function cargarAlertas() {
  try {
    const filtroFechas = _paramsFechaAlertas();
    const url = `${SUPABASE_URL}/rest/v1/alertas?select=*&${filtroFechas}&order=fecha_deteccion.desc&limit=10000`;
    const r = await fetch(url, {
      headers: { apikey: SUPABASE_KEY, Authorization: `Bearer ${SUPABASE_KEY}` }
    });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    todasAlertas = await r.json();
    aplicarFiltros();
  } catch(e) {
    document.getElementById('lista-alertas').innerHTML = `<div class="loading">Error: ${e.message}</div>`;
  }
}

// ── Deforestación GEE ─────────────────────────────────────────────────────────
async function toggleDeforestacion(visible) {
  deforVisible=visible;
  const _sd=document.getElementById('sub-deforestacion'); if(_sd){ if(visible)_sd.classList.remove('hidden'); else _sd.classList.add('hidden'); }
  if(!visible){marcadoresDefor.forEach(m=>map.removeLayer(m));marcadoresDefor=[];document.getElementById('defor-stats').innerHTML='';aplicarFiltros();return;}
  await cargarDeforestacion();
}

async function cargarDeforestacion() {
  if(!deforVisible)return;
  try {
    const dias=parseInt(document.getElementById('sel-defor-periodo').value);
    const ini=new Date(Date.now()-dias*864e5).toISOString().slice(0,10);
    const r=await fetch(`${SUPABASE_URL}/rest/v1/alertas_deforestacion?select=*&fecha_deteccion=gte.${ini}&order=fecha_deteccion.desc&limit=500`,{headers:{apikey:SUPABASE_KEY,Authorization:`Bearer ${SUPABASE_KEY}`}});
    if(!r.ok)throw new Error('HTTP '+r.status);
    alertasDefor=await r.json();filtrarDeforestacion();aplicarFiltros();
  } catch(e){console.error('Error defor:',e);document.getElementById('defor-stats').textContent='Error cargando datos.';}
}

function filtrarDeforestacion() {
  if(!deforVisible)return;
  const sev=document.getElementById('sel-defor-severidad').value;
  let f=sev==='all'?alertasDefor:alertasDefor.filter(a=>a.severidad===sev);
  
  if (areaAnalisisActiva === 'dibujo' && dibujoPoligonoCoords && dibujoPoligonoCoords.length >= 3) {
    f = f.filter(a => pasaFiltroAreaDibujo(a.latitud, a.longitud));
  }
  
  marcadoresDefor.forEach(m=>map.removeLayer(m));marcadoresDefor=[];
  f.forEach(a=>{
    const color=COLORES_SEV[a.severidad]||'#f97316';
    const r=a.severidad==='severa'?10:a.severidad==='moderada'?7:5;
    const m=L.circleMarker([a.latitud,a.longitud],{radius:r,fillColor:color,color:'#fff',weight:1.5,fillOpacity:0.85}).addTo(map);
    const fecha=new Date(a.fecha_deteccion+'T12:00:00').toLocaleDateString('es-CO');
    m.bindPopup(`<div class="popup-title" style="color:${color}">Deforestación · ${a.severidad}</div><div class="popup-row">Predio <span>${a.predio_id}</span></div><div class="popup-row">Fecha <span>${fecha}</span></div><div class="popup-row">Δ NDVI <span>${a.cambio_ndvi!=null?Number(a.cambio_ndvi).toFixed(3):'-'}</span></div><div class="popup-row">Área <span>${a.area_afectada_ha??'-'} ha</span></div>`);
    marcadoresDefor.push(m);
  });
  const ss=f.filter(a=>a.severidad==='severa').length;
  const sm=f.filter(a=>a.severidad==='moderada').length;
  const sl=f.filter(a=>a.severidad==='leve').length;
  document.getElementById('defor-stats').innerHTML=`${f.length} alertas · <span style="color:#dc2626">● ${ss} severa</span> <span style="color:#f97316">● ${sm} mod.</span> <span style="color:#eab308">● ${sl} leve</span>`;
}

// ══════════════════════════════════════════════════════════════
// GFW — Global Forest Watch Integration
// ══════════════════════════════════════════════════════════════
function setGFWStatus(msg,tipo=''){const el=document.getElementById('gfw-status');el.textContent=msg;el.className='gfw-status'+(tipo?' '+tipo:'');}

function gfwFechas(dias){
  if (modoPeriodo === 'rango' && fechaInicio && fechaFin) {
    return {
      startDate: fechaInicio.toISOString().slice(0,10),
      endDate:   fechaFin.toISOString().slice(0,10)
    };
  }
  const fin=new Date(),ini=new Date(Date.now()-dias*864e5);
  return{startDate:ini.toISOString().slice(0,10),endDate:fin.toISOString().slice(0,10)};
}

async function toggleGFW(key,visible){
  const capa=gfwCapas[key];capa.visible=visible;
  document.getElementById(`sub-${key}`).classList.toggle('visible',visible);
  if(!visible){
    capa.marcadores.forEach(m=>map.removeLayer(m));capa.marcadores=[];
    if(capa.tileLayer){map.removeLayer(capa.tileLayer);capa.tileLayer=null;}
    document.getElementById(`stats-${key}`).innerHTML='Sin datos cargados';
    if(!Object.values(gfwCapas).some(c=>c.visible))setGFWStatus('—');
    return;
  }
  if(key==='hansen') cargarGFWTilesHansen();
  else await cargarGFWAlertas(key);
}

function cargarGFWTilesHansen(){
  const capa=gfwCapas.hansen;
  if(capa.tileLayer){map.removeLayer(capa.tileLayer);capa.tileLayer=null;}
  const K=GFW_API_KEY;
  capa.tileLayer=L.layerGroup([
    L.tileLayer(`https://tiles.globalforestwatch.org/umd_tree_cover_density_2000/latest/dynamic/{z}/{x}/{y}.png?x-api-key=${K}`,{opacity:hansenOpacity,maxZoom:13}),
    L.tileLayer(`https://tiles.globalforestwatch.org/umd_tree_cover_loss/latest/dynamic/{z}/{x}/{y}.png?x-api-key=${K}`,{opacity:hansenOpacity,maxZoom:13})
  ]).addTo(map);
  document.getElementById('stats-hansen').innerHTML=`Cobertura 2000 <span style="color:#22c55e">●</span> &nbsp; Pérdida 2001-2023 <span style="color:#dc2626">●</span><br>Opacidad: <span>${Math.round(hansenOpacity*100)}%</span>`;
  setGFWStatus('✓ OK','ok');
}

function actualizarOpacidadHansen(val){
  hansenOpacity=val/100;document.getElementById('val-hansen').textContent=val+'%';
  const capa=gfwCapas.hansen;
  if(capa.tileLayer)capa.tileLayer.getLayers().forEach(l=>{if(l.setOpacity)l.setOpacity(hansenOpacity);});
  document.getElementById('stats-hansen').innerHTML=`Cobertura 2000 <span style="color:#22c55e">●</span> &nbsp; Pérdida 2001-2023 <span style="color:#dc2626">●</span><br>Opacidad: <span>${val}%</span>`;
}

const GFW_CFG = {
  glad:{
    dataset:'gfw_integrated_alerts',
    sqlFn:(s,e)=>`SELECT latitude,longitude,gfw_integrated_alerts__date AS fecha,gfw_integrated_alerts__confidence AS confianza,gfw_integrated_alerts__intensity AS intensidad FROM results WHERE gfw_integrated_alerts__date>='${s}' AND gfw_integrated_alerts__date<='${e}'`,
    colorFn:row=>(row.confianza||'').toLowerCase().includes('high')?'#dc2626':'#f97316',
    radiusFn:row=>(row.confianza||'').toLowerCase().includes('high')?8:5,
    popupFn:(row,lat,lng)=>`<div class="popup-title gfw">Alerta GLAD</div><div class="popup-row">Coords <span>${lat.toFixed(4)}, ${lng.toFixed(4)}</span></div><div class="popup-row">Fecha <span>${row.fecha||'N/D'}</span></div><div class="popup-row">Confianza <span>${row.confianza||'N/D'}</span></div><div class="popup-row">Intensidad <span>${row.intensidad||'N/D'}</span></div>`
  },
  radd:{
    dataset:'wur_radd_alerts',
    sqlFn:(s,e)=>`SELECT latitude,longitude,wur_radd_alerts__date AS fecha,wur_radd_alerts__confidence AS confianza FROM results WHERE wur_radd_alerts__date>='${s}' AND wur_radd_alerts__date<='${e}'`,
    colorFn:row=>(row.confianza||'').toLowerCase()==='confirmed'?'#7c3aed':'#a78bfa',
    radiusFn:row=>(row.confianza||'').toLowerCase()==='confirmed'?8:5,
    popupFn:(row,lat,lng)=>`<div class="popup-title gfw" style="color:#a78bfa">Alerta RADD</div><div class="popup-row">Coords <span>${lat.toFixed(4)}, ${lng.toFixed(4)}</span></div><div class="popup-row">Fecha <span>${row.fecha||'N/D'}</span></div><div class="popup-row">Confianza <span>${row.confianza||'N/D'}</span></div>`
  },
  fires:{dataset:'nasa_viirs_fire_alerts',sqlFn:(s,e)=>`SELECT latitude,longitude,acq_date AS fecha,confidence__cat AS confianza FROM results WHERE acq_date>='${s}' AND acq_date<='${e}'`,colorFn:()=>'#ff2200',radiusFn:()=>6,popupFn:(row,lat,lng)=>`<div class="popup-title gfw">Fire GFW</div><div class="popup-row">Coords <span>${lat.toFixed(4)}, ${lng.toFixed(4)}</span></div>`},
  hansen:{dataset:'umd_tree_cover_loss',sqlFn:()=>'',colorFn:()=>'',radiusFn:()=>0,popupFn:()=>''}
};

async function cargarGFWAlertas(key){
  setGFWStatus('Cargando…','loading');
  const statsEl=document.getElementById(`stats-${key}`);
  statsEl.innerHTML='<span style="color:#ca8a04">⏳ Consultando GFW…</span>';
  const cfg=GFW_CFG[key];
  const {startDate,endDate}=gfwFechas(diasActual);
  const geom=await obtenerGeomActiva();
  let sql=cfg.sqlFn(startDate,endDate);
  if(key==='glad'){const conf=document.getElementById('sel-glad-conf')?.value;if(conf==='high')sql+=` AND gfw_integrated_alerts__confidence IN ('high','highest')`;if(conf==='highest')sql+=` AND gfw_integrated_alerts__confidence='highest'`;}
  sql+=' LIMIT 2000';
  try{
    const resp=await fetch(`${GFW_BASE}/dataset/${cfg.dataset}/latest/query/json`,{method:'POST',headers:{'Content-Type':'application/json','x-api-key':GFW_API_KEY},body:JSON.stringify({sql,geometry:geom})});
    if(!resp.ok)throw new Error(`HTTP ${resp.status}`);
    const data=await resp.json();
    let datos = data.data || [];
    if (RECORTAR_DIBUJO_A_AREA_ESTUDIO && areaAnalisisActiva === 'dibujo' && areaEstudioGeom) {
      datos = datos.filter(r => {
        const lat = _numCoord(r.latitude), lng = _numCoord(r.longitude);
        if (lat === null || lng === null) return false;
        return puntoEnGeoJSON(lat, lng, areaEstudioGeom);
      });
    }
    gfwCapas[key].datos = datos;
    renderGFWMarcadores(key, gfwCapas[key].datos);
  }catch(e){
    console.error(`GFW ${key}:`,e);
    statsEl.innerHTML=`<span style="color:var(--fuego)">❌ ${e.message}</span>`;
    setGFWStatus('Error','err');
  }
}

function renderGFWMarcadores(key,rows){
  const capa=gfwCapas[key];
  capa.marcadores.forEach(m=>map.removeLayer(m));capa.marcadores=[];
  const cfg=GFW_CFG[key];
  const gladConf=key==='glad'?(document.getElementById('sel-glad-conf')?.value||''):'';
  let counts={};
  rows.forEach(row=>{
    const lat=parseFloat(row.latitude),lng=parseFloat(row.longitude);
    if(isNaN(lat)||isNaN(lng))return;
    const conf=(row.confianza||'').toLowerCase();
    if(key==='glad'&&gladConf==='high'&&conf==='nominal')return;
    if(key==='glad'&&gladConf==='highest'&&conf!=='highest')return;
    counts[conf]=(counts[conf]||0)+1;
    const m=L.circleMarker([lat,lng],{radius:cfg.radiusFn(row),fillColor:cfg.colorFn(row),color:'#fff',weight:1.2,fillOpacity:.88}).addTo(map);
    m.bindPopup(cfg.popupFn(row,lat,lng));
    capa.marcadores.push(m);
  });
  const tot=capa.marcadores.length;
  const {startDate,endDate}=gfwFechas(diasActual);
  const area = areaAnalisisActiva==='estudio'?'área de estudio'
             : areaAnalisisActiva==='nucleos'?'núcleos boscosos'
             : areaAnalisisActiva==='municipio'?(municipioActual?`municipio ${municipioActual}`:'todos los municipios')
             : areaAnalisisActiva==='dibujo'?`polígono personalizado (${dibujoArea_ha.toFixed(1)} ha)`
             : 'área activa';
  if(tot===0){
    document.getElementById(`stats-${key}`).innerHTML=`<span>Sin alertas en ${area}</span>`;
    setGFWStatus('✓ 0','ok');
  }else{
    const confRes=Object.entries(counts).map(([c,n])=>`<span>${n} ${c||'?'}</span>`).join(' &nbsp; ');
    document.getElementById(`stats-${key}`).innerHTML=`<b style="color:var(--gfw-l)">${tot} alertas</b> en ${area}<br>${startDate} → ${endDate}<br>${confRes}`;
    setGFWStatus(`✓ ${tot}`,'ok');
  }
}

function filtrarGFW(key){if(!gfwCapas[key].visible)return;if(key==='hansen')return;if(gfwCapas[key].datos.length)renderGFWMarcadores(key,gfwCapas[key].datos);else cargarGFWAlertas(key);}
function recargarGFWTile(key){if(gfwCapas[key].visible){if(gfwCapas[key].datos.length)renderGFWMarcadores(key,gfwCapas[key].datos);else cargarGFWAlertas(key);}}

// ── Descarga GFW (3 archivos) ──────────────────────────────────────────────────
const _gfwLocks = { glad: false, radd: false, hansen: false };

async function descargarGFW(key){
  if (_gfwLocks[key]) { return; }
  _gfwLocks[key] = true;

  const hoy=new Date().toISOString().slice(0,10);
  const btnId = (key === 'glad') ? 'btn-dl-gfw-glad' : 'btn-dl-gfw-radd';
  const btn = document.getElementById(btnId);
  const origText = btn ? btn.textContent : '';
  if(btn){btn.textContent='⏳ Generando…';btn.disabled=true;}
  try{
    if(key==='hansen'){
      window.open('https://data.globalforestwatch.org/datasets/gfw::tree-cover-loss/about','_blank');
      if(btn){btn.textContent=origText;btn.disabled=false;}
      _gfwLocks[key] = false;
      return;
    }
    const datos=gfwCapas[key].datos;
    if(!datos.length){
      alert(`Activa la capa ${key.toUpperCase()} primero y espera que cargue.`);
      if(btn){btn.textContent=origText;btn.disabled=false;}
      _gfwLocks[key] = false;
      return;
    }

    const AREA_HA_POR_PIXEL = 0.09;
    let veredasMpioGFW = [];
    if (municipioActual && veredasGJ) {
      veredasMpioGFW = veredasGJ.features.filter(
        f => (f.properties.NOMB_MPIO||'').toUpperCase() === municipioActual.toUpperCase()
      );
    }
    if (!municipiosGJ) await cargarMunicipios();

    function municipioDeAlerta(lat, lng) {
      if (!veredasGJ) return '';
      const vf = veredasGJ.features.find(f => puntoEnPoligono(lat, lng, vf));
      return vf ? (vf.properties.NOMB_MPIO || '') : 'Fuera de límites';
    }
    function veredaDeAlerta(lat, lng) {
      if (veredasMpioGFW.length) {
        const vf = veredasMpioGFW.find(vf => puntoEnPoligono(lat, lng, vf));
        return vf ? (vf.properties.NOMBRE_VER || 'Sin vereda') : 'Sin vereda';
      }
      if (veredasGJ) {
        const vf = veredasGJ.features.find(f => puntoEnPoligono(lat, lng, f));
        return vf ? (vf.properties.NOMBRE_VER || '') : '';
      }
      return '';
    }

    let cab, rows;
    if (key === 'glad') {
      cab = 'fuente,fecha,municipio,vereda,latitud,longitud,confianza,intensidad,area_ha';
      rows = datos.map(r => {
        const lat = parseFloat(r.latitude||0), lng = parseFloat(r.longitude||0);
        const mpio = municipioActual || municipioDeAlerta(lat, lng);
        const ver  = veredaDeAlerta(lat, lng);
        return ['GFW_GLAD', r.fecha||'', mpio, ver, lat, lng,
                r.confianza||'', r.intensidad||'',
                AREA_HA_POR_PIXEL.toFixed(2)].join(',');
      });
    } else {
      cab = 'fuente,fecha,municipio,vereda,latitud,longitud,confianza,area_ha';
      rows = datos.map(r => {
        const lat = parseFloat(r.latitude||0), lng = parseFloat(r.longitude||0);
        const mpio = municipioActual || municipioDeAlerta(lat, lng);
        const ver  = veredaDeAlerta(lat, lng);
        return ['GFW_RADD', r.fecha||'', mpio, ver, lat, lng,
                r.confianza||'',
                AREA_HA_POR_PIXEL.toFixed(2)].join(',');
      });
    }
    _dl([cab,...rows].join('\n'), `gfw_${key}_puntos_chameza_${hoy}.csv`);

    const parches = dissolverPixelesEnParches(datos, AREA_HA_POR_PIXEL, key);
    const geojson = {
      type: 'FeatureCollection',
      features: parches.map((p, i) => ({
        type: 'Feature',
        properties: {
          parche_id: `${key.toUpperCase()}_${i+1}`,
          fuente: key === 'glad' ? 'GFW_GLAD' : 'GFW_RADD',
          n_pixeles: p.n_pixeles,
          area_ha: parseFloat(p.area_ha.toFixed(3)),
          fecha_min: p.fecha_min,
          fecha_max: p.fecha_max,
          municipio: municipioActual || municipioDeAlerta(p.centroide[1], p.centroide[0]),
          vereda: veredaDeAlerta(p.centroide[1], p.centroide[0])
        },
        geometry: { type: 'Polygon', coordinates: [p.poligono] }
      }))
    };
    const blobGJ = new Blob([JSON.stringify(geojson, null, 2)], {type:'application/geo+json'});
    const urlGJ = URL.createObjectURL(blobGJ);
    const aGJ = document.createElement('a');
    aGJ.href = urlGJ; aGJ.download = `gfw_${key}_parches_chameza_${hoy}.geojson`;
    document.body.appendChild(aGJ);
    aGJ.click();
    document.body.removeChild(aGJ);
    URL.revokeObjectURL(urlGJ);

    await descargarGFWResumen(key, datos, AREA_HA_POR_PIXEL, hoy);

    if(btn){
      btn.textContent='✓ 3 archivos descargados';
      setTimeout(()=>{btn.textContent=origText;btn.disabled=false;}, 2500);
    }
  }catch(e){
    console.error('Error descarga GFW:', e);
    if(btn){btn.textContent=origText;btn.disabled=false;}
    alert('Error generando descarga: '+e.message);
  } finally {
    setTimeout(()=>{ _gfwLocks[key] = false; }, 1500);
  }
}

function dissolverPixelesEnParches(datos, areaPorPixel, key) {
  if (!datos.length) return [];
  const PIXEL_DEG = 0.00028;
  const TOLERANCIA = PIXEL_DEG * 1.5;

  const puntos = datos.map(d => ({
    lat: parseFloat(d.latitude),
    lng: parseFloat(d.longitude),
    fecha: d.fecha || '',
    raw: d,
    cluster: -1
  })).filter(p => !isNaN(p.lat) && !isNaN(p.lng));

  let clusterId = 0;
  for (let i = 0; i < puntos.length; i++) {
    if (puntos[i].cluster !== -1) continue;
    puntos[i].cluster = clusterId;
    const cola = [i];
    while (cola.length) {
      const idx = cola.shift();
      const a = puntos[idx];
      for (let j = 0; j < puntos.length; j++) {
        if (puntos[j].cluster !== -1) continue;
        const b = puntos[j];
        if (Math.abs(a.lat - b.lat) <= TOLERANCIA && Math.abs(a.lng - b.lng) <= TOLERANCIA) {
          b.cluster = clusterId;
          cola.push(j);
        }
      }
    }
    clusterId++;
  }

  const parches = [];
  for (let c = 0; c < clusterId; c++) {
    const miembros = puntos.filter(p => p.cluster === c);
    if (!miembros.length) continue;

    const lats = miembros.map(m => m.lat);
    const lngs = miembros.map(m => m.lng);
    const half = PIXEL_DEG / 2;
    const minLat = Math.min(...lats) - half;
    const maxLat = Math.max(...lats) + half;
    const minLng = Math.min(...lngs) - half;
    const maxLng = Math.max(...lngs) + half;

    const pol = [
      [minLng, minLat], [maxLng, minLat],
      [maxLng, maxLat], [minLng, maxLat],
      [minLng, minLat]
    ];
    const cLat = lats.reduce((s,v)=>s+v,0) / lats.length;
    const cLng = lngs.reduce((s,v)=>s+v,0) / lngs.length;
    const fechas = miembros.map(m => m.fecha).filter(Boolean).sort();

    parches.push({
      n_pixeles: miembros.length,
      area_ha: miembros.length * areaPorPixel,
      centroide: [cLng, cLat],
      poligono: pol,
      fecha_min: fechas[0] || '',
      fecha_max: fechas[fechas.length-1] || ''
    });
  }
  parches.sort((a,b) => b.area_ha - a.area_ha);
  return parches;
}

async function descargarGFWResumen(key, datos, areaPorPixel, hoy) {
  if (!municipiosGJ) await cargarMunicipios();
  if (!veredasGJ) return;

  const agg = {};
  datos.forEach(d => {
    const lat = parseFloat(d.latitude), lng = parseFloat(d.longitude);
    if (isNaN(lat) || isNaN(lng)) return;

    let mpio = 'Fuera de límites', ver = 'Sin vereda';
    const vf = veredasGJ.features.find(f => puntoEnPoligono(lat, lng, f));
    if (vf) {
      mpio = vf.properties.NOMB_MPIO || 'Sin nombre';
      ver  = vf.properties.NOMBRE_VER || 'Sin nombre';
    }
    const k = `${mpio}||${ver}`;
    if (!agg[k]) agg[k] = { mpio, ver, n_pixeles: 0, fechas: [] };
    agg[k].n_pixeles++;
    if (d.fecha) agg[k].fechas.push(d.fecha);
  });

  const esc = v => {
    const s = String(v ?? '');
    return s.includes(',') ? '"'+s.replace(/"/g,'""')+'"' : s;
  };

  const cabVer = ['fuente','municipio','vereda','n_pixeles','area_ha','fecha_min','fecha_max'].join(',');
  const rowsVer = Object.values(agg)
    .sort((a,b) => b.n_pixeles - a.n_pixeles)
    .map(r => {
      const fs = r.fechas.sort();
      return [
        key === 'glad' ? 'GFW_GLAD' : 'GFW_RADD',
        esc(r.mpio), esc(r.ver), r.n_pixeles,
        (r.n_pixeles * areaPorPixel).toFixed(2),
        fs[0]||'', fs[fs.length-1]||''
      ].join(',');
    });

  const aggM = {};
  Object.values(agg).forEach(r => {
    if (!aggM[r.mpio]) aggM[r.mpio] = { mpio: r.mpio, n_pixeles: 0, n_veredas: 0 };
    aggM[r.mpio].n_pixeles += r.n_pixeles;
    aggM[r.mpio].n_veredas++;
  });
  const cabMun = ['fuente','municipio','n_pixeles','area_ha','n_veredas_afectadas'].join(',');
  const rowsMun = Object.values(aggM)
    .sort((a,b) => b.n_pixeles - a.n_pixeles)
    .map(r => [
      key === 'glad' ? 'GFW_GLAD' : 'GFW_RADD',
      esc(r.mpio), r.n_pixeles,
      (r.n_pixeles * areaPorPixel).toFixed(2),
      r.n_veredas
    ].join(','));

  const lines = [
    '## RESUMEN POR VEREDA',
    cabVer, ...rowsVer,
    '',
    '## RESUMEN POR MUNICIPIO',
    cabMun, ...rowsMun
  ];
  _dl(lines.join('\n'), `gfw_${key}_resumen_chameza_${hoy}.csv`);
}

// ── CUNAGÜITO ────────────────────────────────────────────────────────
const CUNA_DATOS={
  mapabase:{img:'mapabase.png',msg:`Para cambiar el mapa base, usa el control de capas en la <b>esquina inferior derecha</b> del mapa. Encontrarás: <b>Google Maps</b> (predeterminado), <b>Satélite</b>, <b>Relieve</b> y <b>Oscuro</b>.`},
  incendios:{img:'incendios.png',msg:`Selecciona <b>Incendios</b> en "Tipo de alerta", elige tu período (24h/7d/30d) y el área de análisis. Luego usa los botones <b>⬇ CSV</b> o <b>⬇ PDF</b> para descargar solo los datos de tu zona.`},
  deforestacion:{img:'deforestacion.png',msg:`Selecciona <b>Pérdida cobertura</b> en "Tipo de alerta" y activa <b>GLAD</b> o <b>RADD</b>. Una vez carguen las alertas en el mapa, descarga con <b>⬇ CSV GFW</b>. Para la deforestación GEE, actívala en "Capas adicionales".`},
  conectividad:{img:'conectividad.png',msg:`En <b>Capas adicionales</b> (al fondo del panel), marca <b>Conectividad (Deciles)</b>. Los deciles del 1 al 10 representan la importancia de conectividad del paisaje — cuantil más alto = mayor importancia.`},
  gfw:{img:'gfw.png',msg:`Selecciona <b>Pérdida cobertura</b> en "Tipo de alerta". Allí encontrarás:<br><br>🌲 <b>GLAD</b> — Alertas Landsat semanales<br>📡 <b>RADD</b> — Radar Sentinel-1, funciona bajo nubes<br>🌳 <b>Hansen</b> — Cobertura y pérdida 2001-2023<br><br>Todas las alertas se filtran automáticamente al <b>área de análisis</b> que hayas seleccionado.`},
  sobre: {
    img: null,
    esMulti: true,
    slides: [
      { titulo: '¿Qué hace la plataforma?', img: 'kuna_workflow.png' },
      { titulo: '¿En qué estamos ahora?',   img: 'kuna_workflow_estado.png' }
    ]
  },
  funcionalidades: {
    img: null,
    esMulti: true,
    slides: [
      {
        titulo: 'Cambiar el mapa base',
        img: 'mapabase.png',
        kunaImg: 'kuna_senala.png',
        texto: `Usa el control de capas en la <b>esquina inferior derecha</b> del mapa. Tienes 5 opciones: <b>CartoDB Claro</b> (predeterminado), <b>Google Maps</b>, <b>Satélite</b>, <b>Relieve</b> y <b>Oscuro</b>. Cambia según la mejor visualización para tu análisis.`
      },
      {
        titulo: 'Global Forest Watch — GLAD & RADD',
        img: 'gfw_incon.png',
        kunaImg: 'kuna_saludando.png',
        texto: `<b class="gfw">Global Forest Watch</b> (WRI) provee alertas satelitales de deforestación casi en tiempo real.<br><br>🌲 <b>GLAD</b> — Landsat · Universidad de Maryland · detección semanal<br>📡 <b>RADD</b> — Sentinel-1 SAR · Univ. Wageningen · funciona bajo nubes<br><br>Créditos: © Global Forest Watch / World Resources Institute. Datos bajo licencia CC BY 4.0.`
      },
      {
        titulo: 'Alertas de Incendios — NASA FIRMS',
        img: 'incendios.png',
        kunaImg: 'kuna_senala.png',
        texto: `Los puntos de calor provienen de <b>NASA FIRMS</b> (Fire Information for Resource Management System), usando sensores <b>VIIRS</b> (375 m) y <b>MODIS</b> (1 km) actualizados cada 3 horas.<br><br>Cada punto muestra: coordenadas, nivel de confianza (Alta / Nominal / Baja) y potencia radiativa del fuego (<b>FRP en MW</b>).`
      },
      {
        titulo: 'Capas Adicionales',
        img: 'capas_adicionales.png',
        kunaImg: 'kuna_saludando.png',
        texto: `En <b>Capas adicionales</b> (panel izquierdo) puedes activar:<br><br>📍 <b>Límites municipales</b> — contornos administrativos<br>📐 <b>Área de estudio</b> — polígono de monitoreo CBC Cunaguaro<br>🌲 <b>Núcleos boscosos</b> — fragmentos prioritarios<br>📉 <b>Pérdidas 2020-2024</b> — pérdida forestal por año<br>🔗 <b>Conectividad</b> — modelo de deciles de importancia ecológica<br>🌿 <b>Alertas de GEE</b> — basado en cambio NDVI Sentinel-2`
      }
    ]
  }
};

function cunaAbrir(){
  const el=document.getElementById('cuna-overlay');
  if (el) { el.classList.remove('hidden'); el.classList.add('flex'); }
}
function cunaCerrar(){
  const el=document.getElementById('cuna-overlay');
  if (el) { el.classList.add('hidden'); el.classList.remove('flex'); }
  setTimeout(cunaVolver,300);
}
function cunaOverlayClick(e){if(e.target===document.getElementById('cuna-overlay'))cunaCerrar();}

function cunaIr(key){
  const d=CUNA_DATOS[key];
  if(!d) return;

  const avatar=document.getElementById('cuna-avatar');
  if (avatar) {
    avatar.style.opacity='0';
    setTimeout(()=>{ avatar.src='kuna_saludando.png'; avatar.style.opacity='1'; },150);
  }

  const menu=document.getElementById('cuna-menu'),detail=document.getElementById('cuna-detail');
  if (menu) menu.style.opacity='0';

  setTimeout(()=>{
    if (menu) menu.style.display='none';
    if (detail) {
      detail.innerHTML='';
      detail.classList.remove('hidden');
      detail.classList.add('flex');

      if (d.esMulti) {
        _cunaCurrentKey = key;
        detail.innerHTML = cunaRenderMulti(d, key);
        cunaSlide(0, d);
      } else {
        detail.innerHTML = `
          <img id="cuna-detail-img" class="w-full h-44 object-cover rounded-xl mb-4 border border-slate-100" src="${d.img||''}" alt="">
          <div id="cuna-detail-text" class="text-sm text-slate-600 leading-relaxed cuna-detail-text">${d.msg||''}</div>
          <button onclick="cunaVolver()" class="mt-5 w-full py-3 border-2 border-slate-200 rounded-xl text-xs font-bold text-slate-600 hover:bg-slate-50 transition-all">← Volver al menú</button>`;
      }
      detail.style.opacity='0';
      setTimeout(()=>{ detail.style.opacity='1'; },20);
    }
  },180);
}

let _cunaSlideIdx = 0;
let _cunaCurrentKey = null;

function cunaRenderMulti(d, key) {
  return `
    <div class="flex items-center justify-between mb-3 w-full">
      <div class="text-[9px] font-bold text-slate-400 uppercase tracking-widest" id="kuna-slide-label">Cargando…</div>
      <div class="flex gap-1" id="kuna-dots"></div>
    </div>
    <div id="kuna-slide-content" class="flex flex-col gap-3 w-full"></div>
    <div class="flex gap-2 mt-4 w-full">
      <button onclick="cunaSlideNav(-1)" id="btn-prev" class="flex-1 py-2.5 border-2 border-slate-200 rounded-xl text-xs font-bold text-slate-500 hover:bg-slate-50 transition-all">← Anterior</button>
      <button onclick="cunaSlideNav(1)"  id="btn-next" class="flex-1 py-2.5 bg-emerald-600 text-white rounded-xl text-xs font-bold hover:bg-emerald-700 transition-all">Siguiente →</button>
    </div>
    <button onclick="cunaVolver()" class="mt-2 w-full py-2.5 border-2 border-slate-100 rounded-xl text-[10px] font-bold text-slate-400 hover:bg-slate-50 transition-all">← Volver al menú</button>`;
}

function cunaSlide(idx, d) {
  if (!d) return;
  const slides = d.slides || [];
  _cunaSlideIdx = Math.max(0, Math.min(idx, slides.length-1));
  const s = slides[_cunaSlideIdx];
  if (!s) return;

  const lbl = document.getElementById('kuna-slide-label');
  if (lbl) lbl.textContent = s.titulo;

  const dots = document.getElementById('kuna-dots');
  if (dots) {
    dots.innerHTML = slides.map((_, i) =>
      `<div class="w-1.5 h-1.5 rounded-full transition-all ${i===_cunaSlideIdx?'bg-emerald-600':'bg-slate-200'}"></div>`
    ).join('');
  }

  const cont = document.getElementById('kuna-slide-content');
  if (cont) {
    if (!s.kunaImg && !s.texto) {
      cont.innerHTML = `<img src="${s.img}" alt="${s.titulo}" class="w-full rounded-xl border border-slate-100 shadow-md" style="width:100%;object-fit:contain;background:#f8faf9;display:block">`;
    } else {
      cont.innerHTML = `
        <div class="flex items-start gap-3">
          <img src="${s.kunaImg||'kuna_saludando.png'}" alt="Kuna" class="w-14 h-14 rounded-xl object-cover shrink-0 border border-emerald-100">
          <div class="text-sm font-semibold text-slate-700 leading-relaxed">${s.texto||''}</div>
        </div>
        <img src="${s.img}" alt="${s.titulo}" class="w-full rounded-xl border border-slate-100 shadow-sm" style="max-height:200px;object-fit:contain;background:#f8faf9">`;
    }
  }

  const prev = document.getElementById('btn-prev');
  const next = document.getElementById('btn-next');
  if (prev) prev.style.visibility = _cunaSlideIdx === 0 ? 'hidden' : 'visible';
  if (next) next.textContent = _cunaSlideIdx === slides.length-1 ? '✓ Listo' : 'Siguiente →';
  if (next) next.onclick = _cunaSlideIdx === slides.length-1
    ? cunaVolver
    : () => cunaSlideNav(1);
}

function cunaSlideNav(dir) {
  if (!_cunaCurrentKey) return;
  const d = CUNA_DATOS[_cunaCurrentKey];
  if (!d || !d.slides) return;
  const newIdx = _cunaSlideIdx + dir;
  if (newIdx >= 0 && newIdx < d.slides.length) {
    cunaSlide(newIdx, d);
  }
}

function cunaVolver(){
  const avatar=document.getElementById('cuna-avatar');
  if (avatar) {
    avatar.style.opacity='0';
    setTimeout(()=>{avatar.src='Cunaguito1.png';avatar.style.opacity='1';},150);
  }
  const msg=document.getElementById('cuna-msg');
  if (msg) {
    msg.style.opacity='0';
    setTimeout(()=>{msg.textContent='¡Hola! Soy Kuna. Estoy aquí para guiarte en el manejo del aplicativo. ¿Qué quieres hacer?';msg.style.opacity='1';},150);
  }
  const menu=document.getElementById('cuna-menu'),detail=document.getElementById('cuna-detail');
  if (detail) detail.style.opacity='0';
  setTimeout(()=>{
    if (detail) detail.classList.add('hidden');
    if (menu) {
      menu.style.display='flex';
      menu.style.opacity='0';
      setTimeout(()=>{menu.style.opacity='1';},20);
    }
  },180);
}

// ── Inicio ────────────────────────────────────────────────────────────────────
inicializarRangoFechas();

obtenerGeomAreaEstudio().then(geom => {
  if (geom) {
    try {
      const gjLayer = L.geoJSON({ type: 'Feature', geometry: geom });
      map.fitBounds(gjLayer.getBounds(), { padding: [30, 30] });
    } catch(e) { console.warn('fitBounds:', e); }
  }
});

toggleCapa('estudio', true);
cargarMunicipios();
cargarAlertas();

setInterval(cargarAlertas, 5*60*1000);
setInterval(()=>{ if(deforVisible)cargarDeforestacion(); }, 10*60*1000);
setInterval(()=>{
  Object.keys(gfwCapas).forEach(k=>{
    if(!gfwCapas[k].visible)return;
    if(k!=='hansen')cargarGFWAlertas(k);
  });
}, 15*60*1000);
