// ── Configuración ─────────────────────────────────────────────────────────────
const SUPABASE_URL = 'https://qryktfqnicnwiuwbijvd.supabase.co';
const SUPABASE_KEY = 'sb_publishable_tVzVULeeVvP-QRCXviO7cg_EOuOfX-0';
const BASE_URL     = 'https://raw.githubusercontent.com/vmartinezarias/monitoreo_tfc_cunaguaro/main/';

const GFW_API_KEY  = '6b196681-4bfb-4c71-8757-b745b9290f95';
const GFW_BASE     = 'https://data-api.globalforestwatch.org';
const CHAMEZA_BBOX = { xmin: -72.80, ymin: 4.85, xmax: -72.20, ymax: 5.35 };

const URLS = {
  municipio:    BASE_URL + 'Chameza4326.geojson',
  estudio:      BASE_URL + 'area_estudio.geojson',
  bosque:       BASE_URL + 'Nucleos_boscosos.geojson',
  perdidas:     BASE_URL + 'Perdidas2020-2024.geojson',
  conectividad: BASE_URL + 'cum_currmap_deciles3_4326_web.tif'
};

// ── FIX PRINCIPAL: NO recortar polígono dibujado al Área de Estudio ───────────
// El usuario puede dibujar donde quiera. Los datos de Supabase vienen sin
// filtro geográfico, así que el único filtro puntual es el polígono mismo.
const RECORTAR_DIBUJO_A_AREA_ESTUDIO = false;

// ── Paleta ────────────────────────────────────────────────────────────────────
const PALETA = ['#4caf7d','#f5a623','#e8480a','#378add','#a855f7','#ec4899','#14b8a6','#f97316','#84cc16','#64748b'];
const mapaColores = {};
function obtenerColor(attr, val) {
  if (!mapaColores[attr]) mapaColores[attr] = {};
  const mc = mapaColores[attr];
  if (!mc[val]) mc[val] = PALETA[Object.keys(mc).length % PALETA.length];
  return mc[val];
}

// ── Mapa ──────────────────────────────────────────────────────────────────────
const baseMaps = {
  'Claro (CartoDB)': L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',{attribution:'© CARTO',subdomains:'abcd',maxZoom:19}),
  'Google Maps':     L.tileLayer('https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}',{attribution:'© Google'}),
  'Satélite':        L.tileLayer('https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',{attribution:'© Google'}),
  'Relieve':         L.tileLayer('https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png',{attribution:'© OpenStreetMap'}),
  'Oscuro':          L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',{attribution:'© CARTO',subdomains:'abcd',maxZoom:19})
};
const map = L.map('map',{zoomControl:false,layers:[baseMaps['Claro (CartoDB)']]}).setView([5.09,-72.47],11);
L.control.zoom({position:'bottomright'}).addTo(map);
L.control.layers(baseMaps,null,{position:'bottomright'}).addTo(map);

// ── Estado ────────────────────────────────────────────────────────────────────
let todasAlertas = [], marcadores = [];
let filtroActual = 'all', diasActual = 30;
let modoPeriodo = 'rango';
let fechaInicio = null, fechaFin = null;
const FECHA_MIN_DATA = '2026-01-01';
let areaFiltro = null;
let tabActual = 'incendios', tipoActual = 'incendios';
let geomAreaEstudio = null, geomNucleos = null;
let municipiosGJ = null, veredasGJ = null, municipioActual = '';
let capaMunicipioViz = null, capaVeredasViz = null;
let areaAnalisisActiva = 'estudio';

// Dibujo
let dibujoPoligonoLayer = null, dibujoPoligonoCoords = null;
let dibujoHandler = null, dibujoArea_ha = 0;

const drawnItems   = new L.FeatureGroup().addTo(map);
const capaTodosGeo = new L.FeatureGroup().addTo(map);

const capasInfo = {
  municipio:    L.layerGroup().addTo(map),
  estudio:      L.layerGroup().addTo(map),
  bosque:       L.layerGroup().addTo(map),
  perdidas:     L.layerGroup().addTo(map),
  conectividad: L.layerGroup().addTo(map)
};

// ── FIX: Registro de capas activas + pills ────────────────────────────────────
// Las capas del mapa permanecen activas aunque el panel se colapse.
// Las pills muestran qué capas están encendidas en todo momento.
const CAPAS_LABELS = {
  municipio: 'Municipios', estudio: 'Área estudio',
  bosque: 'Núcleos', perdidas: 'Pérdidas', conectividad: 'Conectividad'
};
const capasActivas = new Set();

function actualizarPills() {
  const cont = document.getElementById('capas-pills');
  if (!cont) return;
  cont.innerHTML = '';
  capasActivas.forEach(k => {
    const p = document.createElement('span');
    p.className = 'capa-pill';
    p.textContent = CAPAS_LABELS[k] || k;
    cont.appendChild(p);
  });
}

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

// ── GFW área de estudio ───────────────────────────────────────────────────────
let areaEstudioGeom = null, areaEstudioCargando = false;

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

// Algoritmo PiP robusto con detección automática de orden de coordenadas
function isPointInPolygon(lat, lng, polygonCoords) {
  let x = _numCoord(lng), y = _numCoord(lat);
  if (x === null || y === null || !Array.isArray(polygonCoords) || polygonCoords.length < 3) return false;
  let pts = [];
  for (const p of polygonCoords) {
    if (!p) continue;
    let plat = null, plng = null;
    if (typeof p.lat === 'number' && typeof p.lng === 'number') {
      plat = p.lat; plng = p.lng;
    } else if (Array.isArray(p) && p.length >= 2) {
      if (Math.abs(p[0]) > Math.abs(p[1])) { plng = p[0]; plat = p[1]; }
      else { plat = p[0]; plng = p[1]; }
    }
    if (plat !== null && plng !== null) pts.push({ lat: plat, lng: plng });
  }
  let inside = false;
  for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
    const xi=pts[i].lng, yi=pts[i].lat, xj=pts[j].lng, yj=pts[j].lat;
    if (((yi > y) !== (yj > y)) && (x < (xj-xi)*(y-yi)/((yj-yi)||1e-12)+xi)) inside = !inside;
  }
  return inside;
}

function puntoEnGeoJSON(lat, lng, geom) {
  if (!geom) return false;
  if (geom.type === 'Feature') return puntoEnGeoJSON(lat, lng, geom.geometry);
  if (geom.type === 'FeatureCollection') {
    if (!Array.isArray(geom.features)) return false;
    for (const f of geom.features) { if (puntoEnGeoJSON(lat, lng, f)) return true; }
    return false;
  }
  if (geom.type === 'GeometryCollection') {
    if (!Array.isArray(geom.geometries)) return false;
    for (const g of geom.geometries) { if (puntoEnGeoJSON(lat, lng, g)) return true; }
    return false;
  }
  if (geom.type === 'Polygon') {
    const rings = geom.coordinates;
    if (!Array.isArray(rings) || !rings.length) return false;
    if (!isPointInPolygon(lat, lng, rings[0])) return false;
    for (let i = 1; i < rings.length; i++) { if (isPointInPolygon(lat, lng, rings[i])) return false; }
    return true;
  }
  if (geom.type === 'MultiPolygon') {
    for (const poly of geom.coordinates) {
      if (!Array.isArray(poly) || !poly.length) continue;
      if (isPointInPolygon(lat, lng, poly[0])) {
        let hole = false;
        for (let i = 1; i < poly.length; i++) { if (isPointInPolygon(lat, lng, poly[i])) { hole = true; break; } }
        if (!hole) return true;
      }
    }
    return false;
  }
  return false;
}

function puntoEnPoligono(lat, lng, feature) { return puntoEnGeoJSON(lat, lng, feature); }

function puntoEnArea(lat, lng, area) {
  lat = _numCoord(lat); lng = _numCoord(lng);
  if (lat === null || lng === null) return false;
  if (!area) return true;
  if (area.tipo === 'circulo') {
    const al=_numCoord(area.lat),aln=_numCoord(area.lng),r=_numCoord(area.radio);
    if (al===null||aln===null||r===null) return true;
    const d=Math.sqrt(Math.pow((lat-al)*111,2)+Math.pow((lng-aln)*111*Math.cos(al*Math.PI/180),2));
    return d<=r;
  }
  if (area.tipo === 'poligono' && Array.isArray(area.coords)) return isPointInPolygon(lat, lng, area.coords);
  return true;
}

async function obtenerGeomAreaEstudio() {
  if (areaEstudioGeom) return areaEstudioGeom;
  if (areaEstudioCargando) {
    await new Promise(r=>{const t=setInterval(()=>{if(!areaEstudioCargando){clearInterval(t);r();}},100);});
    return areaEstudioGeom;
  }
  areaEstudioCargando = true;
  try {
    const resp = await fetch(URLS.estudio);
    const gj   = await resp.json();
    let geom = null;
    if (gj.type==='FeatureCollection'&&gj.features?.length) geom=gj.features[0].geometry;
    else if (gj.type==='Feature') geom=gj.geometry;
    else if (['Polygon','MultiPolygon','GeometryCollection'].includes(gj.type)) geom=gj;
    else if (gj.geometry) geom=gj.geometry;
    areaEstudioGeom = geom || {type:'Polygon',coordinates:[[[-72.80,4.85],[-72.20,4.85],[-72.20,5.35],[-72.80,5.35],[-72.80,4.85]]]};
  } catch(e) {
    areaEstudioGeom = {type:'Polygon',coordinates:[[[-72.85,4.85],[-72.20,4.85],[-72.20,5.35],[-72.85,5.35],[-72.85,4.85]]]};
  }
  areaEstudioCargando = false;
  return areaEstudioGeom;
}

// ── SVG hatch ─────────────────────────────────────────────────────────────────
const svgNS='http://www.w3.org/2000/svg';
const pat=document.createElementNS(svgNS,'pattern');
pat.setAttribute('id','hatch');pat.setAttribute('patternUnits','userSpaceOnUse');
pat.setAttribute('width','8');pat.setAttribute('height','8');pat.setAttribute('patternTransform','rotate(45)');
const ln=document.createElementNS(svgNS,'line');
ln.setAttribute('x1','0');ln.setAttribute('y1','0');ln.setAttribute('x2','0');ln.setAttribute('y2','8');
ln.setAttribute('stroke','#4caf7d');ln.setAttribute('stroke-width','2');
pat.appendChild(ln);
map.on('layeradd',()=>{
  const svg=document.querySelector('#map svg');
  if(svg&&!svg.querySelector('#hatch')){let d=svg.querySelector('defs');if(!d){d=document.createElementNS(svgNS,'defs');svg.insertBefore(d,svg.firstChild);}d.appendChild(pat);}
});

// ── UI ────────────────────────────────────────────────────────────────────────
function cambiarTipo(tipo) {
  tipoActual=tipo;
  document.getElementById('btn-tipo-incendios').classList.toggle('activo',tipo==='incendios');
  document.getElementById('btn-tipo-gfw').classList.toggle('activo',tipo==='gfw');
  const pi=document.getElementById('panel-incendios'),pg=document.getElementById('panel-gfw');
  if(tipo==='incendios'){pi.classList.remove('hidden');pg.classList.add('hidden');}
  else{pg.classList.remove('hidden');pi.classList.add('hidden');}
  if(tipo==='incendios')cambiarTab('incendios',document.getElementById('tab-incendios-btn'));
  else cambiarTab('deforestacion',document.getElementById('tab-defor-btn'));
  const gd=document.getElementById('gfw-dl-btns');
  if(gd){if(tipo==='gfw')gd.classList.remove('hidden');else gd.classList.add('hidden');}
}

// FIX: toggleCapasExtra — colapsar/expandir el panel NO afecta las capas del mapa
function toggleCapasExtra(btn) {
  const body=document.getElementById('capas-extra-body'),ico=document.getElementById('ico-capas');
  const open=!body.classList.contains('hidden');
  if(open){body.classList.add('hidden');ico.textContent='▸';}
  else{body.classList.remove('hidden');ico.textContent='▾';}
  // Las capas del mapa NO se tocan aquí
}

// ── Área de análisis ──────────────────────────────────────────────────────────
let _cambiandoArea = false;

function cambiarAreaAnalisis() {
  if(_cambiandoArea)return;
  _cambiandoArea=true;

  const chkE=document.getElementById('chk-area-estudio');
  const chkN=document.getElementById('chk-area-nucleos');
  const chkM=document.getElementById('chk-area-municipio');
  const chkD=document.getElementById('chk-area-dibujo');
  const previo=areaAnalisisActiva;
  const activos=[chkE,chkN,chkM,chkD].filter(c=>c.checked);

  if(activos.length>1){
    [chkE,chkN,chkM,chkD].forEach(c=>{
      if(c.checked&&(
        (c.id==='chk-area-estudio'  &&previo==='estudio')||
        (c.id==='chk-area-nucleos'  &&previo==='nucleos')||
        (c.id==='chk-area-municipio'&&previo==='municipio')||
        (c.id==='chk-area-dibujo'   &&previo==='dibujo')
      ))c.checked=false;
    });
  }

  const eE=chkE.checked,eN=chkN.checked,eM=chkM.checked,eD=chkD.checked;
  let nueva=null;
  if(eE)nueva='estudio';else if(eN)nueva='nucleos';else if(eM)nueva='municipio';else if(eD)nueva='dibujo';
  if(!nueva){chkE.checked=true;nueva='estudio';}
  areaAnalisisActiva=nueva;

  const wM=document.getElementById('sel-municipio-wrap');
  if(wM){if(eM)wM.classList.remove('hidden');else wM.classList.add('hidden');}
  const sD=document.getElementById('sub-dibujo');
  if(sD){if(eD)sD.classList.remove('hidden');else sD.classList.add('hidden');}

  if(!eM){limpiarCapaMunicipio();municipioActual='';}
  if(!eD&&dibujoPoligonoLayer){
    if(map.hasLayer(dibujoPoligonoLayer))map.removeLayer(dibujoPoligonoLayer);
    if(dibujoHandler){dibujoHandler.disable();dibujoHandler=null;}
    document.getElementById('dibujo-hint').style.display='none';
  }
  if(eD&&dibujoPoligonoLayer&&!map.hasLayer(dibujoPoligonoLayer))map.addLayer(dibujoPoligonoLayer);
  if(eM&&!municipiosGJ)cargarMunicipios();

  document.getElementById('opt-estudio').classList.toggle('activo',eE);
  document.getElementById('opt-nucleos').classList.toggle('activo',eN);
  document.getElementById('opt-municipio').classList.toggle('activo',eM);
  document.getElementById('opt-dibujo').classList.toggle('activo',eD);

  const txt=document.getElementById('area-activa-txt');
  const mLabel=municipioActual?`Municipio: ${municipioActual}`:'Todos los municipios';
  let badge='';
  if(nueva==='estudio')badge='Área de estudio activa';
  else if(nueva==='nucleos')badge='Núcleos boscosos activos';
  else if(nueva==='municipio')badge=mLabel;
  else if(nueva==='dibujo')badge=dibujoPoligonoLayer?`Polígono dibujado · ${dibujoArea_ha.toFixed(1)} ha`:'Polígono pendiente';
  if(txt)txt.textContent=badge;

  if(eE)toggleCapa('estudio',true);else capasInfo.estudio.clearLayers();
  if(eN)toggleCapa('bosque',true); else capasInfo.bosque.clearLayers();
  capaTodosGeo.clearLayers();

  _cambiandoArea=false;
  aplicarFiltros();
  Object.keys(gfwCapas).forEach(k=>{if(gfwCapas[k].visible&&k!=='hansen')cargarGFWAlertas(k);});
}

async function obtenerGeomActiva() {
  if(!areaAnalisisActiva||areaAnalisisActiva==='estudio')return await obtenerGeomAreaEstudio();
  if(areaAnalisisActiva==='nucleos'){
    if(!geomNucleos){
      try{const gj=await(await fetch(URLS.bosque)).json();geomNucleos=gj.features[0].geometry;}
      catch(e){return await obtenerGeomAreaEstudio();}
    }
    return geomNucleos;
  }
  if(areaAnalisisActiva==='municipio'){
    if(!municipioActual)return await obtenerGeomAreaEstudio();
    try{
      if(!veredasGJ)await cargarVeredas();
      const vMun=veredasGJ.features.filter(f=>(f.properties.NOMB_MPIO||'').toUpperCase()===municipioActual.toUpperCase());
      if(vMun.length===1)return vMun[0].geometry;
      if(vMun.length>1)return{type:'GeometryCollection',geometries:vMun.map(f=>f.geometry)};
    }catch(e){console.warn('geom municipio:',e);}
    return await obtenerGeomAreaEstudio();
  }
  if(areaAnalisisActiva==='dibujo'){
    if(dibujoPoligonoCoords&&dibujoPoligonoCoords.length>=3){
      const ring=[];
      for(const p of dibujoPoligonoCoords){
        let plat=null,plng=null;
        if(Array.isArray(p)){if(Math.abs(p[0])>Math.abs(p[1])){plng=p[0];plat=p[1];}else{plat=p[0];plng=p[1];}}
        if(plat!==null&&plng!==null)ring.push([plng,plat]);
      }
      if(ring.length>=3){
        const first=ring[0],last=ring[ring.length-1];
        if(first[0]!==last[0]||first[1]!==last[1])ring.push([first[0],first[1]]);
        return{type:'Polygon',coordinates:[ring]};
      }
    }
    return await obtenerGeomAreaEstudio();
  }
  return await obtenerGeomAreaEstudio();
}

// ── Dibujo polígono ───────────────────────────────────────────────────────────
function iniciarDibujoPoligono() {
  if(dibujoPoligonoLayer){if(map.hasLayer(dibujoPoligonoLayer))map.removeLayer(dibujoPoligonoLayer);dibujoPoligonoLayer=null;dibujoPoligonoCoords=null;dibujoArea_ha=0;}
  if(dibujoHandler){dibujoHandler.disable();dibujoHandler=null;}
  dibujoHandler=new L.Draw.Polygon(map,{shapeOptions:{color:'#7c3aed',weight:2.5,fillColor:'#7c3aed',fillOpacity:0.15,dashArray:'4 4'},allowIntersection:false,showArea:false});
  dibujoHandler.enable();
  document.getElementById('dibujo-hint').style.display='flex';
  const btn=document.getElementById('btn-iniciar-dibujo');if(btn)btn.textContent='⏸ Dibujando… (doble-clic para cerrar)';
  document.getElementById('dibujo-info').textContent='Haz clic en el mapa para añadir vértices';
}

function limpiarDibujoPoligono() {
  if(dibujoPoligonoLayer&&map.hasLayer(dibujoPoligonoLayer))map.removeLayer(dibujoPoligonoLayer);
  dibujoPoligonoLayer=null;dibujoPoligonoCoords=null;dibujoArea_ha=0;
  document.getElementById('dibujo-info').textContent='Sin polígono dibujado';
  document.getElementById('btn-limpiar-dibujo').style.display='none';
  document.getElementById('btn-iniciar-dibujo').textContent='✏️ Empezar a dibujar';
  aplicarFiltros();
  Object.keys(gfwCapas).forEach(k=>{if(gfwCapas[k].visible&&k!=='hansen')cargarGFWAlertas(k);});
  const txt=document.getElementById('area-activa-txt');if(txt&&areaAnalisisActiva==='dibujo')txt.textContent='Polígono pendiente';
}

function calcularAreaPoligonoHa(coords) {
  if(!coords||coords.length<3)return 0;
  const R=6378137;let area=0;const n=coords.length;
  for(let i=0;i<n;i++){const[lat1,lng1]=coords[i],[lat2,lng2]=coords[(i+1)%n];area+=(lng2-lng1)*Math.PI/180*(2+Math.sin(lat1*Math.PI/180)+Math.sin(lat2*Math.PI/180));}
  return Math.abs(area*R*R/2)/10000;
}

map.on(L.Draw.Event.CREATED,function(e){
  if(!e||e.layerType!=='polygon')return;
  if(areaAnalisisActiva!=='dibujo')return;
  const layer=e.layer;
  let latlngs=layer.getLatLngs();
  while(latlngs.length>0&&Array.isArray(latlngs[0]))latlngs=latlngs[0];
  if(!latlngs||latlngs.length<3){alert('El polígono debe tener al menos 3 vértices.');return;}
  dibujoPoligonoCoords=latlngs.map(p=>[p.lat,p.lng]);
  const first=dibujoPoligonoCoords[0],last=dibujoPoligonoCoords[dibujoPoligonoCoords.length-1];
  if(first[0]!==last[0]||first[1]!==last[1])dibujoPoligonoCoords.push([first[0],first[1]]);
  dibujoArea_ha=calcularAreaPoligonoHa(dibujoPoligonoCoords);
  dibujoPoligonoLayer=L.polygon(latlngs,{color:'#7c3aed',weight:2.5,fillColor:'#7c3aed',fillOpacity:0.12,dashArray:'4 4'})
    .bindTooltip(`Área personalizada · ${dibujoArea_ha.toFixed(1)} ha`,{sticky:true});
  map.addLayer(dibujoPoligonoLayer);
  if(dibujoHandler){dibujoHandler.disable();dibujoHandler=null;}
  document.getElementById('dibujo-hint').style.display='none';
  document.getElementById('btn-iniciar-dibujo').textContent='✏️ Empezar a dibujar';
  document.getElementById('btn-limpiar-dibujo').style.display='block';
  // FIX: solo mostrar el área, sin advertencia de intersección con Área de Estudio
  document.getElementById('dibujo-info').innerHTML=`✓ Polígono · <span class="mono text-violet-600">${dibujoArea_ha.toFixed(1)} ha</span>`;
  const txt=document.getElementById('area-activa-txt');if(txt)txt.textContent=`Polígono dibujado · ${dibujoArea_ha.toFixed(1)} ha`;
  aplicarFiltros();
  Object.keys(gfwCapas).forEach(k=>{if(gfwCapas[k].visible&&k!=='hansen')cargarGFWAlertas(k);});
});

// ── Filtros ───────────────────────────────────────────────────────────────────
// FIX PRINCIPAL: pasaFiltroAreaDibujo solo usa el polígono dibujado.
// NO intersecta con Área de Estudio (RECORTAR_DIBUJO_A_AREA_ESTUDIO=false).
function pasaFiltroAreaDibujo(lat, lng) {
  if(areaAnalisisActiva!=='dibujo')return true;
  if(!dibujoPoligonoCoords||dibujoPoligonoCoords.length<3)return true;
  lat=_numCoord(lat);lng=_numCoord(lng);
  if(lat===null||lng===null)return false;
  const insideDrawn=isPointInPolygon(lat,lng,dibujoPoligonoCoords);
  if(!insideDrawn)return false;
  if(RECORTAR_DIBUJO_A_AREA_ESTUDIO&&areaEstudioGeom&&typeof areaEstudioGeom==='object'){
    if(!puntoEnGeoJSON(lat,lng,areaEstudioGeom))return false;
  }
  return true;
}

function alertasEnPeriodo(arr) {
  if(modoPeriodo==='rango'&&fechaInicio&&fechaFin)
    return arr.filter(a=>{const f=new Date(a.fecha_deteccion);return f>=fechaInicio&&f<=fechaFin;});
  const ini=new Date(Date.now()-diasActual*864e5);
  return arr.filter(a=>new Date(a.fecha_deteccion)>=ini);
}

function aplicarFiltros() {
  let firms=alertasEnPeriodo(todasAlertas);
  if(filtroActual!=='all')firms=firms.filter(a=>(a.firms_confidence||'').toLowerCase()===filtroActual);
  const aplicaFiltroDibujo=(areaAnalisisActiva==='dibujo'&&dibujoPoligonoCoords&&dibujoPoligonoCoords.length>=3);
  const firmsArea=aplicaFiltroDibujo?firms.filter(a=>{const p=_alertaLatLng(a);return p?pasaFiltroAreaDibujo(p.lat,p.lng):false;}):firms;
  const deforPeriodo=alertasEnPeriodo(alertasDefor);
  const deforArea=aplicaFiltroDibujo?deforPeriodo.filter(a=>{const p=_alertaLatLng(a);return p?pasaFiltroAreaDibujo(p.lat,p.lng):false;}):deforPeriodo;
  actualizarStats(firms,firmsArea,deforArea);
  renderMarcadores(firmsArea);
  if(tabActual==='incendios')renderLista(firmsArea,'incendios');else renderLista(deforArea,'deforestacion');
  setTimeout(filtrarDeforestacion,0);
  Object.keys(gfwCapas).forEach(k=>{if(gfwCapas[k].visible&&k!=='hansen')filtrarGFW(k);});
}

function actualizarStats(firms,firmsArea,deforArea) {
  const aplicaFiltroDibujo=(areaAnalisisActiva==='dibujo'&&dibujoPoligonoCoords);
  document.getElementById('total-alertas').textContent=aplicaFiltroDibujo?firmsArea.length:firms.length;
  document.getElementById('total-defor').textContent=deforArea.length;
  document.getElementById('total-high').textContent=firms.filter(a=>(a.firms_confidence||'').toLowerCase()==='high').length;
  const frps=firms.map(a=>a.firms_frp||0).filter(Boolean);
  document.getElementById('total-frp').textContent=frps.length?Math.max(...frps).toFixed(0):'0';
}

function colorConfianza(c) {
  if(!c)return'#f5a623';const cl=c.toLowerCase();
  if(cl==='high')return'#e8480a';if(cl==='nominal')return'#f5a623';return'#4caf7d';
}

function renderMarcadores(alertas) {
  marcadores.forEach(m=>map.removeLayer(m));marcadores=[];
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
  ['tab-incendios-btn','tab-defor-btn'].forEach(id=>{const el=document.getElementById(id);if(el){el.classList.remove('tab-act');el.classList.add('tab-inact');}});
  if(btn){btn.classList.remove('tab-inact');btn.classList.add('tab-act');}
  aplicarFiltros();
}

function renderLista(alertas,tipo) {
  const el=document.getElementById('lista-alertas');
  if(!alertas.length){el.innerHTML='<div class="flex items-center justify-center py-10 text-slate-400 text-xs italic">Sin alertas en este período</div>';return;}
  if(tipo==='incendios'){
    el.innerHTML='<div class="divide-y divide-slate-50">'+alertas.map((a,i)=>{
      const conf=(a.firms_confidence||'low').toLowerCase();
      const fecha=new Date(a.fecha_deteccion).toLocaleString('es-CO',{month:'short',day:'numeric',hour:'2-digit',minute:'2-digit',timeZone:'America/Bogota'});
      const dot=conf==='high'?'bg-red-500 shadow-red-200':conf==='nominal'?'bg-amber-400':'bg-emerald-400';
      return`<div class="flex items-center gap-3 px-4 py-3 hover:bg-emerald-50/50 cursor-pointer transition-colors" onclick="centrarEn(${i})"><div class="w-2 h-2 rounded-full ${dot} shadow-sm shrink-0"></div><div class="flex-1 min-w-0"><div class="mono text-[10px] text-slate-700 font-bold">${a.latitud.toFixed(4)}, ${a.longitud.toFixed(4)}</div><div class="text-[9px] text-slate-400 mt-0.5">${fecha} · ${a.firms_satellite||'VIIRS'}</div></div><div class="mono text-[10px] text-red-500 font-bold shrink-0">${a.firms_frp?a.firms_frp.toFixed(0)+' MW':'—'}</div></div>`;
    }).join('')+'</div>';
  }else{
    el.innerHTML='<div class="divide-y divide-slate-50">'+alertas.map((a,i)=>{
      const fecha=new Date(a.fecha_deteccion+'T12:00:00').toLocaleDateString('es-CO');
      const dot=a.severidad==='severa'?'bg-red-500':a.severidad==='moderada'?'bg-orange-400':'bg-yellow-400';
      return`<div class="flex items-center gap-3 px-4 py-3 hover:bg-emerald-50/50 cursor-pointer transition-colors" onclick="centrarEnDefor(${i})"><div class="w-2 h-2 rounded-sm ${dot} shrink-0"></div><div class="flex-1 min-w-0"><div class="mono text-[10px] text-slate-700 font-bold">${a.latitud.toFixed(4)}, ${a.longitud.toFixed(4)}</div><div class="text-[9px] text-slate-400 mt-0.5">${fecha} · ${a.predio_id}</div></div><div class="mono text-[10px] text-amber-500 font-bold shrink-0">${a.area_afectada_ha||'?'} ha</div></div>`;
    }).join('')+'</div>';
  }
}

function centrarEn(i){if(marcadores[i]){map.setView(marcadores[i].getLatLng(),13);marcadores[i].openPopup();}}
function centrarEnDefor(i){const a=alertasDefor[i];if(a)map.setView([a.latitud,a.longitud],13);}

function cambiarPeriodo(btn) {
  document.querySelectorAll('.btn-per').forEach(b=>b.classList.remove('activo'));
  btn.classList.add('activo');diasActual=parseInt(btn.dataset.dias);
  modoPeriodo='preset';
  const blq=document.getElementById('modo-rango');if(blq){blq.classList.add('rango-inactivo');blq.classList.remove('rango-activo');}
  cargarAlertas();if(deforVisible)cargarDeforestacion();else aplicarFiltros();
  Object.keys(gfwCapas).forEach(k=>{if(gfwCapas[k].visible&&k!=='hansen')cargarGFWAlertas(k);});
}

function cambiarRangoFechas() {
  const inpI=document.getElementById('fecha-inicio'),inpF=document.getElementById('fecha-fin');
  if(!inpI||!inpF)return;
  const sI=inpI.value,sF=inpF.value;if(!sI||!sF)return;
  if(sI<FECHA_MIN_DATA){inpI.value=FECHA_MIN_DATA;return cambiarRangoFechas();}
  if(sI>sF){inpF.value=sI;return cambiarRangoFechas();}
  fechaInicio=new Date(sI+'T00:00:00');fechaFin=new Date(sF+'T23:59:59');
  diasActual=Math.max(1,Math.ceil((new Date()-fechaInicio)/864e5));
  modoPeriodo='rango';
  document.querySelectorAll('.btn-per').forEach(b=>b.classList.remove('activo'));
  const blq=document.getElementById('modo-rango');if(blq){blq.classList.add('rango-activo');blq.classList.remove('rango-inactivo');}
  const dias=Math.ceil((fechaFin-fechaInicio)/864e5);
  const info=document.getElementById('rango-info');if(info)info.textContent=`${dias} día${dias!==1?'s':''} seleccionado${dias!==1?'s':''}`;
  cargarAlertas();if(deforVisible)cargarDeforestacion();else aplicarFiltros();
  Object.keys(gfwCapas).forEach(k=>{if(gfwCapas[k].visible&&k!=='hansen')cargarGFWAlertas(k);});
}

function inicializarRangoFechas() {
  const inpI=document.getElementById('fecha-inicio'),inpF=document.getElementById('fecha-fin');if(!inpI||!inpF)return;
  const hoy=new Date(),hoyStr=hoy.toISOString().slice(0,10);
  let ini=new Date(Date.now()-30*864e5),iniStr=ini.toISOString().slice(0,10);
  if(iniStr<FECHA_MIN_DATA)iniStr=FECHA_MIN_DATA;
  inpI.max=hoyStr;inpF.max=hoyStr;inpI.value=iniStr;inpF.value=hoyStr;
  cambiarRangoFechas();
}

function filtrar(btn) {
  document.querySelectorAll('.filtro-btn').forEach(b=>{b.classList.remove('activo','bg-red-500','text-white','border-red-400');b.classList.add('bg-white','text-slate-600','border-slate-200');});
  btn.classList.add('activo','bg-red-500','text-white','border-red-400');btn.classList.remove('bg-white','text-slate-600','border-slate-200');
  filtroActual=btn.dataset.filtro;aplicarFiltros();
}

// ── Descarga ──────────────────────────────────────────────────────────────────
async function descargarActual(formato) {
  if(tipoActual==='incendios'){if(formato==='csv')descargarCSV('incendios');else descargarPDF('incendios');}
  else{const a=Object.keys(gfwCapas).filter(k=>gfwCapas[k].visible&&k!=='hansen'&&gfwCapas[k].datos.length);if(a.length)a.forEach(k=>descargarGFW(k));else{if(formato==='csv')descargarCSV('deforestacion');else descargarPDF('deforestacion');}}
}
function descargarGFWActivo(){const a=Object.keys(gfwCapas).filter(k=>gfwCapas[k].visible&&k!=='hansen'&&gfwCapas[k].datos.length);if(!a.length){alert('Activa primero una capa GFW (GLAD o RADD) y espera que cargue.');return;}a.forEach(k=>descargarGFW(k));}

// ── Descarga ──────────────────────────────────────────────────────────────────
async function descargarActual(formato) {
  if(tipoActual==='incendios'){
    if(formato==='csv')descargarCSV('incendios');
    else descargarPDF('incendios');
  }
  else{
    const a=Object.keys(gfwCapas).filter(k=>gfwCapas[k].visible&&k!=='hansen'&&gfwCapas[k].datos.length);
    if(a.length)a.forEach(k=>descargarGFW(k));
    else{
      if(formato==='csv')descargarCSV('deforestacion');
      else descargarPDF('deforestacion');
    }
  }
}

function descargarGFWActivo(){
  const a=Object.keys(gfwCapas).filter(k=>gfwCapas[k].visible&&k!=='hansen'&&gfwCapas[k].datos.length);
  if(!a.length){
    alert('Activa primero una capa GFW (GLAD o RADD) y espera que cargue.');
    return;
  }
  a.forEach(k=>descargarGFW(k));
}

// funcion conectividad
async function asegurarRasterConectividad() {
  if (georasterConectividad) return georasterConectividad;

  try {
    const resp = await fetch(URLS.conectividad);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

    const buffer = await resp.arrayBuffer();
    georasterConectividad = await parseGeoraster(buffer);

    return georasterConectividad;
  } catch (e) {
    console.warn('No fue posible cargar el ráster de conectividad:', e);
    return null;
  }
}

function muestrearConectividad(lat, lng, raster) {
  if (!raster) return null;

  const x = Number(lng);
  const y = Number(lat);

  if (!Number.isFinite(x) || !Number.isFinite(y)) return null;

  const col = Math.floor((x - raster.xmin) / raster.pixelWidth);
  const row = Math.floor((raster.ymax - y) / raster.pixelHeight);

  const banda = raster.values?.[0];

  if (!banda || row < 0 || col < 0 || row >= banda.length || col >= banda[0].length) {
    return null;
  }

  const valor = banda[row][col];

  if (
    valor === null ||
    valor === undefined ||
    Number.isNaN(Number(valor)) ||
    valor === raster.noDataValue
  ) {
    return null;
  }

  const decil = Math.round(Number(valor));

  if (decil < 1 || decil > 10) return null;

  return decil;
}

function clasificarConectividad(decil) {
  if (!Number.isFinite(decil)) return 'Sin dato';

  if (decil === 10) return 'Crítica';
  if (decil >= 7) return 'Alta';
  if (decil >= 4) return 'Media';
  if (decil >= 1) return 'Baja';

  return 'Sin dato';
}

// 👇 AQUÍ VA LA NUEVA FUNCIÓN
async function abrirAnalisisGFW() {
  const capas = ['glad', 'radd'];
  const AREA_HA_POR_PIXEL = 0.09;

  const datosCrudos = capas.flatMap(key => {
    const rows = gfwCapas[key]?.datos || [];

    return rows.map(r => ({
      ...r,
      fuente: key.toUpperCase()
    }));
  });

  if (!datosCrudos.length) {
    alert('Activa primero GLAD o RADD y espera a que carguen los datos.');
    return;
  }

  // Cargar veredas si todavía no están disponibles
  try {
    if (!veredasGJ) await cargarVeredas();
  } catch (e) {
    console.warn('No fue posible cargar veredas para el análisis:', e);
  }

  function ubicarPuntoEnVereda(lat, lng) {
    if (!veredasGJ || !Array.isArray(veredasGJ.features)) {
      return {
        municipio: municipioActual || 'Sin municipio',
        vereda: 'Sin vereda'
      };
    }

    const vf = veredasGJ.features.find(f => puntoEnPoligono(lat, lng, f));

    if (!vf) {
      return {
        municipio: municipioActual || 'Fuera de límites',
        vereda: 'Fuera de límites'
      };
    }

    return {
      municipio: vf.properties.NOMB_MPIO || municipioActual || 'Sin municipio',
      vereda: vf.properties.NOMBRE_VER || 'Sin vereda'
    };
  }
  const rasterConectividad = await asegurarRasterConectividad();
  const datos = datosCrudos.map(r => {
    const lat = parseFloat(r.latitude ?? r.latitud ?? r.lat);
    const lng = parseFloat(r.longitude ?? r.longitud ?? r.lng ?? r.lon);
    const decilConectividad = muestrearConectividad(lat, lng, rasterConectividad);
    const ubicacion = Number.isFinite(lat) && Number.isFinite(lng)
      ? ubicarPuntoEnVereda(lat, lng)
      : {
          municipio: municipioActual || 'Sin municipio',
          vereda: 'Sin vereda'
        };

    return {
      ...r,
      latitude: lat,
      longitude: lng,
      municipio: ubicacion.municipio,
      vereda: ubicacion.vereda,
      area_ha: AREA_HA_POR_PIXEL,
      conectividad_decile: decilConectividad,
      conectividad_clase: clasificarConectividad(decilConectividad)
    };
  });

  const paquete = {
    generado_en: new Date().toISOString(),
    area: areaAnalisisActiva,
    municipio: municipioActual || null,
    fecha_inicio: fechaInicio ? fechaInicio.toISOString().slice(0, 10) : null,
    fecha_fin: fechaFin ? fechaFin.toISOString().slice(0, 10) : null,
    datos
  };

  sessionStorage.setItem('gfw_analisis_datos', JSON.stringify(paquete));
  window.open('analisis.html', '_blank');
}
function alertasActualesIncendios() {
  let r=alertasEnPeriodo(todasAlertas);
  if(filtroActual!=='all')r=r.filter(a=>(a.firms_confidence||'').toLowerCase()===filtroActual);
  if(areaAnalisisActiva==='dibujo'&&dibujoPoligonoCoords&&dibujoPoligonoCoords.length>=3)
    r=r.filter(a=>{const p=_alertaLatLng(a);return p?pasaFiltroAreaDibujo(p.lat,p.lng):false;});
  return r;
}

function alertasActualesIncendios() {
  let r=alertasEnPeriodo(todasAlertas);
  if(filtroActual!=='all')r=r.filter(a=>(a.firms_confidence||'').toLowerCase()===filtroActual);
  if(areaAnalisisActiva==='dibujo'&&dibujoPoligonoCoords&&dibujoPoligonoCoords.length>=3)
    r=r.filter(a=>{const p=_alertaLatLng(a);return p?pasaFiltroAreaDibujo(p.lat,p.lng):false;});
  return r;
}
function alertasActualesDefor() {
  let r=alertasEnPeriodo(alertasDefor);
  if(areaAnalisisActiva==='dibujo'&&dibujoPoligonoCoords&&dibujoPoligonoCoords.length>=3)
    r=r.filter(a=>{const p=_alertaLatLng(a);return p?pasaFiltroAreaDibujo(p.lat,p.lng):false;});
  return r;
}

async function descargarCSV(modo) {
  const hoy=new Date().toISOString().slice(0,10);
  const esc=v=>{const s=String(v??'');return s.includes(',')||s.includes('"')||s.includes('\n')?'"'+s.replace(/"/g,'""')+'"':s;};
  function vDA(lat,lng,vM){for(const vf of vM){if(puntoEnPoligono(lat,lng,vf))return vf.properties.NOMBRE_VER||'Sin nombre';}return'Sin vereda';}
  const fi=alertasActualesIncendios(),fd=alertasActualesDefor();
  if(!veredasGJ)await cargarVeredas();

  if(modo==='incendios'){
    if(!fi.length){alert('No hay alertas de incendio para exportar.');return;}
    if(municipioActual&&veredasGJ){
      const vM=veredasGJ.features.filter(f=>(f.properties.NOMB_MPIO||'').toUpperCase()===municipioActual.toUpperCase());
      const cab=['vereda','municipio','fecha_deteccion','latitud','longitud','confianza','frp_mw','satelite','estado'].join(',');
      const rows=fi.map(a=>[esc(vDA(a.latitud,a.longitud,vM)),esc(municipioActual),esc(a.fecha_deteccion),a.latitud,a.longitud,esc(a.firms_confidence||''),a.firms_frp||'',esc(a.firms_satellite||''),esc(a.estado)].join(','));
      _dl([cab,...rows].join('\n'),`incendios_${municipioActual.replace(/ /g,'_')}_${hoy}.csv`);
    }else{
      if(!municipiosGJ)await cargarMunicipios();
      function mpA(lat,lng){if(!veredasGJ)return'Sin datos';const vf=veredasGJ.features.find(f=>puntoEnPoligono(lat,lng,f));return vf?(vf.properties.NOMB_MPIO||'Sin nombre'):'Fuera de límites';}
      const cab=['municipio','fecha_deteccion','latitud','longitud','confianza','frp_mw','satelite','estado'].join(',');
      const rows=fi.map(a=>[esc(mpA(a.latitud,a.longitud)),esc(a.fecha_deteccion),a.latitud,a.longitud,esc(a.firms_confidence||''),a.firms_frp||'',esc(a.firms_satellite||''),esc(a.estado)].join(','));
      const cnt={};fi.forEach(a=>{const m=mpA(a.latitud,a.longitud);cnt[m]=(cnt[m]||0)+1;});
      const cabR=['municipio','n_incendios'].join(',');
      const rowsR=Object.entries(cnt).sort((a,b)=>b[1]-a[1]).map(([m,n])=>[esc(m),n].join(','));
      _dl(['## DETALLE POR ALERTA',cab,...rows,'','## RESUMEN POR MUNICIPIO',cabR,...rowsR].join('\n'),`incendios_area_estudio_${hoy}.csv`);
    }
  }else if(modo==='deforestacion'){
    if(!fd.length){alert('No hay alertas de deforestación para exportar.');return;}
    if(municipioActual&&veredasGJ){
      const vM=veredasGJ.features.filter(f=>(f.properties.NOMB_MPIO||'').toUpperCase()===municipioActual.toUpperCase());
      const cab=['vereda','municipio','fecha_deteccion','predio_id','latitud','longitud','severidad','area_ha','cambio_ndvi','estado'].join(',');
      const rows=fd.map(a=>[esc(vDA(a.latitud,a.longitud,vM)),esc(municipioActual),esc(a.fecha_deteccion),esc(a.predio_id||''),a.latitud,a.longitud,esc(a.severidad||''),a.area_afectada_ha||'',a.cambio_ndvi!=null?Number(a.cambio_ndvi).toFixed(3):'',esc(a.estado)].join(','));
      _dl([cab,...rows].join('\n'),`deforestacion_${municipioActual.replace(/ /g,'_')}_${hoy}.csv`);
    }else{
      if(!municipiosGJ)await cargarMunicipios();
      function mpD(lat,lng){if(!veredasGJ)return'Sin datos';const vf=veredasGJ.features.find(f=>puntoEnPoligono(lat,lng,f));return vf?(vf.properties.NOMB_MPIO||'Sin nombre'):'Fuera de límites';}
      const cab=['municipio','fecha_deteccion','predio_id','latitud','longitud','severidad','area_ha','cambio_ndvi','estado'].join(',');
      const rows=fd.map(a=>[esc(mpD(a.latitud,a.longitud)),esc(a.fecha_deteccion),esc(a.predio_id||''),a.latitud,a.longitud,esc(a.severidad||''),a.area_afectada_ha||'',a.cambio_ndvi!=null?Number(a.cambio_ndvi).toFixed(3):'',esc(a.estado)].join(','));
      const cnt={};fd.forEach(a=>{const m=mpD(a.latitud,a.longitud);cnt[m]=(cnt[m]||0)+1;});
      const cabR=['municipio','n_deforestacion'].join(',');
      const rowsR=Object.entries(cnt).sort((a,b)=>b[1]-a[1]).map(([m,n])=>[esc(m),n].join(','));
      _dl(['## DETALLE POR ALERTA',cab,...rows,'','## RESUMEN POR MUNICIPIO',cabR,...rowsR].join('\n'),`deforestacion_area_estudio_${hoy}.csv`);
    }
  }else if(modo==='combinado'){
    if(!fi.length&&!fd.length){alert('No hay datos para exportar.');return;}
    if(fi.length){descargarCSV('incendios');await new Promise(r=>setTimeout(r,400));}
    if(fd.length)descargarCSV('deforestacion');
  }
}

function _dl(content,filename){
  const blob=new Blob([content],{type:'text/csv;charset=utf-8;'});
  const url=URL.createObjectURL(blob);
  const a=document.createElement('a');a.href=url;a.download=filename;
  document.body.appendChild(a);a.click();document.body.removeChild(a);URL.revokeObjectURL(url);
}

function descargarPDF(modo) {
  const{jsPDF}=window.jspdf;const doc=new jsPDF();
  const hoy=new Date().toLocaleDateString('es-CO',{timeZone:'America/Bogota'});
  const periodo=diasActual===1?'Últimas 24 h':diasActual===7?'Últimos 7 días':'Últimos 30 días';
  const areaLbl=areaAnalisisActiva==='estudio'?'Área de estudio':areaAnalisisActiva==='nucleos'?'Núcleos boscosos':areaAnalisisActiva==='municipio'?(municipioActual?`Municipio: ${municipioActual}`:'Todos los municipios'):areaAnalisisActiva==='dibujo'?`Polígono personalizado (${dibujoArea_ha.toFixed(1)} ha)`:'Área completa';
  doc.setFillColor(26,74,46);doc.rect(0,0,210,36,'F');
  doc.setTextColor(76,175,125);doc.setFontSize(15);doc.setFont('helvetica','bold');
  doc.text(modo==='incendios'?'Reporte de Incendios':modo==='deforestacion'?'Reporte de Deforestación':'Reporte Combinado',14,14);
  doc.setFontSize(9);doc.setFont('helvetica','normal');doc.setTextColor(122,171,138);
  doc.text(`Chámeza, Casanare · ${hoy} · ${periodo} · ${areaLbl}`,14,24);
  doc.setTextColor(180,220,195);doc.text('Fuente: NASA FIRMS + GEE · CBC Cunaguaro / TFCA Colombia',14,32);
  let y=44;
  if(modo!=='deforestacion'){
    const fi=alertasActualesIncendios();
    doc.setFillColor(30,53,40);doc.rect(0,y,210,9,'F');
    doc.setTextColor(255,112,67);doc.setFontSize(11);doc.setFont('helvetica','bold');
    doc.text(`INCENDIOS (${fi.length})`,14,y+6);y+=13;
    if(!fi.length){doc.setFont('helvetica','italic');doc.setTextColor(122,171,138);doc.setFontSize(9);doc.text('Sin alertas en este período.',14,y);y+=10;}
    else{
      doc.setFontSize(8);doc.setFont('helvetica','bold');doc.setFillColor(220,240,228);doc.rect(14,y,182,7,'F');doc.setTextColor(40,40,40);
      doc.text('Fecha',16,y+5);doc.text('Lat',68,y+5);doc.text('Lon',92,y+5);doc.text('Confianza',116,y+5);doc.text('FRP (MW)',146,y+5);doc.text('Satélite',172,y+5);
      y+=9;doc.setFont('helvetica','normal');
      fi.slice(0,40).forEach((a,i)=>{
        if(y>272){doc.addPage();y=20;}
        if(i%2===0){doc.setFillColor(248,252,249);doc.rect(14,y-4,182,7,'F');}
        const f=new Date(a.fecha_deteccion).toLocaleString('es-CO',{timeZone:'America/Bogota',month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'});
        doc.setTextColor(40,40,40);
        doc.text(f,16,y);doc.text(a.latitud.toFixed(4),68,y);doc.text(a.longitud.toFixed(4),92,y);
        doc.text(a.firms_confidence||'—',116,y);doc.text(a.firms_frp?a.firms_frp.toFixed(1):'—',146,y);doc.text(a.firms_satellite||'—',172,y);y+=7;
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
      doc.setFontSize(9);doc.setFont('helvetica','normal');doc.setTextColor(40,40,40);
      doc.text(`Severa: ${sev.severa} · Moderada: ${sev.moderada} · Leve: ${sev.leve}`,14,y);y+=8;
      doc.setFontSize(8);doc.setFont('helvetica','bold');doc.setFillColor(220,240,228);doc.rect(14,y,182,7,'F');doc.setTextColor(40,40,40);
      doc.text('Fecha',16,y+5);doc.text('Predio',54,y+5);doc.text('Lat',110,y+5);doc.text('ΔNDVI',128,y+5);doc.text('Ha',148,y+5);doc.text('Severidad',172,y+5);
      y+=9;doc.setFont('helvetica','normal');
      fd.slice(0,40).forEach((a,i)=>{
        if(y>272){doc.addPage();y=20;}
        if(i%2===0){doc.setFillColor(248,252,249);doc.rect(14,y-4,182,7,'F');}
        const f=new Date(a.fecha_deteccion+'T12:00:00').toLocaleDateString('es-CO');
        const sc={'severa':[220,38,38],'moderada':[249,115,22],'leve':[234,179,8]}[a.severidad]||[40,40,40];
        doc.setTextColor(40,40,40);
        doc.text(f,16,y);doc.text((a.predio_id||'').slice(0,22),54,y);doc.text(a.latitud?.toFixed(4)||'—',110,y);
        doc.text(a.cambio_ndvi!=null?Number(a.cambio_ndvi).toFixed(3):'—',128,y);doc.text(a.area_afectada_ha!=null?String(a.area_afectada_ha):'—',148,y);
        doc.setTextColor(...sc);doc.text(a.severidad||'—',172,y);y+=7;
      });
    }
  }
  doc.setFontSize(7);doc.setTextColor(120,120,120);doc.setFont('helvetica','normal');
  doc.text('CBC Cunaguaro · TFCA Colombia · datos: NASA FIRMS + GEE Sentinel-2',14,290);
  doc.save(`reporte_${modo}_chameza_${new Date().toISOString().slice(0,10)}.pdf`);
}

// ── Capas informativas ────────────────────────────────────────────────────────
// FIX: toggleCapa registra/quita del Set y actualiza las pills
async function toggleCapa(key, visible) {
  const group = capasInfo[key];
  if (!visible) {
    group.clearLayers();
    capasActivas.delete(key);
    actualizarPills();
    const sub = document.getElementById(`sub-${key}`);
    if (sub) sub.classList.remove('visible');
    return;
  }
  const sub = document.getElementById(`sub-${key}`);
  if (sub) sub.classList.add('visible');
  capasActivas.add(key);
  actualizarPills();
  try {
    if (key === 'conectividad') {
      if (!georasterConectividad) { const r=await fetch(URLS.conectividad); georasterConectividad=await parseGeoraster(await r.arrayBuffer()); }
      renderRasterConectividad(); return;
    }
    const data = await (await fetch(URLS[key])).json();
    if (key === 'perdidas') { datosPerdidas = data; actualizarPerdidas(); return; }
    let style = {};
    if (key === 'municipio') style = {color:'#333',weight:2,fillOpacity:0};
    if (key === 'estudio')   style = {color:'#2563eb',weight:2.5,fillOpacity:0.04,dashArray:'6 4'};
    if (key === 'bosque')    style = {color:'#2d7a4f',weight:1,fillOpacity:0.5};
    const layer = L.geoJSON(data,{style,onEachFeature:(f,l)=>{if(key==='bosque')l.on('add',()=>{if(l._path)l._path.setAttribute('fill','url(#hatch)');});}});
    group.addLayer(layer);
  } catch(e) { console.error(`Error capa ${key}:`, e); }
}

function actualizarPerdidas() {
  if (!datosPerdidas) return;
  const group=capasInfo.perdidas; group.clearLayers();
  const ys=document.getElementById('sel-year').value;
  const cy={'20':'#eab308','21':'#f97316','22':'#ef4444','23':'#ec4899','24':'#a855f7'};
  L.geoJSON(datosPerdidas,{
    filter:f=>ys==='all'||String(f.properties.year)===ys,
    pointToLayer:(f,ll)=>L.circleMarker(ll,{radius:4,fillColor:cy[String(f.properties.year)]||'#fff',color:'#000',weight:1,fillOpacity:0.8}).bindPopup(`Pérdida · Año: 20${f.properties.year}`)
  }).addTo(group);
}

function actualizarConectividad(val) {
  cuantilMinimo=parseInt(val);document.getElementById('val-cuantil').textContent=val;
  if(capasInfo.conectividad.getLayers().length>0)renderRasterConectividad();
}
function renderRasterConectividad() {
  if (!georasterConectividad) return;
  const g=capasInfo.conectividad; g.clearLayers();
  g.addLayer(new GeoRasterLayer({georaster:georasterConectividad,opacity:0.7,pixelValuesToColorFn:v=>{const n=Math.round(v[0]);if(n<cuantilMinimo||n>10||isNaN(n))return null;return turboColors[n];},resolution:256}));
}

// ── Municipios y Veredas ──────────────────────────────────────────────────────
async function cargarMunicipios() {
  if (veredasGJ) { municipiosGJ=veredasGJ; return veredasGJ; }
  await cargarVeredas();
  municipiosGJ = veredasGJ;
  const sel = document.getElementById('sel-municipio');
  if (sel && veredasGJ) {
    sel.innerHTML = '<option value="">— Todos los municipios —</option>';
    const nombres=[...new Set((veredasGJ.features||[]).map(f=>(f.properties.NOMB_MPIO||'').trim()).filter(Boolean))].sort();
    nombres.forEach(nom=>{const o=document.createElement('option');o.value=nom;o.textContent=nom;sel.appendChild(o);});
  }
  return veredasGJ;
}

async function cargarVeredas() {
  if (veredasGJ) return veredasGJ;
  try { const resp=await fetch(BASE_URL+'veredas.geojson'); veredasGJ=await resp.json(); return veredasGJ; }
  catch(e) { console.error('Error cargando veredas:',e); return null; }
}

function limpiarCapaMunicipio() {
  if(capaMunicipioViz){map.removeLayer(capaMunicipioViz);capaMunicipioViz=null;}
  if(capaVeredasViz){map.removeLayer(capaVeredasViz);capaVeredasViz=null;}
}

async function seleccionarMunicipio(nombre) {
  municipioActual=nombre; limpiarCapaMunicipio();
  const info=document.getElementById('municipio-veredas-info');
  if(!nombre){if(info)info.textContent='';cambiarAreaAnalisis();aplicarFiltros();return;}
  await cargarVeredas();
  if(veredasGJ){
    const vFeat=veredasGJ.features.filter(f=>(f.properties.NOMB_MPIO||'').toUpperCase()===nombre.toUpperCase());
    if(vFeat.length>0){
      capaVeredasViz=L.geoJSON({type:'FeatureCollection',features:vFeat},{
        style:{color:'#7c3aed',weight:1,fillColor:'#7c3aed',fillOpacity:0.04,dashArray:'3 3'},
        onEachFeature:(f,l)=>l.bindTooltip(f.properties.NOMBRE_VER||'',{permanent:false})
      }).addTo(map);
      if(info)info.textContent=`${vFeat.length} vereda${vFeat.length!==1?'s':''} en ${nombre}`;
    }else{if(info)info.textContent='Sin veredas en este municipio';}
  }
  cambiarAreaAnalisis(); aplicarFiltros();
}

// ── Alertas Supabase ──────────────────────────────────────────────────────────
function _paramsFechaAlertas() {
  let ini,fin;
  if(modoPeriodo==='rango'&&fechaInicio&&fechaFin){ini=fechaInicio.toISOString();fin=fechaFin.toISOString();}
  else{ini=new Date(Date.now()-diasActual*864e5).toISOString();fin=new Date().toISOString();}
  return`fecha_deteccion=gte.${encodeURIComponent(ini)}&fecha_deteccion=lte.${encodeURIComponent(fin)}`;
}

async function cargarAlertas() {
  try{
    const filtroFechas=_paramsFechaAlertas();
    const url=`${SUPABASE_URL}/rest/v1/alertas?select=*&${filtroFechas}&order=fecha_deteccion.desc&limit=10000`;
    const r=await fetch(url,{headers:{apikey:SUPABASE_KEY,Authorization:`Bearer ${SUPABASE_KEY}`}});
    if(!r.ok)throw new Error(`HTTP ${r.status}`);
    todasAlertas=await r.json();
    aplicarFiltros();
  }catch(e){
    document.getElementById('lista-alertas').innerHTML=`<div class="flex items-center justify-center py-8 text-red-400 text-xs">Error: ${e.message}</div>`;
  }
}

// ── Deforestación GEE ─────────────────────────────────────────────────────────
async function toggleDeforestacion(visible) {
  deforVisible=visible;
  const _sd=document.getElementById('sub-deforestacion');if(_sd){if(visible)_sd.classList.remove('hidden');else _sd.classList.add('hidden');}
  if(!visible){marcadoresDefor.forEach(m=>map.removeLayer(m));marcadoresDefor=[];document.getElementById('defor-stats').innerHTML='';aplicarFiltros();return;}
  await cargarDeforestacion();
}
async function cargarDeforestacion() {
  if(!deforVisible)return;
  try{
    const dias=parseInt(document.getElementById('sel-defor-periodo').value);
    const ini=new Date(Date.now()-dias*864e5).toISOString().slice(0,10);
    const r=await fetch(`${SUPABASE_URL}/rest/v1/alertas_deforestacion?select=*&fecha_deteccion=gte.${ini}&order=fecha_deteccion.desc&limit=500`,{headers:{apikey:SUPABASE_KEY,Authorization:`Bearer ${SUPABASE_KEY}`}});
    if(!r.ok)throw new Error('HTTP '+r.status);
    alertasDefor=await r.json();filtrarDeforestacion();aplicarFiltros();
  }catch(e){console.error('Error defor:',e);document.getElementById('defor-stats').textContent='Error cargando datos.';}
}
function filtrarDeforestacion() {
  if(!deforVisible)return;
  const sev=document.getElementById('sel-defor-severidad').value;
  let f=sev==='all'?alertasDefor:alertasDefor.filter(a=>a.severidad===sev);
  if(areaAnalisisActiva==='dibujo'&&dibujoPoligonoCoords&&dibujoPoligonoCoords.length>=3)
    f=f.filter(a=>pasaFiltroAreaDibujo(a.latitud,a.longitud));
  marcadoresDefor.forEach(m=>map.removeLayer(m));marcadoresDefor=[];
  f.forEach(a=>{
    const color=COLORES_SEV[a.severidad]||'#f97316';
    const r=a.severidad==='severa'?10:a.severidad==='moderada'?7:5;
    const m=L.circleMarker([a.latitud,a.longitud],{radius:r,fillColor:color,color:'#fff',weight:1.5,fillOpacity:0.85}).addTo(map);
    const fecha=new Date(a.fecha_deteccion+'T12:00:00').toLocaleDateString('es-CO');
    m.bindPopup(`<div class="popup-title" style="color:${color}">Deforestación · ${a.severidad}</div><div class="popup-row">Predio <span>${a.predio_id}</span></div><div class="popup-row">Fecha <span>${fecha}</span></div><div class="popup-row">Δ NDVI <span>${a.cambio_ndvi!=null?Number(a.cambio_ndvi).toFixed(3):'-'}</span></div><div class="popup-row">Área <span>${a.area_afectada_ha??'-'} ha</span></div>`);
    marcadoresDefor.push(m);
  });
  const ss=f.filter(a=>a.severidad==='severa').length,sm=f.filter(a=>a.severidad==='moderada').length,sl=f.filter(a=>a.severidad==='leve').length;
  document.getElementById('defor-stats').innerHTML=`${f.length} alertas · <span style="color:#dc2626">● ${ss} severa</span> <span style="color:#f97316">● ${sm} mod.</span> <span style="color:#eab308">● ${sl} leve</span>`;
}

// ── GFW ───────────────────────────────────────────────────────────────────────
function setGFWStatus(msg,tipo=''){const el=document.getElementById('gfw-status');el.textContent=msg;el.className='gfw-status'+(tipo?' '+tipo:'');}
function gfwFechas(dias){
  if(modoPeriodo==='rango'&&fechaInicio&&fechaFin)return{startDate:fechaInicio.toISOString().slice(0,10),endDate:fechaFin.toISOString().slice(0,10)};
  const fin=new Date(),ini=new Date(Date.now()-dias*864e5);
  return{startDate:ini.toISOString().slice(0,10),endDate:fin.toISOString().slice(0,10)};
}

async function toggleGFW(key,visible){
  const capa=gfwCapas[key];capa.visible=visible;
  document.getElementById(`sub-${key}`).classList.toggle('visible',visible);
  if(!visible){capa.marcadores.forEach(m=>map.removeLayer(m));capa.marcadores=[];if(capa.tileLayer){map.removeLayer(capa.tileLayer);capa.tileLayer=null;}document.getElementById(`stats-${key}`).innerHTML='Sin datos cargados';if(!Object.values(gfwCapas).some(c=>c.visible))setGFWStatus('—');return;}
  if(key==='hansen')cargarGFWTilesHansen();else await cargarGFWAlertas(key);
}

function cargarGFWTilesHansen(){
  const capa=gfwCapas.hansen;if(capa.tileLayer){map.removeLayer(capa.tileLayer);capa.tileLayer=null;}
  const K=GFW_API_KEY;
  capa.tileLayer=L.layerGroup([L.tileLayer(`https://tiles.globalforestwatch.org/umd_tree_cover_density_2000/latest/dynamic/{z}/{x}/{y}.png?x-api-key=${K}`,{opacity:hansenOpacity,maxZoom:13}),L.tileLayer(`https://tiles.globalforestwatch.org/umd_tree_cover_loss/latest/dynamic/{z}/{x}/{y}.png?x-api-key=${K}`,{opacity:hansenOpacity,maxZoom:13})]).addTo(map);
  document.getElementById('stats-hansen').innerHTML=`Cobertura 2000 <span style="color:#22c55e">●</span> &nbsp; Pérdida 2001-2023 <span style="color:#dc2626">●</span><br>Opacidad: <span>${Math.round(hansenOpacity*100)}%</span>`;
  setGFWStatus('✓ OK','ok');
}
function actualizarOpacidadHansen(val){
  hansenOpacity=val/100;document.getElementById('val-hansen').textContent=val+'%';
  const capa=gfwCapas.hansen;if(capa.tileLayer)capa.tileLayer.getLayers().forEach(l=>{if(l.setOpacity)l.setOpacity(hansenOpacity);});
  document.getElementById('stats-hansen').innerHTML=`Cobertura 2000 <span style="color:#22c55e">●</span> &nbsp; Pérdida 2001-2023 <span style="color:#dc2626">●</span><br>Opacidad: <span>${val}%</span>`;
}

const GFW_CFG={
  glad:{dataset:'gfw_integrated_alerts',sqlFn:(s,e)=>`SELECT latitude,longitude,gfw_integrated_alerts__date AS fecha,gfw_integrated_alerts__confidence AS confianza,gfw_integrated_alerts__intensity AS intensidad FROM results WHERE gfw_integrated_alerts__date>='${s}' AND gfw_integrated_alerts__date<='${e}'`,colorFn:row=>(row.confianza||'').toLowerCase().includes('high')?'#dc2626':'#f97316',radiusFn:row=>(row.confianza||'').toLowerCase().includes('high')?8:5,popupFn:(row,lat,lng)=>`<div class="popup-title gfw">Alerta GLAD</div><div class="popup-row">Coords <span>${lat.toFixed(4)}, ${lng.toFixed(4)}</span></div><div class="popup-row">Fecha <span>${row.fecha||'N/D'}</span></div><div class="popup-row">Confianza <span>${row.confianza||'N/D'}</span></div><div class="popup-row">Intensidad <span>${row.intensidad||'N/D'}</span></div>`},
  radd:{dataset:'wur_radd_alerts',sqlFn:(s,e)=>`SELECT latitude,longitude,wur_radd_alerts__date AS fecha,wur_radd_alerts__confidence AS confianza FROM results WHERE wur_radd_alerts__date>='${s}' AND wur_radd_alerts__date<='${e}'`,colorFn:row=>(row.confianza||'').toLowerCase()==='confirmed'?'#7c3aed':'#a78bfa',radiusFn:row=>(row.confianza||'').toLowerCase()==='confirmed'?8:5,popupFn:(row,lat,lng)=>`<div class="popup-title gfw" style="color:#a78bfa">Alerta RADD</div><div class="popup-row">Coords <span>${lat.toFixed(4)}, ${lng.toFixed(4)}</span></div><div class="popup-row">Fecha <span>${row.fecha||'N/D'}</span></div><div class="popup-row">Confianza <span>${row.confianza||'N/D'}</span></div>`},
  fires:{dataset:'nasa_viirs_fire_alerts',sqlFn:(s,e)=>`SELECT latitude,longitude,acq_date AS fecha,confidence__cat AS confianza FROM results WHERE acq_date>='${s}' AND acq_date<='${e}'`,colorFn:()=>'#ff2200',radiusFn:()=>6,popupFn:(row,lat,lng)=>`<div class="popup-title gfw">Fire GFW</div><div class="popup-row">Coords <span>${lat.toFixed(4)}, ${lng.toFixed(4)}</span></div>`},
  hansen:{dataset:'umd_tree_cover_loss',sqlFn:()=>'',colorFn:()=>'',radiusFn:()=>0,popupFn:()=>''}
};

// FIX: cargarGFWAlertas — sin re-filtro por Área de Estudio en modo dibujo
async function cargarGFWAlertas(key){
  setGFWStatus('Cargando…','loading');
  const statsEl=document.getElementById(`stats-${key}`);
  statsEl.innerHTML='<span style="color:#ca8a04">⏳ Consultando GFW…</span>';
  const cfg=GFW_CFG[key];
  const{startDate,endDate}=gfwFechas(diasActual);
  const geom=await obtenerGeomActiva();
  let sql=cfg.sqlFn(startDate,endDate);
  if(key==='glad'){const conf=document.getElementById('sel-glad-conf')?.value;if(conf==='high')sql+=` AND gfw_integrated_alerts__confidence IN ('high','highest')`;if(conf==='highest')sql+=` AND gfw_integrated_alerts__confidence='highest'`;}
  sql+=' LIMIT 2000';
  try{
    const resp=await fetch(`${GFW_BASE}/dataset/${cfg.dataset}/latest/query/json`,{method:'POST',headers:{'Content-Type':'application/json','x-api-key':GFW_API_KEY},body:JSON.stringify({sql,geometry:geom})});
    if(!resp.ok)throw new Error(`HTTP ${resp.status}`);
    const data=await resp.json();
    // La API ya filtró por el polígono activo — no re-filtrar aquí
    let datos=data.data||[];
    if(RECORTAR_DIBUJO_A_AREA_ESTUDIO&&areaAnalisisActiva==='dibujo'&&areaEstudioGeom){
      datos=datos.filter(r=>{const lat=_numCoord(r.latitude),lng=_numCoord(r.longitude);if(lat===null||lng===null)return false;return puntoEnGeoJSON(lat,lng,areaEstudioGeom);});
    }
    gfwCapas[key].datos=datos;
    renderGFWMarcadores(key,datos);
  }catch(e){
    console.error(`GFW ${key}:`,e);
    statsEl.innerHTML=`<span style="color:var(--fuego)">❌ ${e.message}</span>`;
    setGFWStatus('Error','err');
  }
}

function renderGFWMarcadores(key,rows){
  const capa=gfwCapas[key];capa.marcadores.forEach(m=>map.removeLayer(m));capa.marcadores=[];
  const cfg=GFW_CFG[key];const gladConf=key==='glad'?(document.getElementById('sel-glad-conf')?.value||''):'';
  let counts={};
  rows.forEach(row=>{
    const lat=parseFloat(row.latitude),lng=parseFloat(row.longitude);if(isNaN(lat)||isNaN(lng))return;
    const conf=(row.confianza||'').toLowerCase();
    if(key==='glad'&&gladConf==='high'&&conf==='nominal')return;
    if(key==='glad'&&gladConf==='highest'&&conf!=='highest')return;
    counts[conf]=(counts[conf]||0)+1;
    const m=L.circleMarker([lat,lng],{radius:cfg.radiusFn(row),fillColor:cfg.colorFn(row),color:'#fff',weight:1.2,fillOpacity:.88}).addTo(map);
    m.bindPopup(cfg.popupFn(row,lat,lng));capa.marcadores.push(m);
  });
  const tot=capa.marcadores.length;const{startDate,endDate}=gfwFechas(diasActual);
  const area=areaAnalisisActiva==='estudio'?'área de estudio':areaAnalisisActiva==='nucleos'?'núcleos boscosos':areaAnalisisActiva==='municipio'?(municipioActual?`municipio ${municipioActual}`:'todos los municipios'):areaAnalisisActiva==='dibujo'?`polígono personalizado (${dibujoArea_ha.toFixed(1)} ha)`:'área activa';
  if(tot===0){document.getElementById(`stats-${key}`).innerHTML=`<span>Sin alertas en ${area}</span>`;setGFWStatus('✓ 0','ok');}
  else{const confRes=Object.entries(counts).map(([c,n])=>`<span>${n} ${c||'?'}</span>`).join(' &nbsp; ');document.getElementById(`stats-${key}`).innerHTML=`<b style="color:var(--gfw-l)">${tot} alertas</b> en ${area}<br>${startDate} → ${endDate}<br>${confRes}`;setGFWStatus(`✓ ${tot}`,'ok');}
}

function filtrarGFW(key){if(!gfwCapas[key].visible)return;if(key==='hansen')return;if(gfwCapas[key].datos.length)renderGFWMarcadores(key,gfwCapas[key].datos);else cargarGFWAlertas(key);}
function recargarGFWTile(key){if(gfwCapas[key].visible){if(gfwCapas[key].datos.length)renderGFWMarcadores(key,gfwCapas[key].datos);else cargarGFWAlertas(key);}}

// ── Descarga GFW (3 archivos) ─────────────────────────────────────────────────
const _gfwLocks={glad:false,radd:false,hansen:false};

async function descargarGFW(key){
  if(_gfwLocks[key])return;
  _gfwLocks[key]=true;
  const hoy=new Date().toISOString().slice(0,10);
  const btnId=(key==='glad')?'btn-dl-gfw-glad':'btn-dl-gfw-radd';
  const btn=document.getElementById(btnId);const origText=btn?btn.textContent:'';
  if(btn){btn.textContent='⏳ Generando…';btn.disabled=true;}
  try{
    if(key==='hansen'){window.open('https://data.globalforestwatch.org/datasets/gfw::tree-cover-loss/about','_blank');if(btn){btn.textContent=origText;btn.disabled=false;}_gfwLocks[key]=false;return;}
    const datos=gfwCapas[key].datos;
    if(!datos.length){alert(`Activa la capa ${key.toUpperCase()} primero.`);if(btn){btn.textContent=origText;btn.disabled=false;}_gfwLocks[key]=false;return;}
    const AREA_HA_POR_PIXEL=0.09;
    let veredasMpioGFW=[];
    if(municipioActual&&veredasGJ)veredasMpioGFW=veredasGJ.features.filter(f=>(f.properties.NOMB_MPIO||'').toUpperCase()===municipioActual.toUpperCase());
    if(!municipiosGJ)await cargarMunicipios();
    function mDeA(lat,lng){if(!veredasGJ)return'';const vf=veredasGJ.features.find(f=>puntoEnPoligono(lat,lng,f));return vf?(vf.properties.NOMB_MPIO||''):'Fuera de límites';}
    function vDeA(lat,lng){if(veredasMpioGFW.length){const vf=veredasMpioGFW.find(v=>puntoEnPoligono(lat,lng,v));return vf?(vf.properties.NOMBRE_VER||'Sin vereda'):'Sin vereda';}if(veredasGJ){const vf=veredasGJ.features.find(f=>puntoEnPoligono(lat,lng,f));return vf?(vf.properties.NOMBRE_VER||''):'';} return'';}
    let cab,rows;
    if(key==='glad'){cab='fuente,fecha,municipio,vereda,latitud,longitud,confianza,intensidad,area_ha';rows=datos.map(r=>{const lat=parseFloat(r.latitude||0),lng=parseFloat(r.longitude||0);return['GFW_GLAD',r.fecha||'',municipioActual||mDeA(lat,lng),vDeA(lat,lng),lat,lng,r.confianza||'',r.intensidad||'',AREA_HA_POR_PIXEL.toFixed(2)].join(',');});}
    else{cab='fuente,fecha,municipio,vereda,latitud,longitud,confianza,area_ha';rows=datos.map(r=>{const lat=parseFloat(r.latitude||0),lng=parseFloat(r.longitude||0);return['GFW_RADD',r.fecha||'',municipioActual||mDeA(lat,lng),vDeA(lat,lng),lat,lng,r.confianza||'',AREA_HA_POR_PIXEL.toFixed(2)].join(',');});}
    _dl([cab,...rows].join('\n'),`gfw_${key}_puntos_chameza_${hoy}.csv`);
    const parches=dissolverPixelesEnParches(datos,AREA_HA_POR_PIXEL,key);
    const geojson={type:'FeatureCollection',features:parches.map((p,i)=>({type:'Feature',properties:{parche_id:`${key.toUpperCase()}_${i+1}`,fuente:key==='glad'?'GFW_GLAD':'GFW_RADD',n_pixeles:p.n_pixeles,area_ha:parseFloat(p.area_ha.toFixed(3)),fecha_min:p.fecha_min,fecha_max:p.fecha_max,municipio:municipioActual||mDeA(p.centroide[1],p.centroide[0]),vereda:vDeA(p.centroide[1],p.centroide[0])},geometry:{type:'Polygon',coordinates:[p.poligono]}}))};
    const blobGJ=new Blob([JSON.stringify(geojson,null,2)],{type:'application/geo+json'});const urlGJ=URL.createObjectURL(blobGJ);const aGJ=document.createElement('a');aGJ.href=urlGJ;aGJ.download=`gfw_${key}_parches_chameza_${hoy}.geojson`;document.body.appendChild(aGJ);aGJ.click();document.body.removeChild(aGJ);URL.revokeObjectURL(urlGJ);
    await descargarGFWResumen(key,datos,AREA_HA_POR_PIXEL,hoy);
    if(btn){btn.textContent='✓ 3 archivos descargados';setTimeout(()=>{btn.textContent=origText;btn.disabled=false;},2500);}
  }catch(e){console.error('Error descarga GFW:',e);if(btn){btn.textContent=origText;btn.disabled=false;}alert('Error generando descarga: '+e.message);}
  finally{setTimeout(()=>{_gfwLocks[key]=false;},1500);}
}

function dissolverPixelesEnParches(datos,areaPorPixel,key){
  if(!datos.length)return[];
  const PIXEL_DEG=0.00028,TOL=PIXEL_DEG*1.5;
  const puntos=datos.map(d=>({lat:parseFloat(d.latitude),lng:parseFloat(d.longitude),fecha:d.fecha||'',cluster:-1})).filter(p=>!isNaN(p.lat)&&!isNaN(p.lng));
  let cId=0;
  for(let i=0;i<puntos.length;i++){
    if(puntos[i].cluster!==-1)continue;
    puntos[i].cluster=cId;const cola=[i];
    while(cola.length){const idx=cola.shift();const a=puntos[idx];for(let j=0;j<puntos.length;j++){if(puntos[j].cluster!==-1)continue;const b=puntos[j];if(Math.abs(a.lat-b.lat)<=TOL&&Math.abs(a.lng-b.lng)<=TOL){b.cluster=cId;cola.push(j);}}}
    cId++;
  }
  const parches=[];
  for(let c=0;c<cId;c++){
    const mb=puntos.filter(p=>p.cluster===c);if(!mb.length)continue;
    const lats=mb.map(m=>m.lat),lngs=mb.map(m=>m.lng),h=PIXEL_DEG/2;
    const pol=[[Math.min(...lngs)-h,Math.min(...lats)-h],[Math.max(...lngs)+h,Math.min(...lats)-h],[Math.max(...lngs)+h,Math.max(...lats)+h],[Math.min(...lngs)-h,Math.max(...lats)+h],[Math.min(...lngs)-h,Math.min(...lats)-h]];
    const cLat=lats.reduce((s,v)=>s+v,0)/lats.length,cLng=lngs.reduce((s,v)=>s+v,0)/lngs.length;
    const fechas=mb.map(m=>m.fecha).filter(Boolean).sort();
    parches.push({n_pixeles:mb.length,area_ha:mb.length*areaPorPixel,centroide:[cLng,cLat],poligono:pol,fecha_min:fechas[0]||'',fecha_max:fechas[fechas.length-1]||''});
  }
  parches.sort((a,b)=>b.area_ha-a.area_ha);return parches;
}

async function descargarGFWResumen(key,datos,areaPorPixel,hoy){
  if(!municipiosGJ)await cargarMunicipios();if(!veredasGJ)return;
  const agg={};
  datos.forEach(d=>{
    const lat=parseFloat(d.latitude),lng=parseFloat(d.longitude);if(isNaN(lat)||isNaN(lng))return;
    let mpio='Fuera de límites',ver='Sin vereda';
    const vf=veredasGJ.features.find(f=>puntoEnPoligono(lat,lng,f));
    if(vf){mpio=vf.properties.NOMB_MPIO||'Sin nombre';ver=vf.properties.NOMBRE_VER||'Sin nombre';}
    const k=`${mpio}||${ver}`;if(!agg[k])agg[k]={mpio,ver,n_pixeles:0,fechas:[]};agg[k].n_pixeles++;if(d.fecha)agg[k].fechas.push(d.fecha);
  });
  const esc=v=>{const s=String(v??'');return s.includes(',')?'"'+s.replace(/"/g,'""')+'"':s;};
  const cabV=['fuente','municipio','vereda','n_pixeles','area_ha','fecha_min','fecha_max'].join(',');
  const rowsV=Object.values(agg).sort((a,b)=>b.n_pixeles-a.n_pixeles).map(r=>{const fs=r.fechas.sort();return[key==='glad'?'GFW_GLAD':'GFW_RADD',esc(r.mpio),esc(r.ver),r.n_pixeles,(r.n_pixeles*areaPorPixel).toFixed(2),fs[0]||'',fs[fs.length-1]||''].join(',');});
  const aggM={};Object.values(agg).forEach(r=>{if(!aggM[r.mpio])aggM[r.mpio]={mpio:r.mpio,n_pixeles:0,n_veredas:0};aggM[r.mpio].n_pixeles+=r.n_pixeles;aggM[r.mpio].n_veredas++;});
  const cabM=['fuente','municipio','n_pixeles','area_ha','n_veredas_afectadas'].join(',');
  const rowsM=Object.values(aggM).sort((a,b)=>b.n_pixeles-a.n_pixeles).map(r=>[key==='glad'?'GFW_GLAD':'GFW_RADD',esc(r.mpio),r.n_pixeles,(r.n_pixeles*areaPorPixel).toFixed(2),r.n_veredas].join(','));
  _dl(['## RESUMEN POR VEREDA',cabV,...rowsV,'','## RESUMEN POR MUNICIPIO',cabM,...rowsM].join('\n'),`gfw_${key}_resumen_chameza_${hoy}.csv`);
}

// ── CUNAGÜITO ─────────────────────────────────────────────────────────────────
const CUNA_DATOS={
  mapabase:{img:'mapabase.png',msg:`Para cambiar el mapa base, usa el control de capas en la <b>esquina inferior derecha</b> del mapa. Encontrarás: <b>Google Maps</b>, <b>Satélite</b>, <b>Relieve</b> y <b>Oscuro</b>.`},
  incendios:{img:'incendios.png',msg:`Selecciona <b>Incendios</b> en "Tipo de alerta", elige tu período y el área de análisis. Usa los botones <b>⬇ CSV</b> o <b>⬇ PDF</b> para descargar.`},
  deforestacion:{img:'deforestacion.png',msg:`Selecciona <b>Global Forest Watch</b> y activa <b>GLAD</b> o <b>RADD</b>. Una vez carguen las alertas, descarga con <b>⬇ GFW</b>.`},
  conectividad:{img:'conectividad.png',msg:`En <b>Capas adicionales</b>, marca <b>Conectividad (Deciles)</b>. Los deciles del 1 al 10 representan la importancia de conectividad del paisaje.`},
  gfw:{img:'gfw.png',msg:`Selecciona <b>Global Forest Watch</b>. Allí encontrarás:<br><br>🌲 <b>GLAD</b> — Alertas Landsat semanales<br>📡 <b>RADD</b> — Radar Sentinel-1, funciona bajo nubes<br>🌳 <b>Hansen</b> — Cobertura y pérdida 2001-2023`},
  sobre:{img:null,esMulti:true,slides:[{titulo:'¿Qué hace la plataforma?',img:'kuna_workflow.png'},{titulo:'¿En qué estamos ahora?',img:'kuna_workflow_estado.png'}]},
  funcionalidades:{img:null,esMulti:true,slides:[
    {titulo:'Cambiar el mapa base',img:'mapabase.png',kunaImg:'kuna_senala.png',texto:`Usa el control de capas en la <b>esquina inferior derecha</b> del mapa. Tienes 5 opciones: <b>CartoDB Claro</b>, <b>Google Maps</b>, <b>Satélite</b>, <b>Relieve</b> y <b>Oscuro</b>.`},
    {titulo:'Global Forest Watch — GLAD & RADD',img:'gfw_incon.png',kunaImg:'kuna_saludando.png',texto:`<b class="gfw">Global Forest Watch</b> (WRI) provee alertas satelitales de deforestación casi en tiempo real.<br><br>🌲 <b>GLAD</b> — Landsat · Universidad de Maryland · detección semanal<br>📡 <b>RADD</b> — Sentinel-1 SAR · Univ. Wageningen · funciona bajo nubes<br><br>Créditos: © Global Forest Watch / World Resources Institute. Datos bajo licencia CC BY 4.0.`},
    {titulo:'Alertas de Incendios — NASA FIRMS',img:'incendios.png',kunaImg:'kuna_senala.png',texto:`Los puntos de calor provienen de <b>NASA FIRMS</b> (Fire Information for Resource Management System), usando sensores <b>VIIRS</b> (375 m) y <b>MODIS</b> (1 km) actualizados cada 3 horas.`},
    {titulo:'Capas Adicionales',img:'capas_adicionales.png',kunaImg:'kuna_saludando.png',texto:`En <b>Capas adicionales</b> puedes activar:<br><br>📍 <b>Límites municipales</b><br>📐 <b>Área de estudio</b><br>🌲 <b>Núcleos boscosos</b><br>📉 <b>Pérdidas 2020-2024</b><br>🔗 <b>Conectividad</b> (deciles)<br>🌿 <b>Alertas deforestación GEE</b>`}
  ]}
};

function cunaAbrir(){const el=document.getElementById('cuna-overlay');if(el){el.classList.remove('hidden');el.classList.add('flex');}}
function cunaCerrar(){const el=document.getElementById('cuna-overlay');if(el){el.classList.add('hidden');el.classList.remove('flex');}setTimeout(cunaVolver,300);}
function cunaOverlayClick(e){if(e.target===document.getElementById('cuna-overlay'))cunaCerrar();}

function cunaIr(key){
  const d=CUNA_DATOS[key];if(!d)return;
  const avatar=document.getElementById('cuna-avatar');
  if(avatar){avatar.style.opacity='0';setTimeout(()=>{avatar.src='kuna_saludando.png';avatar.style.opacity='1';},150);}
  const menu=document.getElementById('cuna-menu'),detail=document.getElementById('cuna-detail');
  if(menu)menu.style.opacity='0';
  setTimeout(()=>{
    if(menu)menu.style.display='none';
    if(detail){
      detail.innerHTML='';detail.classList.remove('hidden');detail.classList.add('flex');
      if(d.esMulti){_cunaCurrentKey=key;detail.innerHTML=cunaRenderMulti(d,key);cunaSlide(0,d);}
      else{detail.innerHTML=`<img id="cuna-detail-img" class="w-full h-44 object-cover rounded-xl mb-4 border border-slate-100" src="${d.img||''}" alt=""><div id="cuna-detail-text" class="text-sm text-slate-600 leading-relaxed cuna-detail-text">${d.msg||''}</div><button onclick="cunaVolver()" class="mt-5 w-full py-3 border-2 border-slate-200 rounded-xl text-xs font-bold text-slate-600 hover:bg-slate-50 transition-all">← Volver al menú</button>`;}
      detail.style.opacity='0';setTimeout(()=>{detail.style.opacity='1';},20);
    }
  },180);
}

let _cunaSlideIdx=0,_cunaCurrentKey=null;

function cunaRenderMulti(d,key){
  return`<div class="flex items-center justify-between mb-3 w-full"><div class="text-[9px] font-bold text-slate-400 uppercase tracking-widest" id="kuna-slide-label">Cargando…</div><div class="flex gap-1" id="kuna-dots"></div></div><div id="kuna-slide-content" class="flex flex-col gap-3 w-full"></div><div class="flex gap-2 mt-4 w-full"><button onclick="cunaSlideNav(-1)" id="btn-prev" class="flex-1 py-2.5 border-2 border-slate-200 rounded-xl text-xs font-bold text-slate-500 hover:bg-slate-50 transition-all">← Anterior</button><button onclick="cunaSlideNav(1)" id="btn-next" class="flex-1 py-2.5 bg-emerald-600 text-white rounded-xl text-xs font-bold hover:bg-emerald-700 transition-all">Siguiente →</button></div><button onclick="cunaVolver()" class="mt-2 w-full py-2.5 border-2 border-slate-100 rounded-xl text-[10px] font-bold text-slate-400 hover:bg-slate-50 transition-all">← Volver al menú</button>`;
}

function cunaSlide(idx,d){
  if(!d)return;const slides=d.slides||[];
  _cunaSlideIdx=Math.max(0,Math.min(idx,slides.length-1));const s=slides[_cunaSlideIdx];if(!s)return;
  const lbl=document.getElementById('kuna-slide-label');if(lbl)lbl.textContent=s.titulo;
  const dots=document.getElementById('kuna-dots');if(dots)dots.innerHTML=slides.map((_,i)=>`<div class="w-1.5 h-1.5 rounded-full transition-all ${i===_cunaSlideIdx?'bg-emerald-600':'bg-slate-200'}"></div>`).join('');
  const cont=document.getElementById('kuna-slide-content');
  if(cont){if(!s.kunaImg&&!s.texto){cont.innerHTML=`<img src="${s.img}" alt="${s.titulo}" class="w-full rounded-xl border border-slate-100 shadow-md" style="width:100%;object-fit:contain;background:#f8faf9;display:block">`;}else{cont.innerHTML=`<div class="flex items-start gap-3"><img src="${s.kunaImg||'kuna_saludando.png'}" alt="Kuna" class="w-14 h-14 rounded-xl object-cover shrink-0 border border-emerald-100"><div class="text-sm font-semibold text-slate-700 leading-relaxed">${s.texto||''}</div></div><img src="${s.img}" alt="${s.titulo}" class="w-full rounded-xl border border-slate-100 shadow-sm" style="max-height:200px;object-fit:contain;background:#f8faf9">`;}}
  const prev=document.getElementById('btn-prev'),next=document.getElementById('btn-next');
  if(prev)prev.style.visibility=_cunaSlideIdx===0?'hidden':'visible';
  if(next){next.textContent=_cunaSlideIdx===slides.length-1?'✓ Listo':'Siguiente →';next.onclick=_cunaSlideIdx===slides.length-1?cunaVolver:()=>cunaSlideNav(1);}
}
function cunaSlideNav(dir){if(!_cunaCurrentKey)return;const d=CUNA_DATOS[_cunaCurrentKey];if(!d||!d.slides)return;const ni=_cunaSlideIdx+dir;if(ni>=0&&ni<d.slides.length)cunaSlide(ni,d);}
function cunaVolver(){
  const avatar=document.getElementById('cuna-avatar');if(avatar){avatar.style.opacity='0';setTimeout(()=>{avatar.src='Cunaguito1.png';avatar.style.opacity='1';},150);}
  const msg=document.getElementById('cuna-msg');if(msg){msg.style.opacity='0';setTimeout(()=>{msg.textContent='¡Hola! Soy Kuna. Estoy aquí para guiarte en el manejo del aplicativo. ¿Qué quieres hacer?';msg.style.opacity='1';},150);}
  const menu=document.getElementById('cuna-menu'),detail=document.getElementById('cuna-detail');
  if(detail)detail.style.opacity='0';
  setTimeout(()=>{if(detail)detail.classList.add('hidden');if(menu){menu.style.display='flex';menu.style.opacity='0';setTimeout(()=>{menu.style.opacity='1';},20);}},180);
}

// ── Inicio ────────────────────────────────────────────────────────────────────
inicializarRangoFechas();

obtenerGeomAreaEstudio().then(geom=>{
  if(geom){try{const gjLayer=L.geoJSON({type:'Feature',geometry:geom});map.fitBounds(gjLayer.getBounds(),{padding:[30,30]});}catch(e){console.warn('fitBounds:',e);}}
});

toggleCapa('estudio',true);
cargarMunicipios();
cargarAlertas();

setInterval(cargarAlertas,5*60*1000);
setInterval(()=>{if(deforVisible)cargarDeforestacion();},10*60*1000);
setInterval(()=>{Object.keys(gfwCapas).forEach(k=>{if(!gfwCapas[k].visible)return;if(k!=='hansen')cargarGFWAlertas(k);});},15*60*1000);
