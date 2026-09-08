// Proxy serverless para las consultas GFW (GLAD/RADD).
//
// Por qué existe: GitHub Actions (usado para el reporte mensual) corre en
// rangos de IP de Azure que GFW parece bloquear específicamente para
// peticiones servidor-a-servidor (confirmado: la misma key, mismo body,
// misma redirección 307, funcionan perfecto desde una terminal normal e
// incluso en incógnito, pero siempre fallan con 403 desde GitHub Actions).
// Vercel también es "la nube", pero es la infraestructura donde ya sabemos
// que este proyecto funciona sin problemas (así lo usa app.js en el
// navegador), así que probamos si desde aquí GFW sí deja pasar la consulta.
//
// Uso: POST /api/gfw-proxy  body: { "dataset": "...", "sql": "...", "geometry": {...} }

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Solo POST' });
  }

  const { dataset, sql, geometry } = req.body || {};
  if (!dataset || !sql || !geometry) {
    return res.status(400).json({ error: 'Faltan dataset, sql o geometry en el body' });
  }

  const GFW_API_KEY = process.env.GFW_API_KEY || '6b196681-4bfb-4c71-8757-b745b9290f95';
  const url = `https://data-api.globalforestwatch.org/dataset/${dataset}/latest/query/json`;

  try {
    const gfwResp = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': GFW_API_KEY,
      },
      body: JSON.stringify({ sql, geometry }),
    });

    const texto = await gfwResp.text();
    res.status(gfwResp.status);
    res.setHeader('Content-Type', 'application/json');
    return res.send(texto);
  } catch (e) {
    return res.status(502).json({ error: 'Error llamando a GFW', detalle: String(e) });
  }
}
