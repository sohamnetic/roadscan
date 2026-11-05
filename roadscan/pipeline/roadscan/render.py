"""Exports: a self-contained Leaflet map, GeoJSON for GIS teams, CSV for spreadsheets."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

EXPORT_COLUMNS = ["id", "lat", "lon", "cls", "status", "severity", "area_m2", "first_seen", "last_seen", "days_open", "n_sessions", "misses", "priority"]
PLACE_COLUMNS = ["ward", "road", "road_type"]  # present after places.tag_places


def _columns(potholes: pd.DataFrame) -> list[str]:
    return EXPORT_COLUMNS + [c for c in PLACE_COLUMNS if c in potholes]


def to_geojson(potholes: pd.DataFrame) -> dict:
    features = []
    for _, r in potholes.iterrows():
        props = {k: (None if pd.isna(r[k]) else r[k]) for k in _columns(potholes) if k not in ("lat", "lon")}
        props = {k: (v.item() if hasattr(v, "item") else v) for k, v in props.items()}
        features.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [float(r["lon"]), float(r["lat"])]}, "properties": props})
    return {"type": "FeatureCollection", "features": features}


def write_exports(potholes: pd.DataFrame, out_dir: str | Path) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ranked = potholes.sort_values(["priority", "severity"], ascending=False)
    csv_path, geo_path = out_dir / "potholes.csv", out_dir / "potholes.geojson"
    ranked[_columns(ranked)].to_csv(csv_path, index=False)
    geo_path.write_text(json.dumps(to_geojson(ranked)), encoding="utf-8")
    paths = [csv_path, geo_path]
    if "ward" in ranked:
        from .places import ward_summary

        wards_path = out_dir / "wards.csv"
        ward_summary(ranked).to_csv(wards_path, index=False)
        paths.append(wards_path)
    return paths


def write_map(potholes: pd.DataFrame, out_path: str | Path, title: str = "Road damage map") -> Path:
    data = json.dumps(to_geojson(potholes)).replace("</", "<\\/")
    html = MAP_TEMPLATE.replace("__TITLE__", title).replace("__DATA__", data)
    out_path = Path(out_path)
    out_path.write_text(html, encoding="utf-8")
    return out_path


MAP_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  :root { --bg:#ffffff; --fg:#1d232b; --muted:#5f6b7a; --line:#e3e7ec; --panel:#f7f8fa; }
  @media (prefers-color-scheme: dark) { :root { --bg:#14181d; --fg:#e8ecf1; --muted:#9aa6b4; --line:#2a313a; --panel:#1b2027; } }
  * { box-sizing: border-box; }
  body { margin:0; font:14px/1.4 system-ui, -apple-system, "Segoe UI", sans-serif; background:var(--bg); color:var(--fg); display:flex; height:100vh; }
  aside { width:340px; flex:none; overflow:auto; border-right:1px solid var(--line); background:var(--panel); padding:16px; }
  #map { flex:1; }
  h1 { font-size:17px; margin:0 0 4px; }
  .sub { color:var(--muted); margin:0 0 14px; }
  .stats { display:grid; grid-template-columns:repeat(3,1fr); gap:8px; margin-bottom:14px; }
  .stat { background:var(--bg); border:1px solid var(--line); border-radius:8px; padding:8px; }
  .stat b { display:block; font-size:18px; }
  .stat span { color:var(--muted); font-size:12px; }
  fieldset { border:1px solid var(--line); border-radius:8px; margin:0 0 14px; padding:8px 10px; }
  legend { color:var(--muted); font-size:12px; padding:0 4px; }
  #wardrow { margin:0 0 14px; } #wardrow select { flex:1; min-width:0; padding:4px; background:var(--bg); color:var(--fg); border:1px solid var(--line); border-radius:6px; }
  label { display:flex; align-items:center; gap:6px; margin:3px 0; cursor:pointer; }
  .dot { width:10px; height:10px; border-radius:50%; display:inline-block; }
  h2 { font-size:13px; text-transform:uppercase; letter-spacing:.04em; color:var(--muted); margin:6px 0 8px; }
  ol { margin:0; padding:0; list-style:none; }
  li { padding:8px; border:1px solid var(--line); border-radius:8px; margin-bottom:6px; background:var(--bg); cursor:pointer; }
  li:hover { border-color:var(--muted); }
  li .meta { color:var(--muted); font-size:12px; }
  @media (max-width: 720px) { body { flex-direction:column; } aside { width:100%; height:45vh; border-right:0; border-bottom:1px solid var(--line); } }
</style>
</head>
<body>
<aside>
  <h1>__TITLE__</h1>
  <p class="sub" id="asof"></p>
  <div class="stats" id="stats"></div>
  <fieldset id="filters"><legend>Show</legend></fieldset>
  <label id="wardrow" hidden>Ward <select id="ward"><option value="">All wards</option></select></label>
  <h2>Repair priority</h2>
  <ol id="list"></ol>
</aside>
<div id="map"></div>
<script>
const DATA = __DATA__;
const STATUS = {
  confirmed:      { label: "Confirmed",      color: "#d1352b", show: true },
  open:           { label: "Seen once",      color: "#e8912d", show: true },
  likely_fixed:   { label: "Likely fixed",   color: "#3a7bd5", show: true },
  verified_fixed: { label: "Verified fixed", color: "#2e9a5b", show: false },
  dismissed:      { label: "Dismissed",      color: "#8a94a3", show: false },
};
const fmt = (c) => c.replace(/_/g, " ");
const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[ch]);
const place = (p) => [p.road, p.ward && `ward ${p.ward}`].filter(Boolean).map(esc).join(" &middot; ");
const map = L.map("map");
L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 19, attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
}).addTo(map);

const layers = {};
for (const key of Object.keys(STATUS)) layers[key] = L.layerGroup();
const markers = new Map();
let wardFilter = "";
const inWard = (f) => !wardFilter || f.properties.ward === wardFilter;
for (const f of DATA.features) {
  const p = f.properties, s = STATUS[p.status] || STATUS.open;
  const [lon, lat] = f.geometry.coordinates;
  const m = L.circleMarker([lat, lon], { radius: 4 + 2 * (p.severity || 1), color: s.color, fillColor: s.color, fillOpacity: 0.55, weight: 2 })
    .bindPopup(`<b>#${p.id} ${fmt(p.cls)}</b><br>${s.label} &middot; severity ${p.severity}/5<br>` +
      (place(p) ? `${place(p)}<br>` : "") + `First seen ${String(p.first_seen).slice(0, 10)} (${p.days_open} days)<br>Seen on ${p.n_sessions} ride(s), missed ${p.misses}` +
      (p.area_m2 ? `<br>~${p.area_m2.toFixed(2)} m&sup2;` : "") + `<br><a href="https://www.google.com/maps?q=${lat},${lon}" target="_blank" rel="noopener">Open in Google Maps</a>`);
  m.feature = f;
  (layers[p.status] || layers.open).addLayer(m);
  markers.set(p.id, m);
}

const filters = document.getElementById("filters");
for (const [key, s] of Object.entries(STATUS)) {
  const n = DATA.features.filter((f) => f.properties.status === key).length;
  const row = document.createElement("label");
  row.innerHTML = `<input type="checkbox" ${s.show ? "checked" : ""}><span class="dot" style="background:${s.color}"></span>${s.label} (${n})`;
  const box = row.querySelector("input");
  const sync = () => (box.checked ? layers[key].addTo(map) : layers[key].remove());
  box.addEventListener("change", sync);
  sync();
  filters.appendChild(row);
}

function render() {
  for (const m of markers.values()) {
    const group = layers[m.feature.properties.status] || layers.open;
    if (inWard(m.feature)) group.addLayer(m); else group.removeLayer(m);
  }
  const active = DATA.features.filter((f) => ["open", "confirmed"].includes(f.properties.status) && inWard(f));
  const severe = active.filter((f) => f.properties.severity >= 4).length;
  const old = active.filter((f) => f.properties.days_open > 30).length;
  document.getElementById("stats").innerHTML =
    `<div class="stat"><b>${active.length}</b><span>open defects</span></div>` +
    `<div class="stat"><b>${severe}</b><span>severity 4–5</span></div>` +
    `<div class="stat"><b>${old}</b><span>open &gt; 30 days</span></div>`;
  const list = document.getElementById("list");
  list.innerHTML = "";
  for (const f of [...active].sort((a, b) => b.properties.priority - a.properties.priority).slice(0, 25)) {
    const p = f.properties, li = document.createElement("li");
    li.innerHTML = `<b>#${p.id} ${fmt(p.cls)}</b> &middot; severity ${p.severity}/5<div class="meta">` +
      (place(p) ? `${place(p)}<br>` : "") + `${p.days_open} days open &middot; ${p.n_sessions} ride(s) &middot; priority ${p.priority}</div>`;
    li.addEventListener("click", () => { map.setView(markers.get(p.id).getLatLng(), 18); markers.get(p.id).openPopup(); });
    list.appendChild(li);
  }
}
const wards = [...new Set(DATA.features.map((f) => f.properties.ward).filter(Boolean))].sort();
if (wards.length) {
  const select = document.getElementById("ward");
  for (const w of wards) select.add(new Option(w, w));
  document.getElementById("wardrow").hidden = false;
  select.addEventListener("change", () => {
    wardFilter = select.value;
    render();
    const shown = DATA.features.filter(inWard).map((f) => [f.geometry.coordinates[1], f.geometry.coordinates[0]]);
    if (shown.length) map.fitBounds(shown, { padding: [30, 30], maxZoom: 17 });
  });
}
render();
const lastSeen = DATA.features.map((f) => f.properties.last_seen).sort().pop();
document.getElementById("asof").textContent = lastSeen ? `Latest observation ${String(lastSeen).slice(0, 16).replace("T", " ")} UTC` : "No observations yet";


const all = DATA.features.map((f) => [f.geometry.coordinates[1], f.geometry.coordinates[0]]);
const frame = () => {
  map.invalidateSize();
  if (all.length) map.fitBounds(all, { padding: [30, 30], maxZoom: 17 }); else map.setView([20.59, 78.96], 5);
};
frame();
window.addEventListener("load", frame, { once: true });
</script>
</body>
</html>
"""
