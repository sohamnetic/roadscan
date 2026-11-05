"""Which road and which ward each defect is on, from OpenStreetMap (or a city's own ward file).

Ward engineers plan by ward, and a pothole on a main road matters more than one in a lane,
so each defect gets the nearest drivable road (name, OSM class) and the ward containing it,
and the road class scales its repair priority.

    fetch_osm(bbox, out_dir)  -> roads.geojson + wards.geojson (Overpass API, cached files)
    tag_places(potholes, roads, wards) -> potholes + road, road_type, road_dist_m, ward
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .geo import to_enu

DRIVABLE = (
    "motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential",
    "living_street", "service", "road",
    "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link",
)
# Repair priority multiplier by road class (traffic proxy); unlisted classes count 1.0.
ROAD_WEIGHT = {"motorway": 1.5, "trunk": 1.5, "primary": 1.4, "secondary": 1.3, "tertiary": 1.15}

ENDPOINTS = (
    "https://overpass-api.de/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
)
USER_AGENT = "roadscan/0.1 (road-damage mapping)"


def road_weight(highway: str | None) -> float:
    if not isinstance(highway, str):
        return 1.0
    return ROAD_WEIGHT.get(highway.removesuffix("_link"), 1.0)


# ---------------------------------------------------------------- fetching

def _overpass(query: str, endpoints=ENDPOINTS, timeout: int = 180) -> dict:
    import requests

    errors = []
    for url in endpoints:
        try:
            r = requests.post(url, data={"data": query}, headers={"User-Agent": USER_AGENT}, timeout=timeout)
            if r.ok and r.content:
                return r.json()
            errors.append(f"{url}: HTTP {r.status_code}")
        except Exception as e:  # noqa: BLE001 - try the next mirror
            errors.append(f"{url}: {type(e).__name__}")
    raise ConnectionError("all Overpass servers failed: " + "; ".join(errors))


def _ring_polygons(members: list[dict]):
    """Polygons from a boundary relation's member ways (outer rings minus inner ones)."""
    from shapely.geometry import LineString, MultiPolygon, Polygon
    from shapely.ops import polygonize, unary_union

    def lines(role):
        out = []
        for m in members:
            if m.get("type") == "way" and m.get("role", "outer") in role and len(m.get("geometry", [])) >= 2:
                out.append(LineString([(p["lon"], p["lat"]) for p in m["geometry"]]))
        return out

    outer = list(polygonize(unary_union(lines(("outer", "")))))
    inner = list(polygonize(unary_union(lines(("inner",))))) if lines(("inner",)) else []
    if not outer:
        return None
    shape = unary_union(outer)
    if inner:
        shape = shape.difference(unary_union(inner))
    return shape if isinstance(shape, (Polygon, MultiPolygon)) and not shape.is_empty else None


def fetch_osm(bbox: tuple[float, float, float, float], out_dir: str | Path, ward_level: str = "10",
              endpoints=ENDPOINTS) -> tuple[Path, Path]:
    """Download drivable roads and ward boundaries for bbox = (south, west, north, east)."""
    from shapely.geometry import mapping

    s, w, n, e = bbox
    box = f"{s:.6f},{w:.6f},{n:.6f},{e:.6f}"
    highway = "|".join(DRIVABLE)
    query = (
        f'[out:json][timeout:170];(way["highway"~"^({highway})$"]({box}););out geom;'
        f'rel["boundary"="administrative"]["admin_level"="{ward_level}"]({box});out geom;'
    )
    data = _overpass(query, endpoints)

    roads, wards = [], []
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        if el["type"] == "way" and len(el.get("geometry", [])) >= 2:
            roads.append({
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": [[p["lon"], p["lat"]] for p in el["geometry"]]},
                "properties": {"osm_id": el["id"], "highway": tags.get("highway"), "name": tags.get("name"), "ref": tags.get("ref")},
            })
        elif el["type"] == "relation":
            shape = _ring_polygons(el.get("members", []))
            if shape is not None:
                wards.append({
                    "type": "Feature",
                    "geometry": mapping(shape),
                    "properties": {"osm_id": el["id"], "name": tags.get("name:en") or tags.get("name"), "ref": tags.get("ref")},
                })

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"source": "OpenStreetMap contributors, ODbL", "bbox": [s, w, n, e]}
    roads_path, wards_path = out_dir / "roads.geojson", out_dir / "wards.geojson"
    roads_path.write_text(json.dumps({"type": "FeatureCollection", **meta, "features": roads}), encoding="utf-8")
    wards_path.write_text(json.dumps({"type": "FeatureCollection", **meta, "features": wards}), encoding="utf-8")
    return roads_path, wards_path


def bbox_around(lat, lon, margin_m: float = 300.0) -> tuple[float, float, float, float]:
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    dlat = np.degrees(margin_m / 6_371_000.0)
    dlon = dlat / np.cos(np.radians(lat.mean()))
    return float(lat.min() - dlat), float(lon.min() - dlon), float(lat.max() + dlat), float(lon.max() + dlon)


# ---------------------------------------------------------------- tagging

def _load(path) -> list[dict]:
    if path is None or not Path(path).exists():
        return []
    return json.loads(Path(path).read_text(encoding="utf-8")).get("features", [])


def _ward_name(props: dict) -> str | None:
    for key in ("name", "ward_name", "WARD_NAME", "Ward_Name", "KGISWardName", "name_en"):
        if props.get(key):
            return str(props[key])
    return None


def tag_places(potholes: pd.DataFrame, roads_path=None, wards_path=None, max_road_m: float = 25.0) -> pd.DataFrame:
    """Add road, road_type, road_dist_m and ward columns; scale `priority` by road class.

    A defect with no drivable road within max_road_m keeps road empty and weight 1.
    `wards_path` can be OSM output or any GeoJSON of ward polygons with a name property.
    """
    from shapely import STRtree
    from shapely.geometry import LineString, Point, shape

    out = potholes.copy()
    out["road"], out["road_type"], out["road_dist_m"], out["ward"] = None, None, np.nan, None
    if out.empty:
        return out

    roads = _load(roads_path)
    if roads:
        lat0, lon0 = float(out["lat"].mean()), float(out["lon"].mean())
        lines, props = [], []
        for f in roads:
            xy = np.asarray(f["geometry"]["coordinates"], float)
            ex, ny = to_enu(xy[:, 1], xy[:, 0], lat0, lon0)
            lines.append(LineString(np.column_stack([ex, ny])))
            props.append(f["properties"])
        tree = STRtree(lines)
        ex, ny = to_enu(out["lat"], out["lon"], lat0, lon0)
        points = [Point(x, y) for x, y in zip(ex, ny)]
        (pi, ri), dist = tree.query_nearest(points, max_distance=max_road_m, return_distance=True, all_matches=False)
        for k, (i, j) in enumerate(zip(pi, ri)):
            p = props[j]
            label = p.get("name") or p.get("ref") or f"unnamed {(p.get('highway') or 'road').replace('_', ' ')}"
            row = out.index[i]
            out.at[row, "road"], out.at[row, "road_type"], out.at[row, "road_dist_m"] = label, p.get("highway"), round(float(dist[k]), 1)

    wards = _load(wards_path)
    if wards:
        polys = [shape(f["geometry"]) for f in wards]
        names = [_ward_name(f.get("properties", {})) for f in wards]
        tree = STRtree(polys)
        pts = [Point(lon, lat) for lat, lon in zip(out["lat"], out["lon"])]
        pi, wi = tree.query(pts, predicate="within")
        for i, j in zip(pi, wi):
            out.at[out.index[i], "ward"] = names[j]

    if "priority" in out:
        out["priority"] = (out["priority"] * out["road_type"].map(road_weight).astype(float)).round(2)
    return out


def ward_summary(potholes: pd.DataFrame) -> pd.DataFrame:
    """One row per ward: open defects, how severe, how old: the view a ward engineer works from."""
    cols = ["ward", "open", "confirmed", "severity_4_5", "open_over_30_days", "oldest_days", "total_priority"]
    if potholes.empty or "ward" not in potholes:
        return pd.DataFrame(columns=cols)
    active = potholes[potholes["status"].isin(["open", "confirmed"])].copy()
    active["ward"] = active["ward"].fillna("(outside known wards)")
    g = active.groupby("ward")
    out = pd.DataFrame({
        "open": g.size(),
        "confirmed": g["status"].apply(lambda s: int((s == "confirmed").sum())),
        "severity_4_5": g["severity"].apply(lambda s: int((s >= 4).sum())),
        "open_over_30_days": g["days_open"].apply(lambda s: int((s > 30).sum())),
        "oldest_days": g["days_open"].max(),
        "total_priority": g["priority"].sum().round(1),
    }).reset_index()
    return out.sort_values("total_priority", ascending=False)[cols].reset_index(drop=True)
