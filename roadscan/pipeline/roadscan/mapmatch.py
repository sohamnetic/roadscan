"""Snap a ride's GPS track onto the road network (hidden-Markov map-matching).

Phone GPS drifts 3-10 m, enough to put a ride on the wrong side of a divided road or on a
parallel lane. Matching the whole track (not each fix alone) uses the fact that a vehicle
moves along connected roads: a fix that drifts toward a parallel road is still matched to
the road the ride is actually on.

Each fix is then *clamped* into its matched road's corridor rather than pulled onto the
centre line: a fix already within the road keeps its lane position; one that drifted off
the road is moved back to the road edge. Corridor half-widths are rough per road class.

Algorithm (Newson & Krumm 2009, simplified): candidates are road segments within reach of
each fix; emission ~ Gaussian in the fix-to-road distance; transition ~ exponential in
|distance along roads - straight-line distance|; Viterbi picks the best path. Roads are
treated as two-way. When no connected path fits, the chain restarts (a gap in the map).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .geo import EARTH_RADIUS_M, to_enu

HALF_WIDTH_M = {
    "motorway": 9.0, "trunk": 8.0, "primary": 6.0, "secondary": 5.0, "tertiary": 4.0,
    "unclassified": 3.0, "residential": 3.0, "living_street": 2.5, "service": 2.0, "road": 3.0,
}
MATCH_COLUMNS = ["t", "lat", "lon", "raw_lat", "raw_lon", "osm_id", "road", "road_type", "shift_m", "matched"]


@dataclass(frozen=True)
class MatchConfig:
    radius_m: float = 30.0  # search radius around each fix
    max_candidates: int = 6
    sigma_m: float = 5.0  # GPS noise (emission)
    beta_m: float = 8.0  # tolerance for route vs straight-line distance (transition)
    min_step_m: float = 5.0  # fixes closer than this to the last used one are not matched separately
    detour_factor: float = 3.0  # give up on paths longer than this x straight line (+ 50 m)


def half_width(highway: str | None) -> float:
    if not isinstance(highway, str):
        return 3.0
    return HALF_WIDTH_M.get(highway.removesuffix("_link"), 3.0)


class RoadNetwork:
    """Road segments in local metres, with a graph over their end points."""

    def __init__(self, roads: list[dict], lat0: float, lon0: float):
        import networkx as nx
        from shapely import STRtree
        from shapely.geometry import LineString

        self.lat0, self.lon0 = lat0, lon0
        self.graph = nx.Graph()
        seg_a, seg_b, props, lines = [], [], [], []
        for f in roads:
            xy = np.asarray(f["geometry"]["coordinates"], float)
            east, north = to_enu(xy[:, 1], xy[:, 0], lat0, lon0)
            pts = np.column_stack([east, north])
            nodes = [(round(x, 2), round(y, 2)) for x, y in pts]  # shared OSM nodes -> same key
            for a, b in zip(nodes[:-1], nodes[1:]):
                if a == b:
                    continue
                length = float(np.hypot(b[0] - a[0], b[1] - a[1]))
                self.graph.add_edge(a, b, length=length)
                seg_a.append(a)
                seg_b.append(b)
                props.append(f["properties"])
                lines.append(LineString([a, b]))
        self.a = np.asarray(seg_a, float).reshape(-1, 2)
        self.b = np.asarray(seg_b, float).reshape(-1, 2)
        self.a_key, self.b_key = seg_a, seg_b
        self.props = props
        self.tree = STRtree(lines) if lines else None
        self._sp_cache: dict = {}

    @classmethod
    def from_geojson(cls, path: str | Path, lat0: float, lon0: float) -> "RoadNetwork":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")).get("features", []), lat0, lon0)

    def project(self, seg: int, p: np.ndarray) -> tuple[np.ndarray, float, float]:
        """Closest point on segment, its distance from p, and its offset along the segment."""
        a, b = self.a[seg], self.b[seg]
        ab = b - a
        u = float(np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-9), 0.0, 1.0))
        q = a + u * ab
        return q, float(np.hypot(*(p - q))), u * float(np.hypot(*ab))

    def candidates(self, p: np.ndarray, cfg: MatchConfig) -> list[tuple[int, np.ndarray, float, float]]:
        from shapely.geometry import box

        if self.tree is None:
            return []
        idx = self.tree.query(box(p[0] - cfg.radius_m, p[1] - cfg.radius_m, p[0] + cfg.radius_m, p[1] + cfg.radius_m))
        out = []
        for seg in idx:
            q, d, s = self.project(int(seg), p)
            if d <= cfg.radius_m:
                out.append((int(seg), q, d, s))
        out.sort(key=lambda c: c[2])
        return out[: cfg.max_candidates]

    def _dist_from(self, node, cutoff: float) -> dict:
        import networkx as nx

        key = (node, cutoff)
        if key not in self._sp_cache:
            if len(self._sp_cache) > 20000:
                self._sp_cache.clear()
            self._sp_cache[key] = nx.single_source_dijkstra_path_length(self.graph, node, cutoff=cutoff, weight="length")
        return self._sp_cache[key]

    def route_m(self, c1, c2, cutoff: float) -> float:
        """Shortest distance along roads between two candidate points (inf if not connected)."""
        s1, _, _, off1 = c1
        s2, _, _, off2 = c2
        len1 = float(np.hypot(*(self.b[s1] - self.a[s1])))
        len2 = float(np.hypot(*(self.b[s2] - self.a[s2])))
        if s1 == s2:
            return abs(off2 - off1)
        cutoff = float(np.ceil(cutoff / 25.0) * 25.0)  # bucket so the cache hits
        best = np.inf
        for n1, d1 in ((self.a_key[s1], off1), (self.b_key[s1], len1 - off1)):
            dist = self._dist_from(n1, cutoff)
            for n2, d2 in ((self.a_key[s2], off2), (self.b_key[s2], len2 - off2)):
                if n2 in dist:
                    best = min(best, d1 + dist[n2] + d2)
        return best


def match_track(gps: pd.DataFrame, network: RoadNetwork, cfg: MatchConfig = MatchConfig()) -> pd.DataFrame:
    """One row per GPS fix: corrected lat/lon, matched road, and how far the fix was moved."""
    east, north = to_enu(gps["lat"], gps["lon"], network.lat0, network.lon0)
    pts = np.column_stack([east, north])

    # Fixes that barely moved reuse the previous decision (stationary jitter confuses matching).
    used, last = [], None
    for i, p in enumerate(pts):
        if last is None or np.hypot(*(p - pts[last])) >= cfg.min_step_m:
            used.append(i)
            last = i
    owner = np.searchsorted(used, np.arange(len(pts)), side="right") - 1

    cands = [network.candidates(pts[i], cfg) for i in used]
    choice: list[tuple | None] = [None] * len(used)
    k = 0
    while k < len(used):
        if not cands[k]:
            k += 1
            continue
        # Viterbi over a run of steps that stay connected.
        score = [-0.5 * (c[2] / cfg.sigma_m) ** 2 for c in cands[k]]
        back = [[None] * len(cands[k])]
        steps = [k]
        j = k + 1
        while j < len(used) and cands[j]:
            gc = float(np.hypot(*(pts[used[j]] - pts[used[steps[-1]]])))
            cutoff = cfg.detour_factor * gc + 50.0
            prev = cands[steps[-1]]
            new_score, new_back = [], []
            for c in cands[j]:
                emit = -0.5 * (c[2] / cfg.sigma_m) ** 2
                best, arg = -np.inf, None
                for pi, pc in enumerate(prev):
                    if not np.isfinite(score[pi]):
                        continue
                    route = network.route_m(pc, c, cutoff)
                    if not np.isfinite(route):
                        continue
                    s = score[pi] - abs(route - gc) / cfg.beta_m
                    if s > best:
                        best, arg = s, pi
                new_score.append(best + emit)
                new_back.append(arg)
            if not np.isfinite(max(new_score)):
                break  # no connected path: end this run, start a new one here
            score, steps = new_score, steps + [j]
            back.append(new_back)
            j += 1
        # Backtrack.
        ci = int(np.argmax(score))
        for depth in range(len(steps) - 1, -1, -1):
            choice[steps[depth]] = cands[steps[depth]][ci]
            if depth:
                ci = back[depth][ci]
        k = steps[-1] + 1

    rows = []
    lat_scale = 1.0 / EARTH_RADIUS_M
    lon_scale = 1.0 / (EARTH_RADIUS_M * np.cos(np.radians(network.lat0)))
    for i in range(len(pts)):
        c = choice[owner[i]] if owner[i] >= 0 else None
        p = pts[i]
        raw_lat, raw_lon = float(gps["lat"].iloc[i]), float(gps["lon"].iloc[i])
        if c is None:
            rows.append((gps["t"].iloc[i], raw_lat, raw_lon, raw_lat, raw_lon, None, None, None, 0.0, False))
            continue
        seg = c[0]
        q, d, _ = network.project(seg, p)  # re-project this fix onto the chosen segment
        props = network.props[seg]
        limit = half_width(props.get("highway"))
        corrected = p if d <= limit else q + (p - q) * (limit / d)
        lat = network.lat0 + np.degrees(corrected[1] * lat_scale)
        lon = network.lon0 + np.degrees(corrected[0] * lon_scale)
        label = props.get("name") or props.get("ref") or f"unnamed {(props.get('highway') or 'road').replace('_', ' ')}"
        rows.append((gps["t"].iloc[i], float(lat), float(lon), raw_lat, raw_lon, props.get("osm_id"), label,
                     props.get("highway"), round(float(np.hypot(*(corrected - p))), 2), True))
    return pd.DataFrame(rows, columns=MATCH_COLUMNS)


def matched_gps(gps: pd.DataFrame, roads_path: str | Path, cfg: MatchConfig = MatchConfig()) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The session's gps table with lat/lon corrected, plus the full match table."""
    network = RoadNetwork.from_geojson(roads_path, float(gps["lat"].mean()), float(gps["lon"].mean()))
    match = match_track(gps, network, cfg)
    out = gps.copy()
    out["lat"], out["lon"] = match["lat"].to_numpy(), match["lon"].to_numpy()
    return out, match
