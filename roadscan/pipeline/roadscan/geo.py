"""Geometry: lat/lon offsets, GPS track interpolation and ground-plane camera projection.

Camera convention: x right, y down, z forward in the image. The vehicle frame is
lateral (right of travel), forward, up. The camera sits `height_m` above a flat road,
pitched down by `pitch_deg`.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from .session import Camera

EARTH_RADIUS_M = 6_371_008.8


def offset_latlon(lat, lon, east_m, north_m):
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    lat2 = lat + np.degrees(np.asarray(north_m, float) / EARTH_RADIUS_M)
    lon2 = lon + np.degrees(np.asarray(east_m, float) / (EARTH_RADIUS_M * np.cos(np.radians(lat))))
    return lat2, lon2


def to_enu(lat, lon, lat0: float, lon0: float):
    """Local east/north metres around (lat0, lon0); fine for a few km."""
    east = np.radians(np.asarray(lon, float) - lon0) * EARTH_RADIUS_M * np.cos(np.radians(lat0))
    north = np.radians(np.asarray(lat, float) - lat0) * EARTH_RADIUS_M
    return east, north


def haversine_m(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(np.asarray(lon2, float) - np.asarray(lon1, float))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def cluster_points(lat, lon, eps_m: float) -> np.ndarray:
    """Single-linkage-style DBSCAN on the sphere; every point gets a cluster label."""
    lat = np.asarray(lat, float)
    if len(lat) == 0:
        return np.empty(0, int)
    coords = np.radians(np.column_stack([lat, np.asarray(lon, float)]))
    return DBSCAN(eps=eps_m / EARTH_RADIUS_M, min_samples=1, metric="haversine", algorithm="ball_tree").fit_predict(coords)


def pixel_to_ground(cam: Camera, u, v):
    """Pixel -> (forward_m, lateral_m) on the road plane; NaN at or above the horizon."""
    xn = (np.asarray(u, float) - cam.cx) / cam.fx
    yn = (np.asarray(v, float) - cam.cy) / cam.fy
    p = np.radians(cam.pitch_deg)
    ray_lateral = xn
    ray_forward = np.cos(p) - yn * np.sin(p)
    ray_up = -yn * np.cos(p) - np.sin(p)
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(ray_up < -1e-3, cam.height_m / -ray_up, np.nan)
    return scale * ray_forward, scale * ray_lateral


def ground_to_pixel(cam: Camera, forward, lateral):
    """Road-plane point -> pixel (u, v); NaN if behind the camera."""
    forward = np.asarray(forward, float)
    p = np.radians(cam.pitch_deg)
    xc = np.asarray(lateral, float)
    yc = -forward * np.sin(p) + cam.height_m * np.cos(p)
    zc = forward * np.cos(p) + cam.height_m * np.sin(p)
    with np.errstate(divide="ignore", invalid="ignore"):
        zc = np.where(zc > 1e-3, zc, np.nan)
        return cam.cx + cam.fx * xc / zc, cam.cy + cam.fy * yc / zc


def course_deg(lat, lon) -> np.ndarray:
    """Direction of travel between consecutive fixes, degrees clockwise from north."""
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    east, north = to_enu(lat, lon, lat[0], lon[0])
    course = np.degrees(np.arctan2(np.diff(east), np.diff(north))) % 360
    return np.append(course, course[-1])


class TrackPoint(NamedTuple):
    lat: np.ndarray
    lon: np.ndarray
    speed: np.ndarray
    bearing: np.ndarray
    valid: np.ndarray


class Track:
    """GPS fixes interpolated to arbitrary times."""

    def __init__(self, gps: pd.DataFrame):
        self.t = gps["t"].to_numpy(float)
        self.lat = gps["lat"].to_numpy(float)
        self.lon = gps["lon"].to_numpy(float)

        east, north = to_enu(self.lat, self.lon, self.lat[0], self.lon[0])
        step = np.hypot(np.diff(east), np.diff(north)) / np.maximum(np.diff(self.t), 1e-3)
        derived_speed = np.append(step, step[-1])
        speed = gps["speed_mps"].to_numpy(float)
        self.speed = np.where(np.isnan(speed), derived_speed, speed)

        # Phones report no bearing when stationary; fall back to the course between fixes.
        bearing = gps["bearing_deg"].to_numpy(float)
        bearing = np.where(np.isnan(bearing), course_deg(self.lat, self.lon), bearing)
        self._bearing_unwrapped = np.degrees(np.unwrap(np.radians(bearing)))

    def at(self, t) -> TrackPoint:
        t = np.asarray(t, float)
        return TrackPoint(
            lat=np.interp(t, self.t, self.lat),
            lon=np.interp(t, self.t, self.lon),
            speed=np.interp(t, self.t, self.speed),
            bearing=np.interp(t, self.t, self._bearing_unwrapped) % 360,
            valid=(t >= self.t[0]) & (t <= self.t[-1]),
        )

    def length_m(self) -> float:
        return float(np.sum(haversine_m(self.lat[:-1], self.lon[:-1], self.lat[1:], self.lon[1:])))

    def min_distance_m(self, lat: float, lon: float, min_speed: float = 0.0) -> float:
        """Closest approach of the track to a point, over segments driven at >= min_speed."""
        east, north = to_enu(self.lat, self.lon, lat, lon)
        p0 = np.column_stack([east[:-1], north[:-1]])
        d = np.column_stack([np.diff(east), np.diff(north)])
        moving = (self.speed[:-1] >= min_speed) & (self.speed[1:] >= min_speed)
        if not moving.any():
            return float("inf")
        p0, d = p0[moving], d[moving]
        length2 = np.sum(d * d, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = np.where(length2 > 0, np.clip(-np.sum(p0 * d, axis=1) / length2, 0, 1), 0)
        closest = p0 + frac[:, None] * d
        return float(np.min(np.hypot(closest[:, 0], closest[:, 1])))


def project_detections(det: pd.DataFrame, track: Track, cam: Camera) -> pd.DataFrame:
    """Place each bounding box on the map.

    Adds lat/lon (box centre on the road), forward_m/lateral_m from the camera,
    area_m2 (ellipse fitted to the box footprint), speed_mps, hit_t (when the
    vehicle reaches the spot at current speed) and valid.
    """
    out = det.copy()
    uc = (det["x1"].to_numpy(float) + det["x2"].to_numpy(float)) / 2
    vc = (det["y1"].to_numpy(float) + det["y2"].to_numpy(float)) / 2
    forward, lateral = pixel_to_ground(cam, uc, vc)

    far, _ = pixel_to_ground(cam, uc, det["y1"].to_numpy(float))
    near, _ = pixel_to_ground(cam, uc, det["y2"].to_numpy(float))
    _, left = pixel_to_ground(cam, det["x1"].to_numpy(float), vc)
    _, right = pixel_to_ground(cam, det["x2"].to_numpy(float), vc)
    area = np.pi / 4 * (right - left) * (far - near)

    t = det["t"].to_numpy(float)
    p = track.at(t)
    b = np.radians(p.bearing)
    east = forward * np.sin(b) + lateral * np.cos(b)
    north = forward * np.cos(b) - lateral * np.sin(b)
    out["lat"], out["lon"] = offset_latlon(p.lat, p.lon, east, north)
    out["forward_m"] = forward
    out["lateral_m"] = lateral
    out["area_m2"] = area
    out["speed_mps"] = p.speed
    with np.errstate(divide="ignore", invalid="ignore"):
        out["hit_t"] = np.where(p.speed >= 0.5, t + forward / p.speed, np.nan)
    out["valid"] = p.valid & np.isfinite(forward)
    return out
