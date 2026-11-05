"""Synthetic rides for exercising the pipeline end-to-end without real data.

Builds a ~4.4 km loop of road with potholes, cracks and lookalikes (speed breakers,
manholes, shadows), then rides it several times on a bike or an auto: GPS with
slowly drifting error, a tilted phone's accelerometer, and the boxes a YOLO model
might emit for each processed frame. Some potholes are repaired partway through.

These rides check that the code behaves as designed. They say nothing about
accuracy on real roads: the detector's behaviour here is an assumption.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, sosfilt
from scipy.spatial.transform import Rotation

from .geo import ground_to_pixel, offset_latlon
from .session import Camera, write_meta

ORIGIN = (12.9352, 77.6245)  # Bengaluru
LOOP_ENU = [(0, 0), (900, 0), (900, 700), (300, 700), (300, 1200), (-200, 1200), (-200, 300)]
CAMERA = Camera(width=1920, height=1080, hfov_deg=66.0, vfov_deg=40.0, height_m=1.1, pitch_deg=12.0)
VIDEO_FPS = 30
DETECT_EVERY = 6  # 5 processed frames per second
IMU_HZ = 100
CONDITIONS = ["day", "day", "night", "rain", "day", "day", "night", "day"]
CRACK_CLASSES = ["alligator_crack", "longitudinal_crack", "transverse_crack"]
# Wheel tracks relative to the vehicle centreline: (lateral_m, metres behind the front wheel)
WHEELS = {"bike": [(0.0, 0.0), (0.0, 1.3)], "auto": [(0.0, 0.0), (-0.6, 2.0), (0.6, 2.0)]}


@dataclass
class RoadObject:
    id: str
    kind: str  # pothole | crack | speed_breaker | manhole | shadow
    s: float  # metres along the loop
    lateral: float  # metres right of the centreline
    radius: float
    depth: float = 0.0
    crack_cls: str | None = None
    fixed_after_ride: int | None = None
    lat: float = 0.0
    lon: float = 0.0


class Route:
    def __init__(self, loop_enu=LOOP_ENU, smooth_m: float = 10.0):
        pts = np.array(list(loop_enu) + [loop_enu[0]], float)
        cum = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))])
        s = np.arange(0, cum[-1], 1.0)
        x = gaussian_filter1d(np.interp(s, cum, pts[:, 0]), smooth_m, mode="wrap")
        y = gaussian_filter1d(np.interp(s, cum, pts[:, 1]), smooth_m, mode="wrap")
        step = np.hypot(np.diff(x), np.diff(y))
        self.s = np.concatenate([[0], np.cumsum(step)])
        self.x, self.y = x, y
        dx = np.roll(x, -1) - np.roll(x, 1)
        dy = np.roll(y, -1) - np.roll(y, 1)
        self._heading = np.degrees(np.unwrap(np.arctan2(dx, dy)))
        self.length = float(self.s[-1])

    def heading(self, s) -> np.ndarray:
        """Direction of travel, degrees clockwise from north."""
        return np.interp(s, self.s, self._heading)

    def pos(self, s, lateral=0.0) -> tuple[np.ndarray, np.ndarray]:
        h = np.radians(self.heading(s))
        x = np.interp(s, self.s, self.x) + np.asarray(lateral) * np.cos(h)
        y = np.interp(s, self.s, self.y) - np.asarray(lateral) * np.sin(h)
        return x, y


def make_world(route: Route, rng: np.random.Generator, rides: int, repair_after: int | None) -> list[RoadObject]:
    counts = {"pothole": 30, "crack": 10, "speed_breaker": 6, "manhole": 6, "shadow": 15}
    taken: list[float] = []
    objects: list[RoadObject] = []
    for kind, n in counts.items():
        for i in range(n):
            while True:
                s = float(rng.uniform(30, route.length - 30))
                if all(abs(s - o) >= 15 for o in taken):
                    break
            taken.append(s)
            obj = RoadObject(id=f"{kind}-{i:02d}", kind=kind, s=s, lateral=float(rng.uniform(-3, 3)), radius=0.5)
            if kind == "pothole":
                obj.radius = float(rng.uniform(0.15, 0.8))
                obj.depth = float(rng.uniform(0.03, 0.15))
            elif kind == "crack":
                obj.radius = float(rng.uniform(0.5, 1.5))
                obj.crack_cls = str(rng.choice(CRACK_CLASSES))
            elif kind == "speed_breaker":
                obj.lateral, obj.radius = 0.0, 0.4
            elif kind == "manhole":
                obj.lateral = float(rng.uniform(-2.5, 2.5))
                obj.radius = 0.3
                obj.depth = float(rng.choice([0.0, 0.03]))
            x, y = route.pos(obj.s, obj.lateral)
            lat, lon = offset_latlon(ORIGIN[0], ORIGIN[1], x, y)
            obj.lat, obj.lon = float(lat), float(lon)
            objects.append(obj)

    if repair_after is not None and repair_after < rides:
        potholes = [o for o in objects if o.kind == "pothole"]
        for o in rng.choice(potholes, size=8, replace=False):
            o.fixed_after_ride = repair_after
    return objects


def _impulse(signal, t, t0, amp, freq, tau):
    i0 = int(np.searchsorted(t, t0))
    i1 = min(len(t), i0 + int(0.8 * IMU_HZ))
    dt = t[i0:i1] - t0
    signal[i0:i1] += amp * np.exp(-dt / tau) * np.sin(2 * np.pi * freq * dt)


def _detect(obj: RoadObject, condition: str, forward: float, waterlogged: bool, rng) -> tuple[str, float] | None:
    near = 1.0 if forward < 12 else 0.6
    if obj.kind == "pothole":
        if waterlogged:
            return ("puddle", rng.normal(0.55, 0.1)) if rng.random() < 0.7 * near else None
        p, mu = {"day": (0.8, 0.70), "night": (0.45, 0.42), "rain": (0.6, 0.50)}[condition]
        return ("pothole", rng.normal(mu, 0.12)) if rng.random() < p * near else None
    if obj.kind == "crack":
        p, mu = {"day": (0.6, 0.62), "night": (0.3, 0.45), "rain": (0.4, 0.5)}[condition]
        return (obj.crack_cls, rng.normal(mu, 0.12)) if rng.random() < p * near else None
    if obj.kind == "speed_breaker":
        if rng.random() > 0.85:
            return None
        return ("pothole", rng.normal(0.38, 0.08)) if rng.random() < 0.15 else ("speed_breaker", rng.normal(0.75, 0.1))
    if obj.kind == "manhole":
        if rng.random() > 0.7:
            return None
        return ("pothole", rng.normal(0.42, 0.1)) if rng.random() < 0.25 else ("manhole", rng.normal(0.7, 0.1))
    if obj.kind == "shadow" and condition == "day" and rng.random() < 0.35:
        return ("pothole", rng.normal(0.40, 0.12))
    return None


def simulate_ride(route: Route, objects: list[RoadObject], ride: int, rng, out_dir: Path) -> dict:
    vehicle = "bike" if ride % 2 else "auto"
    condition = CONDITIONS[(ride - 1) % len(CONDITIONS)]
    cam = replace(CAMERA, height_m=1.1 if vehicle == "bike" else 1.3)
    present = [o for o in objects if o.fixed_after_ride is None or ride <= o.fixed_after_ride]
    waterlogged = {o.id for o in present if o.kind == "pothole" and condition == "rain" and rng.random() < 0.5}

    # Speed along the loop: cruise with drift, slow for speed breakers and one signal stop.
    s_grid = np.arange(0, route.length - 5, 1.0)
    v = np.clip(9.0 + gaussian_filter1d(rng.normal(0, 15, len(s_grid)), 30), 4, 13)
    s_stop = 0.4 * route.length
    for o in present:
        if o.kind == "speed_breaker":
            d = s_grid - o.s
            near = (d > -30) & (d < 5)
            v[near] = np.minimum(v[near], 2.5 + 0.25 * np.abs(d[near]))
    d = s_grid - s_stop
    v = np.where((d > -40) & (d < 0), np.minimum(v, 1.0 + 0.2 * np.abs(d)), v)
    v = np.minimum(v, np.minimum(1.0 + 0.8 * s_grid, 1.0 + 0.8 * (s_grid[-1] - s_grid)))
    t_of_s = np.concatenate([[0], np.cumsum(1.0 / v[:-1])])
    i_stop = int(np.searchsorted(s_grid, s_stop))
    s_pts = np.concatenate([s_grid[: i_stop + 1], s_grid[i_stop:]])
    t_pts = np.concatenate([t_of_s[: i_stop + 1], t_of_s[i_stop:] + 20.0])
    duration = float(t_pts[-1])

    def s_at(t):
        return np.interp(t, t_pts, s_pts)

    t = np.arange(0, duration, 1 / IMU_HZ)
    s_t = s_at(t)
    speed_t = np.gradient(s_t, 1 / IMU_HZ)
    lane = rng.normal(0, 0.4)
    phase = rng.uniform(0, 2 * np.pi)
    wander_t = lane + 0.3 * np.sin(2 * np.pi * t / 40 + phase) + gaussian_filter1d(rng.normal(0, 5, len(t)), 200)

    def wander_at(tq):
        return np.interp(tq, t, wander_t)

    def pass_time(s):
        return float(np.interp(s, s_grid, t_of_s) + (20.0 if s > s_stop else 0.0))

    # IMU: road vibration plus impulses, seen through a tilted phone.
    band = sosfilt(butter(2, [2, 30], "bandpass", fs=IMU_HZ, output="sos"), rng.normal(0, 1, len(t)))
    vib = {"bike": 0.9, "auto": 0.7}[vehicle]
    dyn = band / band.std() * vib * np.sqrt(np.clip(speed_t / 8, 0.03, 2)) + 0.3 * np.sin(2 * np.pi * 35 * t)
    gain = 1.0 if vehicle == "bike" else 0.8
    for o in present:
        tp = pass_time(o.s)
        vp = max(float(np.interp(tp, t, speed_t)), 1.0)
        for wheel_lat, behind in WHEELS[vehicle]:
            th = tp + behind / vp
            hit = abs(wander_at(th) + wheel_lat - o.lateral) < o.radius + 0.1
            rear = 0.7 if behind else 1.0
            if o.kind == "speed_breaker":
                _impulse(dyn, t, th, rng.uniform(4, 6) * rear, 3.0, 0.25)
            elif o.kind in ("pothole", "manhole") and o.depth > 0 and hit:
                _impulse(dyn, t, th, -110 * o.depth * np.sqrt(vp / 8) * gain * rear, 12.0, 0.08)
    moving = np.flatnonzero(speed_t > 3)
    for i in rng.choice(moving, size=rng.poisson(route.length / 250), replace=False):
        _impulse(dyn, t, t[i], rng.uniform(2, 5) * gain, 10.0, 0.06)

    world = np.column_stack(
        [
            0.3 * rng.normal(0, 1, len(t)),
            np.gradient(gaussian_filter1d(speed_t, 0.5 * IMU_HZ), 1 / IMU_HZ) + 0.2 * rng.normal(0, 1, len(t)),
            9.81 + dyn,
        ]
    )
    mount = Rotation.from_euler("xyz", [rng.uniform(60, 85), rng.uniform(-8, 8), rng.uniform(-5, 5)], degrees=True)
    acc = world @ mount.as_matrix() + rng.normal(0, 0.05, world.shape)
    imu = pd.DataFrame({"t": t, "ax": acc[:, 0], "ay": acc[:, 1], "az": acc[:, 2]})
    for c in ("gx", "gy", "gz"):
        imu[c] = rng.normal(0, 0.02, len(t))

    # GPS at 1 Hz with slowly drifting bias plus jitter.
    t_gps = np.arange(0, duration, 1.0)
    x, y = route.pos(s_at(t_gps), wander_at(t_gps))
    sigma = 3.5 if condition == "rain" else 2.5
    rho = np.exp(-1 / 30)
    bias = np.zeros((len(t_gps), 2))
    bias[0] = rng.normal(0, sigma, 2)
    for k in range(1, len(t_gps)):
        bias[k] = rho * bias[k - 1] + np.sqrt(1 - rho**2) * rng.normal(0, sigma, 2)
    lat, lon = offset_latlon(ORIGIN[0], ORIGIN[1], x + bias[:, 0] + rng.normal(0, 0.8, len(x)), y + bias[:, 1] + rng.normal(0, 0.8, len(y)))
    gps_speed = np.clip(np.interp(t_gps, t, speed_t) + rng.normal(0, 0.3, len(t_gps)), 0, None)
    bearing = (route.heading(s_at(t_gps)) + rng.normal(0, 2, len(t_gps))) % 360
    gps = pd.DataFrame(
        {
            "t": t_gps,
            "lat": lat,
            "lon": lon,
            "accuracy_m": 3.5 + np.abs(rng.normal(0, 1.5, len(t_gps))),
            "speed_mps": gps_speed,
            "bearing_deg": np.where(gps_speed < 1.0, np.nan, bearing),
        }
    )

    # Detections on every 6th frame.
    video_t0 = 0.5
    obj_xy = {o.id: route.pos(o.s, o.lateral) for o in present}
    rows = []
    for frame in range(0, int((duration - video_t0) * VIDEO_FPS), DETECT_EVERY):
        tf = video_t0 + frame / VIDEO_FPS
        sf = float(s_at(tf))
        cx, cy = route.pos(sf, wander_at(tf))
        h = np.radians(route.heading(sf) + rng.normal(0, 1.0))
        fcam = replace(cam, pitch_deg=cam.pitch_deg + rng.normal(0, 0.4))
        for o in present:
            if abs(o.s - sf) > 30:
                continue
            ox, oy = obj_xy[o.id]
            rx, ry = ox - cx, oy - cy
            forward = rx * np.sin(h) + ry * np.cos(h)
            lateral = rx * np.cos(h) - ry * np.sin(h)
            if not 2.5 < forward < 22:
                continue
            u, vv = ground_to_pixel(
                fcam,
                forward + np.array([-1, -1, 1, 1]) * o.radius,
                lateral + np.array([-1, 1, -1, 1]) * o.radius,
            )
            if np.isnan(u).any():
                continue
            x1, x2 = np.clip([u.min(), u.max()], 0, cam.width)
            y1, y2 = np.clip([vv.min(), vv.max()], 0, cam.height)
            if x2 - x1 < 4 or y2 - y1 < 2:
                continue
            found = _detect(o, condition, forward, o.id in waterlogged, rng)
            if found is None:
                continue
            cls, conf = found
            jitter = rng.normal(0, 3, 4)
            rows.append((tf, frame, cls, conf, x1 + jitter[0], y1 + jitter[1], x2 + jitter[2], y2 + jitter[3]))
        if rng.random() < 0.03:
            bx, by = rng.uniform(200, 1600), rng.uniform(650, 950)
            rows.append((tf, frame, "pothole", rng.uniform(0.25, 0.5), bx, by, bx + rng.uniform(40, 160), by + rng.uniform(20, 80)))
    det = pd.DataFrame(rows, columns=["t", "frame", "cls", "conf", "x1", "y1", "x2", "y2"])
    det["conf"] = det["conf"].clip(0.05, 0.99)
    det = det[det["conf"] >= 0.25]

    session_id = f"ride_{ride:02d}"
    path = out_dir / session_id
    path.mkdir(parents=True, exist_ok=True)
    start = datetime(2026, 9, 1, 3, 30, tzinfo=timezone.utc) + timedelta(days=3 * (ride - 1), hours=12 if condition == "night" else 0)
    write_meta(path, session_id=session_id, vehicle_type=vehicle, device="synthetic", start_time=start, camera=cam, video_t0=video_t0)
    gps.to_csv(path / "gps.csv", index=False, float_format="%.7f")
    imu.to_csv(path / "imu.csv", index=False, float_format="%.4f")
    det.round({"t": 3, "conf": 3, "x1": 1, "y1": 1, "x2": 1, "y2": 1}).to_csv(path / "detections.csv", index=False)
    return {"session_id": session_id, "ride": ride, "vehicle_type": vehicle, "condition": condition, "start_time_utc": start.isoformat()}


def generate(out_dir: str | Path, rides: int = 6, repair_after: int | None = 3, seed: int = 7) -> Path:
    """Write ride_01 .. ride_NN session folders plus ground_truth.json."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    route = Route()
    objects = make_world(route, rng, rides, repair_after)
    ride_info = [simulate_ride(route, objects, r, rng, out_dir) for r in range(1, rides + 1)]
    truth = {"origin": ORIGIN, "loop_length_m": route.length, "rides": ride_info, "objects": [asdict(o) for o in objects]}
    (out_dir / "ground_truth.json").write_text(json.dumps(truth, indent=2), encoding="utf-8")
    return out_dir
