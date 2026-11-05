"""Turn one session's detections and jolts into road-damage candidates.

Per-frame boxes are placed on the map, clustered (one pothole is seen in many
frames), and each cluster is checked against the accelerometer: when a box sits
in the wheel path, the vehicle should feel a jolt at the moment it reaches it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, fields, replace
from pathlib import Path

import numpy as np
import pandas as pd

from .geo import Track, cluster_points, project_detections
from .session import Session

DAMAGE_CLASSES = ("pothole", "alligator_crack", "longitudinal_crack", "transverse_crack")
LINEAR_CRACKS = ("longitudinal_crack", "transverse_crack")
NEGATIVE_CLASSES = ("speed_breaker", "manhole", "patch", "puddle")

CANDIDATE_COLUMNS = [
    "lat", "lon", "cls", "source", "confirmed", "severity", "n_obs", "max_conf",
    "jolt_strength", "area_m2", "forward_min_m", "lateral_m", "in_wheel_path", "t",
]


@dataclass(frozen=True)
class FusionConfig:
    mode: str = "fusion"  # "fusion" (vision + IMU) or "vision" (baseline, boxes only)
    min_forward_m: float = 2.0
    max_forward_m: float = 25.0  # beyond this a pixel spans too much road
    cluster_eps_m: float = 5.0
    jolt_window_s: float = 0.6
    wheel_path_m: float = 0.9
    high_conf: float = 0.6  # confirms on vision alone
    medium_conf: float = 0.3  # confirms only with a matching jolt
    crack_conf: float = 0.5
    vision_conf: float = 0.5  # single threshold used in "vision" mode
    min_frames: int = 2
    imu_only_strength: float = 1.5  # unexplained jolts this strong are reported, unconfirmed


def load_config(path: str | Path) -> FusionConfig:
    """FusionConfig with overrides from a JSON file, e.g. thresholds calibrated for one model.

    Keys starting with "_" are comments; any other unknown key is an error.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    known = {f.name for f in fields(FusionConfig)}
    overrides = {k: v for k, v in raw.items() if not k.startswith("_")}
    unknown = sorted(set(overrides) - known)
    if unknown:
        raise ValueError(f"unknown FusionConfig keys in {path}: {unknown}")
    return replace(FusionConfig(), **overrides)


def severity_score(cls: str, area_m2: float, jolt_strength: float) -> int:
    """1 (cosmetic) to 5 (dangerous), from footprint and, when felt, the jolt."""
    if cls not in DAMAGE_CLASSES:
        return 0
    area = 2 if np.isnan(area_m2) else 1 + int(np.searchsorted([0.1, 0.3, 0.7, 1.5], area_m2))
    if cls in LINEAR_CRACKS:
        return min(area, 2)
    if np.isnan(jolt_strength):
        return area
    return max(area, 2 + int(np.searchsorted([1.5, 2.5, 4.0], jolt_strength)))


def _clusters(session: Session, track: Track, config: FusionConfig) -> pd.DataFrame:
    if session.detections.empty:
        return pd.DataFrame()
    det = project_detections(session.detections, track, session.camera)
    det = det[det["valid"] & det["forward_m"].between(config.min_forward_m, config.max_forward_m)]
    if det.empty:
        return pd.DataFrame()
    det = det.assign(label=cluster_points(det["lat"], det["lon"], config.cluster_eps_m))

    rows = []
    for _, g in det.groupby("label"):
        cls = g.groupby("cls")["conf"].sum().idxmax()
        own = g[g["cls"] == cls]
        weight = g["conf"] / g["forward_m"].clip(lower=3.0)  # near, confident boxes place it best
        closest = g.loc[g["forward_m"].idxmin()]
        rows.append(
            {
                "lat": float(np.average(g["lat"], weights=weight)),
                "lon": float(np.average(g["lon"], weights=weight)),
                "cls": cls,
                "n_obs": len(own),
                "max_conf": float(own["conf"].max()),
                "area_m2": float(g.nsmallest(3, "forward_m")["area_m2"].median()),
                "forward_min_m": float(closest["forward_m"]),
                "lateral_m": float(closest["lateral_m"]),
                "hit_t": float(closest["hit_t"]),
                "t": float(g["t"].max()),
            }
        )
    return pd.DataFrame(rows)


def _match_jolts(clusters: pd.DataFrame, jolts: pd.DataFrame, config: FusionConfig) -> set[int]:
    """Give each cluster in the wheel path the nearest-in-time jolt; returns used jolt indices."""
    clusters["jolt_strength"] = np.nan
    clusters["in_wheel_path"] = clusters["lateral_m"].abs() <= config.wheel_path_m
    if clusters.empty or jolts.empty:
        return set()
    eligible = clusters.index[clusters["in_wheel_path"] & np.isfinite(clusters["hit_t"])]
    pairs = []
    for ci in eligible:
        dt = np.abs(jolts["t"].to_numpy() - clusters.at[ci, "hit_t"])
        for ji in np.flatnonzero(dt <= config.jolt_window_s):
            pairs.append((dt[ji], ci, jolts.index[ji]))
    used_c, used_j = set(), set()
    for _, ci, ji in sorted(pairs):
        if ci in used_c or ji in used_j:
            continue
        used_c.add(ci)
        used_j.add(ji)
        clusters.at[ci, "jolt_strength"] = jolts.at[ji, "strength"]
    return used_j


def _confirm(row, config: FusionConfig) -> bool:
    cls, conf, frames = row["cls"], row["max_conf"], row["n_obs"]
    if cls not in DAMAGE_CLASSES:
        return False
    if config.mode == "vision":
        return conf >= config.vision_conf and frames >= config.min_frames
    if cls == "pothole":
        felt = not np.isnan(row["jolt_strength"])
        return (conf >= config.high_conf and frames >= config.min_frames) or (conf >= config.medium_conf and felt)
    return conf >= config.crack_conf and frames >= config.min_frames


def fuse(session: Session, jolts: pd.DataFrame, config: FusionConfig = FusionConfig()) -> pd.DataFrame:
    """Candidates for one session; `confirmed` marks the ones worth putting on the map."""
    track = Track(session.gps)
    clusters = _clusters(session, track, config)
    if clusters.empty:
        clusters = pd.DataFrame(columns=CANDIDATE_COLUMNS + ["hit_t"])
    jolts = jolts.reset_index(drop=True)
    used = _match_jolts(clusters, jolts, config) if config.mode == "fusion" else set()
    if config.mode != "fusion":
        clusters["jolt_strength"] = np.nan
        clusters["in_wheel_path"] = clusters["lateral_m"].abs() <= config.wheel_path_m

    felt = clusters["jolt_strength"].notna()
    clusters["source"] = np.where(felt, "vision+imu", "vision")
    # A "puddle" the wheel drops into is a waterlogged pothole: common in the monsoon.
    submerged = (clusters["cls"] == "puddle") & felt
    clusters.loc[submerged, "cls"] = "pothole"
    clusters["confirmed"] = [_confirm(r, config) for _, r in clusters.iterrows()]
    hit_t = clusters["hit_t"].astype(float)
    clusters["t"] = hit_t.where(np.isfinite(hit_t), clusters["t"])

    frames = [clusters]
    if config.mode == "fusion" and not jolts.empty:
        unexplained = jolts[~jolts.index.isin(used) & (jolts["strength"] >= config.imu_only_strength)]
        if not unexplained.empty:
            p = track.at(unexplained["t"].to_numpy())
            frames.append(
                pd.DataFrame(
                    {
                        "lat": p.lat,
                        "lon": p.lon,
                        "cls": "pothole",
                        "source": "imu",
                        "confirmed": False,
                        "n_obs": 0,
                        "max_conf": np.nan,
                        "jolt_strength": unexplained["strength"].to_numpy(),
                        "area_m2": np.nan,
                        "forward_min_m": np.nan,
                        "lateral_m": 0.0,
                        "in_wheel_path": True,
                        "t": unexplained["t"].to_numpy(),
                    }
                )
            )
    non_empty = [f for f in frames if not f.empty]
    out = pd.concat(non_empty, ignore_index=True) if non_empty else clusters
    out["severity"] = [severity_score(r["cls"], r["area_m2"], r["jolt_strength"]) for _, r in out.iterrows()]
    out = out.reindex(columns=CANDIDATE_COLUMNS).reset_index(drop=True)
    return out.astype({"confirmed": bool, "in_wheel_path": bool})
