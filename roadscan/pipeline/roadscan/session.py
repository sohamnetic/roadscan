"""Ride session folders: the contract between the capture app and the pipeline.

A session folder holds:
  meta.json        session id, vehicle type, start time, camera mount geometry
  gps.csv          t, lat, lon, accuracy_m, speed_mps, bearing_deg
  imu.csv          t, ax, ay, az, gx, gy, gz   (m/s^2 including gravity, rad/s)
  detections.csv   t, frame, cls, conf, x1, y1, x2, y2   (pixels; written by `roadscan detect`)
  video.mp4        optional; the source for detections.csv

`t` is seconds since session start, on one monotonic clock shared by every file.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

GPS_COLUMNS = ["t", "lat", "lon", "accuracy_m", "speed_mps", "bearing_deg"]
IMU_COLUMNS = ["t", "ax", "ay", "az", "gx", "gy", "gz"]
DETECTION_COLUMNS = ["t", "frame", "cls", "conf", "x1", "y1", "x2", "y2"]
VEHICLE_TYPES = ("bike", "auto")
MAX_GPS_ACCURACY_M = 30.0


@dataclass(frozen=True)
class Camera:
    """Pinhole camera on a mount: image size, field of view, height above road, downward pitch."""

    width: int
    height: int
    hfov_deg: float
    vfov_deg: float
    height_m: float
    pitch_deg: float

    @property
    def fx(self) -> float:
        return (self.width / 2) / math.tan(math.radians(self.hfov_deg) / 2)

    @property
    def fy(self) -> float:
        return (self.height / 2) / math.tan(math.radians(self.vfov_deg) / 2)

    @property
    def cx(self) -> float:
        return self.width / 2

    @property
    def cy(self) -> float:
        return self.height / 2


@dataclass
class Session:
    path: Path
    session_id: str
    vehicle_type: str
    device: str
    start_time: datetime
    camera: Camera
    video_t0: float
    gps: pd.DataFrame
    imu: pd.DataFrame
    detections: pd.DataFrame

    def time_at(self, t: float) -> datetime:
        return self.start_time + timedelta(seconds=float(t))

    @property
    def end_time(self) -> datetime:
        return self.time_at(self.gps["t"].iloc[-1])


def _read_csv(path: Path, columns: list[str], required: bool = True) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return pd.DataFrame({c: pd.Series(dtype="object" if c == "cls" else "float") for c in columns})
    df = pd.read_csv(path)
    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
    return df[columns].sort_values("t", kind="stable").reset_index(drop=True)


def load_session(path: str | Path) -> Session:
    path = Path(path)
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    vehicle = meta.get("vehicle_type", "bike")
    if vehicle not in VEHICLE_TYPES:
        raise ValueError(f"vehicle_type must be one of {VEHICLE_TYPES}, got {vehicle!r}")

    gps = _read_csv(path / "gps.csv", GPS_COLUMNS)
    gps = gps[gps["accuracy_m"].fillna(0) <= MAX_GPS_ACCURACY_M].reset_index(drop=True)
    if len(gps) < 2:
        raise ValueError(f"{path}: fewer than 2 usable GPS fixes")

    return Session(
        path=path,
        session_id=meta["session_id"],
        vehicle_type=vehicle,
        device=meta.get("device", ""),
        start_time=datetime.fromisoformat(meta["start_time_utc"].replace("Z", "+00:00")),
        camera=Camera(**meta["camera"]),
        video_t0=float(meta.get("video_t0", 0.0)),
        gps=gps,
        imu=_read_csv(path / "imu.csv", IMU_COLUMNS),
        detections=_read_csv(path / "detections.csv", DETECTION_COLUMNS, required=False),
    )


def write_meta(
    path: str | Path,
    *,
    session_id: str,
    vehicle_type: str,
    device: str,
    start_time: datetime,
    camera: Camera,
    video_t0: float = 0.0,
) -> None:
    meta = {
        "session_id": session_id,
        "vehicle_type": vehicle_type,
        "device": device,
        "start_time_utc": start_time.isoformat(),
        "camera": asdict(camera),
        "video_t0": video_t0,
    }
    Path(path, "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
