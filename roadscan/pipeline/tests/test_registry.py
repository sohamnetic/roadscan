from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from roadscan.fusion import CANDIDATE_COLUMNS
from roadscan.geo import offset_latlon
from roadscan.registry import Registry
from roadscan.session import Camera, Session

LAT0, LON0 = 12.9, 77.6
CAM = Camera(width=1920, height=1080, hfov_deg=66, vfov_deg=40, height_m=1.1, pitch_deg=12)


def _session(day: int) -> Session:
    lat, lon = offset_latlon(LAT0, LON0, np.linspace(0, 200, 21), np.zeros(21))
    gps = pd.DataFrame({"t": np.arange(21.0), "lat": lat, "lon": lon, "accuracy_m": 4.0, "speed_mps": 10.0, "bearing_deg": 90.0})
    return Session(path=None, session_id=f"s{day}", vehicle_type="bike", device="test",
                   start_time=datetime(2026, 9, 1, tzinfo=timezone.utc) + timedelta(days=day),
                   camera=CAM, video_t0=0.0, gps=gps, imu=pd.DataFrame(), detections=pd.DataFrame())


def _candidates(*east_m: float) -> pd.DataFrame:
    rows = []
    for e in east_m:
        lat, lon = offset_latlon(LAT0, LON0, e, 1.0)
        rows.append({"lat": float(lat), "lon": float(lon), "cls": "pothole", "source": "vision", "confirmed": True,
                     "severity": 3, "n_obs": 5, "max_conf": 0.8, "jolt_strength": np.nan, "area_m2": 0.4,
                     "forward_min_m": 4.0, "lateral_m": 1.0, "in_wheel_path": False, "t": e / 10})
    return pd.DataFrame(rows, columns=CANDIDATE_COLUMNS)


def _status(reg: Registry) -> dict:
    return dict(zip(reg.potholes()["id"], reg.potholes()["status"]))


def test_lifecycle(tmp_path):
    with Registry(tmp_path / "r.db") as reg:
        reg.ingest(_session(0), _candidates(50, 120))
        assert set(_status(reg).values()) == {"open"}

        reg.ingest(_session(1), _candidates(52, 120))  # 2 m off: same defects
        assert set(_status(reg).values()) == {"confirmed"}
        assert len(reg.potholes()) == 2

        for day in (2, 3, 4):  # the one at 50 m is repaired
            reg.ingest(_session(day), _candidates(120))
        status = reg.potholes().set_index("id")["status"]
        assert status[1] == "likely_fixed" and status[2] == "confirmed"

        reg.ingest(_session(5), _candidates(51, 120))  # it was not fixed after all
        assert reg.potholes().set_index("id").at[1, "status"] == "confirmed"

        reg.mark_fixed(1)
        assert reg.potholes().set_index("id").at[1, "status"] == "verified_fixed"


def test_single_sighting_is_dismissed(tmp_path):
    with Registry(tmp_path / "r.db") as reg:
        reg.ingest(_session(0), _candidates(80))
        reg.ingest(_session(1), _candidates())
        reg.ingest(_session(2), _candidates())
        assert _status(reg) == {1: "dismissed"}


def test_same_session_twice_is_rejected(tmp_path):
    with Registry(tmp_path / "r.db") as reg:
        reg.ingest(_session(0), _candidates(80))
        with pytest.raises(ValueError):
            reg.ingest(_session(0), _candidates(80))


def test_priority_grows_with_age(tmp_path):
    with Registry(tmp_path / "r.db") as reg:
        reg.ingest(_session(0), _candidates(80))
        later = datetime(2026, 10, 1, tzinfo=timezone.utc)
        df = reg.potholes(as_of=later)
        assert df.at[0, "days_open"] == pytest.approx(30, abs=0.1)
        assert df.at[0, "priority"] == pytest.approx(3 * 2, abs=0.05)
