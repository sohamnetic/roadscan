import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from roadscan.geo import Track
from roadscan.imu import detect_jolts


def _ride(jolt_times, seed=0, duration=60.0):
    rng = np.random.default_rng(seed)
    t = np.arange(0, duration, 0.01)
    vertical = rng.normal(0, 0.4, len(t))
    for t0 in jolt_times:
        dt = t - t0
        on = (dt >= 0) & (dt < 0.6)
        vertical[on] += -9.0 * np.exp(-dt[on] / 0.08) * np.sin(2 * np.pi * 12 * dt[on])
    world = np.column_stack([rng.normal(0, 0.2, len(t)), rng.normal(0, 0.2, len(t)), 9.81 + vertical])
    acc = world @ Rotation.from_euler("xyz", [72, 5, -3], degrees=True).as_matrix()
    imu = pd.DataFrame({"t": t, "ax": acc[:, 0], "ay": acc[:, 1], "az": acc[:, 2], "gx": 0.0, "gy": 0.0, "gz": 0.0})
    gps = pd.DataFrame({"t": np.arange(0, duration + 1), "lat": 12.9, "lon": np.linspace(77.6, 77.605, int(duration) + 1),
                        "accuracy_m": 4.0, "speed_mps": 8.0, "bearing_deg": 90.0})
    return imu, Track(gps)


def test_finds_jolts_on_a_tilted_phone():
    imu, track = _ride([10.0, 25.0, 41.5])
    jolts = detect_jolts(imu, track, "bike")
    assert len(jolts) == 3
    np.testing.assert_allclose(jolts["t"], [10.0, 25.0, 41.5], atol=0.1)
    assert (jolts["strength"] > 1).all()


def test_quiet_road_has_no_jolts():
    imu, track = _ride([])
    assert detect_jolts(imu, track, "bike").empty
