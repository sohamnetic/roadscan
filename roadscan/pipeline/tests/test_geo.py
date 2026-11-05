import numpy as np
import pandas as pd

from roadscan.geo import Track, ground_to_pixel, haversine_m, offset_latlon, pixel_to_ground
from roadscan.session import Camera

CAM = Camera(width=1920, height=1080, hfov_deg=66, vfov_deg=40, height_m=1.1, pitch_deg=12)


def test_pixel_ground_round_trip():
    forward = np.array([3.0, 6.0, 12.0, 20.0])
    lateral = np.array([-2.0, 0.0, 1.5, 3.0])
    u, v = ground_to_pixel(CAM, forward, lateral)
    f2, l2 = pixel_to_ground(CAM, u, v)
    np.testing.assert_allclose(f2, forward, atol=1e-6)
    np.testing.assert_allclose(l2, lateral, atol=1e-6)


def test_pixels_above_horizon_are_nan():
    f, l = pixel_to_ground(CAM, [960], [0])
    assert np.isnan(f[0]) and np.isnan(l[0])


def test_offset_matches_haversine():
    lat, lon = offset_latlon(12.9, 77.6, 30.0, 40.0)
    assert abs(haversine_m(12.9, 77.6, lat, lon) - 50.0) < 0.05


def test_track_interpolates_bearing_across_north():
    gps = pd.DataFrame({"t": [0.0, 1.0], "lat": [12.9, 12.9001], "lon": [77.6, 77.6],
                        "accuracy_m": [4, 4], "speed_mps": [10, 10], "bearing_deg": [350.0, 10.0]})
    p = Track(gps).at([0.5])
    assert abs(p.bearing[0]) < 1e-6 or abs(p.bearing[0] - 360) < 1e-6


def test_min_distance_to_track():
    lat, lon = offset_latlon(12.9, 77.6, np.array([0.0, 100.0]), np.array([0.0, 0.0]))
    gps = pd.DataFrame({"t": [0.0, 10.0], "lat": lat, "lon": lon, "accuracy_m": 4,
                        "speed_mps": 10.0, "bearing_deg": 90.0})
    plat, plon = offset_latlon(12.9, 77.6, 50.0, 6.0)
    assert abs(Track(gps).min_distance_m(float(plat), float(plon)) - 6.0) < 0.05
