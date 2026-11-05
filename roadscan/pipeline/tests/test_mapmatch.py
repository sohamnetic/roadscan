import numpy as np
import pandas as pd
import pytest

pytest.importorskip("shapely")
pytest.importorskip("networkx")

from roadscan.geo import offset_latlon, to_enu  # noqa: E402
from roadscan.mapmatch import RoadNetwork, match_track  # noqa: E402

LAT0, LON0 = 12.9352, 77.6245


def _line(osm_id, highway, pts):
    coords = []
    for e, n in pts:
        lat, lon = offset_latlon(LAT0, LON0, e, n)
        coords.append([float(lon), float(lat)])
    return {"type": "Feature", "geometry": {"type": "LineString", "coordinates": coords},
            "properties": {"osm_id": osm_id, "highway": highway, "name": f"road {osm_id}", "ref": None}}


def _gps(east, north):
    lat, lon = offset_latlon(LAT0, LON0, np.asarray(east), np.asarray(north))
    n = len(east)
    return pd.DataFrame({"t": np.arange(n, dtype=float), "lat": lat, "lon": lon, "accuracy_m": 5.0,
                         "speed_mps": 8.0, "bearing_deg": np.nan})


@pytest.fixture()
def parallel_roads():
    # Main road at north=0, a lane 25 m north; joined only at the far ends.
    roads = [
        _line(1, "primary", [(-500, 0), (0, 0), (500, 0)]),
        _line(2, "residential", [(-500, 25), (0, 25), (500, 25)]),
        _line(3, "residential", [(-500, 0), (-500, 25)]),
        _line(4, "residential", [(500, 0), (500, 25)]),
    ]
    return RoadNetwork(roads, LAT0, LON0)


def test_drift_toward_parallel_road_stays_on_true_road(parallel_roads):
    rng = np.random.default_rng(1)
    east = np.arange(-400, 400, 8.0)
    north = rng.normal(0, 2.5, len(east))
    north[40:55] += 16  # GPS drifts 16 m north: now nearer the lane (9 m) than the main road
    match = match_track(_gps(east, north), parallel_roads)
    assert match["matched"].all()
    assert set(match["osm_id"]) == {1}
    e, n = to_enu(match["lat"], match["lon"], LAT0, LON0)
    assert np.abs(n).max() <= 6.0 + 1e-6  # clamped into the primary road's corridor


def test_fix_inside_corridor_keeps_lane_position(parallel_roads):
    east = np.arange(-200, 200, 8.0)
    north = np.full(len(east), -3.0)  # driving in the left lane, 3 m from the centre line
    match = match_track(_gps(east, north), parallel_roads)
    assert match["shift_m"].max() < 1e-6


def test_follows_a_turn():
    roads = [_line(10, "secondary", [(0, 0), (300, 0)]), _line(11, "tertiary", [(300, 0), (300, 300)])]
    net = RoadNetwork(roads, LAT0, LON0)
    east = np.r_[np.arange(0, 300, 8.0), np.full(30, 300.0)]
    north = np.r_[np.zeros(38), np.arange(8, 248, 8.0)]
    rng = np.random.default_rng(2)
    match = match_track(_gps(east + rng.normal(0, 2, len(east)), north + rng.normal(0, 2, len(east))), net)
    assert (match["osm_id"].iloc[:30] == 10).all() and (match["osm_id"].iloc[-25:] == 11).all()


def test_far_from_any_road_is_left_alone(parallel_roads):
    match = match_track(_gps([0.0, 8.0, 16.0], [200.0, 200.0, 200.0]), parallel_roads)
    assert not match["matched"].any()
    assert np.allclose(match["lat"], match["raw_lat"])
