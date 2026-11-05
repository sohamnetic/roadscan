import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("shapely")

from roadscan.geo import offset_latlon  # noqa: E402
from roadscan.places import tag_places, ward_summary  # noqa: E402

LAT0, LON0 = 12.9352, 77.6245


def _ll(east, north):
    lat, lon = offset_latlon(LAT0, LON0, east, north)
    return [float(lon), float(lat)]


@pytest.fixture()
def places(tmp_path):
    # An east-west main road at north=0 and a north-south lane at east=100.
    roads = [
        {"type": "Feature", "geometry": {"type": "LineString", "coordinates": [_ll(-500, 0), _ll(500, 0)]},
         "properties": {"osm_id": 1, "highway": "primary", "name": "80 Feet Road", "ref": None}},
        {"type": "Feature", "geometry": {"type": "LineString", "coordinates": [_ll(100, 10), _ll(100, 500)]},
         "properties": {"osm_id": 2, "highway": "residential", "name": None, "ref": None}},
    ]
    # Two wards split at east=0.
    west = [_ll(-600, -600), _ll(0, -600), _ll(0, 600), _ll(-600, 600), _ll(-600, -600)]
    east = [_ll(0, -600), _ll(600, -600), _ll(600, 600), _ll(0, 600), _ll(0, -600)]
    wards = [
        {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [west]}, "properties": {"name": "West Ward"}},
        {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [east]}, "properties": {"name": "East Ward"}},
    ]
    r, w = tmp_path / "roads.geojson", tmp_path / "wards.geojson"
    r.write_text(json.dumps({"type": "FeatureCollection", "features": roads}))
    w.write_text(json.dumps({"type": "FeatureCollection", "features": wards}))
    return r, w


def _potholes(points):
    rows = []
    for i, (e, n) in enumerate(points, 1):
        lon, lat = _ll(e, n)
        rows.append({"id": i, "lat": lat, "lon": lon, "status": "confirmed", "severity": 4, "days_open": 40, "priority": 10.0})
    return pd.DataFrame(rows)


def test_tags_road_ward_and_scales_priority(places):
    df = tag_places(_potholes([(-200, 3), (103, 200), (300, 300)]), *places)
    assert df["road"].tolist()[:2] == ["80 Feet Road", "unnamed residential"]
    assert df["road"].isna().tolist() == [False, False, True]  # 3rd is 200 m from any road
    assert df["road_dist_m"].iloc[0] == pytest.approx(3, abs=0.5)
    assert df["ward"].tolist() == ["West Ward", "East Ward", "East Ward"]
    assert df["priority"].tolist() == [14.0, 10.0, 10.0]  # primary road x1.4


def test_ward_summary(places):
    df = tag_places(_potholes([(-200, 3), (103, 200), (300, 300)]), *places)
    summary = ward_summary(df).set_index("ward")
    assert summary.loc["East Ward", "open"] == 2 and summary.loc["West Ward", "open"] == 1
    assert summary.index[0] == "East Ward"  # highest total priority first
    assert summary.loc["West Ward", "open_over_30_days"] == 1


def test_missing_files_leave_columns_empty():
    df = tag_places(_potholes([(0, 0)]), None, None)
    assert df["road"].isna().all() and df["ward"].isna().all() and df["priority"].tolist() == [10.0]
    assert np.isnan(df["road_dist_m"].iloc[0])
