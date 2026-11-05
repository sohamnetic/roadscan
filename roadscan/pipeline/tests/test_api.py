import io
import shutil
import zipfile

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from roadscan.api import create_app  # noqa: E402
from roadscan.registry import Registry  # noqa: E402
from roadscan.service import process_session  # noqa: E402
from roadscan.synth import generate  # noqa: E402


@pytest.fixture(scope="module")
def rides(tmp_path_factory):
    return generate(tmp_path_factory.mktemp("rides"), rides=3, repair_after=2, seed=3)


@pytest.fixture()
def client(rides, tmp_path):
    db = tmp_path / "r.db"
    with Registry(db) as reg:
        for name in ("ride_01", "ride_02"):
            process_session(rides / name, reg)
    return TestClient(create_app(db, uploads=tmp_path / "uploads"))


def test_dashboard_and_listing(client):
    assert "RoadScan" in client.get("/").text
    fc = client.get("/api/potholes", params={"status": "open,confirmed"}).json()
    assert fc["type"] == "FeatureCollection" and fc["features"]
    prios = [f["properties"]["priority"] for f in fc["features"]]
    assert prios == sorted(prios, reverse=True)
    s = client.get("/api/summary").json()
    assert s["rides"] == 2 and s["open"] == len(fc["features"])


def test_detail_fix_and_workorder(client):
    first = client.get("/api/potholes", params={"status": "confirmed"}).json()["features"][0]["properties"]["id"]
    d = client.get(f"/api/potholes/{first}").json()
    assert d["properties"]["id"] == first and d["observations"]
    assert client.post(f"/api/potholes/{first}/fix").json()["status"] == "verified_fixed"
    assert client.get(f"/api/potholes/{first}").json()["properties"]["status"] == "verified_fixed"
    assert client.get("/api/potholes/999999").status_code == 404
    wo = client.get("/workorder").text
    assert "work order" in wo and f"#{first}<" not in wo  # fixed one left off the order
    assert client.get("/api/export/potholes.csv").text.startswith("id,")


def test_upload_processes_a_ride(client, rides, tmp_path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for f in (rides / "ride_03").iterdir():
            if f.suffix in (".json", ".csv"):
                zf.write(f, f"ride_03/{f.name}")
    r = client.post("/api/sessions", files={"file": ("ride_03.zip", buf.getvalue(), "application/zip")})
    assert r.status_code == 200
    jobs = client.get("/api/jobs").json()
    assert jobs[0]["state"] == "done", jobs[0]["message"]
    assert client.get("/api/summary").json()["rides"] == 3
    # Same ride again is refused by the registry, and reported, not crashed.
    client.post("/api/sessions", files={"file": ("again.zip", buf.getvalue(), "application/zip")})
    assert client.get("/api/jobs").json()[0]["state"] == "failed"
    assert client.post("/api/sessions", files={"file": ("x.zip", b"not a zip", "application/zip")}).status_code == 400


def _files(ride):
    out = [("files", (n, (ride / n).read_bytes())) for n in ("meta.json", "gps.csv", "imu.csv", "detections.csv")]
    return out


def test_phone_upload_as_files(client, rides):
    r = client.post("/api/sessions/files", files=_files(rides / "ride_03"))
    assert r.status_code == 200, r.text
    assert client.get("/api/jobs").json()[0]["state"] == "done"
    bad = client.post("/api/sessions/files", files=[("files", ("evil.sh", b"x")), *_files(rides / "ride_03")])
    assert bad.status_code == 400
    assert "RoadScan Capture" in client.get("/capture").text


def test_token_required_from_other_devices(rides, tmp_path):
    db = tmp_path / "t.db"
    with Registry(db) as reg:
        process_session(rides / "ride_01", reg)
    app = create_app(db, uploads=tmp_path / "up", token="s3cret")
    phone = TestClient(app, client=("192.168.1.20", 50000))
    assert phone.get("/api/summary").status_code == 401
    assert phone.get("/api/summary", params={"token": "nope"}).status_code == 401
    assert phone.get("/api/summary", params={"token": "s3cret"}).status_code == 200
    assert phone.get("/api/summary", headers={"X-RoadScan-Token": "s3cret"}).status_code == 200
    assert TestClient(app).get("/api/summary").status_code == 200  # the laptop itself
