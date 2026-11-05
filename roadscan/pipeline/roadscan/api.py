"""Web backend: the registry as a JSON API, plus the dashboard and ride uploads.

    python -m roadscan serve --db roadscan.db --places osm --config models/<model>.fusion.json

MVP scope: SQLite registry (PostGIS later), no user logins. It binds to 127.0.0.1 by default.
With `--lan` (for recording from a phone on the same Wi-Fi) every request that does not come
from this laptop must carry the access token printed at start-up (?token=... or X-RoadScan-Token).
"""
from __future__ import annotations

import hmac
import html
import io
import json
import re
import threading
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, Response

from .fusion import FusionConfig
from .registry import Registry

STATIC = Path(__file__).parent / "static"
ACTIVE = ("open", "confirmed")
SESSION_FILES = {"meta.json", "gps.csv", "imu.csv", "detections.csv", "video.mp4", "video.webm", "video.mov"}
LOOPBACK = {"127.0.0.1", "::1", "localhost", "testclient"}


def create_app(db: str | Path, places: str | Path | None = None, config: FusionConfig = FusionConfig(),
               uploads: str | Path = "uploads", weights: str | Path | None = None, token: str | None = None):
    from .places import tag_places, ward_summary
    from .render import _columns, to_geojson

    db, uploads = Path(db), Path(uploads)
    places = Path(places) if places else None
    roads = places / "roads.geojson" if places and (places / "roads.geojson").exists() else None
    wards = places / "wards.geojson" if places and (places / "wards.geojson").exists() else None
    jobs: dict[str, dict] = {}
    lock = threading.Lock()  # one ingest at a time: the registry merge is not concurrent-safe

    app = FastAPI(title="RoadScan", version="0.1")

    @app.middleware("http")
    async def require_token(request: Request, call_next):
        """Requests from other devices need the token; this laptop itself never does."""
        client = request.client.host if request.client else ""
        if token and client not in LOOPBACK:
            given = request.query_params.get("token") or request.headers.get("x-roadscan-token") or ""
            if not hmac.compare_digest(given, token):
                return PlainTextResponse("Missing or wrong access token: open the link printed by `roadscan serve`.", 401)
        return await call_next(request)

    def load() -> pd.DataFrame:
        with Registry(db) as reg:
            df = reg.potholes(as_of=reg.latest_session_end())
        return tag_places(df, roads, wards) if (roads or wards) else df

    def filtered(status: str | None, ward: str | None, min_severity: int) -> pd.DataFrame:
        df = load()
        if status:
            df = df[df["status"].isin(status.split(","))]
        if ward and "ward" in df:
            df = df[df["ward"] == ward]
        return df[df["severity"].fillna(0) >= min_severity]

    @app.get("/", response_class=HTMLResponse)
    def dashboard():
        return FileResponse(STATIC / "dashboard.html")

    @app.get("/capture", response_class=HTMLResponse)
    def capture():
        return FileResponse(STATIC / "capture.html")

    @app.get("/api/potholes")
    def potholes(status: str | None = None, ward: str | None = None, min_severity: int = 0):
        df = filtered(status, ward, min_severity).sort_values(["priority", "severity"], ascending=False)
        return JSONResponse(to_geojson(df))

    @app.get("/api/potholes/{pothole_id}")
    def pothole(pothole_id: int):
        df = load()
        row = df[df["id"] == pothole_id]
        if row.empty:
            raise HTTPException(404, f"no defect #{pothole_id}")
        with Registry(db) as reg:
            obs = reg.observations(pothole_id)
        feature = to_geojson(row)["features"][0]
        feature["observations"] = obs.drop(columns=["pothole_id"]).astype(object).where(obs.notna(), None).to_dict("records")
        return feature

    @app.post("/api/potholes/{pothole_id}/fix")
    def fix(pothole_id: int):
        with Registry(db) as reg:
            try:
                reg.mark_fixed(pothole_id)
            except KeyError:
                raise HTTPException(404, f"no defect #{pothole_id}") from None
        return {"id": pothole_id, "status": "verified_fixed"}

    @app.get("/api/wards")
    def wards_api():
        df = load()
        if "ward" not in df:
            return []
        return ward_summary(df).astype(object).to_dict("records")

    @app.get("/api/summary")
    def summary():
        df = load()
        active = df[df["status"].isin(ACTIVE)]
        with Registry(db) as reg:
            sessions = pd.read_sql_query("SELECT COUNT(*) AS n, SUM(distance_km) AS km FROM sessions", reg.conn).iloc[0]
            latest = reg.latest_session_end()
        return {
            "open": int(len(active)),
            "severe": int((active["severity"] >= 4).sum()),
            "aging": {f"over_{d}": int((active["days_open"] > d).sum()) for d in (30, 60, 90)},
            "fixed": int(df["status"].isin(["likely_fixed", "verified_fixed"]).sum()),
            "rides": int(sessions["n"] or 0),
            "km": round(float(sessions["km"] or 0), 1),
            "latest": latest.isoformat() if latest else None,
            "has_wards": bool("ward" in df and df["ward"].notna().any()),
        }

    @app.get("/api/export/potholes.csv")
    def export_csv(status: str | None = None, ward: str | None = None):
        df = filtered(status, ward, 0).sort_values(["priority", "severity"], ascending=False)
        return Response(df[_columns(df)].to_csv(index=False), media_type="text/csv",
                        headers={"Content-Disposition": "attachment; filename=potholes.csv"})

    @app.get("/api/export/potholes.geojson")
    def export_geojson(status: str | None = None, ward: str | None = None):
        df = filtered(status, ward, 0)
        return JSONResponse(to_geojson(df), headers={"Content-Disposition": "attachment; filename=potholes.geojson"})

    @app.get("/api/export/wards.csv")
    def export_wards():
        df = load()
        out = ward_summary(df) if "ward" in df else pd.DataFrame()
        return Response(out.to_csv(index=False), media_type="text/csv",
                        headers={"Content-Disposition": "attachment; filename=wards.csv"})

    @app.get("/workorder", response_class=HTMLResponse)
    def workorder(ward: str | None = None):
        df = filtered("open,confirmed", ward, 0).sort_values(["priority", "severity"], ascending=False)
        return work_order_html(df, ward)

    # ---------------------------------------------------------------- uploads
    @app.post("/api/sessions")
    async def upload(background: BackgroundTasks, file: UploadFile = File(...)):
        data = await file.read()
        try:
            zf = zipfile.ZipFile(io.BytesIO(data))
        except zipfile.BadZipFile:
            raise HTTPException(400, "upload a .zip of one session folder") from None
        metas = [n for n in zf.namelist() if Path(n).name == "meta.json"]
        if len(metas) != 1:
            raise HTTPException(400, "the zip must contain exactly one session (one meta.json)")
        job = uuid.uuid4().hex[:8]
        dest = uploads / f"upload_{job}"
        dest.mkdir(parents=True, exist_ok=True)
        zf.extractall(dest)  # zipfile drops absolute paths and ".." parts
        session_dir = dest / Path(metas[0]).parent
        jobs[job] = {"job": job, "file": file.filename, "state": "queued", "message": "", "at": _now()}
        background.add_task(_run_job, job, session_dir)
        return jobs[job]

    @app.post("/api/sessions/files")
    async def upload_files(background: BackgroundTasks, files: list[UploadFile] = File(...)):
        """A ride sent as separate files (the phone capture page): no zipping on the phone."""
        names = {Path(f.filename or "").name for f in files}
        if not {"meta.json", "gps.csv", "imu.csv"} <= names or not names <= SESSION_FILES:
            raise HTTPException(400, f"need meta.json, gps.csv, imu.csv (+ video); got {sorted(names)}")
        meta = next(f for f in files if Path(f.filename).name == "meta.json")
        try:
            session_id = json.loads(await meta.read())["session_id"]
        except Exception:
            raise HTTPException(400, "meta.json is not valid") from None
        await meta.seek(0)
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", session_id):
            raise HTTPException(400, "bad session_id")
        job = uuid.uuid4().hex[:8]
        session_dir = uploads / f"upload_{job}" / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        for f in files:
            name = Path(f.filename).name
            if name.startswith("video."):
                name = "video.mp4"  # one name for the pipeline; ffmpeg reads the real container
            with open(session_dir / name, "wb") as out:
                while chunk := await f.read(1 << 20):
                    out.write(chunk)
        jobs[job] = {"job": job, "file": session_id, "state": "queued", "message": "", "at": _now()}
        background.add_task(_run_job, job, session_dir)
        return jobs[job]

    @app.get("/api/jobs")
    def list_jobs():
        return sorted(jobs.values(), key=lambda j: j["at"], reverse=True)

    def _run_job(job: str, session_dir: Path) -> None:
        from .service import process_session

        try:
            if not (session_dir / "detections.csv").exists():
                if weights is None or not (session_dir / "video.mp4").exists():
                    raise ValueError("no detections.csv, and no video + --weights to make one")
                jobs[job].update(state="detecting")
                from .detect import run_yolo

                run_yolo(session_dir, weights)
            jobs[job].update(state="processing")
            with lock, Registry(db) as reg:
                result = process_session(session_dir, reg, config, roads)
            jobs[job].update(state="done", message=result["message"])
        except Exception as e:  # noqa: BLE001 - reported to the uploader
            jobs[job].update(state="failed", message=f"{type(e).__name__}: {e}")

    return app


def lan_addresses() -> list[str]:
    """IPv4 addresses of this machine's network adapters (the phone uses the Wi-Fi one)."""
    import socket

    found = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            found.add(info[4][0])
    except OSError:
        pass
    try:  # the adapter that routes to the internet: almost always the Wi-Fi one
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))  # UDP connect sends nothing
            primary = s.getsockname()[0]
    except OSError:
        primary = None
    ordered = ([primary] if primary else []) + sorted(found - {primary, "127.0.0.1"})
    return ordered


def self_signed_cert(folder: str | Path, addresses: list[str]) -> tuple[Path, Path]:
    """A self-signed HTTPS certificate for this laptop's LAN addresses (iPhone camera needs HTTPS)."""
    import ipaddress
    from datetime import timedelta

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    cert_path, key_path = folder / "roadscan-lan.crt", folder / "roadscan-lan.key"
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "RoadScan laptop")])
    sans = [x509.DNSName("localhost")] + [x509.IPAddress(ipaddress.ip_address(a)) for a in addresses + ["127.0.0.1"]]
    now = datetime.now(timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
            .serial_number(x509.random_serial_number()).not_valid_before(now - timedelta(days=1))
            .not_valid_after(now + timedelta(days=397))
            .add_extension(x509.SubjectAlternativeName(sans), critical=False)
            .sign(key, hashes.SHA256()))
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                           serialization.NoEncryption()))
    return cert_path, key_path


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def work_order_html(df: pd.DataFrame, ward: str | None) -> str:
    """A printable work order (browser Print -> Save as PDF)."""
    e = html.escape
    title = f"Road repair work order: {ward}" if ward else "Road repair work order: all wards"
    rows = []
    for n, (_, r) in enumerate(df.iterrows(), 1):
        maps = f"https://www.google.com/maps?q={r['lat']:.6f},{r['lon']:.6f}"
        where = " · ".join(e(str(v)) for v in (r.get("road"), r.get("ward") if not ward else None) if isinstance(v, str))
        rows.append(
            f"<tr><td>{n}</td><td>#{int(r['id'])}</td><td>{e(str(r['cls']).replace('_', ' '))}</td>"
            f"<td>{int(r['severity'])}/5</td><td>{where}</td>"
            f"<td><a href='{maps}'>{r['lat']:.5f}, {r['lon']:.5f}</a></td><td>{r['days_open']:.0f}</td>"
            f"<td></td><td></td></tr>"
        )
    body = "".join(rows) or "<tr><td colspan='9'>No open defects.</td></tr>"
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{e(title)}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
 body{{font:13px/1.4 system-ui,"Segoe UI",sans-serif;margin:24px;color:#1d232b;background:#fff}}
 h1{{font-size:18px;margin:0 0 4px}} p{{margin:0 0 14px;color:#5f6b7a}}
 table{{border-collapse:collapse;width:100%}} th,td{{border:1px solid #c9d0d8;padding:6px 8px;text-align:left;vertical-align:top}}
 th{{background:#f1f3f5}} td:nth-child(8),td:nth-child(9){{min-width:90px}}
 a{{color:#1d4ed8}} .print{{margin-bottom:12px}} @media print{{.print{{display:none}} body{{margin:10mm}}}}
</style></head><body>
<button class="print" onclick="window.print()">Print / save as PDF</button>
<h1>{e(title)}</h1>
<p>Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC · {len(df)} open defects, highest priority first · RoadScan</p>
<table><thead><tr><th>#</th><th>ID</th><th>Type</th><th>Severity</th><th>Road</th><th>Location</th><th>Days open</th>
<th>Repaired on</th><th>Signature</th></tr></thead><tbody>{body}</tbody></table></body></html>"""
