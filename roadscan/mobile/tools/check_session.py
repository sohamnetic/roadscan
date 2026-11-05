"""Check that a session folder recorded by the capture app matches the pipeline's format.

    python check_session.py <session_dir> [--pipeline ../../pipeline]

Standard library only. Uses ffprobe for the video checks if it is on PATH. With --pipeline,
also loads the session through roadscan.session.load_session (needs pandas).
Exit code 1 if any check FAILs; WARNs are things worth a look but not format errors.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path

GPS_COLUMNS = ["t", "lat", "lon", "accuracy_m", "speed_mps", "bearing_deg"]
IMU_COLUMNS = ["t", "ax", "ay", "az", "gx", "gy", "gz"]
META_KEYS = {"session_id", "vehicle_type", "device", "start_time_utc", "camera", "video_t0"}
CAMERA_KEYS = {"width", "height", "hfov_deg", "vfov_deg", "height_m", "pitch_deg"}


class Report:
    def __init__(self):
        self.failed = False

    def ok(self, msg):
        print(f"  ok    {msg}")

    def warn(self, msg):
        print(f"  WARN  {msg}")

    def fail(self, msg):
        self.failed = True
        print(f"  FAIL  {msg}")

    def check(self, cond, msg, warn_only=False):
        if cond:
            self.ok(msg)
        elif warn_only:
            self.warn(msg)
        else:
            self.fail(msg)
        return cond


def is_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def check_meta(path: Path, r: Report) -> dict | None:
    print("meta.json")
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        r.fail(f"cannot read: {e}")
        return None
    r.check(set(meta) == META_KEYS, f"top-level keys are exactly {sorted(META_KEYS)} (got {sorted(meta)})")
    r.check(isinstance(meta.get("session_id"), str) and meta["session_id"], "session_id is a non-empty string")
    r.check(meta.get("session_id") == path.parent.name, "session_id matches the folder name", warn_only=True)
    r.check(meta.get("vehicle_type") in ("bike", "auto"), f"vehicle_type is bike|auto ({meta.get('vehicle_type')!r})")
    r.check(isinstance(meta.get("device"), str), f"device is a string ({meta.get('device')!r})")
    try:
        start = datetime.fromisoformat(str(meta.get("start_time_utc")).replace("Z", "+00:00"))
        r.check(start.utcoffset() is not None and start.utcoffset().total_seconds() == 0,
                f"start_time_utc is ISO 8601 UTC ({meta['start_time_utc']})")
    except ValueError:
        r.fail(f"start_time_utc does not parse as ISO 8601 ({meta.get('start_time_utc')!r})")
    cam = meta.get("camera")
    if not isinstance(cam, dict):
        r.fail("camera is an object")
    else:
        r.check(set(cam) == CAMERA_KEYS, f"camera keys are exactly {sorted(CAMERA_KEYS)} (got {sorted(cam)})")
        r.check(all(isinstance(cam.get(k), int) and cam[k] > 0 for k in ("width", "height")),
                f"camera width/height are positive ints ({cam.get('width')}x{cam.get('height')})")
        r.check(all(is_number(cam.get(k)) for k in CAMERA_KEYS - {"width", "height"}), "camera angles/height are numbers")
        if all(is_number(cam.get(k)) for k in ("hfov_deg", "vfov_deg", "height_m", "pitch_deg")):
            r.check(30 <= cam["hfov_deg"] <= 120 and 15 <= cam["vfov_deg"] <= 100,
                    f"FOV is plausible ({cam['hfov_deg']} deg x {cam['vfov_deg']} deg)", warn_only=True)
            r.check(cam["vfov_deg"] < cam["hfov_deg"], "vfov < hfov (landscape video)", warn_only=True)
            r.check(0.3 <= cam["height_m"] <= 3, f"height_m is plausible ({cam['height_m']} m)", warn_only=True)
            r.check(0 <= cam["pitch_deg"] <= 60, f"pitch_deg is plausible ({cam['pitch_deg']} deg)", warn_only=True)
    t0 = meta.get("video_t0")
    if r.check(is_number(t0), f"video_t0 is a number ({t0})"):
        r.check(0 < t0 < 10, "video_t0 is within the first 10 s (0 means it was never measured)", warn_only=True)
    return meta


def read_csv(path: Path, columns: list[str], r: Report) -> list[dict] | None:
    print(path.name)
    try:
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not r.check(header == columns, f"header is {','.join(columns)} (got {header and ','.join(header)})"):
                return None
            rows, bad = [], 0
            for line in reader:
                if len(line) != len(columns):
                    bad += 1
                    continue
                try:
                    rows.append({c: (float(v) if v != "" else None) for c, v in zip(columns, line)})
                except ValueError:
                    bad += 1
    except OSError as e:
        r.fail(f"cannot read: {e}")
        return None
    r.check(bad == 0, f"every row has {len(columns)} numeric-or-empty fields ({bad} bad rows)")
    return rows


def check_times(rows: list[dict], name: str, min_hz: float, max_median_dt: float, r: Report):
    t = [row["t"] for row in rows]
    if not r.check(len(t) >= 2, f"{len(t)} rows"):
        return None
    r.check(all(x is not None for x in t), "every row has t")
    r.check(min(t) >= 0, f"t starts at or after 0 ({min(t):.3f} s)")
    r.check(all(b >= a for a, b in zip(t, t[1:])), "t never goes backwards")
    dts = [b - a for a, b in zip(t, t[1:])]
    median = statistics.median(dts)
    rate = (len(t) - 1) / (t[-1] - t[0]) if t[-1] > t[0] else 0
    r.check(median <= max_median_dt, f"median interval {median * 1000:.1f} ms (want <= {max_median_dt * 1000:.0f} ms)")
    r.check(rate >= min_hz, f"average rate {rate:.1f} Hz (want >= {min_hz} Hz)", warn_only=True)
    gap = max(dts)
    r.check(gap <= 5 * max_median_dt + 1, f"longest gap {gap:.2f} s", warn_only=True)
    return t[0], t[-1]


def check_gps(rows, r):
    span = check_times(rows, "gps", 0.9, 1.1, r)
    lat_ok = all(row["lat"] is not None and -90 <= row["lat"] <= 90 for row in rows)
    lon_ok = all(row["lon"] is not None and -180 <= row["lon"] <= 180 for row in rows)
    r.check(lat_ok and lon_ok, "lat/lon present and in range")
    acc = [row["accuracy_m"] for row in rows if row["accuracy_m"] is not None]
    if acc:
        good = sum(a <= 30 for a in acc) / len(rows)
        r.check(good >= 0.8, f"{good:.0%} of fixes have accuracy <= 30 m (the pipeline drops the rest)", warn_only=True)
    speeds = [row["speed_mps"] for row in rows if row["speed_mps"] is not None]
    if speeds:
        r.ok(f"speed present on {len(speeds)}/{len(rows)} fixes, max {max(speeds) * 3.6:.0f} km/h")
    return span


def check_imu(rows, r):
    span = check_times(rows, "imu", 50, 0.025, r)  # iPhone Safari ~60 Hz, Android ~100 Hz; pipeline needs >= 40 Hz
    mags = [math.sqrt(row["ax"] ** 2 + row["ay"] ** 2 + row["az"] ** 2) for row in rows if None not in (row["ax"], row["ay"], row["az"])]
    if mags:
        g = statistics.median(mags)
        r.check(8.8 <= g <= 10.8, f"median |accel| {g:.2f} m/s^2 (gravity included, ~9.8)")
    gyro = sum(row["gx"] is not None for row in rows)
    r.check(gyro == len(rows), f"gyroscope present on {gyro}/{len(rows)} rows", warn_only=True)
    return span


def check_video(path: Path, meta: dict | None, sensor_end: float | None, r: Report):
    print("video.mp4")
    if not r.check(path.exists() and path.stat().st_size > 0, "exists and is not empty"):
        return
    if not shutil.which("ffprobe"):
        r.warn("ffprobe not on PATH; skipping size, frame-rate and duration checks")
        return
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
         "stream=width,height,avg_frame_rate,nb_read_packets,duration:stream_side_data=rotation:stream_tags=rotate",
         "-of", "json", str(path)],
        capture_output=True, text=True,
    )
    try:
        stream = json.loads(out.stdout)["streams"][0]
    except (ValueError, KeyError, IndexError):
        r.fail(f"ffprobe could not read the video: {out.stderr.strip()[:200]}")
        return
    w, h = stream["width"], stream["height"]
    rotation = 0
    for sd in stream.get("side_data_list", []):
        rotation = int(sd.get("rotation", rotation))
    rotation = int(stream.get("tags", {}).get("rotate", rotation))
    if rotation % 180:
        w, h = h, w
    num, den = (int(x) for x in stream["avg_frame_rate"].split("/"))
    fps = num / den if den else 0
    frames = int(stream.get("nb_read_packets", 0))
    duration = float(stream.get("duration", 0))
    r.check(min(w, h) >= 720 and w > h, f"displayed size {w}x{h} (want landscape, 720p or more)", warn_only=True)
    if meta and isinstance(meta.get("camera"), dict):
        r.check((w, h) == (meta["camera"].get("width"), meta["camera"].get("height")), "size matches meta.json camera width/height")
    r.check(abs(fps - 30) < 1.5, f"average frame rate {fps:.2f} fps over {frames} frames, {duration:.1f} s")
    if duration and frames:
        expected = duration * 30
        r.check(frames >= 0.97 * expected, f"{frames} frames for {duration:.1f} s ({frames / expected:.1%} of 30 fps; "
                "dropped frames shift `video_t0 + frame / fps` times)", warn_only=True)
    if meta and is_number(meta.get("video_t0")) and sensor_end is not None:
        end = meta["video_t0"] + duration
        r.check(abs(end - sensor_end) < 3, f"video ends at t={end:.1f} s, sensor logs at t={sensor_end:.1f} s", warn_only=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session_dir", type=Path)
    ap.add_argument("--pipeline", type=Path, help="path to roadscan/pipeline, to also run load_session()")
    args = ap.parse_args(argv)
    d: Path = args.session_dir
    r = Report()
    print(f"Session {d}")

    meta = check_meta(d / "meta.json", r)
    gps = read_csv(d / "gps.csv", GPS_COLUMNS, r)
    gps_span = check_gps(gps, r) if gps else None
    imu = read_csv(d / "imu.csv", IMU_COLUMNS, r)
    imu_span = check_imu(imu, r) if imu else None
    ends = [s[1] for s in (gps_span, imu_span) if s]
    check_video(d / "video.mp4", meta, max(ends) if ends else None, r)

    if args.pipeline:
        print("pipeline load_session")
        sys.path.insert(0, str(args.pipeline.resolve()))
        try:
            from roadscan.session import load_session

            s = load_session(d)
            r.ok(f"loaded: {len(s.gps)} usable GPS fixes, {len(s.imu)} IMU rows, camera fx={s.camera.fx:.0f}px")
        except Exception as e:  # report any loader error as a failure
            r.fail(f"load_session raised {type(e).__name__}: {e}")

    print("RESULT:", "FAIL" if r.failed else "PASS")
    return 1 if r.failed else 0


if __name__ == "__main__":
    sys.exit(main())
