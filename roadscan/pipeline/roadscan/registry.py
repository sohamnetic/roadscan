"""Road-damage registry: one row per physical defect, tracked across rides.

Lifecycle:
  open            seen on one ride
  confirmed       seen on two or more rides
  likely_fixed    driven past `miss_limit` times in a row without being seen
  dismissed       seen once, then missed twice: probably a false detection
  verified_fixed  closed by a person (photo or site visit)

SQLite keeps the pilot simple; the schema maps directly onto PostGIS later.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .geo import Track, haversine_m
from .session import Session

ACTIVE = ("open", "confirmed")
MATCHABLE = ("open", "confirmed", "likely_fixed", "dismissed")

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    vehicle_type TEXT,
    start_time TEXT,
    end_time TEXT,
    distance_km REAL,
    ingested_at TEXT
);
CREATE TABLE IF NOT EXISTS potholes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    lat REAL NOT NULL,
    lon REAL NOT NULL,
    cls TEXT NOT NULL,
    status TEXT NOT NULL,
    severity INTEGER,
    area_m2 REAL,
    first_seen TEXT NOT NULL,
    last_seen TEXT NOT NULL,
    n_sessions INTEGER NOT NULL DEFAULT 1,
    misses INTEGER NOT NULL DEFAULT 0,
    closed_at TEXT
);
CREATE TABLE IF NOT EXISTS observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pothole_id INTEGER NOT NULL REFERENCES potholes(id),
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    seen_at TEXT NOT NULL,
    lat REAL, lon REAL, cls TEXT, source TEXT,
    max_conf REAL, n_obs INTEGER, jolt_strength REAL, severity INTEGER
);
CREATE INDEX IF NOT EXISTS idx_obs_pothole ON observations(pothole_id);
"""


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _nullable(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else x


class Registry:
    def __init__(self, path: str | Path):
        self.conn = sqlite3.connect(str(path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _load(self, statuses) -> pd.DataFrame:
        marks = ",".join("?" * len(statuses))
        return pd.read_sql_query(f"SELECT * FROM potholes WHERE status IN ({marks})", self.conn, params=list(statuses))

    def ingest(
        self,
        session: Session,
        candidates: pd.DataFrame,
        match_m: float = 8.0,
        pass_m: float = 8.0,
        miss_limit: int = 3,
    ) -> dict:
        """Merge one session's candidates; returns counts of what changed."""
        if self.conn.execute("SELECT 1 FROM sessions WHERE session_id = ?", (session.session_id,)).fetchone():
            raise ValueError(f"session {session.session_id} is already in the registry")
        track = Track(session.gps)
        end = _iso(session.end_time)
        summary = {"new": 0, "updated": 0, "reopened": 0, "missed": 0, "likely_fixed": 0, "dismissed": 0}

        with self.conn:
            self.conn.execute(
                "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?)",
                (session.session_id, session.vehicle_type, _iso(session.start_time), end,
                 track.length_m() / 1000, _iso(datetime.now(timezone.utc))),
            )
            known = self._load(MATCHABLE)
            confirmed = candidates[candidates["confirmed"]].reset_index(drop=True)
            # Unconfirmed but non-trivial evidence (several frames, or a jolt) still blocks a miss.
            damage = candidates[(candidates["severity"] > 0) & ((candidates["n_obs"] >= 2) | candidates["jolt_strength"].notna())]

            # Greedy one-to-one matching of this ride's confirmed candidates to known defects.
            pairs = []
            if len(known) and len(confirmed):
                d = haversine_m(
                    confirmed["lat"].to_numpy()[:, None], confirmed["lon"].to_numpy()[:, None],
                    known["lat"].to_numpy()[None, :], known["lon"].to_numpy()[None, :],
                )
                ci, ki = np.nonzero(d <= match_m)
                pairs = sorted(zip(d[ci, ki], ci, ki))
            matched_c, matched_k = {}, set()
            for _, ci, ki in pairs:
                if ci not in matched_c and ki not in matched_k:
                    matched_c[ci] = ki
                    matched_k.add(ki)

            for ci, cand in confirmed.iterrows():
                seen = _iso(session.time_at(cand["t"]))
                if ci in matched_c:
                    row = known.iloc[matched_c[ci]]
                    n = int(row["n_sessions"]) + 1
                    status = "confirmed" if n >= 2 else "open"
                    summary["reopened" if row["status"] not in ACTIVE else "updated"] += 1
                    self.conn.execute(
                        "UPDATE potholes SET lat=?, lon=?, cls=?, status=?, severity=?, area_m2=?, last_seen=?,"
                        " n_sessions=?, misses=0, closed_at=NULL WHERE id=?",
                        ((row["lat"] * (n - 1) + cand["lat"]) / n, (row["lon"] * (n - 1) + cand["lon"]) / n,
                         cand["cls"], status, int(cand["severity"]), _nullable(cand["area_m2"]), seen, n, int(row["id"])),
                    )
                    pothole_id = int(row["id"])
                else:
                    summary["new"] += 1
                    pothole_id = self.conn.execute(
                        "INSERT INTO potholes (lat, lon, cls, status, severity, area_m2, first_seen, last_seen)"
                        " VALUES (?, ?, ?, 'open', ?, ?, ?, ?)",
                        (cand["lat"], cand["lon"], cand["cls"], int(cand["severity"]), _nullable(cand["area_m2"]), seen, seen),
                    ).lastrowid
                self.conn.execute(
                    "INSERT INTO observations (pothole_id, session_id, seen_at, lat, lon, cls, source, max_conf, n_obs,"
                    " jolt_strength, severity) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (pothole_id, session.session_id, seen, cand["lat"], cand["lon"], cand["cls"], cand["source"],
                     _nullable(cand["max_conf"]), int(cand["n_obs"]), _nullable(cand["jolt_strength"]), int(cand["severity"])),
                )

            # Active defects this ride drove past without confirming count as a miss.
            for ki, row in known.iterrows():
                if ki in matched_k or row["status"] not in ACTIVE:
                    continue
                if len(damage) and haversine_m(row["lat"], row["lon"], damage["lat"], damage["lon"]).min() <= match_m:
                    continue  # a weak sighting: not proof it is still there, not proof it is gone
                if track.min_distance_m(row["lat"], row["lon"], min_speed=2.0) > pass_m:
                    continue
                misses = int(row["misses"]) + 1
                summary["missed"] += 1
                status, closed = row["status"], None
                if row["n_sessions"] == 1 and misses >= 2:
                    status, closed = "dismissed", end
                elif misses >= miss_limit:
                    status, closed = "likely_fixed", end
                if status != row["status"]:
                    summary[status] += 1
                self.conn.execute(
                    "UPDATE potholes SET misses=?, status=?, closed_at=? WHERE id=?", (misses, status, closed, int(row["id"]))
                )
        return summary

    def mark_fixed(self, pothole_id: int, when: datetime | None = None) -> None:
        with self.conn:
            cur = self.conn.execute(
                "UPDATE potholes SET status='verified_fixed', closed_at=? WHERE id=?",
                (_iso(when or datetime.now(timezone.utc)), pothole_id),
            )
        if cur.rowcount == 0:
            raise KeyError(pothole_id)

    def potholes(self, as_of: datetime | None = None) -> pd.DataFrame:
        """All defects with days_open and a repair priority (0 once closed)."""
        df = pd.read_sql_query("SELECT * FROM potholes ORDER BY id", self.conn)
        as_of = as_of or datetime.now(timezone.utc)
        first = pd.to_datetime(df["first_seen"], utc=True, format="ISO8601")
        df["days_open"] = ((pd.Timestamp(as_of) - first).dt.total_seconds() / 86400).round(1)
        active = df["status"].isin(ACTIVE)
        df["priority"] = np.where(active, df["severity"] * (1 + df["days_open"].clip(lower=0) / 30), 0.0).round(2)
        return df

    def observations(self, pothole_id: int | None = None) -> pd.DataFrame:
        if pothole_id is None:
            return pd.read_sql_query("SELECT * FROM observations ORDER BY id", self.conn)
        return pd.read_sql_query("SELECT * FROM observations WHERE pothole_id = ? ORDER BY id", self.conn, params=[pothole_id])

    def latest_session_end(self) -> datetime | None:
        row = self.conn.execute("SELECT MAX(end_time) FROM sessions").fetchone()
        return datetime.fromisoformat(row[0]) if row and row[0] else None
