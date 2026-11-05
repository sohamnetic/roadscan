"""Process one ride session into the registry; shared by the CLI and the web backend."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from .fusion import FusionConfig, fuse
from .geo import Track
from .imu import detect_jolts
from .registry import Registry
from .session import load_session


def process_session(path: str | Path, registry: Registry, config: FusionConfig = FusionConfig(),
                    roads: str | Path | None = None) -> dict:
    """Jolts + fusion (+ GPS snapped to roads when `roads` is given), merged into the registry."""
    path = Path(path)
    session = load_session(path)
    notes = []
    if roads is not None:
        from .mapmatch import matched_gps

        gps, match = matched_gps(session.gps, roads)
        match.to_csv(path / "gps_matched.csv", index=False)
        moved = match.loc[match["matched"], "shift_m"]
        pulled = moved[moved > 0]
        notes.append(f"{match['matched'].mean():.0%} of GPS fixes on a road, "
                     f"{(len(pulled) / len(moved)) if len(moved) else 0:.0%} pulled back onto it"
                     f" (median {pulled.median() if len(pulled) else 0:.1f} m)")
        session = replace(session, gps=gps)
    jolts = detect_jolts(session.imu, Track(session.gps), session.vehicle_type)
    candidates = fuse(session, jolts, config)
    candidates.to_csv(path / "candidates.csv", index=False)
    summary = registry.ingest(session, candidates)
    confirmed = int(candidates["confirmed"].sum())
    notes.append(f"{len(jolts)} jolts, {len(candidates)} candidates, {confirmed} confirmed -> {summary}")
    return {"session_id": session.session_id, "summary": summary, "confirmed": confirmed,
            "message": f"{session.session_id}: " + "; ".join(notes)}
