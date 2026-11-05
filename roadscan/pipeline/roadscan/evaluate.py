"""Score the pipeline against a ground-truth set of rides (synthetic now, a walked route later).

ground_truth.json lists every object with `kind`, `lat`, `lon` and `fixed_after_ride`;
only kind == "pothole" counts as a true pothole.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .fusion import FusionConfig, fuse
from .geo import Track, haversine_m
from .imu import detect_jolts
from .registry import ACTIVE, Registry
from .session import load_session

RIDE_MATCH_M = 6.0
REGISTRY_MATCH_M = 8.0


def greedy_match(pred: pd.DataFrame, truth: pd.DataFrame, radius_m: float) -> list[tuple[int, int, float]]:
    """One-to-one (pred_index, truth_index, distance) pairs, closest first."""
    if pred.empty or truth.empty:
        return []
    d = haversine_m(
        pred["lat"].to_numpy()[:, None], pred["lon"].to_numpy()[:, None],
        truth["lat"].to_numpy()[None, :], truth["lon"].to_numpy()[None, :],
    )
    pi, ti = np.nonzero(d <= radius_m)
    out, used_p, used_t = [], set(), set()
    for dist, p, t in sorted(zip(d[pi, ti], pi, ti)):
        if p not in used_p and t not in used_t:
            used_p.add(p)
            used_t.add(t)
            out.append((pred.index[p], truth.index[t], float(dist)))
    return out


def _scores(pred: pd.DataFrame, truth: pd.DataFrame) -> dict:
    matches = greedy_match(pred, truth, RIDE_MATCH_M)
    tp = len(matches)
    precision = tp / len(pred) if len(pred) else 0.0
    recall = tp / len(truth) if len(truth) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if tp else 0.0
    err = float(np.median([m[2] for m in matches])) if matches else float("nan")
    return {"tp": tp, "fp": len(pred) - tp, "fn": len(truth) - tp, "precision": precision, "recall": recall, "f1": f1, "median_err_m": err}


def evaluate(rides_dir: str | Path, db_path: str | Path | None = None) -> dict:
    rides_dir = Path(rides_dir)
    truth = json.loads((rides_dir / "ground_truth.json").read_text(encoding="utf-8"))
    objects = pd.DataFrame(truth["objects"])
    potholes = objects[objects["kind"] == "pothole"]
    configs = {
        "vision@0.30": FusionConfig(mode="vision", vision_conf=0.30),
        "vision@0.45": FusionConfig(mode="vision", vision_conf=0.45),
        "vision@0.60": FusionConfig(mode="vision", vision_conf=0.60),
        "fusion": FusionConfig(),
    }

    db_path = Path(db_path) if db_path else rides_dir / "eval.db"
    db_path.unlink(missing_ok=True)
    per_ride = []
    with Registry(db_path) as registry:
        for info in truth["rides"]:
            ride = info["ride"]
            session = load_session(rides_dir / info["session_id"])
            jolts = detect_jolts(session.imu, Track(session.gps), session.vehicle_type)
            fixed = potholes["fixed_after_ride"]
            present = potholes[fixed.isna() | (ride <= fixed.fillna(0))]
            for name, cfg in configs.items():
                cands = fuse(session, jolts, cfg)
                pred = cands[cands["confirmed"] & (cands["cls"] == "pothole")]
                per_ride.append({"ride": ride, "condition": info["condition"], "vehicle": info["vehicle_type"], "method": name, **_scores(pred, present)})
                if name == "fusion":
                    registry.ingest(session, cands)
        entities = registry.potholes(as_of=registry.latest_session_end())

    rides = pd.DataFrame(per_ride)
    totals = rides.groupby("method")[["tp", "fp", "fn"]].sum()
    totals["precision"] = totals["tp"] / (totals["tp"] + totals["fp"])
    totals["recall"] = totals["tp"] / (totals["tp"] + totals["fn"])
    totals["f1"] = 2 * totals["precision"] * totals["recall"] / (totals["precision"] + totals["recall"])
    totals["median_err_m"] = rides.groupby("method")["median_err_m"].median()

    # Registry: one entity per real pothole, and repairs noticed.
    pot_entities = entities[(entities["cls"] == "pothole") & (entities["status"] != "dismissed")]
    matches = greedy_match(pot_entities, potholes, REGISTRY_MATCH_M)
    status_of = {t: pot_entities.at[e, "status"] for e, t, _ in matches}
    matched_entities = {e for e, _, _ in matches}
    unmatched = pot_entities[~pot_entities.index.isin(matched_entities)]
    near_truth = [
        haversine_m(r["lat"], r["lon"], potholes["lat"], potholes["lon"]).min() <= REGISTRY_MATCH_M
        for _, r in unmatched.iterrows()
    ]
    was_fixed = potholes["fixed_after_ride"].notna()
    fixed_idx, open_idx = potholes.index[was_fixed], potholes.index[~was_fixed]
    registry_summary = {
        "true_potholes": len(potholes),
        "found": len(matches),
        "duplicates": int(np.sum(near_truth)),
        "false_entities": int(np.sum(~np.array(near_truth, bool)) if near_truth else 0),
        "false_entities_active": int(((~np.array(near_truth, bool)) & unmatched["status"].isin(ACTIVE).to_numpy()).sum()) if near_truth else 0,
        "median_err_m": float(np.median([m[2] for m in matches])) if matches else float("nan"),
        "repaired": len(fixed_idx),
        "repair_noticed": sum(status_of.get(i) == "likely_fixed" for i in fixed_idx),
        "open_wrongly_closed": sum(status_of.get(i) == "likely_fixed" for i in open_idx),
        "status_counts": entities["status"].value_counts().to_dict(),
    }
    return {"per_ride": rides, "totals": totals, "registry": registry_summary, "db_path": str(db_path)}


def format_report(result: dict) -> str:
    lines = ["Per-ride pothole detection (all rides pooled)", ""]
    t = result["totals"].copy()
    lines.append(t.to_string(float_format=lambda x: f"{x:.3f}"))
    lines += ["", "Fusion by ride condition", ""]
    rides = result["per_ride"]
    fusion = rides[rides["method"] == "fusion"]
    lines.append(fusion[["ride", "condition", "vehicle", "tp", "fp", "fn", "precision", "recall", "median_err_m"]].to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    r = result["registry"]
    lines += [
        "",
        "Registry after all rides",
        f"  true potholes found       {r['found']}/{r['true_potholes']}",
        f"  duplicate entities        {r['duplicates']}",
        f"  false entities            {r['false_entities']} ({r['false_entities_active']} still active)",
        f"  median location error     {r['median_err_m']:.2f} m",
        f"  repairs noticed           {r['repair_noticed']}/{r['repaired']}",
        f"  open potholes closed      {r['open_wrongly_closed']}",
        f"  status counts             {r['status_counts']}",
    ]
    return "\n".join(lines)
