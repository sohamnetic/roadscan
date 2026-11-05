"""Convert a Label Studio JSON export into a YOLO dataset for training.

    python to_yolo.py export.json --images D:/pothole_mapping/data/labeling/images \
        --out D:/pothole_mapping/data/own_yolo

Splits train/val **by ride session**, not by frame: frames from one ride look alike, so a
frame-level split would leak near-copies into val and inflate the scores.
Tasks marked blurry_or_dark, and tasks nobody has labelled yet, are skipped. A labelled
frame with no boxes stays in as a background example (a clean road).
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from classes import CLASSES


def image_name(url: str) -> str:
    query = parse_qs(urlparse(url).query)
    path = query["d"][0] if "d" in query else urlparse(url).path
    return Path(unquote(path)).name


def yolo_lines(results: list[dict]) -> tuple[list[str], bool]:
    lines, bad_frame = [], False
    for r in results:
        if r.get("type") == "choices" and "blurry_or_dark" in r["value"].get("choices", []):
            bad_frame = True
        if r.get("type") != "rectanglelabels":
            continue
        v = r["value"]
        label = v["rectanglelabels"][0]
        if label not in CLASSES:
            continue
        x, y = max(0.0, v["x"]), max(0.0, v["y"])
        w, h = min(100.0 - x, v["width"]), min(100.0 - y, v["height"])
        if w <= 0.1 or h <= 0.1:
            continue
        lines.append(f"{CLASSES.index(label)} {(x + w / 2) / 100:.6f} {(y + h / 2) / 100:.6f} {w / 100:.6f} {h / 100:.6f}")
    return lines, bad_frame


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("export", type=Path, help="Label Studio export (JSON format)")
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tasks = json.loads(args.export.read_text(encoding="utf-8"))
    frames = []  # (session, image name, lines)
    skipped = Counter()
    for task in tasks:
        done = [a for a in task.get("annotations", []) if not a.get("was_cancelled")]
        if not done:
            skipped["unlabelled"] += 1
            continue
        lines, bad = yolo_lines(done[-1]["result"])
        if bad:
            skipped["blurry_or_dark"] += 1
            continue
        name = image_name(task["data"]["image"])
        session = task["data"].get("session") or name.rsplit("_", 1)[0]
        frames.append((session, name, lines))

    sessions = sorted({s for s, _, _ in frames})
    random.Random(args.seed).shuffle(sessions)
    n_val = max(1, round(len(sessions) * args.val_fraction)) if len(sessions) > 1 else 0
    val_sessions = set(sessions[:n_val])

    counts = Counter()
    for session, name, lines in frames:
        split = "val" if session in val_sessions else "train"
        for sub in ("images", "labels"):
            (args.out / sub / split).mkdir(parents=True, exist_ok=True)
        src = args.images / name
        if not src.exists():
            skipped["image missing"] += 1
            continue
        shutil.copy2(src, args.out / "images" / split / name)
        (args.out / "labels" / split / (Path(name).stem + ".txt")).write_text("\n".join(lines), encoding="utf-8")
        counts[split] += 1
        for line in lines:
            counts[CLASSES[int(line.split()[0])]] += 1

    yaml = [f"path: {args.out.resolve().as_posix()}", "train: images/train", "val: images/val", "names:"]
    yaml += [f"  {i}: {n}" for i, n in enumerate(CLASSES)]
    (args.out / "data.yaml").write_text("\n".join(yaml) + "\n", encoding="utf-8")
    print(f"frames: train {counts['train']}, val {counts['val']} (val sessions: {sorted(val_sessions) or 'none'})")
    print("boxes:", {c: counts[c] for c in CLASSES})
    if skipped:
        print("skipped:", dict(skipped))
    if not val_sessions:
        print("note: only one session, so no val split; label at least 2 rides")


if __name__ == "__main__":
    main()
