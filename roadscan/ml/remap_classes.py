"""Copy a YOLO dataset onto the shared RoadScan class list (labeling/classes.py), keeping its split.

    python remap_classes.py D:/pothole_mapping/data/rdd_india_yolo D:/pothole_mapping/data/rdd_india_v2_yolo

RDD's longitudinal and transverse cracks become one `crack` class: transverse had only 57
training boxes, too few to learn on its own. The output uses the same 7-class list as our own
labelled frames (pothole, alligator_crack, crack, speed_breaker, manhole, patch, puddle), so
the two datasets can later be trained together. Images are hard-linked (no extra disk).
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "labeling"))
from classes import CLASSES, canonical  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    args = ap.parse_args()

    names = yaml.safe_load((args.src / "data.yaml").read_text(encoding="utf-8"))["names"]
    names = names if isinstance(names, dict) else dict(enumerate(names))
    mapping = {int(i): CLASSES.index(canonical(n)) for i, n in names.items()}
    print("class mapping:", {names[i]: CLASSES[j] for i, j in mapping.items()})

    counts = Counter()
    for split in ("train", "val"):
        for sub in ("images", "labels"):
            (args.dst / sub / split).mkdir(parents=True, exist_ok=True)
        for img in (args.src / "images" / split).iterdir():
            dst = args.dst / "images" / split / img.name
            if not dst.exists():
                try:
                    os.link(img, dst)
                except OSError:
                    shutil.copy2(img, dst)
            label = args.src / "labels" / split / (img.stem + ".txt")
            lines = []
            if label.exists():
                for line in label.read_text(encoding="utf-8").split("\n"):
                    if line.strip():
                        k, *box = line.split()
                        lines.append(" ".join([str(mapping[int(k)]), *box]))
                        counts[(split, CLASSES[mapping[int(k)]])] += 1
            (args.dst / "labels" / split / (img.stem + ".txt")).write_text("\n".join(lines), encoding="utf-8")
            counts[(split, "images")] += 1

    out = [f"path: {args.dst.resolve().as_posix()}", "train: images/train", "val: images/val", "names:"]
    out += [f"  {i}: {n}" for i, n in enumerate(CLASSES)]
    (args.dst / "data.yaml").write_text("\n".join(out) + "\n", encoding="utf-8")
    for split in ("train", "val"):
        print(split, {k: v for (s, k), v in sorted(counts.items()) if s == split})


if __name__ == "__main__":
    main()
