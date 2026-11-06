"""Show how precision and recall change with the confidence cutoff, per class, on the val set.

    python calibrate.py --weights D:/pothole_mapping/models/roadscan_rdd_india_y11m.pt --data D:/pothole_mapping/data/rdd_india_yolo/data.yaml

Use it to pick the pipeline's FusionConfig thresholds for a given model (box-level numbers,
IoU 0.5). --augment scores test-time augmentation (flips + scales) instead of plain inference.
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("YOLO_OFFLINE", "1")

import numpy as np  # noqa: E402
from ultralytics import YOLO  # noqa: E402

GRID = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--augment", action="store_true", help="test-time augmentation")
    args = ap.parse_args()

    m = YOLO(args.weights).val(data=args.data, imgsz=args.imgsz, batch=args.batch, workers=4,
                               augment=args.augment, plots=False, verbose=False)
    box = m.box
    print(f"\n{'TTA' if args.augment else 'plain'} @ imgsz {args.imgsz}: mAP50 {box.map50:.3f}  mAP50-95 {box.map:.3f}")
    # Curves and per-class arrays are indexed by position among the classes present in val.
    present = [(i, m.names[int(k)]) for i, k in enumerate(box.ap_class_index)]
    for i, name in present:
        print(f"  {name:20s} AP50 {box.ap50[i]:.3f}")

    px = np.asarray(box.px)
    for i, name in present:
        p, r, f1 = (np.asarray(c)[i] for c in (box.p_curve, box.r_curve, box.f1_curve))
        best = int(f1.argmax())
        print(f"\n{name}: best F1 {f1[best]:.3f} at conf {px[best]:.2f} (P {p[best]:.2f}, R {r[best]:.2f})")
        print("   conf   precision  recall")
        for c in GRID:
            k = int(np.searchsorted(px, c))
            print(f"   {c:4.2f}   {p[k]:9.2f}  {r[k]:6.2f}")


if __name__ == "__main__":
    main()
