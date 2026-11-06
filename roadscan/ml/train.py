"""Fine-tune a COCO-pretrained YOLO on the RDD YOLO dataset, then report per-class validation scores.

    python train.py --data D:/pothole_mapping/data/rdd_india_yolo/data.yaml --model D:/pothole_mapping/models/yolo11m.pt

Runs with YOLO_OFFLINE=1 so Ultralytics neither downloads assets nor sends analytics.
The best weights are copied to <models>/roadscan_<name>.pt for `roadscan detect`.
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

os.environ.setdefault("YOLO_OFFLINE", "1")

from ultralytics import YOLO  # noqa: E402  (must follow the env var)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True, help="pretrained .pt to start from")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--fraction", type=float, default=1.0, help="use part of the train set (smoke tests)")
    ap.add_argument("--project", default="D:/pothole_mapping/runs")
    ap.add_argument("--name", default="rdd_india_y11m")
    ap.add_argument("--models-dir", default="D:/pothole_mapping/models")
    ap.add_argument("--resume", action="store_true", help="continue <project>/<name> from its last.pt")
    args = ap.parse_args()

    last = Path(args.project) / args.name / "weights" / "last.pt"
    if args.resume:
        if not last.exists():
            raise SystemExit(f"nothing to resume: {last} not found")
        YOLO(str(last)).train(resume=True)  # restores epoch, optimizer and all original settings
    else:
        _train(args)
    _report(args)


def _train(args) -> None:
    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        fraction=args.fraction,
        project=args.project,
        name=args.name,
        exist_ok=True,
        patience=20,
        cos_lr=True,
        close_mosaic=10,
        fliplr=0.5,
        flipud=0.0,  # roads are never upside down
        degrees=0.0,
        seed=0,
        plots=True,
    )


def _report(args) -> None:
    best = Path(args.project) / args.name / "weights" / "best.pt"
    metrics = YOLO(str(best)).val(data=args.data, imgsz=args.imgsz, batch=args.batch, workers=args.workers, plots=False)
    print("\nValidation (best.pt)")
    print(f"  mAP50     {metrics.box.map50:.3f}")
    print(f"  mAP50-95  {metrics.box.map:.3f}")
    # class_result takes a position among the classes present in val, not a class id
    # (our shared 7-class list has classes RDD never labels).
    for i, k in enumerate(metrics.box.ap_class_index):
        p, r, ap50, ap = metrics.box.class_result(i)
        print(f"  {metrics.names[int(k)]:20s} P {p:.3f}  R {r:.3f}  AP50 {ap50:.3f}")

    out = Path(args.models_dir) / f"roadscan_{args.name}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, out)
    print(f"\nweights: {out}")


if __name__ == "__main__":
    main()
