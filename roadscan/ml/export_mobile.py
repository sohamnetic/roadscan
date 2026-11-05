"""Export a trained model for on-phone use (TFLite), then report size, speed and accuracy.

Run with the export virtualenv (TensorFlow lives there, not in the main Python). Pin
ultralytics==8.4.80 there: from 8.4.83 TFLite export became "LiteRT", which is Linux/macOS-only.

    .venv-export/Scripts/python roadscan/ml/export_mobile.py --weights models/roadscan_rdd_india_y11m.pt \
        --data data/rdd_india_yolo/data.yaml

Writes float16 and int8 .tflite files next to the weights. Int8 is about 4x smaller and
faster on phone NPUs/DSPs; it is calibrated on training images, and its accuracy is
checked on the val split here, because int8 can lose a few points.
Laptop CPU speed is only a rough guide: phone speed must be measured on the phone.
"""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import numpy as np


def bench(path: Path, imgsz: int, runs: int = 20) -> float:
    """Median ms per image on this CPU (TFLite interpreter, 4 threads)."""
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        from tensorflow.lite.python.interpreter import Interpreter
    it = Interpreter(model_path=str(path), num_threads=4)
    it.allocate_tensors()
    inp = it.get_input_details()[0]
    x = np.random.rand(*inp["shape"]).astype(np.float32)
    if inp["dtype"] != np.float32:
        scale, zero = inp["quantization"]
        x = (x / scale + zero).astype(inp["dtype"])
    times = []
    for _ in range(runs + 3):
        t0 = time.perf_counter()
        it.set_tensor(inp["index"], x)
        it.invoke()
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times[3:]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, type=Path)
    ap.add_argument("--data", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--skip-val", action="store_true")
    args = ap.parse_args()

    from ultralytics import YOLO

    out = {}
    for kind, kw in (("fp16", {"half": True}), ("int8", {"int8": True, "data": args.data, "fraction": 0.1})):
        exported = Path(YOLO(str(args.weights)).export(format="tflite", imgsz=args.imgsz, **kw))
        if exported.is_dir():
            exported = max(exported.rglob("*.tflite"), key=lambda f: f.stat().st_mtime)
        final = args.weights.with_name(f"{args.weights.stem}_{kind}.tflite")
        shutil.copy2(exported, final)
        out[kind] = final

    print(f"\n{'model':28s} {'size MB':>8s} {'CPU ms':>8s} {'mAP50':>7s} {'pothole AP50':>13s}")
    base = YOLO(str(args.weights)).val(data=args.data, imgsz=args.imgsz, batch=8, plots=False, verbose=False) if not args.skip_val else None
    if base is not None:
        i = list(base.names.values()).index("pothole")
        print(f"{args.weights.name:28s} {args.weights.stat().st_size / 1e6:8.1f} {'-':>8s} {base.box.map50:7.3f} {base.box.ap50[i]:13.3f}")
    for kind, path in out.items():
        ms = bench(path, args.imgsz)
        if args.skip_val:
            print(f"{path.name:28s} {path.stat().st_size / 1e6:8.1f} {ms:8.0f}")
            continue
        m = YOLO(str(path), task="detect").val(data=args.data, imgsz=args.imgsz, batch=1, plots=False, verbose=False)
        i = list(m.names.values()).index("pothole")
        print(f"{path.name:28s} {path.stat().st_size / 1e6:8.1f} {ms:8.0f} {m.box.map50:7.3f} {m.box.ap50[i]:13.3f}")


if __name__ == "__main__":
    main()
