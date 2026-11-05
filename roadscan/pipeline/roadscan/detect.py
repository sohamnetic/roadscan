"""Run a YOLO road-damage model over a session's video and write detections.csv.

Needs the optional dependency: pip install "roadscan[yolo]". RDD2022-style class
names (D00, D10, D20, D40) are mapped onto the pipeline's names.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .session import DETECTION_COLUMNS, load_session

CLASS_ALIASES = {
    "D00": "longitudinal_crack",
    "D10": "transverse_crack",
    "D20": "alligator_crack",
    "D40": "pothole",
    "speed_bump": "speed_breaker",
    "speedbreaker": "speed_breaker",
    "manhole_cover": "manhole",
}


def run_yolo(
    session_dir: str | Path,
    weights: str | Path,
    every: int = 6,
    conf: float = 0.10,
    imgsz: int = 640,
    tta: bool = True,
) -> Path:
    """`conf` is only a floor: FusionConfig applies the real thresholds, so weak boxes must reach it.

    Use the imgsz the model was trained at (scores drop when it differs), and calibrate the
    fusion thresholds with the same tta setting (ml/calibrate.py --augment).
    """
    import cv2
    from ultralytics import YOLO

    session_dir = Path(session_dir)
    session = load_session(session_dir)
    video = session_dir / "video.mp4"
    if not video.exists():
        raise FileNotFoundError(video)

    model = YOLO(str(weights))
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    rows, frame = [], 0
    try:
        while cap.grab():
            if frame % every == 0:
                ok, image = cap.retrieve()
                if not ok:
                    break
                result = model.predict(image, conf=conf, imgsz=imgsz, augment=tta, verbose=False)[0]
                t = session.video_t0 + frame / fps
                for box, c, k in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), result.boxes.cls.tolist()):
                    name = model.names[int(k)]
                    rows.append((t, frame, CLASS_ALIASES.get(name, name), c, *box))
            frame += 1
    finally:
        cap.release()

    out = session_dir / "detections.csv"
    pd.DataFrame(rows, columns=DETECTION_COLUMNS).round({"t": 3, "conf": 3, "x1": 1, "y1": 1, "x2": 1, "y2": 1}).to_csv(out, index=False)
    return out
