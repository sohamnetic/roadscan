"""Hide people and vehicles in ride video before it is stored or shared (India's DPDP Act 2023).

The product only needs the road surface. So every person (face) and every vehicle
(number plate) is pixelated whole, rather than chasing faces and plates one by one:
a COCO-pretrained YOLO finds them reliably, and an occluding vehicle hides no road anyway.
Boxes from the previous frame are blurred as well, so a one-frame miss does not leak.
Audio is dropped (voices).

Run `roadscan detect` on the original video first, then blur, then delete the original
before anything leaves the device or laptop.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

# COCO class ids: person, bicycle, car, motorcycle, bus, truck (autos come out as car/motorcycle/truck).
PRIVATE_CLASSES = (0, 1, 2, 3, 5, 7)


def pad_boxes(boxes: np.ndarray, pad: float, width: int, height: int) -> np.ndarray:
    """Grow each x1,y1,x2,y2 box by `pad` of its size on every side, clipped to the image, as ints."""
    boxes = np.asarray(boxes, float).reshape(-1, 4)
    if not len(boxes):
        return np.zeros((0, 4), int)
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    grown = boxes + np.stack([-w, -h, w, h], axis=1) * pad
    grown[:, [0, 2]] = grown[:, [0, 2]].clip(0, width)
    grown[:, [1, 3]] = grown[:, [1, 3]].clip(0, height)
    return np.round(grown).astype(int)


def pixelate(image: np.ndarray, boxes: np.ndarray, blocks: int = 10) -> np.ndarray:
    """Replace each box with a coarse mosaic (about `blocks` cells across its longer side), in place."""
    import cv2

    for x1, y1, x2, y2 in np.asarray(boxes, int).reshape(-1, 4):
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        region = image[y1:y2, x1:x2]
        h, w = region.shape[:2]
        cell = max(2, max(w, h) // blocks)
        small = cv2.resize(region, (max(1, w // cell), max(1, h // cell)), interpolation=cv2.INTER_AREA)
        image[y1:y2, x1:x2] = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    return image


def find_private(model, image: np.ndarray, conf: float = 0.2, imgsz: int = 640) -> np.ndarray:
    result = model.predict(image, conf=conf, imgsz=imgsz, classes=list(PRIVATE_CLASSES), verbose=False)[0]
    return result.boxes.xyxy.cpu().numpy().reshape(-1, 4)


def blur_video(
    video: str | Path,
    out: str | Path,
    weights: str | Path,
    conf: float = 0.2,
    pad: float = 0.15,
    imgsz: int = 640,
) -> dict:
    """Write a copy of `video` with people and vehicles pixelated; returns frame and box counts."""
    import cv2

    os.environ.setdefault("YOLO_OFFLINE", "1")  # no downloads, update checks or analytics
    from ultralytics import YOLO

    video, out = Path(video), Path(out)
    if not video.exists():
        raise FileNotFoundError(video)
    if video.resolve() == out.resolve():
        raise ValueError("write the blurred video to a new file, then delete the original")
    model = YOLO(str(weights))
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise ValueError(f"cannot write {out}")
    frames = boxes_total = 0
    previous = np.zeros((0, 4))
    try:
        while True:
            ok, image = cap.read()
            if not ok:
                break
            found = find_private(model, image, conf=conf, imgsz=imgsz)
            boxes_total += len(found)
            pixelate(image, pad_boxes(np.vstack([found, previous]), pad, width, height))
            writer.write(image)
            previous = found
            frames += 1
    finally:
        cap.release()
        writer.release()
    if frames == 0:
        out.unlink(missing_ok=True)
        raise ValueError(f"no frames could be read from {video}")
    return {"frames": frames, "boxes": boxes_total, "out": out}
