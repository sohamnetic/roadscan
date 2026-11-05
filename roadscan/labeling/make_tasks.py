"""Pick frames worth labelling from ride sessions and write a Label Studio task file.

    python make_tasks.py D:/rides/ride_01 D:/rides/ride_02 --out D:/pothole_mapping/data/labeling \
        --weights D:/pothole_mapping/models/roadscan_rdd_india_y11m.pt

Labelling every frame wastes time: neighbouring frames are near-identical. Instead it takes
  - one frame every --every-s seconds (coverage), and
  - every frame where the model is unsure (a box between --low and --high confidence):
    these teach the model the most.
The model's boxes are attached as pre-annotations, so labelling is mostly fixing, not drawing.
Frames come from video_blurred.mp4 when present (run `roadscan blur` first), never raw video
unless --allow-raw is given.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("YOLO_OFFLINE", "1")

from classes import canonical  # noqa: E402


def pick_video(session: Path, allow_raw: bool) -> Path:
    blurred = session / "video_blurred.mp4"
    if blurred.exists():
        return blurred
    if allow_raw and (session / "video.mp4").exists():
        return session / "video.mp4"
    raise FileNotFoundError(f"{blurred} not found: run `roadscan blur {session}` first (or pass --allow-raw)")


def to_ls_boxes(result, names, width: int, height: int) -> list[dict]:
    out = []
    for (x1, y1, x2, y2), conf, k in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), result.boxes.cls.tolist()):
        label = canonical(names[int(k)])
        if label is None:
            continue
        out.append({
            "from_name": "label", "to_name": "image", "type": "rectanglelabels",
            "original_width": width, "original_height": height, "score": round(conf, 3),
            "value": {"x": 100 * x1 / width, "y": 100 * y1 / height,
                      "width": 100 * (x2 - x1) / width, "height": 100 * (y2 - y1) / height,
                      "rotation": 0, "rectanglelabels": [label]},
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sessions", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True, help="labelling folder (Label Studio document root)")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--every-s", type=float, default=2.0)
    ap.add_argument("--low", type=float, default=0.10)
    ap.add_argument("--high", type=float, default=0.50)
    ap.add_argument("--scan-every", type=int, default=6, help="check every Nth frame for unsure boxes")
    ap.add_argument("--max-per-session", type=int, default=300)
    ap.add_argument("--allow-raw", action="store_true", help="use unblurred video (stays on this laptop!)")
    args = ap.parse_args()

    import cv2
    from ultralytics import YOLO

    model = YOLO(args.weights)
    images_dir = args.out / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    tasks_path = args.out / "tasks.json"
    tasks = json.loads(tasks_path.read_text(encoding="utf-8")) if tasks_path.exists() else []
    have = {t["data"]["image"] for t in tasks}

    for session in args.sessions:
        video = pick_video(session, args.allow_raw)
        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, round(args.every_s * fps))
        picked, frame = 0, 0
        try:
            while picked < args.max_per_session and cap.grab():
                coverage = frame % step == 0
                if coverage or frame % args.scan_every == 0:
                    ok, image = cap.retrieve()
                    if not ok:
                        break
                    h, w = image.shape[:2]
                    result = model.predict(image, conf=args.low, imgsz=640, verbose=False)[0]
                    confs = result.boxes.conf.tolist()
                    unsure = any(args.low <= c < args.high for c in confs)
                    if coverage or unsure:
                        name = f"{session.name}_{frame:06d}.jpg"
                        cv2.imwrite(str(images_dir / name), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
                        url = f"/data/local-files/?d=images/{name}"
                        if url not in have:
                            tasks.append({
                                "data": {"image": url, "session": session.name, "frame": frame,
                                         "t_video_s": round(frame / fps, 3), "why": "unsure" if unsure else "coverage"},
                                "predictions": [{"model_version": Path(args.weights).stem,
                                                 "result": to_ls_boxes(result, model.names, w, h)}],
                            })
                            have.add(url)
                        picked += 1
                frame += 1
        finally:
            cap.release()
        print(f"{session.name}: {picked} frames picked from {frame} ({video.name})")

    tasks_path.write_text(json.dumps(tasks, indent=1), encoding="utf-8")
    print(f"{len(tasks)} tasks in {tasks_path}")


if __name__ == "__main__":
    main()
