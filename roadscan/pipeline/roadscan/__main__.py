"""Command line: python -m roadscan <command> ..."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_DB = "roadscan.db"


def _process(args) -> None:
    from .fusion import FusionConfig, fuse, load_config
    from .geo import Track
    from .imu import detect_jolts
    from .registry import Registry
    from .session import load_session

    config = load_config(args.config) if args.config else FusionConfig()
    with Registry(args.db) as registry:
        for path in args.sessions:
            session = load_session(path)
            jolts = detect_jolts(session.imu, Track(session.gps), session.vehicle_type)
            candidates = fuse(session, jolts, config)
            candidates.to_csv(Path(path) / "candidates.csv", index=False)
            summary = registry.ingest(session, candidates)
            confirmed = int(candidates["confirmed"].sum())
            print(f"{session.session_id}: {len(jolts)} jolts, {len(candidates)} candidates, {confirmed} confirmed -> {summary}")


def _map(args) -> None:
    from .registry import Registry
    from .render import write_exports, write_map

    with Registry(args.db) as registry:
        potholes = registry.potholes(as_of=registry.latest_session_end())
    if args.places:
        from .places import tag_places

        places = Path(args.places)
        if not (places / "roads.geojson").exists() and not (places / "wards.geojson").exists():
            raise FileNotFoundError(f"no roads.geojson or wards.geojson in {places} (run `roadscan osm` first)")
        wards = Path(args.wards) if args.wards else places / "wards.geojson"
        potholes = tag_places(potholes, places / "roads.geojson", wards)
        print(f"tagged: {potholes['road'].notna().sum()}/{len(potholes)} on a known road, "
              f"{potholes['ward'].notna().sum()}/{len(potholes)} in a ward")
    out = write_map(potholes, args.out, title=args.title)
    print(f"map: {out}")
    if args.exports:
        for p in write_exports(potholes, args.exports):
            print(f"export: {p}")


def _osm(args) -> None:
    from .places import bbox_around, fetch_osm
    from .registry import Registry

    if args.bbox:
        bbox = tuple(float(v) for v in args.bbox.split(","))
        if len(bbox) != 4:
            raise ValueError("--bbox is south,west,north,east")
    else:
        with Registry(args.db) as registry:
            potholes = registry.potholes()
        if potholes.empty:
            raise ValueError(f"no defects in {args.db} to take an area from; pass --bbox")
        bbox = bbox_around(potholes["lat"], potholes["lon"], margin_m=args.margin)
    roads, wards = fetch_osm(bbox, args.out, ward_level=args.ward_level)
    n_roads = len(json.loads(roads.read_text(encoding="utf-8"))["features"])
    n_wards = len(json.loads(wards.read_text(encoding="utf-8"))["features"])
    print(f"wrote {roads} ({n_roads} roads) and {wards} ({n_wards} wards); data (c) OpenStreetMap contributors, ODbL")


def _fix(args) -> None:
    from .registry import Registry

    with Registry(args.db) as registry:
        registry.mark_fixed(args.id)
    print(f"#{args.id} marked verified_fixed")


def _synth(args) -> None:
    from .synth import generate

    out = generate(args.out, rides=args.rides, repair_after=args.repair_after, seed=args.seed)
    print(f"wrote {args.rides} rides to {out}")


def _eval(args) -> None:
    from .evaluate import evaluate, format_report

    result = evaluate(args.rides_dir, db_path=args.db)
    print(format_report(result))
    print(f"\nregistry written to {result['db_path']}")


def _detect(args) -> None:
    from .detect import run_yolo

    print(f"wrote {run_yolo(args.session, args.weights, every=args.every, conf=args.conf, imgsz=args.imgsz, tta=args.tta)}")


def _blur(args) -> None:
    from .blur import blur_video

    session = Path(args.session)
    video = session / "video.mp4" if session.is_dir() else session
    out = Path(args.out) if args.out else video.with_name(video.stem + "_blurred.mp4")
    stats = blur_video(video, out, args.weights, conf=args.conf)
    print(f"wrote {stats['out']}: {stats['frames']} frames, {stats['boxes']} people/vehicles hidden")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="roadscan", description="Road-damage mapping pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("process", help="fuse detections and jolts, then merge into the registry")
    p.add_argument("sessions", nargs="+", help="session folders, oldest first")
    p.add_argument("--db", default=DEFAULT_DB)
    p.add_argument("--config", help="JSON of FusionConfig overrides, e.g. models/<model>.fusion.json")
    p.set_defaults(func=_process)

    p = sub.add_parser("map", help="write an HTML map (and optionally CSV/GeoJSON) from the registry")
    p.add_argument("--db", default=DEFAULT_DB)
    p.add_argument("--out", default="map.html")
    p.add_argument("--title", default="Road damage map")
    p.add_argument("--exports", help="folder for potholes.csv, potholes.geojson and wards.csv")
    p.add_argument("--places", help="folder from `roadscan osm` (roads.geojson, wards.geojson): tag road + ward")
    p.add_argument("--wards", help="ward polygons GeoJSON to use instead of the OSM ones (e.g. the city's own)")
    p.set_defaults(func=_map)

    p = sub.add_parser("osm", help="download roads and ward boundaries from OpenStreetMap for the mapped area")
    p.add_argument("--out", default="osm", help="folder for roads.geojson and wards.geojson")
    p.add_argument("--db", default=DEFAULT_DB, help="take the area from this registry's defects")
    p.add_argument("--bbox", help="south,west,north,east instead of --db")
    p.add_argument("--margin", type=float, default=300.0, help="metres around the defects")
    p.add_argument("--ward-level", default="10", help="OSM admin_level of wards (10 in Bengaluru)")
    p.set_defaults(func=_osm)

    p = sub.add_parser("fix", help="mark a defect as repaired after a site check")
    p.add_argument("id", type=int)
    p.add_argument("--db", default=DEFAULT_DB)
    p.set_defaults(func=_fix)

    p = sub.add_parser("synth", help="generate synthetic rides with ground truth")
    p.add_argument("out")
    p.add_argument("--rides", type=int, default=8)
    p.add_argument("--repair-after", type=int, default=3)
    p.add_argument("--seed", type=int, default=7)
    p.set_defaults(func=_synth)

    p = sub.add_parser("eval", help="score the pipeline on rides with a ground_truth.json")
    p.add_argument("rides_dir")
    p.add_argument("--db", help="registry file to (re)create; default <rides_dir>/eval.db")
    p.set_defaults(func=_eval)

    p = sub.add_parser("detect", help="run a YOLO model over a session's video.mp4")
    p.add_argument("session")
    p.add_argument("--weights", required=True)
    p.add_argument("--every", type=int, default=6, help="process every Nth frame")
    p.add_argument("--conf", type=float, default=0.10, help="keep boxes above this; fusion applies the real thresholds")
    p.add_argument("--imgsz", type=int, default=640, help="match the size the model was trained at")
    p.add_argument("--tta", action=argparse.BooleanOptionalAction, default=True,
                   help="test-time augmentation (flips + scales): slower, slightly more accurate")
    p.set_defaults(func=_detect)

    p = sub.add_parser("blur", help="pixelate people and vehicles in a session's video (privacy)")
    p.add_argument("session", help="session folder (uses its video.mp4) or a video file")
    p.add_argument("--weights", required=True, help="COCO-pretrained YOLO, e.g. models/yolo11n.pt")
    p.add_argument("--out", help="default: <video>_blurred.mp4 next to the input")
    p.add_argument("--conf", type=float, default=0.2, help="low on purpose: a missed face costs more than extra blur")
    p.set_defaults(func=_blur)

    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (FileNotFoundError, ValueError, KeyError, ConnectionError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
