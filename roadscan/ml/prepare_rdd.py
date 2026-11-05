"""Convert an RDD2022 country zip (Pascal VOC) into a YOLO dataset.

    python -I prepare_rdd.py D:/pothole_mapping/data/rdd2022_raw/RDD2022_full.zip D:/pothole_mapping/data/rdd_india_yolo --country India

Only the labelled train split is used; it is divided into train/val because the
official test split ships without annotations. Class codes become the pipeline's names.
"""
from __future__ import annotations

import argparse
import collections
import os
import random
import shutil
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

# RDD code -> pipeline class name; order sets the YOLO class index.
CLASSES = {"D00": "longitudinal_crack", "D10": "transverse_crack", "D20": "alligator_crack", "D40": "pothole"}
NAMES = list(CLASSES.values())


def extract(zip_path: Path, dest: Path, country: str | None) -> Path:
    """Extract the zip, or only one country's files from the multi-country release.

    The full release may hold country folders or per-country zips; both are handled.
    zipfile strips absolute paths and ".." components, so members cannot escape `dest`.
    """
    marker = dest / ".extracted"
    if marker.exists():
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        if country is None:
            zf.extractall(dest)
        else:
            members = [m for m in zf.namelist() if f"/{country}/" in f"/{m}"]
            nested = [m for m in zf.namelist() if m.lower().endswith(".zip") and country.lower() in Path(m).stem.lower()]
            if members:
                zf.extractall(dest, members)
            elif nested:
                inner = Path(zf.extract(nested[0], dest))
                extract(inner, dest, None)
                inner.unlink()
            else:
                raise FileNotFoundError(f"no {country} files in {zip_path}")
    marker.touch()
    return dest


def find_train_dirs(root: Path) -> tuple[Path, Path]:
    for xml_dir in root.rglob("xmls"):
        if "train" in xml_dir.parts:
            images = xml_dir.parent.parent / "images"
            if images.is_dir():
                return images, xml_dir
    raise FileNotFoundError(f"no train/images + train/annotations/xmls under {root}")


def voc_to_yolo(xml_path: Path, counts: collections.Counter) -> tuple[list[str], int, int]:
    root = ET.parse(xml_path).getroot()
    w = float(root.findtext("size/width"))
    h = float(root.findtext("size/height"))
    lines = []
    for obj in root.iter("object"):
        code = (obj.findtext("name") or "").strip()
        counts[code] += 1
        if code not in CLASSES:
            continue
        box = obj.find("bndbox")
        x1, y1, x2, y2 = (float(box.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        x1, x2 = sorted((max(0.0, x1), min(w, x2)))
        y1, y2 = sorted((max(0.0, y1), min(h, y2)))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        cls = NAMES.index(CLASSES[code])
        lines.append(f"{cls} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
    return lines, int(w), int(h)


def place(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)  # same drive: no extra space
    except OSError:
        shutil.copy2(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("zip", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--country", help="e.g. India; extract only this country from the full release")
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    name = f"rdd2022_{args.country.lower()}_extracted" if args.country else args.zip.stem + "_extracted"
    extracted = extract(args.zip, args.out.parent / name, args.country)
    images_dir, xml_dir = find_train_dirs(extracted)
    xmls = sorted(xml_dir.glob("*.xml"))
    random.Random(args.seed).shuffle(xmls)
    n_val = int(len(xmls) * args.val_fraction)
    splits = {"val": xmls[:n_val], "train": xmls[n_val:]}

    counts: collections.Counter = collections.Counter()
    kept: collections.Counter = collections.Counter()
    for split, files in splits.items():
        (args.out / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for xml_path in files:
            image = images_dir / (xml_path.stem + ".jpg")
            if not image.exists():
                continue
            lines, _, _ = voc_to_yolo(xml_path, counts)
            kept[split] += 1
            place(image, args.out / "images" / split / image.name)
            # Images with no damage stay in as background examples.
            (args.out / "labels" / split / (xml_path.stem + ".txt")).write_text("\n".join(lines), encoding="utf-8")

    yaml = [f"path: {args.out.resolve().as_posix()}", "train: images/train", "val: images/val", "names:"]
    yaml += [f"  {i}: {n}" for i, n in enumerate(NAMES)]
    (args.out / "data.yaml").write_text("\n".join(yaml) + "\n", encoding="utf-8")

    print(f"images: train {kept['train']}, val {kept['val']}")
    print("labels in source (code: count):")
    for code, n in counts.most_common():
        print(f"  {code or '<empty>'}: {n}{'' if code in CLASSES else '  (ignored)'}")
    print(f"wrote {args.out / 'data.yaml'}")


if __name__ == "__main__":
    main()
