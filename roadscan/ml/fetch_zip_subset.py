"""Download only some members of a huge remote zip, using HTTP range requests.

    python -I fetch_zip_subset.py https://ndownloader.figshare.com/files/38030910 D:/pothole_mapping/data/rdd2022_raw/RDD2022_partial.zip --match /India/

Creates a local file the size of the remote zip, fills in its tail (the central
directory) and the byte ranges of the matching members, and leaves the rest as
zeros. zipfile can then open it normally and extract the matching members.
Many parallel connections work around per-connection throttling.
"""
from __future__ import annotations

import argparse
import threading
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

CHUNK = 4 * 1024 * 1024
TAIL = 1024 * 1024  # enough for the end-of-central-directory records
MARGIN = 64 * 1024  # local header + extra field slack before/after each member

_lock = threading.Lock()


def remote_size(url: str) -> int:
    req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Range"].rsplit("/", 1)[1])


def fetch(url: str, start: int, end: int, out: Path, retries: int = 6) -> int:
    """Write bytes [start, end] of url into out at the same offset."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=120) as r:
                if r.status != 206 or not r.headers.get("Content-Range", "").startswith(f"bytes {start}-"):
                    raise IOError(f"server ignored range {start}-{end}")
                data = r.read()
            if len(data) != end - start + 1:
                raise IOError(f"short read {len(data)} for {start}-{end}")
            with _lock, open(out, "r+b") as f:
                f.seek(start)
                f.write(data)
            return len(data)
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))
    return 0


def merge(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[list[int]] = []
    for s, e in sorted(ranges):
        if out and s <= out[-1][1] + 1:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def download(url: str, out: Path, ranges: list[tuple[int, int]], workers: int, label: str) -> None:
    chunks = [(s, min(s + CHUNK - 1, e)) for rs, e in ranges for s in range(rs, e + 1, CHUNK)]
    total = sum(e - s + 1 for s, e in chunks)
    done, t0 = 0, time.time()
    with ThreadPoolExecutor(workers) as pool:
        for fut in as_completed([pool.submit(fetch, url, s, e, out) for s, e in chunks]):
            done += fut.result()
            rate = done / max(time.time() - t0, 1e-3) / 1e6
            print(f"\r{label}: {done / 1e6:8.1f} / {total / 1e6:.1f} MB  {rate:5.2f} MB/s", end="", flush=True)
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("out", type=Path)
    ap.add_argument("--match", required=True, help="substring of member paths to fetch, e.g. /India/")
    ap.add_argument("--size", type=int, help="remote size in bytes (skips a probe request)")
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()

    size = args.size or remote_size(args.url)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if not args.out.exists() or args.out.stat().st_size != size:
        with open(args.out, "wb") as f:
            f.truncate(size)

    # The end record says where the central directory starts; then fetch the whole directory.
    download(args.url, args.out, [(max(0, size - TAIL), size - 1)], args.workers, "end record")
    with open(args.out, "rb") as f:
        end = zipfile._EndRecData(f)
    cd_start = end[zipfile._ECD_OFFSET]
    if cd_start < size - TAIL:
        download(args.url, args.out, [(cd_start, size - TAIL - 1)], args.workers, "index")
    with zipfile.ZipFile(args.out) as zf:
        infos = zf.infolist()
    offsets = sorted({i.header_offset for i in infos} | {cd_start})
    next_offset = {o: offsets[k + 1] for k, o in enumerate(offsets[:-1])}
    wanted = [i for i in infos if args.match in f"/{i.filename}"]
    if not wanted:
        raise SystemExit(f"no members match {args.match!r}")
    # Each member runs from its local header to the next member's header.
    ranges = merge([(i.header_offset, min(next_offset.get(i.header_offset, size) - 1, i.header_offset + i.compress_size + MARGIN)) for i in wanted])
    print(f"{len(wanted)} of {len(infos)} members match; {sum(e - s + 1 for s, e in ranges) / 1e6:.1f} MB in {len(ranges)} range(s)")
    download(args.url, args.out, ranges, args.workers, "members")

    with zipfile.ZipFile(args.out) as zf:
        bad = [i.filename for i in wanted if not i.is_dir() and _crc_bad(zf, i)]
    print("CRC check: all matching members OK" if not bad else f"CRC check: {len(bad)} bad, e.g. {bad[:3]}")


def _crc_bad(zf: zipfile.ZipFile, info: zipfile.ZipInfo) -> bool:
    try:
        with zf.open(info) as f:
            while f.read(1 << 20):
                pass
        return False
    except (zipfile.BadZipFile, OSError):
        return True


if __name__ == "__main__":
    main()
