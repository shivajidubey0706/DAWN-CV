"""Download PASCAL VOC 2007 / 2012 from documented public endpoints.

VOC is the paper's training source (Sec. V-A). The official host is
http://host.robots.ox.ac.uk/pascal/VOC/ ; Ultralytics mirrors the same archives
on GitHub releases, which is usually faster. Both are tried in order.

Usage:
    python scripts/download_voc.py --sets 2007trainval 2007test
    python scripts/download_voc.py --sets 2012trainval      # ~1.9 GB
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SOURCES = {
    "2007trainval": [
        "https://github.com/ultralytics/assets/releases/download/v0.0.0/VOCtrainval_06-Nov-2007.zip",
        "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar",
    ],
    "2007test": [
        "https://github.com/ultralytics/assets/releases/download/v0.0.0/VOCtest_06-Nov-2007.zip",
        "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar",
    ],
    "2012trainval": [
        "https://github.com/ultralytics/assets/releases/download/v0.0.0/VOCtrainval_11-May-2012.zip",
        "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar",
    ],
}


def _download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e6:8.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print()
    tmp.replace(dest)


def _extract(archive: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            for m in z.namelist():  # refuse path traversal
                if Path(m).is_absolute() or ".." in Path(m).parts:
                    raise RuntimeError(f"unsafe path in archive: {m}")
            z.extractall(out)
    else:
        with tarfile.open(archive) as t:
            t.extractall(out, filter="data")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=["2007trainval", "2007test"], choices=list(SOURCES))
    ap.add_argument("--out", default=str(ROOT / "data" / "VOC"))
    ap.add_argument("--keep-archives", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    cache = ROOT / "data" / "downloads"
    cache.mkdir(parents=True, exist_ok=True)
    for name in args.sets:
        for url in SOURCES[name]:
            archive = cache / url.rsplit("/", 1)[-1]
            try:
                if not archive.exists():
                    print(f"[{name}] downloading {url}")
                    _download(url, archive)
                print(f"[{name}] extracting {archive.name} -> {out}")
                _extract(archive, out)
                break
            except Exception as e:  # try next mirror
                print(f"[{name}] failed from {url}: {e}", file=sys.stderr)
                archive.unlink(missing_ok=True)
        else:
            print(f"[{name}] all sources failed; download manually (see docs/DATASETS.md)", file=sys.stderr)
            return 1
        if not args.keep_archives:
            archive.unlink(missing_ok=True)
    # Ultralytics zips extract to VOC/VOCdevkit/...; official tars to VOCdevkit/... — normalise.
    nested = out / "VOC" / "VOCdevkit"
    if nested.exists():
        for year_dir in nested.iterdir():
            target = out / "VOCdevkit" / year_dir.name
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(year_dir), str(target))
            else:  # merge (trainval + test share a year dir)
                for p in year_dir.rglob("*"):
                    if p.is_file():
                        dst = target / p.relative_to(year_dir)
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        if not dst.exists():
                            shutil.move(str(p), str(dst))
        shutil.rmtree(out / "VOC", ignore_errors=True)
    print("done:", [p.name for p in (out / "VOCdevkit").iterdir()] if (out / "VOCdevkit").exists() else "nothing")
    return 0


if __name__ == "__main__":
    sys.exit(main())
