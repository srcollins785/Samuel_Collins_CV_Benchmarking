"""Rebuild the RGB working dataset from Kaggle's Animals-10.

Downloads `alessiocorrado99/animals10`, then writes a reproducible stratified
subset to ``data/`` together with CSV, JSON and JSONL manifests. The images
themselves are not committed to this repository; run this script once before
executing the RGB demonstration.

Requires Kaggle API credentials at ~/.kaggle/kaggle.json (Kaggle > Settings >
API > Create New Token).

Usage
-----
    python scripts/download_animals10.py                 # 500 images/class
    python scripts/download_animals10.py --per-class 10  # fast smoke set
    python scripts/download_animals10.py --keep-archive  # retain the 614 MB zip

Each tier is written to its own directory (``data/animals10_n<N>``) so several
sizes can coexist.

Sampling is deterministic and nested. The shuffle seed depends only on the
class name, never on N, so each tier is a prefix of the same fixed ordering:
n10 is byte-identical to the first 10 images of n100, filenames included.
That makes a comparison across tiers a genuine learning curve rather than
three unrelated samples.
"""

import argparse
import csv
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from PIL import Image

DATASET = "alessiocorrado99/animals10"

# Animals-10 ships Italian folder names.
IT_TO_EN = {
    "cane": "dog",           "cavallo": "horse",   "elefante": "elephant",
    "farfalla": "butterfly", "gallina": "chicken", "gatto": "cat",
    "mucca": "cow",          "pecora": "sheep",    "ragno": "spider",
    "scoiattolo": "squirrel",
}
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"


def download(work_dir: Path) -> Path:
    """Fetch and extract the archive, returning the raw-img directory."""
    raw = work_dir / "extracted" / "raw-img"
    if raw.is_dir():
        print(f"reusing existing extract at {raw}")
        return raw

    archive = work_dir / "animals10.zip"
    if not archive.is_file():
        work_dir.mkdir(parents=True, exist_ok=True)
        print(f"downloading {DATASET} (614 MB) ...")
        try:
            subprocess.run(
                ["kaggle", "datasets", "download", "-d", DATASET,
                 "-p", str(work_dir)],
                check=True,
            )
        except FileNotFoundError:
            sys.exit("kaggle CLI not found. Install it with: pip install kaggle")
        except subprocess.CalledProcessError:
            sys.exit("Kaggle download failed. Check ~/.kaggle/kaggle.json.")

    print("extracting ...")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(work_dir / "extracted")
    return raw


def build(raw: Path, per_class: int, out_dir: Path) -> None:
    """Sample `per_class` verified images per class into `out_dir`."""
    import random

    out_dir.mkdir(parents=True, exist_ok=True)
    images_root = out_dir / "images"
    if images_root.exists():
        shutil.rmtree(images_root)
    images_root.mkdir(parents=True)

    records, rejected = [], 0
    for italian, english in sorted(IT_TO_EN.items(), key=lambda kv: kv[1]):
        candidates = sorted(
            p for p in (raw / italian).iterdir()
            if p.is_file() and p.suffix.lower() in EXTS
        )
        random.Random(english).shuffle(candidates)

        class_dir = images_root / english
        class_dir.mkdir()

        picked = 0
        for path in candidates:
            if picked >= per_class:
                break
            try:  # verify the file genuinely decodes
                with Image.open(path) as im:
                    im.verify()
                with Image.open(path) as im:
                    im.convert("RGB").load()
            except Exception:
                rejected += 1
                continue

            picked += 1
            ext = ".jpeg" if path.suffix.lower() == ".jpg" else path.suffix.lower()
            name = f"{english}_{picked:03d}{ext}"
            shutil.copy2(path, class_dir / name)
            records.append({"image_path": f"images/{english}/{name}",
                            "class_name": english})

        if picked < per_class:
            print(f"!! {english}: only {picked} valid images available")

    # Manifest paths resolve relative to the manifest location.
    with open(out_dir / "labels.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_path", "class_name"])
        w.writeheader()
        w.writerows(records)
    with open(out_dir / "labels.json", "w") as f:
        json.dump(records, f, indent=2)
    with open(out_dir / "labels.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"wrote {len(records)} images across {len(IT_TO_EN)} classes to {out_dir}")
    print(f"skipped {rejected} undecodable source files")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-class", type=int, default=500,
                    help="images per class (default: 500)")
    ap.add_argument("--out", type=Path, default=None,
                    help="output directory (default: data/animals10_n<per-class>)")
    ap.add_argument("--work-dir", type=Path, default=REPO_ROOT.parent / "animals10_source",
                    help="where the archive is downloaded and extracted")
    ap.add_argument("--keep-archive", action="store_true",
                    help="keep the 614 MB zip and extracted copy afterward")
    args = ap.parse_args()

    out_dir = args.out or (DATA_DIR / f"animals10_n{args.per_class}")
    raw = download(args.work_dir)
    build(raw, args.per_class, out_dir)

    if not args.keep_archive:
        print(f"note: source archive retained at {args.work_dir} "
              f"(delete it manually, or pass --keep-archive to silence this)")


if __name__ == "__main__":
    main()
