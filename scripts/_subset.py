"""Shared subset building for the dataset download scripts.

Both datasets arrive as a folder per class and leave as a reproducible
stratified subset with manifests. Only the source layout differs - Animals-10
uses Italian folder names at the archive root, Intel nests its classes under
``seg_train`` - so the download scripts differ in how they locate the class
folders and share everything after that.

The sampling is deterministic and nested: the shuffle seed comes from the
output class name and never from N, so the file ordering per class is fixed
and a smaller subset is a strict prefix of a larger one. That is what makes
two tiers comparable as a learning curve rather than as two unrelated draws.
"""

import csv
import json
import random
import shutil
from pathlib import Path

from PIL import Image

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def build_subset(class_dirs: dict, per_class: int, out_dir: Path) -> int:
    """Copy `per_class` verified images per class into ``out_dir``.

    Parameters
    ----------
    class_dirs
        Maps the output class name to the source directory holding its
        images. The output name is what ends up in the manifests, so a
        dataset with unhelpful folder names is renamed here.
    per_class
        Images to take from each class.
    out_dir
        Destination. ``images/<class>/`` plus the three manifests.

    Returns
    -------
    int
        Number of images written.
    """
    out_dir = Path(out_dir)
    images_root = out_dir / "images"
    if images_root.exists():
        shutil.rmtree(images_root)
    images_root.mkdir(parents=True)

    records, rejected = [], 0

    for class_name in sorted(class_dirs):
        source = Path(class_dirs[class_name])
        candidates = sorted(
            path for path in source.iterdir()
            if path.is_file() and path.suffix.lower() in EXTENSIONS
        )
        # Seeded on the class name only, so the ordering does not depend on N.
        random.Random(class_name).shuffle(candidates)

        class_dir = images_root / class_name
        class_dir.mkdir()

        picked = 0
        for path in candidates:
            if picked >= per_class:
                break
            try:  # verify the file genuinely decodes before counting it
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    image.convert("RGB").load()
            except Exception:
                rejected += 1
                continue

            picked += 1
            extension = ".jpeg" if path.suffix.lower() == ".jpg" else path.suffix.lower()
            name = f"{class_name}_{picked:04d}{extension}"
            shutil.copy2(path, class_dir / name)
            records.append({"image_path": f"images/{class_name}/{name}",
                            "class_name": class_name})

        if picked < per_class:
            print(f"!! {class_name}: only {picked} valid images available")

    write_manifests(out_dir, records)
    print(f"wrote {len(records)} images across {len(class_dirs)} classes to {out_dir}")
    if rejected:
        print(f"skipped {rejected} undecodable source files")
    return len(records)


def write_manifests(out_dir: Path, records: list) -> None:
    """Write CSV, JSON and JSONL manifests beside the images.

    Paths inside them are relative to the manifest's own directory, so the
    whole tier can be moved without editing a row.
    """
    out_dir = Path(out_dir)
    with open(out_dir / "labels.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_path", "class_name"])
        writer.writeheader()
        writer.writerows(records)
    with open(out_dir / "labels.json", "w") as handle:
        json.dump(records, handle, indent=2)
    with open(out_dir / "labels.jsonl", "w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def describe_shapes(images_root: Path, limit: int = 2000) -> dict:
    """Aspect-ratio statistics, to say what letterboxing will cost.

    Fitting a rectangular image into a square canvas leaves padding equal to
    ``1 - short / long``, so the aspect ratios predict the waste before any
    training happens.
    """
    paths = [p for p in Path(images_root).rglob("*") if p.is_file()][:limit]
    ratios, padding = [], []
    for path in paths:
        try:
            with Image.open(path) as image:
                width, height = image.size
        except Exception:
            continue
        long_side, short_side = max(width, height), min(width, height)
        ratios.append(long_side / short_side)
        padding.append(1 - short_side / long_side)

    if not ratios:
        return {}
    ordered = sorted(padding)
    return {
        "images_measured": len(ratios),
        "median_aspect_ratio": sorted(ratios)[len(ratios) // 2],
        "mean_padding_share": sum(padding) / len(padding),
        "p90_padding_share": ordered[int(len(ordered) * 0.9)],
        "max_padding_share": ordered[-1],
    }
