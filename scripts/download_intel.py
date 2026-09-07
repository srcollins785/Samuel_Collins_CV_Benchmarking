"""Build an RGB dataset tier from Kaggle's Intel Image Classification.

Six natural-scene classes - buildings, forest, glacier, mountain, sea, street -
at 150x150 pixels. The size is the point: the images are square and larger
than the 64x64 the package standardises to, so every one downscales into the
canvas with no padding at all.

That is the contrast with Animals-10, whose photographs are mostly 4:3 and
lose about 27% of the 64x64 canvas to the zero padding that preserves their
aspect ratio. Running both isolates what that padding costs, which no single
dataset can show.

Requires Kaggle API credentials at ~/.kaggle/kaggle.json.

Usage
-----
    python scripts/download_intel.py                  # 500 images/class
    python scripts/download_intel.py --per-class 100
"""

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _subset import build_subset, describe_shapes  # noqa: E402

DATASET = "puneet6060/intel-image-classification"
CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"


def download(work_dir: Path) -> Path:
    """Fetch and extract, returning the directory holding the class folders."""
    # The archive nests the training split two levels deep.
    training = work_dir / "extracted" / "seg_train" / "seg_train"
    if training.is_dir():
        print(f"reusing existing extract at {training}")
        return training

    archive = work_dir / "intel-image-classification.zip"
    if not archive.is_file():
        work_dir.mkdir(parents=True, exist_ok=True)
        print(f"downloading {DATASET} (about 350 MB) ...")
        try:
            subprocess.run(
                ["kaggle", "datasets", "download", "-d", DATASET, "-p", str(work_dir)],
                check=True,
            )
        except FileNotFoundError:
            sys.exit("kaggle CLI not found. Install it with: pip install kaggle")
        except subprocess.CalledProcessError:
            sys.exit("Kaggle download failed. Check ~/.kaggle/kaggle.json.")

    print("extracting ...")
    with zipfile.ZipFile(archive) as archive_file:
        archive_file.extractall(work_dir / "extracted")

    if not training.is_dir():
        sys.exit(
            f"Expected class folders at {training}, which is not there. "
            "The archive layout may have changed."
        )
    return training


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-class", type=int, default=500,
                        help="images per class (default: 500)")
    parser.add_argument("--out", type=Path, default=None,
                        help="output directory (default: data/intel_n<per-class>)")
    parser.add_argument("--work-dir", type=Path,
                        default=REPO_ROOT.parent / "intel_source",
                        help="where the archive is downloaded and extracted")
    arguments = parser.parse_args()

    training = download(arguments.work_dir)

    missing = [name for name in CLASSES if not (training / name).is_dir()]
    if missing:
        found = sorted(p.name for p in training.iterdir() if p.is_dir())
        sys.exit(f"Class folder(s) not found: {missing}. Present: {found}.")

    out_dir = arguments.out or (DATA_DIR / f"intel_n{arguments.per_class}")
    build_subset({name: training / name for name in CLASSES},
                 arguments.per_class, out_dir)

    shapes = describe_shapes(out_dir / "images")
    if shapes:
        print(
            f"\naspect ratio median {shapes['median_aspect_ratio']:.2f}, "
            f"padding at 64x64: mean {shapes['mean_padding_share']:.1%}, "
            f"p90 {shapes['p90_padding_share']:.1%}, "
            f"max {shapes['max_padding_share']:.1%}"
        )


if __name__ == "__main__":
    main()
