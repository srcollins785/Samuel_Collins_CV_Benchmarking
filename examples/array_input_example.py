"""Benchmark an in-memory NumPy array.

Array input arrives already decoded, so there are no files to read. Accepted
shapes are ``(N, H, W)``, ``(N, H, W, 1)`` and ``(N, H, W, 3)``, and
``target_labels`` is the label vector itself - one entry per image.

This is the form to use when the images come from somewhere other than a
filesystem: a database, a camera, or another library's loader.

    python examples/array_input_example.py
"""

import numpy as np
from PIL import Image

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from samuel_collins_cv_benchmarking import benchmark_image_classification


def pick_dataset():
    """The smallest downloaded tier, or the committed fixtures.

    Smallest rather than best: an example is there to show the call working,
    and the 500-per-class tier takes about twelve minutes where the
    100-per-class one takes forty seconds. The report is where the full runs
    are presented.

    The image sets are not committed - they are third-party collections - so
    an example that required them would fail on a fresh clone. The fixtures
    are tiny and the resulting scores are meaningless, but the call, the
    outputs and the returned dictionary are exactly the real ones.
    """
    data = REPO_ROOT / "data"
    tiers = []
    if data.is_dir():
        for tier in data.iterdir():
            manifest = tier / "labels.csv"
            if manifest.is_file():
                rows = sum(1 for _ in manifest.open()) - 1
                tiers.append((rows, tier))
    if tiers:
        return min(tiers)[1], False
    return REPO_ROOT / "tests" / "fixtures", True


def report(results, note=""):
    """Print the parts of the result a reader wants to see."""
    information = results["dataset_information"]
    print(f"\n{information['number_of_images']} images, "
          f"{information['number_of_classes']} classes, "
          f"shape {information['image_shape']}, "
          f"mode {information['color_mode']}{note}")
    print(f"best model: {results['best_model']}\n")
    print(results["summary"].to_string(index=False))
    print(f"\noutputs written to {results['output_directory']}")


def load_into_memory(root, limit_per_class=60):
    """Read images off disk into one tensor, the way a caller might.

    Everything is resized to a common size first: an array has one shape, so
    a ragged collection cannot be stacked. The package resizes again to its
    own 64x64 afterwards.
    """
    images, labels = [], []
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for path in sorted(class_dir.iterdir())[:limit_per_class]:
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}:
                continue
            try:
                with Image.open(path) as image:
                    images.append(np.asarray(image.convert("RGB").resize((96, 96))))
            except Exception:
                continue          # a corrupt file is not this example's subject
            labels.append(class_dir.name)
    return np.stack(images), labels


def main():
    tier, using_fixtures = pick_dataset()
    root = tier / ("mini" if using_fixtures else "images")

    tensor, labels = load_into_memory(root)
    print(f"array input: tensor {tensor.shape}, dtype {tensor.dtype}, "
          f"{len(set(labels))} classes")

    results = benchmark_image_classification(
        dataset=tensor,              # the images themselves, not a path
        dataset_type="array",
        target_labels=labels,        # one label per image
        color_mode="rgb",
    )
    report(results, " (test fixtures)" if using_fixtures else "")


if __name__ == "__main__":
    main()
