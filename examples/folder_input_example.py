"""Benchmark a class-folder dataset.

Each subfolder is a class and its name is the label::

    animals/
    |-- cat/cat_001.jpg
    |-- dog/dog_001.jpg
    `-- horse/horse_001.jpg

Here ``target_labels`` is the list of class-folder names to load.

    python examples/folder_input_example.py
"""

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


def main():
    tier, using_fixtures = pick_dataset()
    root = tier / ("mini" if using_fixtures else "images")
    classes = sorted(p.name for p in root.iterdir() if p.is_dir())

    print(f"folder input: {root}")
    results = benchmark_image_classification(
        dataset=str(root),
        dataset_type="folder",
        target_labels=classes,       # a list of class-folder names
        color_mode="rgb",
    )
    report(results, " (test fixtures)" if using_fixtures else "")


if __name__ == "__main__":
    main()
