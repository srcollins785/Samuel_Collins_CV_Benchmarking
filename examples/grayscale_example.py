"""The single-channel demonstration.

One-channel input: every image becomes ``(64, 64, 1)``, and the classical
models see 64 x 64 = 4,096 features each, a third of the RGB count.

Run ``rgb_example.py`` as well and compare. On Animals-10 color is worth
roughly a quarter of the CNN's macro F1 and costs the SVM close to forty
times its training run; on Intel Image Classification the same comparison
comes out differently for the Decision Tree, which is why the report treats
that effect as a property of the data rather than of the model.

    python examples/grayscale_example.py
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
    manifest = tier / ("mini_labels.csv" if using_fixtures else "labels.csv")

    print(f"grayscale demonstration: {manifest}")
    results = benchmark_image_classification(
        dataset=str(manifest),
        dataset_type="csv",
        target_labels="class_name",
        color_mode="grayscale",      # one channel, 4,096 features
    )
    report(results, " (test fixtures)" if using_fixtures else "")

    features = results["dataset_information"]["image_shape"]
    print(f"flattened feature width: {features[0] * features[1] * features[2]:,}")


if __name__ == "__main__":
    main()
