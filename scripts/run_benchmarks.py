"""Run the benchmark over one or more dataset tiers, keeping every result.

The public function writes to ``./benchmark_results`` and takes no output
path, because section 2.3 says a caller must configure nothing beyond the four
documented parameters. That is the right constraint for a library and an
awkward one for a study that runs several tiers, since each run would
overwrite the last.

This script resolves it from the outside rather than by widening the public
signature: each tier runs in its own temporary working directory, and the
directory it produces is moved into ``benchmark_results/<tier>/`` afterwards.
The library keeps its simple contract, and the runs accumulate.

Usage
-----
    python scripts/run_benchmarks.py                     # every built tier
    python scripts/run_benchmarks.py animals10_n100      # just one

Build a tier first with ``scripts/download_animals10.py``.
"""

import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from samuel_collins_cv_benchmarking import (  # noqa: E402
    benchmark_image_classification,
)

DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "benchmark_results"


def available_tiers() -> list:
    """Every built dataset tier, smallest first."""
    if not DATA_DIR.is_dir():
        return []
    tiers = [
        path.name for path in sorted(DATA_DIR.iterdir())
        if path.is_dir() and (path / "labels.csv").is_file()
    ]

    def size(name):
        digits = "".join(c for c in name if c.isdigit())
        return int(digits) if digits else 0

    return sorted(tiers, key=size)


def run_tier(tier: str) -> dict:
    """Benchmark one tier and file its results under that tier's name."""
    manifest = DATA_DIR / tier / "labels.csv"
    if not manifest.is_file():
        raise FileNotFoundError(
            f"No manifest at {manifest}. Build the tier first with "
            f"scripts/download_animals10.py --per-class <N>"
        )

    print(f"\n{'=' * 68}\n{tier}\n{'=' * 68}")
    started = time.perf_counter()

    # Run somewhere disposable so the benchmark's own output directory cannot
    # collide with a previous tier's.
    workspace = Path(tempfile.mkdtemp(prefix=f"{tier}_"))
    previous = Path.cwd()
    try:
        os.chdir(workspace)
        results = benchmark_image_classification(
            dataset=str(manifest),          # absolute, so the cwd move is safe
            dataset_type="csv",
            target_labels="class_name",
            color_mode="rgb",
        )
        produced = workspace / "benchmark_results"
    finally:
        os.chdir(previous)

    destination = RESULTS_DIR / tier
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(produced), str(destination))
    shutil.rmtree(workspace, ignore_errors=True)

    elapsed = time.perf_counter() - started
    print(f"\n{tier}: {elapsed:.1f}s, best {results['best_model']}, "
          f"results in {destination.relative_to(REPO_ROOT)}")
    return results


def main() -> None:
    tiers = sys.argv[1:] or available_tiers()
    if not tiers:
        sys.exit(
            "No dataset tiers found under data/. Build one first:\n"
            "  python scripts/download_animals10.py --per-class 500"
        )

    summaries = {}
    for tier in tiers:
        summaries[tier] = run_tier(tier)

    if len(summaries) > 1:
        # Tiers are nested subsets of one ordering, so reading across them is
        # a learning curve rather than a comparison of unrelated samples.
        print(f"\n{'=' * 68}\nmacro F1 across tiers\n{'=' * 68}")
        models = sorted(
            next(iter(summaries.values()))["model_results"],
            key=lambda key: -(list(summaries.values())[-1]["model_results"][key]["macro_f1"] or 0),
        )
        header = "  ".join(f"{tier:>16}" for tier in summaries)
        print(f"{'model':22}{header}")
        for key in models:
            cells = "  ".join(
                f"{(summary['model_results'][key]['macro_f1'] or 0):>16.4f}"
                for summary in summaries.values()
            )
            name = next(iter(summaries.values()))["model_results"][key]["name"]
            print(f"{name:22}{cells}")


if __name__ == "__main__":
    main()
