"""Run the deep CNN benchmark and rebuild the combined comparison.

    python run_benchmark.py                        # all architectures
    python run_benchmark.py --model resnet50       # one architecture
    python run_benchmark.py --model all            # explicit form of the default
    python run_benchmark.py --tables-only          # rebuild tables and plots

Lives at the repository root because the assignment names this command. The
work itself is in the package, so this file is argument parsing, a preflight,
and progress printing.

The preflight is the part worth reading. Section 1A requires the deep
architectures to be compared against the Part 1 baselines on identical
samples, and this script does not re-run Part 1 - it rebuilds the split from
the same manifest at the same seed and trusts it to come out the same. That
trust is checkable, so it is checked. Part 1's published artifacts record,
for each model, the dataset index behind several test positions, and the
Simple CNN's history records the exact rows it held back for validation.
Both are compared against the freshly rebuilt split before any training
starts. If the data on disk has changed since Part 1 ran, this stops with an
explanation rather than producing a comparison that looks valid and is not.
"""

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from samuel_collins_cv_benchmarking import (  # noqa: E402
    combined,
    deep_data,
    deep_models,
    deep_visualization,
)
from samuel_collins_cv_benchmarking._config import (  # noqa: E402
    CNN_EPOCHS,
    CNN_LEARNING_RATE,
    PLOTS_DIR,
)
from samuel_collins_cv_benchmarking.benchmark import make_split  # noqa: E402
from samuel_collins_cv_benchmarking.data_loader import load_csv  # noqa: E402
from samuel_collins_cv_benchmarking.preprocessing import preprocess  # noqa: E402

DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "benchmark_results"

# The configuration the deep architectures run on. Ten classes, 5,000 images,
# RGB - pretrained ImageNet backbones need three channels, and the Part 1
# results for this same configuration are what the master table compares
# against. The other Part 1 configurations remain the tier and color-mode
# ablation they already were.
DEFAULT_CONFIG = "animals10_n500_rgb"


def parse_config(name: str) -> tuple:
    """Split a result directory name into its tier and color mode."""
    for mode in ("grayscale", "rgb"):
        if name.endswith(f"_{mode}"):
            return name[: -len(mode) - 1], mode
    raise SystemExit(
        f"Cannot read a color mode from {name!r}. Expected a name like "
        "'animals10_n500_rgb'."
    )


def rebuild_split(tier: str, color_mode: str):
    """Reproduce the Part 1 split from the same manifest at the same seed.

    Loaded through the manifest as ``csv`` input, exactly as Part 1's runner
    did, because sample order depends on how the dataset was read and the
    split depends on sample order.
    """
    manifest = DATA_DIR / tier / "labels.csv"
    if not manifest.is_file():
        raise SystemExit(
            f"No manifest at {manifest.relative_to(REPO_ROOT)}. Build the "
            f"tier first:\n  python scripts/download_animals10.py --per-class 500"
        )
    prepared = preprocess(load_csv(str(manifest), "class_name"), color_mode)
    return make_split(prepared)


def verify_against_part1(split, config_dir: Path) -> dict:
    """Prove the rebuilt split matches the one Part 1 published.

    Two independent checks, because they fail for different reasons. The
    test-position check catches a changed dataset or a changed read order;
    the validation-row check catches a drift between the fraction the Simple
    CNN used and the one the deep models use, which would leave the two model
    families validating on different images.
    """
    metrics_path = config_dir / "benchmark_metrics.json"
    if not metrics_path.is_file():
        raise SystemExit(
            f"No Part 1 results at {metrics_path.relative_to(REPO_ROOT)}. The "
            "combined table needs them. Run the Part 1 benchmark first:\n"
            "  python scripts/run_benchmarks.py"
        )

    stored = json.loads(metrics_path.read_text(encoding="utf-8"))
    checked = 0

    # Check 1: every recorded (test_position -> dataset_index) pair must still
    # hold in the rebuilt split.
    for key, entry in stored.items():
        examples = entry.get("prediction_examples") or {}
        for bucket in ("correct", "incorrect"):
            for example in examples.get(bucket, []):
                position = example.get("test_position")
                expected = example.get("dataset_index")
                if position is None or expected is None:
                    continue
                if position >= len(split.test_index):
                    raise SystemExit(
                        f"Part 1 recorded test position {position} for {key}, "
                        f"but the rebuilt test half has only "
                        f"{len(split.test_index)} images. The dataset under "
                        "data/ has changed since Part 1 ran."
                    )
                actual = int(split.test_index[position])
                if actual != expected:
                    raise SystemExit(
                        f"Split mismatch. Part 1 recorded dataset index "
                        f"{expected} at test position {position} for {key}; "
                        f"the rebuilt split has {actual}. The dataset under "
                        "data/ has changed since Part 1 ran, so the deep "
                        "results would not be comparable with the Part 1 "
                        "results. Re-run Part 1 before continuing:\n"
                        "  python scripts/run_benchmarks.py"
                    )
                checked += 1

    # Check 2: the deep validation rows must be the rows the Simple CNN
    # actually held back, as recorded in its published history.
    recorded = ((stored.get("simple_cnn") or {}).get("training_history")
                or {}).get("validation_index")
    validation_match = None
    if recorded:
        _, ours = deep_data.validation_split(split.labels_train)
        validation_match = set(int(i) for i in ours) == set(int(i) for i in recorded)
        if not validation_match:
            raise SystemExit(
                "The deep validation slice is not the slice the Part 1 Simple "
                "CNN held back. Both read the same fraction at the same seed, "
                "so this means one of them has changed. The two model "
                "families would be validating on different images and their "
                "convergence curves would not be comparable."
            )

    return {
        "test_positions_checked": checked,
        "validation_rows_match_simple_cnn": validation_match,
        "test_half_size": int(len(split.test_index)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--model", default="all",
        help="architecture key, or 'all' (default). One of: "
             + ", ".join(deep_models.architecture_names()))
    parser.add_argument("--config", default=DEFAULT_CONFIG,
                        help=f"result directory name (default {DEFAULT_CONFIG})")
    parser.add_argument("--epochs", type=int, default=CNN_EPOCHS,
                        help=f"training epochs (default {CNN_EPOCHS}, the "
                             "assignment minimum)")
    parser.add_argument("--learning-rate", type=float, default=CNN_LEARNING_RATE,
                        help=f"initial learning rate (default {CNN_LEARNING_RATE}). "
                             "Any change from the default must be documented "
                             "in the report.")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="train from scratch instead of fine-tuning")
    parser.add_argument("--tables-only", action="store_true",
                        help="rebuild the combined table, rankings and plots "
                             "from existing results without training")
    parser.add_argument("--skip-verify", action="store_true",
                        help="skip the split verification (not recommended)")
    arguments = parser.parse_args()

    tier, color_mode = parse_config(arguments.config)
    config_dir = RESULTS_DIR / arguments.config

    if arguments.model != "all" and arguments.model not in deep_models.architecture_names():
        raise SystemExit(
            f"Unknown architecture {arguments.model!r}. Available: "
            + ", ".join(deep_models.architecture_names())
            + ", or 'all'."
        )

    started = time.perf_counter()

    if not arguments.tables_only:
        print(f"configuration: {arguments.config}  ({tier}, {color_mode})")
        print("rebuilding the Part 1 split from the same manifest at seed 42")
        split = rebuild_split(tier, color_mode)

        if arguments.skip_verify:
            print("  split verification SKIPPED - results may not be "
                  "comparable with Part 1")
        else:
            verified = verify_against_part1(split, config_dir)
            print(f"  verified: {verified['test_positions_checked']} recorded "
                  f"test positions match, test half {verified['test_half_size']} "
                  f"images, validation rows match the Simple CNN: "
                  f"{verified['validation_rows_match_simple_cnn']}")

        keys = (deep_models.architecture_names() if arguments.model == "all"
                else [arguments.model])

        from samuel_collins_cv_benchmarking import deep_benchmark

        outcome = deep_benchmark.benchmark_all(
            split,
            color_mode,
            config_dir,
            keys=keys,
            epochs=arguments.epochs,
            learning_rate=arguments.learning_rate,
            pretrained=not arguments.no_pretrained,
            on_progress=lambda message: print(message, flush=True),
        )
        print(f"\nwrote deep artifacts to {outcome['output_directory']}")

    # The combined table is rebuilt every time, including after a single
    # architecture, so the master table is never stale relative to the
    # per-model artifacts beside it.
    print("\nbuilding the combined Part 1 + Part 2 comparison")
    built = combined.build_all(config_dir)
    rows = built["rows"]

    plots = deep_visualization.plot_all_comparisons(rows, config_dir / PLOTS_DIR)
    print(f"  {built['combined_csv']}")
    if built["per_class_csv"]:
        print(f"  {built['per_class_csv']}")
    print(f"  {built['rankings_json']}")
    print(f"  {len(plots)} comparison plots in {config_dir / PLOTS_DIR}")

    ready = [row for row in rows if row["status"] == "ok"
             and row["accuracy"] is not None]
    ready.sort(key=lambda row: row["accuracy"], reverse=True)
    if ready:
        print(f"\n{'model':22}{'family':19}{'accuracy':>10}{'macro F1':>10}"
              f"{'params':>12}{'ms/img':>9}")
        print("-" * 82)
        for row in ready:
            parameters = (f"{row['total_parameters'] / 1e6:.1f}M"
                          if row["total_parameters"] else "N/A")
            latency = f"{row['latency_ms']:.3f}" if row["latency_ms"] else "N/A"
            print(f"{row['name']:22}{row['family']:19}{row['accuracy']:>10.4f}"
                  f"{row['macro_f1']:>10.4f}{parameters:>12}{latency:>9}")

    failed = [row for row in rows if row["status"] != "ok"]
    if failed:
        print(f"\n{len(failed)} model(s) did not produce results:")
        for row in failed:
            print(f"  {row['name']}: {row['status']}")

    print(f"\ntotal {(time.perf_counter() - started) / 60:.1f} minutes")


if __name__ == "__main__":
    main()
