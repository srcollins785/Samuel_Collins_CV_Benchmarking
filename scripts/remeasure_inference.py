"""Re-measure inference cost for every trained architecture, uniformly.

Exists because the first memory measurement was wrong in a way that only
showed up once all nine architectures had run.

It read ``torch.mps.driver_allocated_memory()``, which is the process-wide
allocator pool rather than any model's footprint. The pool grows as models are
loaded and is never handed back, so reading it once per architecture in a
nine-architecture run produced a column that increased monotonically in run
order: AlexNet 2.4 GB, then every later architecture between 12 and 15 GB
regardless of its size. DenseNet121 appeared to need 12.7 GB, which was simply
everything allocated before it.

Memory feeds the deployment score in Ranking E, so leaving that in place would
have produced a ranking derived from the order the architectures happened to
run in. The fix is in ``deep_metrics``; this script applies it to the results
already on disk without retraining anything, by reloading each saved
checkpoint and re-running only the measurement passes.

Timing is re-measured at the same time, so latency and memory for every
architecture come from one consistent pass under the same conditions rather
than from whatever state the process was in when that architecture's turn came.

    python scripts/remeasure_inference.py
    python scripts/remeasure_inference.py --config animals10_n500_rgb
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from samuel_collins_cv_benchmarking import (  # noqa: E402
    combined,
    deep_data,
    deep_metrics,
    deep_models,
    deep_visualization,
)
from samuel_collins_cv_benchmarking._config import (  # noqa: E402
    CHECKPOINTS_DIR,
    PLOTS_DIR,
)

DEFAULT_CONFIG = "animals10_n500_rgb"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    arguments = parser.parse_args()

    from run_benchmark import parse_config, rebuild_split, verify_against_part1

    tier, color_mode = parse_config(arguments.config)
    config_dir = REPO_ROOT / "benchmark_results" / arguments.config
    metrics_path = config_dir / "deep_metrics.json"
    if not metrics_path.is_file():
        raise SystemExit(f"No deep results at {metrics_path}")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    device = deep_models.device_for()

    print(f"configuration: {arguments.config}   device: {device}")
    split = rebuild_split(tier, color_mode)
    verified = verify_against_part1(split, config_dir)
    print(f"  split verified: {verified['test_positions_checked']} test "
          f"positions match")

    # Only the test half is needed, but build_loaders caches all three in one
    # decode pass and the cost is a few seconds.
    bundle = deep_data.build_loaders(split, color_mode)
    class_names = bundle["deep_split"].class_names
    test_loader = bundle["test"]

    updated = 0
    print(f"\n{'architecture':22}{'latency ms':>12}{'img/s':>10}"
          f"{'weights MB':>11}{'peak MB':>11}{'1-img ms':>11}")
    print("-" * 77)

    for key in list(metrics):
        entry = metrics[key]
        if not entry.get("succeeded"):
            continue
        checkpoint = config_dir / CHECKPOINTS_DIR / f"best_{key}.pt"
        if not checkpoint.is_file():
            # YOLO has no checkpoint here - it trains in its own environment
            # and reports its own figures.
            print(f"{entry.get('name', key):22}{'(no checkpoint - skipped)':>45}")
            continue

        import torch

        model = deep_models.get_model(key, num_classes=len(class_names),
                                      pretrained=False)
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"))

        inference = deep_metrics.inference_benchmark(model, test_loader, device)
        single = deep_metrics.single_image_latency(model, test_loader, device)
        # Measured in its own pass: the hooks it needs would inflate the
        # latency figures above.
        memory = deep_metrics.peak_activation_memory(model, test_loader, device)

        # The headline memory number is the working set - weights plus the
        # largest intermediate tensors alive at once - which is what a
        # deployment target has to fit. Sampling between batches instead
        # reproduces the checkpoint size and tells the reader nothing new.
        inference["peak_memory_mb"] = memory["peak_memory_mb"]
        inference["memory_measurement"] = memory["measurement"]

        entry["inference"] = inference
        entry["single_image_inference"] = single
        entry["memory"] = memory
        entry["inference_time_ms_per_image"] = inference["latency_ms_per_image"]
        updated += 1

        print(f"{entry.get('name', key):22}"
              f"{inference['latency_ms_per_image']:>12.4f}"
              f"{inference['throughput_images_per_second']:>10.1f}"
              f"{memory['weights_baseline_mb']:>11.1f}"
              f"{memory['peak_memory_mb']:>11.1f}"
              f"{single.get('median_latency_ms', 0):>11.3f}")

        # Released before the next architecture loads, so each measurement
        # describes one model rather than an accumulation.
        del model
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()

    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nre-measured {updated} architecture(s)")

    built = combined.build_all(config_dir)
    deep_visualization.plot_all_comparisons(built["rows"], config_dir / PLOTS_DIR)
    print(f"rebuilt {built['combined_csv']}")


if __name__ == "__main__":
    main()
