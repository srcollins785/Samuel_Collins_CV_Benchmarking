"""Benchmark YOLO classification against the identical split.

Three stages, because YOLO cannot run in the main environment (see
requirements-yolo.txt) and its metrics must nonetheless be computed the same
way as every other architecture's.

1. *Export* (this environment). Rebuild the Part 1 split, verify it against
   Part 1's published artifacts, and materialize the three halves as a
   directory tree of symlinks - which is the only dataset format ultralytics
   classification accepts. Symlinks rather than copies: the tree is 5,000
   entries and copying would duplicate a gigabyte for no reason. Each link is
   named with its dataset index, so a prediction can always be traced back to
   the exact row of the split that produced it.

2. *Train* (the YOLO environment). A subprocess runs scripts/_yolo_train.py,
   which trains on train/, validates on val/, and writes one predicted class
   name per test image.

3. *Score* (this environment). Map the predicted names onto this package's
   label encoding, in the order of ``deep_split.test_index``, and score them
   with ``deep_benchmark.score_predictions`` - the same function that scores
   AlexNet and ConvNeXt. The result is merged into deep_metrics.json as
   ``yolo_cls`` and the combined table is rebuilt.

The deviation to keep in view: ultralytics brings its own augmentation
defaults, its own learning-rate schedule and its own library versions. Epochs,
batch size, image size, optimizer, initial learning rate and seed are aligned
with section 6; the rest is not alignable without reimplementing its trainer,
so it is recorded in the results instead of being smoothed over.

    python scripts/run_yolo.py
    python scripts/run_yolo.py --epochs 20 --config animals10_n500_rgb
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from samuel_collins_cv_benchmarking import (  # noqa: E402
    combined,
    deep_benchmark,
    deep_data,
    deep_visualization,
    visualization,
)
from samuel_collins_cv_benchmarking._config import (  # noqa: E402
    CNN_BATCH_SIZE,
    CNN_EPOCHS,
    CNN_IMAGE_SIZE,
    CNN_LEARNING_RATE,
    PLOTS_DIR,
    RANDOM_SEED,
    YOLO_MODEL,
)
from samuel_collins_cv_benchmarking.evaluation import (  # noqa: E402
    classification_report_frame,
)

YOLO_PYTHON = REPO_ROOT / ".venv-yolo" / "bin" / "python"
DEFAULT_CONFIG = "animals10_n500_rgb"


def export_tree(split, deep_split, destination: Path) -> dict:
    """Write the three halves as a symlink tree ultralytics can read.

    Link names carry the dataset index so a prediction is traceable to the row
    it came from, and so two images sharing a filename in different source
    folders cannot collide.
    """
    import numpy as np

    sources = list(split.dataset.sources)
    class_names = deep_split.class_names
    labels = split.dataset.labels

    if destination.exists():
        import shutil

        shutil.rmtree(destination)

    manifest = {"train": {}, "val": {}, "test": {}}
    halves = (
        ("train", deep_split.train_index),
        ("val", deep_split.validation_index),
        ("test", deep_split.test_index),
    )

    for half, index in halves:
        for dataset_index in index:
            dataset_index = int(dataset_index)
            source = Path(sources[dataset_index]).resolve()
            class_name = class_names[int(labels[dataset_index])]
            folder = destination / half / class_name
            folder.mkdir(parents=True, exist_ok=True)
            link = folder / f"{dataset_index:06d}_{source.name}"
            link.symlink_to(source)
            manifest[half][link.name] = {
                "dataset_index": dataset_index,
                "class_name": class_name,
            }

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--epochs", type=int, default=CNN_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=CNN_LEARNING_RATE)
    parser.add_argument("--model", default=YOLO_MODEL)
    parser.add_argument("--keep-tree", action="store_true",
                        help="keep the exported symlink tree for inspection")
    arguments = parser.parse_args()

    if not YOLO_PYTHON.is_file():
        raise SystemExit(
            f"No YOLO environment at {YOLO_PYTHON.relative_to(REPO_ROOT)}. "
            "Create it with a Python 3.11+ interpreter:\n"
            "  python3 -m venv .venv-yolo\n"
            "  .venv-yolo/bin/pip install -r requirements-yolo.txt"
        )

    sys.path.insert(0, str(REPO_ROOT))
    from run_benchmark import parse_config, rebuild_split, verify_against_part1

    tier, color_mode = parse_config(arguments.config)
    config_dir = REPO_ROOT / "benchmark_results" / arguments.config

    print(f"configuration: {arguments.config}")
    split = rebuild_split(tier, color_mode)
    verified = verify_against_part1(split, config_dir)
    print(f"  split verified against Part 1: "
          f"{verified['test_positions_checked']} test positions, "
          f"validation rows match the Simple CNN: "
          f"{verified['validation_rows_match_simple_cnn']}")

    deep_split = deep_data.make_deep_split(split)
    class_names = deep_split.class_names

    workspace = REPO_ROOT / "yolo_workspace"
    tree = workspace / "dataset"
    workspace.mkdir(exist_ok=True)

    print(f"exporting the split as a symlink tree at "
          f"{tree.relative_to(REPO_ROOT)}")
    manifest = export_tree(split, deep_split, tree)
    print(f"  {len(manifest['train'])} train / {len(manifest['val'])} val / "
          f"{len(manifest['test'])} test symlinks")

    predictions_path = workspace / "predictions.json"
    command = [
        str(YOLO_PYTHON), str(REPO_ROOT / "scripts" / "_yolo_train.py"),
        "--data", str(tree),
        "--output", str(predictions_path),
        "--model", arguments.model,
        "--epochs", str(arguments.epochs),
        "--batch", str(CNN_BATCH_SIZE),
        "--imgsz", str(CNN_IMAGE_SIZE[0]),
        "--lr0", str(arguments.learning_rate),
        "--seed", str(RANDOM_SEED),
        "--project", str(workspace / "runs"),
    ]
    print(f"\ntraining in the YOLO environment ({arguments.epochs} epochs)")
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=workspace)
    if completed.returncode != 0:
        raise SystemExit(
            f"The YOLO subprocess failed with exit code "
            f"{completed.returncode}. Its output is above."
        )
    print(f"  subprocess finished in {(time.perf_counter() - started) / 60:.1f} min")

    payload = json.loads(predictions_path.read_text(encoding="utf-8"))
    merge_results(payload, manifest, deep_split, class_names, config_dir,
                  arguments)
    import_yolo_history(workspace / "runs" / "train", config_dir)

    print("\nrebuilding the combined comparison")
    built = combined.build_all(config_dir)
    deep_visualization.plot_all_comparisons(built["rows"], config_dir / PLOTS_DIR)
    print(f"  {built['combined_csv']}")

    if not arguments.keep_tree:
        import shutil

        shutil.rmtree(tree, ignore_errors=True)
        print(f"  removed the symlink tree (use --keep-tree to retain it)")


def import_yolo_history(run_dir, config_dir) -> None:
    """Bring ultralytics' per-epoch log in, so section 16 covers YOLO too.

    Its trainer owns its loop and writes its own results.csv rather than
    reporting through this project's callback, so the history is imported
    afterward instead of recorded live. It logs training loss, validation
    loss, validation top-1 accuracy, elapsed time and learning rate - every
    field section 11 asks for except training accuracy, which it does not
    compute for classification. That field is left absent rather than filled
    with a substitute, and the curve omits the series instead of drawing it
    as zero.
    """
    import csv

    from samuel_collins_cv_benchmarking import deep_visualization
    from samuel_collins_cv_benchmarking._config import LOGS_DIR, PLOTS_DIR

    results_csv = Path(run_dir) / "results.csv"
    if not results_csv.is_file():
        print(f"  no results.csv at {results_csv}; skipping YOLO curves")
        return

    records, previous = [], 0.0
    with open(results_csv, encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            elapsed = float(row.get("time", 0) or 0)
            records.append({
                "epoch": int(float(row["epoch"])),
                "train_loss": float(row["train/loss"]),
                "validation_loss": float(row["val/loss"]),
                # Not recorded by ultralytics for classification.
                "train_accuracy": None,
                "validation_accuracy": float(row["metrics/accuracy_top1"]),
                "epoch_seconds": round(elapsed - previous, 3),
                "learning_rate": float(row.get("lr/pg0", 0) or 0),
            })
            previous = elapsed

    if not records:
        return

    metrics_path = Path(config_dir) / "deep_metrics.json"
    stored = json.loads(metrics_path.read_text(encoding="utf-8"))
    entry = stored.get("yolo_cls")
    if entry is None:
        return

    best = max(records, key=lambda r: r["validation_accuracy"])
    history = entry.setdefault("training_history", {})
    history.update({
        "per_epoch": records,
        "epochs_run": len(records),
        "best_epoch": best["epoch"],
        "best_validation_accuracy": best["validation_accuracy"],
        "mean_epoch_seconds": round(
            sum(r["epoch_seconds"] for r in records) / len(records), 3),
        "train_accuracy_recorded": False,
        "note": "Per-epoch values imported from ultralytics' own results.csv; "
                "its trainer owns the loop. Training accuracy is not among "
                "the fields it logs for classification.",
    })

    final = records[-1]
    losses = [r["validation_loss"] for r in records]
    entry["overfitting"] = {
        "generalization_gap": None,
        "final_train_accuracy": None,
        "final_validation_accuracy": round(final["validation_accuracy"], 4),
        "epochs_after_best": len(records) - best["epoch"],
        "validation_loss_rise_from_minimum": round(
            final["validation_loss"] - min(losses), 4),
        "minimum_validation_loss": round(min(losses), 4),
        "minimum_validation_loss_epoch": losses.index(min(losses)) + 1,
        "note": "Generalization gap needs a training accuracy, which "
                "ultralytics does not log for classification.",
    }
    metrics_path.write_text(json.dumps(stored, indent=2), encoding="utf-8")

    logs_dir = Path(config_dir) / LOGS_DIR
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "yolo_cls.csv").write_text(
        results_csv.read_text(encoding="utf-8"), encoding="utf-8")

    class Shim:
        key = "yolo_cls"
        name = "YOLO Classification"
        succeeded = True

    shim = Shim()
    shim.history = history
    shim.overfitting = entry["overfitting"]
    deep_visualization.plot_training_curves(
        shim, Path(config_dir) / PLOTS_DIR / "curves_yolo_cls.png")
    print(f"  imported {len(records)} epochs of YOLO history; "
          f"best epoch {best['epoch']} at {best['validation_accuracy']:.4f}")


def merge_results(payload, manifest, deep_split, class_names, config_dir,
                  arguments) -> None:
    """Score YOLO's predictions with the shared scorer and merge them in."""
    import numpy as np

    encoding = {name: index for index, name in enumerate(class_names)}
    by_index = {entry["dataset_index"]: name
                for name, entry in manifest["test"].items()}

    missing = []
    predicted = []
    for dataset_index in deep_split.test_index:
        link_name = by_index.get(int(dataset_index))
        name = payload["predictions"].get(link_name) if link_name else None
        if name is None:
            missing.append(int(dataset_index))
            predicted.append(-1)
        else:
            predicted.append(encoding.get(name, -1))

    if missing:
        raise SystemExit(
            f"YOLO returned no prediction for {len(missing)} test image(s), "
            f"for example dataset index {missing[0]}. The comparison needs a "
            "prediction for every test image in the shared split."
        )

    unknown = sorted(set(payload["class_names"]) - set(class_names))
    if unknown:
        raise SystemExit(
            f"YOLO predicted class name(s) this dataset does not have: "
            f"{unknown}. Its label set has diverged from the split's."
        )

    predictions = np.asarray(predicted, dtype=np.int64)
    truth = deep_split.labels_test
    scored = deep_benchmark.score_predictions(truth, predictions, class_names)

    images = payload["images"]
    inference_seconds = payload["inference_seconds"]
    checkpoint_mb = (round(payload["checkpoint_bytes"] / 1e6, 2)
                     if payload.get("checkpoint_bytes") else None)

    entry = {
        "name": "YOLO Classification",
        "family": "Modern Classifier",
        "year": 2024,
        "kind": "deep_cnn",
        "succeeded": True,
        "error": None,
        "accuracy": scored["accuracy"],
        "macro_precision": scored["macro_precision"],
        "macro_recall": scored["macro_recall"],
        "macro_f1": scored["macro_f1"],
        "weighted_precision": scored["weighted_precision"],
        "weighted_recall": scored["weighted_recall"],
        "weighted_f1": scored["weighted_f1"],
        "training_time_seconds": payload["training_seconds"],
        "inference_time_ms_per_image": round(
            inference_seconds * 1000 / max(1, images), 4),
        "parameter_counts": {
            "total_parameters": payload["total_parameters"],
            "trainable_parameters": payload["trainable_parameters"],
            "total_parameters_millions": round(
                payload["total_parameters"] / 1e6, 3),
            "frozen_parameters": (payload["total_parameters"]
                                  - payload["trainable_parameters"]),
        },
        # ultralytics reports its own FLOPs figure during training but not in a
        # form this script consumes, and section 22 says to report N/A rather
        # than invent one.
        "complexity": {
            "macs": None, "gmacs": None, "flops_estimate": None,
            "gflops_estimate": None, "profiler": None,
            "unsupported_operators": [],
            "note": "Not measured. YOLO trains in a separate environment and "
                    "its model is not passed through this project's profiler.",
        },
        "inference": {
            "images": images,
            "meets_minimum_images": images >= 1000,
            "total_seconds": round(inference_seconds, 4),
            "latency_ms_per_image": round(
                inference_seconds * 1000 / max(1, images), 4),
            "throughput_images_per_second": round(images / inference_seconds, 2)
            if inference_seconds else None,
            "batch_size": CNN_BATCH_SIZE,
            "device": payload["device"],
            "peak_memory_mb": None,
            "memory_measurement": "not_measured_separate_environment",
        },
        "single_image_inference": {},
        "checkpoint_megabytes": checkpoint_mb,
        "confusion_matrix": scored["confusion_matrix"].tolist(),
        "classification_report": scored["classification_report"],
        "per_class_accuracy": scored["per_class_accuracy"],
        "training_history": {
            "total_training_seconds": payload["training_seconds"],
            "epochs_run": arguments.epochs,
            "epochs_requested": arguments.epochs,
            "checkpoint_megabytes": checkpoint_mb,
            "device": payload["device"],
            "selection_criterion": "ultralytics internal best.pt selection",
            "note": "Per-epoch curves are written by ultralytics to "
                    "yolo_workspace/runs/train/results.csv rather than "
                    "recorded here; its trainer owns the loop.",
        },
        "overfitting": {},
        "parameters": {
            "architecture": "yolo_cls",
            "pretrained": True,
            "pretrained_weights": payload["protocol"]["model"],
            "num_classes": len(class_names),
            "input_size": list(CNN_IMAGE_SIZE),
            **payload["protocol"],
            "environment": payload["environment"],
        },
        "architecture_metadata": {
            "key": "yolo_cls",
            "name": "YOLO Classification",
            "family": "Modern Classifier",
            "year": 2024,
            "ideas": [
                "a detection-family backbone run in classification mode",
                "anchor-free, single-pass design inherited from detection",
                "aggressive built-in augmentation and its own training recipe",
                "engineered for deployment throughput rather than benchmark "
                "accuracy alone",
            ],
            "addressed": "Tests whether a model family designed for real-time "
                         "detection is competitive at plain image "
                         "classification, which is what an edge deployment "
                         "would actually have to choose between.",
        },
        "deviation": {
            "environment": payload["environment"],
            "note": "Trained in a separate virtual environment on different "
                    "library versions, using ultralytics' own training loop "
                    "and augmentation defaults. Epochs, batch size, image "
                    "size, optimizer, initial learning rate and seed match "
                    "section 6; the remainder of its recipe is not alignable "
                    "without reimplementing its trainer. The split, the test "
                    "half and the scoring code are identical to every other "
                    "architecture's.",
        },
    }

    metrics_path = config_dir / "deep_metrics.json"
    stored = (json.loads(metrics_path.read_text(encoding="utf-8"))
              if metrics_path.is_file() else {})
    stored["yolo_cls"] = entry
    metrics_path.write_text(json.dumps(stored, indent=2), encoding="utf-8")

    class Shim:
        """Just enough of a result for the shared plotting and report code."""

        key = "yolo_cls"
        name = "YOLO Classification"
        succeeded = True
        confusion_matrix = scored["confusion_matrix"]
        classification_report = scored["classification_report"]

    visualization.plot_confusion_matrix(
        Shim(), class_names,
        config_dir / "confusion_matrices" / "yolo_cls.png")
    classification_report_frame(Shim()).to_csv(
        config_dir / "classification_reports" / "yolo_cls.csv", index=False)

    print(f"\nYOLO Classification: accuracy {scored['accuracy']:.4f}  "
          f"macro F1 {scored['macro_f1']:.4f}  "
          f"{payload['training_seconds']:.0f}s training")


if __name__ == "__main__":
    main()
