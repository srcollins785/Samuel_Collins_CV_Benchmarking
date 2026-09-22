"""The combined Part 1 + Part 2 comparison, section 23's master table.

Section 1A is emphatic that the CNN results must never be reported in
isolation: every final table and every major result discussion has to carry
the traditional ML baselines alongside. This module is what makes that cheap -
it reads both halves of the benchmark back from the artifacts on disk and
emits one table, one set of rankings and one per-class comparison.

Reading from disk rather than from objects in memory is deliberate, and it is
the same choice the Part 1 report generator made. It means the combined table
can be rebuilt in a second without retraining anything, and it means the
traditional ML numbers in the master table are literally the numbers Part 1
published rather than a fresh run that might differ.

One derivation is worth flagging. Section 13 asks for weighted precision and
weighted recall, which Part 1 recorded neither of - it stored macro precision,
macro recall, macro F1 and weighted F1. Those two missing averages are exactly
recoverable from the confusion matrix Part 1 *did* store, because a confusion
matrix determines every per-class and averaged classification metric there is.
So they are recomputed rather than left blank, and
:func:`metrics_from_confusion_matrix` is checked against Part 1's own stored
macro F1 to prove the derivation reproduces what Part 1 measured before any of
its other outputs are trusted.
"""

import json
from pathlib import Path

import numpy as np

from ._config import COMBINED_RESULTS_CSV
from .deep_visualization import group_for

# Section 23's Family column for the Part 1 models. The deep architectures
# carry their own family in deep_models.ARCHITECTURES.
PART1_FAMILIES = {
    "logistic_regression": "Traditional ML",
    "decision_tree": "Traditional ML",
    "random_forest": "Traditional ML",
    "svm": "Traditional ML",
    "neural_network": "Neural Baseline",
    "simple_cnn": "CNN Baseline",
}

# The order section 23's table uses: prior methods first, then the new
# architectures roughly chronologically.
ROW_ORDER = (
    "logistic_regression", "decision_tree", "random_forest", "svm",
    "neural_network", "simple_cnn",
    "alexnet", "vgg16", "googlenet", "resnet18", "resnet50", "densenet121",
    "mobilenet_v3_large", "efficientnet_b0", "convnext_tiny", "yolo_cls",
)


def metrics_from_confusion_matrix(matrix) -> dict:
    """Every averaged classification metric, from the matrix alone.

    A confusion matrix fully determines accuracy and per-class precision,
    recall and F1, hence every macro and weighted average of them. Computed
    directly rather than by re-running a model, with the same zero-division
    convention Part 1 used: a class the model never predicted scores zero
    precision rather than being skipped, which is what stops a model that
    ignores a class from being rewarded for it.
    """
    matrix = np.asarray(matrix, dtype=float)
    if matrix.size == 0:
        return {}

    true_positive = np.diag(matrix)
    predicted = matrix.sum(axis=0)
    actual = matrix.sum(axis=1)
    total = matrix.sum()

    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.divide(true_positive, predicted,
                              out=np.zeros_like(true_positive), where=predicted > 0)
        recall = np.divide(true_positive, actual,
                           out=np.zeros_like(true_positive), where=actual > 0)
        denominator = precision + recall
        f1 = np.divide(2 * precision * recall, denominator,
                       out=np.zeros_like(precision), where=denominator > 0)

    # Weighted averages use the true class support, matching scikit-learn.
    weights = actual / total if total else np.zeros_like(actual)

    return {
        "accuracy": float(true_positive.sum() / total) if total else 0.0,
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "weighted_precision": float((precision * weights).sum()),
        "weighted_recall": float((recall * weights).sum()),
        "weighted_f1": float((f1 * weights).sum()),
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
    }


def _row(key, name, family, metrics, **extra) -> dict:
    row = {
        "key": key,
        "name": name,
        "family": family,
        "group": group_for(key),
        "accuracy": metrics.get("accuracy"),
        "macro_precision": metrics.get("macro_precision"),
        "macro_recall": metrics.get("macro_recall"),
        "macro_f1": metrics.get("macro_f1"),
        "weighted_precision": metrics.get("weighted_precision"),
        "weighted_recall": metrics.get("weighted_recall"),
        "weighted_f1": metrics.get("weighted_f1"),
        "total_parameters": None,
        "trainable_parameters": None,
        "checkpoint_mb": None,
        "training_seconds": None,
        "latency_ms": None,
        "throughput": None,
        "memory_mb": None,
        "memory_measurement": None,
        "gmacs": None,
        "gflops_estimate": None,
        "best_epoch": None,
        "generalization_gap": None,
        "status": "ok",
    }
    row.update(extra)
    return row


def load_baseline_sizes(config_dir) -> dict:
    """Parameter counts and sizes for the two parameterized baselines.

    Part 1 recorded neither, and the first version of the combined table
    reported all six of its models as N/A. That is right for Logistic
    Regression, a Decision Tree, a Random Forest and an SVM, none of which has
    a parameter count in the sense a network does. It is wrong for the fully
    connected network and the Simple CNN, which are parameterized networks and
    which section 23's own template leaves blank rather than marking N/A.

    Measured separately by scripts/measure_baselines.py and read from its
    artifact, so Part 1's published metrics file stays untouched.
    """
    path = Path(config_dir) / "baseline_model_sizes.json"
    if not path.is_file():
        return {}
    return (json.loads(path.read_text(encoding="utf-8")) or {}).get("models", {})


def load_part1_rows(config_dir) -> list:
    """The six prior methods, read back from Part 1's artifacts.

    Raises
    ------
    ValueError
        The derived metrics disagree with Part 1's stored ones, which would
        mean this module is misreading the artifact rather than reproducing
        it. Better to stop than to publish a table built on a misreading.
    """
    path = Path(config_dir) / "benchmark_metrics.json"
    if not path.is_file():
        return []

    stored = json.loads(path.read_text(encoding="utf-8"))
    sizes = load_baseline_sizes(config_dir)
    rows = []

    for key, entry in stored.items():
        name = entry.get("name", key)
        family = PART1_FAMILIES.get(key, "Traditional ML")
        measured = sizes.get(key, {})

        if not entry.get("succeeded", False):
            rows.append(_row(key, name, family, {},
                             status=entry.get("error") or "failed"))
            continue

        matrix = entry.get("confusion_matrix")
        derived = metrics_from_confusion_matrix(matrix) if matrix else {}

        # The derivation has to reproduce what Part 1 measured, or it is not a
        # derivation. Checked on macro F1 and accuracy, which Part 1 stored.
        for metric in ("accuracy", "macro_f1", "macro_precision", "macro_recall",
                       "weighted_f1"):
            recorded = entry.get(metric)
            if recorded is not None and derived.get(metric) is not None:
                if abs(recorded - derived[metric]) > 1e-6:
                    raise ValueError(
                        f"Derived {metric} for {key} is {derived[metric]:.8f} but "
                        f"Part 1 recorded {recorded:.8f}. The confusion matrix in "
                        f"{path} is being misread; the combined table would be "
                        "wrong."
                    )

        latency = entry.get("inference_time_ms_per_image")
        rows.append(_row(
            key, name, family, derived,
            # Present for the two parameterized baselines, absent for the four
            # traditional models, where N/A is the correct answer.
            total_parameters=measured.get("total_parameters"),
            trainable_parameters=measured.get("trainable_parameters"),
            checkpoint_mb=measured.get("size_megabytes"),
            training_seconds=entry.get("training_time_seconds"),
            latency_ms=latency,
            # Exact arithmetic on a measured latency, not a separate
            # measurement - Part 1 timed latency, so throughput is derived.
            throughput=round(1000 / latency, 2) if latency else None,
            best_epoch=(entry.get("training_history") or {}).get("epochs_run"),
        ))

    return rows


def load_part2_rows(config_dir) -> list:
    """The deep architectures, read back from Part 2's artifacts."""
    path = Path(config_dir) / "deep_metrics.json"
    if not path.is_file():
        return []

    stored = json.loads(path.read_text(encoding="utf-8"))
    rows = []

    for key, entry in stored.items():
        name = entry.get("name", key)
        family = entry.get("family", "Deep CNN")

        if not entry.get("succeeded", False):
            rows.append(_row(key, name, family, {},
                             status=entry.get("error") or "failed"))
            continue

        counts = entry.get("parameter_counts") or {}
        complexity = entry.get("complexity") or {}
        inference = entry.get("inference") or {}
        history = entry.get("training_history") or {}
        overfitting = entry.get("overfitting") or {}

        rows.append(_row(
            key, name, family, entry,
            total_parameters=counts.get("total_parameters"),
            trainable_parameters=counts.get("trainable_parameters"),
            checkpoint_mb=entry.get("checkpoint_megabytes"),
            training_seconds=history.get("total_training_seconds"),
            latency_ms=inference.get("latency_ms_per_image"),
            throughput=inference.get("throughput_images_per_second"),
            memory_mb=inference.get("peak_memory_mb"),
            memory_measurement=inference.get("memory_measurement"),
            gmacs=complexity.get("gmacs"),
            gflops_estimate=complexity.get("gflops_estimate"),
            best_epoch=history.get("best_epoch"),
            generalization_gap=overfitting.get("generalization_gap"),
        ))

    return rows


def combined_rows(config_dir) -> list:
    """Both halves of the benchmark, in section 23's row order."""
    rows = load_part1_rows(config_dir) + load_part2_rows(config_dir)
    position = {key: index for index, key in enumerate(ROW_ORDER)}
    return sorted(rows, key=lambda row: position.get(row["key"], 999))


def _display(value, digits=4, missing="N/A"):
    """Format a cell, writing N/A where a metric does not apply."""
    if value is None:
        return missing
    if isinstance(value, float):
        return round(value, digits)
    return value


def write_combined_csv(rows, path) -> Path:
    """Section 23's ``combined_ml_cnn_benchmark_results.csv``.

    N/A rather than a blank or a zero wherever a metric is not meaningful for
    a model, which section 23 asks for explicitly. A Random Forest has no
    parameter count in the sense a CNN does, no checkpoint on disk and no
    device memory figure, and writing 0 in those cells would put three
    fabricated measurements into the headline table.
    """
    import pandas as pd

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    frame = pd.DataFrame([{
        "Model": row["name"],
        "Family": row["family"],
        "Accuracy": _display(row["accuracy"]),
        "Macro Precision": _display(row["macro_precision"]),
        "Macro Recall": _display(row["macro_recall"]),
        "Macro F1": _display(row["macro_f1"]),
        "Weighted Precision": _display(row["weighted_precision"]),
        "Weighted Recall": _display(row["weighted_recall"]),
        "Weighted F1": _display(row["weighted_f1"]),
        "Total Parameters": _display(row["total_parameters"]),
        "Trainable Parameters": _display(row["trainable_parameters"]),
        "Size MB": _display(row["checkpoint_mb"], 2),
        "GMACs": _display(row["gmacs"], 3),
        "Train Time (s)": _display(row["training_seconds"], 1),
        "Latency (ms/img)": _display(row["latency_ms"], 3),
        "Throughput (img/s)": _display(row["throughput"], 1),
        "Memory MB": _display(row["memory_mb"], 1),
        "Best Epoch": _display(row["best_epoch"], 0),
        "Status": row["status"],
    } for row in rows])

    frame.to_csv(path, index=False)
    return path


# -- the required final rankings --------------------------------------------

def _ranked(rows, key, reverse=True, label=None):
    """Rows that reported a metric, ordered by it."""
    present = [row for row in rows if row.get(key) is not None
               and row["status"] == "ok"]
    present.sort(key=lambda row: row[key], reverse=reverse)
    return [{
        "rank": index,
        "model": row["name"],
        "family": row["family"],
        "value": row[key],
        "metric": label or key,
    } for index, row in enumerate(present, start=1)]


# Cost metrics that span orders of magnitude are normalized on a log scale.
#
# Straight min-max on these produces a term that cannot discriminate. Latency
# in this benchmark runs from 0.002 ms/image for a Decision Tree to 89.96 for
# an RBF SVM on 12,288 features - four orders of magnitude - so the SVM alone
# defines the range and every other model lands within a percent or two of the
# best possible score. The latency term then contributes the same amount to
# everyone and the stated 25% weight does no work, which makes the published
# weighting a misdescription of how the ranking was actually produced.
#
# Log scaling is the same reasoning that puts these axes on a log scale in the
# plots: a model twice as fast as another is the interesting comparison, not
# one that is 0.001 ms closer to the fastest. Accuracy and F1 stay linear -
# they are bounded on [0, 1] and already commensurable.
LOG_SCALED_METRICS = {
    "latency_ms", "throughput", "checkpoint_mb", "memory_mb",
    "total_parameters", "training_seconds",
}


def _normalize(values, higher_is_better=True, log=False):
    """Min-max to 0-1, so metrics on different scales can be combined.

    A flat set maps to 1.0 rather than dividing by zero: if every model has
    the same value for a metric, that metric cannot discriminate between them
    and should not drag the whole score toward zero.

    ``log`` normalizes the base-10 logarithm instead of the raw value, for the
    orders-of-magnitude cost metrics listed in ``LOG_SCALED_METRICS``. Values
    at or below zero cannot be log scaled, so a metric containing any is
    normalized linearly and the fallback is silent by design - it affects the
    spacing of a score, never whether a value is included.
    """
    array = np.asarray(values, dtype=float)
    if log and np.all(array > 0):
        array = np.log10(array)
    low, high = array.min(), array.max()
    if high - low < 1e-12:
        return np.ones_like(array)
    scaled = (array - low) / (high - low)
    return scaled if higher_is_better else 1.0 - scaled


# Ranking E, part one: the three metrics every model in the benchmark
# reports, so all sixteen can be scored on the same basis.
UNIVERSAL_WEIGHTS = {"accuracy": 0.40, "macro_f1": 0.35, "latency_ms": 0.25}

# Ranking E, part two: the full deployment score, for the models that report
# size and memory. Weighted for an embedded or UAV target.
DEPLOYMENT_WEIGHTS = {
    "accuracy": 0.30,
    "macro_f1": 0.25,
    "throughput": 0.20,
    "checkpoint_mb": 0.15,
    "memory_mb": 0.10,
}

COST_METRICS = {"latency_ms", "checkpoint_mb", "memory_mb", "training_seconds"}


def weighted_score(rows, weights: dict) -> list:
    """Score and rank rows on a weighted blend of normalized metrics.

    Only rows reporting every metric in ``weights`` are scored - a partial
    score renormalized over fewer metrics is not comparable with a full one,
    and silently mixing the two would produce a ranking whose order depends
    on which measurements happened to be possible.
    """
    metrics = list(weights)
    eligible = [row for row in rows
                if row["status"] == "ok"
                and all(row.get(metric) is not None for metric in metrics)]
    if not eligible:
        return []

    normalized = {
        metric: _normalize([row[metric] for row in eligible],
                           higher_is_better=metric not in COST_METRICS,
                           log=metric in LOG_SCALED_METRICS)
        for metric in metrics
    }

    scored = []
    for position, row in enumerate(eligible):
        contributions = {
            metric: round(float(normalized[metric][position]) * weight, 4)
            for metric, weight in weights.items()
        }
        scored.append({
            "model": row["name"],
            "family": row["family"],
            "score": round(sum(contributions.values()), 4),
            "contributions": contributions,
        })

    scored.sort(key=lambda entry: entry["score"], reverse=True)
    for index, entry in enumerate(scored, start=1):
        entry["rank"] = index
    return scored


def accuracy_per_million_parameters(rows) -> list:
    """Ranking D. A comparative indicator, not a quality metric.

    The assignment says as much, and it is worth repeating wherever this
    appears: the ratio rewards small models so strongly that a tiny network
    with mediocre accuracy outranks an excellent large one. It answers "what
    does each million parameters buy here", which is a real question for
    embedded work, and nothing more than that.
    """
    present = [row for row in rows
               if row.get("accuracy") is not None
               and row.get("total_parameters") and row["status"] == "ok"]
    for row in present:
        row["accuracy_per_million"] = round(
            row["accuracy"] / (row["total_parameters"] / 1e6), 4)
    present.sort(key=lambda row: row["accuracy_per_million"], reverse=True)
    return [{
        "rank": index,
        "model": row["name"],
        "family": row["family"],
        "accuracy": round(row["accuracy"], 4),
        "parameters_millions": round(row["total_parameters"] / 1e6, 3),
        "accuracy_per_million_parameters": row["accuracy_per_million"],
    } for index, row in enumerate(present, start=1)]


def all_rankings(rows) -> dict:
    """Rankings A through E, with their methodology stated alongside."""
    return {
        "A_highest_accuracy": {
            "basis": "Test accuracy on the shared test half, highest first.",
            "ranking": _ranked(rows, "accuracy", True, "accuracy"),
        },
        "B_fastest_inference": {
            "basis": "Batched throughput in images per second, highest first. "
                     "Traditional ML throughput is derived from the latency "
                     "Part 1 measured.",
            "ranking": _ranked(rows, "throughput", True, "images_per_second"),
        },
        "C_smallest_model": {
            "basis": "Saved checkpoint size in MB, smallest first. Only "
                     "models that write a checkpoint appear.",
            "ranking": _ranked(rows, "checkpoint_mb", False, "megabytes"),
        },
        "D_accuracy_per_million_parameters": {
            "basis": "Accuracy divided by parameters in millions. A "
                     "comparative indicator only - it rewards small models "
                     "heavily and is not a measure of model quality.",
            "ranking": accuracy_per_million_parameters(rows),
        },
        "E_overall_recommendation": {
            "universal": {
                "basis": (
                    "Every model in the benchmark scored on the three metrics "
                    "all of them report. Accuracy 40% and macro F1 35% because "
                    "the task is classification and being right dominates; "
                    "macro F1 is weighted separately from accuracy because it "
                    "is the one that catches a model quietly abandoning a "
                    "class. Latency 25% because a classifier too slow for its "
                    "deployment is not a usable classifier. Metrics are "
                    "min-max normalized across the models scored, so the "
                    "numbers are relative to this benchmark and not absolute."
                ),
                "weights": UNIVERSAL_WEIGHTS,
                "ranking": weighted_score(rows, UNIVERSAL_WEIGHTS),
            },
            "deployment": {
                "basis": (
                    "The edge-deployment score, for models that report size "
                    "and memory. Accuracy 30% and macro F1 25% keep "
                    "correctness dominant at 55%. Throughput 20% and "
                    "checkpoint size 15% are the two constraints that "
                    "actually stop a model shipping to a UAV or an embedded "
                    "board - flash budget and frame rate. Memory is weighted "
                    "lowest at 10% because on this host it is a sampled MPS "
                    "allocation rather than a true peak, and a weight should "
                    "not exceed the confidence in its measurement. "
                    "Throughput, checkpoint size and memory are normalized on "
                    "a log scale for the same reason latency is in the "
                    "universal score; accuracy and macro F1 are normalized "
                    "directly."
                ),
                "weights": DEPLOYMENT_WEIGHTS,
                "ranking": weighted_score(rows, DEPLOYMENT_WEIGHTS),
            },
        },
    }


# -- section 15: the per-class comparison ----------------------------------

def per_class_table(config_dir, class_names=None) -> dict:
    """Per-class F1 for every model, plus the classes everything finds hard.

    Section 15 asks which classes stay difficult across both traditional ML
    and the CNNs, so the interesting output is not the table itself but the
    ranking underneath it: mean F1 per class across model families, which
    says whether depth actually fixed the errors shallow models made or
    merely shrank them.
    """
    config_dir = Path(config_dir)
    per_model = {}

    for filename, loader in (("benchmark_metrics.json", PART1_FAMILIES),
                             ("deep_metrics.json", None)):
        path = config_dir / filename
        if not path.is_file():
            continue
        stored = json.loads(path.read_text(encoding="utf-8"))
        for key, entry in stored.items():
            if not entry.get("succeeded"):
                continue
            report = entry.get("classification_report") or {}
            scores = {
                name: values.get("f1-score")
                for name, values in report.items()
                if isinstance(values, dict) and "f1-score" in values
                and name not in ("accuracy", "macro avg", "weighted avg")
            }
            if scores:
                per_model[key] = {
                    "name": entry.get("name", key),
                    "group": group_for(key),
                    "per_class_f1": scores,
                }

    if not per_model:
        return {"models": {}, "difficulty": []}

    names = class_names or sorted(
        next(iter(per_model.values()))["per_class_f1"].keys())

    difficulty = []
    for class_name in names:
        by_group = {}
        for entry in per_model.values():
            score = entry["per_class_f1"].get(class_name)
            if score is not None:
                by_group.setdefault(entry["group"], []).append(score)
        overall = [score for scores in by_group.values() for score in scores]
        if not overall:
            continue
        difficulty.append({
            "class": class_name,
            "mean_f1_all_models": round(float(np.mean(overall)), 4),
            "mean_f1_by_group": {
                group: round(float(np.mean(scores)), 4)
                for group, scores in sorted(by_group.items())
            },
            "worst_f1": round(float(np.min(overall)), 4),
            "best_f1": round(float(np.max(overall)), 4),
        })

    difficulty.sort(key=lambda entry: entry["mean_f1_all_models"])
    return {"models": per_model, "class_names": list(names),
            "difficulty": difficulty}


def write_per_class_csv(config_dir, path, class_names=None) -> Path:
    """The section 15 table: one row per class, one column per model."""
    import pandas as pd

    table = per_class_table(config_dir, class_names)
    if not table["models"]:
        return None

    position = {key: index for index, key in enumerate(ROW_ORDER)}
    ordered = sorted(table["models"].items(),
                     key=lambda item: position.get(item[0], 999))

    rows = []
    for class_name in table["class_names"]:
        row = {"Class": class_name}
        for key, entry in ordered:
            score = entry["per_class_f1"].get(class_name)
            row[entry["name"]] = round(score, 4) if score is not None else "N/A"
        rows.append(row)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def build_all(config_dir, output_dir=None) -> dict:
    """Write the master table, the rankings and the per-class comparison."""
    config_dir = Path(config_dir)
    output_dir = Path(output_dir or config_dir)
    rows = combined_rows(config_dir)

    combined_csv = write_combined_csv(rows, output_dir / COMBINED_RESULTS_CSV)
    per_class_csv = write_per_class_csv(
        config_dir, output_dir / "per_class_f1_comparison.csv")
    rankings = all_rankings(rows)
    (output_dir / "rankings.json").write_text(
        json.dumps(rankings, indent=2), encoding="utf-8")

    return {
        "rows": rows,
        "rankings": rankings,
        "combined_csv": str(combined_csv),
        "per_class_csv": str(per_class_csv) if per_class_csv else None,
        "rankings_json": str(output_dir / "rankings.json"),
    }
