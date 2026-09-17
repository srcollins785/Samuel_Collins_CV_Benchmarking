"""Orchestration for the deep CNN benchmark.

Drives each architecture through the same sequence - build, fine-tune on the
training half, select on validation accuracy, score once on the test half,
then measure what it costs - and writes the artifacts the report reads.

Two inherited rules from Part 1 are kept deliberately.

*Failures are results.* An architecture that raises keeps its row in the
comparison carrying its error message, rather than vanishing. A benchmark that
silently compared eight architectures while claiming to compare nine would be
worse than one that admits a failure.

*The report reads files, not objects.* Everything needed to write the report
is serialized to disk here, so ``generate_report.py`` can stay a consumer of
artifacts rather than an importer of pipeline internals. That is what lets the
report be regenerated in a second without retraining anything.

The one thing that happens exactly once, outside the per-architecture loop, is
the image cache. Decoding the dataset is shared work; doing it per
architecture would multiply it by nine for no benefit.
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ._config import (
    CHECKPOINTS_DIR,
    CNN_BATCH_SIZE,
    CNN_EPOCHS,
    CNN_IMAGE_SIZE,
    CNN_LEARNING_RATE,  # noqa: F401  - used in the run configuration
    CNN_LOSS,
    CNN_OPTIMIZER,
    CNN_PRETRAINED,
    CNN_WEIGHT_DECAY,
    LOGS_DIR,
    PLOTS_DIR,
    RANDOM_SEED,
)


@dataclass
class DeepResult:
    """Everything the benchmark learned about one deep architecture.

    Field names deliberately mirror Part 1's ``ModelResult`` where they
    overlap, so the shared plotting code and the combined table can treat a
    traditional model and a deep architecture the same way.
    """

    key: str
    name: str
    family: str = "Deep CNN"
    year: int = 0
    kind: str = "deep_cnn"
    succeeded: bool = True
    error: str = None

    accuracy: float = None
    macro_precision: float = None
    macro_recall: float = None
    macro_f1: float = None
    weighted_precision: float = None
    weighted_recall: float = None
    weighted_f1: float = None

    predictions: np.ndarray = None
    confusion_matrix: np.ndarray = None
    classification_report: dict = field(default_factory=dict)
    per_class_accuracy: dict = field(default_factory=dict)

    history: dict = field(default_factory=dict)
    overfitting: dict = field(default_factory=dict)
    parameters: dict = field(default_factory=dict)
    parameter_counts: dict = field(default_factory=dict)
    complexity: dict = field(default_factory=dict)
    inference: dict = field(default_factory=dict)
    single_image: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    @property
    def training_time_seconds(self):
        return self.history.get("total_training_seconds")

    @property
    def inference_time_ms_per_image(self):
        return self.inference.get("latency_ms_per_image")

    @property
    def checkpoint_megabytes(self):
        return self.history.get("checkpoint_megabytes")

    def metrics(self) -> dict:
        """The section 13 metrics, macro and weighted for all three."""
        return {
            "accuracy": self.accuracy,
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "macro_f1": self.macro_f1,
            "weighted_precision": self.weighted_precision,
            "weighted_recall": self.weighted_recall,
            "weighted_f1": self.weighted_f1,
            "training_time_seconds": self.training_time_seconds,
            "inference_time_ms_per_image": self.inference_time_ms_per_image,
        }


def score_predictions(truth, predictions, class_names) -> dict:
    """Every metric sections 13, 14 and 15 require, from one pair of arrays.

    ``zero_division=0`` throughout, matching Part 1. A class the model never
    predicts has undefined precision; scoring it zero is what makes the macro
    averages penalize ignoring a class rather than quietly skipping it.
    """
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    labels = list(range(len(class_names)))
    matrix = confusion_matrix(truth, predictions, labels=labels)

    # Section 15 asks for per-class accuracy, which is the row-normalized
    # diagonal: of the images truly of this class, the share predicted
    # correctly. That is recall per class, named as the assignment names it.
    row_totals = matrix.sum(axis=1)
    per_class = {
        class_names[index]: (
            round(float(matrix[index, index] / row_totals[index]), 6)
            if row_totals[index] else None
        )
        for index in labels
    }

    def averaged(function, average):
        return float(function(truth, predictions, average=average,
                             labels=labels, zero_division=0))

    return {
        "accuracy": float(accuracy_score(truth, predictions)),
        "macro_precision": averaged(precision_score, "macro"),
        "macro_recall": averaged(recall_score, "macro"),
        "macro_f1": averaged(f1_score, "macro"),
        "weighted_precision": averaged(precision_score, "weighted"),
        "weighted_recall": averaged(recall_score, "weighted"),
        "weighted_f1": averaged(f1_score, "weighted"),
        "confusion_matrix": matrix,
        "per_class_accuracy": per_class,
        "classification_report": classification_report(
            truth, predictions, labels=labels, target_names=list(class_names),
            output_dict=True, zero_division=0),
    }


def _jsonable(value):
    """numpy scalars and arrays into something json can write."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def benchmark_architecture(
    key: str,
    bundle: dict,
    class_names: list,
    output_dir: Path,
    epochs: int = CNN_EPOCHS,
    learning_rate: float = CNN_LEARNING_RATE,
    device=None,
    pretrained: bool = CNN_PRETRAINED,
    on_progress=None,
) -> DeepResult:
    """Fine-tune, select, score and measure one architecture.

    The order of operations is the part that matters: the test half is only
    touched after :func:`~.deep_training.train_model` has restored the
    best-validation-accuracy weights, which is what section 12 means by using
    the test set only once model selection is complete.
    """
    from . import deep_metrics, deep_models, deep_training

    # Looked up defensively, and outside the metadata's own assumptions. An
    # unknown key has to come back as a failed row like any other failure -
    # raising here would end a nine-architecture run because of one bad name,
    # which is precisely the behavior "failures are results" rules out.
    try:
        metadata = deep_models.architecture_metadata(key)
    except KeyError:
        return DeepResult(
            key=key,
            name=key,
            succeeded=False,
            error=(
                f"KeyError: unknown architecture {key!r}. Available: "
                + ", ".join(deep_models.architecture_names())
            ),
        )

    result = DeepResult(
        key=key,
        name=metadata["name"],
        family=metadata["family"],
        year=metadata["year"],
        metadata=metadata,
    )

    device = device or deep_models.device_for()
    output_dir = Path(output_dir)
    checkpoint = output_dir / CHECKPOINTS_DIR / f"best_{key}.pt"
    log_path = output_dir / LOGS_DIR / f"{key}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def say(message):
        if on_progress is not None:
            on_progress(message)

    try:
        model = deep_models.get_model(
            key, num_classes=len(class_names), pretrained=pretrained)
        result.parameter_counts = deep_models.parameter_counts(model)

        # Complexity is measured on the untrained graph: MAC count is a
        # property of the architecture, not of the weights, and measuring it
        # before training keeps a profiler failure from costing a trained
        # model.
        result.complexity = deep_metrics.complexity(model, CNN_IMAGE_SIZE)

        result.parameters = {
            "architecture": key,
            "pretrained": pretrained,
            "pretrained_weights": deep_models.weights_identifier(key)
                                  if pretrained else None,
            "num_classes": len(class_names),
            "input_size": list(CNN_IMAGE_SIZE),
            "optimizer": CNN_OPTIMIZER,
            "learning_rate": learning_rate,
            "weight_decay": CNN_WEIGHT_DECAY,
            "loss": CNN_LOSS,
            "epochs": epochs,
            "batch_size": CNN_BATCH_SIZE,
            "freeze_backbone": False,
            "early_stopping": False,
            "random_seed": RANDOM_SEED,
            "device": str(device),
        }

        # One append-only log line per epoch, so a long run can be watched
        # from another terminal and a crash leaves the completed epochs
        # behind rather than losing them with the process.
        with open(log_path, "w", encoding="utf-8") as log:
            def on_epoch(record):
                log.write(json.dumps(record) + "\n")
                log.flush()
                say(f"    epoch {record['epoch']:2d}/{epochs}  "
                    f"train {record['train_loss']:.4f}/"
                    f"{record['train_accuracy']:.3f}  "
                    f"val {record['validation_loss']:.4f}/"
                    f"{record['validation_accuracy']:.3f}  "
                    f"{record['epoch_seconds']:.1f}s")

            result.history = deep_training.train_model(
                model,
                bundle["train"],
                bundle["validation"],
                epochs=epochs,
                learning_rate=learning_rate,
                device=device,
                checkpoint_path=checkpoint,
                on_epoch=on_epoch,
            )

        result.overfitting = deep_training.overfitting_signals(result.history)

        # Model selection is finished. Only now is the test half touched.
        predictions, truth = deep_training.predict(model, bundle["test"], device)
        scored = score_predictions(truth, predictions, class_names)

        result.predictions = predictions
        result.accuracy = scored["accuracy"]
        result.macro_precision = scored["macro_precision"]
        result.macro_recall = scored["macro_recall"]
        result.macro_f1 = scored["macro_f1"]
        result.weighted_precision = scored["weighted_precision"]
        result.weighted_recall = scored["weighted_recall"]
        result.weighted_f1 = scored["weighted_f1"]
        result.confusion_matrix = scored["confusion_matrix"]
        result.classification_report = scored["classification_report"]
        result.per_class_accuracy = scored["per_class_accuracy"]

        result.inference = deep_metrics.inference_benchmark(
            model, bundle["test"], device)
        result.single_image = deep_metrics.single_image_latency(
            model, bundle["test"], device)

    except Exception as error:  # failures are results, per Part 1
        result.succeeded = False
        result.error = f"{type(error).__name__}: {error}"
        say(f"    FAILED - {result.error}")

    return result


def benchmark_all(
    split,
    color_mode: str,
    output_dir,
    keys=None,
    epochs: int = CNN_EPOCHS,
    learning_rate: float = CNN_LEARNING_RATE,
    pretrained: bool = CNN_PRETRAINED,
    device=None,
    on_progress=None,
) -> dict:
    """Run every requested architecture against one shared split and cache.

    Parameters
    ----------
    split
        A Part 1 :class:`~.benchmark.Split`. Untouched; the validation slice
        is derived from its training half.
    color_mode
        Must be ``"rgb"``; the deep path refuses anything else.
    output_dir
        The configuration's result directory, shared with Part 1 so that
        confusion matrices and per-class reports for both halves of the
        assignment land side by side.
    keys
        Architectures to run. Defaults to all nine constructible ones.

    Returns
    -------
    dict
        Results, the deep split summary, the environment, and where things
        were written.
    """
    from . import deep_data, deep_metrics, deep_models

    keys = list(keys or deep_models.architecture_names())
    device = device or deep_models.device_for()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def say(message):
        if on_progress is not None:
            on_progress(message)

    say(f"device: {device}")
    say("decoding images once, shared by every architecture")
    started = time.perf_counter()
    bundle = deep_data.build_loaders(split, color_mode, on_progress=say)
    deep_split = bundle["deep_split"]
    say(f"cache ready in {time.perf_counter() - started:.1f}s "
        f"({bundle['cache_shapes']['megabytes']} MB): "
        f"{len(deep_split.train_index)} train / "
        f"{len(deep_split.validation_index)} val / "
        f"{len(deep_split.test_index)} test")

    class_names = deep_split.class_names
    results = []
    for position, key in enumerate(keys, start=1):
        say(f"[{position}/{len(keys)}] {key}")
        result = benchmark_architecture(
            key, bundle, class_names, output_dir,
            epochs=epochs, learning_rate=learning_rate,
            device=device, pretrained=pretrained, on_progress=say)
        results.append(result)
        if result.succeeded:
            say(f"    accuracy {result.accuracy:.4f}  macro F1 "
                f"{result.macro_f1:.4f}  best epoch "
                f"{result.history.get('best_epoch')}  "
                f"{result.history.get('total_training_seconds', 0):.0f}s")

    environment = deep_metrics.environment_report()
    written = write_deep_artifacts(
        results, deep_split, bundle, environment, output_dir,
        epochs=epochs, learning_rate=learning_rate, device=device,
        color_mode=color_mode)

    return {
        "results": results,
        "deep_split": deep_split,
        "environment": environment,
        "output_directory": str(output_dir.resolve()),
        "written": written,
    }


def write_deep_artifacts(results, deep_split, bundle, environment, output_dir,
                         epochs, learning_rate, device, color_mode) -> dict:
    """Write everything the report will read back.

    Confusion matrices and per-class reports go into the directories Part 1
    already uses, because the assignment's required output layout wants
    ``confusion_matrices/alexnet.png`` and the Part 1 model keys cannot
    collide with architecture names. The deep-specific files are named
    separately so that re-running Part 1 cannot overwrite them and
    re-running Part 2 cannot overwrite Part 1.
    """
    from . import deep_visualization, visualization
    from .evaluation import classification_report_frame

    output_dir = Path(output_dir)
    (output_dir / "classification_reports").mkdir(parents=True, exist_ok=True)
    (output_dir / "confusion_matrices").mkdir(parents=True, exist_ok=True)
    (output_dir / PLOTS_DIR).mkdir(parents=True, exist_ok=True)

    class_names = deep_split.class_names
    written = {"confusion_matrices": {}, "training_curves": {}}

    for result in results:
        if not result.succeeded:
            continue
        written["confusion_matrices"][result.key] = str(
            visualization.plot_confusion_matrix(
                result, class_names,
                output_dir / "confusion_matrices" / f"{result.key}.png"))
        classification_report_frame(result).to_csv(
            output_dir / "classification_reports" / f"{result.key}.csv",
            index=False)
        # Named curves_<arch>, not training_<arch>: section 24 requires a
        # comparison figure called training_time.png in this same directory,
        # and a reader scanning plots/ should not have to work out which
        # training_*.png files are per-model curves and which is the
        # cross-model comparison.
        written["training_curves"][result.key] = str(
            deep_visualization.plot_training_curves(
                result, output_dir / PLOTS_DIR / f"curves_{result.key}.png"))

    metrics = {
        result.key: {
            "name": result.name,
            "family": result.family,
            "year": result.year,
            "kind": result.kind,
            "succeeded": result.succeeded,
            "error": result.error,
            **_jsonable(result.metrics()),
            "parameter_counts": _jsonable(result.parameter_counts),
            "complexity": _jsonable(result.complexity),
            "inference": _jsonable(result.inference),
            "single_image_inference": _jsonable(result.single_image),
            "checkpoint_megabytes": result.checkpoint_megabytes,
            "confusion_matrix": _jsonable(result.confusion_matrix),
            "classification_report": _jsonable(result.classification_report),
            "per_class_accuracy": _jsonable(result.per_class_accuracy),
            "training_history": _jsonable(result.history),
            "overfitting": _jsonable(result.overfitting),
            "parameters": _jsonable(result.parameters),
            "architecture_metadata": _jsonable(result.metadata),
        }
        for result in results
    }
    # Merged into whatever is already there, not written over it. The
    # assignment requires single-architecture runs
    # (`run_benchmark.py --model resnet50`), and the learning-rate remediation
    # depends on them: it re-runs two architectures out of ten. Replacing the
    # file would delete the other eight results, and the master table built
    # immediately afterward would look complete while describing two models.
    metrics_path = output_dir / "deep_metrics.json"
    if metrics_path.is_file():
        existing = json.loads(metrics_path.read_text(encoding="utf-8"))
        existing.update(metrics)
        metrics = existing
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    configuration = {
        "part": 2,
        "color_mode": color_mode,
        "input_size": list(CNN_IMAGE_SIZE),
        "cache_shapes": bundle["cache_shapes"],
        "random_seed": RANDOM_SEED,
        "split": _jsonable(deep_split.summary()),
        "class_names": list(class_names),
        "protocol": {
            "epochs": epochs,
            "batch_size": CNN_BATCH_SIZE,
            "optimizer": CNN_OPTIMIZER,
            # The rate section 6 prescribes, not the rate this particular
            # invocation used. A remediation run passes a lower rate for one
            # architecture, and recording that here would misdescribe the
            # other nine. What each architecture actually trained at is in
            # its own entry's "parameters.learning_rate".
            "learning_rate": CNN_LEARNING_RATE,
            "learning_rate_this_run": learning_rate,
            "learning_rate_note": (
                "Per-architecture rates are recorded in each entry's "
                "parameters.learning_rate; any departure from the prescribed "
                "rate is recorded in deep_lr_deviations.json."
            ),
            "weight_decay": CNN_WEIGHT_DECAY,
            "loss": CNN_LOSS,
            "pretrained": CNN_PRETRAINED,
            "freeze_backbone": False,
            "early_stopping": False,
            "selection_criterion": "highest_validation_accuracy",
            "train_transforms": [
                f"Resize({list(CNN_IMAGE_SIZE)}) via a {bundle['cache_shapes']['train'][1]}"
                "px cache",
                "RandomCrop(224)",
                "RandomHorizontalFlip(p=0.5)",
                "ToTensor",
                "Normalize(ImageNet mean/std)",
            ],
            "eval_transforms": [
                f"Resize({list(CNN_IMAGE_SIZE)})",
                "ToTensor",
                "Normalize(ImageNet mean/std)",
            ],
        },
        "device": str(device),
        "environment": environment,
        "deviations": _deviations(),
    }
    (output_dir / "deep_run_configuration.json").write_text(
        json.dumps(configuration, indent=2), encoding="utf-8")

    written["deep_metrics"] = str(output_dir / "deep_metrics.json")
    written["deep_run_configuration"] = str(
        output_dir / "deep_run_configuration.json")
    return written


def _deviations() -> list:
    """Every documented departure from the letter of the protocol.

    Section 6 permits adjustments provided they are documented, and the
    honest place for that list is the artifact the report reads, not a
    paragraph somebody has to remember to update.
    """
    return [
        {
            "item": "GPU memory measurement (section 21)",
            "deviation": "No CUDA device on this host, so "
                         "torch.cuda.max_memory_allocated cannot be called.",
            "handling": "Reported as the peak working set - weights plus the "
                        "largest set of intermediate tensors alive at once - "
                        "sampled inside the forward pass through per-module "
                        "hooks and kept as a maximum, in a pass separate from "
                        "the timed one. Sampling between batches instead was "
                        "tried and discarded: it reproduces the checkpoint "
                        "size (r = 1.000) because activations are already "
                        "released by then. Labeled as a sampled maximum, not "
                        "presented as a CUDA peak.",
        },
        {
            "item": "GoogLeNet auxiliary classifiers",
            "deviation": "Disabled so that every architecture's forward pass "
                         "returns a single logit tensor and one shared "
                         "training loop can serve all nine.",
            "handling": "Removes GoogLeNet's auxiliary loss and 4,329,984 "
                        "parameters. Inference-time architecture is "
                        "unchanged, since the auxiliary branches are "
                        "discarded at inference anyway.",
        },
        {
            "item": "Training device",
            "deviation": "The Part 1 Simple CNN trains on CPU for "
                         "cross-machine determinism; the deep architectures "
                         "train on MPS.",
            "handling": "Unavoidable - nine networks at 224x224 are not "
                        "tractable on CPU. Recorded in every result.",
        },
        {
            "item": "Validation split",
            "deviation": "The Part 1 split is train/test only, and section 12 "
                         "requires selection on validation accuracy.",
            "handling": "Validation is carved from the training half at the "
                        "same fraction and seed the Part 1 Simple CNN already "
                        "used, so the test half is unchanged and both model "
                        "families validate on identical rows.",
        },
        {
            "item": "Computational complexity (section 22)",
            "deviation": "fvcore and thop both count multiply-accumulates "
                         "and label the total 'flops'.",
            "handling": "Reported as MACs, with FLOPs derived explicitly as "
                        "2x MACs. Operators the profiler could not account "
                        "for are listed per architecture.",
        },
        {
            "item": "YOLO classification",
            "deviation": "ultralytics excludes numpy 2.0-2.3.4 on macOS and "
                         "numpy >= 2.3.5 needs Python 3.11+, so installing it "
                         "would downgrade numpy beneath the environment that "
                         "produced the Part 1 results.",
            "handling": "Run as a subprocess in a separate virtual "
                        "environment against the identical split, writing the "
                        "same artifacts. Its environment is recorded "
                        "separately.",
        },
    ]
