"""Orchestration for the public benchmarking entry point.

Also home to the stratified split, because section 10 names no module for it
and the fairness rule it enforces belongs with the orchestrator.

The important idea is that the split produces *indices*, not data. Section 4.3
requires two views of the same images - flattened vectors for the classical
models, tensors for the CNN - and section 7 requires every model to see
exactly the same training and testing samples. Splitting the two
representations separately would satisfy neither: two calls to
``train_test_split`` could diverge, and a divergence would be invisible in the
results. Splitting one index array once and slicing both views with it makes
that failure impossible rather than merely unlikely.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ._config import (
    COLOR_MODES,
    DATASET_TYPES,
    DISTRIBUTION_NAME,
    IMAGE_SIZE,
    RANDOM_SEED,
    RESULTS_DIR,
    TEST_SIZE,
    VERSION,
)
from .preprocessing import PreparedDataset, preprocess


@dataclass
class Split:
    """One stratified train/test division, reused by every model.

    Holds positions into a :class:`PreparedDataset` rather than copies of it,
    so the training and testing views of both representations are guaranteed
    to come from the same division.
    """

    dataset: PreparedDataset
    train_index: np.ndarray = field(default_factory=lambda: np.empty(0, np.int64))
    test_index: np.ndarray = field(default_factory=lambda: np.empty(0, np.int64))
    random_seed: int = RANDOM_SEED
    test_size: float = TEST_SIZE

    # Flattened vectors, for Logistic Regression, Decision Tree, Random
    # Forest, SVM and the fully connected network.
    @property
    def features_train(self) -> np.ndarray:
        return self.dataset.features[self.train_index]

    @property
    def features_test(self) -> np.ndarray:
        return self.dataset.features[self.test_index]

    # Image tensors, for the CNN.
    @property
    def images_train(self) -> np.ndarray:
        return self.dataset.images[self.train_index]

    @property
    def images_test(self) -> np.ndarray:
        return self.dataset.images[self.test_index]

    @property
    def labels_train(self) -> np.ndarray:
        return self.dataset.labels[self.train_index]

    @property
    def labels_test(self) -> np.ndarray:
        return self.dataset.labels[self.test_index]

    def class_distribution(self, labels: np.ndarray) -> dict:
        """Counts per class name for any label array from this dataset."""
        counts = {name: 0 for name in self.dataset.class_names}
        for encoded in labels:
            counts[self.dataset.class_names[encoded]] += 1
        return counts

    def distributions(self) -> dict:
        """The three distributions section 5 requires in the returned result."""
        return {
            "full": self.class_distribution(self.dataset.labels),
            "training": self.class_distribution(self.labels_train),
            "testing": self.class_distribution(self.labels_test),
        }

    def summary(self) -> dict:
        """Split information for the returned dictionary and the run config."""
        return {
            "training_samples": int(len(self.train_index)),
            "testing_samples": int(len(self.test_index)),
            "test_size": self.test_size,
            "random_seed": self.random_seed,
            "stratified": True,
        }


def make_split(
    dataset: PreparedDataset,
    test_size: float = TEST_SIZE,
    random_seed: int = RANDOM_SEED,
) -> Split:
    """Divide a prepared dataset once, stratified by label.

    Parameters
    ----------
    dataset
        Output of :func:`~.preprocessing.preprocess`.
    test_size
        Fraction held out for testing. Section 5 fixes this at 0.20.
    random_seed
        Section 5 fixes this at 42.

    Returns
    -------
    Split
        Index arrays for the training and testing halves.

    Raises
    ------
    ValueError
        A class has too few samples for a stratified split, or the test
        fraction is too small to give every class a testing sample.
    """
    from sklearn.model_selection import train_test_split

    total = len(dataset)
    class_count = len(dataset.class_names)
    counts = dataset.count_by_class()

    # Section 4.1 asks for a clear error when a stratified split cannot be
    # created. scikit-learn's own messages describe the constraint without
    # naming the class that violates it, which is the one thing the caller
    # needs in order to fix their data.
    thin = {name: count for name, count in counts.items() if count < 2}
    if thin:
        raise ValueError(
            f"Cannot build a stratified split: {thin} - every class needs at "
            "least 2 images so that it can appear in both the training and "
            "testing sets."
        )

    expected_test = int(np.floor(total * test_size))
    if expected_test < class_count:
        raise ValueError(
            f"Cannot build a stratified split: a {test_size:.0%} test fraction of "
            f"{total} image(s) is {expected_test} sample(s), fewer than the "
            f"{class_count} classes. Use more images per class."
        )

    # Split positions, not pixels. Both representations are then sliced with
    # the same indices, so they cannot disagree about who is in which half.
    positions = np.arange(total)
    train_index, test_index = train_test_split(
        positions,
        test_size=test_size,
        stratify=dataset.labels,
        random_state=random_seed,
    )

    # Sorted so the indices are stable to read, compare and serialise into
    # run_configuration.json. Membership is what stratification fixes; order
    # within each half carries no meaning.
    return Split(
        dataset=dataset,
        train_index=np.sort(train_index),
        test_index=np.sort(test_index),
        random_seed=random_seed,
        test_size=test_size,
    )


def _load(dataset, dataset_type: str, target_labels):
    """Hand the input to the loader that understands it.

    ``target_labels`` means something different to each of these - a list of
    class-folder names, a column name, a field name, or a label vector - so
    each loader is responsible for saying so when it is given the wrong shape.
    """
    from . import data_loader

    if dataset_type == "folder":
        return data_loader.load_folder(dataset, target_labels)
    if dataset_type == "csv":
        return data_loader.load_csv(dataset, target_labels)
    if dataset_type == "json":
        return data_loader.load_json(dataset, target_labels)
    return data_loader.load_array(dataset, target_labels)


def _jsonable(value):
    """Convert numpy scalars and arrays into something json can write."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_artifacts(results, split, dataset_type, color_mode, output_dir: Path) -> dict:
    """Write the result directory section 9 requires.

    Everything here is written from the results already computed - nothing is
    recalculated - so the files and the returned dictionary cannot disagree.
    """
    from . import visualization
    from .evaluation import (
        classification_report_frame,
        prediction_examples,
        summary_frame,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "classification_reports").mkdir(exist_ok=True)

    summary = summary_frame(results)
    summary.to_csv(output_dir / "benchmark_summary.csv", index=False)

    metrics = {
        result.key: {
            "name": result.name,
            "succeeded": result.succeeded,
            "error": result.error,
            **_jsonable(result.metrics()),
            "confusion_matrix": _jsonable(result.confusion_matrix),
            "classification_report": _jsonable(result.classification_report),
            "training_history": _jsonable(result.history),
            # Section 11 asks for examples of correct and incorrect
            # predictions, and the report reads files rather than importing
            # the pipeline, so they have to be written rather than only
            # returned.
            "prediction_examples": _jsonable(prediction_examples(result, split)),
        }
        for result in results
    }
    (output_dir / "benchmark_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8")

    # Section 9: the configuration file records image size, color mode, seed,
    # split, model parameters and package version - everything a reader needs
    # to reproduce the run without reading the source.
    configuration = {
        "package": {"name": DISTRIBUTION_NAME, "version": VERSION},
        "dataset_type": dataset_type,
        "color_mode": color_mode,
        "image_size": list(IMAGE_SIZE),
        "random_seed": split.random_seed,
        "split": _jsonable(split.summary()),
        "class_names": list(split.dataset.class_names),
        "models": {result.key: _jsonable(result.parameters) for result in results},
    }
    (output_dir / "run_configuration.json").write_text(
        json.dumps(configuration, indent=2), encoding="utf-8")

    for result in results:
        if result.succeeded:
            classification_report_frame(result).to_csv(
                output_dir / "classification_reports" / f"{result.key}.csv",
                index=False)

    visualization.save_all(results, split, output_dir)
    return metrics


def benchmark_image_classification(
    dataset,
    dataset_type: str,
    target_labels,
    color_mode: str,
) -> dict:
    """Load an image dataset, train all required classifiers,
    and return a complete benchmark comparison.

    Runs the whole pipeline: read the dataset in whichever of the four
    organizations it arrives in, standardize every image to 64x64 in the
    requested color mode, build one stratified 80/20 split at seed 42, train
    and score all six models on that same split, write the result directory,
    and return the comparison.

    Parameters
    ----------
    dataset
        Dataset root directory, CSV/JSON/JSONL manifest path, Pandas
        DataFrame, or NumPy image array.
    dataset_type
        One of "folder", "csv", "json", or "array".
    target_labels
        Class-folder names, manifest label-field name, DataFrame label
        column, or a label vector.
    color_mode
        Either "grayscale" for one channel or "rgb" for three channels.

    Returns
    -------
    dict
        The benchmark table, best model, dataset and split information,
        per-model results, confusion matrices, class-level reports and
        warnings. Output files are written to ``benchmark_results/``.
    """
    from .classical_models import classical_model_specs
    from .evaluation import (
        best_model,
        evaluate_all,
        prediction_examples,
        summary_frame,
    )
    from .neural_models import neural_model_specs

    if dataset_type not in DATASET_TYPES:
        raise ValueError(
            f"dataset_type must be one of {DATASET_TYPES}, got {dataset_type!r}"
        )
    if color_mode not in COLOR_MODES:
        raise ValueError(
            f"color_mode must be one of {COLOR_MODES}, got {color_mode!r}"
        )

    loaded = _load(dataset, dataset_type, target_labels)
    print(f"loaded {len(loaded)} sample(s) across {len(loaded.class_names)} class(es)")

    prepared = preprocess(loaded, color_mode)
    if prepared.skipped:
        print(f"skipped {len(prepared.skipped)} undecodable image(s)")
    print(f"standardized to {prepared.images.shape[1:]} in {color_mode}")

    split = make_split(prepared)
    print(f"split {len(split.train_index)} training / {len(split.test_index)} testing")

    specs = classical_model_specs() + neural_model_specs()

    # Printed as each model finishes: an RBF SVM on 12,288 features takes
    # minutes, and a run that prints nothing for that long looks hung.
    def announce(result):
        if result.succeeded:
            print(f"  {result.name:20} macro F1 {result.macro_f1:.4f}  "
                  f"fit {result.training_time_seconds:.2f}s")
        else:
            print(f"  {result.name:20} FAILED - {result.error}")

    results = evaluate_all(specs, split, on_progress=announce)

    output_dir = Path(RESULTS_DIR)
    _write_artifacts(results, split, dataset_type, color_mode, output_dir)
    print(f"wrote results to {output_dir.resolve()}")

    channels = prepared.images.shape[3]
    return {
        "package_information": {"name": DISTRIBUTION_NAME, "version": VERSION},
        "summary": summary_frame(results),
        "best_model": best_model(results),
        "dataset_information": {
            "dataset_type": dataset_type,
            "number_of_images": len(prepared),
            "number_of_classes": len(prepared.class_names),
            "class_names": list(prepared.class_names),
            "color_mode": color_mode,
            "image_shape": [*IMAGE_SIZE, channels],
            "class_distribution": split.distributions()["full"],
            "skipped_images": list(prepared.skipped),
        },
        "split_information": {
            **split.summary(),
            "training_distribution": split.distributions()["training"],
            "testing_distribution": split.distributions()["testing"],
        },
        "model_results": {
            result.key: {
                "name": result.name,
                "succeeded": result.succeeded,
                "error": result.error,
                **result.metrics(),
                "parameters": result.parameters,
                "training_history": result.history,
                "prediction_examples": prediction_examples(result, split),
            }
            for result in results
        },
        "confusion_matrices": {
            result.key: result.confusion_matrix
            for result in results if result.succeeded
        },
        "classification_reports": {
            result.key: result.classification_report
            for result in results if result.succeeded
        },
        "warnings": list(prepared.warnings),
        "output_directory": str(output_dir.resolve()),
    }
