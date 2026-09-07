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

from dataclasses import dataclass, field

import numpy as np

from ._config import COLOR_MODES, DATASET_TYPES, RANDOM_SEED, TEST_SIZE
from .preprocessing import PreparedDataset


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


def benchmark_image_classification(
    dataset,
    dataset_type: str,
    target_labels,
    color_mode: str,
) -> dict:
    """Load an image dataset, train all required classifiers,
    and return a complete benchmark comparison.

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
    """
    if dataset_type not in DATASET_TYPES:
        raise ValueError(
            f"dataset_type must be one of {DATASET_TYPES}, got {dataset_type!r}"
        )
    if color_mode not in COLOR_MODES:
        raise ValueError(
            f"color_mode must be one of {COLOR_MODES}, got {color_mode!r}"
        )
    raise NotImplementedError("Benchmark pipeline not yet implemented.")
