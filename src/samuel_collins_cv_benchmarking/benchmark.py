"""Orchestration for the public benchmarking entry point."""

from ._config import COLOR_MODES, DATASET_TYPES


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
