"""Load folder, CSV, JSON/JSONL and array datasets into a common form.

Every loader answers the same question - *which samples are in this dataset,
and what is each one labeled?* - and answers it the same way, so that nothing
downstream needs to know which of the four organizations the caller used.

Loaders deliberately do not open images. They report *structural* problems
(a class folder that is absent, a manifest row naming a file that is not on
disk, an unsupported extension), because those are visible from the filesystem
alone. Whether a file's bytes actually decode is a *content* problem, and
preprocessing reports it after trying to read the pixels. Both stages append to
the same warning list, which the benchmark returns to the caller.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

from ._config import SUPPORTED_EXTENSIONS

# A sample is a path to decode later, paired with its class name as a string.
# Label encoding to integers happens once, in preprocessing, so that every
# loader stays free of that concern.
Sample = tuple[Path, str]


@dataclass
class LoadedDataset:
    """What every loader returns, whatever the input organization was."""

    samples: list[Sample] = field(default_factory=list)
    class_names: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.samples)

    def count_by_class(self) -> dict[str, int]:
        """Samples retained per class, used for the distribution report."""
        counts = {name: 0 for name in self.class_names}
        for _, label in self.samples:
            counts[label] += 1
        return counts


def load_folder(
    dataset: Union[str, Path],
    target_labels: list[str],
) -> LoadedDataset:
    """Load a class-folder dataset, where each subfolder name is the label.

    Expects the layout described in the assignment::

        animals/
        |-- cat/cat_001.jpg
        |-- dog/dog_001.jpg
        `-- horse/horse_001.jpg

    Parameters
    ----------
    dataset
        Path to the directory holding one subfolder per class.
    target_labels
        The class-folder names to load, in the caller's preferred order.

    Returns
    -------
    LoadedDataset
        Samples as ``(path, class_name)`` pairs, sorted for reproducibility.

    Raises
    ------
    FileNotFoundError
        The dataset directory does not exist.
    NotADirectoryError
        The path exists but is a file.
    ValueError
        ``target_labels`` is not a list of names, repeats a class name, names
        a class folder that is missing, or fewer than two classes survive
        validation.
    """
    root = Path(dataset)
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(
            f'dataset_type="folder" needs a directory, but {root} is a file.'
        )

    if isinstance(target_labels, (str, bytes)) or not isinstance(target_labels, (list, tuple)):
        raise ValueError(
            'dataset_type="folder" needs target_labels to be a list of class-folder '
            f"names, for example [\"cat\", \"dog\"]. Got {type(target_labels).__name__}."
        )

    # A repeated class name would walk the same folder twice, putting every one
    # of its images into the dataset twice. The duplicates would then be split
    # independently, so the same image could land in both the training and the
    # testing half - the data leakage section 7 prohibits. Reject it outright.
    seen, duplicates = set(), []
    for name in target_labels:
        if name in seen and name not in duplicates:
            duplicates.append(name)
        seen.add(name)
    if duplicates:
        raise ValueError(
            f"target_labels contains repeated class name(s): {duplicates}. "
            "Each class must appear exactly once."
        )

    missing = [name for name in target_labels if not (root / name).is_dir()]
    if missing:
        available = sorted(p.name for p in root.iterdir() if p.is_dir())
        raise ValueError(
            f"Class folder(s) not found under {root}: {missing}. "
            f"Available subfolders: {available or 'none'}."
        )

    result = LoadedDataset()
    empty_classes = []

    for label in target_labels:
        class_dir = root / label
        kept = 0

        # Sorted so that two runs on the same folder produce the same ordering.
        # Everything downstream is seeded, but a seed only reproduces a result
        # if the input order it shuffles is itself stable.
        for path in sorted(class_dir.iterdir()):
            if path.is_dir():
                continue
            if path.name.startswith("."):
                continue  # .DS_Store and friends are not worth warning about
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                result.warnings.append(
                    f"skipped (unsupported extension {path.suffix!r}): {path}"
                )
                continue
            result.samples.append((path, label))
            kept += 1

        if kept == 0:
            empty_classes.append(label)
        else:
            result.class_names.append(label)

    for label in empty_classes:
        result.warnings.append(f"dropped class {label!r}: no supported image files")

    # Section 4.1: at least two classes must remain, or there is nothing to
    # classify and the stratified split cannot be built.
    if len(result.class_names) < 2:
        raise ValueError(
            f"At least two classes with images are required, found "
            f"{len(result.class_names)}: {result.class_names}. "
            f"Checked {len(target_labels)} folder(s) under {root}."
        )

    # Section 4.1 recommends five images per class. Fewer is allowed but is
    # worth surfacing, because a class with one image cannot appear in both
    # the training and testing halves of the split.
    for label, count in result.count_by_class().items():
        if count < 5:
            result.warnings.append(
                f"class {label!r} has only {count} image(s); at least 5 is recommended"
            )

    return result
