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

    return _finalize(result, f"Checked {len(target_labels)} folder(s) under {root}.")


def _finalize(result: LoadedDataset, context: str) -> LoadedDataset:
    """Apply the section 4.1 checks every loader owes, whatever its input.

    Keeping these here rather than in each loader means the four dataset
    organizations cannot drift into four different ideas of what counts as a
    usable dataset.
    """
    # At least two classes must remain, or there is nothing to classify and
    # the stratified split cannot be built.
    if len(result.class_names) < 2:
        raise ValueError(
            f"At least two classes with images are required, found "
            f"{len(result.class_names)}: {result.class_names}. {context}"
        )

    # Five images per class is recommended. Fewer is allowed but is worth
    # surfacing, because a class with one image cannot appear in both the
    # training and testing halves of the split.
    for label, count in result.count_by_class().items():
        if count < 5:
            result.warnings.append(
                f"class {label!r} has only {count} image(s); at least 5 is recommended"
            )

    return result


def load_csv(
    dataset: Union[str, Path],
    target_labels: str,
) -> LoadedDataset:
    """Load a CSV manifest listing one image per row.

    Expects the layout described in the assignment::

        image_path,class_name
        images/image001.jpg,cat
        images/image002.jpg,dog

    Relative paths are resolved against the manifest's own directory, so a
    manifest and its images can be moved together without editing any rows.
    Absolute paths are used as given.

    Note that ``target_labels`` means something different here than it does
    for a class-folder dataset: this is the *name of the label column*, a
    single string, not a list of class names.

    Parameters
    ----------
    dataset
        Path to the CSV file.
    target_labels
        Name of the column holding each row's class.

    Returns
    -------
    LoadedDataset
        Samples as ``(path, class_name)`` pairs. Class names are sorted
        alphabetically, matching how scikit-learn's ``LabelEncoder`` assigns
        integers, so ``class_names[i]`` is the class the models call ``i``.

    Raises
    ------
    FileNotFoundError
        The manifest does not exist.
    IsADirectoryError
        The path points at a directory.
    ValueError
        ``target_labels`` is not a column name, the manifest is empty or
        unparseable, a required column is absent, or fewer than two classes
        survive validation.
    """
    import pandas as pd

    manifest = Path(dataset)
    if not manifest.exists():
        raise FileNotFoundError(f"CSV manifest not found: {manifest}")
    if manifest.is_dir():
        raise IsADirectoryError(
            f'dataset_type="csv" needs a CSV file, but {manifest} is a directory.'
        )

    if not isinstance(target_labels, str):
        raise ValueError(
            'dataset_type="csv" needs target_labels to be the name of the label '
            f'column, for example "class_name". Got {type(target_labels).__name__}. '
            "(A list of class names is only used with dataset_type=\"folder\".)"
        )

    # dtype=str keeps a column of values like 1, 2, 3 as class *names* rather
    # than letting pandas infer integers, which would break the string label
    # contract every other loader follows.
    try:
        frame = pd.read_csv(manifest, dtype=str)
    except pd.errors.EmptyDataError:
        raise ValueError(f"CSV manifest is empty: {manifest}") from None
    except pd.errors.ParserError as error:
        raise ValueError(f"Could not parse CSV manifest {manifest}: {error}") from None

    for column in ("image_path", target_labels):
        if column not in frame.columns:
            raise ValueError(
                f"CSV manifest {manifest} has no {column!r} column. "
                f"Columns present: {list(frame.columns)}."
            )

    result = LoadedDataset()
    seen_paths = set()

    for position, row in enumerate(frame.itertuples(index=False), start=2):
        # start=2 so the number matches the line a spreadsheet would show,
        # counting the header as line 1.
        raw_path = getattr(row, "image_path", None)
        raw_label = getattr(row, target_labels, None)

        if not isinstance(raw_path, str) or not raw_path.strip():
            result.warnings.append(f"skipped (blank image_path): row {position}")
            continue
        if not isinstance(raw_label, str) or not raw_label.strip():
            result.warnings.append(f"skipped (blank label): row {position}")
            continue

        label = raw_label.strip()
        path = Path(raw_path.strip())
        if not path.is_absolute():
            path = manifest.parent / path

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            result.warnings.append(
                f"skipped (unsupported extension {path.suffix!r}): row {position}, {path}"
            )
            continue

        # The one structural failure a class-folder dataset cannot have: a
        # manifest can name a file that is simply not there.
        if not path.exists():
            result.warnings.append(f"skipped (not found): row {position}, {path}")
            continue
        if not path.is_file():
            result.warnings.append(f"skipped (not a file): row {position}, {path}")
            continue

        # A repeated path would put the same image into the dataset twice, so
        # copies could land in both halves of the split - the leakage section 7
        # prohibits. The first occurrence is kept.
        resolved = path.resolve()
        if resolved in seen_paths:
            result.warnings.append(
                f"skipped (duplicate of an earlier row): row {position}, {path}"
            )
            continue
        seen_paths.add(resolved)

        result.samples.append((path, label))

    # Sorted, because scikit-learn's LabelEncoder sorts when it assigns
    # integers. Any other order would leave class_names[i] naming a different
    # class than the models mean by i, mislabelling every confusion matrix.
    result.class_names = sorted({label for _, label in result.samples})

    return _finalize(result, f"Read {len(frame)} row(s) from {manifest}.")
