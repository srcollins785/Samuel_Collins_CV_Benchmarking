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

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import numpy as np

from ._config import SUPPORTED_EXTENSIONS

# A sample is a source of pixels paired with its class name as a string.
#
# For the three file-based organizations the source is a Path - a *promise* of
# pixels that preprocessing redeems by opening the file, resizing to 64x64 and
# discarding the full-resolution original. That laziness is what keeps peak
# memory near 60 MB for a 5,000-image RGB set instead of the ~3 GB the same
# images would occupy decoded at full size.
#
# Array input arrives already decoded, so there is no file to open and the
# source is the pixel array itself. Preprocessing tells the two apart with a
# single isinstance check at the point of decoding.
#
# Label encoding to integers happens once, in preprocessing, so that every
# loader stays free of that concern.
Sample = tuple[Union[Path, np.ndarray], str]


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


def _collect_samples(rows, base_dir: Path, result: LoadedDataset) -> None:
    """Validate ``(where, raw_path, raw_label)`` triples into samples.

    Shared by the CSV and JSON/JSONL loaders so the three manifest formats
    cannot develop three different ideas of what a bad row is. ``where``
    is a human-readable location - "row 4", "line 9", "record 2" - used
    verbatim in warnings, since each format numbers its contents differently.

    Appends to ``result`` in place; the caller sets ``class_names``.
    """
    seen_paths = set()

    for where, raw_path, raw_label in rows:
        if not isinstance(raw_path, str) or not raw_path.strip():
            result.warnings.append(f"skipped (blank image_path): {where}")
            continue
        if not isinstance(raw_label, str) or not raw_label.strip():
            result.warnings.append(f"skipped (blank label): {where}")
            continue

        label = raw_label.strip()
        path = Path(raw_path.strip())
        if not path.is_absolute():
            path = base_dir / path

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            result.warnings.append(
                f"skipped (unsupported extension {path.suffix!r}): {where}, {path}"
            )
            continue

        # The one structural failure a class-folder dataset cannot have: a
        # manifest can name a file that is simply not there.
        if not path.exists():
            result.warnings.append(f"skipped (not found): {where}, {path}")
            continue
        if not path.is_file():
            result.warnings.append(f"skipped (not a file): {where}, {path}")
            continue

        # A repeated path would put the same image into the dataset twice, so
        # copies could land in both halves of the split - the leakage section 7
        # prohibits. The first occurrence is kept.
        resolved = path.resolve()
        if resolved in seen_paths:
            result.warnings.append(
                f"skipped (duplicate of an earlier row): {where}, {path}"
            )
            continue
        seen_paths.add(resolved)

        result.samples.append((path, label))


def _sorted_class_names(result: LoadedDataset) -> list:
    """Class names in the order scikit-learn's LabelEncoder will use.

    LabelEncoder assigns integers alphabetically. Any other order would leave
    ``class_names[i]`` naming a different class than the models mean by ``i``,
    mislabelling every confusion matrix.
    """
    return sorted({label for _, label in result.samples})


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

    # Read the columns directly rather than through itertuples(), which
    # renames anything that is not a valid Python identifier: a label column
    # called "class name" would arrive as "_1", every row would look blank,
    # and the error would be about having no classes rather than about the
    # column name.
    raw_paths = frame["image_path"].tolist()
    raw_labels = frame[target_labels].tolist()

    # start=2 so the number matches the line a spreadsheet shows, counting
    # the header as line 1.
    rows = (
        (f"row {position}", raw_path, raw_label)
        for position, (raw_path, raw_label)
        in enumerate(zip(raw_paths, raw_labels), start=2)
    )
    _collect_samples(rows, manifest.parent, result)
    result.class_names = _sorted_class_names(result)

    return _finalize(result, f"Read {len(frame)} row(s) from {manifest}.")


def _parse_json_array(text: str) -> list:
    """Parse a whole-file JSON array of records."""
    parsed = json.loads(text)
    if not isinstance(parsed, list):
        raise ValueError(
            f"a JSON manifest must hold a list of records, found "
            f"{type(parsed).__name__}"
        )
    return parsed


def _parse_jsonl(text: str) -> list:
    """Parse one JSON record per line, ignoring blank lines."""
    records = []
    for number, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"line {number} is not valid JSON: {error.msg}") from None
        # Each line must be an object. Without this a whole JSON array written
        # on one line would parse "successfully" as a single JSONL record and
        # yield a list where a record belongs, so a .jsonl file holding an
        # array would load as zero samples instead of falling back to JSON.
        if not isinstance(parsed, dict):
            raise ValueError(
                f"line {number} is a JSON {type(parsed).__name__}, not an object"
            )
        records.append((number, parsed))
    if not records:
        raise ValueError("no JSON records found")
    return records


def load_json(
    dataset: Union[str, Path],
    target_labels: str,
) -> LoadedDataset:
    """Load a JSON or JSONL manifest.

    Both forms are reached through ``dataset_type="json"``. A ``.json`` file
    holds one list of records::

        [{"image_path": "images/a.jpg", "class_name": "cat"}]

    while a ``.jsonl`` file holds one record per line::

        {"image_path": "images/a.jpg", "class_name": "cat"}
        {"image_path": "images/b.jpg", "class_name": "dog"}

    The format is chosen by file extension, then confirmed by parsing. If the
    extension is absent, unfamiliar or simply wrong, the other form is tried
    before giving up, so a JSONL file saved as ``.txt`` still loads.

    As with CSV, ``target_labels`` is the *name of the label field*, a single
    string, not a list of class names.

    Parameters
    ----------
    dataset
        Path to the ``.json`` or ``.jsonl`` file.
    target_labels
        Name of the field holding each record's class.

    Returns
    -------
    LoadedDataset
        Samples as ``(path, class_name)`` pairs, with class names sorted.

    Raises
    ------
    FileNotFoundError
        The manifest does not exist.
    IsADirectoryError
        The path points at a directory.
    ValueError
        ``target_labels`` is not a field name, the file parses as neither
        JSON nor JSONL, no record carries a required field, or fewer than two
        classes survive validation.
    """
    manifest = Path(dataset)
    if not manifest.exists():
        raise FileNotFoundError(f"JSON manifest not found: {manifest}")
    if manifest.is_dir():
        raise IsADirectoryError(
            f'dataset_type="json" needs a JSON or JSONL file, but {manifest} '
            "is a directory."
        )

    if not isinstance(target_labels, str):
        raise ValueError(
            'dataset_type="json" needs target_labels to be the name of the label '
            f'field, for example "class_name". Got {type(target_labels).__name__}. '
            "(A list of class names is only used with dataset_type=\"folder\".)"
        )

    text = manifest.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"JSON manifest is empty: {manifest}")

    # Try the form the extension advertises first, then fall back to the other.
    # Extension is the better guess when it is present, but a misnamed file is
    # a reason to warn rather than to refuse.
    jsonl_first = manifest.suffix.lower() == ".jsonl"
    attempts = [("JSONL", _parse_jsonl), ("JSON", _parse_json_array)]
    if not jsonl_first:
        attempts.reverse()

    records = None
    failures = []
    for index, (name, parser) in enumerate(attempts):
        try:
            parsed = parser(text)
        except (ValueError, json.JSONDecodeError) as error:
            failures.append(f"as {name}: {error}")
            continue
        if index == 1:
            expected = "JSONL" if jsonl_first else "JSON"
            result_warning = (
                f"{manifest.name} does not parse as {expected} but does parse as "
                f"{name}; loaded as {name}"
            )
        else:
            result_warning = None
        records = parsed
        break

    if records is None:
        raise ValueError(
            f"Could not parse {manifest} as JSON or JSONL. " + "; ".join(failures)
        )

    # Normalise both forms to (where, record) so the rest is shared. A JSON
    # array numbers records by position; JSONL numbers them by line, which is
    # what a text editor shows.
    if records and isinstance(records[0], tuple):
        located = [(f"line {number}", record) for number, record in records]
    else:
        located = [(f"record {position}", record)
                   for position, record in enumerate(records, start=1)]

    result = LoadedDataset()
    if result_warning:
        result.warnings.append(result_warning)

    usable = [(where, record) for where, record in located if isinstance(record, dict)]
    for where, record in located:
        if not isinstance(record, dict):
            result.warnings.append(
                f"skipped (not an object): {where}, found {type(record).__name__}"
            )

    # A field absent from *every* record is a manifest-level mistake, the
    # equivalent of a missing CSV column, so it earns an error naming the
    # fields that are present rather than a warning per record.
    for field_name in ("image_path", target_labels):
        if usable and not any(field_name in record for _, record in usable):
            present = sorted({key for _, record in usable for key in record})
            raise ValueError(
                f"No record in {manifest} has an {field_name!r} field. "
                f"Fields present: {present}."
            )

    rows = (
        (where, record.get("image_path"), record.get(target_labels))
        for where, record in usable
    )
    _collect_samples(rows, manifest.parent, result)
    result.class_names = _sorted_class_names(result)

    return _finalize(result, f"Read {len(located)} record(s) from {manifest}.")


def _is_blank(value) -> bool:
    """True for the several ways a label vector can say 'nothing here'."""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return isinstance(value, str) and not value.strip()


def _is_whole_number(value) -> bool:
    """True for an integer, or a float that happens to be whole."""
    if isinstance(value, (bool, np.bool_)):
        return False
    if isinstance(value, (int, np.integer)):
        return True
    if isinstance(value, (float, np.floating)):
        return float(value).is_integer()
    return False


def _class_names_from_vector(values: list, result: LoadedDataset) -> list:
    """Turn a raw label vector into class-name strings, preserving order.

    Numeric labels are zero-padded to a common width. Without that, ``str()``
    sorts 0..10 as "0", "1", "10", "2", and since class names are sorted to
    match ``LabelEncoder``, ``class_names[1]`` would name "10" while the models
    mean something else - mislabelling every confusion matrix in a way that
    still looks plausible. Padding makes alphabetical order agree with numeric
    order at any class count.

    Blanks come back as ``None`` for the caller to skip.
    """
    present = [value for value in values if not _is_blank(value)]

    numeric = bool(present) and all(
        _is_whole_number(value) and int(value) >= 0 for value in present
    )
    if numeric:
        width = len(str(max(int(value) for value in present)))
        if width > 1:
            example = format(0, f"0{width}d")
            result.warnings.append(
                f"numeric labels zero-padded to width {width} so that sorting "
                f"matches numeric order (0 becomes {example!r})"
            )
        return [None if _is_blank(v) else f"{int(v):0{width}d}" for v in values]

    return [None if _is_blank(v) else str(v).strip() for v in values]


def load_array(
    dataset,
    target_labels,
) -> LoadedDataset:
    """Load an in-memory image tensor with its label vector.

    Accepted shapes, per the assignment::

        (N, Height, Width)      one channel, implicit
        (N, Height, Width, 1)   one channel
        (N, Height, Width, 3)   three channels

    Unlike the file-based loaders this receives pixels rather than paths, so
    there is nothing to open and no such thing as a missing file. The
    equivalent content failure is an image carrying NaN or infinity, which
    would poison every metric computed from it, so those are skipped.

    Pixel values are left exactly as given. Preprocessing decides how to
    normalise them by looking at the dtype, so an integer array of 0-255 and a
    float array already scaled to 0-1 both survive this stage untouched.

    Parameters
    ----------
    dataset
        Image tensor of shape ``(N, H, W)``, ``(N, H, W, 1)`` or
        ``(N, H, W, 3)``. Nested lists are accepted and converted.
    target_labels
        One-dimensional label vector with one entry per image.

    Returns
    -------
    LoadedDataset
        Samples as ``(image_array, class_name)`` pairs.

    Raises
    ------
    TypeError
        ``dataset`` is not array-like.
    ValueError
        The tensor has an unusable shape, the label vector length does not
        match the number of images, or fewer than two classes survive.
    """
    if isinstance(dataset, (list, tuple)):
        try:
            dataset = np.asarray(dataset)
        except Exception as error:  # ragged nesting, mixed types
            raise ValueError(f"Could not read dataset as an image array: {error}") from None
    if not isinstance(dataset, np.ndarray):
        raise TypeError(
            f'dataset_type="array" needs a NumPy image array, got '
            f"{type(dataset).__name__}."
        )

    if dataset.ndim == 3:
        channels = 1
    elif dataset.ndim == 4:
        channels = dataset.shape[3]
        if channels not in (1, 3):
            raise ValueError(
                f"Image arrays must have 1 or 3 channels, got {channels}. "
                f"Accepted shapes are (N, H, W), (N, H, W, 1) and (N, H, W, 3); "
                f"this array is {dataset.shape}."
            )
    else:
        raise ValueError(
            f"Image arrays must be 3- or 4-dimensional, got {dataset.ndim} "
            f"dimension(s) with shape {dataset.shape}. Accepted shapes are "
            f"(N, H, W), (N, H, W, 1) and (N, H, W, 3)."
        )

    if dataset.shape[0] == 0:
        raise ValueError("Image array is empty; there is nothing to classify.")

    if target_labels is None:
        raise ValueError(
            'dataset_type="array" needs target_labels to be the label vector, got None.'
        )
    if isinstance(target_labels, (str, bytes)):
        raise ValueError(
            'dataset_type="array" needs target_labels to be a label vector with one '
            f"entry per image, not a single name. Got {type(target_labels).__name__}."
        )

    labels = np.asarray(target_labels, dtype=object)
    if labels.ndim != 1:
        raise ValueError(
            f"target_labels must be one-dimensional, got shape {labels.shape}."
        )

    # Section 4.1: the number of images and the number of labels must agree.
    # Silently zipping to the shorter of the two would mislabel every image
    # after the first mismatch.
    if len(labels) != dataset.shape[0]:
        raise ValueError(
            f"Number of labels ({len(labels)}) does not match number of images "
            f"({dataset.shape[0]})."
        )

    result = LoadedDataset()
    names = _class_names_from_vector(list(labels), result)

    for index, (image, name) in enumerate(zip(dataset, names)):
        if name is None:
            result.warnings.append(f"skipped (blank label): image {index}")
            continue
        # The array equivalent of an undecodable file. NaN or infinity would
        # propagate through scaling and training into every reported metric.
        if np.issubdtype(image.dtype, np.floating) and not np.isfinite(image).all():
            result.warnings.append(
                f"skipped (contains NaN or infinity): image {index}"
            )
            continue
        result.samples.append((image, name))

    result.class_names = _sorted_class_names(result)

    return _finalize(
        result,
        f"Read {dataset.shape[0]} image(s) of shape "
        f"{tuple(dataset.shape[1:])} with {channels} channel(s).",
    )
