"""Validate images, resize to a fixed size and normalise channels.

This is where the loaders' *promises* are redeemed. A loader reports only what
the filesystem can tell it; here every image is actually opened, and a file
that exists with a plausible extension but will not decode finally fails.
That means two things worth remembering:

* The class counts a loader reports are provisional. The real distribution -
  the one section 5 requires in the returned dictionary - is computed here,
  after decoding, and can be smaller.
* A class can disappear entirely at this stage if all of its images were
  corrupt, so the "at least two classes" rule is checked again.

Every image ends up as float32 in [0, 1] with shape (64, 64, 1) or
(64, 64, 3). Aspect ratio is preserved by letterboxing - scale the longest
side to 64, centre the result, pad the remainder with zeros - rather than
stretching, per section 4.2.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

from ._config import COLOR_MODES, IMAGE_SIZE
from .data_loader import LoadedDataset

# PIL's own names for the two colour modes the public API exposes.
_PIL_MODE = {"grayscale": "L", "rgb": "RGB"}
_CHANNELS = {"grayscale": 1, "rgb": 3}


@dataclass
class PreparedDataset:
    """Decoded, standardised images ready for training.

    ``images`` is the tensor the CNN consumes. ``features`` is the flattened
    view the classical models consume - a reshape of the same buffer, not a
    copy, so both representations cost one allocation rather than two.
    """

    images: np.ndarray = field(default_factory=lambda: np.empty((0, 0, 0, 0), np.float32))
    labels: np.ndarray = field(default_factory=lambda: np.empty((0,), np.int64))
    class_names: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    skipped: list = field(default_factory=list)
    # Where each surviving image came from, parallel to ``images``. A file
    # path for the three file-based organizations, a positional name for
    # array input. The report needs it to say which image an example was,
    # and it costs one string per image.
    sources: list = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.labels)

    @property
    def features(self) -> np.ndarray:
        """Flattened feature vectors: 4,096 grayscale or 12,288 RGB.

        A reshape of a contiguous array returns a view, so this shares memory
        with ``images`` instead of doubling it.
        """
        return self.images.reshape(len(self.images), -1)

    def count_by_class(self) -> dict:
        """The true distribution, counted after decoding."""
        counts = {name: 0 for name in self.class_names}
        for index in self.labels:
            counts[self.class_names[index]] += 1
        return counts


def _array_to_uint8(array: np.ndarray) -> tuple:
    """Bring an in-memory image to uint8 0-255, reporting any assumption made.

    The scale of an incoming array is genuinely ambiguous: 0-255 integers and
    0-1 floats are both common. Dividing an already-normalised float array by
    255 a second time would compress every image to near-black and quietly
    destroy accuracy, so the range is inspected rather than assumed.
    """
    if array.dtype == np.bool_:
        return (array.astype(np.uint8) * 255), None
    if np.issubdtype(array.dtype, np.integer):
        return np.clip(array, 0, 255).astype(np.uint8), None

    if np.issubdtype(array.dtype, np.floating):
        finite = array[np.isfinite(array)]
        peak = float(finite.max()) if finite.size else 0.0
        if peak <= 1.0:
            return np.clip(array * 255.0, 0, 255).astype(np.uint8), None
        if peak <= 255.0:
            return np.clip(array, 0, 255).astype(np.uint8), None
        return (
            np.clip(array, 0, 255).astype(np.uint8),
            f"values above 255 (max {peak:.4g}) were clipped",
        )

    raise ValueError(f"unsupported image dtype {array.dtype}")


def _to_image(source: Union[Path, np.ndarray]) -> tuple:
    """Open a path or wrap an array as a PIL image, plus any note about it.

    Arrays pass through uint8 because the geometric step uses PIL. For data
    that was 8-bit to begin with - which every photograph is - that costs
    nothing; for float input it is under 0.4% of the value range.
    """
    if isinstance(source, np.ndarray):
        array = source
        if array.ndim == 3 and array.shape[2] == 1:
            array = array[:, :, 0]
        if array.ndim not in (2, 3):
            raise ValueError(f"image array must be 2- or 3-dimensional, got {array.shape}")
        pixels, note = _array_to_uint8(array)
        return Image.fromarray(pixels), note

    # .load() forces the decode. Image.open() only reads the header, so a file
    # that is truncated or has garbage past the magic bytes opens fine and
    # fails later, somewhere far less helpful.
    image = Image.open(source)
    image.load()
    return image, None


def _letterbox(image: Image.Image, size: tuple) -> Image.Image:
    """Fit an image into ``size`` without distorting it.

    Scales the longest side to fit, centres the result, and leaves the
    remainder at zero. A 150x64 photograph becomes 64x27 of content with 37
    rows of padding: shapes stay true at the cost of constant pixels. The
    alternative, stretching to square, keeps every pixel meaningful but makes
    a tall animal short, which is a distortion the models would have to learn
    around.
    """
    target_width, target_height = size
    width, height = image.size

    scale = min(target_width / width, target_height / height)
    # max(1, ...) because a very lopsided image can round a side to zero,
    # which PIL rejects.
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))

    # LANCZOS because this is almost always downscaling, where it keeps
    # detail that a plain bilinear average smears.
    resized = image.resize(new_size, Image.LANCZOS)

    canvas = Image.new(image.mode, size, 0)
    canvas.paste(
        resized,
        ((target_width - new_size[0]) // 2, (target_height - new_size[1]) // 2),
    )
    return canvas


def _describe(source: Union[Path, np.ndarray], index: int) -> str:
    """Name a sample in a warning, however it arrived."""
    return str(source) if isinstance(source, Path) else f"image {index}"


def preprocess(dataset: LoadedDataset, color_mode: str) -> PreparedDataset:
    """Decode, standardise and encode a loaded dataset.

    Parameters
    ----------
    dataset
        Output of any of the four loaders.
    color_mode
        ``"grayscale"`` for one channel or ``"rgb"`` for three.

    Returns
    -------
    PreparedDataset
        Images as float32 in [0, 1] with shape ``(N, 64, 64, C)``, integer
        labels, and the class names in encoded order.

    Raises
    ------
    ValueError
        ``color_mode`` is not one of the two accepted values, or fewer than
        two classes still have a decodable image.
    """
    if color_mode not in COLOR_MODES:
        raise ValueError(
            f"color_mode must be one of {COLOR_MODES}, got {color_mode!r}"
        )

    pil_mode = _PIL_MODE[color_mode]
    channels = _CHANNELS[color_mode]
    total = len(dataset.samples)

    result = PreparedDataset(warnings=list(dataset.warnings))

    # Allocated for every sample up front and sliced back at the end. Growing
    # a list and stacking it would hold the images twice at the moment of the
    # stack, which for a 5,000-image RGB set is the difference between about
    # 245 MB and 490 MB.
    buffer = np.empty((total, *IMAGE_SIZE, channels), dtype=np.float32)
    kept_labels = []
    kept_sources = []
    kept = 0

    for index, (source, label) in enumerate(dataset.samples):
        try:
            image, note = _to_image(source)
        except Exception as error:
            # Section 4.1: skip corrupted files safely and record their paths.
            name = _describe(source, index)
            result.skipped.append(name)
            result.warnings.append(
                f"skipped (could not decode): {name} - "
                f"{type(error).__name__}: {error}"
            )
            continue

        if note:
            result.warnings.append(f"{_describe(source, index)}: {note}")

        # Convert first so the padding added next is in the target mode, and
        # so a palette or CMYK source becomes something predictable early.
        if image.mode != pil_mode:
            image = image.convert(pil_mode)

        pixels = np.asarray(_letterbox(image, IMAGE_SIZE), dtype=np.float32) / 255.0
        if pixels.ndim == 2:
            pixels = pixels[:, :, np.newaxis]

        buffer[kept] = pixels
        kept_labels.append(label)
        kept_sources.append(_describe(source, index))
        kept += 1

    # A class whose images were all corrupt is gone now, so the encoding is
    # built from what actually survived rather than from what was loaded.
    result.class_names = sorted(set(kept_labels))

    if len(result.class_names) < 2:
        raise ValueError(
            f"At least two classes with decodable images are required, found "
            f"{len(result.class_names)}: {result.class_names}. "
            f"{len(result.skipped)} of {total} image(s) could not be decoded."
        )

    # Sorted class names indexed by position is exactly what scikit-learn's
    # LabelEncoder produces, so our class_names[i] is the class the models
    # mean by i. A test asserts that equivalence rather than trusting it.
    encoding = {name: position for position, name in enumerate(result.class_names)}

    result.images = buffer[:kept]
    result.labels = np.array([encoding[label] for label in kept_labels], dtype=np.int64)
    result.sources = kept_sources

    if result.skipped:
        result.warnings.append(
            f"{len(result.skipped)} of {total} image(s) could not be decoded"
        )

    return result
