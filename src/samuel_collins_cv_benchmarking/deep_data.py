"""The 224x224 representation the pretrained CNNs consume.

This is the third view of one split. Part 1 established two - flattened
vectors for the classical models, 64x64 tensors for the Simple CNN - and both
are sliced out of a single index array so that they cannot disagree about
which images are in which half. The deep architectures need a third view at a
different resolution, and the rule that makes the whole comparison defensible
is that they get it *without touching the split*: the same ``train_index`` and
``test_index`` that produced every Part 1 number select the pixels here.

Three things in this file are worth understanding before changing it.

*The validation set comes out of the training half.* Section 12 requires the
checkpoint with the best validation accuracy, and section 3 requires the test
set to stay unseen until model selection is finished. Those two together mean
a validation set is mandatory and it cannot come from the test half. Part 1's
Simple CNN already solved this by carving 15% off its training data, so the
deep models reuse that exact fraction and seed and validate on the identical
rows. The consequence - neural models train on 68% of the data where the
classical models get 80% - is real, was already true in Part 1, and belongs in
the report rather than in a comment.

*Pixels are decoded once and cached.* The GPU cost of nine architectures at 20
epochs is around an hour on this machine; decoding 3,400 JPEGs from disk
nine times over, twenty times each, would dwarf it. So every image is decoded
exactly once into a uint8 array and augmented from memory thereafter. The
training half is cached at 256x256 and the validation and test halves at
224x224, for the reason in the next paragraph. Together that is roughly 900 MB
at uint8, against 48 GB of RAM.

*The two cache sizes are not an inconsistency.* Section 7 asks for
``Resize((224, 224))`` then ``RandomCrop(...)`` on the training transform,
which does nothing at all unless the source is bigger than the crop - so the
training cache is held at 256x256 to give the crop the headroom the transform
exists to provide. The validation and test transform in section 7 has no
augmentation in it, so those images are cached at exactly 224x224 and the
only thing left to do to them at load time is normalize. Caching them at 256
and resizing down would have made their pipeline a 256->224 resample rather
than the original->224 resize section 7 specifies.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ._config import (
    CNN_BATCH_SIZE,
    CNN_CACHE_SIZE,
    CNN_IMAGE_SIZE,
    CNN_VALIDATION_FRACTION,
    IMAGENET_MEAN,
    IMAGENET_STD,
    RANDOM_SEED,
)


def validation_split(
    labels: np.ndarray,
    fraction: float = CNN_VALIDATION_FRACTION,
    seed: int = RANDOM_SEED,
) -> tuple:
    """Carve a stratified validation slice out of training labels.

    Deliberately identical in behavior to ``SimpleCNN._validation_split``,
    including its refusal to split a set too small to give every class a
    validation sample. Sharing the behavior is the point: the Simple CNN and
    the deep architectures then hold back the same rows at the same seed, so
    the report can compare their convergence curves against each other rather
    than each against itself. A test asserts the two agree on real data.

    Parameters
    ----------
    labels
        Encoded labels of the *training half only*.
    fraction
        Share of the training half held back. Defaults to the one value both
        implementations read.
    seed
        Section 3 fixes this at 42.

    Returns
    -------
    tuple
        ``(train_positions, validation_positions)`` as positions *within the
        training half*, not dataset indices. The caller maps them through
        ``split.train_index``.
    """
    from sklearn.model_selection import train_test_split

    positions = np.arange(len(labels))
    counts = np.bincount(labels)
    class_count = int((counts > 0).sum())

    # Stratifying needs every class on both sides, so the validation slice has
    # to be at least as large as the class count - enough samples is not
    # enough. When the training half cannot spare a usable slice, train on all
    # of it and let the epoch cap end the run.
    validation_size = int(np.floor(len(labels) * fraction))
    if len(labels) < 2 or counts.min() < 2 or validation_size < class_count:
        return positions, np.empty(0, dtype=int)

    return train_test_split(
        positions,
        test_size=fraction,
        stratify=labels,
        random_state=seed,
    )


@dataclass
class DeepSplit:
    """A three-way train/validation/test view over the Part 1 split.

    ``test_index`` is copied straight from the Part 1 split and is never
    recomputed, so the test half every deep model is scored on is the same
    half every traditional model was scored on. ``train_index`` is the Part 1
    training half minus the validation slice.
    """

    split: object
    train_index: np.ndarray = field(default_factory=lambda: np.empty(0, np.int64))
    validation_index: np.ndarray = field(default_factory=lambda: np.empty(0, np.int64))
    test_index: np.ndarray = field(default_factory=lambda: np.empty(0, np.int64))

    @property
    def class_names(self) -> list:
        return list(self.split.dataset.class_names)

    def labels_for(self, index: np.ndarray) -> np.ndarray:
        return self.split.dataset.labels[index]

    @property
    def labels_train(self) -> np.ndarray:
        return self.labels_for(self.train_index)

    @property
    def labels_validation(self) -> np.ndarray:
        return self.labels_for(self.validation_index)

    @property
    def labels_test(self) -> np.ndarray:
        return self.labels_for(self.test_index)

    def summary(self) -> dict:
        """Split sizes for the run configuration and the report."""
        total = len(self.split.dataset)
        return {
            "training_samples": int(len(self.train_index)),
            "validation_samples": int(len(self.validation_index)),
            "testing_samples": int(len(self.test_index)),
            "training_fraction": round(len(self.train_index) / max(1, total), 4),
            "validation_fraction": round(len(self.validation_index) / max(1, total), 4),
            "testing_fraction": round(len(self.test_index) / max(1, total), 4),
            "validation_fraction_of_training": CNN_VALIDATION_FRACTION,
            "random_seed": self.split.random_seed,
            "stratified": True,
            # Stated because it is the asymmetry a reader should know about:
            # the classical models trained on the full training half.
            "note": (
                "Validation is carved from the training half only; the test "
                "half is identical to the Part 1 split and unseen until model "
                "selection is complete."
            ),
        }


def make_deep_split(split) -> DeepSplit:
    """Add a validation slice to a Part 1 split without altering it.

    Parameters
    ----------
    split
        A :class:`~.benchmark.Split` from ``make_split``.

    Returns
    -------
    DeepSplit
        Dataset-level indices for the three halves.
    """
    train_positions, validation_positions = validation_split(split.labels_train)

    # Map positions-within-the-training-half back to dataset indices, which is
    # what the image cache and the label lookups both need.
    train_index = np.sort(split.train_index[train_positions])
    validation_index = np.sort(split.train_index[validation_positions])

    return DeepSplit(
        split=split,
        train_index=train_index,
        validation_index=validation_index,
        # Copied, not recomputed. This is the line that keeps Part 1 and
        # Part 2 comparable.
        test_index=np.asarray(split.test_index),
    )


def _openable_sources(split) -> list:
    """The dataset's sources, checked to be real files.

    Deep CNNs read the original images from disk, because the in-memory Part 1
    array is 64x64 and upsampling it to 224x224 would feed the pretrained
    weights detail that no longer exists. That requires file paths, so array
    input cannot run this path at all - and saying so is better than
    silently producing plausible numbers from upsampled thumbnails.
    """
    sources = list(split.dataset.sources)
    missing = [s for s in sources[:50] if not Path(str(s)).is_file()]
    if missing:
        raise ValueError(
            "The deep CNN benchmark reads the original image files at "
            f"224x224, and {len(missing)} of the first {min(50, len(sources))} "
            f"sources are not readable files (for example {missing[0]!r}). "
            "This happens with dataset_type='array', which carries pixels "
            "rather than paths. Use 'folder', 'csv' or 'json' input for the "
            "deep architectures."
        )
    return sources


def cache_pixels(split, index: np.ndarray, size: tuple, on_progress=None) -> np.ndarray:
    """Decode the given dataset rows once, at ``size``, as uint8.

    Parameters
    ----------
    split
        The Part 1 split, whose dataset carries the source paths.
    index
        Dataset indices to decode, in the order they should be cached.
    size
        ``(height, width)`` to resize to.
    on_progress
        Called with ``(done, total)`` occasionally, so a 5,000-image decode
        does not look hung.

    Returns
    -------
    numpy.ndarray
        ``(len(index), height, width, 3)`` uint8, parallel to ``index``.

    Notes
    -----
    Bilinear resampling, to match the interpolation ``torchvision.Resize``
    uses by default; PIL's bilinear filter antialiases on downscale, as
    torchvision's does. Images are converted to RGB on the way in, so a
    grayscale or palettized file in an otherwise RGB dataset still produces
    three channels rather than breaking the batch.
    """
    from PIL import Image

    sources = _openable_sources(split)
    height, width = size
    cache = np.empty((len(index), height, width, 3), dtype=np.uint8)

    for position, dataset_index in enumerate(index):
        with Image.open(str(sources[int(dataset_index)])) as handle:
            image = handle.convert("RGB").resize(
                (width, height), Image.Resampling.BILINEAR)
            cache[position] = np.asarray(image, dtype=np.uint8)
        if on_progress is not None and (position + 1) % 500 == 0:
            on_progress(position + 1, len(index))

    return cache


class CachedImageDataset:
    """A torch dataset over cached uint8 pixels.

    Two modes, matching the two transform pipelines in section 7. ``train``
    applies ``RandomCrop`` and ``RandomHorizontalFlip`` to the 256x256 cache
    before normalizing; ``eval`` normalizes the 224x224 cache and nothing
    else, because section 7 forbids random augmentation in the validation and
    test pipeline.

    The augmentation runs per access rather than once, so each epoch sees
    different crops and flips of the same cached decode - which is the whole
    point of augmentation and the reason the cache stores pixels rather than
    finished tensors.
    """

    def __init__(self, pixels: np.ndarray, labels: np.ndarray, mode: str):
        if mode not in ("train", "eval"):
            raise ValueError(f"mode must be 'train' or 'eval', got {mode!r}")

        import torch
        from torchvision.transforms import v2

        self.pixels = pixels
        self.labels = torch.from_numpy(np.asarray(labels).astype(np.int64))
        self.mode = mode

        # v2 transforms operate on tensors, so the cached array never has to
        # become a PIL image again. ToDtype with scale=True is the ToTensor
        # step section 7 names: uint8 0-255 to float 0-1.
        normalize = [
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD)),
        ]
        if mode == "train":
            self.transform = v2.Compose([
                v2.RandomCrop(CNN_IMAGE_SIZE),
                v2.RandomHorizontalFlip(p=0.5),
                *normalize,
            ])
        else:
            self.transform = v2.Compose(normalize)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, position: int):
        import torch

        # Channels-last uint8 cache to channels-first, which is what every
        # torchvision transform and every conv layer expects.
        image = torch.from_numpy(
            np.ascontiguousarray(self.pixels[position])).permute(2, 0, 1)
        return self.transform(image), self.labels[position]


def make_loader(dataset: CachedImageDataset, shuffle: bool,
                batch_size: int = CNN_BATCH_SIZE, seed: int = RANDOM_SEED):
    """Wrap a cached dataset in a DataLoader.

    ``num_workers=0`` on purpose. Workers exist to hide file I/O behind
    computation, and the cache has already removed the file I/O; spawning
    them would add process startup and would copy a 668 MB array into every
    worker for no gain.

    Shuffling draws from a generator seeded from ``seed``, so two runs of the
    same architecture see batches in the same order and their loss curves are
    comparable rather than merely similar.
    """
    import torch
    from torch.utils.data import DataLoader

    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        drop_last=False,
    )


def build_loaders(split, color_mode: str, on_progress=None) -> dict:
    """Everything the deep training loop needs, from a Part 1 split.

    Decodes the three halves once and returns their loaders along with the
    :class:`DeepSplit` that defined them.

    Parameters
    ----------
    split
        A :class:`~.benchmark.Split` from ``make_split``.
    color_mode
        The Part 1 color mode. Only ``"rgb"`` is supported here, for the
        reason in the raised message.
    on_progress
        Passed through to :func:`cache_pixels`.

    Returns
    -------
    dict
        ``deep_split``, ``train``, ``validation``, ``test`` loaders, and the
        cache shapes for the run configuration.
    """
    if color_mode != "rgb":
        raise ValueError(
            f"The deep CNN benchmark requires color_mode='rgb', got "
            f"{color_mode!r}. Every pretrained backbone was trained on "
            "three-channel ImageNet data; running grayscale would mean "
            "replicating one channel into three and feeding the weights "
            "something they never saw, which costs accuracy and makes the "
            "transfer-learning comparison meaningless. Benchmark the deep "
            "architectures on the RGB configuration instead."
        )

    deep = make_deep_split(split)

    def report(label):
        def hook(done, total):
            if on_progress is not None:
                on_progress(f"  caching {label}: {done}/{total}")
        return hook

    # The training half only is cached larger, to give RandomCrop its headroom.
    train_pixels = cache_pixels(split, deep.train_index, CNN_CACHE_SIZE,
                                report("train"))
    validation_pixels = cache_pixels(split, deep.validation_index, CNN_IMAGE_SIZE,
                                     report("validation"))
    test_pixels = cache_pixels(split, deep.test_index, CNN_IMAGE_SIZE,
                               report("test"))

    return {
        "deep_split": deep,
        "train": make_loader(
            CachedImageDataset(train_pixels, deep.labels_train, "train"),
            shuffle=True),
        "validation": make_loader(
            CachedImageDataset(validation_pixels, deep.labels_validation, "eval"),
            shuffle=False),
        "test": make_loader(
            CachedImageDataset(test_pixels, deep.labels_test, "eval"),
            shuffle=False),
        "cache_shapes": {
            "train": list(train_pixels.shape),
            "validation": list(validation_pixels.shape),
            "test": list(test_pixels.shape),
            "megabytes": round(
                (train_pixels.nbytes + validation_pixels.nbytes
                 + test_pixels.nbytes) / 1e6, 1),
        },
    }
