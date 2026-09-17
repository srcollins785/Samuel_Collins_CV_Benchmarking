"""Tests for the 224x224 representation the deep CNNs consume.

The first test here is the one that matters most in the whole Part 2
extension. Section 1A requires the deep architectures to be compared against
the Part 1 baselines on the same samples, and section 12 requires a validation
set that the Part 1 split does not contain. Those two requirements pull in
opposite directions, and the resolution - carve validation out of the training
half, leave the test half alone - is only trustworthy if it is actually true
of the indices rather than merely intended.

So the assertions are about set membership, not about accuracy: the test half
is unchanged, the three halves are disjoint, together they cover the dataset,
and the validation rows are the same rows the Part 1 Simple CNN already held
back. A comment claiming those things would not survive a refactor. These do.
"""

import numpy as np
import pytest

from samuel_collins_cv_benchmarking import deep_data
from samuel_collins_cv_benchmarking._config import (
    CNN_CACHE_SIZE,
    CNN_IMAGE_SIZE,
    CNN_VALIDATION_FRACTION,
)
from samuel_collins_cv_benchmarking.benchmark import make_split
from samuel_collins_cv_benchmarking.data_loader import load_array, load_folder
from samuel_collins_cv_benchmarking.neural_models import (
    VALIDATION_FRACTION,
    SimpleCNN,
)
from samuel_collins_cv_benchmarking.preprocessing import preprocess

CLASSES = ["alpha", "beta", "gamma"]
PER_CLASS = 20


@pytest.fixture(scope="module")
def image_folder(tmp_path_factory):
    """A small class-folder dataset written to disk.

    Written rather than synthesized in memory because the deep path reads the
    original files - that is the whole reason it exists - so a fixture that
    only lived in an array would not exercise it. Images are 300x300 so that
    caching to 256 and to 224 are both genuine downscales, as they are on the
    real dataset.
    """
    from PIL import Image

    root = tmp_path_factory.mktemp("deep_images") / "images"
    generator = np.random.default_rng(0)
    for band, name in zip((40, 110, 180), CLASSES):
        (root / name).mkdir(parents=True)
        for number in range(PER_CLASS):
            pixels = np.clip(
                generator.normal(band, 8, (300, 300, 3)), 0, 255).astype(np.uint8)
            Image.fromarray(pixels).save(root / name / f"{name}_{number:03d}.png")
    return root


@pytest.fixture(scope="module")
def split(image_folder):
    return make_split(preprocess(load_folder(image_folder, CLASSES), "rgb"))


@pytest.fixture(scope="module")
def deep(split):
    return deep_data.make_deep_split(split)


# -- the validation slice ----------------------------------------------------

def test_validation_slice_is_the_rows_simple_cnn_already_held_back(split):
    """deep_data and SimpleCNN must hold back the identical rows.

    Both read the same fraction at the same seed, so the Simple CNN and the
    deep architectures validate on the same images and their convergence
    curves are comparable. If this drifts, every training curve in the report
    is being compared against a slightly different yardstick, and nothing
    would fail loudly to say so.
    """
    mine_train, mine_validation = deep_data.validation_split(split.labels_train)
    theirs_train, theirs_validation = SimpleCNN()._validation_split(split.labels_train)

    assert np.array_equal(np.sort(mine_validation), np.sort(theirs_validation))
    assert np.array_equal(np.sort(mine_train), np.sort(theirs_train))


def test_the_two_validation_fractions_are_the_same_constant():
    """The equivalence above depends on these agreeing, so assert it directly."""
    assert CNN_VALIDATION_FRACTION == VALIDATION_FRACTION


def test_validation_slice_is_empty_when_it_cannot_cover_every_class():
    """Too small to stratify means train on everything, not crash.

    A 15% slice of 12 samples is 1, which cannot represent 3 classes.
    scikit-learn refuses that outright, so the split declines to make one
    rather than letting the refusal surface as a failed model.
    """
    labels = np.array([0, 1, 2] * 4)
    train, validation = deep_data.validation_split(labels)
    assert len(validation) == 0
    assert np.array_equal(train, np.arange(len(labels)))


# -- the three-way split -----------------------------------------------------

def test_test_half_is_unchanged_from_part_one(split, deep):
    """Section 1A: the deep models are scored on the Part 1 test half."""
    assert np.array_equal(deep.test_index, split.test_index)


def test_halves_are_disjoint_and_cover_the_dataset(split, deep):
    train = set(deep.train_index.tolist())
    validation = set(deep.validation_index.tolist())
    test = set(deep.test_index.tolist())

    assert not train & validation
    assert not train & test
    assert not validation & test
    # Validation came out of the training half and nowhere else.
    assert train | validation == set(split.train_index.tolist())
    assert train | validation | test == set(range(len(split.dataset)))


def test_validation_came_only_from_the_training_half(split, deep):
    """Stated separately because this is the rule section 3 actually cares about."""
    assert set(deep.validation_index.tolist()) <= set(split.train_index.tolist())
    assert not set(deep.validation_index.tolist()) & set(split.test_index.tolist())


def test_every_half_keeps_every_class(deep):
    """Stratification has to survive the second division, not just the first."""
    for index in (deep.train_index, deep.validation_index, deep.test_index):
        counts = np.bincount(deep.labels_for(index), minlength=len(CLASSES))
        assert counts.min() > 0


def test_summary_reports_the_three_way_proportions(deep):
    summary = deep.summary()
    assert summary["validation_fraction_of_training"] == CNN_VALIDATION_FRACTION
    assert summary["random_seed"] == 42
    total = (summary["training_samples"] + summary["validation_samples"]
             + summary["testing_samples"])
    assert total == len(CLASSES) * PER_CLASS


# -- the caches and the loaders ---------------------------------------------

def test_training_cache_is_larger_than_the_crop(split):
    """RandomCrop needs headroom or it is a no-op, which is why the sizes differ."""
    bundle = deep_data.build_loaders(split, "rgb")
    shapes = bundle["cache_shapes"]
    assert tuple(shapes["train"][1:3]) == CNN_CACHE_SIZE
    assert tuple(shapes["validation"][1:3]) == CNN_IMAGE_SIZE
    assert tuple(shapes["test"][1:3]) == CNN_IMAGE_SIZE
    assert CNN_CACHE_SIZE[0] > CNN_IMAGE_SIZE[0]


def test_batches_arrive_at_the_pretrained_input_size(split):
    import torch

    bundle = deep_data.build_loaders(split, "rgb")
    for name in ("train", "validation", "test"):
        images, labels = next(iter(bundle[name]))
        assert images.shape[1:] == (3, *CNN_IMAGE_SIZE)
        assert images.dtype == torch.float32
        assert len(images) == len(labels)


def test_training_augments_and_evaluation_does_not(split):
    """Section 7 forbids random augmentation in the validation/test pipeline.

    Checked by reading the same row twice: the training pipeline should give
    two different tensors, the evaluation pipeline the same tensor exactly.
    """
    import torch

    bundle = deep_data.build_loaders(split, "rgb")
    training = bundle["train"].dataset
    assert not torch.equal(training[0][0], training[0][0])

    for name in ("validation", "test"):
        evaluation = bundle[name].dataset
        assert torch.equal(evaluation[0][0], evaluation[0][0])


def test_labels_stay_aligned_with_their_pixels(split, deep):
    """A cache that reorders rows without reordering labels would train on noise."""
    bundle = deep_data.build_loaders(split, "rgb")
    dataset = bundle["test"].dataset
    expected = deep.labels_test
    got = np.array([int(dataset[position][1]) for position in range(len(dataset))])
    assert np.array_equal(got, expected)


# -- the refusals ------------------------------------------------------------

def test_grayscale_is_refused_with_a_reason(split):
    """Replicating one channel into three would quietly weaken transfer learning."""
    with pytest.raises(ValueError, match="rgb"):
        deep_data.build_loaders(split, "grayscale")


def test_array_input_is_refused_because_it_has_no_files():
    """Upsampling the 64x64 array to 224x224 would invent detail. Refuse instead."""
    generator = np.random.default_rng(1)
    images = generator.integers(0, 255, (30, 40, 40, 3), dtype=np.uint8)
    labels = ["alpha", "beta", "gamma"] * 10
    array_split = make_split(preprocess(load_array(images, labels), "rgb"))

    with pytest.raises(ValueError, match="array"):
        deep_data.build_loaders(array_split, "rgb")
