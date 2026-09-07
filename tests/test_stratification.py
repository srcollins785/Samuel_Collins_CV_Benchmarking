"""Tests for the single stratified train/test split.

Covers the assignment's requirements that training and testing sets do not
overlap, that the split is stratified, and that the same split is used for
every model.

That last requirement is the reason the split produces indices rather than
data. Section 4.3 needs two views of the same images - flattened vectors for
the classical models, tensors for the CNN - and if those were split
separately the two could disagree about who is in which half without anything
failing. Slicing both with one index array makes the disagreement impossible,
and the tests below check that property directly rather than checking that
two split calls happened to match.
"""

import numpy as np
import pytest

from samuel_collins_cv_benchmarking.data_loader import load_array
from samuel_collins_cv_benchmarking.preprocessing import preprocess
from samuel_collins_cv_benchmarking.benchmark import Split, make_split


def prepared(labels, size=(8, 8, 3), seed=0):
    """A prepared dataset carrying the given labels and random pixels."""
    generator = np.random.default_rng(seed)
    tensor = generator.integers(0, 256, size=(len(labels), *size), dtype=np.uint8)
    return preprocess(load_array(tensor, list(labels)), "rgb")


BALANCED = ["cat"] * 40 + ["dog"] * 40 + ["horse"] * 40
IMBALANCED = ["cat"] * 40 + ["dog"] * 40 + ["horse"] * 20


@pytest.fixture
def split():
    return make_split(prepared(IMBALANCED))


# --------------------------------------------------------------------------
# The fairness rule
# --------------------------------------------------------------------------

class TestNoLeakage:

    def test_training_and_testing_do_not_overlap(self, split):
        assert set(split.train_index).isdisjoint(set(split.test_index))

    def test_every_sample_is_used_exactly_once(self, split):
        combined = np.concatenate([split.train_index, split.test_index])
        assert sorted(combined) == list(range(len(split.dataset)))

    def test_no_test_image_appears_in_the_training_pixels(self, split):
        # The property that actually matters, checked on the pixels rather
        # than on the bookkeeping.
        train_rows = {row.tobytes() for row in split.features_train}
        test_rows = {row.tobytes() for row in split.features_test}
        assert train_rows.isdisjoint(test_rows)

    def test_split_sizes_match_the_requested_fraction(self, split):
        total = len(split.dataset)
        assert len(split.test_index) == round(total * 0.20)
        assert len(split.train_index) == total - len(split.test_index)


# --------------------------------------------------------------------------
# Stratification
# --------------------------------------------------------------------------

class TestStratification:

    def test_class_proportions_are_preserved(self, split):
        full = split.class_distribution(split.dataset.labels)
        total = sum(full.values())
        for half in ("training", "testing"):
            distribution = split.distributions()[half]
            half_total = sum(distribution.values())
            for name, count in distribution.items():
                assert count / half_total == pytest.approx(full[name] / total, abs=0.02)

    def test_an_imbalanced_dataset_stays_imbalanced_in_both_halves(self, split):
        # 40/40/20 in, 40/40/20 in each half. Stratification preserves the
        # imbalance rather than correcting it.
        assert split.distributions()["training"] == {"cat": 32, "dog": 32, "horse": 16}
        assert split.distributions()["testing"] == {"cat": 8, "dog": 8, "horse": 4}

    def test_every_class_appears_in_both_halves(self, split):
        for half in ("training", "testing"):
            assert all(count > 0 for count in split.distributions()[half].values())

    def test_all_three_distributions_are_reported(self, split):
        # Section 5 requires the full, training and testing distributions.
        assert set(split.distributions()) == {"full", "training", "testing"}


# --------------------------------------------------------------------------
# One split, reused
# --------------------------------------------------------------------------

class TestOneSplitForEveryModel:

    def test_both_representations_use_the_same_rows(self, split):
        # The classical models see features_train and the CNN sees
        # images_train. They must be the same images.
        flattened = split.images_train.reshape(len(split.images_train), -1)
        assert np.array_equal(flattened, split.features_train)

    def test_both_representations_use_the_same_test_rows(self, split):
        flattened = split.images_test.reshape(len(split.images_test), -1)
        assert np.array_equal(flattened, split.features_test)

    def test_labels_line_up_with_features(self, split):
        assert len(split.labels_train) == len(split.features_train)
        assert len(split.labels_test) == len(split.features_test)

    def test_labels_line_up_with_images(self, split):
        assert len(split.labels_train) == len(split.images_train)
        assert len(split.labels_test) == len(split.images_test)

    def test_indices_address_the_original_dataset(self, split):
        # Every view is a slice of one PreparedDataset, so a row fetched by
        # index must equal the row the split hands out.
        for position, index in enumerate(split.train_index[:5]):
            assert np.array_equal(split.features_train[position],
                                  split.dataset.features[index])


# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------

class TestReproducibility:

    def test_the_same_seed_gives_the_same_split(self):
        dataset = prepared(BALANCED)
        first, second = make_split(dataset), make_split(dataset)
        assert np.array_equal(first.train_index, second.train_index)
        assert np.array_equal(first.test_index, second.test_index)

    def test_a_different_seed_gives_a_different_split(self):
        dataset = prepared(BALANCED)
        assert not np.array_equal(
            make_split(dataset).train_index,
            make_split(dataset, random_seed=7).train_index,
        )

    def test_a_different_seed_is_still_stratified(self):
        dataset = prepared(IMBALANCED)
        split = make_split(dataset, random_seed=7)
        assert split.distributions()["testing"] == {"cat": 8, "dog": 8, "horse": 4}

    def test_the_defaults_are_the_required_ones(self, split):
        # Section 5 fixes 80/20 and seed 42.
        assert split.summary()["random_seed"] == 42
        assert split.summary()["test_size"] == 0.20

    def test_indices_are_sorted(self, split):
        # Membership is what stratification fixes; sorting makes the indices
        # stable to read, compare and serialise.
        assert list(split.train_index) == sorted(split.train_index)
        assert list(split.test_index) == sorted(split.test_index)


# --------------------------------------------------------------------------
# Refusing an impossible split
# --------------------------------------------------------------------------

class TestImpossibleSplits:

    def test_a_class_with_one_image_raises(self):
        # Section 4.1 asks for a clear error when a stratified split cannot
        # be created.
        dataset = prepared(["cat"] + ["dog"] * 20)
        with pytest.raises(ValueError, match="at least 2 images"):
            make_split(dataset)

    def test_the_error_names_the_offending_class(self):
        # scikit-learn says "the least populated class" without saying which,
        # and which is the one thing needed to fix the data.
        dataset = prepared(["cat"] + ["dog"] * 20)
        with pytest.raises(ValueError) as excinfo:
            make_split(dataset)
        assert "'cat'" in str(excinfo.value)

    def test_too_few_images_for_the_class_count_raises(self):
        dataset = prepared(["cat"] * 3 + ["dog"] * 3)
        with pytest.raises(ValueError, match="fewer than the 2 classes"):
            make_split(dataset)

    def test_the_smallest_workable_dataset_still_splits(self):
        dataset = prepared(["cat"] * 5 + ["dog"] * 5)
        split = make_split(dataset)
        assert split.distributions()["testing"] == {"cat": 1, "dog": 1}


class TestSplitSummary:

    def test_summary_reports_sample_counts(self, split):
        summary = split.summary()
        assert summary["training_samples"] == len(split.train_index)
        assert summary["testing_samples"] == len(split.test_index)

    def test_summary_values_are_json_friendly(self, split):
        # This goes into run_configuration.json, where numpy integers are not
        # serialisable.
        import json

        json.dumps(split.summary())
        json.dumps(split.distributions())

    def test_an_empty_split_is_constructible(self):
        # The dataclass defaults have to be valid on their own.
        assert len(Split(dataset=prepared(BALANCED)).train_index) == 0
