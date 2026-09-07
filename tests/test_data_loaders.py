"""Tests for the dataset loaders.

Covers the assignment's requirement that folder, CSV, JSON/JSONL and array
inputs are all accepted, and that missing files are handled rather than
crashing the run.

Two sources of test data are used deliberately:

* ``tests/fixtures`` holds committed images, for cases where realistic input
  matters - a real mix of extensions, real directory nesting.
* ``tmp_path`` builds throwaway trees, for cases where the *shape* of the
  input matters and the bytes do not - an empty class folder, a stray ``.txt``.
  Those are easier to read as five lines of setup than as another committed
  fixture, and they keep the repository small.

Because the loaders never decode images, a zero-byte file with an image
extension is a perfectly good sample as far as this module is concerned.
The synthetic tests below rely on that, which is itself a useful check that
the structural/content split holds.
"""

import pytest

from samuel_collins_cv_benchmarking.data_loader import LoadedDataset, load_folder

CLASSES = ["cat", "dog", "horse"]


def make_tree(root, layout):
    """Build a class-folder tree from ``{class_name: [filenames]}``.

    Files are created empty; only their names and extensions matter here.
    """
    for class_name, filenames in layout.items():
        class_dir = root / class_name
        class_dir.mkdir(parents=True)
        for filename in filenames:
            (class_dir / filename).touch()
    return root


# --------------------------------------------------------------------------
# Loading valid data
# --------------------------------------------------------------------------

class TestLoadFolderHappyPath:

    def test_loads_every_image(self, fixtures_dir):
        dataset = load_folder(fixtures_dir / "mini", CLASSES)
        assert len(dataset) == 15

    def test_counts_images_per_class(self, fixtures_dir):
        dataset = load_folder(fixtures_dir / "mini", CLASSES)
        assert dataset.count_by_class() == {"cat": 5, "dog": 5, "horse": 5}

    def test_clean_dataset_produces_no_warnings(self, fixtures_dir):
        dataset = load_folder(fixtures_dir / "mini", CLASSES)
        assert dataset.warnings == []

    def test_supports_every_required_extension(self, fixtures_dir):
        # Section 3.1 names PNG, JPG, JPEG, BMP and TIFF. The mini fixture
        # holds one of each, so a regression in SUPPORTED_EXTENSIONS shows up
        # as a missing extension rather than a silently smaller dataset.
        dataset = load_folder(fixtures_dir / "mini", CLASSES)
        found = {path.suffix.lower() for path, _ in dataset.samples}
        assert found == {".jpeg", ".jpg", ".png", ".bmp", ".tiff"}

    def test_samples_pair_a_path_with_its_class_name(self, fixtures_dir):
        dataset = load_folder(fixtures_dir / "mini", CLASSES)
        for path, label in dataset.samples:
            assert label in CLASSES
            # The label must match the folder the file actually came from.
            assert path.parent.name == label

    def test_class_names_follow_the_requested_order(self, fixtures_dir):
        dataset = load_folder(fixtures_dir / "mini", ["horse", "cat", "dog"])
        assert dataset.class_names == ["horse", "cat", "dog"]

    def test_can_load_a_subset_of_the_available_classes(self, fixtures_dir):
        dataset = load_folder(fixtures_dir / "mini", ["cat", "dog"])
        assert dataset.class_names == ["cat", "dog"]
        assert len(dataset) == 10

    def test_ordering_is_stable_across_calls(self, fixtures_dir):
        # A fixed seed only reproduces a split if the order being shuffled is
        # itself reproducible, so this guards the whole reproducibility story.
        first = [p for p, _ in load_folder(fixtures_dir / "mini", CLASSES).samples]
        second = [p for p, _ in load_folder(fixtures_dir / "mini", CLASSES).samples]
        assert first == second

    def test_files_come_back_in_sorted_order(self, tmp_path):
        # Created deliberately out of alphabetical sequence. Comparing two
        # calls to each other is not enough: the filesystem returns the same
        # order twice whether or not the loader sorts. This asserts the
        # loader *imposes* an order, which is what makes seed 42 reproduce
        # the same split on a different machine.
        make_tree(tmp_path, {
            "cat": ["c.jpg", "a.jpg", "b.jpg"],
            "dog": ["z.jpg", "y.jpg"],
        })
        dataset = load_folder(tmp_path, ["cat", "dog"])
        names = [path.name for path, label in dataset.samples if label == "cat"]
        assert names == ["a.jpg", "b.jpg", "c.jpg"]

    def test_accepts_a_string_path(self, fixtures_dir):
        dataset = load_folder(str(fixtures_dir / "mini"), CLASSES)
        assert len(dataset) == 15


# --------------------------------------------------------------------------
# Rejecting bad arguments
# --------------------------------------------------------------------------

class TestLoadFolderValidation:

    def test_missing_directory_raises(self, fixtures_dir):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_folder(fixtures_dir / "does_not_exist", CLASSES)

    def test_file_instead_of_directory_raises(self, fixtures_dir):
        with pytest.raises(NotADirectoryError, match="is a file"):
            load_folder(fixtures_dir / "mini_labels.csv", CLASSES)

    def test_string_target_labels_raises(self, fixtures_dir):
        # "cat" is iterable, so without an explicit guard this would look for
        # folders named 'c', 'a' and 't' and report a baffling error.
        with pytest.raises(ValueError, match="list of class-folder names"):
            load_folder(fixtures_dir / "mini", "cat")

    def test_non_sequence_target_labels_raises(self, fixtures_dir):
        with pytest.raises(ValueError, match="list of class-folder names"):
            load_folder(fixtures_dir / "mini", 42)

    def test_duplicate_class_names_raise(self, fixtures_dir):
        # A repeat would load the same folder twice, putting identical images
        # into both halves of the split. Section 7 forbids that leakage.
        with pytest.raises(ValueError, match="repeated class name"):
            load_folder(fixtures_dir / "mini", ["cat", "cat", "dog"])

    def test_duplicate_error_lists_every_repeat(self, fixtures_dir):
        with pytest.raises(ValueError) as excinfo:
            load_folder(fixtures_dir / "mini", ["cat", "dog", "cat", "dog"])
        assert "'cat'" in str(excinfo.value)
        assert "'dog'" in str(excinfo.value)

    def test_absent_class_folder_raises(self, fixtures_dir):
        with pytest.raises(ValueError, match="Class folder"):
            load_folder(fixtures_dir / "mini", ["cat", "zebra"])

    def test_absent_class_error_names_what_is_available(self, fixtures_dir):
        # The grader runs this on their own layout; the error should orient
        # them rather than just say no.
        with pytest.raises(ValueError) as excinfo:
            load_folder(fixtures_dir / "mini", ["cat", "zebra"])
        message = str(excinfo.value)
        assert "zebra" in message
        assert "'cat', 'dog', 'horse'" in message

    def test_single_class_raises(self, fixtures_dir):
        # Section 4.1: at least two classes must survive validation.
        with pytest.raises(ValueError, match="At least two classes"):
            load_folder(fixtures_dir / "mini", ["cat"])


# --------------------------------------------------------------------------
# Warning rather than failing
# --------------------------------------------------------------------------

class TestLoadFolderWarnings:

    def test_unsupported_extension_is_skipped_and_reported(self, tmp_path):
        make_tree(tmp_path, {
            "cat": ["a.jpg", "notes.txt"],
            "dog": ["b.jpg"],
        })
        dataset = load_folder(tmp_path, ["cat", "dog"])
        assert len(dataset) == 2
        assert any("notes.txt" in w for w in dataset.warnings)

    def test_hidden_files_are_ignored_without_a_warning(self, tmp_path):
        # .DS_Store appears in every macOS folder; warning about it would
        # bury the warnings that matter.
        make_tree(tmp_path, {
            "cat": ["a.jpg", ".DS_Store"],
            "dog": ["b.jpg"],
        })
        dataset = load_folder(tmp_path, ["cat", "dog"])
        assert len(dataset) == 2
        assert not any("DS_Store" in w for w in dataset.warnings)

    def test_empty_class_is_dropped_and_reported(self, tmp_path):
        make_tree(tmp_path, {
            "cat": ["a.jpg"],
            "dog": ["b.jpg"],
            "horse": [],
        })
        dataset = load_folder(tmp_path, CLASSES)
        assert dataset.class_names == ["cat", "dog"]
        assert any("horse" in w and "dropped" in w for w in dataset.warnings)

    def test_dropping_below_two_classes_raises_rather_than_warns(self, tmp_path):
        make_tree(tmp_path, {
            "cat": ["a.jpg"],
            "dog": [],
            "horse": [],
        })
        with pytest.raises(ValueError, match="At least two classes"):
            load_folder(tmp_path, CLASSES)

    def test_thin_class_warns_about_the_recommended_minimum(self, tmp_path):
        # Section 4.1 recommends five images per class.
        make_tree(tmp_path, {
            "cat": [f"c{i}.jpg" for i in range(5)],
            "dog": ["d0.jpg", "d1.jpg"],
        })
        dataset = load_folder(tmp_path, ["cat", "dog"])
        warnings = " ".join(dataset.warnings)
        assert "'dog'" in warnings and "recommended" in warnings
        assert "'cat'" not in warnings

    def test_nested_subdirectories_are_not_treated_as_images(self, tmp_path):
        make_tree(tmp_path, {"cat": ["a.jpg"], "dog": ["b.jpg"]})
        (tmp_path / "cat" / "extra").mkdir()
        dataset = load_folder(tmp_path, ["cat", "dog"])
        assert len(dataset) == 2


# --------------------------------------------------------------------------
# The contract shared by every loader
# --------------------------------------------------------------------------

class TestLoaderContract:

    def test_undecodable_file_is_still_returned_as_a_sample(self, fixtures_dir):
        # Documents the structural/content split: broken/cat/cat_corrupt.jpeg
        # is a real file with a real extension, so the loader has no basis to
        # reject it. Preprocessing catches it when the pixels fail to decode.
        dataset = load_folder(fixtures_dir / "broken", CLASSES)
        names = [path.name for path, _ in dataset.samples]
        assert "cat_corrupt.jpeg" in names

    def test_counts_reported_here_are_provisional(self, fixtures_dir):
        # Following from the above: the corrupt file is counted, so cat reads
        # as 4 even though only 3 of its images will survive preprocessing.
        # The distribution reported to the user must be computed after
        # decoding, not from this number.
        dataset = load_folder(fixtures_dir / "broken", CLASSES)
        assert dataset.count_by_class()["cat"] == 4


class TestLoadedDataset:

    def test_empty_dataset_has_no_samples(self):
        assert len(LoadedDataset()) == 0

    def test_count_by_class_starts_from_the_class_names(self):
        dataset = LoadedDataset(class_names=["cat", "dog"])
        assert dataset.count_by_class() == {"cat": 0, "dog": 0}
