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

import json

import numpy as np
import pytest

from samuel_collins_cv_benchmarking.data_loader import (
    LoadedDataset,
    load_csv,
    load_folder,
    load_json,
    load_array,
)

CLASSES = ["cat", "dog", "horse"]


def write_jsonl(path, rows, label_field="class_name"):
    """Write a JSONL manifest from ``[(path, label), ...]``."""
    lines = [json.dumps({"image_path": image, label_field: label}) for image, label in rows]
    path.write_text("\n".join(lines) + "\n")
    return path


def write_json(path, rows, label_field="class_name"):
    """Write a JSON array manifest from ``[(path, label), ...]``."""
    records = [{"image_path": image, label_field: label} for image, label in rows]
    path.write_text(json.dumps(records, indent=2))
    return path


def write_csv(path, rows, header="image_path,class_name"):
    """Write a manifest from ``[(path, label), ...]`` and return its path."""
    lines = [header] + [f"{image},{label}" for image, label in rows]
    path.write_text("\n".join(lines) + "\n")
    return path


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


# --------------------------------------------------------------------------
# CSV manifests
# --------------------------------------------------------------------------

class TestLoadCsvHappyPath:

    def test_loads_every_row(self, fixtures_dir):
        dataset = load_csv(fixtures_dir / "mini_labels.csv", "class_name")
        assert len(dataset) == 15

    def test_counts_images_per_class(self, fixtures_dir):
        dataset = load_csv(fixtures_dir / "mini_labels.csv", "class_name")
        assert dataset.count_by_class() == {"cat": 5, "dog": 5, "horse": 5}

    def test_clean_manifest_produces_no_warnings(self, fixtures_dir):
        dataset = load_csv(fixtures_dir / "mini_labels.csv", "class_name")
        assert dataset.warnings == []

    def test_class_names_are_sorted_not_in_row_order(self, tmp_path):
        # scikit-learn's LabelEncoder assigns integers in sorted order. If
        # class_names used row order instead, class_names[i] would name a
        # different class than the models mean by i, and every confusion
        # matrix would be mislabelled.
        make_tree(tmp_path, {"cat": ["a.jpg"], "dog": ["b.jpg"], "horse": ["c.jpg"]})
        manifest = write_csv(tmp_path / "m.csv", [
            ("horse/c.jpg", "horse"),
            ("cat/a.jpg", "cat"),
            ("dog/b.jpg", "dog"),
        ])
        dataset = load_csv(manifest, "class_name")
        assert dataset.class_names == ["cat", "dog", "horse"]

    def test_relative_paths_resolve_from_the_manifest_directory(self, tmp_path):
        nested = tmp_path / "somewhere" / "deep"
        make_tree(nested, {"cat": ["a.jpg", "b.jpg"], "dog": ["c.jpg", "d.jpg"]})
        manifest = write_csv(nested / "m.csv", [
            ("cat/a.jpg", "cat"), ("cat/b.jpg", "cat"),
            ("dog/c.jpg", "dog"), ("dog/d.jpg", "dog"),
        ])
        dataset = load_csv(manifest, "class_name")
        assert len(dataset) == 4
        assert all(path.exists() for path, _ in dataset.samples)

    def test_absolute_paths_are_used_as_given(self, tmp_path):
        make_tree(tmp_path, {"cat": ["a.jpg"], "dog": ["b.jpg"]})
        manifest = write_csv(tmp_path / "m.csv", [
            (str(tmp_path / "cat" / "a.jpg"), "cat"),
            (str(tmp_path / "dog" / "b.jpg"), "dog"),
        ])
        assert len(load_csv(manifest, "class_name")) == 2

    def test_result_does_not_depend_on_the_working_directory(self, fixtures_dir, monkeypatch, tmp_path):
        # Section 13 forbids hard-coded machine-specific paths; the grader will
        # run this from a directory we have never seen.
        manifest = (fixtures_dir / "mini_labels.csv").resolve()
        monkeypatch.chdir(tmp_path)
        assert len(load_csv(manifest, "class_name")) == 15

    def test_numeric_looking_labels_stay_strings(self, tmp_path):
        # Without dtype=str pandas would read these as int64 and break the
        # string-label contract the other loaders follow.
        make_tree(tmp_path, {"1": ["a.jpg"], "2": ["b.jpg"]})
        manifest = write_csv(tmp_path / "m.csv", [("1/a.jpg", "1"), ("2/b.jpg", "2")])
        dataset = load_csv(manifest, "class_name")
        assert dataset.class_names == ["1", "2"]
        assert all(isinstance(label, str) for _, label in dataset.samples)


class TestLoadCsvValidation:

    def test_missing_manifest_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_csv(tmp_path / "nope.csv", "class_name")

    def test_directory_instead_of_file_raises(self, fixtures_dir):
        with pytest.raises(IsADirectoryError, match="is a directory"):
            load_csv(fixtures_dir / "mini", "class_name")

    def test_list_target_labels_raises(self, fixtures_dir):
        # The mirror image of the folder loader: here a list is the mistake
        # and a string is correct.
        with pytest.raises(ValueError, match="name of the label column"):
            load_csv(fixtures_dir / "mini_labels.csv", ["cat", "dog"])

    def test_empty_file_raises(self, tmp_path):
        empty = tmp_path / "empty.csv"
        empty.write_text("")
        with pytest.raises(ValueError, match="empty"):
            load_csv(empty, "class_name")

    def test_missing_image_path_column_raises(self, tmp_path):
        manifest = write_csv(tmp_path / "m.csv", [("a.jpg", "cat")], header="file,class_name")
        with pytest.raises(ValueError, match="no 'image_path' column"):
            load_csv(manifest, "class_name")

    def test_missing_label_column_raises(self, tmp_path):
        manifest = write_csv(tmp_path / "m.csv", [("a.jpg", "cat")], header="image_path,label")
        with pytest.raises(ValueError, match="no 'class_name' column"):
            load_csv(manifest, "class_name")

    def test_column_error_names_the_columns_present(self, tmp_path):
        manifest = write_csv(tmp_path / "m.csv", [("a.jpg", "cat")], header="file,label")
        with pytest.raises(ValueError) as excinfo:
            load_csv(manifest, "class_name")
        assert "'file', 'label'" in str(excinfo.value)

    def test_header_only_manifest_raises(self, tmp_path):
        manifest = write_csv(tmp_path / "m.csv", [])
        with pytest.raises(ValueError, match="At least two classes"):
            load_csv(manifest, "class_name")

    def test_single_class_raises(self, tmp_path):
        make_tree(tmp_path, {"cat": ["a.jpg"]})
        manifest = write_csv(tmp_path / "m.csv", [("cat/a.jpg", "cat")])
        with pytest.raises(ValueError, match="At least two classes"):
            load_csv(manifest, "class_name")


class TestLoadCsvWarnings:

    def test_row_naming_an_absent_file_is_skipped(self, tmp_path):
        # The one structural failure a class-folder dataset cannot produce.
        make_tree(tmp_path, {"cat": ["a.jpg"], "dog": ["b.jpg"]})
        manifest = write_csv(tmp_path / "m.csv", [
            ("cat/a.jpg", "cat"), ("dog/b.jpg", "dog"), ("cat/gone.jpg", "cat"),
        ])
        dataset = load_csv(manifest, "class_name")
        assert len(dataset) == 2
        assert any("not found" in w and "gone.jpg" in w for w in dataset.warnings)

    def test_duplicate_rows_are_skipped(self, tmp_path):
        # The same image twice could land in both halves of the split.
        make_tree(tmp_path, {"cat": ["a.jpg"], "dog": ["b.jpg"]})
        manifest = write_csv(tmp_path / "m.csv", [
            ("cat/a.jpg", "cat"), ("cat/a.jpg", "cat"), ("dog/b.jpg", "dog"),
        ])
        dataset = load_csv(manifest, "class_name")
        assert len(dataset) == 2
        assert any("duplicate" in w for w in dataset.warnings)

    def test_blank_image_path_is_skipped(self, tmp_path):
        make_tree(tmp_path, {"cat": ["a.jpg"], "dog": ["b.jpg"]})
        manifest = write_csv(tmp_path / "m.csv", [
            ("cat/a.jpg", "cat"), ("", "cat"), ("dog/b.jpg", "dog"),
        ])
        dataset = load_csv(manifest, "class_name")
        assert len(dataset) == 2
        assert any("blank image_path" in w for w in dataset.warnings)

    def test_blank_label_is_skipped(self, tmp_path):
        make_tree(tmp_path, {"cat": ["a.jpg", "c.jpg"], "dog": ["b.jpg"]})
        manifest = write_csv(tmp_path / "m.csv", [
            ("cat/a.jpg", "cat"), ("cat/c.jpg", ""), ("dog/b.jpg", "dog"),
        ])
        dataset = load_csv(manifest, "class_name")
        assert len(dataset) == 2
        assert any("blank label" in w for w in dataset.warnings)

    def test_unsupported_extension_is_skipped(self, tmp_path):
        make_tree(tmp_path, {"cat": ["a.jpg", "notes.txt"], "dog": ["b.jpg"]})
        manifest = write_csv(tmp_path / "m.csv", [
            ("cat/a.jpg", "cat"), ("cat/notes.txt", "cat"), ("dog/b.jpg", "dog"),
        ])
        dataset = load_csv(manifest, "class_name")
        assert len(dataset) == 2
        assert any("unsupported extension" in w for w in dataset.warnings)

    def test_warning_names_the_spreadsheet_row_number(self, tmp_path):
        # Row 2 is the first data row, counting the header as row 1, so the
        # number matches what a spreadsheet shows.
        make_tree(tmp_path, {"cat": ["a.jpg"], "dog": ["b.jpg"]})
        manifest = write_csv(tmp_path / "m.csv", [
            ("cat/a.jpg", "cat"), ("cat/gone.jpg", "cat"), ("dog/b.jpg", "dog"),
        ])
        dataset = load_csv(manifest, "class_name")
        assert any("row 3" in w for w in dataset.warnings)


class TestLoadersAgree:

    def test_folder_and_csv_return_the_same_samples(self, fixtures_dir):
        # The whole point of the shared contract: two organizations describing
        # the same images must produce the same dataset.
        from_folder = load_folder(fixtures_dir / "mini", CLASSES)
        from_csv = load_csv(fixtures_dir / "mini_labels.csv", "class_name")
        assert sorted(p.resolve() for p, _ in from_folder.samples) == \
               sorted(p.resolve() for p, _ in from_csv.samples)
        assert from_folder.count_by_class() == from_csv.count_by_class()


# --------------------------------------------------------------------------
# JSON and JSONL manifests
# --------------------------------------------------------------------------

class TestLoadJsonHappyPath:

    def test_loads_a_json_array(self, fixtures_dir):
        dataset = load_json(fixtures_dir / "mini_labels.json", "class_name")
        assert len(dataset) == 15

    def test_loads_a_jsonl_file(self, fixtures_dir):
        dataset = load_json(fixtures_dir / "mini_labels.jsonl", "class_name")
        assert len(dataset) == 15

    def test_both_forms_agree(self, fixtures_dir):
        as_json = load_json(fixtures_dir / "mini_labels.json", "class_name")
        as_jsonl = load_json(fixtures_dir / "mini_labels.jsonl", "class_name")
        assert as_json.samples == as_jsonl.samples

    def test_clean_manifest_produces_no_warnings(self, fixtures_dir):
        dataset = load_json(fixtures_dir / "mini_labels.json", "class_name")
        assert dataset.warnings == []

    def test_class_names_are_sorted(self, tmp_path):
        make_tree(tmp_path, {"cat": ["a.jpg"], "dog": ["b.jpg"], "horse": ["c.jpg"]})
        manifest = write_json(tmp_path / "m.json", [
            ("horse/c.jpg", "horse"), ("cat/a.jpg", "cat"), ("dog/b.jpg", "dog"),
        ])
        assert load_json(manifest, "class_name").class_names == ["cat", "dog", "horse"]

    def test_relative_paths_resolve_from_the_manifest_directory(self, tmp_path):
        nested = tmp_path / "deep"
        make_tree(nested, {"cat": ["a.jpg"], "dog": ["b.jpg"]})
        manifest = write_json(nested / "m.json", [("cat/a.jpg", "cat"), ("dog/b.jpg", "dog")])
        dataset = load_json(manifest, "class_name")
        assert len(dataset) == 2
        assert all(path.exists() for path, _ in dataset.samples)

    def test_blank_lines_in_jsonl_are_ignored(self, tmp_path):
        make_tree(tmp_path, {"cat": ["a.jpg"], "dog": ["b.jpg"]})
        manifest = tmp_path / "m.jsonl"
        manifest.write_text(
            '{"image_path": "cat/a.jpg", "class_name": "cat"}\n'
            "\n"
            '{"image_path": "dog/b.jpg", "class_name": "dog"}\n'
            "\n"
        )
        dataset = load_json(manifest, "class_name")
        assert len(dataset) == 2
        # Only the small-class advisory; nothing about parsing or skipped rows.
        assert not any("skipped" in w or "parse" in w for w in dataset.warnings)


class TestJsonFormatDetection:
    """Extension chooses the parser; content decides if that guess was wrong."""

    ARRAY = '[{"image_path": "cat/a.jpg", "class_name": "cat"}, ' \
            '{"image_path": "dog/b.jpg", "class_name": "dog"}]'
    LINES = '{"image_path": "cat/a.jpg", "class_name": "cat"}\n' \
            '{"image_path": "dog/b.jpg", "class_name": "dog"}\n'

    @pytest.fixture
    def images(self, tmp_path):
        make_tree(tmp_path, {"cat": ["a.jpg"], "dog": ["b.jpg"]})
        return tmp_path

    def test_matching_extensions_load_without_warning(self, images):
        for name, text in (("m.json", self.ARRAY), ("m.jsonl", self.LINES)):
            manifest = images / name
            manifest.write_text(text)
            dataset = load_json(manifest, "class_name")
            assert len(dataset) == 2
            assert not any("parse" in w for w in dataset.warnings)

    def test_jsonl_content_in_a_json_file_still_loads(self, images):
        manifest = images / "m.json"
        manifest.write_text(self.LINES)
        dataset = load_json(manifest, "class_name")
        assert len(dataset) == 2
        assert any("loaded as JSONL" in w for w in dataset.warnings)

    def test_array_content_in_a_jsonl_file_still_loads(self, images):
        # A JSON array on one line parses as a single JSONL record, so without
        # requiring each line to be an object this would silently yield zero
        # samples rather than falling back.
        manifest = images / "m.jsonl"
        manifest.write_text(self.ARRAY)
        dataset = load_json(manifest, "class_name")
        assert len(dataset) == 2
        assert any("loaded as JSON" in w for w in dataset.warnings)

    def test_a_bare_object_is_rejected_as_not_a_list(self, images):
        # Section 3.3: a JSON manifest holds a *list* of records. A single
        # object spread over several lines parses as neither form, and the
        # error should say which shape was expected rather than crash on an
        # index into a dict.
        manifest = images / "m.json"
        manifest.write_text('{\n  "image_path": "cat/a.jpg",\n  "class_name": "cat"\n}')
        with pytest.raises(ValueError, match="list of records"):
            load_json(manifest, "class_name")

    def test_a_dict_of_records_is_rejected(self, images):
        manifest = images / "m.json"
        manifest.write_text(
            '{\n  "first": {"image_path": "cat/a.jpg", "class_name": "cat"}\n}')
        with pytest.raises(ValueError, match="list of records"):
            load_json(manifest, "class_name")

    def test_unknown_extension_is_sniffed(self, images):
        manifest = images / "m.txt"
        manifest.write_text(self.LINES)
        assert len(load_json(manifest, "class_name")) == 2

    def test_neither_format_raises_naming_both_attempts(self, images):
        manifest = images / "m.json"
        manifest.write_text("this is not JSON at all")
        with pytest.raises(ValueError) as excinfo:
            load_json(manifest, "class_name")
        assert "as JSON" in str(excinfo.value) and "as JSONL" in str(excinfo.value)


class TestLoadJsonValidation:

    def test_missing_manifest_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_json(tmp_path / "nope.json", "class_name")

    def test_directory_instead_of_file_raises(self, fixtures_dir):
        with pytest.raises(IsADirectoryError, match="is a directory"):
            load_json(fixtures_dir / "mini", "class_name")

    def test_list_target_labels_raises(self, fixtures_dir):
        with pytest.raises(ValueError, match="name of the label field"):
            load_json(fixtures_dir / "mini_labels.json", ["cat", "dog"])

    def test_empty_file_raises(self, tmp_path):
        manifest = tmp_path / "m.json"
        manifest.write_text("   \n")
        with pytest.raises(ValueError, match="empty"):
            load_json(manifest, "class_name")

    def test_missing_image_path_field_raises(self, tmp_path):
        manifest = tmp_path / "m.json"
        manifest.write_text(json.dumps([{"file": "a.jpg", "class_name": "cat"}]))
        with pytest.raises(ValueError, match="image_path"):
            load_json(manifest, "class_name")

    def test_field_error_names_the_fields_present(self, tmp_path):
        manifest = tmp_path / "m.json"
        manifest.write_text(json.dumps([{"file": "a.jpg", "label": "cat"}]))
        with pytest.raises(ValueError) as excinfo:
            load_json(manifest, "class_name")
        assert "'file', 'label'" in str(excinfo.value)

    def test_single_class_raises(self, tmp_path):
        make_tree(tmp_path, {"cat": ["a.jpg"]})
        manifest = write_json(tmp_path / "m.json", [("cat/a.jpg", "cat")])
        with pytest.raises(ValueError, match="At least two classes"):
            load_json(manifest, "class_name")


class TestLoadJsonWarnings:

    def test_record_naming_an_absent_file_is_skipped(self, tmp_path):
        make_tree(tmp_path, {"cat": ["a.jpg"], "dog": ["b.jpg"]})
        manifest = write_json(tmp_path / "m.json", [
            ("cat/a.jpg", "cat"), ("dog/b.jpg", "dog"), ("cat/gone.jpg", "cat"),
        ])
        dataset = load_json(manifest, "class_name")
        assert len(dataset) == 2
        assert any("not found" in w for w in dataset.warnings)

    def test_non_object_entries_are_skipped(self, tmp_path):
        make_tree(tmp_path, {"cat": ["a.jpg"], "dog": ["b.jpg"]})
        manifest = tmp_path / "m.json"
        manifest.write_text(json.dumps([
            {"image_path": "cat/a.jpg", "class_name": "cat"},
            "not a record",
            {"image_path": "dog/b.jpg", "class_name": "dog"},
        ]))
        dataset = load_json(manifest, "class_name")
        assert len(dataset) == 2
        assert any("not an object" in w for w in dataset.warnings)

    def test_json_warnings_are_numbered_by_record(self, tmp_path):
        make_tree(tmp_path, {"cat": ["a.jpg"], "dog": ["b.jpg"]})
        manifest = write_json(tmp_path / "m.json", [
            ("cat/a.jpg", "cat"), ("cat/gone.jpg", "cat"), ("dog/b.jpg", "dog"),
        ])
        dataset = load_json(manifest, "class_name")
        assert any("record 2" in w for w in dataset.warnings)

    def test_jsonl_warnings_are_numbered_by_line(self, tmp_path):
        # A text editor shows line numbers, so JSONL warnings use them.
        make_tree(tmp_path, {"cat": ["a.jpg"], "dog": ["b.jpg"]})
        manifest = write_jsonl(tmp_path / "m.jsonl", [
            ("cat/a.jpg", "cat"), ("cat/gone.jpg", "cat"), ("dog/b.jpg", "dog"),
        ])
        dataset = load_json(manifest, "class_name")
        assert any("line 2" in w for w in dataset.warnings)


class TestAllFourOrganizationsAgree:

    def test_folder_csv_json_and_jsonl_return_the_same_samples(self, fixtures_dir):
        # Section 3: the package changes how it reads the data, not what the
        # data is. Three of the four organizations are covered here; array
        # input is checked separately once that loader exists.
        results = {
            "folder": load_folder(fixtures_dir / "mini", CLASSES),
            "csv": load_csv(fixtures_dir / "mini_labels.csv", "class_name"),
            "json": load_json(fixtures_dir / "mini_labels.json", "class_name"),
            "jsonl": load_json(fixtures_dir / "mini_labels.jsonl", "class_name"),
        }
        expected = sorted(p.resolve() for p, _ in results["folder"].samples)
        for name, dataset in results.items():
            assert sorted(p.resolve() for p, _ in dataset.samples) == expected, name
            assert dataset.count_by_class() == {"cat": 5, "dog": 5, "horse": 5}, name

    def test_array_input_describes_the_same_dataset(self, fixtures_dir):
        # Array input cannot reference the same *files* - it carries pixels -
        # so equivalence is checked at the level of labels and counts, which
        # is what every downstream stage actually consumes.
        from PIL import Image

        from_folder = load_folder(fixtures_dir / "mini", CLASSES)
        # The fixture images are deliberately different sizes, so they have to
        # be brought to a common shape before they can stack into one tensor.
        tensor = np.stack([
            np.asarray(Image.open(path).convert("RGB").resize((16, 16)))
            for path, _ in from_folder.samples
        ])
        labels = [label for _, label in from_folder.samples]

        from_array = load_array(tensor, labels)
        assert from_array.class_names == from_folder.class_names
        assert from_array.count_by_class() == from_folder.count_by_class()
        assert len(from_array) == len(from_folder)


# --------------------------------------------------------------------------
# NumPy array / in-memory input
# --------------------------------------------------------------------------

def images(count, *shape, dtype=np.uint8):
    """A tensor of `count` blank images with the given per-image shape."""
    return np.zeros((count, *shape), dtype=dtype)


LABELS_6 = ["cat", "cat", "cat", "dog", "dog", "dog"]


class TestLoadArrayHappyPath:

    @pytest.mark.parametrize("shape", [(8, 8), (8, 8, 1), (8, 8, 3)])
    def test_accepts_every_documented_shape(self, shape):
        # Section 3.4 lists (N,H,W), (N,H,W,1) and (N,H,W,3).
        dataset = load_array(images(6, *shape), LABELS_6)
        assert len(dataset) == 6
        assert dataset.class_names == ["cat", "dog"]

    def test_samples_carry_arrays_not_paths(self, tmp_path):
        # The widened Sample type: array input has no file to open, so the
        # source is the pixel array itself.
        dataset = load_array(images(4, 4, 4), ["a", "a", "b", "b"])
        assert all(isinstance(source, np.ndarray) for source, _ in dataset.samples)

    def test_pixel_values_are_left_untouched(self):
        # Preprocessing decides how to normalise by looking at the dtype, so
        # an integer 0-255 array must arrive unchanged.
        raw = np.array([[[10, 200]], [[30, 40]], [[50, 60]], [[70, 80]]], dtype=np.uint8)
        dataset = load_array(raw, ["a", "a", "b", "b"])
        assert dataset.samples[0][0].dtype == np.uint8
        assert list(dataset.samples[0][0].ravel()) == [10, 200]

    def test_accepts_a_nested_list(self):
        data = [[[0, 0], [0, 0]] for _ in range(4)]
        assert len(load_array(data, ["a", "a", "b", "b"])) == 4

    def test_accepts_a_numpy_label_vector(self):
        dataset = load_array(images(4, 4, 4), np.array(["cat", "cat", "dog", "dog"]))
        assert dataset.class_names == ["cat", "dog"]

    def test_counts_images_per_class(self):
        dataset = load_array(images(6, 4, 4), LABELS_6)
        assert dataset.count_by_class() == {"cat": 3, "dog": 3}


class TestNumericLabelPadding:
    """Numeric labels must sort numerically once turned into class names."""

    def test_single_digit_labels_are_not_padded(self):
        dataset = load_array(images(6, 4, 4), [0, 0, 0, 1, 1, 1])
        assert dataset.class_names == ["0", "1"]

    def test_double_digit_labels_are_padded(self):
        # Without padding these sort as "0","1","10","2",..., so class_names[1]
        # would name "10" while the models mean 1, mislabelling every
        # confusion matrix in a way that still looks plausible.
        labels = [i % 11 for i in range(22)]
        dataset = load_array(images(22, 4, 4), labels)
        assert dataset.class_names == [f"{i:02d}" for i in range(11)]

    def test_padding_matches_numeric_order(self):
        labels = [i % 11 for i in range(22)]
        dataset = load_array(images(22, 4, 4), labels)
        assert dataset.class_names == sorted(dataset.class_names)
        assert [int(name) for name in dataset.class_names] == list(range(11))

    def test_padding_is_reported(self):
        labels = [i % 11 for i in range(22)]
        dataset = load_array(images(22, 4, 4), labels)
        assert any("zero-padded" in w for w in dataset.warnings)

    def test_whole_floats_are_treated_as_numeric(self):
        dataset = load_array(images(4, 4, 4), [0.0, 0.0, 1.0, 1.0])
        assert dataset.class_names == ["0", "1"]

    def test_string_labels_are_left_alone(self):
        dataset = load_array(images(4, 4, 4), ["cat", "cat", "dog", "dog"])
        assert dataset.class_names == ["cat", "dog"]
        assert not any("zero-padded" in w for w in dataset.warnings)

    def test_negative_labels_fall_back_to_plain_strings(self):
        dataset = load_array(images(4, 4, 4), [-1, -1, 2, 2])
        assert dataset.class_names == ["-1", "2"]


class TestLoadArrayValidation:

    def test_non_array_raises(self):
        with pytest.raises(TypeError, match="NumPy image array"):
            load_array("some/path", ["a", "b"])

    def test_two_dimensional_array_raises(self):
        with pytest.raises(ValueError, match="3- or 4-dimensional"):
            load_array(np.zeros((4, 4)), ["a", "b"])

    def test_five_dimensional_array_raises(self):
        with pytest.raises(ValueError, match="3- or 4-dimensional"):
            load_array(np.zeros((2, 4, 4, 3, 1)), ["a", "b"])

    def test_unsupported_channel_count_raises(self):
        with pytest.raises(ValueError, match="1 or 3 channels"):
            load_array(np.zeros((4, 8, 8, 4)), ["a", "a", "b", "b"])

    def test_empty_array_raises(self):
        with pytest.raises(ValueError, match="empty"):
            load_array(np.zeros((0, 4, 4)), [])

    def test_none_labels_raise(self):
        with pytest.raises(ValueError, match="label vector"):
            load_array(images(4, 4, 4), None)

    def test_string_labels_raise(self):
        # The mirror of the folder loader again: here a single name is wrong
        # because one label per image is required.
        with pytest.raises(ValueError, match="one entry per image"):
            load_array(images(4, 4, 4), "cat")

    def test_two_dimensional_labels_raise(self):
        with pytest.raises(ValueError, match="one-dimensional"):
            load_array(images(4, 4, 4), np.zeros((2, 2)))

    @pytest.mark.parametrize("labels", [["a", "b", "c"], ["a", "b", "c", "d", "e"]])
    def test_label_count_must_match_image_count(self, labels):
        # Section 4.1. Zipping to the shorter of the two would mislabel every
        # image after the mismatch instead of failing.
        with pytest.raises(ValueError, match="does not match number of images"):
            load_array(images(4, 4, 4), labels)

    def test_ragged_nested_list_raises(self):
        with pytest.raises(ValueError, match="Could not read dataset"):
            load_array([[[1, 2]], [[1, 2, 3]]], ["a", "b"])

    def test_single_class_raises(self):
        with pytest.raises(ValueError, match="At least two classes"):
            load_array(images(4, 4, 4), ["a", "a", "a", "a"])


class TestLoadArrayWarnings:

    def test_non_finite_images_are_skipped(self):
        # The array equivalent of an undecodable file: NaN would propagate
        # through scaling and training into every reported metric.
        data = np.zeros((6, 4, 4), dtype=np.float32)
        data[2, 0, 0] = np.nan
        data[4] = np.inf
        dataset = load_array(data, LABELS_6)
        assert len(dataset) == 4
        assert sum("NaN or infinity" in w for w in dataset.warnings) == 2

    def test_integer_arrays_are_not_checked_for_nan(self):
        # np.isfinite is meaningless on integer dtypes; they cannot hold NaN.
        dataset = load_array(images(6, 4, 4), LABELS_6)
        assert len(dataset) == 6
        assert not any("NaN" in w for w in dataset.warnings)

    def test_blank_labels_are_skipped(self):
        dataset = load_array(images(6, 4, 4), ["cat", "cat", None, "dog", "dog", ""])
        assert len(dataset) == 4
        assert sum("blank label" in w for w in dataset.warnings) == 2

    def test_nan_labels_are_skipped(self):
        dataset = load_array(
            images(6, 4, 4), ["cat", "cat", float("nan"), "dog", "dog", "dog"])
        assert len(dataset) == 5
        assert any("blank label" in w for w in dataset.warnings)
