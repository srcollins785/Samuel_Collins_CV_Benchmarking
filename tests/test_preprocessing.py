"""Tests for image standardization and label encoding.

Covers the assignment's requirements that corrupted files are handled, that
grayscale output has one channel and RGB three, that every image has the same
dimensions afterwards, and that label encoding and class-name recovery stay
consistent.
"""

import numpy as np
import pytest
from PIL import Image

from samuel_collins_cv_benchmarking.data_loader import load_array, load_folder
from samuel_collins_cv_benchmarking.preprocessing import preprocess

CLASSES = ["cat", "dog", "horse"]

CORRUPT_BYTES = b"\xff\xd8\xff\xe0 this is not a valid JPEG payload"


def write_image(path, size=(20, 20), color=128, mode="RGB"):
    """Write a small real image so PIL can actually decode it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fill = (color, color, color) if mode == "RGB" else color
    Image.new(mode, size, fill).save(path)
    return path


def pair(array):
    """Wrap an array of 4 images as a two-class dataset."""
    return load_array(array, ["a", "a", "b", "b"])


# --------------------------------------------------------------------------
# Output shape and range
# --------------------------------------------------------------------------

class TestStandardizedOutput:

    @pytest.mark.parametrize("mode,channels", [("grayscale", 1), ("rgb", 3)])
    def test_channel_count_matches_color_mode(self, fixtures_dir, mode, channels):
        prepared = preprocess(load_folder(fixtures_dir / "mini", CLASSES), mode)
        assert prepared.images.shape == (15, 64, 64, channels)

    def test_every_image_has_identical_dimensions(self, fixtures_dir):
        # The mini fixture is deliberately five different sizes.
        prepared = preprocess(load_folder(fixtures_dir / "mini", CLASSES), "rgb")
        assert len({image.shape for image in prepared.images}) == 1

    def test_values_are_normalized_to_zero_one(self, fixtures_dir):
        prepared = preprocess(load_folder(fixtures_dir / "mini", CLASSES), "rgb")
        assert prepared.images.dtype == np.float32
        assert prepared.images.min() >= 0.0
        assert prepared.images.max() <= 1.0

    @pytest.mark.parametrize("mode,width", [("grayscale", 4096), ("rgb", 12288)])
    def test_feature_vector_width(self, fixtures_dir, mode, width):
        # Section 4.3: 64 x 64 = 4,096 and 64 x 64 x 3 = 12,288.
        prepared = preprocess(load_folder(fixtures_dir / "mini", CLASSES), mode)
        assert prepared.features.shape == (15, width)

    def test_features_share_memory_with_images(self, fixtures_dir):
        # The two representations section 4.3 asks for are one buffer viewed
        # two ways, not two copies of the same pixels.
        prepared = preprocess(load_folder(fixtures_dir / "mini", CLASSES), "rgb")
        assert np.shares_memory(prepared.images, prepared.features)

    def test_rejects_an_unknown_color_mode(self, fixtures_dir):
        with pytest.raises(ValueError, match="color_mode"):
            preprocess(load_folder(fixtures_dir / "mini", CLASSES), "cmyk")


class TestColorConversion:

    def test_single_channel_source_becomes_three(self):
        prepared = preprocess(pair(np.full((4, 8, 8), 200, np.uint8)), "rgb")
        assert prepared.images.shape[-1] == 3

    def test_three_channel_source_becomes_one(self):
        prepared = preprocess(pair(np.full((4, 8, 8, 3), 200, np.uint8)), "grayscale")
        assert prepared.images.shape[-1] == 1

    def test_grayscale_of_a_grey_image_keeps_its_value(self):
        prepared = preprocess(pair(np.full((4, 8, 8, 3), 128, np.uint8)), "grayscale")
        assert prepared.images.mean() == pytest.approx(128 / 255, abs=0.01)


# --------------------------------------------------------------------------
# Aspect ratio
# --------------------------------------------------------------------------

class TestLetterboxing:

    def blank_rows_and_columns(self, image):
        plane = image[:, :, 0]
        return int((plane.sum(axis=1) == 0).sum()), int((plane.sum(axis=0) == 0).sum())

    def test_square_input_needs_no_padding(self, tmp_path):
        write_image(tmp_path / "cat" / "a.png", size=(40, 40), color=200)
        write_image(tmp_path / "dog" / "b.png", size=(40, 40), color=200)
        prepared = preprocess(load_folder(tmp_path, ["cat", "dog"]), "grayscale")
        rows, columns = self.blank_rows_and_columns(prepared.images[0])
        assert (rows, columns) == (0, 0)

    def test_wide_input_is_padded_top_and_bottom(self, tmp_path):
        write_image(tmp_path / "cat" / "a.png", size=(150, 64), color=200)
        write_image(tmp_path / "dog" / "b.png", size=(150, 64), color=200)
        prepared = preprocess(load_folder(tmp_path, ["cat", "dog"]), "grayscale")
        rows, columns = self.blank_rows_and_columns(prepared.images[0])
        assert rows > 0 and columns == 0

    def test_tall_input_is_padded_left_and_right(self, tmp_path):
        write_image(tmp_path / "cat" / "a.png", size=(64, 150), color=200)
        write_image(tmp_path / "dog" / "b.png", size=(64, 150), color=200)
        prepared = preprocess(load_folder(tmp_path, ["cat", "dog"]), "grayscale")
        rows, columns = self.blank_rows_and_columns(prepared.images[0])
        assert columns > 0 and rows == 0

    def test_content_keeps_its_aspect_ratio(self, tmp_path):
        # A 2:1 image must occupy a 2:1 region of the 64x64 output rather than
        # being stretched to fill it.
        write_image(tmp_path / "cat" / "a.png", size=(120, 60), color=200)
        write_image(tmp_path / "dog" / "b.png", size=(120, 60), color=200)
        prepared = preprocess(load_folder(tmp_path, ["cat", "dog"]), "grayscale")
        plane = prepared.images[0][:, :, 0]
        filled_rows = int((plane.sum(axis=1) > 0).sum())
        filled_columns = int((plane.sum(axis=0) > 0).sum())
        assert filled_columns / filled_rows == pytest.approx(2.0, abs=0.15)

    def test_padding_is_exactly_zero(self, tmp_path):
        write_image(tmp_path / "cat" / "a.png", size=(150, 64), color=200)
        write_image(tmp_path / "dog" / "b.png", size=(150, 64), color=200)
        prepared = preprocess(load_folder(tmp_path, ["cat", "dog"]), "grayscale")
        assert prepared.images[0].min() == 0.0


# --------------------------------------------------------------------------
# Normalization of in-memory arrays
# --------------------------------------------------------------------------

class TestArrayNormalization:
    """The scale of an incoming array is ambiguous and must be inspected."""

    def test_integer_arrays_are_divided_by_255(self):
        prepared = preprocess(pair(np.full((4, 8, 8), 128, np.uint8)), "grayscale")
        assert prepared.images.mean() == pytest.approx(128 / 255, abs=0.01)

    def test_floats_already_in_zero_one_are_not_divided_again(self):
        # A second divide would compress every image to near-black - mean
        # about 0.002 instead of 0.5 - and quietly destroy accuracy.
        prepared = preprocess(pair(np.full((4, 8, 8), 0.5, np.float32)), "grayscale")
        assert prepared.images.mean() == pytest.approx(0.5, abs=0.01)

    def test_floats_on_a_zero_255_scale_are_divided(self):
        prepared = preprocess(pair(np.full((4, 8, 8), 128.0, np.float32)), "grayscale")
        assert prepared.images.mean() == pytest.approx(128 / 255, abs=0.01)

    def test_values_above_255_are_clipped_and_reported(self):
        prepared = preprocess(pair(np.full((4, 8, 8), 400.0, np.float32)), "grayscale")
        assert prepared.images.max() == pytest.approx(1.0)
        assert any("clipped" in w for w in prepared.warnings)

    def test_boolean_arrays_become_black_and_white(self):
        prepared = preprocess(pair(np.ones((4, 8, 8), bool)), "grayscale")
        assert prepared.images.max() == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Corrupt files
# --------------------------------------------------------------------------

class TestCorruptImages:

    def test_undecodable_file_is_skipped(self, fixtures_dir):
        # The loader returned this as a valid sample; a file with a real
        # extension is only revealed as corrupt when something opens it.
        prepared = preprocess(load_folder(fixtures_dir / "broken", CLASSES), "rgb")
        assert len(prepared) == 9
        assert len(prepared.skipped) == 1
        assert "cat_corrupt.jpeg" in prepared.skipped[0]

    def test_skipped_paths_are_recorded(self, fixtures_dir):
        # Section 4.1 asks for the number *and paths* of skipped samples.
        prepared = preprocess(load_folder(fixtures_dir / "broken", CLASSES), "rgb")
        assert any("could not decode" in w for w in prepared.warnings)

    def test_true_counts_differ_from_the_loader_counts(self, fixtures_dir):
        # The loader's numbers are provisional; these are the ones section 5
        # requires in the returned dictionary.
        loaded = load_folder(fixtures_dir / "broken", CLASSES)
        prepared = preprocess(loaded, "rgb")
        assert loaded.count_by_class()["cat"] == 4
        assert prepared.count_by_class()["cat"] == 3

    def test_loader_warnings_are_carried_forward(self, fixtures_dir):
        loaded = load_folder(fixtures_dir / "broken", CLASSES)
        prepared = preprocess(loaded, "rgb")
        for warning in loaded.warnings:
            assert warning in prepared.warnings

    def test_a_class_lost_to_corruption_raises(self, tmp_path):
        # If every image in a class fails to decode the class disappears here,
        # so the two-class rule has to be checked again after decoding.
        write_image(tmp_path / "cat" / "a.png", size=(20, 20))
        write_image(tmp_path / "cat" / "b.png", size=(20, 20))
        (tmp_path / "dog").mkdir()
        (tmp_path / "dog" / "bad.jpeg").write_bytes(CORRUPT_BYTES)
        loaded = load_folder(tmp_path, ["cat", "dog"])
        assert len(loaded) == 3  # the loader still sees three samples
        with pytest.raises(ValueError, match="At least two classes"):
            preprocess(loaded, "rgb")

    def test_truncated_file_is_caught(self, tmp_path):
        # Image.open() reads only the header, so a truncated file opens
        # cleanly and fails on decode. Forcing the decode here is what turns
        # that into a warning rather than a crash during training.
        write_image(tmp_path / "cat" / "a.png", size=(20, 20))
        write_image(tmp_path / "cat" / "b.png", size=(20, 20))
        write_image(tmp_path / "dog" / "c.png", size=(20, 20))
        good = write_image(tmp_path / "dog" / "d.png", size=(60, 60))
        good.write_bytes(good.read_bytes()[: len(good.read_bytes()) // 2])
        prepared = preprocess(load_folder(tmp_path, ["cat", "dog"]), "rgb")
        assert len(prepared) == 3
        assert len(prepared.skipped) == 1


# --------------------------------------------------------------------------
# Label encoding
# --------------------------------------------------------------------------

class TestLabelEncoding:

    def test_labels_are_integers_from_zero(self, fixtures_dir):
        prepared = preprocess(load_folder(fixtures_dir / "mini", CLASSES), "rgb")
        assert prepared.labels.dtype == np.int64
        assert set(prepared.labels) == {0, 1, 2}

    def test_class_names_are_sorted(self, fixtures_dir):
        prepared = preprocess(load_folder(fixtures_dir / "mini", CLASSES), "rgb")
        assert prepared.class_names == sorted(prepared.class_names)

    def test_encoding_matches_sklearn_label_encoder(self, fixtures_dir):
        # Everything downstream reports per-class results by index. If our
        # order and LabelEncoder's ever diverged, every confusion matrix and
        # classification report would carry the wrong class names.
        from sklearn.preprocessing import LabelEncoder

        loaded = load_folder(fixtures_dir / "mini", CLASSES)
        prepared = preprocess(loaded, "rgb")
        raw = [label for _, label in loaded.samples]

        encoder = LabelEncoder().fit(raw)
        assert list(encoder.classes_) == prepared.class_names
        assert list(encoder.transform(raw)) == list(prepared.labels)

    def test_class_name_is_recoverable_from_its_integer(self, fixtures_dir):
        loaded = load_folder(fixtures_dir / "mini", CLASSES)
        prepared = preprocess(loaded, "rgb")
        for (_, original), encoded in zip(loaded.samples, prepared.labels):
            assert prepared.class_names[encoded] == original

    def test_counts_by_class_use_the_recovered_names(self, fixtures_dir):
        prepared = preprocess(load_folder(fixtures_dir / "mini", CLASSES), "rgb")
        assert prepared.count_by_class() == {"cat": 5, "dog": 5, "horse": 5}

    def test_labels_and_images_stay_aligned_after_skips(self, fixtures_dir):
        prepared = preprocess(load_folder(fixtures_dir / "broken", CLASSES), "rgb")
        assert len(prepared.images) == len(prepared.labels)
        assert len(prepared) == 9
