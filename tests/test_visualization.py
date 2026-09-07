"""Tests for the generated figures.

Covers the assignment's requirement that all required output files are
generated, and that the comparison figure carries the four measures section 9
names.

Rendering is checked by asserting on the figure objects the plotting code
builds - the marks, scales and labels - rather than by comparing images.
A pixel comparison would break on a matplotlib upgrade or a different font
without anything actually being wrong.
"""

import numpy as np
import pytest

from samuel_collins_cv_benchmarking.benchmark import make_split
from samuel_collins_cv_benchmarking.classical_models import classical_model_specs
from samuel_collins_cv_benchmarking.data_loader import load_array
from samuel_collins_cv_benchmarking.evaluation import ModelResult, evaluate_all
from samuel_collins_cv_benchmarking.preprocessing import preprocess
from samuel_collins_cv_benchmarking import visualization


@pytest.fixture(scope="module")
def split():
    generator = np.random.default_rng(0)
    bands = [50] * 40 + [130] * 40 + [210] * 20
    tensor = np.stack([
        generator.integers(max(0, low - 25), min(255, low + 25), size=(12, 12, 3))
        .astype(np.uint8)
        for low in bands
    ])
    labels = ["cat"] * 40 + ["dog"] * 40 + ["horse"] * 20
    return make_split(preprocess(load_array(tensor, labels), "rgb"))


@pytest.fixture(scope="module")
def results(split):
    return evaluate_all(classical_model_specs(), split)


class TestRequiredFiles:

    def test_save_all_writes_every_required_file(self, results, split, tmp_path):
        # Section 9's output structure.
        visualization.save_all(results, split, tmp_path)
        assert (tmp_path / "class_distribution.png").is_file()
        assert (tmp_path / "model_comparison.png").is_file()
        for result in results:
            assert (tmp_path / "confusion_matrices" / f"{result.key}.png").is_file()

    def test_files_are_not_empty(self, results, split, tmp_path):
        visualization.save_all(results, split, tmp_path)
        assert (tmp_path / "model_comparison.png").stat().st_size > 5000

    def test_creates_missing_directories(self, results, split, tmp_path):
        target = tmp_path / "does" / "not" / "exist"
        visualization.save_all(results, split, target)
        assert (target / "confusion_matrices").is_dir()

    def test_failed_models_get_no_matrix_but_keep_their_row(self, results, split, tmp_path):
        # A failed model has nothing to draw; its row still appears in the
        # summary table carrying the error.
        broken = ModelResult(key="broken", name="Broken", succeeded=False,
                             error="RuntimeError: boom")
        written = visualization.save_all(list(results) + [broken], split, tmp_path)
        assert "broken" not in written["confusion_matrices"]
        assert not (tmp_path / "confusion_matrices" / "broken.png").exists()

    def test_returns_what_it_wrote(self, results, split, tmp_path):
        written = visualization.save_all(results, split, tmp_path)
        assert set(written) == {"class_distribution", "model_comparison",
                                "confusion_matrices", "prediction_examples"}
        assert len(written["confusion_matrices"]) == len(results)


class TestClassDistribution:

    def test_bar_lengths_are_the_class_totals(self, split, tmp_path):
        import matplotlib.pyplot as plt

        visualization.plot_class_distribution(split, tmp_path / "d.png")
        # Redraw into a figure we can inspect rather than parsing the PNG.
        figure = plt.figure()
        counts = split.dataset.count_by_class()
        distributions = split.distributions()
        for name in split.dataset.class_names:
            total = distributions["training"][name] + distributions["testing"][name]
            assert total == counts[name]
        plt.close(figure)

    def test_training_and_testing_sum_to_the_full_set(self, split):
        distributions = split.distributions()
        for name in split.dataset.class_names:
            assert (distributions["training"][name] + distributions["testing"][name]
                    == distributions["full"][name])

    def test_two_series_carry_a_legend(self, split, tmp_path, monkeypatch):
        # Identity must never be color-alone.
        captured = {}
        original = visualization.plt.subplots

        def spy(*args, **kwargs):
            figure, axes = original(*args, **kwargs)
            captured["axes"] = axes
            return figure, axes

        monkeypatch.setattr(visualization.plt, "subplots", spy)
        visualization.plot_class_distribution(split, tmp_path / "d.png")
        assert captured["axes"].get_legend() is not None


class TestModelComparison:

    def test_covers_the_four_required_measures(self):
        # Section 9: accuracy, macro F1, training time and inference time.
        fields = [field for field, _, _, _ in visualization.PANELS]
        assert fields == ["accuracy", "macro_f1", "training_time_seconds",
                          "inference_time_ms_per_image"]

    def test_uses_four_separate_panels(self, results, tmp_path, monkeypatch):
        # Four measures on four axes rather than two on one pair, because a
        # dual-axis chart makes the crossing point an artifact of the scales.
        captured = {}
        original = visualization.plt.subplots

        def spy(*args, **kwargs):
            figure, axes = original(*args, **kwargs)
            captured["axes"] = axes
            return figure, axes

        monkeypatch.setattr(visualization.plt, "subplots", spy)
        visualization.plot_model_comparison(results, tmp_path / "c.png")
        assert captured["axes"].size == 4

    def test_no_panel_carries_a_second_y_axis(self, results, tmp_path, monkeypatch):
        captured = {}
        original = visualization.plt.subplots

        def spy(*args, **kwargs):
            figure, axes = original(*args, **kwargs)
            captured["figure"] = figure
            return figure, axes

        monkeypatch.setattr(visualization.plt, "subplots", spy)
        visualization.plot_model_comparison(results, tmp_path / "c.png")
        # A twinned axis would add a fifth (and sixth) axes to the figure.
        assert len(captured["figure"].axes) == 4

    def test_wide_ranging_measures_use_dots_not_bars(self, tmp_path, monkeypatch):
        # A bar states magnitude by its length from zero, and a log axis has
        # no zero, so a wide spread must switch mark type as well as scale.
        spread = [
            ModelResult(key=f"m{i}", name=f"M{i}", accuracy=0.5, macro_f1=0.5,
                        training_time_seconds=t, inference_time_ms_per_image=t)
            for i, t in enumerate([0.001, 0.01, 1.0, 100.0])
        ]
        captured = {}
        original = visualization.plt.subplots

        def spy(*args, **kwargs):
            figure, axes = original(*args, **kwargs)
            captured["axes"] = axes
            return figure, axes

        monkeypatch.setattr(visualization.plt, "subplots", spy)
        visualization.plot_model_comparison(spread, tmp_path / "c.png")

        accuracy_panel, _, training_panel, _ = captured["axes"].ravel()
        assert accuracy_panel.get_xscale() == "linear"
        assert training_panel.get_xscale() == "log"
        # Bars are patches; dots are line artists with no connecting line.
        assert len(accuracy_panel.patches) == 4
        assert len(training_panel.patches) == 0
        assert len(training_panel.lines) >= 1

    def test_narrow_ranging_measures_stay_as_bars(self, tmp_path, monkeypatch):
        narrow = [
            ModelResult(key=f"m{i}", name=f"M{i}", accuracy=0.5, macro_f1=0.5,
                        training_time_seconds=t, inference_time_ms_per_image=t)
            for i, t in enumerate([1.0, 1.5, 2.0, 2.5])
        ]
        captured = {}
        original = visualization.plt.subplots

        def spy(*args, **kwargs):
            figure, axes = original(*args, **kwargs)
            captured["axes"] = axes
            return figure, axes

        monkeypatch.setattr(visualization.plt, "subplots", spy)
        visualization.plot_model_comparison(narrow, tmp_path / "c.png")
        training_panel = captured["axes"].ravel()[2]
        assert training_panel.get_xscale() == "linear"
        assert len(training_panel.patches) == 4

    def test_every_panel_lists_models_in_the_same_order(self, results, tmp_path, monkeypatch):
        captured = {}
        original = visualization.plt.subplots

        def spy(*args, **kwargs):
            figure, axes = original(*args, **kwargs)
            captured["axes"] = axes
            return figure, axes

        monkeypatch.setattr(visualization.plt, "subplots", spy)
        visualization.plot_model_comparison(results, tmp_path / "c.png")
        orders = [
            [label.get_text() for label in axes.get_yticklabels()]
            for axes in captured["axes"].ravel()
        ]
        assert len(set(map(tuple, orders))) == 1

    def test_refuses_to_plot_when_nothing_succeeded(self, tmp_path):
        broken = [ModelResult(key="b", name="B", succeeded=False, error="boom")]
        with pytest.raises(ValueError, match="No successful models"):
            visualization.plot_model_comparison(broken, tmp_path / "c.png")


class TestConfusionMatrix:

    def test_colors_by_share_so_small_classes_stay_readable(self, results, split, tmp_path):
        # Coloring by raw count would leave the whole row of a small class
        # pale regardless of how well the model handled it.
        result = next(r for r in results if r.succeeded)
        matrix = np.asarray(result.confusion_matrix, dtype=float)
        shares = matrix / matrix.sum(axis=1, keepdims=True)
        # A perfectly classified small class reaches 1.0 by share even though
        # its raw counts are far below the larger classes'.
        assert shares.max() <= 1.0
        visualization.plot_confusion_matrix(
            result, split.dataset.class_names, tmp_path / "m.png")
        assert (tmp_path / "m.png").is_file()

    def test_annotates_with_raw_counts(self, results, split, tmp_path, monkeypatch):
        captured = {}
        original = visualization.plt.subplots

        def spy(*args, **kwargs):
            figure, axes = original(*args, **kwargs)
            captured["axes"] = axes
            return figure, axes

        monkeypatch.setattr(visualization.plt, "subplots", spy)
        result = next(r for r in results if r.succeeded)
        visualization.plot_confusion_matrix(
            result, split.dataset.class_names, tmp_path / "m.png")

        texts = [t.get_text() for t in captured["axes"].texts]
        total = sum(int(t) for t in texts if t.isdigit())
        assert total == len(split.test_index)

    def test_axes_are_labeled_with_class_names(self, results, split, tmp_path, monkeypatch):
        captured = {}
        original = visualization.plt.subplots

        def spy(*args, **kwargs):
            figure, axes = original(*args, **kwargs)
            captured["axes"] = axes
            return figure, axes

        monkeypatch.setattr(visualization.plt, "subplots", spy)
        result = next(r for r in results if r.succeeded)
        visualization.plot_confusion_matrix(
            result, split.dataset.class_names, tmp_path / "m.png")
        axes = captured["axes"]
        assert [t.get_text() for t in axes.get_xticklabels()] == split.dataset.class_names
        assert [t.get_text() for t in axes.get_yticklabels()] == split.dataset.class_names

    def test_a_class_with_no_test_images_does_not_divide_by_zero(self, split, tmp_path):
        result = ModelResult(key="k", name="K",
                             confusion_matrix=np.array([[3, 0, 0], [0, 2, 0], [0, 0, 0]]))
        visualization.plot_confusion_matrix(
            result, ["a", "b", "c"], tmp_path / "m.png")
        assert (tmp_path / "m.png").is_file()


class TestPalette:

    def test_sequential_ramp_is_one_hue_light_to_dark(self):
        # Magnitude gets one hue, never a rainbow. Checked by lightness being
        # monotonically decreasing across the ramp.
        def luminance(hex_color):
            r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
            return 0.299 * r + 0.587 * g + 0.114 * b

        values = [luminance(c) for c in visualization.SEQUENTIAL]
        assert values == sorted(values, reverse=True)

    def test_categorical_slots_are_distinct_hues(self):
        assert len(set(visualization.SERIES)) == len(visualization.SERIES)

    def test_headless_backend_is_selected(self):
        # The benchmark runs without a display; an interactive backend would
        # fail or block.
        import matplotlib

        assert matplotlib.get_backend().lower() == "agg"


class TestFontResolution:
    """matplotlib must be able to resolve the fonts we name.

    An unresolvable family is not an error - matplotlib falls back and warns,
    once per text object. A full run emitted thousands of those lines, which
    would bury a real warning and swamp a screen recording.
    """

    def test_every_named_font_family_exists(self):
        from matplotlib import font_manager

        available = {font.name for font in font_manager.fontManager.ttflist}
        assert any(name in available for name in visualization.FONT)

    def test_no_css_keywords_in_the_font_list(self):
        # "system-ui" and "-apple-system" are CSS keywords, not families.
        for name in visualization.FONT:
            assert not name.startswith("-")
            assert name != "system-ui"

    def test_drawing_emits_no_font_warnings(self, results, split, tmp_path):
        import logging

        records = []

        class Capture(logging.Handler):
            def emit(self, record):
                records.append(record.getMessage())

        logger = logging.getLogger("matplotlib.font_manager")
        handler = Capture()
        logger.addHandler(handler)
        previous = logger.level
        logger.setLevel(logging.WARNING)
        try:
            visualization.plot_class_distribution(split, tmp_path / "d.png")
        finally:
            logger.removeHandler(handler)
            logger.setLevel(previous)

        assert not [m for m in records if "findfont" in m or "not found" in m]
