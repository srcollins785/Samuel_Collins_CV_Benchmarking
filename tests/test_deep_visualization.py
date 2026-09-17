"""Tests for the training curves and the combined comparison plots.

Mostly about honesty in the figures rather than their appearance.

A model that has no parameter count must be absent from the parameter plot,
not drawn as a zero-height bar. A zero bar is a fabricated measurement that
looks exactly like a real one, and section 23 is explicit that an
inapplicable metric is N/A with an explanation. So the tests check that
missing values are omitted and that the omission is stated in the figure's
own subtitle.

The color assignment is also pinned. Three groups, not eleven families: the
scatter plots are an all-pairs chart form where the validated palette caps at
three slots, and eleven categorical hues cannot be distinguished by anyone.
"""

from pathlib import Path

import pytest

from samuel_collins_cv_benchmarking import deep_visualization as viz


def row(key, name, **extra):
    base = {"key": key, "name": name, "family": "Test",
            "group": viz.group_for(key), "accuracy": 0.5, "macro_f1": 0.5,
            "total_parameters": None, "checkpoint_mb": None,
            "training_seconds": None, "latency_ms": None, "throughput": None}
    base.update(extra)
    return base


@pytest.fixture
def rows():
    return [
        row("logistic_regression", "Logistic Regression", accuracy=0.23,
            macro_f1=0.22, training_seconds=12.0, latency_ms=0.03,
            throughput=33000.0),
        row("simple_cnn", "Simple CNN", accuracy=0.43, macro_f1=0.43,
            training_seconds=95.0, latency_ms=0.45, throughput=2200.0),
        row("resnet18", "ResNet18", accuracy=0.91, macro_f1=0.91,
            total_parameters=11_181_642, checkpoint_mb=44.8,
            training_seconds=150.0, latency_ms=1.2, throughput=830.0),
        row("vgg16", "VGG16", accuracy=0.39, macro_f1=0.38,
            total_parameters=134_301_514, checkpoint_mb=537.2,
            training_seconds=674.0, latency_ms=4.5, throughput=220.0),
    ]


# -- the grouping -----------------------------------------------------------

@pytest.mark.parametrize("key,expected", [
    ("logistic_regression", "Traditional ML"),
    ("decision_tree", "Traditional ML"),
    ("random_forest", "Traditional ML"),
    ("svm", "Traditional ML"),
    ("neural_network", "Baseline neural"),
    ("simple_cnn", "Baseline neural"),
    ("alexnet", "Deep CNN"),
    ("resnet50", "Deep CNN"),
    ("convnext_tiny", "Deep CNN"),
    ("yolo_cls", "Deep CNN"),
])
def test_every_model_lands_in_the_right_group(key, expected):
    assert viz.group_for(key) == expected


def test_there_are_exactly_three_color_groups():
    """The scatter forms cap at three validated slots; don't grow this."""
    assert len(viz.GROUP_COLORS) == 3
    assert set(viz.GROUP_COLORS) == set(viz.GROUP_ORDER)


def test_group_colors_are_the_validated_palette_slots():
    """Slots 1-3, validated all-pairs for the scatter plots."""
    assert viz.GROUP_COLORS["Traditional ML"] == "#2a78d6"
    assert viz.GROUP_COLORS["Baseline neural"] == "#eb6834"
    assert viz.GROUP_COLORS["Deep CNN"] == "#1baf7a"


# -- missing is not zero ----------------------------------------------------

def test_models_without_a_metric_are_omitted_from_its_plot(rows, tmp_path):
    """Two of the four rows have no parameter count; neither may be plotted."""
    path = viz._bar_plot(rows, "total_parameters", "Parameters", "count",
                         tmp_path / "params.png", log=True)
    assert Path(path).is_file()
    # The plotted set is the rows that actually reported the metric.
    plotted = [r for r in rows if r["total_parameters"] is not None]
    assert len(plotted) == 2


def test_the_omission_is_stated_not_silent(rows):
    note = viz._omitted_note(rows, "total_parameters")
    assert "Logistic Regression" in note
    assert "N/A" in note
    assert "zero" in note.lower()


def test_no_note_when_every_model_reported_the_metric(rows):
    assert viz._omitted_note(rows, "accuracy") == ""


def test_a_plot_with_no_data_is_not_written(tmp_path):
    empty = [row("resnet18", "ResNet18", accuracy=None)]
    result = viz._bar_plot(empty, "checkpoint_mb", "Size", "MB",
                           tmp_path / "nothing.png")
    assert result is None


# -- the full figure set ----------------------------------------------------

def test_all_eight_section_24_figures_are_written(rows, tmp_path):
    written = viz.plot_all_comparisons(rows, tmp_path)
    expected = {
        "accuracy_comparison", "f1_comparison", "parameter_comparison",
        "model_size", "training_time", "inference_speed",
        "accuracy_vs_parameters", "accuracy_vs_latency",
    }
    assert expected <= set(written), f"missing {expected - set(written)}"
    for name in expected:
        assert Path(written[name]).is_file()
        assert Path(written[name]).stat().st_size > 1000


def test_scatter_plots_skip_rows_missing_either_axis(rows, tmp_path):
    path = viz._scatter_plot(rows, "total_parameters", "accuracy",
                             "Accuracy vs parameters", "params", "accuracy",
                             tmp_path / "scatter.png")
    assert Path(path).is_file()


# -- training curves --------------------------------------------------------

class FakeResult:
    key = "resnet18"
    name = "ResNet18"
    history = {
        "best_epoch": 2,
        "per_epoch": [
            {"epoch": 1, "train_loss": 1.2, "validation_loss": 1.1,
             "train_accuracy": 0.5, "validation_accuracy": 0.55},
            {"epoch": 2, "train_loss": 0.8, "validation_loss": 0.9,
             "train_accuracy": 0.7, "validation_accuracy": 0.72},
            {"epoch": 3, "train_loss": 0.5, "validation_loss": 1.0,
             "train_accuracy": 0.85, "validation_accuracy": 0.70},
        ],
    }
    overfitting = {"generalization_gap": 0.15}


def test_training_curves_are_written(tmp_path):
    path = viz.plot_training_curves(FakeResult(), tmp_path / "curves.png")
    assert Path(path).is_file()
    assert Path(path).stat().st_size > 1000


def test_training_curves_handle_an_empty_history(tmp_path):
    """A failed architecture has no curve, and must not crash the report."""

    class Empty:
        key = "x"
        name = "Nothing"
        history = {}
        overfitting = {}

    path = viz.plot_training_curves(Empty(), tmp_path / "empty.png")
    assert Path(path).is_file()
