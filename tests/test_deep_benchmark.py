"""Tests for the deep benchmark orchestration.

Two inherited rules from Part 1 are the subject here.

Failures are results: an architecture that raises keeps its row carrying its
error message rather than disappearing. A benchmark that silently compared
eight architectures while claiming to compare nine would be worse than one
that admits a failure.

And the scoring is shared: the deep architectures are scored by the same
function, with the same zero-division convention, as everything else. That is
checked against scikit-learn directly rather than assumed.
"""

import numpy as np
import pytest
from sklearn.metrics import f1_score, precision_score, recall_score

from samuel_collins_cv_benchmarking import deep_benchmark

CLASSES = ["cat", "dog", "horse"]


# -- scoring ----------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2])
def test_scoring_matches_scikit_learn(seed):
    generator = np.random.default_rng(seed)
    truth = generator.integers(0, 3, 150)
    predictions = generator.integers(0, 3, 150)
    scored = deep_benchmark.score_predictions(truth, predictions, CLASSES)

    labels = [0, 1, 2]
    for average in ("macro", "weighted"):
        assert scored[f"{average}_precision"] == pytest.approx(
            precision_score(truth, predictions, average=average,
                            labels=labels, zero_division=0))
        assert scored[f"{average}_recall"] == pytest.approx(
            recall_score(truth, predictions, average=average,
                         labels=labels, zero_division=0))
        assert scored[f"{average}_f1"] == pytest.approx(
            f1_score(truth, predictions, average=average,
                     labels=labels, zero_division=0))


def test_scoring_returns_every_metric_section_13_requires():
    truth = np.array([0, 1, 2, 0, 1, 2])
    predictions = np.array([0, 1, 2, 0, 2, 1])
    scored = deep_benchmark.score_predictions(truth, predictions, CLASSES)
    for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1",
                "weighted_precision", "weighted_recall", "weighted_f1"):
        assert scored[key] is not None


def test_per_class_accuracy_is_the_row_normalized_diagonal():
    """Section 15's per-class accuracy, which equals per-class recall."""
    truth = np.array([0, 0, 0, 0, 1, 1, 2, 2])
    predictions = np.array([0, 0, 0, 1, 1, 1, 2, 0])
    scored = deep_benchmark.score_predictions(truth, predictions, CLASSES)

    assert scored["per_class_accuracy"]["cat"] == pytest.approx(3 / 4)
    assert scored["per_class_accuracy"]["dog"] == pytest.approx(2 / 2)
    assert scored["per_class_accuracy"]["horse"] == pytest.approx(1 / 2)


def test_a_class_with_no_test_images_reports_none_not_zero():
    """Zero accuracy would imply it was tested and failed."""
    truth = np.array([0, 0, 1, 1])
    predictions = np.array([0, 1, 1, 1])
    scored = deep_benchmark.score_predictions(truth, predictions, CLASSES)
    assert scored["per_class_accuracy"]["horse"] is None


def test_confusion_matrix_covers_every_class():
    truth = np.array([0, 1])
    predictions = np.array([0, 1])
    scored = deep_benchmark.score_predictions(truth, predictions, CLASSES)
    assert scored["confusion_matrix"].shape == (3, 3)


# -- failures are results ---------------------------------------------------

def test_an_unknown_architecture_becomes_a_failed_row_not_an_exception(tmp_path):
    result = deep_benchmark.benchmark_architecture(
        "nonexistent_net", {}, CLASSES, tmp_path)
    assert result.succeeded is False
    assert result.error
    assert result.accuracy is None
    # It still has a name and a key, so it can appear in the comparison.
    assert result.key == "nonexistent_net"


def test_a_failed_result_still_reports_its_metric_keys():
    result = deep_benchmark.DeepResult(key="x", name="X", succeeded=False,
                                       error="boom")
    metrics = result.metrics()
    assert set(metrics) >= {"accuracy", "macro_f1", "weighted_f1"}
    assert all(value is None for value in metrics.values())


# -- the result shape -------------------------------------------------------

def test_result_exposes_the_fields_the_shared_code_needs():
    """Part 1's plotting takes .name and .confusion_matrix off a result."""
    result = deep_benchmark.DeepResult(key="resnet18", name="ResNet18")
    assert hasattr(result, "name")
    assert hasattr(result, "confusion_matrix")
    assert hasattr(result, "key")
    assert hasattr(result, "succeeded")


def test_derived_properties_read_from_the_recorded_history():
    result = deep_benchmark.DeepResult(
        key="r", name="R",
        history={"total_training_seconds": 12.5, "checkpoint_megabytes": 44.8},
        inference={"latency_ms_per_image": 1.25})
    assert result.training_time_seconds == 12.5
    assert result.checkpoint_megabytes == 44.8
    assert result.inference_time_ms_per_image == 1.25


def test_derived_properties_are_none_when_nothing_was_recorded():
    result = deep_benchmark.DeepResult(key="r", name="R")
    assert result.training_time_seconds is None
    assert result.inference_time_ms_per_image is None


# -- serialization ----------------------------------------------------------

def test_numpy_values_are_made_json_safe():
    import json

    value = {
        "array": np.array([[1, 2], [3, 4]]),
        "integer": np.int64(7),
        "float": np.float32(1.5),
        "nested": [np.int32(1), {"deep": np.float64(2.5)}],
    }
    converted = deep_benchmark._jsonable(value)
    json.dumps(converted)  # must not raise
    assert converted["array"] == [[1, 2], [3, 4]]
    assert converted["integer"] == 7


# -- the deviations record --------------------------------------------------

def test_every_deviation_states_the_item_the_change_and_the_handling():
    """Section 6 requires documentation, so each entry must be complete."""
    deviations = deep_benchmark._deviations()
    assert deviations
    for entry in deviations:
        assert entry["item"].strip()
        assert entry["deviation"].strip()
        assert entry["handling"].strip()


def test_the_deviations_cover_the_known_departures():
    text = " ".join(
        f"{e['item']} {e['deviation']} {e['handling']}"
        for e in deep_benchmark._deviations()).lower()
    for topic in ("memory", "googlenet", "validation", "yolo", "mac"):
        assert topic in text, f"no deviation recorded about {topic}"


# -- single-architecture runs must not destroy the others -------------------

def test_a_single_architecture_run_merges_into_existing_results(tmp_path):
    """The learning-rate remediation depends on this.

    It re-runs two architectures out of ten by invoking the benchmark once per
    architecture. If writing results replaced the file rather than merging
    into it, the other eight would be deleted and the master table built
    straight afterward would look complete while describing two models.
    """
    import json

    from samuel_collins_cv_benchmarking import deep_data

    existing = {
        "resnet50": {"name": "ResNet50", "succeeded": True, "accuracy": 0.918},
        "densenet121": {"name": "DenseNet121", "succeeded": True,
                        "accuracy": 0.896},
    }
    (tmp_path / "deep_metrics.json").write_text(json.dumps(existing),
                                                encoding="utf-8")

    class FakeSplit:
        class dataset:
            class_names = CLASSES
            labels = np.zeros(4, dtype=np.int64)
        random_seed = 42
        train_index = np.arange(2)
        validation_index = np.arange(2, 3)
        test_index = np.arange(3, 4)

        @staticmethod
        def summary():
            return {"training_samples": 2}

        @property
        def class_names(self):
            return CLASSES

    replacement = deep_benchmark.DeepResult(
        key="alexnet", name="AlexNet", succeeded=False,
        error="re-run at a lower rate")

    deep_benchmark.write_deep_artifacts(
        [replacement], FakeSplit(), {"cache_shapes": {"train": [2, 256, 256, 3]}},
        {"gpu": "test"}, tmp_path, epochs=20, learning_rate=1e-4,
        device="cpu", color_mode="rgb")

    merged = json.loads((tmp_path / "deep_metrics.json").read_text())
    assert set(merged) == {"resnet50", "densenet121", "alexnet"}
    assert merged["resnet50"]["accuracy"] == 0.918
    assert merged["alexnet"]["name"] == "AlexNet"


def test_run_configuration_records_the_prescribed_rate_not_the_run_rate(tmp_path):
    """A remediation run must not relabel the protocol for everyone else."""
    import json

    from samuel_collins_cv_benchmarking._config import CNN_LEARNING_RATE

    class FakeSplit:
        class dataset:
            class_names = CLASSES
            labels = np.zeros(4, dtype=np.int64)
        random_seed = 42
        train_index = np.arange(2)
        validation_index = np.arange(2, 3)
        test_index = np.arange(3, 4)

        @staticmethod
        def summary():
            return {"training_samples": 2}

        @property
        def class_names(self):
            return CLASSES

    deep_benchmark.write_deep_artifacts(
        [], FakeSplit(), {"cache_shapes": {"train": [2, 256, 256, 3]}},
        {"gpu": "test"}, tmp_path, epochs=20, learning_rate=1e-4,
        device="cpu", color_mode="rgb")

    configuration = json.loads(
        (tmp_path / "deep_run_configuration.json").read_text())
    assert configuration["protocol"]["learning_rate"] == CNN_LEARNING_RATE
    assert configuration["protocol"]["learning_rate_this_run"] == 1e-4
