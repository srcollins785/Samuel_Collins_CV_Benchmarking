"""Tests for metrics, timing and the comparison table.

Covers the assignment's requirements that the summary contains six model
rows, that every successful model predicts every test image, and that a
failed model is reported with its error rather than dropped.
"""

import json

import numpy as np
import pytest

from samuel_collins_cv_benchmarking.benchmark import make_split
from samuel_collins_cv_benchmarking.classical_models import (
    ModelSpec,
    classical_model_specs,
)
from samuel_collins_cv_benchmarking.data_loader import load_array
from samuel_collins_cv_benchmarking.evaluation import (
    ModelResult,
    best_model,
    classification_report_frame,
    data_for,
    evaluate_all,
    evaluate_model,
    prediction_examples,
    rank,
    summary_frame,
)
from samuel_collins_cv_benchmarking.neural_models import neural_model_specs
from samuel_collins_cv_benchmarking.preprocessing import preprocess


def make_split_from(bands, labels, size=(16, 16, 3), seed=0, spread=45):
    generator = np.random.default_rng(seed)
    tensor = np.stack([
        generator.integers(max(0, low - spread), min(255, low + spread), size=size)
        .astype(np.uint8)
        for low in bands
    ])
    return make_split(preprocess(load_array(tensor, labels), "rgb"))


@pytest.fixture(scope="module")
def split():
    """Separable, balanced, and small enough to run six models repeatedly."""
    bands = [50] * 40 + [130] * 40 + [210] * 40
    labels = ["cat"] * 40 + ["dog"] * 40 + ["horse"] * 40
    return make_split_from(bands, labels, spread=20)


@pytest.fixture(scope="module")
def imbalanced_split():
    """Imbalanced and only partly separable, so a class gets ignored."""
    bands = [60] * 80 + [120] * 80 + [190] * 20
    labels = ["cat"] * 80 + ["dog"] * 80 + ["horse"] * 20
    return make_split_from(bands, labels)


class ExplodingModel:
    """A model that fails during fit, to exercise the failure path."""

    def fit(self, X, y):
        raise RuntimeError("no space left on device")

    def predict(self, X):  # pragma: no cover - never reached
        raise AssertionError


def exploding_spec():
    return ModelSpec(key="exploding", name="Exploding Model",
                     build=ExplodingModel, scaled=False)


class SlowFitModel:
    """Slow to train, instant to predict, so the two phases are separable."""

    FIT_SECONDS = 0.30

    def fit(self, X, y):
        import time as _time
        _time.sleep(self.FIT_SECONDS)
        self._first = int(np.unique(y)[0])
        return self

    def predict(self, X):
        return np.full(len(X), self._first, dtype=np.int64)


def slow_fit_spec():
    return ModelSpec(key="slow_fit", name="Slow Fit Model",
                     build=SlowFitModel, scaled=False)


class ConstantModel:
    """Always predicts the first class, so the others are never predicted.

    This is the only situation in which zero_division actually fires.
    Precision's denominator is the number of times a class was *predicted*,
    so a class predicted even once wrongly has a well-defined precision of
    zero. It takes a class predicted never for the denominator to reach zero.
    """

    def fit(self, X, y):
        self._first = int(np.unique(y)[0])
        return self

    def predict(self, X):
        return np.full(len(X), self._first, dtype=np.int64)


def constant_spec():
    return ModelSpec(key="constant", name="Constant Model",
                     build=ConstantModel, scaled=False)


class TruncatingModel:
    """A model that returns too few predictions."""

    def fit(self, X, y):
        self._classes = np.unique(y)
        return self

    def predict(self, X):
        return np.zeros(len(X) // 2, dtype=np.int64)


def truncating_spec():
    return ModelSpec(key="truncating", name="Truncating Model",
                     build=TruncatingModel, scaled=False)


# --------------------------------------------------------------------------
# Which data each model receives
# --------------------------------------------------------------------------

class TestDataSelection:

    def test_the_cnn_gets_image_tensors(self, split):
        spec = next(s for s in neural_model_specs() if s.kind == "cnn")
        train, test = data_for(spec, split)
        assert train.shape[1:] == (64, 64, 3)

    def test_everything_else_gets_flattened_features(self, split):
        for spec in classical_model_specs():
            train, test = data_for(spec, split)
            assert train.ndim == 2

    def test_both_representations_cover_the_same_rows(self, split):
        cnn = next(s for s in neural_model_specs() if s.kind == "cnn")
        cnn_train, cnn_test = data_for(cnn, split)
        flat_train, flat_test = data_for(classical_model_specs()[0], split)
        assert len(cnn_train) == len(flat_train)
        assert len(cnn_test) == len(flat_test)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

class TestMetrics:

    @pytest.fixture(scope="class")
    def result(self, split):
        return evaluate_model(classical_model_specs()[0], split)

    def test_every_required_metric_is_present(self, result):
        # Section 8's list.
        for name in ("accuracy", "macro_precision", "macro_recall", "macro_f1",
                     "weighted_f1", "training_time_seconds",
                     "inference_time_ms_per_image"):
            assert result.metrics()[name] is not None

    def test_metrics_are_plain_floats(self, result):
        # numpy scalars are not JSON serializable, and these go into
        # benchmark_metrics.json.
        for value in result.metrics().values():
            assert isinstance(value, float)
        json.dumps(result.metrics())

    def test_metrics_are_in_range(self, result):
        for name in ("accuracy", "macro_precision", "macro_recall",
                     "macro_f1", "weighted_f1"):
            assert 0.0 <= result.metrics()[name] <= 1.0

    def test_times_are_positive(self, result):
        assert result.training_time_seconds > 0
        assert result.inference_time_ms_per_image > 0

    def test_training_and_inference_are_timed_separately(self, split):
        # Section 7 asks for comparable fit and prediction phases. If the
        # timer were not reset between them, inference would carry the whole
        # training cost and the slowest model would look slowest at both.
        result = evaluate_model(slow_fit_spec(), split)
        assert result.training_time_seconds >= SlowFitModel.FIT_SECONDS
        total_inference_ms = result.inference_time_ms_per_image * len(split.test_index)
        assert total_inference_ms < SlowFitModel.FIT_SECONDS * 1000 / 10

    def test_confusion_matrix_is_square_and_totals_the_test_set(self, result, split):
        classes = len(split.dataset.class_names)
        assert result.confusion_matrix.shape == (classes, classes)
        assert result.confusion_matrix.sum() == len(split.test_index)

    def test_classification_report_names_every_class(self, result, split):
        for name in split.dataset.class_names:
            assert name in result.classification_report

    def test_report_frame_has_one_row_per_class_plus_averages(self, result, split):
        frame = classification_report_frame(result)
        assert set(split.dataset.class_names) <= set(frame["class"])
        assert "macro avg" in set(frame["class"])

    def test_predictions_cover_every_test_image(self, result, split):
        assert len(result.predictions) == len(split.test_index)


class TestMacroF1PunishesIgnoredClasses:
    """Why section 8 ranks by macro F1 rather than accuracy."""

    @pytest.fixture(scope="class")
    def result(self, imbalanced_split):
        return evaluate_model(neural_model_specs()[0], imbalanced_split)

    def test_accuracy_stays_high_while_macro_f1_falls(self, result):
        # The model never predicts the smallest class. Accuracy barely
        # notices, because that class is a small share of the test set.
        assert result.accuracy > 0.8
        assert result.macro_f1 < result.accuracy - 0.15

    def test_the_ignored_class_scores_zero_rather_than_being_skipped(self, result):
        # zero_division=0 is what makes an unpredicted class cost something.
        # Without it the class would be dropped from the average entirely.
        per_class = [
            report["f1-score"]
            for name, report in result.classification_report.items()
            if name in ("cat", "dog", "horse")
        ]
        assert 0.0 in per_class

    def test_macro_precision_averages_the_per_class_values(self, result, imbalanced_split):
        # Pins the value rather than only its range. With zero_division=1 the
        # ignored class would contribute a precision of 1.0 to this average
        # instead of 0.0, which would quietly flatter every model that skips
        # a class.
        per_class = [
            result.classification_report[name]["precision"]
            for name in imbalanced_split.dataset.class_names
        ]
        assert result.macro_precision == pytest.approx(float(np.mean(per_class)))

    def test_macro_recall_averages_the_per_class_values(self, result, imbalanced_split):
        per_class = [
            result.classification_report[name]["recall"]
            for name in imbalanced_split.dataset.class_names
        ]
        assert result.macro_recall == pytest.approx(float(np.mean(per_class)))

    def test_weighted_f1_sits_between_the_two(self, result):
        # Weighted F1 counts classes by size, so it tracks accuracy more
        # closely than macro F1 does.
        assert result.macro_f1 < result.weighted_f1


# --------------------------------------------------------------------------
# Failures are results
# --------------------------------------------------------------------------

class TestFailedModels:

    def test_a_failing_model_does_not_end_the_run(self, split):
        result = evaluate_model(exploding_spec(), split)
        assert result.succeeded is False

    def test_the_error_message_is_kept(self, split):
        # Section 7: report the failure and its message.
        result = evaluate_model(exploding_spec(), split)
        assert "no space left on device" in result.error
        assert "RuntimeError" in result.error

    def test_a_failed_model_keeps_its_row(self, split):
        specs = classical_model_specs()[:2] + [exploding_spec()]
        results = evaluate_all(specs, split)
        assert len(results) == 3
        assert "Exploding Model" in {r.name for r in results}

    def test_a_failed_model_has_no_metrics(self, split):
        result = evaluate_model(exploding_spec(), split)
        assert result.macro_f1 is None
        assert result.predictions is None

    def test_a_short_prediction_vector_is_a_failure(self, split):
        # Section 12 requires every successful model to predict every test
        # image, so a model that returns fewer is reported, not scored.
        result = evaluate_model(truncating_spec(), split)
        assert result.succeeded is False
        assert "prediction" in result.error

    def test_failed_models_sort_last(self, split):
        specs = [exploding_spec()] + classical_model_specs()[:2]
        results = evaluate_all(specs, split)
        assert results[-1].name == "Exploding Model"

    def test_best_model_skips_failures(self, split):
        specs = [exploding_spec()] + classical_model_specs()[:1]
        assert best_model(evaluate_all(specs, split)) != "Exploding Model"


# --------------------------------------------------------------------------
# Ranking
# --------------------------------------------------------------------------

class TestRanking:

    def result(self, name, macro_f1, inference, succeeded=True):
        return ModelResult(key=name.lower(), name=name, macro_f1=macro_f1,
                           inference_time_ms_per_image=inference,
                           succeeded=succeeded)

    def test_orders_by_macro_f1_descending(self):
        ordered = rank([
            self.result("low", 0.30, 1.0),
            self.result("high", 0.90, 1.0),
            self.result("mid", 0.60, 1.0),
        ])
        assert [r.name for r in ordered] == ["high", "mid", "low"]

    def test_ties_break_on_lower_inference_time(self):
        # Section 8 says so explicitly.
        ordered = rank([
            self.result("slow", 0.80, 9.0),
            self.result("fast", 0.80, 0.5),
        ])
        assert [r.name for r in ordered] == ["fast", "slow"]

    def test_failures_go_last_even_with_a_high_score_recorded(self):
        ordered = rank([
            self.result("broken", 0.99, 0.1, succeeded=False),
            self.result("working", 0.10, 5.0),
        ])
        assert [r.name for r in ordered] == ["working", "broken"]

    def test_ranking_does_not_mutate_the_input(self):
        results = [self.result("a", 0.1, 1.0), self.result("b", 0.9, 1.0)]
        rank(results)
        assert [r.name for r in results] == ["a", "b"]


# --------------------------------------------------------------------------
# The whole comparison
# --------------------------------------------------------------------------

class TestFullBenchmark:

    @pytest.fixture(scope="class")
    def results(self, split):
        specs = classical_model_specs() + neural_model_specs()
        return evaluate_all(specs, split)

    def test_the_summary_contains_six_model_rows(self, results):
        # Section 12 asks for exactly this.
        assert len(summary_frame(results)) == 6

    def test_every_required_model_appears(self, results):
        assert {r.name for r in results} == {
            "Logistic Regression", "Decision Tree", "Random Forest",
            "SVM", "Neural Network", "Simple CNN",
        }

    def test_summary_columns_match_the_required_table(self, results):
        assert list(summary_frame(results).columns) == [
            "Model", "Accuracy", "Macro Precision", "Macro Recall", "Macro F1",
            "Weighted F1", "Training Time (s)", "Inference Time (ms/img)",
            "Status",
        ]

    def test_results_come_back_ranked(self, results):
        scores = [r.macro_f1 for r in results if r.succeeded]
        assert scores == sorted(scores, reverse=True)

    def test_every_model_predicts_every_test_image(self, results, split):
        for result in results:
            if result.succeeded:
                assert len(result.predictions) == len(split.test_index)

    def test_every_model_saw_the_same_split(self, results, split):
        # Section 7's fairness rule, checked at the far end: each confusion
        # matrix must account for the identical test set.
        for result in results:
            if result.succeeded:
                assert result.confusion_matrix.sum() == len(split.test_index)

    def test_progress_callback_fires_once_per_model(self, split):
        seen = []
        evaluate_all(classical_model_specs(), split, on_progress=seen.append)
        assert len(seen) == 4

    def test_the_cnn_history_is_carried_through(self, results):
        cnn = next(r for r in results if r.key == "simple_cnn")
        assert cnn.history["epochs_run"] >= 1


class TestPredictionExamples:

    def test_returns_both_correct_and_incorrect(self, imbalanced_split):
        result = evaluate_model(neural_model_specs()[0], imbalanced_split)
        examples = prediction_examples(result, imbalanced_split, limit=3)
        assert examples["correct"] and examples["incorrect"]

    def test_examples_point_back_into_the_dataset(self, imbalanced_split):
        result = evaluate_model(neural_model_specs()[0], imbalanced_split)
        examples = prediction_examples(result, imbalanced_split, limit=2)
        for entry in examples["correct"]:
            position = entry["test_position"]
            assert entry["dataset_index"] == imbalanced_split.test_index[position]

    def test_incorrect_examples_really_are_wrong(self, imbalanced_split):
        result = evaluate_model(neural_model_specs()[0], imbalanced_split)
        for entry in prediction_examples(result, imbalanced_split)["incorrect"]:
            assert entry["true_class"] != entry["predicted_class"]

    def test_a_failed_model_yields_no_examples(self, split):
        result = evaluate_model(exploding_spec(), split)
        assert prediction_examples(result, split) == {"correct": [], "incorrect": []}

    def test_examples_are_json_serializable(self, imbalanced_split):
        result = evaluate_model(neural_model_specs()[0], imbalanced_split)
        json.dumps(prediction_examples(result, imbalanced_split))


class TestZeroDivisionOnNeverPredictedClasses:
    """Section 8 requires zero_division=0, and this is where it bites.

    A class that is predicted even once, wrongly, has precision 0/1 - well
    defined, and zero_division never applies. Only a class predicted *never*
    gives 0/0, and then the setting decides whether that class contributes 0
    or 1 to the macro average. Scoring it 1 would flatter a model that
    ignored the class completely.
    """

    @pytest.fixture(scope="class")
    def result(self, split):
        return evaluate_model(constant_spec(), split)

    def test_two_classes_are_never_predicted(self, result, split):
        predicted = set(int(p) for p in result.predictions)
        assert len(predicted) == 1
        assert len(split.dataset.class_names) == 3

    def test_never_predicted_classes_score_zero_precision(self, result, split):
        ignored = split.dataset.class_names[1:]
        for name in ignored:
            assert result.classification_report[name]["precision"] == 0.0

    def test_macro_precision_counts_them_as_zero(self, result, split):
        # One class right, two contributing nothing: about a third.
        per_class = [
            result.classification_report[name]["precision"]
            for name in split.dataset.class_names
        ]
        assert result.macro_precision == pytest.approx(float(np.mean(per_class)))
        assert result.macro_precision < 0.4

    def test_macro_f1_counts_them_as_zero(self, result, split):
        per_class = [
            result.classification_report[name]["f1-score"]
            for name in split.dataset.class_names
        ]
        assert result.macro_f1 == pytest.approx(float(np.mean(per_class)))

    def test_accuracy_still_reflects_the_one_class_it_gets_right(self, result):
        # A third of the balanced test set, so accuracy stays near 0.33 while
        # macro precision is dragged down by the two zeros.
        assert result.accuracy == pytest.approx(1 / 3, abs=0.05)


class TestExamplesSpreadAcrossClasses:
    """Examples must illustrate the model, not one alphabetically-first class.

    The split's indices are sorted, so taking the first few test rows returns
    images of whichever class sorts first - five butterflies telling you
    nothing about the other nine classes. Found by looking at the generated
    figure, where every panel carried the same label.
    """

    @pytest.fixture(scope="class")
    def many_classes(self):
        names = ["ant", "bee", "cat", "dog", "eel", "fox"]
        generator = np.random.default_rng(4)
        bands = [30 + i * 35 for i in range(len(names)) for _ in range(20)]
        tensor = np.stack([
            generator.integers(max(0, b - 40), min(255, b + 40), size=(12, 12, 3))
            .astype(np.uint8) for b in bands
        ])
        labels = [name for name in names for _ in range(20)]
        return make_split(preprocess(load_array(tensor, labels), "rgb"))

    def test_correct_examples_cover_several_classes(self, many_classes):
        result = evaluate_model(classical_model_specs()[1], many_classes)
        examples = prediction_examples(result, many_classes, limit=5)
        classes = {entry["true_class"] for entry in examples["correct"]}
        assert len(classes) > 1

    def test_examples_are_not_all_the_first_class(self, many_classes):
        result = evaluate_model(classical_model_specs()[1], many_classes)
        examples = prediction_examples(result, many_classes, limit=5)
        first = many_classes.dataset.class_names[0]
        assert [e["true_class"] for e in examples["correct"]] != [first] * 5

    def test_selection_is_reproducible(self, many_classes):
        result = evaluate_model(classical_model_specs()[1], many_classes)
        first = prediction_examples(result, many_classes, limit=5)
        second = prediction_examples(result, many_classes, limit=5)
        assert first == second

    def test_examples_carry_their_source(self, many_classes):
        result = evaluate_model(classical_model_specs()[1], many_classes)
        for entry in prediction_examples(result, many_classes)["correct"]:
            assert entry["source"] is not None
