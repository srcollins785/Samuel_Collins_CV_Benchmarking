"""Tests for the combined Part 1 + Part 2 comparison.

Two things here carry real risk.

The first is ``metrics_from_confusion_matrix``. Section 13 asks for weighted
precision and weighted recall, and Part 1 recorded neither - it stored the
macro averages and weighted F1 only. Rather than leave two columns blank for
six of the sixteen models, they are recomputed from the confusion matrices
Part 1 did store, which determine every averaged classification metric
exactly. That is sound in principle and easy to get subtly wrong in practice,
so it is checked against scikit-learn on randomized data rather than on one
convenient example.

The second is the N/A convention. Section 23 says to write N/A where a metric
is not meaningful and explain it, rather than invent a value. A zero in the
parameter-count column for a Random Forest would be a fabricated measurement
sitting in the headline table looking exactly like a real one, so the tests
assert the cell says N/A.
"""

import json

import numpy as np
import pytest
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from samuel_collins_cv_benchmarking import combined


# -- the derivation ---------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_derived_metrics_match_scikit_learn_exactly(seed):
    """A confusion matrix determines every averaged metric. Prove it.

    Randomized across seeds so the test is not passing on one lucky example.
    Deliberately includes a class the predictions never use, which is the case
    where a zero-division convention matters: that class has undefined
    precision, and scoring it zero rather than skipping it is what makes a
    macro average penalize a model for abandoning a class.
    """
    generator = np.random.default_rng(seed)
    classes = 6
    truth = generator.integers(0, classes, 400)
    # Never predict the last class, so a zero-division case is always present.
    predictions = generator.integers(0, classes - 1, 400)

    labels = list(range(classes))
    matrix = confusion_matrix(truth, predictions, labels=labels)
    derived = combined.metrics_from_confusion_matrix(matrix)

    assert derived["accuracy"] == pytest.approx(
        accuracy_score(truth, predictions))
    for average in ("macro", "weighted"):
        assert derived[f"{average}_precision"] == pytest.approx(
            precision_score(truth, predictions, average=average,
                            labels=labels, zero_division=0))
        assert derived[f"{average}_recall"] == pytest.approx(
            recall_score(truth, predictions, average=average,
                         labels=labels, zero_division=0))
        assert derived[f"{average}_f1"] == pytest.approx(
            f1_score(truth, predictions, average=average,
                     labels=labels, zero_division=0))


def test_derivation_handles_a_perfect_classifier():
    matrix = np.diag([10, 20, 30])
    derived = combined.metrics_from_confusion_matrix(matrix)
    assert derived["accuracy"] == pytest.approx(1.0)
    assert derived["macro_f1"] == pytest.approx(1.0)
    assert derived["weighted_precision"] == pytest.approx(1.0)


def test_derivation_of_a_single_class_predictor():
    """The shape a collapsed model leaves: everything predicted as class 0."""
    matrix = np.array([[20, 0, 0], [20, 0, 0], [20, 0, 0]])
    derived = combined.metrics_from_confusion_matrix(matrix)
    assert derived["accuracy"] == pytest.approx(20 / 60)
    # Two of three classes score zero on every metric, so macro F1 collapses.
    assert derived["macro_f1"] == pytest.approx(0.5 / 3, abs=1e-6)


def test_derivation_on_an_empty_matrix():
    assert combined.metrics_from_confusion_matrix(np.empty((0, 0))) == {}


# -- reading Part 1 back ----------------------------------------------------

@pytest.fixture
def config_dir(tmp_path):
    """A minimal Part 1 artifact directory, shaped like the real one."""
    matrix = [[8, 2], [3, 7]]
    derived = combined.metrics_from_confusion_matrix(matrix)
    (tmp_path / "benchmark_metrics.json").write_text(json.dumps({
        "logistic_regression": {
            "name": "Logistic Regression",
            "succeeded": True,
            "error": None,
            "accuracy": derived["accuracy"],
            "macro_precision": derived["macro_precision"],
            "macro_recall": derived["macro_recall"],
            "macro_f1": derived["macro_f1"],
            "weighted_f1": derived["weighted_f1"],
            "training_time_seconds": 1.5,
            "inference_time_ms_per_image": 0.25,
            "confusion_matrix": matrix,
            "classification_report": {
                "cat": {"f1-score": 0.8}, "dog": {"f1-score": 0.7}},
            "training_history": {},
        },
        "decision_tree": {
            "name": "Decision Tree", "succeeded": False,
            "error": "ValueError: boom", "confusion_matrix": None,
        },
    }), encoding="utf-8")
    return tmp_path


def test_part1_rows_are_recovered_with_the_missing_averages_filled_in(config_dir):
    rows = combined.load_part1_rows(config_dir)
    row = next(r for r in rows if r["key"] == "logistic_regression")

    assert row["family"] == "Traditional ML"
    assert row["group"] == "Traditional ML"
    # The two Part 1 never recorded.
    assert row["weighted_precision"] is not None
    assert row["weighted_recall"] is not None
    # Throughput is exact arithmetic on the latency Part 1 measured.
    assert row["throughput"] == pytest.approx(1000 / 0.25)


def test_a_failed_part1_model_keeps_its_row_and_its_error(config_dir):
    """Failures are results, as they were in Part 1."""
    rows = combined.load_part1_rows(config_dir)
    row = next(r for r in rows if r["key"] == "decision_tree")
    assert row["status"] == "ValueError: boom"
    assert row["accuracy"] is None


def test_a_misread_confusion_matrix_stops_the_build(config_dir, tmp_path):
    """If the derivation disagrees with Part 1, stop rather than publish.

    A silent disagreement would mean the master table's traditional rows were
    computed from a misreading of the artifact, which is exactly the kind of
    error that survives into a report unnoticed.
    """
    path = config_dir / "benchmark_metrics.json"
    stored = json.loads(path.read_text())
    stored["logistic_regression"]["macro_f1"] = 0.999
    path.write_text(json.dumps(stored), encoding="utf-8")

    with pytest.raises(ValueError, match="misread"):
        combined.load_part1_rows(config_dir)


def test_missing_part1_artifacts_yield_no_rows(tmp_path):
    assert combined.load_part1_rows(tmp_path) == []
    assert combined.load_part2_rows(tmp_path) == []


# -- the master table -------------------------------------------------------

def test_inapplicable_metrics_are_written_as_na(config_dir, tmp_path):
    """Section 23: N/A, never a fabricated zero.

    Also pins the reading convention. "N/A" is in pandas' default missing-value
    list, so a plain read_csv turns every one of these cells into NaN and the
    report renders "nan" where it should say N/A - which is how this was first
    found.
    """
    import pandas as pd

    rows = combined.combined_rows(config_dir)
    path = combined.write_combined_csv(rows, tmp_path / "combined.csv")
    # keep_default_na=False, as the report reads it: pandas would otherwise
    # reinterpret the deliberate "N/A" cells as missing values, which is the
    # bug this assertion originally caught.
    frame = pd.read_csv(path, keep_default_na=False)

    row = frame[frame["Model"] == "Logistic Regression"].iloc[0]
    for column in ("Total Parameters", "Size MB", "Memory MB", "GMACs"):
        assert row[column] == "N/A", f"{column} should be N/A, got {row[column]}"


def test_rows_follow_the_assignment_order(config_dir):
    """Prior methods first, then the new architectures chronologically."""
    rows = combined.combined_rows(config_dir)
    keys = [row["key"] for row in rows]
    assert keys.index("logistic_regression") < keys.index("decision_tree")


# -- rankings ---------------------------------------------------------------

@pytest.fixture
def ranking_rows():
    def row(key, name, **extra):
        base = {
            "key": key, "name": name, "family": "Test", "group": "Deep CNN",
            "accuracy": None, "macro_f1": None, "macro_precision": None,
            "macro_recall": None, "weighted_precision": None,
            "weighted_recall": None, "weighted_f1": None,
            "total_parameters": None, "trainable_parameters": None,
            "checkpoint_mb": None, "training_seconds": None,
            "latency_ms": None, "throughput": None, "memory_mb": None,
            "memory_measurement": None, "gmacs": None,
            "gflops_estimate": None, "best_epoch": None,
            "generalization_gap": None, "status": "ok",
        }
        base.update(extra)
        return base

    return [
        row("big", "Big Net", accuracy=0.90, macro_f1=0.90,
            total_parameters=100_000_000, checkpoint_mb=400.0,
            latency_ms=10.0, throughput=100.0, memory_mb=800.0),
        row("small", "Small Net", accuracy=0.85, macro_f1=0.85,
            total_parameters=5_000_000, checkpoint_mb=20.0,
            latency_ms=2.0, throughput=500.0, memory_mb=200.0),
        row("partial", "Partial Net", accuracy=0.80, macro_f1=0.80,
            latency_ms=1.0, throughput=1000.0),
    ]


def test_accuracy_ranking_is_ordered_highest_first(ranking_rows):
    ranking = combined.all_rankings(ranking_rows)["A_highest_accuracy"]["ranking"]
    assert [entry["model"] for entry in ranking] == [
        "Big Net", "Small Net", "Partial Net"]
    assert ranking[0]["rank"] == 1


def test_smallest_model_ranking_is_ordered_smallest_first(ranking_rows):
    ranking = combined.all_rankings(ranking_rows)["C_smallest_model"]["ranking"]
    assert [entry["model"] for entry in ranking] == ["Small Net", "Big Net"]


def test_models_without_a_metric_are_absent_not_zero(ranking_rows):
    """Partial Net has no checkpoint, so it cannot rank on checkpoint size."""
    ranking = combined.all_rankings(ranking_rows)["C_smallest_model"]["ranking"]
    assert "Partial Net" not in [entry["model"] for entry in ranking]


def test_accuracy_per_million_parameters_favors_the_small_model(ranking_rows):
    ranking = combined.accuracy_per_million_parameters(ranking_rows)
    assert ranking[0]["model"] == "Small Net"
    assert ranking[0]["accuracy_per_million_parameters"] == pytest.approx(
        0.85 / 5.0, abs=1e-4)


def test_deployment_score_only_ranks_models_with_every_metric(ranking_rows):
    """A partial score renormalized over fewer metrics is not comparable."""
    scored = combined.weighted_score(ranking_rows, combined.DEPLOYMENT_WEIGHTS)
    assert {entry["model"] for entry in scored} == {"Big Net", "Small Net"}


def test_universal_score_ranks_every_model(ranking_rows):
    """All three report accuracy, macro F1 and latency."""
    scored = combined.weighted_score(ranking_rows, combined.UNIVERSAL_WEIGHTS)
    assert len(scored) == 3
    assert scored[0]["rank"] == 1
    assert scored[0]["score"] >= scored[-1]["score"]


def test_cost_metrics_are_inverted_before_scoring():
    """Lower latency must score higher, not lower."""
    values = combined._normalize([1.0, 10.0], higher_is_better=False)
    assert values[0] > values[1]


def test_a_flat_metric_does_not_drag_every_score_to_zero(ranking_rows):
    """If a metric cannot discriminate, it should not penalize everyone."""
    values = combined._normalize([5.0, 5.0, 5.0])
    assert list(values) == [1.0, 1.0, 1.0]


def test_score_contributions_sum_to_the_score(ranking_rows):
    scored = combined.weighted_score(ranking_rows, combined.UNIVERSAL_WEIGHTS)
    for entry in scored:
        assert entry["score"] == pytest.approx(
            sum(entry["contributions"].values()), abs=1e-4)


def test_weights_sum_to_one():
    """Both scoring systems should be stated as full weightings."""
    assert sum(combined.UNIVERSAL_WEIGHTS.values()) == pytest.approx(1.0)
    assert sum(combined.DEPLOYMENT_WEIGHTS.values()) == pytest.approx(1.0)


def test_every_ranking_states_its_basis(ranking_rows):
    """A ranking without its methodology is not interpretable."""
    rankings = combined.all_rankings(ranking_rows)
    for key in ("A_highest_accuracy", "B_fastest_inference",
                "C_smallest_model", "D_accuracy_per_million_parameters"):
        assert rankings[key]["basis"].strip()
    overall = rankings["E_overall_recommendation"]
    assert overall["universal"]["basis"].strip()
    assert overall["deployment"]["basis"].strip()


def test_failed_models_are_excluded_from_rankings(ranking_rows):
    broken = dict(ranking_rows[0])
    broken.update({"key": "broken", "name": "Broken Net", "status": "boom"})
    ranking = combined.all_rankings(ranking_rows + [broken])
    names = [e["model"] for e in ranking["A_highest_accuracy"]["ranking"]]
    assert "Broken Net" not in names


# -- log normalization for orders-of-magnitude cost metrics ------------------

def test_log_scaling_keeps_a_cost_term_discriminating():
    """An outlier must not flatten the metric it belongs to.

    Latency in this benchmark spans 0.002 ms to 89.96 ms. Under linear
    normalization the slowest model defines the range and every fast model
    lands within a percent of the best possible score, so a 25% weight on
    latency contributes the same amount to everyone and does no ranking work
    at all. Log scaling restores the spread.
    """
    latencies = [0.002, 0.03, 0.4, 1.5, 4.5, 89.96]

    linear = combined._normalize(latencies, higher_is_better=False, log=False)
    logged = combined._normalize(latencies, higher_is_better=False, log=True)

    # Linear: the five fast models are crushed into a narrow band.
    assert max(linear[:5]) - min(linear[:5]) < 0.06
    # Log: the same five are spread across most of the range.
    assert max(logged[:5]) - min(logged[:5]) > 0.5

    # Both orderings still rank faster as better.
    assert linear[0] == pytest.approx(1.0)
    assert logged[0] == pytest.approx(1.0)


def test_log_scaling_preserves_ordering():
    values = [1.0, 10.0, 100.0, 1000.0]
    scaled = combined._normalize(values, higher_is_better=True, log=True)
    assert list(scaled) == sorted(scaled)
    # Evenly spaced in log space, so evenly spaced after normalization.
    assert scaled[1] - scaled[0] == pytest.approx(scaled[2] - scaled[1])


def test_log_scaling_falls_back_when_a_value_is_not_positive():
    """log10(0) is undefined; normalize linearly rather than emit NaN."""
    scaled = combined._normalize([0.0, 1.0, 2.0], higher_is_better=True, log=True)
    assert all(np.isfinite(scaled))
    assert scaled[0] == pytest.approx(0.0)
    assert scaled[-1] == pytest.approx(1.0)


def test_accuracy_is_not_log_scaled():
    """Bounded [0, 1] metrics are already commensurable."""
    assert "accuracy" not in combined.LOG_SCALED_METRICS
    assert "macro_f1" not in combined.LOG_SCALED_METRICS
    assert "latency_ms" in combined.LOG_SCALED_METRICS
    assert "checkpoint_mb" in combined.LOG_SCALED_METRICS
