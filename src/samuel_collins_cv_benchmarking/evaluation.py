"""Metrics and timing.

Every model - classical or neural - is driven through this one function, so
the comparison is fair by construction rather than by care. A model that had
its own training path would be timed differently and scored differently, and
nothing in the results would show it.

Three rules from the assignment shape this file.

*Fair timing* (section 7). Training time is the ``fit`` call and nothing else;
inference time is the ``predict`` call divided by the number of test images.
Both are measured identically for all six models, so the SVM's minutes and
the Decision Tree's milliseconds are the same kind of number.

*Failures are results* (section 7). A model that raises keeps its row in the
comparison, carrying its error message. Dropping it would leave a benchmark
that silently compared five models while claiming to compare six.

*Ranking* (section 8). Sort by macro F1, and break ties by the lower inference
time. Macro F1 rather than accuracy because it weights every class equally: on
an imbalanced set, a model that ignores the smallest class can still post a
high accuracy, and macro F1 will not let it.
"""

import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class ModelResult:
    """Everything the benchmark learned about one model."""

    key: str
    name: str
    kind: str = "classical"
    succeeded: bool = True
    error: str = None

    accuracy: float = None
    macro_precision: float = None
    macro_recall: float = None
    macro_f1: float = None
    weighted_f1: float = None
    training_time_seconds: float = None
    inference_time_ms_per_image: float = None

    predictions: np.ndarray = None
    confusion_matrix: np.ndarray = None
    classification_report: dict = field(default_factory=dict)
    parameters: dict = field(default_factory=dict)
    history: dict = field(default_factory=dict)

    def metrics(self) -> dict:
        """The section 8 metrics as a plain, JSON-friendly mapping."""
        return {
            "accuracy": self.accuracy,
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "macro_f1": self.macro_f1,
            "weighted_f1": self.weighted_f1,
            "training_time_seconds": self.training_time_seconds,
            "inference_time_ms_per_image": self.inference_time_ms_per_image,
        }


def data_for(spec, split) -> tuple:
    """The representation this model consumes.

    The CNN learns spatial structure and needs the image tensor; everything
    else works on the flattened vectors. Both come from the same split, so
    the choice of representation never changes *which* samples a model sees.
    """
    if spec.kind == "cnn":
        return split.images_train, split.images_test
    return split.features_train, split.features_test


def evaluate_model(spec, split) -> ModelResult:
    """Train one model on the split, score it, and time both phases.

    A model that raises is returned as a failed result rather than allowed to
    end the run: section 7 requires the failure and its message to appear in
    the comparison.
    """
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    result = ModelResult(
        key=spec.key, name=spec.name, kind=spec.kind, parameters=dict(spec.parameters)
    )
    train_x, test_x = data_for(spec, split)
    truth = split.labels_test

    try:
        model = spec()

        started = time.perf_counter()
        model.fit(train_x, split.labels_train)
        result.training_time_seconds = time.perf_counter() - started

        started = time.perf_counter()
        predictions = np.asarray(model.predict(test_x))
        elapsed = time.perf_counter() - started
        result.inference_time_ms_per_image = elapsed * 1000 / max(1, len(predictions))

    except Exception as error:
        result.succeeded = False
        result.error = f"{type(error).__name__}: {error}"
        return result

    # Section 12: every successful model must predict every test image.
    if len(predictions) != len(truth):
        result.succeeded = False
        result.error = (
            f"model returned {len(predictions)} prediction(s) for "
            f"{len(truth)} test image(s)"
        )
        return result

    result.predictions = predictions
    if getattr(model, "history_", None):
        result.history = model.history_

    labels = list(range(len(split.dataset.class_names)))
    names = split.dataset.class_names

    # zero_division=0 throughout, per section 8. A class the model never
    # predicts has undefined precision; scoring it 0 is what makes macro F1
    # penalise ignoring a class rather than quietly skipping it.
    result.accuracy = float(accuracy_score(truth, predictions))
    result.macro_precision = float(
        precision_score(truth, predictions, average="macro", zero_division=0))
    result.macro_recall = float(
        recall_score(truth, predictions, average="macro", zero_division=0))
    result.macro_f1 = float(
        f1_score(truth, predictions, average="macro", zero_division=0))
    result.weighted_f1 = float(
        f1_score(truth, predictions, average="weighted", zero_division=0))

    result.confusion_matrix = confusion_matrix(truth, predictions, labels=labels)
    result.classification_report = classification_report(
        truth, predictions, labels=labels, target_names=names,
        output_dict=True, zero_division=0,
    )
    return result


def rank(results: list) -> list:
    """Order the comparison as section 8 requires.

    Highest macro F1 first; where two models tie, the faster one at inference
    goes first. Failed models sort last - they have no score to rank on, but
    they stay in the list.
    """
    def key(result):
        if not result.succeeded:
            return (1, 0.0, 0.0)
        return (0, -result.macro_f1, result.inference_time_ms_per_image)

    return sorted(results, key=key)


def evaluate_all(specs: list, split, on_progress=None) -> list:
    """Run every model against the one split, ranked.

    ``on_progress`` is called with each finished result, so a caller can
    report as the run proceeds. An RBF SVM on 12,288 features takes minutes,
    and a benchmark that prints nothing for that long looks hung.
    """
    results = []
    for spec in specs:
        result = evaluate_model(spec, split)
        results.append(result)
        if on_progress is not None:
            on_progress(result)
    return rank(results)


def summary_frame(results: list):
    """The comparison table in section 8's column order."""
    import pandas as pd

    rows = []
    for result in results:
        rows.append({
            "Model": result.name,
            "Accuracy": result.accuracy,
            "Macro Precision": result.macro_precision,
            "Macro Recall": result.macro_recall,
            "Macro F1": result.macro_f1,
            "Weighted F1": result.weighted_f1,
            "Training Time (s)": result.training_time_seconds,
            "Inference Time (ms/img)": result.inference_time_ms_per_image,
            "Status": "ok" if result.succeeded else result.error,
        })
    return pd.DataFrame(rows)


def classification_report_frame(result: ModelResult):
    """One model's per-class report, shaped for its CSV file."""
    import pandas as pd

    frame = pd.DataFrame(result.classification_report).transpose()
    frame.index.name = "class"
    return frame.reset_index()


def best_model(results: list) -> str:
    """Name of the highest-ranked model that actually ran."""
    for result in rank(results):
        if result.succeeded:
            return result.name
    return None


def prediction_examples(result: ModelResult, split, limit: int = 5) -> dict:
    """A few correct and incorrect predictions, for the report.

    Section 11 asks for examples of both. They are chosen one class at a time
    rather than in test order: the split's indices are sorted, so taking the
    first few would return five images of whichever class sorts first and
    illustrate nothing about the other nine. Round-robin over the true
    classes gives a spread, and within a class the order is still the fixed
    test order, so the selection is reproducible.

    Positions index into the test half, so a caller can recover the image or
    its source from the split.
    """
    if not result.succeeded or result.predictions is None:
        return {"correct": [], "incorrect": []}

    truth = split.labels_test
    names = split.dataset.class_names
    sources = split.dataset.sources

    by_class = {"correct": {}, "incorrect": {}}
    for position, (actual, predicted) in enumerate(zip(truth, result.predictions)):
        index = int(split.test_index[position])
        entry = {
            "test_position": int(position),
            "dataset_index": index,
            "source": sources[index] if index < len(sources) else None,
            "true_class": names[int(actual)],
            "predicted_class": names[int(predicted)],
        }
        bucket = "correct" if actual == predicted else "incorrect"
        by_class[bucket].setdefault(int(actual), []).append(entry)

    def spread(grouped: dict) -> list:
        """One from each class in turn, until `limit` entries are collected."""
        chosen = []
        depth = 0
        while len(chosen) < limit:
            added = False
            for class_index in sorted(grouped):
                entries = grouped[class_index]
                if depth < len(entries):
                    chosen.append(entries[depth])
                    added = True
                    if len(chosen) == limit:
                        break
            if not added:  # every class exhausted
                break
            depth += 1
        return chosen

    return {"correct": spread(by_class["correct"]),
            "incorrect": spread(by_class["incorrect"])}
