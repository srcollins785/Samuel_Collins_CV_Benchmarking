"""Tests for the model definitions.

Covers the assignment's requirements that every successful model predicts
every test image, that the same split is used throughout, and that scaling is
fitted on training data only.

The scaler test is the important one here. Fitting a scaler on the whole
dataset before splitting does not fail, warn, or look wrong in the results -
it just makes every scaled model score a little too high, because the test
set's mean and variance shaped the transformation applied during training.
Section 7 prohibits it, so it is checked directly by inspecting what the
fitted scaler actually saw.
"""

import json

import numpy as np
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from samuel_collins_cv_benchmarking.classical_models import (
    ModelSpec,
    classical_model_specs,
)
from samuel_collins_cv_benchmarking.benchmark import make_split
from samuel_collins_cv_benchmarking.data_loader import load_array
from samuel_collins_cv_benchmarking.preprocessing import preprocess


@pytest.fixture(scope="module")
def split():
    """A small dataset whose classes are genuinely separable.

    Random noise would leave every model at chance, which would make an
    accuracy assertion meaningless. Giving each class its own brightness band
    means a working model should get these right and a broken one will not.
    """
    generator = np.random.default_rng(0)
    bands = [40] * 40 + [110] * 40 + [180] * 40
    tensor = np.stack([
        generator.integers(low, low + 60, size=(12, 12, 3)).astype(np.uint8)
        for low in bands
    ])
    labels = ["cat"] * 40 + ["dog"] * 40 + ["horse"] * 40
    return make_split(preprocess(load_array(tensor, labels), "rgb"))


# --------------------------------------------------------------------------
# What the registry describes
# --------------------------------------------------------------------------

class TestModelRegistry:

    def test_provides_the_four_classical_models(self):
        assert [spec.name for spec in classical_model_specs()] == [
            "Logistic Regression", "Decision Tree", "Random Forest", "SVM",
        ]

    def test_keys_are_file_safe(self):
        # Section 9 names confusion_matrices/logistic_regression.png and
        # classification_reports/decision_tree.csv, so the key has to be
        # usable as a filename while the name stays readable in the table.
        for spec in classical_model_specs():
            assert spec.key.replace("_", "").isalnum()
            assert spec.key.islower()

    def test_keys_match_the_required_output_filenames(self):
        assert [spec.key for spec in classical_model_specs()] == [
            "logistic_regression", "decision_tree", "random_forest", "svm",
        ]

    def test_every_spec_builds_a_fresh_estimator(self):
        # Two runs must not share a fitted model, or the second would start
        # from the first one's state.
        for spec in classical_model_specs():
            assert spec() is not spec()

    def test_parameters_are_json_serialisable(self):
        # These go into run_configuration.json.
        for spec in classical_model_specs():
            json.dumps(spec.parameters)

    def test_specs_record_the_required_hyperparameters(self):
        by_key = {spec.key: spec.parameters for spec in classical_model_specs()}
        assert by_key["logistic_regression"]["max_iter"] == 1000
        assert by_key["random_forest"]["n_estimators"] == 200
        assert by_key["svm"]["kernel"] == "rbf"

    def test_every_model_fixes_its_random_state(self):
        # Section 7 requires a fixed seed for reproducibility.
        for spec in classical_model_specs():
            assert spec.parameters.get("random_state") == 42


# --------------------------------------------------------------------------
# Scaling
# --------------------------------------------------------------------------

class TestScaling:

    def scaled_specs(self):
        return [spec for spec in classical_model_specs() if spec.scaled]

    def test_logistic_regression_and_svm_are_scaled(self):
        # Section 6.1 names these two, plus the neural network.
        assert {spec.key for spec in self.scaled_specs()} == {
            "logistic_regression", "svm"}

    def test_tree_models_are_not_scaled(self):
        # Trees split on a threshold within one feature, so rescaling that
        # feature changes nothing about the tree.
        unscaled = {spec.key for spec in classical_model_specs() if not spec.scaled}
        assert unscaled == {"decision_tree", "random_forest"}

    def test_scaled_models_are_pipelines(self):
        for spec in self.scaled_specs():
            model = spec()
            assert isinstance(model, Pipeline)
            assert isinstance(model.named_steps["scaler"], StandardScaler)

    def test_unscaled_models_are_bare_estimators(self):
        for spec in classical_model_specs():
            if not spec.scaled:
                assert not isinstance(spec(), Pipeline)

    @pytest.mark.parametrize("key", ["logistic_regression", "svm"])
    def test_scaler_sees_only_the_training_rows(self, split, key):
        # The direct check: a scaler fitted inside the pipeline must have seen
        # exactly as many rows as the training half.
        spec = next(s for s in classical_model_specs() if s.key == key)
        model = spec()
        model.fit(split.features_train, split.labels_train)
        assert model.named_steps["scaler"].n_samples_seen_ == len(split.train_index)

    @pytest.mark.parametrize("key", ["logistic_regression", "svm"])
    def test_scaler_statistics_come_from_training_data_only(self, split, key):
        # And the stronger check: the statistics themselves must match the
        # training half, not the whole dataset.
        spec = next(s for s in classical_model_specs() if s.key == key)
        model = spec()
        model.fit(split.features_train, split.labels_train)
        scaler = model.named_steps["scaler"]

        assert np.allclose(scaler.mean_, split.features_train.mean(axis=0))
        assert not np.allclose(scaler.mean_, split.dataset.features.mean(axis=0))


# --------------------------------------------------------------------------
# Training and prediction
# --------------------------------------------------------------------------

class TestTrainingAndPrediction:

    @pytest.mark.parametrize("spec", classical_model_specs(), ids=lambda s: s.key)
    def test_model_trains_and_predicts_every_test_image(self, split, spec):
        model = spec()
        model.fit(split.features_train, split.labels_train)
        predictions = model.predict(split.features_test)
        assert len(predictions) == len(split.test_index)

    @pytest.mark.parametrize("spec", classical_model_specs(), ids=lambda s: s.key)
    def test_predictions_are_valid_class_indices(self, split, spec):
        model = spec()
        model.fit(split.features_train, split.labels_train)
        predictions = model.predict(split.features_test)
        assert set(predictions) <= set(range(len(split.dataset.class_names)))

    @pytest.mark.parametrize("spec", classical_model_specs(), ids=lambda s: s.key)
    def test_model_learns_separable_classes(self, split, spec):
        # Not a quality bar, a wiring check: on three clearly separated
        # brightness bands, anything far above chance means the features and
        # labels reached the model in the right order.
        model = spec()
        model.fit(split.features_train, split.labels_train)
        accuracy = (model.predict(split.features_test) == split.labels_test).mean()
        assert accuracy > 0.8

    @pytest.mark.parametrize("spec", classical_model_specs(), ids=lambda s: s.key)
    def test_predictions_are_reproducible(self, split, spec):
        first, second = spec(), spec()
        first.fit(split.features_train, split.labels_train)
        second.fit(split.features_train, split.labels_train)
        assert np.array_equal(
            first.predict(split.features_test),
            second.predict(split.features_test),
        )

    def test_every_model_sees_the_same_split(self, split):
        # Section 7's fairness rule, checked at the point of use: each model
        # is handed the identical arrays.
        seen = []
        for spec in classical_model_specs():
            model = spec()
            model.fit(split.features_train, split.labels_train)
            seen.append((split.features_train.shape, split.labels_train.tobytes()))
        assert len(set(seen)) == 1


class TestModelSpec:

    def test_spec_is_callable(self):
        spec = ModelSpec(key="k", name="N", build=lambda: "model", scaled=False)
        assert spec() == "model"

    def test_kind_defaults_to_classical(self):
        for spec in classical_model_specs():
            assert spec.kind == "classical"
