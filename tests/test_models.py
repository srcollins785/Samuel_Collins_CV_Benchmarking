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


# --------------------------------------------------------------------------
# Neural models
# --------------------------------------------------------------------------

from samuel_collins_cv_benchmarking.neural_models import (  # noqa: E402
    SimpleCNN,
    neural_model_specs,
)


@pytest.fixture(scope="module")
def small_split():
    """A smaller separable set, because the CNN actually trains here."""
    generator = np.random.default_rng(1)
    bands = [40] * 30 + [120] * 30 + [200] * 30
    tensor = np.stack([
        generator.integers(low, low + 50, size=(16, 16, 3)).astype(np.uint8)
        for low in bands
    ])
    labels = ["cat"] * 30 + ["dog"] * 30 + ["horse"] * 30
    return make_split(preprocess(load_array(tensor, labels), "rgb"))


class TestNeuralRegistry:

    def test_provides_both_neural_models(self):
        assert [spec.name for spec in neural_model_specs()] == [
            "Neural Network", "Simple CNN"]

    def test_keys_match_the_required_output_filenames(self):
        assert [spec.key for spec in neural_model_specs()] == [
            "neural_network", "simple_cnn"]

    def test_the_mlp_is_scaled_like_the_other_linear_models(self):
        # Section 6.1 names Logistic Regression, SVM and the fully connected
        # network as the three that need a scaler.
        spec = neural_model_specs()[0]
        assert spec.scaled
        assert isinstance(spec().named_steps["scaler"], StandardScaler)

    def test_the_cnn_is_not_scaled(self):
        # Its input is already normalized to [0, 1] by preprocessing.
        assert neural_model_specs()[1].scaled is False

    def test_the_cnn_wants_image_tensors(self):
        # kind is how evaluation knows to hand over images rather than the
        # flattened features every other model consumes.
        assert neural_model_specs()[1].kind == "cnn"
        assert neural_model_specs()[0].kind == "neural"

    def test_mlp_records_the_required_hyperparameters(self):
        parameters = neural_model_specs()[0].parameters
        assert parameters["hidden_layer_sizes"] == [128, 64]
        assert parameters["max_iter"] == 100
        assert parameters["early_stopping"] is True

    def test_cnn_records_its_training_settings(self):
        parameters = neural_model_specs()[1].parameters
        assert parameters["max_epochs"] == 20
        assert parameters["batch_size"] == 32
        assert parameters["early_stopping_patience"] == 3
        assert parameters["optimizer"] == "Adam"

    def test_the_cnn_declares_no_pretrained_weights(self):
        # Section 6.2 forbids them, and run_configuration.json should say so.
        assert neural_model_specs()[1].parameters["pretrained"] is False

    def test_neural_parameters_are_json_serialisable(self):
        for spec in neural_model_specs():
            json.dumps(spec.parameters)


class TestCnnArchitecture:

    def layers(self, channels=3, classes=3):
        cnn = SimpleCNN()
        cnn._image_size_ = (64, 64)
        network = cnn._build(channels, classes)
        flat = list(network[0]) + list(network[1:])
        return flat

    def test_layer_order_matches_the_specification(self):
        import torch.nn as nn

        expected = [nn.Conv2d, nn.ReLU, nn.MaxPool2d, nn.Conv2d, nn.ReLU,
                    nn.MaxPool2d, nn.Flatten, nn.Linear, nn.ReLU, nn.Dropout,
                    nn.Linear]
        assert [type(layer) for layer in self.layers()] == expected

    def test_filter_counts(self):
        convolutions = [l for l in self.layers() if type(l).__name__ == "Conv2d"]
        assert [c.out_channels for c in convolutions] == [32, 64]
        assert all(c.kernel_size == (3, 3) for c in convolutions)

    def test_pooling_is_two_by_two(self):
        pools = [l for l in self.layers() if type(l).__name__ == "MaxPool2d"]
        assert len(pools) == 2
        assert all(p.kernel_size == 2 for p in pools)

    def test_dense_layer_and_dropout(self):
        linears = [l for l in self.layers() if type(l).__name__ == "Linear"]
        dropouts = [l for l in self.layers() if type(l).__name__ == "Dropout"]
        assert linears[0].out_features == 128
        assert dropouts[0].p == 0.30

    def test_output_has_one_unit_per_class(self):
        for class_count in (2, 3, 10):
            linears = [l for l in self.layers(classes=class_count)
                       if type(l).__name__ == "Linear"]
            assert linears[-1].out_features == class_count

    def test_accepts_single_channel_input(self):
        convolutions = [l for l in self.layers(channels=1)
                        if type(l).__name__ == "Conv2d"]
        assert convolutions[0].in_channels == 1


class TestCnnTraining:

    def test_trains_and_predicts_every_test_image(self, small_split):
        model = SimpleCNN(epochs=3)
        model.fit(small_split.images_train, small_split.labels_train)
        predictions = model.predict(small_split.images_test)
        assert len(predictions) == len(small_split.test_index)

    def test_predictions_are_valid_class_indices(self, small_split):
        model = SimpleCNN(epochs=3)
        model.fit(small_split.images_train, small_split.labels_train)
        predictions = model.predict(small_split.images_test)
        assert set(predictions) <= set(range(len(small_split.dataset.class_names)))

    def test_learns_separable_classes(self, small_split):
        model = SimpleCNN(epochs=10)
        model.fit(small_split.images_train, small_split.labels_train)
        accuracy = (model.predict(small_split.images_test)
                    == small_split.labels_test).mean()
        assert accuracy > 0.8

    def test_predictions_are_reproducible(self, small_split):
        # Same seed, same weights, same answers - on CPU, on any machine.
        first = SimpleCNN(epochs=3).fit(small_split.images_train, small_split.labels_train)
        second = SimpleCNN(epochs=3).fit(small_split.images_train, small_split.labels_train)
        assert np.array_equal(
            first.predict(small_split.images_test),
            second.predict(small_split.images_test),
        )

    def test_handles_grayscale_input(self, small_split):
        generator = np.random.default_rng(2)
        bands = [40] * 30 + [200] * 30
        tensor = np.stack([
            generator.integers(low, low + 50, size=(16, 16)).astype(np.uint8)
            for low in bands
        ])
        split = make_split(preprocess(
            load_array(tensor, ["cat"] * 30 + ["dog"] * 30), "grayscale"))
        model = SimpleCNN(epochs=2)
        model.fit(split.images_train, split.labels_train)
        assert len(model.predict(split.images_test)) == len(split.test_index)

    def test_predict_before_fit_raises(self, small_split):
        with pytest.raises(RuntimeError, match="before fit"):
            SimpleCNN().predict(small_split.images_test)


class TestCnnEarlyStopping:

    @pytest.fixture(scope="class")
    def trained(self, small_split):
        model = SimpleCNN(epochs=20)
        model.fit(small_split.images_train, small_split.labels_train)
        return model

    def test_validation_comes_out_of_the_training_half_only(self, trained, small_split):
        # Section 7: the test set must not influence training, and deciding
        # when to stop is influence.
        history = trained.history_
        assert history["training_samples"] + history["validation_samples"] == \
               len(small_split.train_index)

    def test_validation_is_about_fifteen_percent_of_training(self, trained, small_split):
        history = trained.history_
        fraction = history["validation_samples"] / len(small_split.train_index)
        assert fraction == pytest.approx(0.15, abs=0.03)

    def test_never_runs_past_the_epoch_cap(self, trained):
        assert trained.history_["epochs_run"] <= 20

    def test_history_records_every_epoch(self, trained):
        assert len(trained.history_["per_epoch"]) == trained.history_["epochs_run"]

    def test_best_loss_is_the_minimum_seen(self, trained):
        losses = [e["validation_loss"] for e in trained.history_["per_epoch"]]
        assert trained.history_["best_validation_loss"] == pytest.approx(min(losses))

    def test_separable_data_trains_to_the_epoch_cap(self, trained):
        # With any decrease counting as an improvement, a set this clean keeps
        # improving microscopically every epoch, so patience never runs out
        # and the cap is what stops training. That is the cap doing its job,
        # not early stopping failing.
        assert trained.history_["epochs_run"] == 20
        assert trained.history_["stopped_early"] is False


    def test_history_is_json_serialisable(self, trained):
        json.dumps(trained.history_)

    def test_tiny_training_set_skips_validation(self):
        # With too few samples to stratify, training on all of them beats
        # failing outright.
        model = SimpleCNN(epochs=2)
        images = np.zeros((4, 64, 64, 1), dtype=np.float32)
        model.fit(images, np.array([0, 0, 1, 1]))
        assert model.history_["validation_samples"] == 0


class TestCnnEarlyStoppingActuallyFires:
    """Early stopping needs data the network cannot learn.

    Random pixels with shuffled labels leave nothing to generalise, so the
    network can only memorise the training rows and validation loss has to
    turn upward. That is the situation early stopping exists for.
    """

    @pytest.fixture(scope="class")
    def split_for_overfit(self):
        generator = np.random.default_rng(3)
        tensor = generator.integers(0, 256, size=(120, 16, 16, 3), dtype=np.uint8)
        labels = ["cat"] * 60 + ["dog"] * 60
        generator.shuffle(labels)
        return make_split(preprocess(load_array(tensor, labels), "rgb"))

    @pytest.fixture(scope="class")
    def overfitted(self, split_for_overfit):
        return SimpleCNN(epochs=20).fit(
            split_for_overfit.images_train, split_for_overfit.labels_train)

    def test_stops_before_the_epoch_cap(self, overfitted):
        assert overfitted.history_["stopped_early"] is True
        assert overfitted.history_["epochs_run"] < 20

    def test_stops_exactly_patience_epochs_after_the_best(self, overfitted):
        losses = [e["validation_loss"] for e in overfitted.history_["per_epoch"]]
        best_epoch = int(np.argmin(losses)) + 1
        assert overfitted.history_["epochs_run"] - best_epoch == 3

    def test_the_weights_kept_are_the_genuine_minimum(self, overfitted):
        # With a non-zero improvement threshold an epoch that beat the best by
        # a small margin would not count, and the model would end up holding
        # weights that were measurably worse than ones it had already seen.
        losses = [e["validation_loss"] for e in overfitted.history_["per_epoch"]]
        assert overfitted.history_["best_validation_loss"] == pytest.approx(min(losses))

    def test_the_model_actually_holds_the_best_weights(self, overfitted, split_for_overfit):
        # The bookkeeping tests above only prove the best loss was *recorded*.
        # This one re-scores the fitted model on the same validation rows and
        # checks it matches, which is only true if the weights were rewound.
        # Without the rewind the model carries the weights from three epochs
        # later, which are measurably worse.
        import torch
        import torch.nn as nn

        history = overfitted.history_
        rows = np.array(history["validation_index"])
        images = split_for_overfit.images_train[rows]
        labels = split_for_overfit.labels_train[rows]

        overfitted.model_.eval()
        with torch.no_grad():
            logits = overfitted.model_(overfitted._to_tensor(images))
            loss = nn.CrossEntropyLoss()(
                logits, torch.from_numpy(labels.astype(np.int64))).item()

        assert loss == pytest.approx(history["best_validation_loss"], abs=1e-5)

    def test_validation_loss_actually_rose(self, overfitted):
        # Confirms the fixture set up the situation it claims to.
        losses = [e["validation_loss"] for e in overfitted.history_["per_epoch"]]
        assert losses[-1] > min(losses)


class TestCnnValidationGuard:
    """The validation slice must be able to hold every class.

    Having enough samples is not the same as having enough room. A 15% slice
    of 12 images is 2, which cannot cover 3 classes, and scikit-learn refuses
    to stratify it. Found by running the whole pipeline on the 15-image
    fixture, where every unit test had already passed.
    """

    def small_split(self, per_class, class_count=3):
        names = ["cat", "dog", "horse"][:class_count]
        labels = [name for name in names for _ in range(per_class)]
        tensor = np.zeros((len(labels), 16, 16, 3), dtype=np.uint8)
        for index in range(len(labels)):
            tensor[index] = (index % 7) * 30
        return make_split(preprocess(load_array(tensor, labels), "rgb"))

    def test_trains_when_the_slice_cannot_cover_every_class(self):
        # 15 images -> 12 training -> a 15% slice is 2, short of 3 classes.
        split = self.small_split(per_class=5)
        model = SimpleCNN(epochs=2)
        model.fit(split.images_train, split.labels_train)
        assert model.history_["validation_samples"] == 0
        assert model.history_["training_samples"] == len(split.train_index)

    def test_still_predicts_every_test_image(self):
        split = self.small_split(per_class=5)
        model = SimpleCNN(epochs=2).fit(split.images_train, split.labels_train)
        assert len(model.predict(split.images_test)) == len(split.test_index)

    def test_uses_a_validation_slice_once_there_is_room(self):
        split = self.small_split(per_class=20)
        model = SimpleCNN(epochs=2).fit(split.images_train, split.labels_train)
        assert model.history_["validation_samples"] >= 3
