"""Fully connected network and CNN.

Two models with quite different needs.

The fully connected network is one more scikit-learn estimator, scaled like
Logistic Regression and SVM, consuming the same flattened feature vectors as
the classical models.

The CNN is the only model that needs the image tensor rather than flattened
features, so ``ModelSpec.kind`` records which representation each model wants
and evaluation hands over the right one. It is wrapped in a small class with
``fit`` and ``predict`` so that every model in the benchmark - classical or
neural - is driven by identical code and timed the same way. A separate
training path for the CNN would make its numbers quietly incomparable.

Two details worth knowing:

* Section 4.2 stores images channels-last as ``(N, 64, 64, C)``, which is the
  Keras convention. PyTorch convolutions want channels-first, so the tensor is
  permuted at the boundary rather than stored differently. Storage follows the
  assignment; the permute is an implementation detail of this file.
* Section 6.2 asks for a softmax output layer. ``CrossEntropyLoss`` applies
  log-softmax internally and expects raw logits, so the final layer emits
  logits and the softmax lives in the loss. Applying softmax twice would
  flatten the gradients and train badly.

Early stopping watches a validation split carved out of the *training* half
only. Section 7 forbids the test set from influencing training, and choosing
when to stop is influence. The consequence is that the neural models train on
about 68% of the data where the classical models get the full 80% - an
asymmetry worth stating in the report rather than leaving to be discovered.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ._config import RANDOM_SEED
from .classical_models import ModelSpec

# Fraction of the *training* half held back to decide when to stop.
VALIDATION_FRACTION = 0.15

# Section 6.2's training settings.
MAX_EPOCHS = 20
BATCH_SIZE = 32
EARLY_STOPPING_PATIENCE = 3
LEARNING_RATE = 1e-3

# How much better a validation loss must be to count as an improvement.
# scikit-learn's MLPClassifier uses 1e-4 here (its `tol`), which has an
# awkward consequence: an epoch that genuinely beats the previous best by
# less than the threshold is treated as no improvement, so the weights
# finally kept are not the best ones seen. Zero avoids that - any decrease
# counts, patience counts epochs with no decrease at all, and the restored
# weights are the genuine minimum.
MIN_IMPROVEMENT = 0.0


def neural_network_spec() -> ModelSpec:
    """A fully connected network on the flattened features."""
    parameters = {
        "hidden_layer_sizes": (128, 64),
        "max_iter": 100,
        "early_stopping": True,
        "validation_fraction": VALIDATION_FRACTION,
        "n_iter_no_change": EARLY_STOPPING_PATIENCE,
        "random_state": RANDOM_SEED,
    }
    return ModelSpec(
        key="neural_network",
        name="Neural Network",
        build=lambda: Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", MLPClassifier(**parameters)),
        ]),
        scaled=True,
        parameters={**parameters, "hidden_layer_sizes": list(parameters["hidden_layer_sizes"])},
        kind="neural",
    )


class SimpleCNN:
    """The small convolutional network section 6.2 prescribes.

    Exposes ``fit`` and ``predict`` so the benchmark can drive it exactly as
    it drives the scikit-learn models.

    Architecture, in order: Conv2D 32 filters 3x3 with ReLU, 2x2 max pooling,
    Conv2D 64 filters 3x3 with ReLU, 2x2 max pooling, flatten, dense 128 with
    ReLU, dropout 0.30, and a linear output with one unit per class. Trained
    from scratch with Adam and cross-entropy; no pretrained weights are used
    anywhere.
    """

    def __init__(
        self,
        epochs: int = MAX_EPOCHS,
        batch_size: int = BATCH_SIZE,
        patience: int = EARLY_STOPPING_PATIENCE,
        validation_fraction: float = VALIDATION_FRACTION,
        learning_rate: float = LEARNING_RATE,
        random_state: int = RANDOM_SEED,
        device: str = "cpu",
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.learning_rate = learning_rate
        self.random_state = random_state
        # CPU by default so the reported numbers are identical on whatever
        # machine reproduces them. Metal is faster but its floating-point
        # results differ slightly from CPU.
        self.device = device

        self.model_ = None
        self.classes_ = None
        self.history_ = {}

    # -- architecture -------------------------------------------------------

    def _build(self, channels: int, class_count: int):
        import torch.nn as nn

        features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )

        # Measured rather than hard-coded, so the network still assembles if
        # the internal image size ever changes.
        import torch

        with torch.no_grad():
            flat = features(torch.zeros(1, channels, *self._image_size_)).shape[1]

        return nn.Sequential(
            features,
            nn.Linear(flat, 128),
            nn.ReLU(),
            nn.Dropout(0.30),
            # Logits: CrossEntropyLoss applies the softmax section 6.2 asks for.
            nn.Linear(128, class_count),
        )

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _to_tensor(images: np.ndarray):
        """Channels-last (N, H, W, C) storage to channels-first for PyTorch."""
        import torch

        return torch.from_numpy(np.ascontiguousarray(images)).permute(0, 3, 1, 2).float()

    def _validation_split(self, labels: np.ndarray):
        """Stratified hold-out taken from the training half only."""
        from sklearn.model_selection import train_test_split

        positions = np.arange(len(labels))
        counts = np.bincount(labels)
        class_count = int((counts > 0).sum())

        # Stratifying needs every class on both sides of the division, so the
        # validation slice has to be at least as large as the class count -
        # having enough *samples* is not enough. A 15% slice of 12 images is
        # 2, which cannot cover 3 classes, and scikit-learn refuses. When the
        # training set is too small to spare a usable slice, train on all of
        # it and let the epoch cap stop the run.
        validation_size = int(np.floor(len(labels) * self.validation_fraction))
        if len(labels) < 2 or counts.min() < 2 or validation_size < class_count:
            return positions, np.empty(0, dtype=int)
        return train_test_split(
            positions,
            test_size=self.validation_fraction,
            stratify=labels,
            random_state=self.random_state,
        )

    # -- the scikit-learn shaped interface ----------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SimpleCNN":
        import torch
        import torch.nn as nn

        torch.manual_seed(self.random_state)

        self.classes_ = np.unique(y)
        self._image_size_ = X.shape[1:3]
        channels = X.shape[3]

        train_positions, validation_positions = self._validation_split(y)

        device = torch.device(self.device)
        self.model_ = self._build(channels, len(self.classes_)).to(device)

        inputs = self._to_tensor(X).to(device)
        targets = torch.from_numpy(y.astype(np.int64)).to(device)

        train_inputs = inputs[train_positions]
        train_targets = targets[train_positions]
        has_validation = len(validation_positions) > 0

        optimiser = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()

        generator = torch.Generator().manual_seed(self.random_state)
        best_loss, best_state, epochs_without_improvement = np.inf, None, 0
        history = []

        for epoch in range(self.epochs):
            self.model_.train()
            order = torch.randperm(len(train_inputs), generator=generator)
            running = 0.0

            for start in range(0, len(order), self.batch_size):
                batch = order[start:start + self.batch_size]
                optimiser.zero_grad()
                loss = criterion(self.model_(train_inputs[batch]), train_targets[batch])
                loss.backward()
                optimiser.step()
                running += loss.item() * len(batch)

            train_loss = running / max(1, len(train_inputs))

            if has_validation:
                self.model_.eval()
                with torch.no_grad():
                    watched = criterion(
                        self.model_(inputs[validation_positions]),
                        targets[validation_positions],
                    ).item()
            else:
                watched = train_loss

            history.append({"epoch": epoch + 1, "train_loss": train_loss,
                            "validation_loss": watched})

            if watched < best_loss - MIN_IMPROVEMENT:
                best_loss = watched
                epochs_without_improvement = 0
                # Keep the best weights, not the last. Without this, early
                # stopping returns a model three epochs *past* its best.
                best_state = {k: v.detach().clone()
                              for k, v in self.model_.state_dict().items()}
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.patience:
                    break

        if best_state is not None:
            self.model_.load_state_dict(best_state)

        self.history_ = {
            "epochs_run": len(history),
            "epochs_max": self.epochs,
            "stopped_early": len(history) < self.epochs,
            "best_validation_loss": None if best_loss == np.inf else float(best_loss),
            "validation_samples": int(len(validation_positions)),
            "training_samples": int(len(train_positions)),
            # Recorded so the restored weights can be checked against the
            # epoch they came from, and so the report can say exactly which
            # rows early stopping watched.
            "validation_index": [int(i) for i in validation_positions],
            "per_epoch": history,
        }
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        import torch

        if self.model_ is None:
            raise RuntimeError("SimpleCNN.predict called before fit.")

        self.model_.eval()
        device = torch.device(self.device)
        predictions = []

        with torch.no_grad():
            inputs = self._to_tensor(X).to(device)
            # Batched so a large test set cannot exhaust memory in one pass.
            for start in range(0, len(inputs), self.batch_size):
                logits = self.model_(inputs[start:start + self.batch_size])
                predictions.append(logits.argmax(dim=1).cpu().numpy())

        return np.concatenate(predictions) if predictions else np.empty(0, dtype=np.int64)


def simple_cnn_spec() -> ModelSpec:
    parameters = {
        "architecture": "conv32-pool-conv64-pool-dense128-dropout0.30-softmax",
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "loss": "cross_entropy",
        "max_epochs": MAX_EPOCHS,
        "batch_size": BATCH_SIZE,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "min_improvement": MIN_IMPROVEMENT,
        "validation_fraction": VALIDATION_FRACTION,
        "pretrained": False,
        "device": "cpu",
        "random_state": RANDOM_SEED,
    }
    return ModelSpec(
        key="simple_cnn",
        name="Simple CNN",
        build=lambda: SimpleCNN(),
        scaled=False,  # pixels are already normalized to [0, 1]
        parameters=parameters,
        kind="cnn",
    )


def neural_model_specs() -> list:
    """The two neural models, in the order the summary table lists them."""
    return [neural_network_spec(), simple_cnn_spec()]
