"""Logistic Regression, Decision Tree, Random Forest and SVM.

This module only *describes* the models. Fitting, timing and scoring belong to
evaluation, so that every model - classical or neural - is measured by the
same code and the comparison stays fair.

Scaling is the part worth reading carefully. Section 6.1 requires Logistic
Regression, SVM and the fully connected network to use a scaler fitted on the
training data only. The tempting shortcut is::

    scaled = StandardScaler().fit_transform(features)   # WRONG
    train, test = split(scaled)

That fits the scaler on every image, so the mean and variance of the test set
inform the transformation applied to the training set. Nothing errors, the
accuracy simply comes out slightly too high, and section 7 prohibits it.

Wrapping the scaler and the classifier in a ``Pipeline`` removes the choice:
``pipeline.fit(train)`` fits the scaler on the training rows only, and
``pipeline.predict(test)`` reuses those training statistics. The mistake stops
being something to remember not to make.

Decision Tree and Random Forest are deliberately left unscaled. They split on
thresholds within a single feature, so a monotonic rescaling of that feature
changes nothing about the tree; adding a scaler would cost time and imply a
dependence that is not there.
"""

from dataclasses import dataclass, field
from typing import Callable

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from ._config import RANDOM_SEED


@dataclass
class ModelSpec:
    """A model the benchmark will train, with the metadata around it.

    ``key`` is the file-safe form used for the per-model output files section
    9 requires, so that ``name`` can stay readable in the summary table while
    ``key`` names ``confusion_matrices/logistic_regression.png``.
    """

    key: str
    name: str
    build: Callable
    scaled: bool
    parameters: dict = field(default_factory=dict)
    kind: str = "classical"

    def __call__(self):
        """Construct a fresh, unfitted estimator."""
        return self.build()


def _scaled(estimator) -> Pipeline:
    """Attach a scaler that can only ever see training rows."""
    return Pipeline([("scaler", StandardScaler()), ("classifier", estimator)])


def logistic_regression_spec() -> ModelSpec:
    parameters = {"max_iter": 1000, "random_state": RANDOM_SEED}
    return ModelSpec(
        key="logistic_regression",
        name="Logistic Regression",
        build=lambda: _scaled(LogisticRegression(**parameters)),
        scaled=True,
        parameters=parameters,
    )


def decision_tree_spec() -> ModelSpec:
    parameters = {"random_state": RANDOM_SEED}
    return ModelSpec(
        key="decision_tree",
        name="Decision Tree",
        build=lambda: DecisionTreeClassifier(**parameters),
        scaled=False,
        parameters=parameters,
    )


def random_forest_spec() -> ModelSpec:
    # n_jobs=-1 uses every core. It changes how the work is distributed, not
    # what the trees are, so a fixed random_state still reproduces exactly.
    parameters = {"n_estimators": 200, "random_state": RANDOM_SEED, "n_jobs": -1}
    return ModelSpec(
        key="random_forest",
        name="Random Forest",
        build=lambda: RandomForestClassifier(**parameters),
        scaled=False,
        parameters=parameters,
    )


def svm_spec() -> ModelSpec:
    # An RBF SVM on 12,288 features is the slow one: its cost grows with the
    # square of the sample count. On a 5,000-image RGB run expect minutes.
    # That is a result worth reporting, which is why section 8 asks for
    # training time, not a problem to design away.
    parameters = {"kernel": "rbf", "random_state": RANDOM_SEED}
    return ModelSpec(
        key="svm",
        name="SVM",
        build=lambda: _scaled(SVC(**parameters)),
        scaled=True,
        parameters=parameters,
    )


def classical_model_specs() -> list:
    """The four classical models, in the order the summary table lists them."""
    return [
        logistic_regression_spec(),
        decision_tree_spec(),
        random_forest_spec(),
        svm_spec(),
    ]
