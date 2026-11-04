"""Os modelos didáticos por gradiente descendente chegam aonde o scikit-learn chega.

Minimizam a mesma função que o ``LogisticRegression`` e o ``LinearRegression`` do
scikit-learn; com tolerâncias apertadas dos dois lados, os pesos coincidem.
"""

import json
import math
from collections.abc import Callable
from itertools import pairwise

import numpy as np
import pytest
from numpy.typing import NDArray
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression, LogisticRegression

from ml_lab.core.foundation.errors import ModelError
from ml_lab.core.models.supervised.classification.logistic_regression_gd import (
    LogisticRegressionGD,
    LogisticRegressionGDParams,
)
from ml_lab.core.models.supervised.gradient_descent import DIVERGENCE_FACTOR, has_diverged
from ml_lab.core.models.supervised.regression.linear_regression_gd import (
    LinearRegressionGD,
    LinearRegressionGDParams,
)
from ml_lab.core.training.context import RecordingTrainingContext

type Data = tuple[NDArray[np.float64], NDArray[np.int64]]


def standardized(X: NDArray[np.float64]) -> NDArray[np.float64]:
    result: NDArray[np.float64] = (X - X.mean(axis=0)) / X.std(axis=0)
    return result


def iris() -> Data:
    data = load_iris()
    return standardized(np.asarray(data.data, dtype=np.float64)), data.target.astype(np.int64)


def binary() -> Data:
    X, y = iris()
    keep = y > 0  # versicolor contra virginica: as duas que se misturam
    return X[keep], (y[keep] == 2).astype(np.int64)


def regression() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    return X, X @ np.array([1.5, -2.0, 0.5]) + 3.0 + rng.normal(scale=0.3, size=200)


TIGHT_LOGISTIC = LogisticRegressionGDParams(learning_rate=1.0, max_epochs=20_000, tol=1e-8)


@pytest.mark.parametrize("data", [binary, iris], ids=["binária", "multiclasse"])
def test_logistic_gd_reaches_the_scikit_learn_solution(data: Callable[[], Data]) -> None:
    X, y = data()
    ours = LogisticRegressionGD(TIGHT_LOGISTIC).fit(X, y)
    theirs = LogisticRegression(C=1.0, tol=1e-12, max_iter=100_000).fit(X, y)

    assert ours.state()["converged"]
    assert np.allclose(ours.state()["coefficients"], theirs.coef_, atol=1e-4)
    assert np.allclose(ours.state()["intercept"], theirs.intercept_, atol=1e-4)
    assert np.allclose(ours.predict_proba(X), theirs.predict_proba(X), atol=1e-5)
    assert np.array_equal(ours.predict(X), theirs.predict(X))


def test_linear_gd_reaches_the_least_squares_solution() -> None:
    X, y = regression()
    params = LinearRegressionGDParams(learning_rate=0.5, max_epochs=20_000, tol=1e-10)

    ours = LinearRegressionGD(params).fit(X, y)
    theirs = LinearRegression().fit(X, y)

    assert ours.state()["converged"]
    assert np.allclose(ours.state()["coefficients"], theirs.coef_, atol=1e-6)
    assert ours.state()["intercept"] == pytest.approx(theirs.intercept_, abs=1e-6)


def test_with_small_steps_the_loss_only_goes_down() -> None:
    X, y = iris()
    context = RecordingTrainingContext()

    LogisticRegressionGD(LogisticRegressionGDParams(learning_rate=0.1)).fit(X, y, context)

    losses = [loss for _, loss, _ in context.steps if loss is not None]
    assert len(losses) == len(context.steps)
    assert all(after <= before + 1e-12 for before, after in pairwise(losses))
    assert context.steps[0][2]["accuracy"] <= context.steps[-1][2]["accuracy"]


def test_a_learning_rate_too_large_stops_with_a_warning_and_finite_predictions() -> None:
    X, y = regression()
    # Sem padronização, a taxa que serve a uma feature é grande demais para outra.
    X = X * np.array([1.0, 10.0, 100.0])
    params = LinearRegressionGDParams(learning_rate=5.0)

    model = LinearRegressionGD(params).fit(X, y)

    assert [notice.code for notice in model.notices] == ["diverged"]
    assert model.state()["iterations"] < params.max_epochs
    assert np.isfinite(model.predict(X)).all()


def test_the_loss_has_exploded_when_it_is_no_longer_a_number_or_far_worse() -> None:
    assert has_diverged(math.inf, 1.0)
    assert has_diverged(math.nan, 1.0)
    assert has_diverged(DIVERGENCE_FACTOR * 2, 1.0)
    assert not has_diverged(10.0, 1.0)
    assert not has_diverged(0.0, 0.0)


def test_without_enough_epochs_the_model_says_it_did_not_converge() -> None:
    X, y = iris()

    model = LogisticRegressionGD(LogisticRegressionGDParams(max_epochs=10)).fit(X, y)

    assert [notice.code for notice in model.notices] == ["not_converged"]
    assert model.state()["iterations"] == 10
    assert not model.state()["converged"]


@pytest.mark.parametrize("model", [LogisticRegressionGD, LinearRegressionGD])
def test_a_stop_request_keeps_what_was_learned_without_warnings(
    model: type[LogisticRegressionGD] | type[LinearRegressionGD],
) -> None:
    X, y = iris()
    context = RecordingTrainingContext(stop_after=3)

    fitted = model().fit(X, y.astype(np.float64) if model is LinearRegressionGD else y, context)

    assert fitted.state()["iterations"] == 3
    assert fitted.notices == ()
    assert fitted.is_fitted


def test_snapshots_carry_the_predictions_on_the_probe() -> None:
    X, y = binary()
    probe = X[:7]
    context = RecordingTrainingContext(probe)

    LogisticRegressionGD(LogisticRegressionGDParams(max_epochs=20)).fit(X, y, context)

    assert [step for step, _ in context.snapshots] == list(range(1, 21))
    first, last = context.snapshots[0][1], context.snapshots[-1][1]
    assert np.asarray(first["probe"]).shape == (7, 2)
    assert not np.allclose(first["probe"], last["probe"])  # a fronteira mexeu-se
    json.dumps({key: value for key, value in last.items() if key != "probe"})


def test_the_learned_state_names_the_classes_and_one_weight_vector_for_two() -> None:
    X, y = binary()

    state = LogisticRegressionGD().fit(X, y).state()

    assert state["classes"] == [0, 1]
    assert np.asarray(state["coefficients"]).shape == (1, 4)
    json.dumps(state)


def test_a_single_class_is_refused() -> None:
    X, _ = binary()

    with pytest.raises(ModelError, match="two classes"):
        LogisticRegressionGD().fit(X, np.zeros(len(X), dtype=np.int64))
