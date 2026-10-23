"""O contrato LabModel, o adaptador do scikit-learn e a regressão logística."""

import warnings
from typing import Any, override

import numpy as np
import pytest
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression

from ml_lab.core.foundation.errors import (
    CapabilityError,
    HyperparameterError,
    InvalidSeedError,
    ModelError,
)
from ml_lab.core.foundation.types import TaskType
from ml_lab.core.models.base import (
    MODELS,
    Features,
    Implementation,
    LabModel,
    ModelFamily,
    ModelInfo,
    Target,
    capture_warnings,
)
from ml_lab.core.models.supervised.classification.logistic_regression import (
    LogisticRegression,
    LogisticRegressionParams,
)
from ml_lab.core.schemas.hyperparameters import Hyperparameters
from ml_lab.core.training.context import RecordingTrainingContext, TrainingContext


class Threshold(Hyperparameters):
    """Um único hiperparâmetro, para o modelo de teste."""

    value: float = 0.0


class Sign(LabModel[Threshold]):
    """Classe 1 se a primeira feature passar do limiar; não dá probabilidades."""

    info = ModelInfo(
        name="sign",
        display_name="Sinal",
        summary="Para testes.",
        task_types=frozenset({TaskType.BINARY_CLASSIFICATION}),
        family=ModelFamily.LINEAR,
        implementation=Implementation.DIDACTIC,
        iterative=True,
        supports_proba=False,
        hyperparameters=Threshold,
        doc="nenhuma.md",
    )

    @override
    def _fit(self, X: Features, y: Target | None, context: TrainingContext) -> None:
        context.report_step(1, loss=0.5)

    @override
    def _predict(self, X: Features) -> NDArray[Any]:
        return (X[:, 0] > self.hyperparameters.value).astype(np.int64)


def separable(n: int = 40) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, 2))
    return X, (X[:, 0] > 0).astype(np.int64)


# --- contrato -----------------------------------------------------------------------------


def test_hyperparameters_default_to_their_class_and_are_type_checked() -> None:
    assert Sign().hyperparameters == Threshold()
    assert Sign(Threshold(value=1.0)).hyperparameters.value == 1.0
    with pytest.raises(ModelError, match="Threshold"):
        Sign(LogisticRegressionParams())  # type: ignore[arg-type]


def test_seeds_are_validated() -> None:
    with pytest.raises(InvalidSeedError):
        Sign(seed=-1)


@pytest.mark.parametrize(
    ("X", "y", "message"),
    [
        (np.zeros(4), None, "matriz"),
        (np.zeros((0, 2)), None, "matriz"),
        (np.zeros((4, 2)), np.zeros(3), "um valor por linha"),
    ],
)
def test_bad_shapes_are_explained(
    X: NDArray[np.float64], y: NDArray[np.float64] | None, message: str
) -> None:
    with pytest.raises(ModelError, match=message):
        Sign().fit(X, y)


def test_the_context_receives_the_announced_steps() -> None:
    context = RecordingTrainingContext()
    X, y = separable()

    Sign().fit(X, y, context)

    assert context.steps == [(1, 0.5, {})]


def test_models_without_probabilities_say_so() -> None:
    X, y = separable()
    model = Sign().fit(X, y)

    with pytest.raises(CapabilityError, match="não dá probabilidades"):
        model.predict_proba(X)
    assert model.classes() is None
    assert model.state() == {}


def test_capture_warnings_returns_the_result_and_the_warnings() -> None:
    def noisy() -> int:
        warnings.warn("cuidado", UserWarning, stacklevel=1)
        return 42

    result, caught = capture_warnings(noisy)

    assert result == 42
    assert [str(warning.message) for warning in caught] == ["cuidado"]


def test_models_are_discovered_and_filtered_by_task() -> None:
    assert MODELS.get("logistic_regression") is LogisticRegression
    assert "logistic_regression" in MODELS.for_task(TaskType.MULTICLASS_CLASSIFICATION)
    assert MODELS.for_task(TaskType.REGRESSION) == {}


# --- regressão logística -------------------------------------------------------------


def test_logistic_regression_is_the_scikit_learn_model_with_our_hyperparameters() -> None:
    X, y = separable()
    params = LogisticRegressionParams(C=0.5, class_weight="balanced", max_iter=200)

    ours = LogisticRegression(params, seed=3).fit(X, y)
    theirs = SklearnLogisticRegression(C=0.5, class_weight="balanced", max_iter=200).fit(X, y)

    assert np.allclose(ours.state()["coefficients"], theirs.coef_)
    assert np.allclose(ours.predict_proba(X), theirs.predict_proba(X))
    classes = ours.classes()
    assert classes is not None
    assert classes.tolist() == [0, 1]


def test_the_state_shows_what_the_model_learned() -> None:
    X, y = separable()

    state = LogisticRegression().fit(X, y).state()

    assert np.asarray(state["coefficients"]).shape == (1, 2)
    assert len(state["intercept"]) == 1
    assert state["classes"] == [0, 1]
    assert state["iterations"] >= 1
    assert state["coefficients"][0][0] > 0  # a classe 1 é a da primeira feature positiva


def test_library_warnings_become_didactic_notices() -> None:
    # Features em escalas muito diferentes e só 10 iterações: o otimizador não chega lá.
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4)) * np.array([1, 100, 1_000, 10_000])
    y = (X[:, 0] + X[:, 1] / 100 > 0).astype(np.int64)

    model = LogisticRegression(LogisticRegressionParams(max_iter=10)).fit(X, y)

    assert [notice.code for notice in model.notices] == ["convergence"]
    assert "escala as features" in model.notices[0].message
    model.fit(*separable())
    assert model.notices == ()  # cada treino começa sem avisos


def test_invalid_hyperparameters_are_explained_in_portuguese() -> None:
    with pytest.raises(HyperparameterError) as error:
        LogisticRegressionParams.parse(
            {"C": 0, "class_weight": "auto"}, model="logistic_regression"
        )

    assert error.value.problems == (
        "C (inverso da regularização) tem de ser pelo menos 0,0001.",
        "Peso das classes tem de ser um destes: 'none' or 'balanced'.",
    )
