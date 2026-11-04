"""Regressão logística por gradiente descendente: implementação didática em NumPy.

Minimiza a mesma função que o ``LogisticRegression`` do scikit-learn (log-loss mais
a penalização L2 com o mesmo ``C``), por isso, quando converge, chega aos mesmos
pesos. A diferença está no caminho: o scikit-learn usa o otimizador L-BFGS, que lá
chega em poucas dezenas de passos; o gradiente descendente dá centenas de passos
pequenos, sempre na direção que faz descer a loss, e cada um fica fotografado. É
isso que permite ver a fronteira de decisão a mover-se.

Em cada época (uma passagem por todos os exemplos de treino):

1. calcula as probabilidades com os pesos atuais (sigmoide com 2 classes, softmax
   com mais);
2. calcula o gradiente: a média de ``(p - y) · x`` mais ``w / (C · n)``;
3. dá um passo contra o gradiente: ``w ← w - taxa · gradiente``.

Ficha didática: ``docs/algorithms/logistic_regression.md``.
"""

import math
from typing import Any, override

import numpy as np
from numpy.typing import NDArray
from pydantic import Field

from ml_lab.core.foundation.errors import ModelError
from ml_lab.core.foundation.notices import Notice
from ml_lab.core.foundation.types import TaskType
from ml_lab.core.models.base import (
    MODELS,
    Features,
    Implementation,
    LabModel,
    ModelFamily,
    ModelInfo,
    Target,
)
from ml_lab.core.models.supervised.gradient_descent import diverged_notice, has_diverged
from ml_lab.core.schemas.hyperparameters import Hyperparameters
from ml_lab.core.training.context import TrainingContext
from ml_lab.formatting import format_count


class LogisticRegressionGDParams(Hyperparameters):
    """Hiperparâmetros da regressão logística por gradiente descendente."""

    learning_rate: float = Field(
        default=1.0,
        ge=0.0001,
        le=50,
        title="Learning rate",
        description=(
            "The size of each step. Too small, training drags on; too large, the loss jumps "
            "from side to side or blows up."
        ),
        json_schema_extra={"x-scale": "log"},
    )
    max_epochs: int = Field(
        default=2_000,
        ge=10,
        le=20_000,
        title="Maximum epochs",
        description="How many passes over the training data, at most.",
    )
    C: float = Field(
        default=1.0,
        ge=0.0001,
        le=10_000,
        title="C (inverse of regularisation)",
        description="The same C as scikit-learn's: the smaller it is, the smaller the weights.",
        json_schema_extra={"x-scale": "log", "x-search": {"range": [0.01, 100]}},
    )
    tol: float = Field(
        default=1e-6,
        ge=1e-10,
        le=0.01,
        title="Stopping tolerance",
        description="Stops when the gradient gets smaller than this: it reached the minimum.",
        json_schema_extra={"x-scale": "log"},
    )
    fit_intercept: bool = Field(
        default=True,
        title="Learn the intercept (b)",
        description="Without an intercept, the decision boundary has to go through the origin.",
    )


@MODELS.register("logistic_regression_gd")
class LogisticRegressionGD(LabModel[LogisticRegressionGDParams]):
    """Regressão logística treinada por gradiente descendente, época a época."""

    info = ModelInfo(
        name="logistic_regression_gd",
        display_name="Logistic regression (gradient descent)",
        summary=(
            "The same logistic regression, written in NumPy and trained step by step: you see "
            "the loss fall and the boundary move until it reaches scikit-learn's."
        ),
        task_types=frozenset({TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION}),
        family=ModelFamily.LINEAR,
        implementation=Implementation.DIDACTIC,
        iterative=True,
        supports_proba=True,
        hyperparameters=LogisticRegressionGDParams,
        doc="logistic_regression.md",
        step_limit="max_epochs",
    )

    def __init__(
        self, hyperparameters: LogisticRegressionGDParams | None = None, *, seed: int = 0
    ) -> None:
        super().__init__(hyperparameters, seed=seed)
        self._classes = np.zeros(0, dtype=np.int64)
        self._weights = np.zeros((0, 0))
        self._bias = np.zeros(0)
        self._epochs = 0
        self._converged = False

    @override
    def classes(self) -> NDArray[np.int64] | None:
        return self._classes if self._classes.size else None

    @override
    def state(self) -> dict[str, Any]:
        return {
            "coefficients": self._weights.tolist(),
            "intercept": self._bias.tolist(),
            "classes": self._classes.tolist(),
            "iterations": self._epochs,
            "converged": self._converged,
        }

    @override
    def _fit(self, X: Features, y: Target | None, context: TrainingContext) -> None:
        if y is None:
            raise ModelError("Logistic regression needs the target to learn.")
        params = self.hyperparameters
        self._classes = np.unique(y).astype(np.int64)
        n_samples, n_features = X.shape
        n_classes = len(self._classes)
        if n_classes < 2:
            raise ModelError("Logistic regression needs at least two classes in the training data.")
        # Duas classes: um só vetor de pesos (o da classe positiva), como no scikit-learn.
        n_outputs = 1 if n_classes == 2 else n_classes
        targets = _one_hot(np.searchsorted(self._classes, y), n_classes)
        if n_classes == 2:
            targets = targets[:, 1:]
        self._weights = np.zeros((n_outputs, n_features))
        self._bias = np.zeros(n_outputs)
        self._epochs = 0
        self._converged = False
        penalty = 1.0 / (params.C * n_samples)
        probabilities = self._output(X)
        initial_loss = _log_loss(probabilities, targets)

        for epoch in range(1, params.max_epochs + 1):
            error = probabilities - targets
            weights_gradient = error.T @ X / n_samples + penalty * self._weights
            bias_gradient = error.mean(axis=0) if params.fit_intercept else np.zeros(n_outputs)
            self._weights = self._weights - params.learning_rate * weights_gradient
            self._bias = self._bias - params.learning_rate * bias_gradient
            self._epochs = epoch
            probabilities = self._output(X)
            loss = _log_loss(probabilities, targets) + 0.5 * penalty * float(
                np.sum(self._weights**2)
            )
            accuracy = float(np.mean(self._decide(probabilities) == y))
            context.report_step(epoch, loss=loss, metrics={"accuracy": accuracy})
            if context.wants_snapshot(epoch):
                context.report_snapshot(epoch, self.snapshot_state(context.probe))
            if has_diverged(loss, initial_loss):
                self._note(diverged_notice(params.learning_rate))
                return
            gradient_norm = math.sqrt(
                float(np.sum(weights_gradient**2)) + float(np.sum(bias_gradient**2))
            )
            if gradient_norm < params.tol:
                self._converged = True
                return
            if context.should_stop():
                return
        self._note(
            Notice.warning(
                "not_converged",
                f"Gradient descent ran all {format_count(params.max_epochs)} epochs without "
                "reaching the minimum: raise the maximum epochs or the learning rate.",
            )
        )

    @override
    def _predict(self, X: Features) -> NDArray[Any]:
        return self._decide(self._output(X))

    @override
    def _predict_proba(self, X: Features) -> NDArray[np.float64]:
        output = self._output(X)
        if output.shape[1] == 1:  # duas classes: p(positiva) e o complemento
            return np.column_stack([1.0 - output[:, 0], output[:, 0]])
        return output

    def _output(self, X: Features) -> NDArray[np.float64]:
        scores = X @ self._weights.T + self._bias
        if scores.shape[1] == 1:
            return _sigmoid(scores)
        return _softmax(scores)

    def _decide(self, output: NDArray[np.float64]) -> NDArray[np.int64]:
        if output.shape[1] == 1:
            return self._classes[(output[:, 0] >= 0.5).astype(np.int64)]
        return self._classes[np.argmax(output, axis=1)]


def _one_hot(codes: NDArray[np.intp], n_classes: int) -> NDArray[np.float64]:
    return np.eye(n_classes)[codes]


def _sigmoid(scores: NDArray[np.float64]) -> NDArray[np.float64]:
    # 0,5·(1 + tanh(z/2)) é a sigmoide sem overflow para |z| grande.
    result: NDArray[np.float64] = 0.5 * (1.0 + np.tanh(0.5 * scores))
    return result


def _softmax(scores: NDArray[np.float64]) -> NDArray[np.float64]:
    shifted = scores - scores.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    result: NDArray[np.float64] = exponentials / exponentials.sum(axis=1, keepdims=True)
    return result


def _log_loss(probabilities: NDArray[np.float64], targets: NDArray[np.float64]) -> float:
    clipped = np.clip(probabilities, 1e-15, 1 - 1e-15)
    if clipped.shape[1] == 1:
        losses = -(targets * np.log(clipped) + (1 - targets) * np.log(1 - clipped))
        return float(np.mean(losses))
    return float(-np.mean(np.sum(targets * np.log(clipped), axis=1)))
