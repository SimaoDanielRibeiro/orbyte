"""Regressão linear por gradiente descendente: implementação didática em NumPy.

Procura a reta (ou o plano) ``ŷ = w · x + b`` que minimiza o erro quadrático médio,
``EQM = média((ŷ - y)²)``. A solução tem fórmula fechada (mínimos quadrados, o que o
``LinearRegression`` do scikit-learn calcula), mas aqui chega-se lá aos poucos, para se
ver o caminho:

1. calcula os resíduos ``r = ŷ - y`` com os pesos atuais;
2. o gradiente do EQM é ``2 · média(r · x)`` (para ``b``, ``2 · média(r)``); usa-se
   metade, o gradiente de ``EQM / 2``, que tem o mesmo mínimo;
3. dá um passo contra o gradiente: ``w ← w - taxa · gradiente``.

Com uma só feature, cada época é um ponto ``(w, b)`` na superfície do erro, e o treino
é uma trajetória a descer até ao fundo. Ficha didática:
``docs/algorithms/linear_regression.md``.
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


class LinearRegressionGDParams(Hyperparameters):
    """Hiperparâmetros da regressão linear por gradiente descendente."""

    learning_rate: float = Field(
        default=0.1,
        ge=0.0001,
        le=5,
        title="Learning rate",
        description=(
            "The size of each step. Too small, training drags on; too large, the error grows "
            "instead of falling (with standardised features, above ~1 it usually blows up)."
        ),
        json_schema_extra={"x-scale": "log"},
    )
    max_epochs: int = Field(
        default=1_000,
        ge=10,
        le=20_000,
        title="Maximum epochs",
        description="How many passes over the training data, at most.",
    )
    tol: float = Field(
        default=1e-6,
        ge=1e-10,
        le=0.01,
        title="Stopping tolerance",
        description=(
            "Stops when the gradient (measured on the target's scale) gets smaller than this."
        ),
        json_schema_extra={"x-scale": "log"},
    )
    fit_intercept: bool = Field(
        default=True,
        title="Learn the intercept (b)",
        description="Without an intercept, the line has to go through the origin.",
    )


@MODELS.register("linear_regression_gd")
class LinearRegressionGD(LabModel[LinearRegressionGDParams]):
    """Regressão linear treinada por gradiente descendente, época a época."""

    info = ModelInfo(
        name="linear_regression_gd",
        display_name="Linear regression (gradient descent)",
        summary=(
            "Linear regression written in NumPy and trained step by step: you see the line fit "
            "itself and the error fall down to the least-squares solution."
        ),
        task_types=frozenset({TaskType.REGRESSION}),
        family=ModelFamily.LINEAR,
        implementation=Implementation.DIDACTIC,
        iterative=True,
        supports_proba=False,
        hyperparameters=LinearRegressionGDParams,
        doc="linear_regression.md",
        step_limit="max_epochs",
    )

    def __init__(
        self, hyperparameters: LinearRegressionGDParams | None = None, *, seed: int = 0
    ) -> None:
        super().__init__(hyperparameters, seed=seed)
        self._weights = np.zeros(0)
        self._bias = 0.0
        self._epochs = 0
        self._converged = False

    @override
    def state(self) -> dict[str, Any]:
        return {
            "coefficients": self._weights.tolist(),
            "intercept": self._bias,
            "iterations": self._epochs,
            "converged": self._converged,
        }

    @override
    def _fit(self, X: Features, y: Target | None, context: TrainingContext) -> None:
        if y is None:
            raise ModelError("Linear regression needs the target to learn.")
        params = self.hyperparameters
        target = y.astype(np.float64)
        n_samples, n_features = X.shape
        self._weights = np.zeros(n_features)
        self._bias = 0.0
        self._epochs = 0
        self._converged = False
        # O gradiente cresce com a escala do alvo (preços na casa dos milhares, por exemplo):
        # medido nessa escala, o critério de paragem serve para qualquer alvo.
        scale = max(float(np.std(target)), 1e-12)
        residuals = X @ self._weights + self._bias - target
        initial_loss = float(np.mean(residuals**2))

        for epoch in range(1, params.max_epochs + 1):
            weights_gradient = X.T @ residuals / n_samples
            bias_gradient = float(residuals.mean()) if params.fit_intercept else 0.0
            self._weights = self._weights - params.learning_rate * weights_gradient
            self._bias -= params.learning_rate * bias_gradient
            self._epochs = epoch
            residuals = X @ self._weights + self._bias - target
            loss = float(np.mean(residuals**2))
            context.report_step(epoch, loss=loss)
            if context.wants_snapshot(epoch):
                context.report_snapshot(epoch, self.snapshot_state(context.probe))
            if has_diverged(loss, initial_loss):
                self._note(diverged_notice(params.learning_rate))
                return
            gradient_norm = math.sqrt(float(np.sum(weights_gradient**2)) + bias_gradient**2)
            if gradient_norm / scale < params.tol:
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
        result: NDArray[np.float64] = X @ self._weights + self._bias
        return result
