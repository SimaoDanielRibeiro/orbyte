"""Regressão linear por mínimos quadrados (scikit-learn): a solução de uma vez.

Calcula de uma só vez os pesos que minimizam o erro quadrático médio (a fórmula dos
mínimos quadrados, resolvida de forma numericamente estável). É o ponto de chegada da
versão didática por gradiente descendente (``linear_regression_gd.py``). Ficha
didática: ``docs/algorithms/linear_regression.md``.
"""

from typing import Any, override

from pydantic import Field
from sklearn.linear_model import LinearRegression as SklearnLinearRegression

from ml_lab.core.foundation.types import TaskType
from ml_lab.core.models.adapters.sklearn_adapter import SklearnModel
from ml_lab.core.models.base import MODELS, Implementation, ModelFamily, ModelInfo
from ml_lab.core.schemas.hyperparameters import Hyperparameters


class LinearRegressionParams(Hyperparameters):
    """Hiperparâmetros da regressão linear por mínimos quadrados."""

    fit_intercept: bool = Field(
        default=True,
        title="Learn the intercept (b)",
        description="Without an intercept, the line has to go through the origin.",
    )


def linear_state(estimator: Any) -> dict[str, Any]:
    """Os pesos e a ordenada de um modelo linear de regressão do scikit-learn."""
    return {
        "coefficients": estimator.coef_.tolist(),
        "intercept": float(estimator.intercept_),
    }


@MODELS.register("linear_regression")
class LinearRegression(SklearnModel[LinearRegressionParams]):
    """Regressão linear do scikit-learn (mínimos quadrados)."""

    info = ModelInfo(
        name="linear_regression",
        display_name="Linear regression",
        summary=(
            "The line (or plane) that minimises the squared error, computed in one go: the "
            "first model of any regression."
        ),
        task_types=frozenset({TaskType.REGRESSION}),
        family=ModelFamily.LINEAR,
        implementation=Implementation.SKLEARN,
        iterative=False,
        supports_proba=False,
        hyperparameters=LinearRegressionParams,
        doc="linear_regression.md",
    )

    @override
    def build_estimator(self) -> SklearnLinearRegression:
        return SklearnLinearRegression(fit_intercept=self.hyperparameters.fit_intercept)

    @override
    def state(self) -> dict[str, Any]:
        return linear_state(self.estimator_)
