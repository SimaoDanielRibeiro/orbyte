"""Ridge e Lasso (scikit-learn): a regressão linear com os pesos travados.

Os dois minimizam o erro quadrático mais uma penalização dos pesos, de força ``alpha``:

- **Ridge** (L2) soma os pesos ao quadrado: todos encolhem, nenhum chega a zero. Lida
  bem com features correlacionadas, que dividem o peso entre si;
- **Lasso** (L1) soma os pesos em valor absoluto: os menos úteis ficam exatamente a
  zero, por isso também escolhe as features.

Ficha didática: ``docs/algorithms/ridge_lasso.md``.
"""

from typing import Any, override

from pydantic import Field
from sklearn.linear_model import Lasso as SklearnLasso
from sklearn.linear_model import Ridge as SklearnRidge

from ml_lab.core.foundation.types import TaskType
from ml_lab.core.models.adapters.sklearn_adapter import SklearnModel
from ml_lab.core.models.base import MODELS, Implementation, ModelFamily, ModelInfo
from ml_lab.core.models.supervised.regression.linear_regression import (
    LinearRegressionParams,
    linear_state,
)


class RidgeParams(LinearRegressionParams):
    """Hiperparâmetros da Ridge."""

    alpha: float = Field(
        default=1.0,
        ge=0.0001,
        le=10_000,
        title="Alpha (regularisation strength)",
        description=(
            "How much large weights cost. Alpha at zero is linear regression; a huge alpha "
            "shrinks everything and the model predicts little more than the mean."
        ),
        json_schema_extra={"x-scale": "log", "x-search": {"range": [0.01, 100]}},
    )


class LassoParams(LinearRegressionParams):
    """Hiperparâmetros do Lasso."""

    alpha: float = Field(
        default=0.1,
        ge=0.0001,
        le=100,
        title="Alpha (regularisation strength)",
        description=(
            "How much the weights cost. The larger it is, the more weights end up exactly at "
            "zero (the matching features stop counting)."
        ),
        json_schema_extra={"x-scale": "log", "x-search": {"range": [0.0001, 1]}},
    )
    max_iter: int = Field(
        default=1_000,
        ge=100,
        le=100_000,
        title="Maximum iterations",
        description="Lasso has no closed form: the optimiser improves the weights bit by bit.",
    )


@MODELS.register("ridge")
class Ridge(SklearnModel[RidgeParams]):
    """Regressão Ridge do scikit-learn."""

    info = ModelInfo(
        name="ridge",
        display_name="Ridge (L2 regularisation)",
        summary=(
            "Linear regression with shrunken weights: steadier when there are correlated "
            "features or more features than is wise."
        ),
        task_types=frozenset({TaskType.REGRESSION}),
        family=ModelFamily.LINEAR,
        implementation=Implementation.SKLEARN,
        iterative=False,
        supports_proba=False,
        hyperparameters=RidgeParams,
        doc="ridge_lasso.md",
    )

    @override
    def build_estimator(self) -> SklearnRidge:
        params = self.hyperparameters
        return SklearnRidge(alpha=params.alpha, fit_intercept=params.fit_intercept)

    @override
    def state(self) -> dict[str, Any]:
        return linear_state(self.estimator_)


@MODELS.register("lasso")
class Lasso(SklearnModel[LassoParams]):
    """Regressão Lasso do scikit-learn."""

    info = ModelInfo(
        name="lasso",
        display_name="Lasso (L1 regularisation)",
        summary=(
            "Linear regression that sets the weights of the least useful features to zero: it "
            "picks the features while it learns."
        ),
        task_types=frozenset({TaskType.REGRESSION}),
        family=ModelFamily.LINEAR,
        implementation=Implementation.SKLEARN,
        iterative=False,
        supports_proba=False,
        hyperparameters=LassoParams,
        doc="ridge_lasso.md",
    )

    @override
    def build_estimator(self) -> SklearnLasso:
        params = self.hyperparameters
        return SklearnLasso(
            alpha=params.alpha,
            fit_intercept=params.fit_intercept,
            max_iter=params.max_iter,
            random_state=self.seed,
        )

    @override
    def state(self) -> dict[str, Any]:
        return {**linear_state(self.estimator_), "iterations": int(self.estimator_.n_iter_)}
