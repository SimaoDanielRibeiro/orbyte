"""Regressão polinomial (scikit-learn): uma regressão linear sobre potências das features.

Um *pipeline* de três peças: cria as potências e os produtos das features até ao grau
escolhido (``x``, ``x²``, ``x³``, ``x·z``…), padroniza-as (``x¹⁵`` tem uma escala
muito diferente de ``x``) e ajusta uma Ridge. O modelo continua linear nos pesos, mas a
curva já não é uma reta. É a demonstração clássica do overfitting: graus altos passam
por todos os pontos de treino e falham no teste. Ficha didática:
``docs/algorithms/polynomial_regression.md``.
"""

from typing import Any, Final, override

from pydantic import Field
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from ml_lab.core.foundation.errors import ModelError
from ml_lab.core.foundation.types import TaskType
from ml_lab.core.models.adapters.sklearn_adapter import SklearnModel
from ml_lab.core.models.base import (
    MODELS,
    Features,
    Implementation,
    ModelFamily,
    ModelInfo,
    Target,
)
from ml_lab.core.schemas.hyperparameters import Hyperparameters
from ml_lab.core.training.context import TrainingContext
from ml_lab.formatting import format_count

MAX_POLYNOMIAL_FEATURES: Final = 5_000
"""Mais do que isto, e o grau escolhido cria features a mais para tão poucos dados."""


class PolynomialRegressionParams(Hyperparameters):
    """Hiperparâmetros da regressão polinomial."""

    degree: int = Field(
        default=2,
        ge=1,
        le=15,
        title="Polynomial degree",
        description=(
            "The highest power used. Degree 1 is the straight line; high degrees make curves "
            "that go through every training point (overfitting)."
        ),
        # Só até 3: com muitas features, cada grau a mais multiplica as colunas do polinómio.
        json_schema_extra={"x-search": {"values": [1, 2, 3]}},
    )
    alpha: float = Field(
        default=0.000001,
        ge=0.000001,
        le=1_000,
        title="Alpha (Ridge regularisation)",
        description=(
            "Holds back the weights of the powers. Near zero shows overfitting in all its "
            "glory; raising alpha tames the curves of the high degrees."
        ),
        json_schema_extra={"x-scale": "log", "x-search": {"range": [0.000001, 10]}},
    )


@MODELS.register("polynomial_regression")
class PolynomialRegression(SklearnModel[PolynomialRegressionParams]):
    """Regressão polinomial: potências das features, padronização e Ridge."""

    info = ModelInfo(
        name="polynomial_regression",
        display_name="Polynomial regression",
        summary=(
            "A linear regression on the powers of the features: it follows curves, and with "
            "high degrees it shows overfitting like no other model."
        ),
        task_types=frozenset({TaskType.REGRESSION}),
        family=ModelFamily.LINEAR,
        implementation=Implementation.SKLEARN,
        iterative=False,
        supports_proba=False,
        hyperparameters=PolynomialRegressionParams,
        doc="polynomial_regression.md",
    )

    @override
    def build_estimator(self) -> Pipeline:
        params = self.hyperparameters
        return Pipeline(
            [
                ("powers", PolynomialFeatures(degree=params.degree, include_bias=False)),
                ("scale", StandardScaler()),
                ("ridge", Ridge(alpha=params.alpha)),
            ]
        )

    @override
    def _fit(self, X: Features, y: Target | None, context: TrainingContext) -> None:
        powers = PolynomialFeatures(degree=self.hyperparameters.degree, include_bias=False)
        created = powers.fit(X[:1]).n_output_features_
        if created > MAX_POLYNOMIAL_FEATURES:
            raise ModelError(
                f"With {X.shape[1]} features and degree {self.hyperparameters.degree}, the "
                f"polynomial would have {format_count(created)} terms: pick a lower degree or "
                "fewer features."
            )
        super()._fit(X, y, context)

    @override
    def state(self) -> dict[str, Any]:
        ridge = self.estimator_.named_steps["ridge"]
        return {
            "degree": self.hyperparameters.degree,
            "n_terms": int(self.estimator_.named_steps["powers"].n_output_features_),
            "coefficients": ridge.coef_.tolist(),
            "intercept": float(ridge.intercept_),
        }
