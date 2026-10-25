"""Regressão logística (scikit-learn).

Apesar do nome, classifica: calcula uma soma pesada das features, ``z = w·x + b``, e
transforma-a numa probabilidade com a função logística, ``p = 1 / (1 + e^-z)``. Com
mais de duas classes, usa uma soma por classe e a função softmax. Ficha didática:
``docs/algorithms/logistic_regression.md``.

A versão didática em NumPy, que mostra os pesos a mudar época a época, está em
``logistic_regression_gd.py``.
"""

from typing import Any, Literal, override

from pydantic import Field
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression

from ml_lab.core.foundation.types import TaskType
from ml_lab.core.models.adapters.sklearn_adapter import SklearnModel
from ml_lab.core.models.base import MODELS, Implementation, ModelFamily, ModelInfo
from ml_lab.core.schemas.hyperparameters import Hyperparameters


class LogisticRegressionParams(Hyperparameters):
    """Hiperparâmetros da regressão logística."""

    C: float = Field(
        default=1.0,
        ge=0.0001,
        le=10_000,
        title="C (inverso da regularização)",
        description=(
            "Quanto menor, mais forte a regularização: os pesos ficam pequenos e a fronteira "
            "mais simples. Valores grandes deixam o modelo seguir mais de perto o treino."
        ),
        json_schema_extra={"x-scale": "log", "x-search": {"range": [0.01, 100]}},
    )
    max_iter: int = Field(
        default=100,
        ge=10,
        le=5_000,
        title="Máximo de iterações",
        description="Quantos passos o otimizador pode dar até encontrar os melhores pesos.",
    )
    class_weight: Literal["none", "balanced"] = Field(
        default="none",
        title="Peso das classes",
        description=(
            "Com «equilibrado», os erros nas classes raras contam mais, o que ajuda quando as "
            "classes estão desequilibradas."
        ),
        json_schema_extra={
            "x-labels": {"none": "Igual para todas", "balanced": "Equilibrado"},
            "x-search": {"values": ["none", "balanced"]},
        },
    )
    fit_intercept: bool = Field(
        default=True,
        title="Aprender a ordenada (b)",
        description="Sem ordenada, a fronteira de decisão é obrigada a passar pela origem.",
    )


@MODELS.register("logistic_regression")
class LogisticRegression(SklearnModel[LogisticRegressionParams]):
    """Regressão logística do scikit-learn, com regularização L2 e o otimizador L-BFGS."""

    info = ModelInfo(
        name="logistic_regression",
        display_name="Regressão logística",
        summary=(
            "Modelo linear que dá a probabilidade de cada classe: rápido, fácil de interpretar "
            "e um bom primeiro modelo para comparar com os outros."
        ),
        task_types=frozenset({TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION}),
        family=ModelFamily.LINEAR,
        implementation=Implementation.SKLEARN,
        iterative=False,
        supports_proba=True,
        hyperparameters=LogisticRegressionParams,
        doc="logistic_regression.md",
    )

    @override
    def build_estimator(self) -> SklearnLogisticRegression:
        params = self.hyperparameters
        return SklearnLogisticRegression(
            C=params.C,
            max_iter=params.max_iter,
            class_weight=None if params.class_weight == "none" else "balanced",
            fit_intercept=params.fit_intercept,
            random_state=self.seed,
        )

    @override
    def state(self) -> dict[str, Any]:
        estimator = self.estimator_
        return {
            "coefficients": estimator.coef_.tolist(),
            "intercept": estimator.intercept_.tolist(),
            "classes": estimator.classes_.tolist(),
            "iterations": int(max(estimator.n_iter_)),
        }
