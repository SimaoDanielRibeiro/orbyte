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
        title="C (inverse of regularisation)",
        description=(
            "The smaller, the stronger the regularisation: the weights stay small and the "
            "boundary simpler. Large values let the model follow the training data more closely."
        ),
        json_schema_extra={"x-scale": "log", "x-search": {"range": [0.01, 100]}},
    )
    max_iter: int = Field(
        default=100,
        ge=10,
        le=5_000,
        title="Maximum iterations",
        description="How many steps the optimiser can take to find the best weights.",
    )
    class_weight: Literal["none", "balanced"] = Field(
        default="none",
        title="Class weights",
        description=(
            "With “balanced”, the errors on rare classes count more, which helps when the "
            "classes are imbalanced."
        ),
        json_schema_extra={
            "x-labels": {"none": "Same for all", "balanced": "Balanced"},
            "x-search": {"values": ["none", "balanced"]},
        },
    )
    fit_intercept: bool = Field(
        default=True,
        title="Learn the intercept (b)",
        description="Without an intercept, the decision boundary has to go through the origin.",
    )


@MODELS.register("logistic_regression")
class LogisticRegression(SklearnModel[LogisticRegressionParams]):
    """Regressão logística do scikit-learn, com regularização L2 e o otimizador L-BFGS."""

    info = ModelInfo(
        name="logistic_regression",
        display_name="Logistic regression",
        summary=(
            "A linear model that gives the probability of each class: fast, easy to interpret "
            "and a good first model to compare the others with."
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
