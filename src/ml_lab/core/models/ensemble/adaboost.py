"""AdaBoost de classificação e de regressão (scikit-learn), etapa a etapa.

Treina modelos fracos em sequência (árvores de um só nível, os *stumps*). Depois de cada
um, dá mais peso aos exemplos que ele falhou, para o seguinte se concentrar neles; no
fim, cada modelo vota com um peso que depende de quanto acertou. Ficha didática:
``docs/algorithms/adaboost.md``.
"""

from typing import Any, Literal, override

from pydantic import Field
from sklearn.ensemble import AdaBoostClassifier as SklearnClassifier
from sklearn.ensemble import AdaBoostRegressor as SklearnRegressor

from ml_lab.core.foundation.types import TaskType
from ml_lab.core.models.adapters.growth import GrowingSklearnModel
from ml_lab.core.models.base import (
    MODELS,
    Features,
    Implementation,
    ModelFamily,
    ModelInfo,
    Target,
)
from ml_lab.core.models.ensemble.boosting import AdaBoostParams, boosting_state
from ml_lab.core.training.context import TrainingContext


class AdaBoostRegressionParams(AdaBoostParams):
    """Hiperparâmetros do AdaBoost de regressão."""

    loss: Literal["linear", "square", "exponential"] = Field(
        default="linear",
        title="Error loss",
        description=(
            "How the size of each example's error becomes its weight in the next stage: "
            "proportional, squared (big errors weigh much more) or exponential."
        ),
        json_schema_extra={
            "x-labels": {"linear": "Linear", "square": "Squared", "exponential": "Exponential"},
            "x-search": {"values": ["linear", "square", "exponential"]},
        },
    )


@MODELS.register("adaboost")
class AdaBoost(GrowingSklearnModel[AdaBoostParams]):
    """AdaBoost de classificação do scikit-learn (algoritmo SAMME)."""

    info = ModelInfo(
        name="adaboost",
        display_name="AdaBoost",
        summary=(
            "Combines very simple models, each focused on the examples the previous ones "
            "got wrong: the boundary is built bit by bit."
        ),
        task_types=frozenset({TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION}),
        family=ModelFamily.ENSEMBLE,
        implementation=Implementation.SKLEARN,
        iterative=True,
        supports_proba=True,
        hyperparameters=AdaBoostParams,
        doc="adaboost.md",
        step_limit="n_estimators",
    )

    @override
    def build_estimator(self) -> SklearnClassifier:
        params = self.hyperparameters
        return SklearnClassifier(
            n_estimators=params.n_estimators,
            learning_rate=params.learning_rate,
            random_state=self.seed,
        )

    @override
    def _fit(self, X: Features, y: Target | None, context: TrainingContext) -> None:
        self._replay_stages(self.build_estimator(), X, y, context)

    @override
    def state(self) -> dict[str, Any]:
        return {
            "classes": self.estimator_.classes_.tolist(),
            "estimator_weights": self.estimator_.estimator_weights_.tolist(),
            **boosting_state(self.estimator_),
        }


@MODELS.register("adaboost_regression")
class AdaBoostRegression(GrowingSklearnModel[AdaBoostRegressionParams]):
    """AdaBoost de regressão do scikit-learn (algoritmo AdaBoost.R2)."""

    info = ModelInfo(
        name="adaboost_regression",
        display_name="AdaBoost (regression)",
        summary=(
            "Trees in sequence, each paying more attention to the examples with the most "
            "error; the prediction is the weighted median of the trees."
        ),
        task_types=frozenset({TaskType.REGRESSION}),
        family=ModelFamily.ENSEMBLE,
        implementation=Implementation.SKLEARN,
        iterative=True,
        supports_proba=False,
        hyperparameters=AdaBoostRegressionParams,
        doc="adaboost.md",
        step_limit="n_estimators",
    )

    @override
    def build_estimator(self) -> SklearnRegressor:
        params = self.hyperparameters
        return SklearnRegressor(
            n_estimators=params.n_estimators,
            learning_rate=params.learning_rate,
            loss=params.loss,
            random_state=self.seed,
        )

    @override
    def _fit(self, X: Features, y: Target | None, context: TrainingContext) -> None:
        self._replay_stages(self.build_estimator(), X, y, context)

    @override
    def state(self) -> dict[str, Any]:
        return {
            "estimator_weights": self.estimator_.estimator_weights_.tolist(),
            **boosting_state(self.estimator_),
        }
