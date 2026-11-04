"""Gradient boosting de classificação e de regressão (scikit-learn), etapa a etapa.

Começa por prever uma constante e, em cada etapa, junta uma árvore pequena ajustada ao
que ainda falta acertar (os resíduos, ou o gradiente da log-loss na classificação),
multiplicada pela taxa de aprendizagem. No treino passo a passo, cada passo é uma etapa (ver
``adapters/growth.py``). Ficha didática: ``docs/algorithms/gradient_boosting.md``.
"""

from typing import Any, override

from sklearn.ensemble import GradientBoostingClassifier as SklearnClassifier
from sklearn.ensemble import GradientBoostingRegressor as SklearnRegressor

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
from ml_lab.core.models.ensemble.boosting import GradientBoostingParams, boosting_state
from ml_lab.core.training.context import TrainingContext


def _options(params: GradientBoostingParams, seed: int) -> dict[str, Any]:
    return {
        "n_estimators": params.n_estimators,
        "learning_rate": params.learning_rate,
        "max_depth": params.max_depth,
        "subsample": params.subsample,
        "random_state": seed,
    }


@MODELS.register("gradient_boosting")
class GradientBoosting(GrowingSklearnModel[GradientBoostingParams]):
    """Gradient boosting de classificação do scikit-learn."""

    info = ModelInfo(
        name="gradient_boosting",
        display_name="Gradient boosting",
        summary=(
            "Small trees in sequence, each correcting the errors of the previous ones: one "
            "of the most accurate models for tabular data."
        ),
        task_types=frozenset({TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION}),
        family=ModelFamily.ENSEMBLE,
        implementation=Implementation.SKLEARN,
        iterative=True,
        supports_proba=True,
        hyperparameters=GradientBoostingParams,
        doc="gradient_boosting.md",
        step_limit="n_estimators",
    )

    @override
    def build_estimator(self) -> SklearnClassifier:
        return SklearnClassifier(**_options(self.hyperparameters, self.seed))

    @override
    def _fit(self, X: Features, y: Target | None, context: TrainingContext) -> None:
        self._replay_stages(self.build_estimator(), X, y, context)

    @override
    def state(self) -> dict[str, Any]:
        return {"classes": self.estimator_.classes_.tolist(), **boosting_state(self.estimator_)}


@MODELS.register("gradient_boosting_regression")
class GradientBoostingRegression(GrowingSklearnModel[GradientBoostingParams]):
    """Gradient boosting de regressão do scikit-learn."""

    info = ModelInfo(
        name="gradient_boosting_regression",
        display_name="Gradient boosting (regression)",
        summary=(
            "Small trees in sequence, each fitted to the residuals of the previous ones: the "
            "prediction gets closer to the real values stage by stage."
        ),
        task_types=frozenset({TaskType.REGRESSION}),
        family=ModelFamily.ENSEMBLE,
        implementation=Implementation.SKLEARN,
        iterative=True,
        supports_proba=False,
        hyperparameters=GradientBoostingParams,
        doc="gradient_boosting.md",
        step_limit="n_estimators",
    )

    @override
    def build_estimator(self) -> SklearnRegressor:
        return SklearnRegressor(**_options(self.hyperparameters, self.seed))

    @override
    def _fit(self, X: Features, y: Target | None, context: TrainingContext) -> None:
        self._replay_stages(self.build_estimator(), X, y, context)

    @override
    def state(self) -> dict[str, Any]:
        return boosting_state(self.estimator_)
