"""Random Forest de classificação (scikit-learn), a crescer árvore a árvore.

Muitas árvores de decisão, cada uma treinada com uma amostra diferente das linhas
(com reposição: o *bootstrap*) e com features sorteadas em cada nó. A floresta dá a
classe mais votada. Cada árvore não viu cerca de um terço das linhas (as que ficaram
"fora do saco"): prevê-las dá uma estimativa do erro sem precisar do teste, o erro
OOB. No treino passo a passo, cada passo junta mais árvores e mostra esse erro. Ficha
didática: ``docs/algorithms/random_forest.md``.
"""

from typing import Any, override

from sklearn.ensemble import RandomForestClassifier

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
from ml_lab.core.models.supervised.forests import ForestParams, forest_options, forest_state
from ml_lab.core.training.context import TrainingContext


@MODELS.register("random_forest")
class RandomForest(GrowingSklearnModel[ForestParams]):
    """Random Forest de classificação do scikit-learn."""

    info = ModelInfo(
        name="random_forest",
        display_name="Random forest",
        summary=(
            "Many different decision trees voting: it corrects each tree's overfitting and is "
            "one of the most reliable models for tabular data."
        ),
        task_types=frozenset({TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION}),
        family=ModelFamily.ENSEMBLE,
        implementation=Implementation.SKLEARN,
        iterative=True,
        supports_proba=True,
        hyperparameters=ForestParams,
        doc="random_forest.md",
    )

    @override
    def build_estimator(self) -> RandomForestClassifier:
        return RandomForestClassifier(
            **forest_options(self.hyperparameters), random_state=self.seed
        )

    @override
    def _fit(self, X: Features, y: Target | None, context: TrainingContext) -> None:
        self._grow_by_trees(self.build_estimator(), X, y, context)

    @override
    def state(self) -> dict[str, Any]:
        return {"classes": self.estimator_.classes_.tolist(), **forest_state(self.estimator_)}
