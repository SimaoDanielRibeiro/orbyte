"""Árvore de decisão e Random Forest de regressão (scikit-learn).

As mesmas ideias da classificação, mas cada folha prevê a média do alvo dos exemplos que
lá caem: a previsão é uma função em degraus, que se refina a cada nível (árvore) ou a
cada árvore que se junta (floresta). Fichas didáticas: ``docs/algorithms/decision_tree.md``
e ``docs/algorithms/random_forest.md``.
"""

from typing import Any, Literal, override

from pydantic import Field
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

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
from ml_lab.core.models.supervised.trees import TreeParams, tree_state
from ml_lab.core.training.context import TrainingContext


class DecisionTreeRegressionParams(TreeParams):
    """Hiperparâmetros da árvore de decisão de regressão."""

    criterion: Literal["squared_error", "absolute_error"] = Field(
        default="squared_error",
        title="Erro a minimizar",
        description=(
            "Como se mede a qualidade de uma divisão: o erro quadrático (cada folha prevê a "
            "média) ou o erro absoluto (prevê a mediana, menos sensível a extremos e mais lento)."
        ),
        json_schema_extra={
            "x-labels": {"squared_error": "Quadrático", "absolute_error": "Absoluto"}
        },
    )


@MODELS.register("decision_tree_regression")
class DecisionTreeRegression(GrowingSklearnModel[DecisionTreeRegressionParams]):
    """Árvore de decisão de regressão do scikit-learn."""

    info = ModelInfo(
        name="decision_tree_regression",
        display_name="Árvore de decisão (regressão)",
        summary=(
            "Perguntas sim/não sobre as features até uma folha, que prevê a média dos exemplos "
            "que lá caem: uma função em degraus."
        ),
        task_types=frozenset({TaskType.REGRESSION}),
        family=ModelFamily.TREE,
        implementation=Implementation.SKLEARN,
        iterative=True,
        supports_proba=False,
        hyperparameters=DecisionTreeRegressionParams,
        doc="decision_tree.md",
        step_limit="max_depth",
    )

    @override
    def build_estimator(self) -> DecisionTreeRegressor:
        params = self.hyperparameters
        return DecisionTreeRegressor(
            criterion=params.criterion,
            max_depth=params.max_depth or None,
            min_samples_leaf=params.min_samples_leaf,
            random_state=self.seed,
        )

    @override
    def _fit(self, X: Features, y: Target | None, context: TrainingContext) -> None:
        self._grow_by_depth(self.build_estimator(), X, y, context)

    @override
    def state(self) -> dict[str, Any]:
        return tree_state(self.estimator_)


class RandomForestRegressionParams(ForestParams):
    """Hiperparâmetros da Random Forest de regressão."""

    max_features: Literal["sqrt", "log2", "all"] = Field(
        default="all",
        title="Features por divisão",
        description=(
            "Quantas features, escolhidas ao acaso, cada nó pode experimentar. Na regressão, "
            "usar todas é o habitual; menos tornam as árvores mais diferentes entre si."
        ),
        json_schema_extra={
            "x-labels": {
                "sqrt": "Raiz quadrada do total",
                "log2": "Logaritmo do total",
                "all": "Todas",
            }
        },
    )


@MODELS.register("random_forest_regression")
class RandomForestRegression(GrowingSklearnModel[RandomForestRegressionParams]):
    """Random Forest de regressão do scikit-learn."""

    info = ModelInfo(
        name="random_forest_regression",
        display_name="Random Forest (regressão)",
        summary=(
            "A média de muitas árvores de regressão diferentes: degraus suaves e um dos modelos "
            "mais fiáveis para prever números a partir de tabelas."
        ),
        task_types=frozenset({TaskType.REGRESSION}),
        family=ModelFamily.ENSEMBLE,
        implementation=Implementation.SKLEARN,
        iterative=True,
        supports_proba=False,
        hyperparameters=RandomForestRegressionParams,
        doc="random_forest.md",
    )

    @override
    def build_estimator(self) -> RandomForestRegressor:
        return RandomForestRegressor(**forest_options(self.hyperparameters), random_state=self.seed)

    @override
    def _fit(self, X: Features, y: Target | None, context: TrainingContext) -> None:
        self._grow_by_trees(self.build_estimator(), X, y, context)

    @override
    def state(self) -> dict[str, Any]:
        return forest_state(self.estimator_)
