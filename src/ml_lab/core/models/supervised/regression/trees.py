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
        title="Error to minimise",
        description=(
            "How the quality of a split is measured: the squared error (each leaf predicts the "
            "mean) or the absolute error (it predicts the median, less sensitive to extremes "
            "and slower)."
        ),
        json_schema_extra={"x-labels": {"squared_error": "Squared", "absolute_error": "Absolute"}},
    )


@MODELS.register("decision_tree_regression")
class DecisionTreeRegression(GrowingSklearnModel[DecisionTreeRegressionParams]):
    """Árvore de decisão de regressão do scikit-learn."""

    info = ModelInfo(
        name="decision_tree_regression",
        display_name="Decision tree (regression)",
        summary=(
            "Yes/no questions about the features down to a leaf, which predicts the mean of the "
            "examples that land there: a step function."
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
        title="Features per split",
        description=(
            "How many features, picked at random, each node can try. In regression, using all "
            "of them is the usual; fewer make the trees more different from each other."
        ),
        json_schema_extra={
            "x-labels": {
                "sqrt": "Square root of the total",
                "log2": "Logarithm of the total",
                "all": "All",
            },
            "x-search": {"values": ["sqrt", "log2", "all"]},
        },
    )


@MODELS.register("random_forest_regression")
class RandomForestRegression(GrowingSklearnModel[RandomForestRegressionParams]):
    """Random Forest de regressão do scikit-learn."""

    info = ModelInfo(
        name="random_forest_regression",
        display_name="Random forest (regression)",
        summary=(
            "The average of many different regression trees: smooth steps and one of the most "
            "reliable models for predicting numbers from tables."
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
