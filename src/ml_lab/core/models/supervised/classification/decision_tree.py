"""Árvore de decisão de classificação (scikit-learn), a crescer por profundidade.

Faz perguntas do tipo "``idade`` ≤ 30?" e segue para um lado ou para o outro até chegar
a uma folha, que dá a classe. Em cada nó escolhe a pergunta que deixa os dois lados mais
"puros" (mais dominados por uma só classe). No ecrã Treinar, cada passo é a árvore com
mais um nível de profundidade (ver ``adapters/growth.py``). Ficha didática:
``docs/algorithms/decision_tree.md``.
"""

from typing import Any, Literal, override

from pydantic import Field
from sklearn.tree import DecisionTreeClassifier

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
from ml_lab.core.models.supervised.trees import TreeParams, tree_state
from ml_lab.core.training.context import TrainingContext


class DecisionTreeParams(TreeParams):
    """Hiperparâmetros da árvore de decisão de classificação."""

    criterion: Literal["gini", "entropy"] = Field(
        default="gini",
        title="Medida de impureza",
        description=(
            "Como se mede a mistura de classes num nó. Gini e entropia escolhem quase sempre "
            "as mesmas divisões; a entropia vem da teoria da informação."
        ),
        json_schema_extra={"x-labels": {"gini": "Gini", "entropy": "Entropia"}},
    )


@MODELS.register("decision_tree")
class DecisionTree(GrowingSklearnModel[DecisionTreeParams]):
    """Árvore de decisão de classificação do scikit-learn."""

    info = ModelInfo(
        name="decision_tree",
        display_name="Árvore de decisão",
        summary=(
            "Uma sequência de perguntas sim/não sobre as features que acaba numa classe. Fácil "
            "de ler, mas decora o treino se crescer demais."
        ),
        task_types=frozenset({TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION}),
        family=ModelFamily.TREE,
        implementation=Implementation.SKLEARN,
        iterative=True,
        supports_proba=True,
        hyperparameters=DecisionTreeParams,
        doc="decision_tree.md",
        step_limit="max_depth",
    )

    @override
    def build_estimator(self) -> DecisionTreeClassifier:
        params = self.hyperparameters
        return DecisionTreeClassifier(
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
        return {"classes": self.estimator_.classes_.tolist(), **tree_state(self.estimator_)}
