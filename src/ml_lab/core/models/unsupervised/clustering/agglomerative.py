"""Clustering hierárquico aglomerativo (scikit-learn).

Começa com cada ponto no seu grupo e, em cada passo, junta os dois grupos mais
próximos, até sobrarem ``n_clusters``. A forma de medir a distância entre grupos (a
*ligação*) muda tudo: ``ward`` junta os grupos que menos aumentam a variância (grupos
redondos e equilibrados); ``single`` usa os dois pontos mais próximos (segue cadeias).
O histórico das junções é o **dendrograma**, que fica no estado do modelo.

Guarda as distâncias entre todos os pares de pontos, por isso só aceita até 5000 linhas
de treino. O scikit-learn não prevê pontos novos: aqui, um ponto novo fica no grupo do
ponto de treino mais próximo. Ficha didática: ``docs/algorithms/agglomerative.md``.
"""

from typing import Any, Final, Literal, override

import numpy as np
from numpy.typing import NDArray
from pydantic import Field
from sklearn.cluster import AgglomerativeClustering

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
from ml_lab.core.models.distances import nearest, same_rows
from ml_lab.core.models.parameters import Memory
from ml_lab.core.schemas.hyperparameters import Hyperparameters
from ml_lab.core.training.context import TrainingContext
from ml_lab.formatting import format_count

MAX_ROWS: Final = 5_000
"""Mais linhas de treino do que isto, e a matriz de distâncias deixa de caber em memória."""


class AgglomerativeParams(Hyperparameters):
    """Hiperparâmetros do clustering hierárquico."""

    n_clusters: int = Field(
        default=3,
        ge=2,
        le=20,
        title="Número de grupos",
        description="Em quantos grupos se corta o dendrograma.",
    )
    linkage: Literal["ward", "complete", "average", "single"] = Field(
        default="ward",
        title="Ligação",
        description=(
            "Como se mede a distância entre dois grupos: Ward (o aumento da variância), "
            "completa (os pontos mais afastados), média (a média dos pares) ou simples (os "
            "pontos mais próximos)."
        ),
        json_schema_extra={
            "x-labels": {
                "ward": "Ward",
                "complete": "Completa",
                "average": "Média",
                "single": "Simples",
            }
        },
    )


def linkage_matrix(children: NDArray[np.intp], distances: NDArray[np.float64]) -> list[list[float]]:
    """O dendrograma no formato do SciPy: ``[grupo a, grupo b, distância, pontos]`` por junção.

    Os índices até ``n - 1`` são os pontos; ``n + i`` é o grupo criado na junção ``i``.
    """
    n_points = len(children) + 1
    sizes = np.zeros(len(children))
    for merge, pair in enumerate(children):
        sizes[merge] = sum(1 if node < n_points else sizes[node - n_points] for node in pair)
    return [
        [float(a), float(b), float(distance), float(size)]
        for (a, b), distance, size in zip(children, distances, sizes, strict=True)
    ]


@MODELS.register("agglomerative")
class Agglomerative(SklearnModel[AgglomerativeParams]):
    """Clustering hierárquico aglomerativo do scikit-learn."""

    info = ModelInfo(
        name="agglomerative",
        display_name="Clustering hierárquico",
        summary=(
            "Junta os pontos em grupos cada vez maiores, dos mais próximos para os mais "
            "afastados: o dendrograma mostra a hierarquia toda."
        ),
        task_types=frozenset({TaskType.CLUSTERING}),
        family=ModelFamily.HIERARCHICAL,
        implementation=Implementation.SKLEARN,
        iterative=False,
        supports_proba=False,
        hyperparameters=AgglomerativeParams,
        doc="agglomerative.md",
        keeps_training_rows=True,
    )

    def __init__(
        self, hyperparameters: AgglomerativeParams | None = None, *, seed: int = 0
    ) -> None:
        super().__init__(hyperparameters, seed=seed)
        self._train = np.zeros((0, 0))
        self._labels = np.zeros(0, dtype=np.int64)

    @override
    def build_estimator(self) -> AgglomerativeClustering:
        params = self.hyperparameters
        return AgglomerativeClustering(
            n_clusters=params.n_clusters, linkage=params.linkage, compute_distances=True
        )

    @override
    def _fit(self, X: Features, y: Target | None, context: TrainingContext) -> None:
        if len(X) > MAX_ROWS:
            raise ModelError(
                f"O clustering hierárquico guarda a distância entre todos os pares de pontos: com "
                f"{format_count(len(X))} linhas seriam demasiados pares. Usa no máximo "
                f"{format_count(MAX_ROWS)} linhas de treino (por exemplo, com um teste maior)."
            )
        if self.hyperparameters.n_clusters > len(X):
            raise ModelError(
                f"Pediste {self.hyperparameters.n_clusters} grupos, mas o treino só tem "
                f"{len(X)} linhas."
            )
        super()._fit(X, None, context)
        self._train = X.copy()
        self._labels = np.asarray(self.estimator_.labels_, dtype=np.int64)

    @override
    def _predict(self, X: Features) -> NDArray[Any]:
        if same_rows(X, self._train):
            return self._labels.copy()
        closest, _ = nearest(X, self._train)
        return self._labels[closest]

    def memory(self) -> Memory:
        """As linhas de treino e o grupo de cada uma: um ponto novo fica no do mais próximo."""
        return Memory(rows=self._train.copy(), labels=self._labels.copy())

    @override
    def state(self) -> dict[str, Any]:
        estimator = self.estimator_
        return {
            "n_clusters": int(estimator.n_clusters_),
            "linkage": linkage_matrix(estimator.children_, estimator.distances_),
        }
