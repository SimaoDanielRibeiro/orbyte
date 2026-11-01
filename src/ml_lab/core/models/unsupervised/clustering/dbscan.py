"""DBSCAN: grupos como zonas densas (scikit-learn).

Um ponto com pelo menos ``min_samples`` vizinhos a menos de ``eps`` é um ponto
**núcleo**. Núcleos vizinhos uns dos outros formam um grupo; os pontos perto de um
núcleo mas sem vizinhos suficientes ficam na **fronteira** do grupo; os restantes são
**ruído** (grupo -1). Não é preciso dizer quantos grupos há, e os grupos podem ter
qualquer forma.

O scikit-learn não prevê pontos novos. Aqui, um ponto novo fica no grupo do núcleo mais
próximo, se estiver a menos de ``eps`` dele; senão, é ruído. Os pontos de treino ficam
exatamente com o grupo que o DBSCAN lhes deu. Ficha didática: ``docs/algorithms/dbscan.md``.
"""

from typing import Any, override

import numpy as np
from numpy.typing import NDArray
from pydantic import Field
from sklearn import cluster

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


class DBSCANParams(Hyperparameters):
    """Hiperparâmetros do DBSCAN."""

    eps: float = Field(
        default=0.5,
        ge=0.01,
        le=10,
        title="Raio da vizinhança (eps)",
        description=(
            "Até que distância dois pontos são vizinhos. Pequeno: quase tudo é ruído; grande: "
            "tudo se junta num só grupo. Depende da escala das features (padroniza-as)."
        ),
        json_schema_extra={"x-scale": "log"},
    )
    min_samples: int = Field(
        default=5,
        ge=1,
        le=100,
        title="Vizinhos para ser núcleo",
        description=(
            "Quantos pontos (incluindo o próprio) tem de haver no raio para o ponto ser núcleo "
            "de um grupo. Valores maiores exigem zonas mais densas."
        ),
    )


@MODELS.register("dbscan")
class DBSCAN(SklearnModel[DBSCANParams]):
    """DBSCAN do scikit-learn, com uma regra para pontos novos."""

    info = ModelInfo(
        name="dbscan",
        display_name="DBSCAN",
        summary=(
            "Encontra zonas densas de pontos, de qualquer forma, sem lhe dizeres quantos grupos "
            "há; os pontos isolados ficam como ruído."
        ),
        task_types=frozenset({TaskType.CLUSTERING}),
        family=ModelFamily.DENSITY,
        implementation=Implementation.SKLEARN,
        iterative=False,
        supports_proba=False,
        hyperparameters=DBSCANParams,
        doc="dbscan.md",
        keeps_training_rows=True,
    )

    def __init__(self, hyperparameters: DBSCANParams | None = None, *, seed: int = 0) -> None:
        super().__init__(hyperparameters, seed=seed)
        self._train = np.zeros((0, 0))
        self._core = np.zeros((0, 0))
        self._core_labels = np.zeros(0, dtype=np.int64)

    @override
    def build_estimator(self) -> cluster.DBSCAN:
        params = self.hyperparameters
        return cluster.DBSCAN(eps=params.eps, min_samples=params.min_samples)

    @override
    def _fit(self, X: Features, y: Target | None, context: TrainingContext) -> None:
        super()._fit(X, None, context)
        indices = self.estimator_.core_sample_indices_
        self._train = X.copy()
        self._core = X[indices]
        self._core_labels = np.asarray(self.estimator_.labels_, dtype=np.int64)[indices]

    @override
    def _predict(self, X: Features) -> NDArray[Any]:
        if same_rows(X, self._train):
            labels: NDArray[np.int64] = np.asarray(self.estimator_.labels_, dtype=np.int64)
            return labels
        if not len(self._core):
            return np.full(len(X), -1, dtype=np.int64)
        closest, distances = nearest(X, self._core)
        return np.where(distances <= self.hyperparameters.eps, self._core_labels[closest], -1)

    def memory(self) -> Memory:
        """Os pontos centrais e o grupo de cada um.

        Um ponto novo a menos de ``eps`` do ponto central mais próximo fica no grupo dele;
        senão, é ruído (-1).
        """
        return Memory(rows=self._core.copy(), labels=self._core_labels.copy())

    @override
    def state(self) -> dict[str, Any]:
        labels = np.asarray(self.estimator_.labels_)
        return {
            "n_clusters": len(set(labels.tolist()) - {-1}),
            "n_noise": int(np.sum(labels == -1)),
            "n_core": len(self._core),
        }
