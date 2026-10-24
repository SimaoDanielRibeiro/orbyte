"""K vizinhos mais próximos: implementação didática em NumPy.

Treinar é só guardar os exemplos. Prever um ponto é:

1. medir a distância dele a todos os exemplos de treino;
2. ficar com os k mais próximos (``np.argpartition``, sem ordenar tudo);
3. contar os votos de cada classe (com pesos ``1 / distância``, se pedido) e dar a mais
   votada; a proporção de votos é a "probabilidade".

As distâncias calculam-se aos blocos de linhas (``models/distances.py``). Ficha didática:
``docs/algorithms/knn.md``.
"""

from typing import Any, override

import numpy as np
from numpy.typing import NDArray

from ml_lab.core.foundation.errors import ModelError
from ml_lab.core.foundation.types import TaskType
from ml_lab.core.models.base import (
    MODELS,
    Features,
    Implementation,
    LabModel,
    ModelFamily,
    ModelInfo,
    Target,
)
from ml_lab.core.models.distances import blocks
from ml_lab.core.models.supervised.classification.knn import KnnParams, check_neighbors
from ml_lab.core.training.context import TrainingContext


@MODELS.register("knn_numpy")
class KnnNumpy(LabModel[KnnParams]):
    """K vizinhos mais próximos, com as distâncias e os votos à vista."""

    info = ModelInfo(
        name="knn_numpy",
        display_name="K vizinhos mais próximos (NumPy)",
        summary=(
            "O mesmo KNN, escrito em NumPy: distâncias, os k mais próximos e a votação, passo a "
            "passo, para comparares com o do scikit-learn."
        ),
        task_types=frozenset({TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION}),
        family=ModelFamily.INSTANCE_BASED,
        implementation=Implementation.DIDACTIC,
        iterative=False,
        supports_proba=True,
        hyperparameters=KnnParams,
        doc="knn.md",
    )

    def __init__(self, hyperparameters: KnnParams | None = None, *, seed: int = 0) -> None:
        super().__init__(hyperparameters, seed=seed)
        self._train = np.zeros((0, 0))
        self._codes = np.zeros(0, dtype=np.int64)
        self._classes = np.zeros(0, dtype=np.int64)

    @override
    def classes(self) -> NDArray[np.int64] | None:
        return self._classes if self._classes.size else None

    @override
    def state(self) -> dict[str, Any]:
        return {"classes": self._classes.tolist(), "n_train": len(self._train)}

    @override
    def _fit(self, X: Features, y: Target | None, context: TrainingContext) -> None:
        if y is None:
            raise ModelError("O KNN precisa do alvo para classificar.")
        check_neighbors(self.hyperparameters.n_neighbors, len(X))
        self._classes = np.unique(y).astype(np.int64)
        self._codes = np.searchsorted(self._classes, y)
        self._train = X.copy()

    @override
    def _predict(self, X: Features) -> NDArray[Any]:
        return self._classes[np.argmax(self._votes(X), axis=1)]

    @override
    def _predict_proba(self, X: Features) -> NDArray[np.float64]:
        votes = self._votes(X)
        result: NDArray[np.float64] = votes / votes.sum(axis=1, keepdims=True)
        return result

    def _votes(self, X: Features) -> NDArray[np.float64]:
        """Os votos (com o seu peso) de cada classe, para cada linha de ``X``."""
        params = self.hyperparameters
        k = params.n_neighbors
        votes = np.zeros((len(X), len(self._classes)))
        for start, distances in blocks(X, self._train, params.metric):
            rows = np.arange(len(distances))
            nearest = np.argpartition(distances, k - 1, axis=1)[:, :k]
            near = np.take_along_axis(distances, nearest, axis=1)
            weights = np.ones_like(near) if params.weights == "uniform" else _inverse(near)
            for column in range(k):
                np.add.at(
                    votes[start : start + len(distances)],
                    (rows, self._codes[nearest[:, column]]),
                    weights[:, column],
                )
        return votes


def _inverse(distances: NDArray[np.float64]) -> NDArray[np.float64]:
    """Pesos ``1 / distância``; um vizinho à distância zero leva o voto todo (como no sklearn)."""
    exact = distances == 0
    with np.errstate(divide="ignore"):
        weights: NDArray[np.float64] = np.where(exact, 0.0, 1.0 / distances)
    rows = exact.any(axis=1)
    weights[rows] = exact[rows].astype(np.float64)
    return weights
