"""K-means: implementação didática em NumPy (o algoritmo de Lloyd).

Escolhe k pontos de partida para os centróides e repete dois passos até deixarem de
mexer:

1. **atribuir**: cada ponto vai para o grupo do centróide mais próximo;
2. **atualizar**: cada centróide passa para a média dos pontos do seu grupo.

Cada iteração nunca aumenta a inércia (a soma das distâncias ao quadrado de cada ponto
ao seu centróide), por isso o algoritmo pára sempre, mas num mínimo que depende do
ponto de partida. O k-means++ escolhe os pontos de partida afastados uns dos outros:
cada novo centróide é sorteado com probabilidade proporcional ao quadrado da distância
ao centróide mais próximo já escolhido. Ficha didática: ``docs/algorithms/kmeans.md``.
"""

from typing import Any, Literal, override

import numpy as np
from numpy.typing import NDArray
from pydantic import Field

from ml_lab.core.foundation.errors import ModelError
from ml_lab.core.foundation.notices import Notice
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
from ml_lab.core.models.distances import nearest
from ml_lab.core.schemas.hyperparameters import Hyperparameters
from ml_lab.core.training.context import TrainingContext
from ml_lab.formatting import format_count


class KMeansParams(Hyperparameters):
    """Hiperparâmetros do k-means."""

    n_clusters: int = Field(
        default=3,
        ge=2,
        le=20,
        title="Number of clusters (k)",
        description=(
            "How many clusters to look for. K-means always finds k clusters, even if the data "
            "doesn't have them: compare the silhouette of several k."
        ),
    )
    init: Literal["k-means++", "random"] = Field(
        default="k-means++",
        title="Starting points",
        description=(
            "Where the centroids start: k-means++ spreads them over the data; at random, it "
            "can put two in the same cluster and end in a poor result."
        ),
        json_schema_extra={"x-labels": {"k-means++": "k-means++", "random": "At random"}},
    )
    max_iter: int = Field(
        default=300,
        ge=1,
        le=1_000,
        title="Maximum iterations",
        description="How many times, at most, it repeats assigning and updating.",
    )
    tol: float = Field(
        default=0.0001,
        ge=1e-8,
        le=0.1,
        title="Stopping tolerance",
        description=(
            "Stops when the centroids barely move: the shift falls below this fraction of the "
            "data's variance."
        ),
        json_schema_extra={"x-scale": "log"},
    )


@MODELS.register("kmeans_numpy")
class KMeansNumpy(LabModel[KMeansParams]):
    """K-means de Lloyd, com uma iteração por passo do treino."""

    info = ModelInfo(
        name="kmeans_numpy",
        display_name="K-means (NumPy)",
        summary=(
            "Groups the points around k centroids, written in NumPy: you see the centroids "
            "walk to the middle of their clusters, iteration by iteration."
        ),
        task_types=frozenset({TaskType.CLUSTERING}),
        family=ModelFamily.CENTROID,
        implementation=Implementation.DIDACTIC,
        iterative=True,
        supports_proba=False,
        hyperparameters=KMeansParams,
        doc="kmeans.md",
        step_limit="max_iter",
    )

    def __init__(self, hyperparameters: KMeansParams | None = None, *, seed: int = 0) -> None:
        super().__init__(hyperparameters, seed=seed)
        self._centers = np.zeros((0, 0))
        self._initial = np.zeros((0, 0))
        self._inertia = 0.0
        self._iterations = 0
        self._converged = False

    @override
    def state(self) -> dict[str, Any]:
        return {
            "centers": self._centers.tolist(),
            "initial_centers": self._initial.tolist(),
            "inertia": self._inertia,
            "iterations": self._iterations,
            "converged": self._converged,
        }

    @override
    def _fit(self, X: Features, y: Target | None, context: TrainingContext) -> None:
        params = self.hyperparameters
        k = params.n_clusters
        if k > len(X):
            raise ModelError(
                f"You asked for {k} clusters, but the training data has only {len(X)} rows."
            )
        rng = np.random.default_rng(self.seed)
        self._centers = self._start(X, k, rng)
        self._initial = self._centers.copy()
        self._iterations = 0
        self._converged = False
        threshold = params.tol * float(np.mean(np.var(X, axis=0)))

        for iteration in range(1, params.max_iter + 1):
            labels, _ = nearest(X, self._centers)
            updated = self._centers.copy()
            for group in range(k):
                members = X[labels == group]
                if len(members):
                    updated[group] = members.mean(axis=0)
            shift = float(np.sum((updated - self._centers) ** 2))
            self._centers = updated
            self._iterations = iteration
            self._inertia = float(np.sum((X - self._centers[labels]) ** 2))
            context.report_step(iteration, loss=self._inertia)
            if context.wants_snapshot(iteration):
                context.report_snapshot(iteration, self.snapshot_state(context.probe))
            if shift <= threshold:
                self._converged = True
                break
            if context.should_stop():
                break
        labels, distances = nearest(X, self._centers)
        self._inertia = float(np.sum(distances**2))
        empty = k - len(np.unique(labels))
        if empty:
            self._note(
                Notice.warning(
                    "empty_clusters",
                    f"{format_count(empty)} of the {k} clusters ended up empty: there are more "
                    "clusters than the structure of the data justifies. Try a smaller k.",
                )
            )
        if not self._converged and self._iterations == params.max_iter:
            self._note(
                Notice.warning(
                    "not_converged",
                    "The centroids were still moving after "
                    f"{format_count(params.max_iter)} iterations: raise the maximum iterations.",
                )
            )

    @override
    def _predict(self, X: Features) -> NDArray[Any]:
        labels, _ = nearest(X, self._centers)
        return labels.astype(np.int64)

    def _start(self, X: Features, k: int, rng: np.random.Generator) -> NDArray[np.float64]:
        """Os centróides de partida: k pontos de treino, ao acaso ou pelo k-means++."""
        if self.hyperparameters.init == "random":
            chosen: NDArray[np.float64] = X[rng.choice(len(X), size=k, replace=False)].copy()
            return chosen
        centers = [X[rng.integers(len(X))]]
        for _ in range(1, k):
            _, distances = nearest(X, np.array(centers))
            weights = distances**2
            total = float(weights.sum())
            # Todos os pontos já coincidem com um centróide: qualquer um serve.
            pick = rng.integers(len(X)) if total == 0 else rng.choice(len(X), p=weights / total)
            centers.append(X[pick])
        return np.array(centers, dtype=np.float64)
