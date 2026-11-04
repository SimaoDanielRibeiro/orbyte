"""t-SNE (scikit-learn): um mapa 2D que mantém os vizinhos perto.

Para cada ponto, transforma as distâncias aos outros em probabilidades de "ser vizinho"
(a perplexidade é, grosso modo, o número de vizinhos que conta). Depois procura posições
no plano onde essas probabilidades se repitam, descendo o gradiente da divergência de
Kullback-Leibler entre as duas. Os grupos ficam bem separados, mas as distâncias entre
grupos e o tamanho de cada um não querem dizer nada.

Não aprende uma função: coloca no mapa só os pontos com que treinou. Por isso não
avalia o teste (``supports_new_data`` é falso). Ficha didática: ``docs/algorithms/tsne.md``.
"""

from typing import Any, override

import numpy as np
from numpy.typing import NDArray
from pydantic import Field
from sklearn import manifold

from ml_lab.core.foundation.errors import CapabilityError, ModelError
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
from ml_lab.core.models.distances import same_rows
from ml_lab.core.schemas.hyperparameters import Hyperparameters
from ml_lab.core.training.context import TrainingContext


class TSNEParams(Hyperparameters):
    """Hiperparâmetros do t-SNE."""

    perplexity: float = Field(
        default=30.0,
        ge=2,
        le=100,
        title="Perplexity",
        description=(
            "Roughly how many neighbours each point tries to keep close. Small: many tiny "
            "clusters; large: the global structure, more smudged."
        ),
    )
    max_iter: int = Field(
        default=1_000,
        ge=250,
        le=5_000,
        title="Maximum iterations",
        description="How many gradient steps, at most, to lay out the map.",
    )


@MODELS.register("tsne")
class TSNE(SklearnModel[TSNEParams]):
    """t-SNE do scikit-learn (Barnes-Hut), com o mapa sempre em 2D."""

    # O gradiente (Barnes-Hut) soma-se em paralelo: com várias threads, a mesma seed dava
    # divergências KL diferentes nos últimos dígitos.
    single_thread = True

    info = ModelInfo(
        name="tsne",
        display_name="t-SNE",
        summary=(
            "Draws the data on a 2D map where similar points stay together: great for seeing "
            "clusters, but the distances between clusters mislead."
        ),
        task_types=frozenset({TaskType.DIMENSIONALITY_REDUCTION}),
        family=ModelFamily.PROJECTION,
        implementation=Implementation.SKLEARN,
        iterative=False,
        supports_proba=False,
        hyperparameters=TSNEParams,
        doc="tsne.md",
        supports_new_data=False,
        keeps_training_rows=True,
    )

    def __init__(self, hyperparameters: TSNEParams | None = None, *, seed: int = 0) -> None:
        super().__init__(hyperparameters, seed=seed)
        self._train = np.zeros((0, 0))
        self._embedding = np.zeros((0, 2))

    @override
    def build_estimator(self) -> manifold.TSNE:
        params = self.hyperparameters
        return manifold.TSNE(
            n_components=2,
            perplexity=params.perplexity,
            max_iter=params.max_iter,
            init="pca",
            random_state=self.seed,
        )

    @override
    def _fit(self, X: Features, y: Target | None, context: TrainingContext) -> None:
        if self.hyperparameters.perplexity >= len(X):
            raise ModelError(
                f"The perplexity ({self.hyperparameters.perplexity:g}) has to be smaller than "
                f"the number of training rows ({len(X)}): pick a smaller one."
            )
        estimator = self.build_estimator()
        self.estimator_ = self._fit_estimator(estimator, X, None)
        self._train = X.copy()
        self._embedding = np.asarray(estimator.embedding_, dtype=np.float64)

    @override
    def _transform(self, X: Features) -> NDArray[np.float64]:
        if not same_rows(X, self._train):
            raise CapabilityError(
                "t-SNE only places on the map the points it trained on: to see other points, "
                "train it again with them."
            )
        return self._embedding.copy()

    @override
    def state(self) -> dict[str, Any]:
        return {
            "kl_divergence": float(self.estimator_.kl_divergence_),
            "iterations": int(self.estimator_.n_iter_),
        }
