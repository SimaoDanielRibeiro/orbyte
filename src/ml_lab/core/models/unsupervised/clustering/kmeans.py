"""K-means (scikit-learn).

O mesmo algoritmo de Lloyd da versão didática (``kmeans_numpy.py``), mas repetido com
vários pontos de partida (``n_init``), ficando o resultado com a menor inércia. Ficha
didática: ``docs/algorithms/kmeans.md``.
"""

from typing import Any, override

from pydantic import Field
from sklearn.cluster import KMeans as SklearnKMeans

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
from ml_lab.core.models.unsupervised.clustering.kmeans_numpy import KMeansParams
from ml_lab.core.training.context import TrainingContext


class SklearnKMeansParams(KMeansParams):
    """Hiperparâmetros do k-means do scikit-learn."""

    n_init: int = Field(
        default=10,
        ge=1,
        le=50,
        title="Número de partidas",
        description=(
            "Quantas vezes corre o algoritmo com pontos de partida diferentes; fica com o "
            "resultado de menor inércia."
        ),
    )


@MODELS.register("kmeans")
class KMeans(SklearnModel[SklearnKMeansParams]):
    """K-means do scikit-learn."""

    info = ModelInfo(
        name="kmeans",
        display_name="K-means",
        summary=(
            "Agrupa os pontos à volta de k centróides, com várias partidas para fugir dos maus "
            "começos: rápido e o primeiro método de clustering a experimentar."
        ),
        task_types=frozenset({TaskType.CLUSTERING}),
        family=ModelFamily.CENTROID,
        implementation=Implementation.SKLEARN,
        iterative=False,
        supports_proba=False,
        hyperparameters=SklearnKMeansParams,
        doc="kmeans.md",
    )

    @override
    def build_estimator(self) -> SklearnKMeans:
        params = self.hyperparameters
        return SklearnKMeans(
            n_clusters=params.n_clusters,
            init=params.init,
            n_init=params.n_init,
            max_iter=params.max_iter,
            tol=params.tol,
            random_state=self.seed,
        )

    @override
    def _fit(self, X: Features, y: Target | None, context: TrainingContext) -> None:
        if self.hyperparameters.n_clusters > len(X):
            raise ModelError(
                f"Pediste {self.hyperparameters.n_clusters} grupos, mas o treino só tem "
                f"{len(X)} linhas."
            )
        super()._fit(X, None, context)

    @override
    def state(self) -> dict[str, Any]:
        estimator = self.estimator_
        return {
            "centers": estimator.cluster_centers_.tolist(),
            "inertia": float(estimator.inertia_),
            "iterations": int(estimator.n_iter_),
        }
