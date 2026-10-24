"""PCA (análise de componentes principais): implementação didática em NumPy.

Procura as direções em que os dados variam mais:

1. centra os dados (subtrai a média de cada feature);
2. calcula a matriz de covariância, ``Σ = Xcᵀ Xc / (n - 1)``;
3. os vetores próprios de ``Σ`` são as direções (as **componentes principais**) e os
   valores próprios são a variância dos dados ao longo de cada uma;
4. fica com as k componentes de maior variância e projeta os dados nelas.

O sinal de cada componente é arbitrário (uma direção e a oposta são a mesma reta); como
no scikit-learn, escolhe-se o sinal que torna positivo o maior valor da componente.
Ficha didática: ``docs/algorithms/pca.md``.
"""

from typing import Any, override

import numpy as np
from numpy.typing import NDArray
from pydantic import Field

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
from ml_lab.core.schemas.hyperparameters import Hyperparameters
from ml_lab.core.training.context import TrainingContext


class PCAParams(Hyperparameters):
    """Hiperparâmetros da PCA."""

    n_components: int = Field(
        default=2,
        ge=1,
        le=10,
        title="Número de componentes",
        description=(
            "Quantas direções guardar. Com 2, os dados cabem num gráfico; a variância "
            "explicada diz quanta informação ficou."
        ),
    )


def check_components(n_components: int, X: Features) -> None:
    """Confirma que há dados para ``n_components`` componentes.

    Raises:
        ModelError: Se houver mais componentes do que features (ou linhas).
    """
    limit = min(X.shape)
    if n_components > limit:
        raise ModelError(
            f"Pediste {n_components} componentes, mas os dados só têm {X.shape[1]} features e "
            f"{X.shape[0]} linhas: no máximo {limit}."
        )


@MODELS.register("pca_numpy")
class PCANumpy(LabModel[PCAParams]):
    """PCA pela decomposição da matriz de covariância."""

    info = ModelInfo(
        name="pca_numpy",
        display_name="PCA (NumPy)",
        summary=(
            "Encontra as direções de maior variância com a matriz de covariância e os seus "
            "vetores próprios, em NumPy, para comparares com a do scikit-learn."
        ),
        task_types=frozenset({TaskType.DIMENSIONALITY_REDUCTION}),
        family=ModelFamily.PROJECTION,
        implementation=Implementation.DIDACTIC,
        iterative=False,
        supports_proba=False,
        hyperparameters=PCAParams,
        doc="pca.md",
    )

    def __init__(self, hyperparameters: PCAParams | None = None, *, seed: int = 0) -> None:
        super().__init__(hyperparameters, seed=seed)
        self._mean = np.zeros(0)
        self._components = np.zeros((0, 0))
        self._variances = np.zeros(0)
        self._ratios = np.zeros(0)

    @override
    def state(self) -> dict[str, Any]:
        return {
            "components": self._components.tolist(),
            "explained_variance": self._variances.tolist(),
            "explained_variance_ratio": self._ratios.tolist(),
            "mean": self._mean.tolist(),
        }

    @override
    def _fit(self, X: Features, y: Target | None, context: TrainingContext) -> None:
        k = self.hyperparameters.n_components
        check_components(k, X)
        self._mean = X.mean(axis=0)
        centered = X - self._mean
        covariance = centered.T @ centered / max(1, len(X) - 1)
        values, vectors = np.linalg.eigh(covariance)  # por ordem crescente
        order = np.argsort(values)[::-1]
        values, vectors = np.maximum(values[order], 0.0), vectors[:, order]
        components = vectors[:, :k].T
        signs = np.sign(components[np.arange(k), np.argmax(np.abs(components), axis=1)])
        self._components = components * signs[:, None]
        self._variances = values[:k]
        total = float(values.sum())
        self._ratios = self._variances / total if total > 0 else np.zeros(k)

    @override
    def _transform(self, X: Features) -> NDArray[np.float64]:
        projected: NDArray[np.float64] = (X - self._mean) @ self._components.T
        return projected
