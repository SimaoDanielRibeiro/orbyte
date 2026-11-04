"""PCA (análise de componentes principais), implementação didática em NumPy.

A PCA procura as direções em que os dados variam mais. A primeira componente é a
direção de maior variância; a segunda é a de maior variância entre as direções
perpendiculares à primeira; e assim por diante. Projetar os dados nas duas
primeiras dá a "fotografia" 2D que perde menos informação.

Como se calcula:

1. centrar os dados (subtrair a média de cada coluna, aprendida no treino);
2. decompor a matriz centrada em valores singulares: ``X = U · S · Vᵀ``;
3. as linhas de ``Vᵀ`` são as componentes; ``S² / (n - 1)`` é a variância de cada uma.

O sinal de cada componente é arbitrário (``v`` e ``-v`` são a mesma direção). Fixa-se
como no scikit-learn: o maior peso, em valor absoluto, fica positivo. Assim o
resultado é sempre o mesmo e pode comparar-se com o do scikit-learn.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ml_lab.core.foundation.errors import DataError


@dataclass(frozen=True, slots=True, eq=False)
class PCA:
    """Uma PCA já aprendida.

    Attributes:
        mean: Média de cada coluna no treino, ``(n_features,)``.
        components: As componentes, uma por linha, ``(n_components, n_features)``.
        explained_variance: Variância dos dados ao longo de cada componente.
        explained_variance_ratio: Proporção da variância total explicada por cada componente.
    """

    mean: NDArray[np.float64]
    components: NDArray[np.float64]
    explained_variance: NDArray[np.float64]
    explained_variance_ratio: NDArray[np.float64]

    def transform(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        """Coordenadas de ``values`` nas componentes, ``(n_samples, n_components)``."""
        result: NDArray[np.float64] = (values - self.mean) @ self.components.T
        return result


def fit_pca(values: NDArray[np.float64], n_components: int = 2) -> PCA:
    """Aprende a PCA com ``values`` (as linhas de treino).

    Args:
        values: Matriz ``(n_samples, n_features)`` sem valores em falta.
        n_components: Quantas componentes guardar (no máximo, o menor entre linhas e colunas).

    Raises:
        DataError: Se houver menos de duas linhas.
    """
    n_samples, n_features = values.shape
    if n_samples < 2:
        raise DataError("PCA needs at least two rows to measure the variance.")
    k = min(n_components, n_samples, n_features)
    mean = values.mean(axis=0)
    _, singular_values, vt = np.linalg.svd(values - mean, full_matrices=False)
    largest = np.argmax(np.abs(vt), axis=1)
    signs = np.sign(vt[np.arange(vt.shape[0]), largest])
    signs[signs == 0] = 1.0
    vt = vt * signs[:, np.newaxis]
    variance = singular_values**2 / (n_samples - 1)
    total = variance.sum()
    ratio = variance / total if total > 0 else np.zeros_like(variance)
    return PCA(
        mean=mean,
        components=vt[:k],
        explained_variance=variance[:k],
        explained_variance_ratio=ratio[:k],
    )
