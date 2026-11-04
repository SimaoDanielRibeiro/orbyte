"""A grelha da sonda: os pontos onde se pedem as previsões do modelo para desenhar.

Para ver a fronteira de decisão de um modelo, pergunta-se ao modelo o que preveria em
cada ponto de uma grelha que cobre os dados. Os dados vivem em muitas dimensões, mas
o ecrã só tem duas, por isso a grelha é desenhada no plano da projeção 2D (a mesma do
ecrã Preparar) e cada ponto é levado de volta ao espaço das features com
``Projection.to_features``. Com uma só feature, a grelha é uma reta.
"""

from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
from numpy.typing import NDArray

from ml_lab.core.data.dataset import Dataset
from ml_lab.core.foundation.types import Partition

GRID_RESOLUTION: Final = 50
"""Pontos por eixo numa grelha 2D (50 × 50 = 2500 previsões por snapshot)."""

LINE_RESOLUTION: Final = 150
"""Pontos numa grelha 1D."""

_PADDING: Final = 0.1
"""Margem à volta dos dados, em proporção da sua extensão."""


@dataclass(frozen=True, slots=True, eq=False)
class ProbeGrid:
    """Pontos onde se querem as previsões, no plano do ecrã e no espaço das features.

    Attributes:
        dimensions: 2 (grelha no plano) ou 1 (reta, quando só há uma feature).
        xs: Valores no eixo horizontal.
        ys: Valores no eixo vertical (vazio em 1D).
        points: Os mesmos pontos no espaço das features, ``(n_pontos, n_features)``, por
            linhas da grelha (primeiro todos os ``xs`` do primeiro ``ys``...).
        axis_labels: O nome de cada eixo.
    """

    dimensions: Literal[1, 2]
    xs: NDArray[np.float64]
    ys: NDArray[np.float64]
    points: NDArray[np.float64]
    axis_labels: tuple[str, str]

    @property
    def shape(self) -> tuple[int, int]:
        """Linhas e colunas da grelha (em 1D, uma só linha)."""
        return (len(self.ys) if self.dimensions == 2 else 1, len(self.xs))


def probe_grid(dataset: Dataset, resolution: int = GRID_RESOLUTION) -> ProbeGrid | None:
    """A grelha que cobre os dados de treino de ``dataset``.

    Returns:
        A grelha, ou ``None`` se não houver como desenhar (sem features ou sem projeção).
    """
    train = dataset.train.X
    if dataset.n_features == 1:
        xs = _axis(train[:, 0], LINE_RESOLUTION)
        target = dataset.target_name or "value"
        return ProbeGrid(1, xs, np.zeros(0), xs.reshape(-1, 1), (dataset.feature_names[0], target))
    projection = dataset.projection
    if projection is None or projection.center is None:
        return None
    plane = projection.coordinates[Partition.TRAIN]
    xs = _axis(plane[:, 0], resolution)
    ys = _axis(plane[:, 1], resolution)
    grid_x, grid_y = np.meshgrid(xs, ys)
    flat = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    return ProbeGrid(2, xs, ys, projection.to_features(flat), projection.axis_labels)


def _axis(values: NDArray[np.float64], resolution: int) -> NDArray[np.float64]:
    low, high = float(values.min()), float(values.max())
    margin = (high - low) * _PADDING or 1.0
    return np.linspace(low - margin, high + margin, resolution)
