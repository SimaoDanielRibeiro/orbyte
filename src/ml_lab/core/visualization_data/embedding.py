"""O mapa de uma redução de dimensionalidade: as novas coordenadas dos pontos de treino.

A PCA e o t-SNE não têm fronteira para desenhar: o resultado deles é o próprio mapa.
Desenham-se as duas primeiras coordenadas de cada ponto, pintado com a classe da coluna
de referência (se houver), para se ver se as classes ficam separadas.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ml_lab.core.data.dataset import Dataset
from ml_lab.core.visualization_data.animation import MAX_POINTS, PointCloud


@dataclass(frozen=True, slots=True)
class Embedding:
    """O mapa: os pontos e o nome de cada eixo."""

    points: PointCloud
    axis_labels: tuple[str, str]


def embedding_map(dataset: Dataset, coordinates: NDArray[np.float64], axis_name: str) -> Embedding:
    """O mapa dos pontos de treino com as suas novas coordenadas.

    Args:
        dataset: Os dados (para a coluna de referência e os nomes das classes).
        coordinates: As novas coordenadas dos pontos de treino, ``(n_linhas, k)``.
        axis_name: O nome de cada eixo, sem o número ("Componente", "Dimensão").
    """
    step = max(1, -(-len(coordinates) // MAX_POINTS))
    sample = coordinates[::step]
    y = sample[:, 1] if sample.shape[1] > 1 else np.zeros(len(sample))
    reference = dataset.train.y
    labels: tuple[str, ...] = ()
    if reference is not None and dataset.class_labels:
        labels = tuple(dataset.class_labels[int(code)] for code in reference[::step])
    second = f"{axis_name} 2" if sample.shape[1] > 1 else "—"
    return Embedding(
        points=PointCloud(
            x=tuple(float(v) for v in sample[:, 0]),
            y=tuple(float(v) for v in y),
            labels=labels,
        ),
        axis_labels=(f"{axis_name} 1", second),
    )
