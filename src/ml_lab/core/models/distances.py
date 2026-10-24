"""Distâncias entre pontos, aos blocos, para os modelos que comparam pontos uns com os outros.

O KNN em NumPy, o DBSCAN e o clustering hierárquico (para pontos novos) precisam de saber
quem está mais perto de quem. Uma grelha de 2500 pontos contra 16 000 exemplos de treino
seriam 40 milhões de distâncias de uma vez; aos blocos, a memória fica controlada.
"""

from collections.abc import Iterator
from typing import Final

import numpy as np
from numpy.typing import NDArray

BLOCK_SIZE: Final = 2_000_000
"""Distâncias calculadas de cada vez, no máximo (linhas do bloco × pontos de referência)."""


def pairwise_distances(
    points: NDArray[np.float64], reference: NDArray[np.float64], metric: str = "euclidean"
) -> NDArray[np.float64]:
    """As distâncias de cada ponto a cada ponto de referência, ``(n_pontos, n_referência)``."""
    if metric == "manhattan":
        result: NDArray[np.float64] = np.abs(points[:, None, :] - reference[None, :, :]).sum(axis=2)
        return result
    # |a - b|² = |a|² - 2 a·b + |b|²: sem a matriz das diferenças (pontos × referência × features).
    squared = (
        (points**2).sum(axis=1)[:, None] - 2 * points @ reference.T + (reference**2).sum(axis=1)
    )
    return np.sqrt(np.maximum(squared, 0.0))


def blocks(
    points: NDArray[np.float64], reference: NDArray[np.float64], metric: str = "euclidean"
) -> Iterator[tuple[int, NDArray[np.float64]]]:
    """As distâncias aos blocos de linhas: pares ``(início do bloco, distâncias do bloco)``."""
    per_row = len(reference) * (points.shape[1] if metric == "manhattan" else 1)
    rows = max(1, BLOCK_SIZE // max(1, per_row))
    for start in range(0, len(points), rows):
        yield start, pairwise_distances(points[start : start + rows], reference, metric)


def nearest(
    points: NDArray[np.float64], reference: NDArray[np.float64]
) -> tuple[NDArray[np.intp], NDArray[np.float64]]:
    """Para cada ponto, o índice do ponto de referência mais próximo e a distância a ele."""
    indices = np.zeros(len(points), dtype=np.intp)
    distances = np.zeros(len(points))
    for start, block in blocks(points, reference):
        closest = np.argmin(block, axis=1)
        indices[start : start + len(block)] = closest
        distances[start : start + len(block)] = block[np.arange(len(block)), closest]
    return indices, distances


def same_rows(a: NDArray[np.float64], b: NDArray[np.float64]) -> bool:
    """Se ``a`` e ``b`` são os mesmos dados (por exemplo, os do treino)."""
    return a.shape == b.shape and bool(np.array_equal(a, b))
