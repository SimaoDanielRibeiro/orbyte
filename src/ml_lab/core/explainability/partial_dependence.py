"""Dependência parcial: como a previsão muda, em média, quando uma feature muda.

Para cada valor de uma grelha (do 5.º ao 95.º percentil da feature), põe-se esse valor na
feature de todas as linhas, mantendo as outras como estão, e calcula-se a média das
previsões. A curva mostra o efeito "típico" da feature. As curvas individuais (ICE)
mostram o mesmo para algumas linhas, uma a uma: se forem todas paralelas, o efeito é o
mesmo para toda a gente; se se cruzarem, a feature interage com outras.

Os valores da feature estão na escala depois da preparação (padronizados, se a
preparação os padronizou). Cuidado com features correlacionadas: a grelha cria
combinações que não existem nos dados (uma casa pequena com dez quartos).
"""

from dataclasses import dataclass
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from ml_lab.core.data.dataset import Dataset
from ml_lab.core.foundation.errors import ModelError
from ml_lab.core.foundation.types import TaskType
from ml_lab.core.models.base import LabModel, class_probabilities

GRID_POINTS: Final = 30
"""Valores da feature na grelha, no máximo."""

MAX_ROWS: Final = 500
"""Linhas usadas na média, no máximo (uma amostra, sempre a mesma)."""

ICE_ROWS: Final = 20
"""Linhas com a sua curva individual."""


@dataclass(frozen=True, slots=True)
class PartialDependence:
    """A dependência parcial de uma feature.

    Attributes:
        feature: A feature.
        grid: Os valores da feature (na escala depois da preparação).
        output_labels: O que cada curva média mostra (a probabilidade de cada classe ou o
            valor previsto).
        average: Uma curva por saída, com um valor por ponto da grelha.
        individual: As curvas individuais (ICE) da primeira saída, uma por linha.
        values: Os valores da feature nas linhas usadas (para ver onde há dados).
    """

    feature: str
    grid: tuple[float, ...]
    output_labels: tuple[str, ...]
    average: tuple[tuple[float, ...], ...]
    individual: tuple[tuple[float, ...], ...]
    values: tuple[float, ...]


def partial_dependence(
    model: LabModel[Any], dataset: Dataset, feature: str, *, seed: int
) -> PartialDependence:
    """A dependência parcial de ``feature``, calculada nas linhas de teste.

    Raises:
        ModelError: Se a feature não existir, ou numa tarefa sem alvo.
    """
    if feature not in dataset.feature_names:
        raise ModelError(f"A feature {feature} não existe nos dados preparados.")
    if not dataset.task_type.requires_target:
        raise ModelError("A dependência parcial mostra previsões: só nas tarefas com alvo.")
    column = dataset.feature_names.index(feature)
    rng = np.random.default_rng(seed)
    X = dataset.test.X
    if len(X) > MAX_ROWS:
        X = X[np.sort(rng.choice(len(X), size=MAX_ROWS, replace=False))]
    grid = _grid(X[:, column])
    labels, outputs = _outputs(model, dataset)
    curves = np.zeros((len(grid), len(X), len(labels)))
    for position, value in enumerate(grid):
        changed = X.copy()
        changed[:, column] = value
        curves[position] = outputs(changed)
    ice = rng.choice(len(X), size=min(ICE_ROWS, len(X)), replace=False)
    return PartialDependence(
        feature=feature,
        grid=tuple(float(value) for value in grid),
        output_labels=labels,
        average=tuple(
            tuple(float(v) for v in curves[:, :, k].mean(axis=1)) for k in range(len(labels))
        ),
        individual=tuple(tuple(float(v) for v in curves[:, row, 0]) for row in ice),
        values=tuple(float(value) for value in X[:, column]),
    )


def _grid(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Os valores a experimentar: os próprios, se forem poucos (0 e 1); senão, percentis."""
    distinct = np.unique(values)
    if len(distinct) <= GRID_POINTS:
        return distinct
    return np.unique(np.percentile(values, np.linspace(5, 95, GRID_POINTS)))


def _outputs(model: LabModel[Any], dataset: Dataset) -> tuple[tuple[str, ...], Any]:
    """O nome de cada saída e a função que a calcula para uma matriz de features."""
    task = dataset.task_type
    if task is TaskType.REGRESSION:
        label = f"Previsão de {dataset.target_name}" if dataset.target_name else "Previsão"
        return (label,), lambda X: np.asarray(model.predict(X), dtype=np.float64)[:, None]
    labels = dataset.class_labels
    if model.info.supports_proba:

        def probabilities(X: NDArray[np.float64]) -> NDArray[np.float64]:
            proba = class_probabilities(model, X, len(labels))
            return proba[:, 1:] if len(labels) == 2 else proba

        names = labels[1:] if len(labels) == 2 else labels
        return tuple(f"Probabilidade de «{name}»" for name in names), probabilities

    def shares(X: NDArray[np.float64]) -> NDArray[np.float64]:
        predicted = np.asarray(model.predict(X), dtype=np.int64)
        return np.eye(len(labels))[predicted]

    return tuple(f"Proporção prevista como «{name}»" for name in labels), shares
