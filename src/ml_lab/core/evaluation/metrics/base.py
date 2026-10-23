"""Métricas: funções puras registadas por tipo de tarefa.

Uma métrica recebe os valores reais e os previstos (e, se precisar, as
probabilidades) e devolve um número. Não guarda estado: o mesmo input dá sempre o
mesmo resultado (regra 4 dos contratos).

Para acrescentar uma métrica: uma função em ``metrics/<tipo>.py`` decorada com
``@metric(...)``, e a sua explicação em ``evaluation/explanations.py``. Aparece
sozinha em todos os relatórios dos tipos de tarefa indicados.
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ml_lab.core.foundation.registry import Registry
from ml_lab.core.foundation.types import TaskType


@dataclass(frozen=True, slots=True, eq=False)
class MetricInputs:
    """O que uma métrica recebe.

    Attributes:
        y_true: Valores reais (na classificação, o número de cada classe).
        y_pred: Valores previstos.
        y_proba: Probabilidade de cada classe, ``(n_linhas, n_classes)``, se o modelo as der.
        n_classes: Número de classes (0 fora da classificação).
    """

    y_true: NDArray[Any]
    y_pred: NDArray[Any]
    y_proba: NDArray[np.float64] | None = None
    n_classes: int = 0

    @property
    def labels(self) -> list[int]:
        """Os números de todas as classes (``0 … n_classes - 1``)."""
        return list(range(self.n_classes))


type MetricFunction = Callable[[MetricInputs], float]


@dataclass(frozen=True, slots=True)
class Metric:
    """Uma métrica registada.

    Attributes:
        name: Identificador, igual ao nome no registo.
        display_name: Nome a mostrar.
        function: A função pura que a calcula.
        task_types: Tipos de tarefa em que faz sentido.
        higher_is_better: Se um valor maior é melhor.
        needs_proba: Se precisa das probabilidades (fica de fora se o modelo não as der).
        as_percent: Se se mostra como percentagem (proporções entre 0 e 1).
    """

    name: str
    display_name: str
    function: MetricFunction
    task_types: frozenset[TaskType]
    higher_is_better: bool
    needs_proba: bool = False
    as_percent: bool = False

    def __call__(self, inputs: MetricInputs) -> float:
        """Calcula a métrica."""
        return float(self.function(inputs))


METRICS: Registry[Metric] = Registry(
    "metrics",
    task_types=lambda metric: metric.task_types,
    discover="ml_lab.core.evaluation.metrics",
)
"""Todas as métricas, pelo nome."""


def metric(
    name: str,
    *,
    display_name: str,
    task_types: Iterable[TaskType],
    higher_is_better: bool,
    needs_proba: bool = False,
    as_percent: bool = False,
) -> Callable[[MetricFunction], MetricFunction]:
    """Decorador que regista uma função como métrica."""

    def decorator(function: MetricFunction) -> MetricFunction:
        METRICS.add(
            name,
            Metric(
                name,
                display_name,
                function,
                frozenset(task_types),
                higher_is_better,
                needs_proba,
                as_percent,
            ),
        )
        return function

    return decorator
