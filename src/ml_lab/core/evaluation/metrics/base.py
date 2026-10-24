"""Métricas: funções puras registadas por tipo de tarefa.

Uma métrica recebe os valores reais e os previstos (e, se precisar, as
probabilidades) e devolve um número. Não guarda estado: o mesmo input dá sempre o
mesmo resultado (regra 4 dos contratos).

Para acrescentar uma métrica: uma função em ``metrics/<tipo>.py`` decorada com
``@metric(...)``, e a sua explicação em ``evaluation/explanations.py``. Aparece
sozinha em todos os relatórios dos tipos de tarefa indicados.

Cada métrica declara o que precisa de receber (``needs``): os valores reais, as
probabilidades, as features… Quando falta alguma coisa (um modelo sem probabilidades,
um clustering sem coluna de referência), a métrica fica de fora do relatório.
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from ml_lab.core.foundation.errors import ModelError
from ml_lab.core.foundation.registry import Registry
from ml_lab.core.foundation.types import TaskType

INPUT_NAMES: Final = ("y_true", "y_pred", "y_proba", "X", "embedding", "scores")
"""O que uma métrica pode pedir em ``needs``."""


@dataclass(frozen=True, slots=True, eq=False)
class MetricInputs:
    """O que uma métrica recebe (cada uma usa só o que declara em ``needs``).

    Attributes:
        y_true: Valores reais (na classificação, o número de cada classe). Nas tarefas não
            supervisionadas, a coluna de referência, se houver.
        y_pred: Valores previstos: classes, valores, grupos ou (1 = anomalia, 0 = normal).
        y_proba: Probabilidade de cada classe, ``(n_linhas, n_classes)``, se o modelo as der.
        n_classes: Número de classes (0 fora da classificação).
        X: As features (o clustering mede a separação dos grupos no espaço das features).
        embedding: Na redução de dimensionalidade, as novas coordenadas de cada linha.
        scores: Na deteção de anomalias, a pontuação de cada linha (maior = mais estranha).
    """

    y_true: NDArray[Any] | None = None
    y_pred: NDArray[Any] | None = None
    y_proba: NDArray[np.float64] | None = None
    n_classes: int = 0
    X: NDArray[np.float64] | None = None
    embedding: NDArray[np.float64] | None = None
    scores: NDArray[np.float64] | None = None

    @property
    def labels(self) -> list[int]:
        """Os números de todas as classes (``0 … n_classes - 1``)."""
        return list(range(self.n_classes))

    @property
    def truth(self) -> NDArray[Any]:
        """Os valores reais (para as métricas que declaram ``y_true`` em ``needs``)."""
        return _required(self.y_true, "y_true")

    @property
    def predicted(self) -> NDArray[Any]:
        """Os valores previstos (para as métricas que declaram ``y_pred`` em ``needs``)."""
        return _required(self.y_pred, "y_pred")

    def has(self, name: str) -> bool:
        """Se recebeu ``name`` (um dos ``INPUT_NAMES``)."""
        return getattr(self, name) is not None


def _required(values: NDArray[Any] | None, name: str) -> NDArray[Any]:
    if values is None:
        raise ModelError(f"Esta métrica precisa de {name}, que não foi dado.")
    return values


type MetricFunction = Callable[[MetricInputs], float]


@dataclass(frozen=True, slots=True)
class Metric:
    """Uma métrica registada.

    Attributes:
        name: Identificador, igual ao nome no registo.
        display_name: Nome a mostrar.
        function: A função pura que a calcula.
        task_types: Tipos de tarefa em que faz sentido.
        higher_is_better: Se um valor maior é melhor; ``None`` nas métricas que só
            descrevem (o número de grupos, a proporção de anomalias).
        needs: O que precisa de receber (nomes de ``INPUT_NAMES``).
        as_percent: Se se mostra como percentagem (proporções entre 0 e 1).
    """

    name: str
    display_name: str
    function: MetricFunction
    task_types: frozenset[TaskType]
    higher_is_better: bool | None
    needs: frozenset[str] = frozenset({"y_true", "y_pred"})
    as_percent: bool = False

    @property
    def needs_proba(self) -> bool:
        """Se precisa das probabilidades (fica de fora se o modelo não as der)."""
        return "y_proba" in self.needs

    def applies_to(self, inputs: MetricInputs) -> bool:
        """Se ``inputs`` tem tudo o que a métrica precisa."""
        return all(inputs.has(name) for name in self.needs)

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
    higher_is_better: bool | None,
    needs: Iterable[str] = ("y_true", "y_pred"),
    needs_proba: bool = False,
    as_percent: bool = False,
) -> Callable[[MetricFunction], MetricFunction]:
    """Decorador que regista uma função como métrica.

    Args:
        name: O identificador.
        display_name: O nome a mostrar.
        task_types: Os tipos de tarefa em que faz sentido.
        higher_is_better: Se maior é melhor (``None``: só descreve).
        needs: O que precisa de receber; por omissão, os valores reais e os previstos.
        needs_proba: Atalho para juntar as probabilidades a ``needs``.
        as_percent: Se se mostra como percentagem.
    """
    required = frozenset(needs) | ({"y_proba"} if needs_proba else set())
    unknown = required - set(INPUT_NAMES)
    if unknown:
        raise ValueError(f"A métrica {name} pede entradas que não existem: {sorted(unknown)}.")

    def decorator(function: MetricFunction) -> MetricFunction:
        METRICS.add(
            name,
            Metric(
                name,
                display_name,
                function,
                frozenset(task_types),
                higher_is_better,
                required,
                as_percent,
            ),
        )
        return function

    return decorator
