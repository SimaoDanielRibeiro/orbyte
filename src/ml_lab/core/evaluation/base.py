"""Contrato Evaluator: avaliar um modelo para um tipo de tarefa.

Há um avaliador por tipo de tarefa (regra 5 dos contratos). Cada um recolhe sozinho
todas as métricas registadas para o seu tipo, por isso uma métrica nova aparece nos
relatórios sem mexer no avaliador.
"""

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any, ClassVar, Final

from ml_lab.core.evaluation.explanations import METRIC_EXPLANATIONS
from ml_lab.core.evaluation.metrics.base import METRICS, MetricInputs
from ml_lab.core.evaluation.report import EvaluationReport, MetricResult
from ml_lab.core.foundation.errors import ModelError
from ml_lab.core.foundation.notices import Notice
from ml_lab.core.foundation.registry import Registry
from ml_lab.core.foundation.types import Partition, TaskType
from ml_lab.core.models.base import Features, LabModel, Target

PRIMARY_METRIC: Final[Mapping[TaskType, str]] = MappingProxyType(
    {
        TaskType.BINARY_CLASSIFICATION: "accuracy",
        TaskType.MULTICLASS_CLASSIFICATION: "accuracy",
        TaskType.REGRESSION: "r2",
        TaskType.CLUSTERING: "silhouette",
        TaskType.DIMENSIONALITY_REDUCTION: "trustworthiness",
        TaskType.ANOMALY_DETECTION: "anomaly_rate",
    }
)
"""A métrica que resume cada tipo de tarefa (a do "último resultado" e a do overfitting)."""


class Evaluator(ABC):
    """Avalia modelos de um ou mais tipos de tarefa.

    Attributes:
        task_types: Os tipos de tarefa que sabe avaliar.
    """

    task_types: ClassVar[frozenset[TaskType]]

    @abstractmethod
    def evaluate(
        self,
        model: LabModel[Any],
        X: Features,
        y: Target | None,
        *,
        partition: Partition,
        task_type: TaskType,
        class_labels: tuple[str, ...] = (),
    ) -> EvaluationReport:
        """Avalia ``model`` nas linhas ``X`` (com o alvo ``y``) de uma parte dos dados.

        Args:
            model: O modelo já treinado.
            X: As features.
            y: O alvo real.
            partition: A parte avaliada.
            task_type: O tipo de tarefa.
            class_labels: Na classificação, o nome de cada classe.
        """


EVALUATORS: Registry[type[Evaluator]] = Registry(
    "evaluators",
    task_types=lambda evaluator: evaluator.task_types,
    discover="ml_lab.core.evaluation",
)
"""Os avaliadores, pelo nome; ``evaluator_for`` escolhe o do tipo de tarefa."""


def evaluator_for(task_type: TaskType) -> Evaluator:
    """O avaliador de ``task_type``.

    Raises:
        ModelError: Se ainda não houver avaliador para esse tipo de tarefa.
    """
    found = EVALUATORS.for_task(task_type)
    if not found:
        raise ModelError(f"Ainda não há avaliador para {task_type.label.lower()}.")
    return next(iter(found.values()))()


def measure(
    task_type: TaskType, inputs: MetricInputs, *, order: Sequence[str] = ()
) -> tuple[tuple[MetricResult, ...], tuple[str, ...]]:
    """Calcula todas as métricas registadas para ``task_type``, cada uma com a sua explicação.

    As métricas a que falta alguma coisa (as probabilidades de um modelo que não as dá, a
    coluna de referência de um clustering) ficam de fora.

    Args:
        task_type: O tipo de tarefa.
        inputs: O que as métricas recebem.
        order: A ordem em que aparecem; as que faltarem na lista vêm depois, por nome.

    Returns:
        Os resultados e os nomes (a mostrar) das métricas que não ficaram definidas.
    """

    def rank(name: str) -> tuple[int, str]:
        return (list(order).index(name) if name in order else len(order), name)

    results: list[MetricResult] = []
    undefined: list[str] = []
    for name in sorted(METRICS.for_task(task_type), key=rank):
        metric = METRICS.get(name)
        if not metric.applies_to(inputs):
            continue
        value = metric(inputs)
        if math.isnan(value):
            undefined.append(metric.display_name)
        explanation = METRIC_EXPLANATIONS[name]
        results.append(
            MetricResult(
                name=name,
                display_name=metric.display_name,
                value=None if math.isnan(value) else value,
                higher_is_better=metric.higher_is_better,
                as_percent=metric.as_percent,
                what=explanation.what,
                how_to_read=explanation.how_to_read,
            )
        )
    return tuple(results), tuple(undefined)


def undefined_notices(undefined: Sequence[str], partition: Partition, reason: str) -> list[Notice]:
    """O aviso das métricas que ficaram por definir numa parte dos dados (se houver alguma).

    Args:
        undefined: Os nomes das métricas indefinidas (o segundo valor de ``measure``).
        partition: A parte avaliada.
        reason: Porque é que, nesta tarefa, uma métrica fica por definir.
    """
    if not undefined:
        return []
    message = f"{', '.join(undefined)}: não está definida no {partition.label.lower()} ({reason})."
    return [Notice.info("undefined_metric", message)]
