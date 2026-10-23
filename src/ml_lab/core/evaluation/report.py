"""Relatório de avaliação: as métricas de uma parte dos dados, a matriz de confusão e os avisos.

São estruturas simples (números, texto, tuplos), prontas a mostrar na interface e a
guardar em JSON com a experiência.
"""

import math
from dataclasses import dataclass
from typing import Any

from ml_lab.core.foundation.errors import ComponentNotFoundError
from ml_lab.core.foundation.notices import Notice
from ml_lab.core.foundation.types import Partition, TaskType
from ml_lab.formatting import format_number, format_percent


@dataclass(frozen=True, slots=True)
class MetricResult:
    """O valor de uma métrica, com o que é preciso para o mostrar e explicar.

    Attributes:
        name: Identificador da métrica.
        display_name: Nome a mostrar.
        value: O valor; ``None`` se não estiver definido para estes dados.
        higher_is_better: Se maior é melhor.
        as_percent: Se se mostra como percentagem.
        what: O que mede.
        how_to_read: Como se lê.
    """

    name: str
    display_name: str
    value: float | None
    higher_is_better: bool
    as_percent: bool
    what: str
    how_to_read: str

    @property
    def formatted(self) -> str:
        """O valor escrito à portuguesa (``96,7%``, ``0,12``) ou ``—`` se não existir."""
        if self.value is None or math.isnan(self.value):
            return "—"
        return format_percent(self.value) if self.as_percent else format_number(self.value, 3)


@dataclass(frozen=True, slots=True)
class ConfusionMatrix:
    """Quantos exemplos de cada classe real foram previstos como cada classe.

    Attributes:
        labels: As classes, pela mesma ordem nas linhas e nas colunas.
        counts: ``counts[i][j]`` exemplos da classe real ``labels[i]`` previstos como
            ``labels[j]``. A diagonal são os acertos.
    """

    labels: tuple[str, ...]
    counts: tuple[tuple[int, ...], ...]


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """A avaliação de um modelo numa parte dos dados.

    Attributes:
        partition: A parte avaliada (treino, validação ou teste).
        task_type: O tipo de tarefa.
        n_samples: Quantas linhas foram avaliadas.
        metrics: O valor de cada métrica.
        confusion_matrix: Na classificação, a matriz de confusão.
        notices: Avisos sobre o resultado (por exemplo, não ser melhor do que o acaso).
    """

    partition: Partition
    task_type: TaskType
    n_samples: int
    metrics: tuple[MetricResult, ...]
    confusion_matrix: ConfusionMatrix | None = None
    notices: tuple[Notice, ...] = ()

    def metric(self, name: str) -> MetricResult:
        """O resultado de uma métrica.

        Raises:
            ComponentNotFoundError: Se a métrica não fizer parte do relatório.
        """
        for result in self.metrics:
            if result.name == name:
                return result
        raise ComponentNotFoundError("metrics", name, [m.name for m in self.metrics])

    def values(self) -> dict[str, float | None]:
        """O valor de cada métrica, pelo nome."""
        return {result.name: result.value for result in self.metrics}

    def to_dict(self) -> dict[str, Any]:
        """O relatório em valores simples, para eventos e JSON."""
        return {
            "partition": self.partition.value,
            "task_type": self.task_type.value,
            "n_samples": self.n_samples,
            "metrics": self.values(),
            "confusion_matrix": None
            if self.confusion_matrix is None
            else {
                "labels": list(self.confusion_matrix.labels),
                "counts": [list(row) for row in self.confusion_matrix.counts],
            },
        }
