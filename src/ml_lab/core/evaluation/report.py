"""Relatório de avaliação: as métricas de uma parte dos dados, a matriz de confusão e os avisos.

São estruturas simples (números, texto, tuplos), prontas a mostrar na interface e a
passar a JSON com ``schemas.serialization.to_json`` (para os eventos e para a API).
"""

import math
from dataclasses import dataclass

from ml_lab.core.evaluation.metrics.base import MetricTerm
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
        higher_is_better: Se maior é melhor; ``None`` nas métricas que só descrevem.
        as_percent: Se se mostra como percentagem.
        what: O que mede.
        how_to_read: Como se lê.
        terms: Os termos da fórmula nestes dados (os verdadeiros positivos, a soma dos
            erros…), nas métricas que são uma conta com quantidades que se contam. A fórmula
            está no catálogo das métricas (``CatalogService.metrics``).
    """

    name: str
    display_name: str
    value: float | None
    higher_is_better: bool | None
    as_percent: bool
    what: str
    how_to_read: str
    terms: tuple[MetricTerm, ...] = ()

    @property
    def formatted(self) -> str:
        """O valor escrito à portuguesa (``96,7%``, ``0,12``) ou ``—`` se não existir."""
        if self.value is None or math.isnan(self.value):
            return "—"
        if self.as_percent:
            return format_percent(self.value)
        if self.value.is_integer() and self.higher_is_better is None:  # contagens: 3 grupos
            return format_number(self.value, 0)
        return format_number(self.value, 3)


@dataclass(frozen=True, slots=True)
class ConfusionMatrix:
    """Quantos exemplos de cada classe real foram previstos como cada classe.

    Attributes:
        labels: As classes reais (as linhas) e, se ``column_labels`` estiver vazio, também as
            previstas (as colunas).
        counts: ``counts[i][j]`` exemplos da classe real ``labels[i]`` previstos como
            a coluna ``j``. Com as mesmas classes nos dois lados, a diagonal são os acertos.
        column_labels: Os nomes das colunas, quando não são as mesmas classes (no
            clustering, os grupos encontrados contra as classes da coluna de referência).
    """

    labels: tuple[str, ...]
    counts: tuple[tuple[int, ...], ...]
    column_labels: tuple[str, ...] = ()

    @property
    def columns(self) -> tuple[str, ...]:
        """Os nomes das colunas."""
        return self.column_labels or self.labels


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
        result = self.find(name)
        if result is None:
            raise ComponentNotFoundError("metrics", name, [m.name for m in self.metrics])
        return result

    def find(self, name: str) -> MetricResult | None:
        """O resultado de uma métrica, ou ``None`` se ficou de fora do relatório."""
        return next((result for result in self.metrics if result.name == name), None)

    def values(self) -> dict[str, float | None]:
        """O valor de cada métrica, pelo nome."""
        return {result.name: result.value for result in self.metrics}
