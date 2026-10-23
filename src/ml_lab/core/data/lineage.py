"""Linhagem: o histórico de todas as operações aplicadas aos dados.

Cada operação ("removidas 12 linhas duplicadas", "coluna Age convertida para
número") fica registada como um passo, com o antes e o depois. A interface mostra
este histórico, e os parâmetros de cada passo permitem reproduzir a preparação.

A linhagem é imutável: ``then`` devolve uma linhagem nova com os passos
acrescentados, por isso nenhuma etapa reescreve o que as anteriores registaram.

Exemplo:
    >>> lineage = Lineage().then(
    ...     LineageStep(
    ...         Stage.CLEANING,
    ...         "drop_duplicates",
    ...         "Removidas 12 linhas.",
    ...         rows_before=100,
    ...         rows_after=88,
    ...     )
    ... )
    >>> [step.operation for step in lineage]
    ['drop_duplicates']
    >>> lineage[0].rows_removed
    12
"""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Final, Self


class Stage(StrEnum):
    """Etapa da camada de dados em que a operação aconteceu (ver ``core/data/README.md``)."""

    INPUT = "input"
    CLEANING = "cleaning"
    TASK = "task"
    SPLIT = "split"
    TRANSFORM = "transform"
    PROJECTION = "projection"
    OUTPUT = "output"

    @property
    def label(self) -> str:
        """Nome da etapa em português."""
        return _STAGE_LABELS[self]


_STAGE_LABELS: Final[Mapping[Stage, str]] = MappingProxyType(
    {
        Stage.INPUT: "Entrada",
        Stage.CLEANING: "Limpeza",
        Stage.TASK: "Tarefa",
        Stage.SPLIT: "Divisão",
        Stage.TRANSFORM: "Transformação",
        Stage.PROJECTION: "Projeção",
        Stage.OUTPUT: "Saída",
    }
)


@dataclass(frozen=True, slots=True)
class LineageStep:
    """Uma operação aplicada aos dados.

    Attributes:
        stage: Etapa em que aconteceu.
        operation: Nome estável da operação, por exemplo ``"drop_duplicates"``.
        description: O que aconteceu, em português, com os números concretos.
        params: Parâmetros usados (só valores simples, serializáveis em JSON).
        rows_before: Linhas antes da operação, se a operação as conta.
        rows_after: Linhas depois da operação.
        columns_added: Colunas criadas.
        columns_removed: Colunas removidas.
        columns_changed: Colunas que ficaram com valores ou tipo diferentes.
    """

    stage: Stage
    operation: str
    description: str
    params: Mapping[str, Any] = field(default_factory=dict)
    rows_before: int | None = None
    rows_after: int | None = None
    columns_added: tuple[str, ...] = ()
    columns_removed: tuple[str, ...] = ()
    columns_changed: tuple[str, ...] = ()

    @property
    def rows_removed(self) -> int:
        """Quantas linhas a operação removeu (0 se não removeu ou não conta linhas)."""
        if self.rows_before is None or self.rows_after is None:
            return 0
        return self.rows_before - self.rows_after

    def to_record(self) -> dict[str, Any]:
        """O passo como dicionário simples, para tabelas e JSON."""
        return {
            "stage": self.stage.value,
            "stage_label": self.stage.label,
            "operation": self.operation,
            "description": self.description,
            "params": dict(self.params),
            "rows_before": self.rows_before,
            "rows_after": self.rows_after,
            "columns_added": list(self.columns_added),
            "columns_removed": list(self.columns_removed),
            "columns_changed": list(self.columns_changed),
        }


@dataclass(frozen=True, slots=True)
class Lineage:
    """Sequência ordenada de operações aplicadas aos dados.

    Attributes:
        steps: Os passos, do primeiro ao último.
    """

    steps: tuple[LineageStep, ...] = ()

    def then(self, *steps: LineageStep) -> Self:
        """Linhagem nova com ``steps`` acrescentados no fim."""
        return type(self)((*self.steps, *steps))

    def for_stage(self, stage: Stage) -> tuple[LineageStep, ...]:
        """Os passos de uma etapa."""
        return tuple(step for step in self.steps if step.stage is stage)

    def to_records(self) -> list[dict[str, Any]]:
        """Todos os passos como dicionários simples."""
        return [step.to_record() for step in self.steps]

    def __iter__(self) -> Iterator[LineageStep]:
        return iter(self.steps)

    def __len__(self) -> int:
        return len(self.steps)

    def __getitem__(self, index: int) -> LineageStep:
        return self.steps[index]
