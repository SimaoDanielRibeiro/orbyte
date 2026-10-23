"""Operações de limpeza: registo, receita e aplicação com registo na linhagem.

A limpeza é uma receita: uma lista de passos (``CleaningStep``), cada um com o nome
de uma operação registada e os seus parâmetros. Aplicar a mesma receita aos mesmos
dados dá sempre o mesmo resultado, e cada passo deixa uma entrada na linhagem.

A limpeza acontece antes da divisão porque não aprende nada com os dados: remover
duplicados ou converter tipos dá o mesmo resultado em qualquer amostra.

Para acrescentar uma operação: cria uma função ``(frame, **parâmetros) ->
CleaningResult`` num ficheiro desta pasta e decora-a com ``@cleaning_operation``.
"""

import inspect
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ml_lab.core.data.lineage import Lineage, LineageStep, Stage
from ml_lab.core.foundation.errors import ColumnNotFoundError, InvalidParameterError
from ml_lab.core.foundation.notices import Notice
from ml_lab.core.foundation.registry import Registry


@dataclass(frozen=True, slots=True)
class CleaningStep:
    """Um passo da receita de limpeza.

    Attributes:
        operation: Nome da operação no registo ``CLEANING``.
        params: Parâmetros da operação (só valores simples, para poder ir para JSON).
    """

    operation: str
    params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True, eq=False)
class CleaningResult:
    """O que uma operação de limpeza devolve.

    Attributes:
        frame: A tabela nova (a original nunca é alterada).
        step: O registo do que a operação fez, para a linhagem.
        notices: Avisos sobre o resultado (por exemplo, valores que não se converteram).
    """

    frame: pd.DataFrame
    step: LineageStep
    notices: tuple[Notice, ...] = ()


type CleaningFunction = Callable[..., CleaningResult]
"""Função ``(frame, **parâmetros) -> CleaningResult``."""


@dataclass(frozen=True, slots=True)
class CleaningOperation:
    """Uma operação de limpeza registada.

    Attributes:
        name: Identificador, igual ao nome no registo.
        title: Nome a mostrar.
        description: O que faz e quando usar.
        function: A função que a aplica.
    """

    name: str
    title: str
    description: str
    function: CleaningFunction

    def apply(self, frame: pd.DataFrame, params: Mapping[str, Any]) -> CleaningResult:
        """Aplica a operação com ``params``.

        Raises:
            InvalidParameterError: Se os parâmetros não servirem para esta operação.
            ColumnNotFoundError: Se a operação usar uma coluna que não existe.
        """
        try:
            inspect.signature(self.function).bind(frame, **params)
        except TypeError as error:
            raise InvalidParameterError(
                f"Parâmetros inválidos para a operação '{self.name}': {error}."
            ) from error
        return self.function(frame, **params)


CLEANING: Registry[CleaningOperation] = Registry("cleaning", discover="ml_lab.core.data.cleaning")
"""Operações de limpeza, pelo nome."""


def cleaning_operation(
    name: str, *, title: str, description: str
) -> Callable[[CleaningFunction], CleaningFunction]:
    """Decorador que regista uma função como operação de limpeza."""

    def decorator(function: CleaningFunction) -> CleaningFunction:
        CLEANING.add(name, CleaningOperation(name, title, description, function))
        return function

    return decorator


@dataclass(frozen=True, slots=True, eq=False)
class CleanedTable:
    """A tabela depois de toda a receita de limpeza.

    Attributes:
        frame: A tabela limpa.
        lineage: A linhagem, com um passo por operação aplicada.
        notices: Os avisos de todas as operações.
    """

    frame: pd.DataFrame
    lineage: Lineage
    notices: tuple[Notice, ...] = ()


def apply_cleaning(
    frame: pd.DataFrame, steps: Sequence[CleaningStep], lineage: Lineage | None = None
) -> CleanedTable:
    """Aplica a receita de limpeza, passo a passo.

    Args:
        frame: A tabela original; não é alterada.
        steps: Os passos, pela ordem em que se aplicam.
        lineage: Linhagem anterior, à qual se juntam os passos da limpeza.

    Returns:
        A tabela limpa, a linhagem e os avisos.
    """
    history = lineage if lineage is not None else Lineage()
    notices: list[Notice] = []
    for step in steps:
        result = CLEANING.get(step.operation).apply(frame, step.params)
        frame = result.frame
        history = history.then(result.step)
        notices.extend(result.notices)
    return CleanedTable(frame, history, tuple(notices))


def require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    """Confirma que todas as ``columns`` existem em ``frame``.

    Raises:
        ColumnNotFoundError: Para a primeira que não existir.
    """
    available = [str(name) for name in frame.columns]
    for column in columns:
        if column not in frame.columns:
            raise ColumnNotFoundError(column, available)


def cleaning_step(
    operation: str,
    description: str,
    before: pd.DataFrame,
    after: pd.DataFrame,
    params: Mapping[str, Any],
    *,
    changed: Sequence[str] = (),
) -> LineageStep:
    """Monta o passo de linhagem de uma operação de limpeza, com o antes e o depois."""
    before_columns = [str(name) for name in before.columns]
    after_columns = [str(name) for name in after.columns]
    return LineageStep(
        stage=Stage.CLEANING,
        operation=operation,
        description=description,
        params=dict(params),
        rows_before=len(before),
        rows_after=len(after),
        columns_added=tuple(c for c in after_columns if c not in before_columns),
        columns_removed=tuple(c for c in before_columns if c not in after_columns),
        columns_changed=tuple(changed),
    )
