"""Remoção de linhas duplicadas.

Uma linha repetida conta duas vezes no treino e, pior, pode calhar uma cópia no
treino e outra no teste: o modelo é avaliado num exemplo que já viu, e o teste
parece melhor do que devia.
"""

from collections.abc import Sequence

import pandas as pd

from ml_lab.core.data.cleaning.base import (
    CleaningResult,
    cleaning_operation,
    cleaning_step,
    require_columns,
)
from ml_lab.formatting import format_quantity


@cleaning_operation(
    "drop_duplicates",
    title="Remover linhas duplicadas",
    description=(
        "Remove as linhas repetidas e fica com a primeira de cada grupo. Com colunas "
        "indicadas, duas linhas contam como iguais quando coincidem nessas colunas."
    ),
)
def drop_duplicates(frame: pd.DataFrame, *, columns: Sequence[str] | None = None) -> CleaningResult:
    """Remove as linhas duplicadas (em todas as colunas ou só nas ``columns`` dadas)."""
    subset = list(columns) if columns else None
    if subset:
        require_columns(frame, subset)
    duplicated = frame.duplicated(subset=subset, keep="first")
    n_duplicated = int(duplicated.sum())
    result = frame.loc[~duplicated]
    scope = f" (comparando {', '.join(subset)})" if subset else ""
    if n_duplicated:
        removed = format_quantity(n_duplicated, "linha duplicada", "linhas duplicadas")
        description = f"{_removed(n_duplicated)} {removed}{scope}."
    else:
        description = f"Não havia linhas duplicadas{scope}."
    params = {"columns": subset} if subset else {}
    return CleaningResult(
        result, cleaning_step("drop_duplicates", description, frame, result, params)
    )


def _removed(count: int) -> str:
    return "Removida" if count == 1 else "Removidas"
