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
    title="Drop duplicate rows",
    description=(
        "Drops the repeated rows and keeps the first of each group. With columns given, two "
        "rows count as equal when they match on those columns."
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
    scope = f" (comparing {', '.join(subset)})" if subset else ""
    if n_duplicated:
        removed = format_quantity(n_duplicated, "duplicate row", "duplicate rows")
        description = f"Dropped {removed}{scope}."
    else:
        description = f"There were no duplicate rows{scope}."
    params = {"columns": subset} if subset else {}
    return CleaningResult(
        result, cleaning_step("drop_duplicates", description, frame, result, params)
    )
