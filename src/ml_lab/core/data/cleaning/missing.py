"""Remoção de linhas com valores em falta.

Remover linhas é a forma mais simples de lidar com nulos, mas perde informação:
no Titanic, remover as linhas sem idade deita fora 20% dos passageiros. Para as
features, costuma ser melhor preencher os nulos na transformação (imputação), que
aprende os valores só com o treino. Para o alvo não há alternativa: uma linha sem
alvo não ensina nada a um modelo supervisionado.
"""

from collections.abc import Sequence
from typing import Literal

import pandas as pd

from ml_lab.core.data.cleaning.base import (
    CleaningResult,
    cleaning_operation,
    cleaning_step,
    require_columns,
)
from ml_lab.core.foundation.errors import InvalidParameterError
from ml_lab.formatting import format_percent, format_quantity


@cleaning_operation(
    "drop_missing_rows",
    title="Drop rows with missing values",
    description=(
        "Drops the rows with missing values in the given columns (or in any column). It's "
        "mostly for the target; for the features, imputation in the transformation is usually "
        "better than losing rows."
    ),
)
def drop_missing_rows(
    frame: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    how: Literal["any", "all"] = "any",
) -> CleaningResult:
    """Remove as linhas com nulos em ``columns`` (``how="all"``: só se todas forem nulas)."""
    if how not in ("any", "all"):
        raise InvalidParameterError(f"Invalid value '{how}' for how: use 'any' or 'all'.")
    subset = list(columns) if columns else None
    if subset:
        require_columns(frame, subset)
    result = frame.dropna(subset=subset, how=how)
    removed = len(frame) - len(result)
    where = f"in {', '.join(subset)}" if subset else "in some column"
    if how == "all":
        where = f"in all the{' given' if subset else ''} columns"
    if removed:
        share = format_percent(removed / len(frame))
        description = (
            f"Dropped {format_quantity(removed, 'row', 'rows')} ({share}) with missing values "
            f"{where}."
        )
    else:
        description = f"There were no rows with missing values {where}."
    params: dict[str, object] = {"how": how}
    if subset:
        params["columns"] = subset
    return CleaningResult(
        result, cleaning_step("drop_missing_rows", description, frame, result, params)
    )
