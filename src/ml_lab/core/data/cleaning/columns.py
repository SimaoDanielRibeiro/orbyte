"""Remoção de IDs e de colunas inúteis.

Há colunas que não ajudam o modelo a generalizar: identificadores (diferentes em
cada linha), texto livre, colunas quase vazias ou sempre com o mesmo valor. Um
identificador é especialmente traiçoeiro: um modelo flexível pode decorar "o
passageiro 12 sobreviveu", o que não serve para nenhum passageiro novo.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from ml_lab.core.data.cleaning.base import (
    CleaningResult,
    cleaning_operation,
    cleaning_step,
    require_columns,
)
from ml_lab.core.foundation.errors import InvalidParameterError


def _drop_present(
    frame: pd.DataFrame, params: Mapping[str, Any], _learned: Mapping[str, Any]
) -> pd.DataFrame:
    """Numa linha nova, tira as colunas removidas que ela trouxer (pode não trazer nenhuma)."""
    return frame.drop(columns=[c for c in params["columns"] if c in frame.columns])


@cleaning_operation(
    "drop_columns",
    title="Drop columns",
    description=(
        "Drops columns that don't help the model: identifiers, free text, nearly empty or "
        "constant columns."
    ),
    replay=_drop_present,
)
def drop_columns(frame: pd.DataFrame, *, columns: Sequence[str]) -> CleaningResult:
    """Remove as ``columns`` indicadas."""
    names = list(columns)
    if not names:
        raise InvalidParameterError("Name at least one column to drop.")
    require_columns(frame, names)
    result = frame.drop(columns=names)
    label = "the column" if len(names) == 1 else "the columns"
    description = f"Dropped {label} {', '.join(names)}."
    return CleaningResult(
        result, cleaning_step("drop_columns", description, frame, result, {"columns": names})
    )
