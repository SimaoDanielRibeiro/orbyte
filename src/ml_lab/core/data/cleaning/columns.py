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
    title="Remover colunas",
    description=(
        "Remove colunas que não ajudam o modelo: identificadores, texto livre, colunas "
        "quase vazias ou constantes."
    ),
    replay=_drop_present,
)
def drop_columns(frame: pd.DataFrame, *, columns: Sequence[str]) -> CleaningResult:
    """Remove as ``columns`` indicadas."""
    names = list(columns)
    if not names:
        raise InvalidParameterError("Indica pelo menos uma coluna a remover.")
    require_columns(frame, names)
    result = frame.drop(columns=names)
    label = "a coluna" if len(names) == 1 else "as colunas"
    description = f"Removida{'s' if len(names) > 1 else ''} {label} {', '.join(names)}."
    return CleaningResult(
        result, cleaning_step("drop_columns", description, frame, result, {"columns": names})
    )
