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
    title="Remover linhas com valores em falta",
    description=(
        "Remove as linhas com valores em falta nas colunas indicadas (ou em qualquer "
        "coluna). Usa-se sobretudo para o alvo; para as features, a imputação na "
        "transformação costuma ser melhor do que perder linhas."
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
        raise InvalidParameterError(f"Valor '{how}' inválido para how: usa 'any' ou 'all'.")
    subset = list(columns) if columns else None
    if subset:
        require_columns(frame, subset)
    result = frame.dropna(subset=subset, how=how)
    removed = len(frame) - len(result)
    where = f"em {', '.join(subset)}" if subset else "em alguma coluna"
    if how == "all":
        where = f"em todas as colunas{' indicadas' if subset else ''}"
    if removed:
        share = format_percent(removed / len(frame))
        description = (
            f"{'Removida' if removed == 1 else 'Removidas'} "
            f"{format_quantity(removed, 'linha', 'linhas')} ({share}) com valores em falta "
            f"{where}."
        )
    else:
        description = f"Não havia linhas com valores em falta {where}."
    params: dict[str, object] = {"how": how}
    if subset:
        params["columns"] = subset
    return CleaningResult(
        result, cleaning_step("drop_missing_rows", description, frame, result, params)
    )
