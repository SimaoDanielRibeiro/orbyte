"""Deteção de valores extremos e, se pedido, o seu tratamento.

Usa a regra do IQR (intervalo interquartil): um valor é extremo se estiver abaixo
de ``Q1 - 1,5·IQR`` ou acima de ``Q3 + 1,5·IQR``, em que ``IQR = Q3 - Q1``. Ao
contrário da média e do desvio padrão, os quartis quase não mudam por causa dos
próprios extremos, por isso a regra não se deixa enganar por eles.

Um valor extremo não é necessariamente um erro: no Titanic, as tarifas de 500
libras eram reais (primeira classe). Por isso a deteção só avisa, e o tratamento
só acontece se o utilizador o pedir.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

from ml_lab.core.data.cleaning.base import (
    CleaningResult,
    cleaning_operation,
    cleaning_step,
    require_columns,
)
from ml_lab.core.foundation.errors import DataError, InvalidParameterError
from ml_lab.formatting import format_number, format_quantity

DEFAULT_FACTOR = 1.5
GROSS_FACTOR = 10.0
"""O fator dos valores que já não são extremos, são erros: dez IQR para lá dos quartis."""


@dataclass(frozen=True, slots=True)
class OutlierBounds:
    """Limites da regra do IQR e quantos valores ficam fora deles.

    Attributes:
        lower: Limite inferior, ``Q1 - fator·IQR``.
        upper: Limite superior, ``Q3 + fator·IQR``.
        n_below: Valores abaixo do limite inferior.
        n_above: Valores acima do limite superior.
    """

    lower: float
    upper: float
    n_below: int
    n_above: int

    @property
    def n_outliers(self) -> int:
        """Total de valores extremos."""
        return self.n_below + self.n_above


def iqr_bounds(values: pd.Series, factor: float = DEFAULT_FACTOR) -> OutlierBounds | None:
    """Calcula os limites do IQR para uma coluna numérica (os nulos são ignorados).

    Returns:
        Os limites, ou ``None`` se a coluna não tiver valores.
    """
    present = values.dropna()
    if present.empty:
        return None
    q1, q3 = (float(q) for q in present.quantile([0.25, 0.75]))
    spread = q3 - q1
    lower, upper = q1 - factor * spread, q3 + factor * spread
    return OutlierBounds(
        lower=lower,
        upper=upper,
        n_below=int((present < lower).sum()),
        n_above=int((present > upper).sum()),
    )


def _clip_new_rows(
    frame: pd.DataFrame, params: Mapping[str, Any], learned: Mapping[str, Any]
) -> pd.DataFrame:
    """Numa linha nova, limita o valor aos limites que se calcularam com os dados.

    Uma linha sozinha não tem quartis: os limites são os da tabela. Quando o método foi
    remover, a linha nova fica como está (não se remove uma linha que se quer prever).
    """
    column = params["column"]
    if params.get("method", "clip") != "clip" or "lower" not in learned or column not in frame:
        return frame
    return frame.assign(**{column: frame[column].clip(learned["lower"], learned["upper"])})


@cleaning_operation(
    "treat_outliers",
    title="Treat extreme values",
    description=(
        "Detects the extreme values of a numeric column with the IQR rule and caps them at the "
        "limits (clip) or drops the rows where they appear (remove). The limits are computed "
        "with every row, before the split: use it only when the extremes are errors or when "
        "you want to see their effect."
    ),
    replay=_clip_new_rows,
)
def treat_outliers(
    frame: pd.DataFrame,
    *,
    column: str,
    method: Literal["clip", "remove"] = "clip",
    factor: float = DEFAULT_FACTOR,
) -> CleaningResult:
    """Limita (``clip``) ou remove (``remove``) os valores extremos de ``column``."""
    require_columns(frame, [column])
    if method not in ("clip", "remove"):
        raise InvalidParameterError(f"Unknown method '{method}': use 'clip' or 'remove'.")
    if factor <= 0:
        raise InvalidParameterError("The IQR factor must be positive (1.5 is the usual one).")
    if not pd.api.types.is_numeric_dtype(frame[column]) or pd.api.types.is_bool_dtype(
        frame[column]
    ):
        raise DataError(
            f"The column '{column}' isn't numeric: convert it first (the convert_type operation)."
        )
    bounds = iqr_bounds(frame[column], factor)
    params = {"column": column, "method": method, "factor": factor}
    # Os limites ficam guardados mesmo sem extremos: uma linha nova pode trazer um.
    learned = {} if bounds is None else {"lower": bounds.lower, "upper": bounds.upper}
    if bounds is None or bounds.n_outliers == 0:
        description = f"The column {column} has no extreme values (IQR rule)."
        return CleaningResult(
            frame,
            cleaning_step("treat_outliers", description, frame, frame, params),
            learned=learned,
        )

    # Por extenso ("de -6.69 a 64.81"): com a vírgula dos milhares, "[1,234.5, 2,000]" lia-se mal.
    limits = f"the range {format_number(bounds.lower)} to {format_number(bounds.upper)}"
    n_outliers = bounds.n_outliers
    if method == "clip":
        result = frame.assign(**{column: frame[column].clip(bounds.lower, bounds.upper)})
        values = format_quantity(n_outliers, "extreme value capped", "extreme values capped")
        description = f"{column}: {values} to {limits}."
        step = cleaning_step("treat_outliers", description, frame, result, params, changed=[column])
    else:
        inside = frame[column].isna() | frame[column].between(bounds.lower, bounds.upper)
        result = frame.loc[inside]
        rows = format_quantity(n_outliers, "row", "rows")
        description = f"{column}: dropped {rows} with values outside {limits}."
        step = cleaning_step("treat_outliers", description, frame, result, params)
    return CleaningResult(result, step, learned=learned)
