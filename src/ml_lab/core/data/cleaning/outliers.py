"""Deteção de valores extremos e, se pedido, o seu tratamento.

Usa a regra do IQR (intervalo interquartil): um valor é extremo se estiver abaixo
de ``Q1 - 1,5·IQR`` ou acima de ``Q3 + 1,5·IQR``, em que ``IQR = Q3 - Q1``. Ao
contrário da média e do desvio padrão, os quartis quase não mudam por causa dos
próprios extremos, por isso a regra não se deixa enganar por eles.

Um valor extremo não é necessariamente um erro: no Titanic, as tarifas de 500
libras eram reais (primeira classe). Por isso a deteção só avisa, e o tratamento
só acontece se o utilizador o pedir.
"""

from dataclasses import dataclass
from typing import Literal

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


@cleaning_operation(
    "treat_outliers",
    title="Tratar valores extremos",
    description=(
        "Deteta os valores extremos de uma coluna numérica com a regra do IQR e limita-os "
        "aos limites (clip) ou remove as linhas onde aparecem (remove). Os limites são "
        "calculados com todas as linhas, antes da divisão: usa só quando os extremos forem "
        "erros ou quando quiseres ver o seu efeito."
    ),
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
        raise InvalidParameterError(f"Método '{method}' desconhecido: usa 'clip' ou 'remove'.")
    if factor <= 0:
        raise InvalidParameterError("O fator do IQR tem de ser positivo (o habitual é 1,5).")
    if not pd.api.types.is_numeric_dtype(frame[column]) or pd.api.types.is_bool_dtype(
        frame[column]
    ):
        raise DataError(
            f"A coluna '{column}' não é numérica: converte-a primeiro (operação convert_type)."
        )
    bounds = iqr_bounds(frame[column], factor)
    params = {"column": column, "method": method, "factor": factor}
    if bounds is None or bounds.n_outliers == 0:
        description = f"A coluna {column} não tem valores extremos (regra do IQR)."
        return CleaningResult(
            frame, cleaning_step("treat_outliers", description, frame, frame, params)
        )

    limits = f"[{format_number(bounds.lower)}; {format_number(bounds.upper)}]"
    n_outliers = bounds.n_outliers
    if method == "clip":
        result = frame.assign(**{column: frame[column].clip(bounds.lower, bounds.upper)})
        values = format_quantity(n_outliers, "valor extremo limitado", "valores extremos limitados")
        description = f"{column}: {values} ao intervalo {limits}."
        step = cleaning_step("treat_outliers", description, frame, result, params, changed=[column])
    else:
        inside = frame[column].isna() | frame[column].between(bounds.lower, bounds.upper)
        result = frame.loc[inside]
        rows = format_quantity(n_outliers, "linha", "linhas")
        verb = "removida" if n_outliers == 1 else "removidas"
        description = f"{column}: {verb} {rows} com valores fora de {limits}."
        step = cleaning_step("treat_outliers", description, frame, result, params)
    return CleaningResult(result, step)
