"""Regras esperadas por coluna: o que uma coluna tem de cumprir para os dados servirem.

As regras descrevem o que se espera (a coluna existe, é numérica, não tem nulos,
os valores estão num intervalo) e a verificação diz o que falhou. Servem para
confirmar que um dataset de exemplo não foi alterado e, mais tarde, que dados
novos têm as colunas com que o modelo foi treinado.

Exemplo:
    >>> import pandas as pd
    >>> frame = pd.DataFrame({"age": [22.0, -1.0]})
    >>> [n.code for n in check_rules(frame, [ColumnRule("age", "numeric", minimum=0)])]
    ['below_minimum']
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from ml_lab.core.foundation.notices import Notice
from ml_lab.formatting import format_count, format_number, format_quantity

type DtypeFamily = Literal["any", "numeric", "integer", "text", "boolean", "datetime"]

_FAMILY_LABELS = {
    "numeric": "numérica",
    "integer": "de números inteiros",
    "text": "de texto",
    "boolean": "booleana",
    "datetime": "de datas",
}


@dataclass(frozen=True, slots=True)
class ColumnRule:
    """O que uma coluna tem de cumprir.

    Attributes:
        name: Nome da coluna.
        family: Família do tipo de dados: ``"numeric"``, ``"integer"`` (números sem parte
            decimal, mesmo que guardados como decimais por causa dos nulos), ``"text"``,
            ``"boolean"``, ``"datetime"`` ou ``"any"``.
        required: Se a coluna tem de existir.
        allow_missing: Se pode ter valores em falta.
        minimum: Menor valor aceite, para colunas numéricas.
        maximum: Maior valor aceite, para colunas numéricas.
        allowed: Valores aceites (comparados como texto), para colunas categóricas.
    """

    name: str
    family: DtypeFamily = "any"
    required: bool = True
    allow_missing: bool = True
    minimum: float | None = None
    maximum: float | None = None
    allowed: tuple[str, ...] | None = None


def check_rules(frame: pd.DataFrame, rules: Sequence[ColumnRule]) -> tuple[Notice, ...]:
    """Verifica ``frame`` contra as regras e devolve um erro por cada regra que falha."""
    notices: list[Notice] = []
    for rule in rules:
        if rule.name not in frame.columns:
            if rule.required:
                notices.append(
                    Notice.error("missing_column", f"Falta a coluna '{rule.name}'.", rule.name)
                )
            continue
        notices.extend(_check_column(frame[rule.name], rule))
    return tuple(notices)


def _check_column(column: pd.Series, rule: ColumnRule) -> list[Notice]:
    name = rule.name
    if not _has_family(column, rule.family):
        return [
            Notice.error(
                "wrong_type",
                f"A coluna '{name}' devia ser {_FAMILY_LABELS[rule.family]}, mas é do tipo "
                f"{column.dtype}.",
                name,
            )
        ]
    notices: list[Notice] = []
    n_missing = int(column.isna().sum())
    if n_missing and not rule.allow_missing:
        notices.append(
            Notice.error(
                "unexpected_missing",
                f"A coluna '{name}' não devia ter valores em falta e tem "
                f"{format_count(n_missing)}.",
                name,
            )
        )
    if rule.minimum is not None and (below := int((column < rule.minimum).sum())):
        notices.append(
            Notice.error(
                "below_minimum",
                f"A coluna '{name}' tem {format_quantity(below, 'valor', 'valores')} abaixo de "
                f"{format_number(rule.minimum)}.",
                name,
            )
        )
    if rule.maximum is not None and (above := int((column > rule.maximum).sum())):
        notices.append(
            Notice.error(
                "above_maximum",
                f"A coluna '{name}' tem {format_quantity(above, 'valor', 'valores')} acima de "
                f"{format_number(rule.maximum)}.",
                name,
            )
        )
    if rule.allowed is not None:
        unexpected = sorted(set(column.dropna().astype(str)) - set(rule.allowed))
        if unexpected:
            notices.append(
                Notice.error(
                    "unexpected_values",
                    f"A coluna '{name}' tem valores inesperados: {', '.join(unexpected[:5])}.",
                    name,
                )
            )
    return notices


def _has_family(column: pd.Series, family: DtypeFamily) -> bool:
    if family == "any":
        return True
    if family == "boolean":
        return pd.api.types.is_bool_dtype(column)
    if family == "datetime":
        return pd.api.types.is_datetime64_any_dtype(column)
    if family == "text":
        return pd.api.types.is_string_dtype(column) or pd.api.types.is_object_dtype(column)
    numeric = pd.api.types.is_numeric_dtype(column) and not pd.api.types.is_bool_dtype(column)
    if family == "numeric" or not numeric:
        return numeric
    values = column.dropna().to_numpy(dtype=np.float64)
    return bool(np.all(values == np.round(values)))
