"""Correção de tipos: texto para número, data ou booleano.

Um preço lido como texto (``"1,5"``) não entra num modelo. A conversão nunca
inventa valores: o que não se consegue converter fica em falta, e a linhagem diz
quantos foram e dá exemplos, para decidires se a conversão faz sentido.

Também converte texto para ``category``: o pandas guarda cada valor distinto uma
só vez, o que poupa memória e marca a coluna como categórica.
"""

from collections.abc import Mapping
from typing import Any, Final, Literal

import pandas as pd

from ml_lab.core.data.cleaning.base import (
    CleaningResult,
    cleaning_operation,
    cleaning_step,
    require_columns,
)
from ml_lab.core.data.profiling.type_inference import BOOLEAN_WORDS
from ml_lab.core.foundation.errors import InvalidParameterError
from ml_lab.core.foundation.notices import Notice
from ml_lab.formatting import format_count, format_quantity

type TargetType = Literal["numeric", "datetime", "boolean", "category", "text"]

TARGET_TYPES: Final[tuple[TargetType, ...]] = ("numeric", "datetime", "boolean", "category", "text")

_LABELS: Final = {
    "numeric": "número",
    "datetime": "data",
    "boolean": "booleano",
    "category": "categoria (category)",
    "text": "texto",
}
_EXAMPLES: Final = 3


def _convert_new_rows(
    frame: pd.DataFrame, params: Mapping[str, Any], _learned: Mapping[str, Any]
) -> pd.DataFrame:
    """Numa linha nova, converte a coluna da mesma maneira (a conversão não aprende nada)."""
    if params["column"] not in frame:
        return frame
    return convert_type(frame, **params).frame


@cleaning_operation(
    "convert_type",
    title="Converter o tipo de uma coluna",
    description=(
        "Converte uma coluna para número, data, booleano, categoria ou texto. Os valores "
        "que não se conseguem converter ficam em falta, e a linhagem diz quantos foram."
    ),
    replay=_convert_new_rows,
)
def convert_type(
    frame: pd.DataFrame,
    *,
    column: str,
    to: TargetType,
    decimal: Literal[".", ","] = ".",
    date_format: str | None = None,
) -> CleaningResult:
    """Converte ``column`` para o tipo ``to``.

    Args:
        frame: A tabela; não é alterada.
        column: Coluna a converter.
        to: ``"numeric"``, ``"datetime"``, ``"boolean"``, ``"category"`` ou ``"text"``.
        decimal: Separador decimal dos números escritos como texto.
        date_format: Formato das datas (``"%d/%m/%Y"``...); por omissão, ISO 8601.

    Returns:
        A tabela com a coluna convertida e o passo de linhagem.
    """
    require_columns(frame, [column])
    if to not in TARGET_TYPES:
        raise InvalidParameterError(
            f"Tipo '{to}' desconhecido. Usa um destes: {', '.join(TARGET_TYPES)}."
        )
    if decimal not in (".", ","):
        raise InvalidParameterError("O separador decimal tem de ser '.' ou ','.")
    original = frame[column]
    converted = _convert(original, to, decimal, date_format)
    failed = converted.isna() & original.notna()
    n_failed = int(failed.sum())
    result = frame.assign(**{column: converted})

    description = f"Coluna {column} convertida para {_LABELS[to]}."
    notices: tuple[Notice, ...] = ()
    if n_failed:
        examples = ", ".join(repr(str(v)) for v in original[failed].unique()[:_EXAMPLES])
        failures = (
            "1 valor não se converteu e ficou em falta"
            if n_failed == 1
            else f"{format_count(n_failed)} valores não se converteram e ficaram em falta"
        )
        description += f" {failures} (por exemplo {examples})."
        notices = (
            Notice.warning(
                "conversion_failures",
                f"Na conversão de {column}, ficaram em falta "
                f"{format_quantity(n_failed, 'valor', 'valores')}: "
                "confirma se o tipo escolhido é o certo.",
                column,
            ),
        )
    params: dict[str, object] = {"column": column, "to": to}
    if to == "numeric":
        params["decimal"] = decimal
    if to == "datetime":
        params["date_format"] = date_format
    step = cleaning_step("convert_type", description, frame, result, params, changed=[column])
    return CleaningResult(result, step, notices)


def _convert(values: pd.Series, to: TargetType, decimal: str, date_format: str | None) -> pd.Series:
    if to == "category":
        return values.astype("category")
    if to == "text":
        return values.astype("str").where(values.notna())
    if to == "boolean":
        return _to_boolean(values)
    text = _as_text(values)
    if to == "numeric":
        if decimal == ",":
            text = text.str.replace(",", ".", regex=False)
        return pd.to_numeric(text, errors="coerce")
    return pd.to_datetime(text, format=date_format or "ISO8601", errors="coerce")


def _as_text(values: pd.Series) -> pd.Series:
    """Os valores como texto sem espaços nas pontas (os nulos continuam nulos)."""
    if pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values):
        return values.astype("str").where(values.notna())
    return values.astype("str").str.strip().where(values.notna())


def _to_boolean(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values
    words = {**BOOLEAN_WORDS, "1": True, "0": False, "1.0": True, "0.0": False}
    mapped = _as_text(values).str.lower().map(words)
    # O tipo "boolean" do pandas aceita nulos; o bool do NumPy não.
    return mapped.astype("boolean")
