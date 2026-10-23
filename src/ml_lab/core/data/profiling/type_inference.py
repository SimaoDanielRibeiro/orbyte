"""Inferência do tipo de cada coluna: numérica, categórica, booleana, data, texto ou ID.

O pandas diz como uma coluna está *guardada* (``int64``, ``float64``, ``str``...); a
inferência diz o que ela *significa*, e explica porquê. Alguns exemplos do Titanic:

- ``Survived`` é ``int64`` mas só vale 0 ou 1: é booleana;
- ``PassengerId`` são inteiros todos diferentes e seguidos: é um identificador;
- ``Name`` é texto com um valor diferente por linha: é texto livre;
- ``Embarked`` tem só três valores: é categórica.

Também deteta valores guardados como texto que são números (``"1,5"``), datas
(``"25/12/2024"``) ou booleanos (``"sim"``/``"não"``), para a limpeza os converter.
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

import pandas as pd

from ml_lab.core.data.schema import ColumnKind, ColumnRole, ColumnSchema, TableSchema
from ml_lab.formatting import format_count, format_percent

CATEGORICAL_MAX_UNIQUE: Final = 50
"""Uma coluna de texto com mais valores distintos do que isto não é tratada como categórica."""

CATEGORICAL_MAX_RATIO: Final = 0.5
"""Nem com mais valores distintos do que esta proporção das linhas."""

FEW_VALUES: Final = 10
"""Inteiros com até este número de valores distintos podem ser categorias (ou classes)."""

PARSE_THRESHOLD: Final = 0.95
"""Proporção mínima de valores convertíveis para um texto ser tratado como número ou data."""

_SAMPLE_SIZE: Final = 2_000
"""Valores usados para testar conversões de texto: chegam para decidir e são rápidos."""

BOOLEAN_WORDS: Final[Mapping[str, bool]] = {
    "true": True,
    "false": False,
    "yes": True,
    "no": False,
    "sim": True,
    "não": False,
    "nao": False,
    "s": True,
    "n": False,
    "y": True,
    "t": True,
    "f": False,
    "verdadeiro": True,
    "falso": False,
    "v": True,
}
"""Palavras reconhecidas como verdadeiro/falso (em minúsculas)."""

DATE_FORMATS: Final = (
    "ISO8601",
    "%d/%m/%Y",
    "%d-%m-%Y",
    "%d.%m.%Y",
    "%d/%m/%Y %H:%M",
    "%d/%m/%Y %H:%M:%S",
    "%m/%d/%Y",
    "%Y/%m/%d",
)
"""Formatos de data testados por ordem. Dia antes do mês primeiro, como em Portugal."""

_ID_WORDS: Final = frozenset({"id", "uuid", "guid", "index", "idx", "key", "code", "codigo"})
_WORDS: Final = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+")


@dataclass(frozen=True, slots=True)
class TypeInference:
    """O tipo inferido de uma coluna e a razão.

    Attributes:
        column: Nome da coluna.
        kind: O que a coluna significa.
        reason: Porque se chegou a este tipo, em português.
        stored_as_text: Se os valores estão guardados como texto, mas são números, datas
            ou booleanos (a limpeza deve convertê-los).
        text_format: Como converter o texto: o separador decimal (``"."`` ou ``","``) dos
            números ou o formato das datas (por exemplo ``"%d/%m/%Y"``).
    """

    column: str
    kind: ColumnKind
    reason: str
    stored_as_text: bool = False
    text_format: str | None = None


def infer_column(series: pd.Series) -> TypeInference:
    """Infere o tipo de uma coluna a partir dos seus valores (os nulos são ignorados)."""
    name = str(series.name)
    values = series.dropna()
    if values.empty:
        kind = ColumnKind.NUMERIC if pd.api.types.is_numeric_dtype(series) else ColumnKind.TEXT
        return TypeInference(name, kind, "A coluna não tem valores, só nulos.")
    # A ordem importa: no pandas, bool conta como numérico e category conta como texto.
    if pd.api.types.is_bool_dtype(series):
        return TypeInference(name, ColumnKind.BOOLEAN, "Valores verdadeiro/falso.")
    if pd.api.types.is_datetime64_any_dtype(series):
        return TypeInference(name, ColumnKind.DATETIME, "Valores do tipo data.")
    if isinstance(series.dtype, pd.CategoricalDtype):
        return TypeInference(name, ColumnKind.CATEGORICAL, "Coluna do tipo category do pandas.")
    if pd.api.types.is_numeric_dtype(series):
        return _infer_numeric(name, values)
    if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
        return _infer_text(name, values.astype(str).str.strip())
    return TypeInference(
        name, ColumnKind.TEXT, f"Tipo {series.dtype} sem tratamento próprio: fica como texto."
    )


def infer_types(frame: pd.DataFrame) -> dict[str, TypeInference]:
    """Infere o tipo de todas as colunas, pela ordem da tabela."""
    return {str(name): infer_column(frame[name]) for name in frame.columns}


def schema_from(inferences: Mapping[str, TypeInference], target: str | None = None) -> TableSchema:
    """Monta o esquema a partir dos tipos inferidos.

    Texto livre e identificadores ficam com o papel ``IGNORED``: não entram no modelo.
    """
    columns = tuple(
        ColumnSchema(
            name,
            inference.kind,
            ColumnRole.FEATURE if inference.kind.is_usable_by_models else ColumnRole.IGNORED,
        )
        for name, inference in inferences.items()
    )
    return TableSchema(columns).with_target(target)


def infer_schema(frame: pd.DataFrame, target: str | None = None) -> TableSchema:
    """Infere o esquema de ``frame``, com ``target`` como alvo (se for dado)."""
    return schema_from(infer_types(frame), target)


def looks_like_classes(series: pd.Series, inference: TypeInference | None = None) -> bool:
    """Se uma coluna, usada como alvo, tem cara de classes.

    É o caso do texto, dos booleanos e dos inteiros com poucos valores distintos
    (regra 2 da deteção automática do tipo de tarefa, em docs/architecture.md).
    """
    inference = inference or infer_column(series)
    if inference.kind in (ColumnKind.BOOLEAN, ColumnKind.CATEGORICAL):
        return True
    if inference.kind is not ColumnKind.NUMERIC or inference.stored_as_text:
        return False
    values = series.dropna()
    integral = bool((values == values.round()).all())
    return integral and int(values.nunique()) <= FEW_VALUES


def looks_like_an_identifier_name(name: str) -> bool:
    """Se o nome sugere um identificador: ``PassengerId``, ``user_id``, ``ID``...

    ``idade`` não conta: as palavras são separadas e comparadas inteiras.
    """
    return any(word.lower() in _ID_WORDS for word in _WORDS.findall(name))


def _infer_numeric(name: str, values: pd.Series) -> TypeInference:
    n_unique = int(values.nunique())
    if n_unique == 2 and set(values.unique().tolist()) <= {0, 1}:
        return TypeInference(name, ColumnKind.BOOLEAN, "Só tem os valores 0 e 1.")
    integral = bool((values == values.round()).all())
    if pd.api.types.is_integer_dtype(values) and n_unique == len(values) > 1:
        consecutive = int(values.max()) - int(values.min()) + 1 == len(values)
        if consecutive or looks_like_an_identifier_name(name):
            why = "seguidos" if consecutive else "com nome de identificador"
            return TypeInference(
                name,
                ColumnKind.IDENTIFIER,
                f"Inteiros todos diferentes e {why}: parece um identificador de cada linha.",
            )
    if integral and n_unique <= FEW_VALUES:
        return TypeInference(
            name,
            ColumnKind.NUMERIC,
            f"Números inteiros com poucos valores distintos ({n_unique}): podem ser categorias.",
        )
    return TypeInference(name, ColumnKind.NUMERIC, "Valores numéricos.")


def _infer_text(name: str, text: pd.Series) -> TypeInference:
    lowered = set(text.str.lower().unique().tolist())
    if len(lowered) == 2 and lowered <= set(BOOLEAN_WORDS):
        return TypeInference(
            name,
            ColumnKind.BOOLEAN,
            f"Texto com valores de verdadeiro/falso ({', '.join(sorted(lowered))}).",
            stored_as_text=True,
        )
    sample = text.sample(_SAMPLE_SIZE, random_state=0) if len(text) > _SAMPLE_SIZE else text
    for decimal in (".", ","):
        candidates = sample if decimal == "." else sample.str.replace(",", ".", regex=False)
        ratio = float(pd.to_numeric(candidates, errors="coerce").notna().mean())
        if ratio >= PARSE_THRESHOLD:
            return TypeInference(
                name,
                ColumnKind.NUMERIC,
                f"{format_percent(ratio)} dos valores são números guardados como texto"
                + (" (com vírgula decimal)." if decimal == "," else "."),
                stored_as_text=True,
                text_format=decimal,
            )
    best_format, best_ratio = _best_date_format(sample)
    if best_format is not None:
        return TypeInference(
            name,
            ColumnKind.DATETIME,
            f"{format_percent(best_ratio)} dos valores são datas guardadas como texto.",
            stored_as_text=True,
            text_format=best_format,
        )

    n_values = len(text)
    n_unique = int(text.nunique())
    if n_unique <= min(CATEGORICAL_MAX_UNIQUE, max(2, CATEGORICAL_MAX_RATIO * n_values)):
        return TypeInference(
            name, ColumnKind.CATEGORICAL, f"Texto com poucos valores distintos ({n_unique})."
        )
    with_spaces = float(text.str.contains(" ", regex=False).mean())
    if n_unique == n_values and with_spaces < 0.1:
        return TypeInference(
            name,
            ColumnKind.IDENTIFIER,
            "Um valor diferente em cada linha, sem espaços: parece um código ou identificador.",
        )
    return TypeInference(
        name,
        ColumnKind.TEXT,
        f"Texto livre: {format_count(n_unique)} valores distintos em "
        f"{format_count(n_values)} linhas, demasiados para categorias.",
    )


def _best_date_format(sample: pd.Series) -> tuple[str | None, float]:
    best: tuple[str | None, float] = (None, 0.0)
    for date_format in DATE_FORMATS:
        parsed = pd.to_datetime(sample, format=date_format, errors="coerce")
        ratio = float(parsed.notna().mean())
        if ratio >= PARSE_THRESHOLD and ratio > best[1]:
            best = (date_format, ratio)
    return best
