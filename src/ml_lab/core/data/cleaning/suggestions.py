"""Sugestões de limpeza a partir do perfil: o sistema propõe, o utilizador decide.

A ordem das sugestões é a ordem em que fazem sentido:

1. **Duplicados primeiro**, com todas as colunas. Depois de remover um
   identificador, passageiros diferentes com os mesmos dados pareceriam duplicados.
2. **Conversões de tipo**, para os números, datas e booleanos guardados como texto.
3. **Remover colunas** que não ajudam: identificadores, texto livre, colunas vazias,
   constantes ou com mais de metade dos valores em falta.
"""

from dataclasses import dataclass

from ml_lab.core.data.cleaning.base import CleaningStep
from ml_lab.core.data.profiling.column_profile import MISSING_SEVERE, ColumnProfile
from ml_lab.core.data.profiling.dataset_profile import DatasetProfile
from ml_lab.core.data.schema import ColumnKind
from ml_lab.formatting import format_percent, format_quantity

_CONVERSIONS = {
    ColumnKind.NUMERIC: "numeric",
    ColumnKind.DATETIME: "datetime",
    ColumnKind.BOOLEAN: "boolean",
}


@dataclass(frozen=True, slots=True)
class CleaningSuggestion:
    """Um passo de limpeza sugerido e a razão.

    Attributes:
        step: O passo, pronto a juntar à receita.
        reason: Porque é sugerido, em português.
        column: A coluna a que diz respeito, se for o caso.
    """

    step: CleaningStep
    reason: str
    column: str | None = None


def suggest_cleaning(
    profile: DatasetProfile, target: str | None = None
) -> tuple[CleaningSuggestion, ...]:
    """Sugere uma receita de limpeza para o dataset descrito por ``profile``.

    Args:
        profile: O perfil do dataset.
        target: A coluna-alvo, que nunca é sugerida para remoção.
    """
    suggestions: list[CleaningSuggestion] = []
    if profile.n_duplicate_rows:
        duplicates = format_quantity(
            profile.n_duplicate_rows, "linha duplicada", "linhas duplicadas"
        )
        suggestions.append(CleaningSuggestion(CleaningStep("drop_duplicates"), f"Há {duplicates}."))
    for column in profile.columns:
        conversion = _conversion(column)
        if conversion is not None:
            suggestions.append(conversion)
    for column in profile.columns:
        if column.name == target:
            continue
        reason = _drop_reason(column)
        if reason is not None:
            step = CleaningStep("drop_columns", {"columns": [column.name]})
            suggestions.append(CleaningSuggestion(step, reason, column.name))
    return tuple(suggestions)


def _conversion(column: ColumnProfile) -> CleaningSuggestion | None:
    to = _CONVERSIONS.get(column.kind)
    if not column.stored_as_text or to is None:
        return None
    params: dict[str, object] = {"column": column.name, "to": to}
    if to == "numeric" and column.text_format == ",":
        params["decimal"] = ","
    if to == "datetime" and column.text_format not in (None, "ISO8601"):
        params["date_format"] = column.text_format
    return CleaningSuggestion(
        CleaningStep("convert_type", params),
        f"{column.name} está guardada como texto, mas é do tipo {column.kind.label.lower()}.",
        column.name,
    )


def _drop_reason(column: ColumnProfile) -> str | None:
    name = column.name
    if column.n_present == 0:
        return f"{name} não tem nenhum valor."
    if column.n_unique == 1:
        return f"{name} tem sempre o mesmo valor."
    if column.kind is ColumnKind.IDENTIFIER:
        return f"{name} é um identificador: um valor diferente em cada linha."
    if column.kind is ColumnKind.TEXT:
        return f"{name} é texto livre, que os modelos não usam diretamente."
    if column.missing_ratio >= MISSING_SEVERE:
        return f"{name} tem {format_percent(column.missing_ratio)} dos valores em falta."
    return None
