"""Sugestões de limpeza a partir do perfil: o sistema propõe, o utilizador decide.

A ordem das sugestões é a ordem em que fazem sentido:

1. **Duplicados primeiro**, com todas as colunas. Depois de remover um
   identificador, passageiros diferentes com os mesmos dados pareceriam duplicados.
2. **Conversões de tipo**, para os números, datas e booleanos guardados como texto.
3. **Remover colunas** que não ajudam: identificadores, texto livre, colunas vazias,
   constantes ou com mais de metade dos valores em falta. No lugar do alvo, que nunca se
   remove, os erros grosseiros de um alvo numérico (as linhas com um preço impossível).
"""

from dataclasses import dataclass
from typing import Final

from ml_lab.core.data.cleaning.base import CleaningStep
from ml_lab.core.data.cleaning.outliers import GROSS_FACTOR
from ml_lab.core.data.profiling.column_profile import MISSING_SEVERE, ColumnProfile
from ml_lab.core.data.profiling.dataset_profile import DatasetProfile
from ml_lab.core.data.profiling.type_inference import repeated_categories
from ml_lab.core.data.schema import ColumnKind
from ml_lab.formatting import format_count, format_number, format_percent, format_quantity

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
        reason: Porque é sugerido, em inglês.
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
        duplicates = format_quantity(profile.n_duplicate_rows, "duplicate row", "duplicate rows")
        suggestions.append(
            CleaningSuggestion(CleaningStep("drop_duplicates"), f"There are {duplicates}.")
        )
    for column in profile.columns:
        conversion = _conversion(column)
        if conversion is not None:
            suggestions.append(conversion)
    for column in profile.columns:
        if column.name == target:
            gross = _gross_target_errors(column)
            if gross is not None:
                suggestions.append(gross)
            continue
        reason = _drop_reason(column)
        if reason is not None:
            step = CleaningStep("drop_columns", {"columns": [column.name]})
            suggestions.append(CleaningSuggestion(step, reason, column.name))
    return tuple(suggestions)


def _gross_target_errors(column: ColumnProfile) -> CleaningSuggestion | None:
    """Os erros num alvo numérico: poucos valores muito para lá dos outros (ADR 0051).

    Um valor extremo pode ser real (por isso a regra do IQR só avisa); dez IQR para lá dos
    quartis já não é extremo, é quase sempre um erro (um preço de 99 999 999). No alvo de uma
    regressão, um só desses valores domina o erro de todos os modelos. Sugere-se remover as
    linhas, e não limitar os valores: limitar o alvo era inventar a resposta.
    """
    summary = column.numeric
    if column.kind is not ColumnKind.NUMERIC or summary is None:
        return None
    bounds = summary.gross_outliers
    if bounds is None or not bounds.n_outliers or bounds.upper <= bounds.lower:
        return None
    if bounds.n_outliers > GROSS_SHARE * column.n_present:
        return None  # muitos: é a forma da distribuição, não erros soltos
    values = format_quantity(bounds.n_outliers, "value", "values")
    extreme = summary.maximum if bounds.n_above else summary.minimum
    reach = format_number(extreme, 0 if float(extreme).is_integer() else 2)
    return CleaningSuggestion(
        CleaningStep(
            "treat_outliers", {"column": column.name, "method": "remove", "factor": GROSS_FACTOR}
        ),
        f"{column.name} has {values} far beyond the rest (up to {reach}): they look like typing "
        "errors, and they would dominate the error of every model.",
        column.name,
    )


GROSS_SHARE: Final = 0.01
"""Até que parte das linhas os valores muito afastados ainda são erros soltos (e não a forma)."""


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
        f"{column.name} is stored as text, but it's {column.kind.label.lower()}.",
        column.name,
    )


def _drop_reason(column: ColumnProfile) -> str | None:
    name = column.name
    if column.n_present == 0:
        return f"{name} has no values at all."
    if column.n_unique == 1:
        return f"{name} always has the same value."
    if column.kind is ColumnKind.IDENTIFIER:
        return f"{name} is an identifier: a different value in each row."
    if column.kind is ColumnKind.TEXT and repeated_categories(column.n_unique, column.n_present):
        return f"{name} has {format_count(column.n_unique)} categories, too many to encode."
    if column.kind is ColumnKind.TEXT:
        return f"{name} is free text, which models don't use directly."
    if column.missing_ratio >= MISSING_SEVERE:
        return f"{name} has {format_percent(column.missing_ratio)} of its values missing."
    return None
