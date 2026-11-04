"""Números escritos em inglês: ponto decimal e vírgula a separar os milhares.

O site está em inglês (ADR 0043), por isso ``0.199`` aparece como ``19.9%`` e ``20640`` como
``20,640``. As regras são as mesmas do ``format.ts`` da interface: um número escrito pelo backend
num aviso e o mesmo número numa tabela da interface aparecem iguais. Usa só a biblioteca padrão
(o módulo ``locale`` depende do sistema operativo) e pode ser usado por todas as camadas, como
``ml_lab.config`` e ``ml_lab.logging``.

Exemplo:
    >>> format_percent(0.1987)
    '19.9%'
    >>> format_count(20640)
    '20,640'
    >>> format_number(3.14159, 2)
    '3.14'
"""

import math
from collections.abc import Sequence

_GROUPING_THRESHOLD = 1_000
"""Em inglês, os milhares separam-se a partir de quatro algarismos (1,234 e 12,345)."""

_SUPERSCRIPTS = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")

_SCIENTIFIC_FROM = 6
"""Ordem de grandeza a partir da qual ``format_significant`` passa a potência de 10."""


def format_number(value: float, decimals: int = 2) -> str:
    """Escreve ``value`` com ``decimals`` casas decimais e ponto decimal.

    Os valores não finitos aparecem como ``"—"`` (em falta) ou ``"∞"``.
    """
    if math.isnan(value):
        return "—"
    if math.isinf(value):
        return "∞" if value > 0 else "-∞"
    grouping = "," if abs(value) >= _GROUPING_THRESHOLD else ""
    return f"{value:{grouping}.{decimals}f}"


def format_count(value: int) -> str:
    """Escreve um número inteiro, com vírgula a separar os milhares (``20,640``)."""
    return format_number(value, 0)


def format_percent(ratio: float, decimals: int = 1) -> str:
    """Escreve uma proporção como percentagem: ``0.1987`` dá ``19.9%``."""
    return f"{format_number(ratio * 100, decimals)}%"


def format_scientific(value: float, digits: int = 3) -> str:
    """Escreve ``value`` como potência de 10: para números muito pequenos ou muito grandes.

    >>> format_scientific(0.000001)
    '1 × 10⁻⁶'
    >>> format_scientific(0.000025)
    '2.5 × 10⁻⁵'
    """
    if value == 0 or not math.isfinite(value):
        return format_number(value, 0)
    mantissa, exponent = f"{value:.{digits - 1}e}".split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    return f"{mantissa} × 10{str(int(exponent)).translate(_SUPERSCRIPTS)}"


def format_significant(value: float, digits: int = 4) -> str:
    """Escreve ``value`` com ``digits`` algarismos significativos (uma loss, por exemplo).

    Os números muito grandes (a partir de um milhão) ou muito pequenos ficam em potência
    de 10, para não encherem o ecrã de algarismos.

    >>> format_significant(0.320912)
    '0.3209'
    >>> format_significant(4661704537.05)
    '4.662 × 10⁹'
    """
    if value == 0 or not math.isfinite(value):
        return format_number(value, 0)
    magnitude = math.floor(math.log10(abs(value)))
    if magnitude >= _SCIENTIFIC_FROM or magnitude < -3:
        return format_scientific(value, digits)
    return format_number(value, max(0, digits - 1 - magnitude))


def format_ordinal(place: int) -> str:
    """Escreve uma posição em inglês, como o ``formatOrdinal`` da interface.

    >>> [format_ordinal(place) for place in (1, 2, 3, 4, 11, 22)]
    ['1st', '2nd', '3rd', '4th', '11th', '22nd']
    """
    tens, units = place % 100, place % 10
    # 11, 12 e 13 são as exceções: "11th", e não "11st".
    suffix = "th" if 11 <= tens <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(units, "th")
    return f"{format_count(place)}{suffix}"


def format_quantity(count: int, singular: str, plural: str) -> str:
    """Escreve uma quantidade com o nome no singular ou no plural.

    >>> format_quantity(1, "duplicate row", "duplicate rows")
    '1 duplicate row'
    >>> format_quantity(12, "duplicate row", "duplicate rows")
    '12 duplicate rows'
    """
    return f"{format_count(count)} {singular if count == 1 else plural}"


def format_value(value: object, digits: int = 3) -> str:
    """Escreve um valor qualquer (o de um hiperparâmetro, por exemplo).

    Os números decimais ficam com ``digits`` algarismos significativos, sem zeros à direita;
    os muito pequenos, em potência de 10.

    >>> format_value(0.0128)
    '0.0128'
    >>> format_value(0.1)
    '0.1'
    >>> format_value(0.00001)
    '1 × 10⁻⁵'
    >>> format_value(True)
    'yes'
    """
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return format_count(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return format_number(value)
        text = f"{value:.{digits}g}"
        return format_scientific(value, digits) if "e" in text else text
    return str(value)


def format_list(items: Sequence[str]) -> str:
    """Escreve uma lista como se diz em inglês, com "and" antes do último.

    >>> format_list(["Sex"])
    'Sex'
    >>> format_list(["Sex", "Embarked", "Pclass"])
    'Sex, Embarked and Pclass'
    """
    if len(items) <= 1:
        return "".join(items)
    return f"{', '.join(items[:-1])} and {items[-1]}"
