"""Números escritos à portuguesa: vírgula decimal e espaço a separar os milhares.

As mensagens didáticas e a interface estão em português, por isso ``0.199`` aparece
como ``19,9%`` e ``20640`` como ``20 640``. Usa só a biblioteca padrão (o módulo
``locale`` depende do sistema operativo) e pode ser usado por todas as camadas,
como ``ml_lab.config`` e ``ml_lab.logging``.

Exemplo:
    >>> format_percent(0.1987)
    '19,9%'
    >>> format_count(20640)
    '20 640'
    >>> format_number(3.14159, 2)
    '3,14'
"""

import math

_GROUPING_THRESHOLD = 10_000
"""Em português, só se separam os milhares a partir de cinco algarismos (1234, mas 12 345)."""

_SUPERSCRIPTS = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")

_SCIENTIFIC_FROM = 6
"""Ordem de grandeza a partir da qual ``format_significant`` passa a potência de 10."""


def format_number(value: float, decimals: int = 2) -> str:
    """Escreve ``value`` com ``decimals`` casas decimais e vírgula decimal.

    Os valores não finitos aparecem como ``"—"`` (em falta) ou ``"∞"``.
    """
    if math.isnan(value):
        return "—"
    if math.isinf(value):
        return "∞" if value > 0 else "-∞"
    grouping = "," if abs(value) >= _GROUPING_THRESHOLD else ""
    text = f"{value:{grouping}.{decimals}f}"
    # Troca em dois passos, para a vírgula dos milhares não se confundir com a decimal.
    return text.replace(",", " ").replace(".", ",")


def format_count(value: int) -> str:
    """Escreve um número inteiro, com espaço a separar os milhares (``20 640``)."""
    return format_number(value, 0)


def format_percent(ratio: float, decimals: int = 1) -> str:
    """Escreve uma proporção como percentagem: ``0.1987`` dá ``19,9%``."""
    return f"{format_number(ratio * 100, decimals)}%"


def format_scientific(value: float, digits: int = 3) -> str:
    """Escreve ``value`` como potência de 10: para números muito pequenos ou muito grandes.

    >>> format_scientific(0.000001)
    '1 × 10⁻⁶'
    >>> format_scientific(0.000025)
    '2,5 × 10⁻⁵'
    """
    if value == 0 or not math.isfinite(value):
        return format_number(value, 0)
    mantissa, exponent = f"{value:.{digits - 1}e}".split("e")
    mantissa = mantissa.rstrip("0").rstrip(".").replace(".", ",")
    return f"{mantissa} × 10{str(int(exponent)).translate(_SUPERSCRIPTS)}"


def format_significant(value: float, digits: int = 4) -> str:
    """Escreve ``value`` com ``digits`` algarismos significativos (uma loss, por exemplo).

    Os números muito grandes (a partir de um milhão) ou muito pequenos ficam em potência
    de 10, para não encherem o ecrã de algarismos.

    >>> format_significant(0.320912)
    '0,3209'
    >>> format_significant(4661704537.05)
    '4,662 × 10⁹'
    """
    if value == 0 or not math.isfinite(value):
        return format_number(value, 0)
    magnitude = math.floor(math.log10(abs(value)))
    if magnitude >= _SCIENTIFIC_FROM or magnitude < -3:
        return format_scientific(value, digits)
    return format_number(value, max(0, digits - 1 - magnitude))


def format_quantity(count: int, singular: str, plural: str) -> str:
    """Escreve uma quantidade com o nome no singular ou no plural.

    >>> format_quantity(1, "linha duplicada", "linhas duplicadas")
    '1 linha duplicada'
    >>> format_quantity(12, "linha duplicada", "linhas duplicadas")
    '12 linhas duplicadas'
    """
    return f"{format_count(count)} {singular if count == 1 else plural}"
