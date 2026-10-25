"""Erros de validação do Pydantic explicados em português.

O Pydantic descreve cada problema com um tipo estável (``greater_than``,
``literal_error``...) e uma mensagem em inglês. Aqui cada tipo vira uma frase em
português que diz o campo, o valor recebido e o que era esperado.
"""

from collections.abc import Callable, Iterable, Mapping
from typing import Any, Final

from pydantic import ValidationError

from ml_lab.formatting import format_number


def _number(value: object) -> str:
    return (
        format_number(value, 4).rstrip("0").rstrip(",") if isinstance(value, float) else str(value)
    )


_TEMPLATES: Final[Mapping[str, Callable[[str, Mapping[str, Any]], str]]] = {
    "greater_than": lambda field, ctx: f"{field} tem de ser maior do que {_number(ctx['gt'])}.",
    "greater_than_equal": lambda field, ctx: f"{field} tem de ser pelo menos {_number(ctx['ge'])}.",
    "less_than": lambda field, ctx: f"{field} tem de ser menor do que {_number(ctx['lt'])}.",
    "less_than_equal": lambda field, ctx: f"{field} não pode passar de {_number(ctx['le'])}.",
    "literal_error": lambda field, ctx: f"{field} tem de ser um destes: {ctx['expected']}.",
    "enum": lambda field, ctx: f"{field} tem de ser um destes: {ctx['expected']}.",
    "missing": lambda field, ctx: f"Falta o valor de {field}.",
    "extra_forbidden": lambda field, ctx: f"{field} não é um parâmetro deste modelo.",
}
_TYPES: Final = {
    "int_type": "um número inteiro",
    "int_parsing": "um número inteiro",
    "int_from_float": "um número inteiro",
    "float_type": "um número",
    "float_parsing": "um número",
    "bool_type": "verdadeiro ou falso",
    "bool_parsing": "verdadeiro ou falso",
    "string_type": "texto",
    "dict_type": "um objeto (com chaves e valores)",
    "model_type": "um objeto (com chaves e valores)",
    "dataclass_type": "um objeto (com chaves e valores)",
    "list_type": "uma lista",
    "tuple_type": "uma lista",
}


def describe_errors(
    error: ValidationError, titles: Mapping[str, str] | None = None
) -> tuple[str, ...]:
    """Uma frase em português por cada problema de ``error``.

    Args:
        error: O erro do Pydantic.
        titles: Nome a mostrar para cada campo (por omissão, o nome do campo).
    """
    return describe_problems(error.errors(include_url=False), titles)


def describe_problems(
    problems: Iterable[Mapping[str, Any]], titles: Mapping[str, str] | None = None
) -> tuple[str, ...]:
    """Uma frase em português por cada problema, no formato do Pydantic.

    É o que ``describe_errors`` usa, e serve também aos erros que chegam já em lista (os
    de um pedido à API, por exemplo).

    Args:
        problems: Os problemas, cada um com ``loc``, ``type``, ``msg`` e, às vezes, ``ctx``.
        titles: Nome a mostrar para cada campo (por omissão, o nome do campo).
    """
    sentences: list[str] = []
    for problem in problems:
        location = ".".join(str(part) for part in problem["loc"]) or "valor"
        field = (titles or {}).get(location, location)
        kind = problem["type"]
        context: Mapping[str, Any] = problem.get("ctx", {})
        if kind in _TEMPLATES:
            sentences.append(_TEMPLATES[kind](field, context))
        elif kind in _TYPES:
            sentences.append(f"{field} tem de ser {_TYPES[kind]}; recebi {problem['input']!r}.")
        elif kind == "value_error":
            # Erros lançados pelo nosso código: a mensagem já está em português.
            sentences.append(str(problem["msg"]).removeprefix("Value error, "))
        else:
            sentences.append(f"{field}: {problem['msg']}.")
    return tuple(sentences)
