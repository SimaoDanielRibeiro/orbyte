"""Erros de validação do Pydantic explicados em inglês simples.

O Pydantic descreve cada problema com um tipo estável (``greater_than``,
``literal_error``...) e uma mensagem técnica. Aqui cada tipo vira uma frase curta que diz o
campo, o valor recebido e o que era esperado.
"""

from collections.abc import Callable, Iterable, Mapping
from typing import Any, Final

from pydantic import ValidationError

from ml_lab.formatting import format_number


def _number(value: object) -> str:
    return (
        format_number(value, 4).rstrip("0").rstrip(".") if isinstance(value, float) else str(value)
    )


_TEMPLATES: Final[Mapping[str, Callable[[str, Mapping[str, Any]], str]]] = {
    "greater_than": lambda field, ctx: f"{field} must be greater than {_number(ctx['gt'])}.",
    "greater_than_equal": lambda field, ctx: f"{field} must be at least {_number(ctx['ge'])}.",
    "less_than": lambda field, ctx: f"{field} must be less than {_number(ctx['lt'])}.",
    "less_than_equal": lambda field, ctx: f"{field} can't be more than {_number(ctx['le'])}.",
    "literal_error": lambda field, ctx: f"{field} must be one of: {ctx['expected']}.",
    "enum": lambda field, ctx: f"{field} must be one of: {ctx['expected']}.",
    "missing": lambda field, ctx: f"{field} is missing a value.",
    "extra_forbidden": lambda field, ctx: f"{field} isn't a parameter of this model.",
}
_TYPES: Final = {
    "int_type": "a whole number",
    "int_parsing": "a whole number",
    "int_from_float": "a whole number",
    "float_type": "a number",
    "float_parsing": "a number",
    "bool_type": "true or false",
    "bool_parsing": "true or false",
    "string_type": "text",
    "dict_type": "an object (with keys and values)",
    "model_type": "an object (with keys and values)",
    "dataclass_type": "an object (with keys and values)",
    "list_type": "a list",
    "tuple_type": "a list",
}


def describe_errors(
    error: ValidationError, titles: Mapping[str, str] | None = None
) -> tuple[str, ...]:
    """Uma frase por cada problema de ``error``.

    Args:
        error: O erro do Pydantic.
        titles: Nome a mostrar para cada campo (por omissão, o nome do campo).
    """
    return describe_problems(error.errors(include_url=False), titles)


def describe_problems(
    problems: Iterable[Mapping[str, Any]], titles: Mapping[str, str] | None = None
) -> tuple[str, ...]:
    """Uma frase por cada problema, no formato do Pydantic.

    É o que ``describe_errors`` usa, e serve também aos erros que chegam já em lista (os
    de um pedido à API, por exemplo).

    Args:
        problems: Os problemas, cada um com ``loc``, ``type``, ``msg`` e, às vezes, ``ctx``.
        titles: Nome a mostrar para cada campo (por omissão, o nome do campo).
    """
    sentences: list[str] = []
    for problem in problems:
        location = ".".join(str(part) for part in problem["loc"]) or "value"
        field = (titles or {}).get(location, location)
        kind = problem["type"]
        context: Mapping[str, Any] = problem.get("ctx", {})
        if kind in _TEMPLATES:
            sentences.append(_TEMPLATES[kind](field, context))
        elif kind in _TYPES:
            sentences.append(f"{field} must be {_TYPES[kind]}; got {problem['input']!r}.")
        elif kind == "value_error":
            # Erros lançados pelo nosso código: a mensagem já vem pronta a mostrar.
            sentences.append(str(problem["msg"]).removeprefix("Value error, "))
        else:
            sentences.append(f"{field}: {problem['msg']}.")
    return tuple(sentences)
