"""Formulários gerados a partir de JSON Schema.

Os parâmetros dos geradores de dados e os hiperparâmetros dos modelos descrevem-se
em JSON Schema (tipo, limites, valor por omissão, título, explicação). Este módulo
desenha um controlo por parâmetro, sem código específico para cada um: a interface
aprimorada fará o mesmo com o mesmo esquema.

Além do JSON Schema normal, entende duas extensões opcionais:

- ``"x-scale": "log"``: números que variam em ordens de grandeza (como o ``C`` da
  regressão logística) ganham um seletor com valores 0,001, 0,01, 0,1…;
- ``"x-labels"``: nomes a mostrar para as opções de um ``enum``.
"""

import math
from collections.abc import Mapping
from typing import Any

import streamlit as st

from ml_lab.formatting import format_number, format_scientific

_LOG_STEPS = (1.0, 2.0, 5.0)


def schema_form(
    schema: Mapping[str, Any], *, key: str, values: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Desenha um controlo por propriedade de ``schema`` e devolve os valores escolhidos.

    Args:
        schema: Um JSON Schema do tipo ``object``.
        key: Prefixo das chaves dos controlos (tem de ser único na página).
        values: Valores iniciais; os que faltam usam o ``default`` do esquema.
    """
    chosen: dict[str, Any] = {}
    definitions: Mapping[str, Any] = schema.get("$defs", {})
    for name, raw in schema.get("properties", {}).items():
        prop = _resolve(raw, definitions)
        initial = (values or {}).get(name, prop.get("default"))
        chosen[name] = _control(name, prop, initial, f"{key}.{name}")
    return chosen


def _resolve(prop: Mapping[str, Any], definitions: Mapping[str, Any]) -> dict[str, Any]:
    """Junta ``$ref`` e ``anyOf`` (como o Pydantic os gera) numa só descrição."""
    resolved = dict(prop)
    reference = resolved.pop("$ref", None)
    if isinstance(reference, str):
        resolved = {**definitions[reference.rsplit("/", 1)[-1]], **resolved}
    options = resolved.pop("anyOf", None)
    if isinstance(options, list):
        concrete: Mapping[str, Any] = next((o for o in options if o.get("type") != "null"), {})
        resolved = {**_resolve(concrete, definitions), **resolved}
    return resolved


def _control(name: str, prop: Mapping[str, Any], initial: Any, key: str) -> Any:
    label = str(prop.get("title") or name)
    help_text = prop.get("description") or None
    if "enum" in prop:
        options = list(prop["enum"])
        labels: Mapping[str, str] = prop.get("x-labels", {})
        index = options.index(initial) if initial in options else 0
        return st.selectbox(
            label,
            options,
            index=index,
            format_func=lambda option: labels.get(str(option), str(option)),
            help=help_text,
            key=key,
            persist_state="session",
        )
    kind = prop.get("type")
    if kind == "boolean":
        return st.toggle(
            label, value=bool(initial), help=help_text, key=key, persist_state="session"
        )
    if kind in ("integer", "number"):
        return _number(label, prop, initial, key, help_text)
    return st.text_input(
        label, value=str(initial or ""), help=help_text, key=key, persist_state="session"
    )


def _number(
    label: str, prop: Mapping[str, Any], initial: Any, key: str, help_text: str | None
) -> Any:
    integer = prop.get("type") == "integer"
    minimum = prop.get("minimum", prop.get("exclusiveMinimum"))
    maximum = prop.get("maximum", prop.get("exclusiveMaximum"))
    cast = int if integer else float
    value = cast(initial if initial is not None else (minimum or 0))
    if minimum is not None:
        value = max(value, cast(minimum))
    if maximum is not None:
        value = min(value, cast(maximum))
    if prop.get("x-scale") == "log" and minimum and maximum:
        options = _log_options(float(minimum), float(maximum), float(value))
        return st.select_slider(
            label,
            options=options,
            value=min(options, key=lambda option: abs(option - value)),
            format_func=_format_option,
            help=help_text,
            key=key,
            persist_state="session",
        )
    if minimum is not None and maximum is not None:
        step = 1 if integer else _nice_step(float(minimum), float(maximum))
        return st.slider(
            label,
            min_value=cast(minimum),
            max_value=cast(maximum),
            value=value,
            step=step,
            help=help_text,
            key=key,
            persist_state="session",
        )
    return st.number_input(
        label,
        min_value=None if minimum is None else cast(minimum),
        max_value=None if maximum is None else cast(maximum),
        value=value,
        help=help_text,
        key=key,
        persist_state="session",
    )


def _log_options(minimum: float, maximum: float, value: float) -> list[float]:
    """Valores 1, 2 e 5 em cada potência de 10 entre ``minimum`` e ``maximum``."""
    options = {value}
    for exponent in range(math.floor(math.log10(minimum)), math.ceil(math.log10(maximum)) + 1):
        for step in _LOG_STEPS:
            option = float(f"{step * 10.0**exponent:.12g}")
            if minimum <= option <= maximum:
                options.add(option)
    return sorted(options)


def _nice_step(minimum: float, maximum: float) -> float:
    """Um passo "redondo" (0,01, 0,05, 0,1…) que dá cerca de cem posições no controlo."""
    raw = (maximum - minimum) / 100
    if raw <= 0:
        return 0.01
    magnitude = 10.0 ** math.floor(math.log10(raw))
    for multiple in (1, 2, 5, 10):
        if raw <= multiple * magnitude:
            return float(f"{multiple * magnitude:.12g}")
    return raw


def _format_option(value: float) -> str:
    if float(value).is_integer():
        return format_number(value, 0)
    if abs(value) < 0.001:  # a tolerância de paragem (1 × 10⁻⁶) ficaria "0"
        return format_scientific(value)
    return format_number(value, 4).rstrip("0").rstrip(",")
