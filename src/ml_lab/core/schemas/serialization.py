"""A fronteira em JSON: o que os serviços devolvem, pronto para a API e para o frontend.

Os serviços devolvem dataclasses e modelos Pydantic com tipos simples: texto, números,
booleanos, enums, tuplos e dicionários. O Pydantic sabe descrever estes tipos (o JSON
Schema, de onde o FastAPI tira o OpenAPI e o frontend tira os tipos em TypeScript) e
passá-los a JSON, sem código de conversão para cada tipo. Este módulo é o único sítio
onde isso se faz (ADR 0019):

- ``to_json`` passa um resultado a valores de JSON. Os números que não são finitos (NaN,
  infinito: uma loss que explodiu) passam a ``None``, porque o JSON não os tem e o
  navegador rejeita-os; os números do NumPy passam a números simples.
- ``from_json`` faz o caminho inverso: valores de JSON de volta ao tipo (por exemplo, um
  resultado guardado no registo de experiências, ou um pedido que chega à API).
- ``json_schema`` descreve um tipo devolvido pelos serviços.
- ``@internal`` marca os métodos dos serviços que só servem outros serviços, porque
  devolvem DataFrames, matrizes ou o modelo treinado: a API não os expõe, e o teste de
  contrato da fronteira não os exige serializáveis.
"""

import json
from collections.abc import Callable
from typing import Any, Final

import numpy as np
from pydantic import TypeAdapter

_INTERNAL: Final = "__ml_lab_internal__"

_ADAPTERS: dict[Any, TypeAdapter[Any]] = {}


def to_json(value: Any) -> Any:
    """``value`` em valores de JSON: dicionários, listas, texto, números, booleanos e ``None``.

    Raises:
        PydanticSerializationError: Se ``value`` tiver alguma coisa que não passa a JSON
            (por exemplo, um DataFrame).
    """
    return json.loads(_adapter(type(value)).dump_json(value, fallback=_plain))


def from_json[T](kind: type[T], data: Any) -> T:
    """Os valores de JSON ``data`` de volta ao tipo ``kind`` (o caminho inverso de ``to_json``).

    Raises:
        pydantic.ValidationError: Se os valores não corresponderem ao tipo.
    """
    value: T = _adapter(kind).validate_python(data)
    return value


def json_schema(kind: Any) -> dict[str, Any]:
    """O JSON Schema de um tipo devolvido pelos serviços.

    Raises:
        PydanticSchemaGenerationError: Se o tipo tiver campos que não passam a JSON (um
            DataFrame, uma matriz do NumPy, um modelo).
    """
    schema: dict[str, Any] = _adapter(kind).json_schema(mode="serialization")
    return schema


def internal[F: Callable[..., Any]](method: F) -> F:
    """Marca um método de um serviço que só serve outros serviços (ver o módulo)."""
    setattr(method, _INTERNAL, True)
    return method


def is_internal(method: Callable[..., Any]) -> bool:
    """Se ``method`` foi marcado com ``@internal``."""
    return bool(getattr(method, _INTERNAL, False))


def _adapter(kind: Any) -> TypeAdapter[Any]:
    """O ``TypeAdapter`` de ``kind`` (criá-lo custa: fica guardado um por tipo)."""
    if kind not in _ADAPTERS:
        _ADAPTERS[kind] = TypeAdapter(kind)
    return _ADAPTERS[kind]


def _plain(value: Any) -> Any:
    """Os valores do NumPy que aparecem em campos livres (por exemplo, o estado de um modelo)."""
    if isinstance(value, np.ndarray | np.generic):
        return value.tolist()
    raise TypeError(f"Um valor do tipo {type(value).__name__} não passa a JSON.")
