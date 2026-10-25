"""O OpenAPI de onde o frontend gera os tipos TypeScript, com as respostas completas.

O FastAPI gera o esquema a partir dos tipos que as rotas anunciam. Num ponto, o esquema diz
menos do que a API faz: nas respostas, um campo com valor por omissão aparece como opcional,
mas o ``to_json`` manda sempre todos os campos. No TypeScript, isso dava
``history?: StepRecord[]`` e obrigava o frontend a tratar um ``undefined`` que nunca chega.

``complete_responses`` corrige o esquema: nos tipos que só aparecem em respostas, todos os
campos são obrigatórios. Os tipos que também se enviam em pedidos (``ExperimentConfig``,
``PreparationConfig``…) ficam como estão, porque num pedido um campo com valor por omissão
pode mesmo faltar (ADR 0026).
"""

from collections.abc import Iterable, Iterator
from typing import Any, Final

from fastapi import FastAPI

_PREFIX: Final = "#/components/schemas/"


def install(app: FastAPI) -> None:
    """Põe a aplicação a servir o OpenAPI com as respostas completas (em ``/openapi.json``)."""
    generate = app.openapi

    def openapi() -> dict[str, Any]:
        if app.openapi_schema is None:
            app.openapi_schema = complete_responses(generate())
        return app.openapi_schema

    app.openapi = openapi  # type: ignore[method-assign]  # é assim que o FastAPI o permite


def complete_responses(schema: dict[str, Any]) -> dict[str, Any]:
    """Marca como obrigatórios todos os campos dos tipos que só aparecem em respostas."""
    components: dict[str, Any] = schema.get("components", {}).get("schemas", {})
    sent = _reachable(_request_parts(schema), components)
    for name, component in components.items():
        properties = component.get("properties")
        if name not in sent and properties:
            component["required"] = list(properties)
    return schema


def _request_parts(schema: dict[str, Any]) -> Iterator[Any]:
    """O que os clientes enviam: os corpos e os parâmetros de cada operação."""
    for operations in schema.get("paths", {}).values():
        for operation in operations.values():
            yield operation.get("requestBody", {})
            yield from operation.get("parameters", [])


def _reachable(roots: Iterable[Any], components: dict[str, Any]) -> set[str]:
    """Os tipos a que se chega a partir de ``roots``, seguindo as referências."""
    found: set[str] = set()
    pending = [name for root in roots for name in _references(root)]
    while pending:
        name = pending.pop()
        if name not in found:
            found.add(name)
            pending.extend(_references(components.get(name, {})))
    return found


def _references(fragment: Any) -> Iterator[str]:
    """Os nomes dos tipos referidos (``$ref``) dentro de um pedaço do esquema."""
    if isinstance(fragment, dict):
        reference = fragment.get("$ref")
        if isinstance(reference, str) and reference.startswith(_PREFIX):
            yield reference.removeprefix(_PREFIX)
        for value in fragment.values():
            yield from _references(value)
    elif isinstance(fragment, list):
        for value in fragment:
            yield from _references(value)
