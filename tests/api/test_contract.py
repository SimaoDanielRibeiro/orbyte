"""O contrato da API com os serviços: cada método público tem a sua rota, documentada.

Como cada operação se chama ``<serviço>_<método>`` (ver ``ml_lab.api.routing``), um método
novo num serviço que fique sem rota faz falhar o primeiro teste: a interface nunca fica sem
acesso a uma coisa que o backend já sabe fazer.
"""

import inspect
import math
import re
from dataclasses import dataclass
from typing import Any, get_type_hints

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ml_lab.api.routing import ServiceRouter, operation_id
from ml_lab.core.schemas.serialization import is_internal
from ml_lab.core.services import Lab

NOT_EXPOSED = {
    "experiments_run": "a API treina sempre em segundo plano (experiments_start)",
    "reinforcement_run": "a API treina sempre em segundo plano (reinforcement_start)",
    "automl_run": "a API procura sempre em segundo plano (automl_start)",
}
"""Os métodos públicos que ficam de fora da API, de propósito, e porquê."""


def service_operations() -> set[str]:
    """``<serviço>_<método>`` de cada método público dos serviços do ``Lab``."""
    found = set()
    for service, service_class in get_type_hints(Lab).items():
        for name, method in inspect.getmembers(service_class, inspect.isfunction):
            if not name.startswith("_") and not is_internal(method):
                found.add(f"{service}_{name}")
    return found


def operations(client: TestClient) -> dict[str, dict[str, Any]]:
    paths = client.get("/openapi.json").json()["paths"]
    return {
        operation["operationId"]: operation
        for path in paths.values()
        for operation in path.values()
    }


def test_every_public_service_method_has_a_route(client: TestClient) -> None:
    missing = service_operations() - set(operations(client)) - set(NOT_EXPOSED)

    assert not missing, f"Métodos dos serviços sem rota na API: {sorted(missing)}"


def test_the_exceptions_are_still_service_methods() -> None:
    # Uma exceção que deixou de existir no serviço só engana quem lê a lista.
    assert set(NOT_EXPOSED) <= service_operations()


def test_every_route_has_a_summary_in_portuguese(client: TestClient) -> None:
    summaries = {name: op.get("summary", "") for name, op in operations(client).items()}

    assert all(summaries.values()), [name for name, text in summaries.items() if not text]
    assert summaries["datasets_samples"] == "Os datasets de exemplo."


def test_the_documentation_and_the_health_check_answer(client: TestClient) -> None:
    docs = client.get("/docs")
    health = client.get("/health")

    assert docs.status_code == 200
    assert "swagger" in docs.text.lower()
    assert client.get("/openapi.json").json()["info"]["title"] == "Orbyte"
    assert health.json()["status"] == "ok"


@dataclass(frozen=True, slots=True)
class Odd:
    loss: float
    state: dict[str, Any]


def test_answers_go_through_the_project_serializer() -> None:
    # O to_json (ADR 0019): NaN e infinito passam a null, e o NumPy a números.
    router = ServiceRouter(tags=["odd"])

    @router.get("/odd")
    def odd() -> Odd:
        return Odd(math.nan, {"weights": np.array([1.5, math.inf])})

    app = FastAPI(generate_unique_id_function=operation_id)
    app.include_router(router)

    response = TestClient(app).get("/odd")

    assert response.json() == {"loss": None, "state": {"weights": [1.5, None]}}
    assert "NaN" not in response.text
    assert "Infinity" not in response.text


def schemas(client: TestClient) -> dict[str, Any]:
    components: dict[str, Any] = client.get("/openapi.json").json()["components"]["schemas"]
    return components


def test_every_type_has_one_schema_with_a_plain_name(client: TestClient) -> None:
    # Um nome como "ExperimentConfig-Input" quer dizer dois esquemas para o mesmo tipo, e dois
    # tipos no TypeScript do frontend (ADR 0026).
    odd = [name for name in schemas(client) if not re.fullmatch(r"[A-Z][A-Za-z0-9_]*", name)]

    assert not odd


def test_responses_promise_every_field_and_requests_only_the_needed_ones(
    client: TestClient,
) -> None:
    # O to_json manda sempre todos os campos: nos tipos que só se recebem, nenhum é opcional.
    # Nos que também se enviam, um campo com valor por omissão pode faltar no pedido.
    components = schemas(client)

    result = components["ExperimentResult"]
    assert set(result["required"]) == set(result["properties"])
    assert "history" in result["required"]
    request = components["ExperimentConfig"]
    assert set(request["required"]) == {"dataset", "model"}
    assert "required" not in components["SearchSettings"]


def test_every_error_answer_is_an_api_error(client: TestClient) -> None:
    operation = client.get("/openapi.json").json()["paths"]["/experiments"]["post"]
    errors = {code: answer for code, answer in operation["responses"].items() if code >= "400"}

    assert set(errors) == {"400", "404", "409", "422", "500"}
    for answer in errors.values():
        schema = answer["content"]["application/json"]["schema"]
        assert schema["$ref"].endswith("/ApiError")
