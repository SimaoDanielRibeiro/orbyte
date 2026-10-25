"""O que os testes da API partilham: os serviços, o cliente HTTP e esperar por um treino."""

import time
from collections.abc import Callable, Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from ml_lab.api.app import create_app
from ml_lab.core.services import Lab
from ml_lab.core.tracking.store import InMemoryStore

type Waiter = Callable[[str], dict[str, Any]]

IRIS: dict[str, Any] = {"kind": "sample", "params": {"name": "iris"}}


@pytest.fixture
def lab() -> Lab:
    # O registo em memória; as pastas já apontam para uma pasta temporária (tests/conftest.py).
    return Lab.create(store=InMemoryStore())


@pytest.fixture
def client(lab: Lab) -> Iterator[TestClient]:
    with TestClient(create_app(lab)) as test_client:
        yield test_client


@pytest.fixture
def finished(client: TestClient) -> Waiter:
    """Espera que um treino termine e devolve o estado final."""

    def wait(run_id: str, prefix: str = "/experiments", timeout: float = 60) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            state: dict[str, Any] = client.get(f"{prefix}/{run_id}/job").json()
            if state["status"] in {"finished", "failed"}:
                return state
            time.sleep(0.02)
        raise AssertionError(f"O treino {run_id} não terminou em {timeout} s.")

    return wait


def experiment(model: str = "logistic_regression_gd", **hyperparameters: Any) -> dict[str, Any]:
    """Uma experiência no Iris, em JSON, como o frontend a envia."""
    return {
        "dataset": IRIS,
        "preparation": {"target": "species"},
        "model": {"name": model, "hyperparameters": hyperparameters},
        "seed": 42,
    }


@pytest.fixture
def config() -> Callable[..., dict[str, Any]]:
    """Monta a experiência de ``experiment`` (os testes não importam o conftest)."""
    return experiment
