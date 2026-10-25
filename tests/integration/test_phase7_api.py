"""Critério da fase 7: pela API, carregar um CSV, lançar o treino e receber os snapshots.

O teste faz o mesmo que se faz à mão na documentação (``/docs``): envia um CSV das luas,
começa o treino de uma rede neuronal e acompanha-o enquanto decorre, primeiro perguntando
(``GET .../updates``) e depois pelo WebSocket. Os snapshots chegam durante o treino, cada
um com a fronteira desse passo, e no fim o resultado tem a animação completa.
"""

import time
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from ml_lab.api.app import create_app
from ml_lab.core.services import Lab
from ml_lab.core.tracking.store import InMemoryStore


def moons_csv(n: int = 300, noise: float = 0.1) -> bytes:
    """As luas num CSV, como um utilizador o enviaria (duas features e a classe)."""
    rng = np.random.default_rng(0)
    angles = rng.uniform(0, np.pi, size=n)
    upper = np.arange(n) < n // 2
    x = np.where(upper, np.cos(angles), 1 - np.cos(angles)) + rng.normal(scale=noise, size=n)
    y = np.where(upper, np.sin(angles), 0.5 - np.sin(angles)) + rng.normal(scale=noise, size=n)
    lines = ["x,y,lua"] + [
        f"{a:.4f},{b:.4f},{'cima' if top else 'baixo'}"
        for a, b, top in zip(x, y, upper, strict=True)
    ]
    return "\n".join(lines).encode()


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(create_app(Lab.create(store=InMemoryStore()))) as test_client:
        yield test_client


def start_training(client: TestClient) -> str:
    uploaded = client.post("/datasets/uploads", files={"file": ("luas.csv", moons_csv())})
    assert uploaded.status_code == 200
    config = {
        "dataset": uploaded.json()["source"],
        "preparation": {"target": "lua"},
        "model": {"name": "mlp", "hyperparameters": {"max_epochs": 400, "early_stopping": False}},
        "seed": 0,
    }
    started = client.post("/experiments", json=config)
    assert started.status_code == 202
    run_id: str = started.json()["run_id"]
    return run_id


def frame_steps(previews: list[dict[str, Any]]) -> list[int]:
    return [preview["frames"][0]["step"] for preview in previews]


def test_the_documentation_shows_the_way(client: TestClient) -> None:
    assert client.get("/docs").status_code == 200
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/datasets/uploads", "/experiments", "/experiments/{run_id}/updates"} <= set(paths)


def test_a_csv_trains_and_its_snapshots_arrive_while_it_trains(client: TestClient) -> None:
    run_id = start_training(client)

    previews: list[dict[str, Any]] = []
    after_step = after_snapshot = 0
    while True:
        update = client.get(
            f"/experiments/{run_id}/updates",
            params={"after_step": after_step, "after_snapshot": after_snapshot},
        ).json()
        if update["steps"]:
            after_step = update["steps"][-1]["step"]
        if update["preview"] is not None:
            previews.append(update["preview"])
            after_snapshot = update["preview"]["frames"][0]["step"]
        if update["state"]["status"] in {"finished", "failed"}:
            break
        time.sleep(0.05)

    result = client.get(f"/experiments/{run_id}/result").json()
    assert len(previews) >= 2  # chegaram fotografias durante o treino
    assert frame_steps(previews) == sorted(set(frame_steps(previews)))  # sempre mais recentes
    grid = previews[0]["frames"][0]["values"]
    assert len(grid) == len(previews[0]["ys"])  # cada uma é a fronteira inteira, na grelha
    assert len(result["animation"]["frames"]) > 10
    assert result["reports"]["test"]["metrics"][0]["value"] > 0.95


def test_the_websocket_streams_the_same_snapshots(client: TestClient) -> None:
    run_id = start_training(client)

    with client.websocket_connect(f"/experiments/{run_id}/stream") as websocket:
        messages: list[dict[str, Any]] = []
        while not messages or messages[-1]["data"]["state"]["status"] == "running":
            messages.append(websocket.receive_json())

    previews = [m["data"]["preview"] for m in messages if m["data"]["preview"] is not None]
    assert messages[-1]["data"]["state"]["status"] == "finished"
    assert len(previews) >= 2
    assert frame_steps(previews) == sorted(set(frame_steps(previews)))
