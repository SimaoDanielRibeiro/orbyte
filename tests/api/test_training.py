"""Treinar pela API: lançar em segundo plano, acompanhar, controlar e pedir o resultado."""

import json
import time
from collections.abc import Callable
from typing import Any

from fastapi.testclient import TestClient

type Waiter = Callable[[str], dict[str, Any]]
type Config = Callable[..., dict[str, Any]]


def strict(text: str) -> Any:
    """O JSON de uma resposta, recusando NaN e infinitos (que o navegador não aceita)."""

    def refuse(constant: str) -> None:
        raise AssertionError(f"A resposta tem {constant}, que não é JSON.")

    return json.loads(text, parse_constant=refuse)


def test_training_runs_in_the_background_and_ends_with_a_result(
    client: TestClient, config: Config, finished: Waiter
) -> None:
    started = client.post("/experiments", json=config())

    assert started.status_code == 202
    run_id = started.json()["run_id"]
    assert finished(run_id)["status"] == "finished"
    result = client.get(f"/experiments/{run_id}/result")
    assert result.status_code == 200
    body = strict(result.text)
    assert body["run_id"] == run_id
    assert body["reports"]["test"]["metrics"][0]["name"] == "accuracy"
    assert len(body["animation"]["frames"]) > 1


def test_updates_bring_only_what_is_new(
    client: TestClient, config: Config, finished: Waiter
) -> None:
    run_id = client.post("/experiments", json=config()).json()["run_id"]
    finished(run_id)

    everything = client.get(f"/experiments/{run_id}/updates").json()
    last = everything["steps"][-1]["step"]
    nothing_new = client.get(f"/experiments/{run_id}/updates", params={"after_step": last}).json()

    assert everything["state"]["status"] == "finished"
    assert [step["step"] for step in everything["steps"]] == list(range(1, last + 1))
    assert nothing_new["steps"] == []


def test_a_paused_training_has_no_result_yet_and_can_be_stopped(
    client: TestClient, config: Config, finished: Waiter
) -> None:
    # O máximo de épocas, a tolerância mínima e passos pequenos: não acaba antes da pausa.
    long = config("linear_regression_gd", max_epochs=20_000, tol=1e-10, learning_rate=0.001)
    long["preparation"] = {"target": "petal_width"}
    run_id = client.post("/experiments", json=long).json()["run_id"]

    paused = client.post(f"/experiments/{run_id}/pause").json()
    early = client.get(f"/experiments/{run_id}/result")
    resumed = client.post(f"/experiments/{run_id}/resume").json()
    time.sleep(0.1)
    client.post(f"/experiments/{run_id}/cancel")

    assert paused["status"] == "paused"
    assert early.status_code == 409
    assert early.json()["code"] == "training_in_progress"
    assert resumed["status"] == "running"
    assert finished(run_id)["status"] == "finished"
    result = client.get(f"/experiments/{run_id}/result").json()
    assert result["stopped_early"] is True
    assert 0 < len(result["history"]) < 20_000


def test_the_websocket_sends_the_training_until_it_ends(client: TestClient, config: Config) -> None:
    run_id = client.post("/experiments", json=config(max_epochs=300)).json()["run_id"]

    messages = []
    with client.websocket_connect(f"/experiments/{run_id}/stream") as websocket:
        while True:
            message = websocket.receive_json()
            messages.append(message)
            if message["data"]["state"]["status"] in {"finished", "failed"}:
                break

    steps = [step["step"] for message in messages for step in message["data"]["steps"]]
    result = client.get(f"/experiments/{run_id}/result").json()
    assert {message["kind"] for message in messages} == {"update"}
    assert messages[-1]["data"]["state"]["status"] == "finished"
    assert steps == [record["step"] for record in result["history"]]  # nenhum perdido ou repetido


def test_the_websocket_explains_an_unknown_training(client: TestClient) -> None:
    with client.websocket_connect("/experiments/run-nada/stream") as websocket:
        message = websocket.receive_json()

    assert message["kind"] == "error"
    assert message["data"]["code"] == "job_not_found"
