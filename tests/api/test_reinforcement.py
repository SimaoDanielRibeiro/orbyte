"""Reinforcement learning pela API: lançar o treino de um agente, acompanhá-lo e o histórico."""

from collections.abc import Callable
from typing import Any

from fastapi.testclient import TestClient

type Waiter = Callable[..., dict[str, Any]]

TRAINING = {
    "environment": {"name": "gridworld", "parameters": {"layout": "penhasco"}},
    "agent": {"name": "q_learning", "hyperparameters": {"episodes": 300}},
    "seed": 5,
}


def test_the_catalogue_lists_environments_and_agents(client: TestClient) -> None:
    environments = client.get("/reinforcement/environments").json()
    agents = client.get("/reinforcement/agents").json()

    assert environments[0]["name"] == "gridworld"
    assert [agent["name"] for agent in agents] == ["q_learning", "dqn"]
    assert agents[0]["implementation"] == "didactic"


def test_the_lesson_comes_step_by_step(client: TestClient) -> None:
    lesson = client.get("/reinforcement/lesson").json()

    first = lesson["kept"][0]
    assert lesson["agent_display_name"] == "Q-learning"
    assert first["episode"] == 1
    assert {"state", "action", "explored", "reward", "surprise", "value"} <= set(first["steps"][0])
    assert lesson["kept"][-1]["reached"] is True


def test_an_agent_trains_in_the_background_and_lands_in_the_history(
    client: TestClient, finished: Waiter
) -> None:
    started = client.post("/reinforcement", json=TRAINING)

    assert started.status_code == 202
    run_id = started.json()["run_id"]
    assert finished(run_id, prefix="/reinforcement")["status"] == "finished"
    result = client.get(f"/reinforcement/{run_id}/result").json()
    assert result["reports"]["test"]["metrics"][0]["name"] == "success_rate"
    assert result["layout"]["cells"][36] == "start"
    assert result["learned"]["reached"] is True
    entry = client.get(f"/history/runs/{run_id}").json()
    assert (entry["task_type"], entry["model_name"]) == ("control", "q_learning")
    reopened = client.get(f"/history/runs/{run_id}/reinforcement-result").json()
    assert reopened["run_id"] == run_id
    assert client.get(f"/history/runs/{run_id}/result").status_code == 422  # não é um modelo


def test_the_websocket_sends_the_episodes_and_the_policy(client: TestClient) -> None:
    run_id = client.post("/reinforcement", json=TRAINING).json()["run_id"]

    messages: list[dict[str, Any]] = []
    with client.websocket_connect(f"/reinforcement/{run_id}/stream") as websocket:
        while not messages or messages[-1]["data"]["state"]["status"] == "running":
            messages.append(websocket.receive_json())

    episodes = [step["step"] for message in messages for step in message["data"]["steps"]]
    assert messages[-1]["data"]["state"]["status"] == "finished"
    assert episodes == list(range(1, 301))


def test_bad_configurations_are_explained(client: TestClient) -> None:
    unknown = client.post("/reinforcement", json={**TRAINING, "agent": {"name": "sarsa"}})
    bad_map = client.post(
        "/reinforcement",
        json={**TRAINING, "environment": {"name": "gridworld", "parameters": {"layout": "lua"}}},
    )

    assert unknown.status_code == 404
    assert unknown.json()["code"] == "component_not_found"
    assert bad_map.status_code == 422
    assert "environment 'gridworld'" in bad_map.json()["message"]
