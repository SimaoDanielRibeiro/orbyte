"""Os agentes pela API: pedir o melhor modelo, seguir a conversa e treinar um candidato à mão."""

from collections.abc import Callable
from typing import Any

from fastapi.testclient import TestClient

type Waiter = Callable[..., dict[str, Any]]

IRIS = {"kind": "sample", "params": {"name": "iris"}}
REQUEST = {
    "dataset": IRIS,
    "target": "species",
    "families": ["linear", "instance_based", "tree"],
    "settings": {"budget": 5, "folds": 3, "finalists": 2},
    "seed": 3,
}


def test_the_options_explain_what_can_be_asked(client: TestClient) -> None:
    binary = client.get("/automl/options", params={"task_type": "binary_classification"}).json()
    clustering = client.get("/automl/options", params={"task_type": "clustering"}).json()

    assert binary["supported"]
    assert binary["objectives"][0]["name"] == "accuracy"
    assert binary["settings"]["budget"] == 24
    assert not clustering["supported"]
    assert "modo manual" in clustering["reason"]


def test_the_plan_shows_what_the_data_agent_decided(client: TestClient) -> None:
    plan = client.post("/automl/plan", json=REQUEST).json()

    assert plan["task_type"] == "multiclass_classification"
    assert plan["decisions"][0]["title"] == "Remover linhas duplicadas"
    assert plan["baseline"] == "logistic_regression"


def test_a_search_runs_in_the_background_and_ends_with_the_winners_run(
    client: TestClient, finished: Waiter
) -> None:
    started = client.post("/automl", json=REQUEST)

    assert started.status_code == 202
    search_id = started.json()["run_id"]
    assert search_id.startswith("search-")
    assert finished(search_id, prefix="/automl", timeout=120)["status"] == "finished"
    result = client.get(f"/automl/{search_id}/result").json()
    assert len(result["candidates"]) == 5
    assert result["winner"] == result["ranking"][0]
    winner = client.get(f"/experiments/{result['run_id']}/result").json()  # um run normal
    assert winner["model_name"] == result["candidates"][result["winner"] - 1]["model"]
    entry = client.get(f"/history/runs/{result['run_id']}").json()
    assert entry["name"].startswith("Agentes · ")


def test_updates_bring_only_the_new_candidates_and_messages(
    client: TestClient, finished: Waiter
) -> None:
    search_id = client.post("/automl", json=REQUEST).json()["run_id"]
    finished(search_id, prefix="/automl", timeout=120)

    later = client.get(
        f"/automl/{search_id}/updates", params={"after_step": 3, "after_message": 5}
    ).json()

    assert [candidate["number"] for candidate in later["candidates"]] == [4, 5]
    assert later["messages"][0]["number"] == 6
    assert later["phase"] == "done"


def test_the_websocket_streams_the_conversation_until_the_end(client: TestClient) -> None:
    search_id = client.post("/automl", json=REQUEST).json()["run_id"]
    updates = []

    with client.websocket_connect(f"/automl/{search_id}/stream") as websocket:
        while True:
            message = websocket.receive_json()
            updates.append(message)
            if message["data"]["state"]["status"] in {"finished", "failed"}:
                break

    numbers = [m["number"] for update in updates for m in update["data"]["messages"]]
    candidates = [c["number"] for update in updates for c in update["data"]["candidates"]]
    result = client.get(f"/automl/{search_id}/result").json()
    assert numbers == [message["number"] for message in result["messages"]]  # nem uma perdida
    assert candidates == [1, 2, 3, 4, 5]


def test_a_candidate_can_be_trained_by_hand(client: TestClient, finished: Waiter) -> None:
    search_id = client.post("/automl", json=REQUEST).json()["run_id"]
    finished(search_id, prefix="/automl", timeout=120)

    started = client.post(f"/automl/{search_id}/candidates/2/train")

    assert started.status_code == 202
    run_id = started.json()["run_id"]
    assert finished(run_id)["status"] == "finished"  # acompanha-se nas rotas do treino
    assert client.get(f"/experiments/{run_id}/result").json()["model_name"] == "knn"


def test_unknown_searches_and_candidates_are_404(client: TestClient, finished: Waiter) -> None:
    missing = client.get("/automl/search-nada/job")
    search_id = client.post("/automl", json=REQUEST).json()["run_id"]
    finished(search_id, prefix="/automl", timeout=120)
    no_candidate = client.post(f"/automl/{search_id}/candidates/99/train")

    assert (missing.status_code, missing.json()["code"]) == (404, "search_not_found")
    assert (no_candidate.status_code, no_candidate.json()["code"]) == (404, "candidate_not_found")


def test_an_unsupervised_request_is_refused_before_starting(client: TestClient) -> None:
    response = client.post("/automl", json={**REQUEST, "task_type": "clustering"})

    assert response.status_code == 422
    assert "modo manual" in response.json()["message"]
