"""Os agentes pela API: pedir o melhor modelo, seguir a conversa, treinar um candidato à mão e
perguntar ao analista sobre a procura acabada."""

import socket
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from ml_lab.api.app import create_app
from ml_lab.config import Settings
from ml_lab.core.services import AutoMLRequest, Lab
from ml_lab.core.tracking.store import InMemoryStore

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
    assert "manual mode" in clustering["reason"]


def test_the_plan_shows_what_the_data_agent_decided(client: TestClient) -> None:
    plan = client.post("/automl/plan", json=REQUEST).json()

    assert plan["task_type"] == "multiclass_classification"
    assert plan["decisions"][0]["title"] == "Drop duplicate rows"
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
    assert entry["name"].startswith("Agents · ")


def test_the_plan_of_a_search_is_there_while_it_runs_and_after(client: TestClient) -> None:
    search_id = client.post("/automl", json=REQUEST).json()["run_id"]

    plan = client.get(f"/automl/{search_id}/plan").json()  # logo a seguir: ainda decorre

    assert plan["target"] == "species"
    assert [agent["family"] for agent in plan["agents"]] == ["linear", "instance_based", "tree"]
    assert client.get("/automl/search-nada/plan").status_code == 404


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
    assert later["working"] is None  # acabou: nenhum agente está a medir um candidato


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
    assert "manual mode" in response.json()["message"]


# --- o analista (fase 13, etapa 13.3) ------------------------------------------------


def saved_replies(winner: int, second: int) -> list[dict[str, Any]]:
    """As respostas de um modelo, em JSON: o quadro, a comparação e a resposta."""
    return [
        {"thinking": "The board first.", "tool_calls": [{"name": "list_candidates"}]},
        {
            "thinking": "The winner against the second.",
            "tool_calls": [
                {"name": "compare_candidates", "arguments": {"first": winner, "second": second}}
            ],
        },
        {"thinking": "Enough.", "content": f"#{winner} is first on the board."},
    ]


def finished_search(client: TestClient, finished: Waiter) -> tuple[str, list[int]]:
    search_id: str = client.post("/automl", json=REQUEST).json()["run_id"]
    finished(search_id, prefix="/automl", timeout=120)
    ranking: list[int] = client.get(f"/automl/{search_id}/result").json()["ranking"]
    return search_id, ranking


def test_the_analyst_answers_with_each_tool_it_used(client: TestClient, finished: Waiter) -> None:
    search_id, ranking = finished_search(client, finished)
    options = client.get(f"/automl/{search_id}/analyst").json()

    asked = client.post(
        f"/automl/{search_id}/questions",
        json={"question": options["suggestions"][0], "replies": saved_replies(*ranking[:2])},
    )

    assert options["suggestions"][0] == "Why did the winner win?"
    assert asked.status_code == 202
    question_id = asked.json()["run_id"]
    assert question_id.startswith("question-")
    assert finished(question_id, prefix="/automl/questions", timeout=60)["status"] == "finished"
    answer = client.get(f"/automl/questions/{question_id}/result").json()
    assert [step["kind"] for step in answer["trace"]] == [
        "thought",
        "tool",
        "thought",
        "tool",
        "thought",
        "answer",
    ]
    tools = [step["tool"] for step in answer["trace"] if step["tool"] is not None]
    assert [tool["label"] for tool in tools] == ["The candidate board", "Compare two candidates"]
    assert tools[1]["arguments"] == {"first": ranking[0], "second": ranking[1]}
    assert answer["answer"] == f"#{ranking[0]} is first on the board."
    assert answer["request"]["replies"][1]["tool_calls"][0]["name"] == "compare_candidates"
    asked_before = client.get(f"/automl/{search_id}/questions").json()
    assert [one["question_id"] for one in asked_before] == [question_id]


def test_a_question_streams_its_steps_until_the_answer(
    client: TestClient, finished: Waiter
) -> None:
    search_id, ranking = finished_search(client, finished)
    question_id = client.post(
        f"/automl/{search_id}/questions",
        json={"question": "Why?", "replies": saved_replies(*ranking[:2])},
    ).json()["run_id"]
    updates = []

    with client.websocket_connect(f"/automl/questions/{question_id}/stream") as websocket:
        while True:
            message = websocket.receive_json()
            updates.append(message)
            if message["data"]["state"]["status"] in {"finished", "failed"}:
                break

    numbers = [step["number"] for update in updates for step in update["data"]["trace"]]
    answer = client.get(f"/automl/questions/{question_id}/result").json()
    assert numbers == [step["number"] for step in answer["trace"]]  # nem um passo perdido
    assert updates[-1]["data"]["question"] == "Why?"
    later = client.get(f"/automl/questions/{question_id}/updates", params={"after_step": 4}).json()
    assert [step["number"] for step in later["trace"]] == [5, 6]


def test_unknown_questions_are_404_and_a_running_search_is_409(client: TestClient) -> None:
    missing = client.get("/automl/questions/question-nada/result")
    running = client.post(
        "/automl", json={**REQUEST, "settings": {"budget": 200, "folds": 3}}
    ).json()["run_id"]
    client.post(f"/automl/{running}/pause")

    asked = client.post(f"/automl/{running}/questions", json={"question": "Why?"})

    client.post(f"/automl/{running}/cancel")
    assert (missing.status_code, missing.json()["code"]) == (404, "question_not_found")
    assert (asked.status_code, asked.json()["code"]) == (409, "training_in_progress")
    empty = client.post(f"/automl/{running}/questions", json={"question": ""})
    assert (empty.status_code, empty.json()["code"]) == (422, "invalid_request")


def test_without_the_ollama_a_question_says_how_to_start_it(tmp_path: Path) -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        closed = f"http://127.0.0.1:{probe.getsockname()[1]}"
    settings = Settings.from_env({"ML_LAB_OLLAMA_URL": closed}, home=tmp_path)
    lab = Lab.create(settings, store=InMemoryStore())
    result = lab.automl.run(AutoMLRequest.model_validate(REQUEST))

    with TestClient(create_app(lab)) as client:
        asked = client.post(f"/automl/{result.search_id}/questions", json={"question": "Why?"})
        status = client.get("/llm/status").json()

    assert (asked.status_code, asked.json()["code"]) == (400, "language_model")
    assert "ollama serve" in asked.json()["message"]
    assert not status["ready"]  # o painel mostra este estado, com o que fazer
