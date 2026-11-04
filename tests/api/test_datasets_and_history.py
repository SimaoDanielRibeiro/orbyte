"""Dados e histórico pela API: carregar, explorar, preparar, guardar, comparar e reproduzir."""

from collections.abc import Callable
from typing import Any

from fastapi.testclient import TestClient

type Waiter = Callable[[str], dict[str, Any]]
type Config = Callable[..., dict[str, Any]]

CSV = b"altura,peso,classe\n1.6,60,a\n1.7,72,b\n1.8,80,b\n1.5,50,a\n1.65,65,a\n1.75,77,b\n"


def test_an_uploaded_csv_is_loaded_explored_and_prepared(client: TestClient) -> None:
    uploaded = client.post("/datasets/uploads", files={"file": ("pessoas.csv", CSV, "text/csv")})

    assert uploaded.status_code == 200
    dataset = uploaded.json()
    dataset_id = dataset["id"]
    assert (dataset["name"], dataset["n_rows"]) == ("pessoas.csv", 6)
    assert [c["name"] for c in dataset["columns"]] == ["altura", "peso", "classe"]
    assert client.get("/datasets").json()[0]["id"] == dataset_id
    assert len(client.get(f"/datasets/{dataset_id}/preview", params={"n_rows": 2}).json()["rows"])
    suggestion = client.get(f"/datasets/{dataset_id}/task-suggestion", params={"target": "classe"})
    assert suggestion.json()["task_type"] == "binary_classification"
    prepared = client.post(
        f"/datasets/{dataset_id}/preparation",
        json={"config": {"target": "classe", "split": {"test_size": 0.34}}, "seed": 1},
    )
    assert prepared.status_code == 200
    assert prepared.json()["task_type"] == "binary_classification"


def test_samples_and_generators_are_listed_and_loaded(client: TestClient) -> None:
    samples = client.get("/datasets/samples").json()
    moons = client.post(
        "/datasets/generated",
        json={"generator": "moons", "parameters": {"n_samples": 100}, "seed": 3},
    )

    assert "iris" in [sample["name"] for sample in samples]
    assert moons.status_code == 200
    assert moons.json()["n_rows"] == 100


def test_the_catalog_answers_for_the_model_screen(client: TestClient) -> None:
    cards = client.get("/catalog/models", params={"task_type": "regression"}).json()
    suggested = client.get("/catalog/suggested-model", params={"task_type": "regression"})
    validated = client.post("/catalog/models/knn/hyperparameters", json={"n_neighbors": 7})

    assert all("regression" in card["task_types"] for card in cards)
    assert suggested.json() == "linear_regression"
    assert validated.json()["n_neighbors"] == 7


def test_the_history_saves_compares_reproduces_and_deletes(
    client: TestClient, config: Config, finished: Waiter
) -> None:
    first = client.post("/experiments", json=config(learning_rate=0.5)).json()["run_id"]
    second = client.post("/experiments", json=config(learning_rate=0.05)).json()["run_id"]
    finished(first)
    finished(second)

    saved = client.post(f"/history/runs/{first}/save", json={"name": "Taxa alta"}).json()
    renamed = client.patch(f"/history/runs/{second}", json={"name": "Taxa baixa"}).json()
    comparison = client.post("/history/comparison", json={"run_ids": [first, second]}).json()
    reproduced = client.post(f"/history/runs/{first}/reproduction")
    opened = client.get(f"/history/runs/{first}/result").json()

    assert (saved["saved"], saved["name"]) == (True, "Taxa alta")
    assert renamed["name"] == "Taxa baixa"
    assert [d["label"] for d in comparison["differences"]] == ["Hyperparameter · Learning rate"]
    assert reproduced.status_code == 202
    assert finished(reproduced.json()["run_id"])["status"] == "finished"
    assert opened["run_id"] == first
    runs = [entry["run_id"] for entry in client.get("/history/runs").json()]
    assert {first, second, reproduced.json()["run_id"]} <= set(runs)
    assert client.delete(f"/history/runs/{second}").status_code == 204
    assert client.get(f"/history/runs/{second}").status_code == 404


def test_the_analysis_answers_for_a_recent_run(
    client: TestClient, config: Config, finished: Waiter
) -> None:
    run_id = client.post("/experiments", json=config()).json()["run_id"]
    finished(run_id)

    charts = client.get(f"/analysis/{run_id}/charts")
    rows = client.get(f"/analysis/{run_id}/test-rows").json()
    explanation = client.get(f"/analysis/{run_id}/test-rows/0/explanation")

    assert charts.status_code == 200
    assert charts.json()["roc"]
    assert rows == 30
    assert explanation.json()["contributions"]


def test_the_laboratory_and_the_why_panel_answer(client: TestClient) -> None:
    knn = client.get("/laboratory/knn", params={"k": 5})
    why = client.get("/explanations/split")

    assert knn.json()["k"] == 5
    assert why.json()["title"]


def test_the_laboratory_says_what_each_control_offers(client: TestClient) -> None:
    demos = client.get("/laboratory/demos").json()
    degree = demos[0]["control"]
    too_far = client.get(
        "/laboratory/overfitting", params={"degree": int(degree["values"][-1]) + 1}
    )

    assert [demo["name"] for demo in demos] == [
        "overfitting",
        "learning_rate",
        "knn",
        "regularization",
        "threshold",
        "data_size",
    ]
    assert degree["default"] in degree["values"]
    # Um valor fora do que a demonstração compara é um pedido que não serve, explicado.
    assert too_far.status_code == 422
    assert too_far.json()["code"] == "invalid_parameter"
