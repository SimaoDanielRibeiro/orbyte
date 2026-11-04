"""A demonstração de um modelo pela API: o pedido, a resposta e os erros."""

from fastapi.testclient import TestClient


def test_a_demo_answers_with_the_example_and_what_the_model_learned(client: TestClient) -> None:
    response = client.post(
        "/catalog/models/decision_tree/demo", json={"hyperparameters": {"max_depth": 3}}
    )

    assert response.status_code == 200
    demo = response.json()
    assert demo["example"]["name"] == "moons"
    assert demo["hyperparameters"]["max_depth"] == 3
    assert demo["learned"]["tree"][0]["node"] == 0
    assert len(demo["animation"]["frames"]) > 1
    assert set(demo["reports"]) == {"train", "test"}


def test_the_body_can_be_empty(client: TestClient) -> None:
    response = client.post("/catalog/models/kmeans/demo", json={})

    assert response.status_code == 200
    assert response.json()["task_type"] == "clustering"


def test_an_example_of_another_task_is_a_422_with_the_choices(client: TestClient) -> None:
    response = client.post("/catalog/models/knn/demo", json={"example": "wave"})

    assert response.status_code == 422
    assert response.json()["code"] == "invalid_parameter"
    assert "moons" in response.json()["message"]


def test_a_misspelled_field_is_refused(client: TestClient) -> None:
    response = client.post("/catalog/models/knn/demo", json={"exemplo": "moons"})

    assert response.status_code == 422
    assert response.json()["code"] == "invalid_request"


def test_the_metrics_of_a_task_come_with_their_formulas(client: TestClient) -> None:
    response = client.get("/catalog/metrics", params={"task_type": "binary_classification"})

    assert response.status_code == 200
    cards = {card["name"]: card for card in response.json()}
    assert "precision" in cards
    assert "r2" not in cards
    assert cards["precision"]["formula"].startswith(r"\text{precision}")
    assert cards["precision"]["symbols"][0]["symbol"] == r"\mathrm{TP}"


def test_a_demo_report_brings_the_terms_of_each_formula(client: TestClient) -> None:
    demo = client.post("/catalog/models/logistic_regression/demo", json={}).json()

    metrics = {metric["name"]: metric for metric in demo["reports"]["test"]["metrics"]}
    symbols = [term["symbol"] for term in metrics["precision"]["terms"]]
    assert symbols == [r"\mathrm{TP}", r"\mathrm{FP}"]
    assert metrics["roc_auc"]["terms"] == []


def test_an_unknown_model_is_a_404(client: TestClient) -> None:
    response = client.post("/catalog/models/knnn/demo", json={})

    assert response.status_code == 404
    assert response.json()["code"] == "component_not_found"
