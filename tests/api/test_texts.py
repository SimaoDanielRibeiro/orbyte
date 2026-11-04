"""Os textos da interface pela API: o vocabulário e os temas do "Porquê?", pedidos uma vez."""

from fastapi.testclient import TestClient


def test_the_vocabulary_names_every_value(client: TestClient) -> None:
    vocabulary = client.get("/vocabulary").json()

    assert vocabulary["labels"]["TaskType"]["binary_classification"] == "Binary classification"
    assert vocabulary["labels"]["JobStatus"]["running"] == "Training"
    regression = next(t for t in vocabulary["tasks"] if t["task_type"] == "regression")
    assert regression["primary_metric"] == "r2"
    assert regression["suggested_model"] == "linear_regression"
    assert regression["agents"] is True


def test_every_topic_comes_at_once_or_one_by_one(client: TestClient) -> None:
    topics = client.get("/explanations").json()
    one = client.get("/explanations/automl").json()

    assert "automl" in topics
    assert "split" in topics
    assert topics["automl"] == one


def test_an_unknown_topic_is_404(client: TestClient) -> None:
    response = client.get("/explanations/nada")

    assert response.status_code == 404
    assert response.json()["code"] == "component_not_found"
