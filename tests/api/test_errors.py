"""Os erros da API: o código HTTP vem da categoria do erro, e a mensagem do serviço."""

from collections.abc import Callable
from typing import Any

import pytest
from fastapi.testclient import TestClient

from ml_lab.api.errors import error_code, status_for
from ml_lab.core.foundation.errors import (
    ColumnNotFoundError,
    HyperparameterError,
    MLLabError,
    RunNotFoundError,
    TrackingError,
    TrainingInProgressError,
)

type Waiter = Callable[[str], dict[str, Any]]


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (TrainingInProgressError("run-1"), 409),
        (ColumnNotFoundError("idade", ["age"]), 422),  # um DataError, embora também LookupError
        (HyperparameterError("knn", ["k tem de ser pelo menos 1."]), 422),
        (RunNotFoundError("run-1"), 404),
        (TrackingError("A base de dados é mais nova."), 500),
        (MLLabError("Outra coisa."), 400),
        (RuntimeError("Não é do ML Lab."), 500),
    ],
)
def test_each_category_of_error_has_its_status(error: Exception, status: int) -> None:
    assert status_for(error) == status


def test_the_code_is_the_name_of_the_error() -> None:
    assert error_code(RunNotFoundError("run-1")) == "run_not_found"
    assert error_code(TrainingInProgressError("run-1")) == "training_in_progress"


def test_unknown_identifiers_answer_404(client: TestClient) -> None:
    dataset = client.get("/datasets/nada")
    model = client.get("/catalog/models/logistic_regresion")
    run = client.get("/history/runs/run-nada")
    job = client.get("/experiments/run-nada/job")
    topic = client.get("/explanations/nada")

    assert [r.status_code for r in (dataset, model, run, job, topic)] == [404] * 5
    assert dataset.json()["code"] == "dataset_not_found"
    assert "Did you mean 'logistic_regression'?" in model.json()["message"]
    assert run.json()["code"] == "run_not_found"
    assert job.json()["code"] == "job_not_found"


def test_values_that_do_not_serve_answer_422_with_the_reason(
    client: TestClient, config: Callable[..., dict[str, Any]]
) -> None:
    hyperparameter = client.post("/experiments", json=config(learning_rate=-1))
    shape = client.post("/experiments", json={"dataset": "iris"})
    column = client.post("/experiments", json={**config(), "preparation": {"target": "especie"}})

    assert hyperparameter.status_code == 422
    assert hyperparameter.json()["code"] == "hyperparameter"
    assert "Learning rate must be at least" in hyperparameter.json()["message"]
    assert shape.status_code == 422
    assert shape.json()["code"] == "invalid_request"
    assert "model is missing a value." in shape.json()["problems"]
    assert column.status_code == 422
    assert column.json()["code"] == "column_not_found"


def test_a_question_that_does_not_fit_the_task_answers_400(
    client: TestClient, config: Callable[..., dict[str, Any]], finished: Waiter
) -> None:
    clustering = {
        **config("kmeans"),
        "preparation": {"target": "species", "task_type": "clustering"},
    }
    run_id = client.post("/experiments", json=clustering).json()["run_id"]
    finished(run_id)

    response = client.get(f"/analysis/{run_id}/cross-validation")

    assert response.status_code == 400
    assert response.json()["code"] == "model"
    assert "measures predictions against a target" in response.json()["message"]
