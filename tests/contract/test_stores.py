"""Contrato do registo de experiências: as duas implementações comportam-se da mesma maneira.

A regra 6 dos contratos diz que o registo é trocável (em memória, SQLite e, um dia,
MLflow) sem mudar quem o usa. Estes testes correm iguais em todas as implementações.
"""

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from ml_lab.core.foundation.errors import RunNotFoundError
from ml_lab.core.foundation.types import Partition, TaskType
from ml_lab.core.tracking.sqlite_store import SqliteStore
from ml_lab.core.tracking.store import ExperimentStore, InMemoryStore, RunEntry

START = datetime(2026, 10, 24, 12, 0, tzinfo=UTC)


def entry(run_id: str, minutes: int = 0, **changes: object) -> RunEntry:
    values: dict[str, object] = {
        "run_id": run_id,
        "created_at": START + timedelta(minutes=minutes),
        "name": f"Run {run_id}",
        "dataset_name": "Iris",
        "task_type": TaskType.MULTICLASS_CLASSIFICATION,
        "model_name": "logistic_regression",
        "model_display_name": "Regressão logística",
        "seed": 42,
        "partition": Partition.TEST,
        "primary_metric": "accuracy",
        "metrics": {"accuracy": 0.9, "roc_auc": None},
        "primary_label": "Exatidão",
        "primary_as_percent": True,
    }
    values.update(changes)
    return RunEntry(**values)  # type: ignore[arg-type]


STORES: dict[str, Callable[[Path], ExperimentStore]] = {
    "memória": lambda _: InMemoryStore(),
    "sqlite": lambda folder: SqliteStore(folder / "registo.db"),
}


@pytest.fixture(params=list(STORES), ids=list(STORES))
def store(request: pytest.FixtureRequest, tmp_path: Path) -> ExperimentStore:
    return STORES[request.param](tmp_path)


def test_a_run_comes_back_as_it_went_in(store: ExperimentStore) -> None:
    document = {"config": {"seed": 42}, "values": [1.5, None, "texto"]}

    store.add(entry("run-a"), document)

    assert store.entry("run-a") == entry("run-a")
    assert store.document("run-a") == document
    assert store.entry("run-a").primary_value == 0.9


def test_the_history_starts_with_the_most_recent_run(store: ExperimentStore) -> None:
    store.add(entry("run-old", minutes=0), {})
    store.add(entry("run-new", minutes=5), {})
    store.add(entry("run-middle", minutes=2), {})

    assert [run.run_id for run in store.entries()] == ["run-new", "run-middle", "run-old"]


def test_adding_the_same_run_again_replaces_it(store: ExperimentStore) -> None:
    store.add(entry("run-a"), {"version": 1})
    store.add(entry("run-a", metrics={"accuracy": 0.5}), {"version": 2})

    assert len(store.entries()) == 1
    assert store.entry("run-a").metrics == {"accuracy": 0.5}
    assert store.document("run-a") == {"version": 2}


def test_a_run_can_be_renamed_and_marked_as_saved(store: ExperimentStore) -> None:
    store.add(entry("run-a"), {})

    renamed = store.update("run-a", name="O melhor")
    saved = store.update("run-a", saved=True)

    assert renamed.name == "O melhor"
    assert (saved.name, saved.saved) == ("O melhor", True)
    assert store.entry("run-a") == saved


def test_deleting_removes_the_run_and_its_metrics(store: ExperimentStore) -> None:
    store.add(entry("run-a"), {})
    store.add(entry("run-b"), {})

    store.delete("run-a")
    store.delete("run-a")  # apagar outra vez não é um erro

    assert [run.run_id for run in store.entries()] == ["run-b"]
    with pytest.raises(RunNotFoundError, match="run-a"):
        store.entry("run-a")


def test_an_unknown_run_is_a_clear_error(store: ExperimentStore) -> None:
    with pytest.raises(RunNotFoundError, match="run-x"):
        store.document("run-x")
    with pytest.raises(RunNotFoundError):
        store.update("run-x", name="nome")


def test_documents_must_be_json(store: ExperimentStore) -> None:
    # O NaN não existe em JSON: um documento com ele seria ilegível noutras ferramentas.
    with pytest.raises(ValueError, match="JSON"):
        store.add(entry("run-a"), {"loss": float("nan")})
