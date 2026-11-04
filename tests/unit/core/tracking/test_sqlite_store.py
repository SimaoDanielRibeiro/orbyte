"""O registo em SQLite: o esquema com versão, a persistência e o uso em vários fios."""

import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ml_lab.core.foundation.errors import TrackingError
from ml_lab.core.foundation.types import Partition, TaskType
from ml_lab.core.tracking.sqlite_store import MIGRATIONS, SqliteStore
from ml_lab.core.tracking.store import RunEntry


def entry(run_id: str) -> RunEntry:
    return RunEntry(
        run_id=run_id,
        created_at=datetime(2026, 10, 24, tzinfo=UTC),
        name=run_id,
        dataset_name="Iris",
        task_type=TaskType.CLUSTERING,
        model_name="kmeans",
        model_display_name="K-means",
        seed=1,
        partition=Partition.TEST,
        primary_metric="silhouette",
        metrics={"silhouette": 0.5, "n_clusters": 3.0},
    )


def test_a_new_database_gets_the_latest_schema(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "pasta" / "registo.db")

    assert store.version == len(MIGRATIONS)
    assert (tmp_path / "pasta" / "registo.db").is_file()


def test_the_runs_survive_closing_and_reopening(tmp_path: Path) -> None:
    path = tmp_path / "registo.db"
    SqliteStore(path).add(entry("run-a"), {"x": 1})

    reopened = SqliteStore(path)  # como fechar a app e voltar a abrir

    assert reopened.entry("run-a").metrics == {"silhouette": 0.5, "n_clusters": 3.0}
    assert reopened.document("run-a") == {"x": 1}


def test_the_metrics_table_answers_questions_in_sql(tmp_path: Path) -> None:
    path = tmp_path / "registo.db"
    store = SqliteStore(path)
    store.add(entry("run-a"), {})

    with sqlite3.connect(path) as connection:
        rows = connection.execute(
            "SELECT run_id, value FROM metrics WHERE name = 'silhouette'"
        ).fetchall()

    assert rows == [("run-a", 0.5)]


def test_a_database_from_a_newer_version_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "registo.db"
    with sqlite3.connect(path) as connection:
        connection.execute(f"PRAGMA user_version = {len(MIGRATIONS) + 1}")

    with pytest.raises(TrackingError, match="update Orbyte"):
        SqliteStore(path)


def test_runs_can_be_recorded_from_another_thread(tmp_path: Path) -> None:
    # O treino corre noutro fio de execução, e é de lá que o registo é chamado.
    store = SqliteStore(tmp_path / "registo.db")
    errors: list[BaseException] = []

    def record(run_id: str) -> None:
        try:
            store.add(entry(run_id), {})
        except BaseException as error:  # o teste quer ver qualquer falha do outro fio
            errors.append(error)

    threads = [threading.Thread(target=record, args=(f"run-{i}",)) for i in range(5)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    assert len(store.entries()) == 5
