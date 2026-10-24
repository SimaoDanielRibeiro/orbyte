"""O registo de experiências num ficheiro SQLite (o módulo ``sqlite3`` da biblioteca padrão).

Duas tabelas (ADR 0020):

- ``runs``: uma linha por run, com o que o histórico mostra e o documento completo (o
  resultado do run em JSON);
- ``metrics``: uma linha por métrica de cada run. Assim, o histórico ordena-se e filtra-se
  por qualquer métrica sem abrir os documentos, por exemplo
  ``SELECT run_id FROM metrics WHERE name = 'accuracy' ORDER BY value DESC``.

O esquema tem um número de versão (``PRAGMA user_version``). Ao abrir uma base de dados
antiga, correm por ordem as migrações que lhe faltam; uma versão nova do esquema é uma
migração nova no fim de ``MIGRATIONS``, e nunca uma alteração das que já existem.

Cada operação abre a sua própria ligação: o treino corre noutro fio de execução, e uma
ligação do ``sqlite3`` não se partilha entre fios. Abrir uma ligação a um ficheiro local
custa pouco.
"""

import json
import sqlite3
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Final

from ml_lab.core.foundation.errors import RunNotFoundError, TrackingError
from ml_lab.core.foundation.types import Partition, TaskType
from ml_lab.core.tracking.store import RunEntry

MIGRATIONS: Final = (
    # Versão 1: os runs e as suas métricas.
    """
    CREATE TABLE runs (
        run_id TEXT PRIMARY KEY,
        created_at TEXT NOT NULL,
        name TEXT NOT NULL,
        dataset_name TEXT NOT NULL,
        task_type TEXT NOT NULL,
        model_name TEXT NOT NULL,
        model_display_name TEXT NOT NULL,
        seed INTEGER NOT NULL,
        partition TEXT NOT NULL,
        primary_metric TEXT,
        primary_label TEXT,
        primary_as_percent INTEGER NOT NULL DEFAULT 0,
        saved INTEGER NOT NULL DEFAULT 0,
        document TEXT NOT NULL
    );
    CREATE INDEX runs_by_date ON runs (created_at);
    CREATE TABLE metrics (
        run_id TEXT NOT NULL REFERENCES runs (run_id) ON DELETE CASCADE,
        name TEXT NOT NULL,
        value REAL,
        PRIMARY KEY (run_id, name)
    );
    """,
)
"""As migrações do esquema, pela ordem: a migração ``i`` leva a base de dados à versão ``i + 1``."""

_COLUMNS: Final = (
    "run_id, created_at, name, dataset_name, task_type, model_name, model_display_name, "
    "seed, partition, primary_metric, primary_label, primary_as_percent, saved"
)


class SqliteStore:
    """Registo de experiências num ficheiro SQLite.

    Args:
        path: O ficheiro da base de dados; é criado (com as pastas) se não existir.

    Raises:
        TrackingError: Se a base de dados for de uma versão mais recente do ML Lab.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            _migrate(connection)

    def add(self, entry: RunEntry, document: Mapping[str, Any]) -> None:
        """Regista um run (ou substitui o que tiver o mesmo ``run_id``)."""
        text = json.dumps(document, allow_nan=False)
        with self._connect() as connection:
            # Apagar primeiro leva também as métricas antigas (ON DELETE CASCADE).
            connection.execute("DELETE FROM runs WHERE run_id = ?", (entry.run_id,))
            connection.execute(
                f"INSERT INTO runs ({_COLUMNS}, document) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    entry.run_id,
                    entry.created_at.isoformat(),
                    entry.name,
                    entry.dataset_name,
                    entry.task_type.value,
                    entry.model_name,
                    entry.model_display_name,
                    entry.seed,
                    entry.partition.value,
                    entry.primary_metric,
                    entry.primary_label,
                    int(entry.primary_as_percent),
                    int(entry.saved),
                    text,
                ),
            )
            connection.executemany(
                "INSERT INTO metrics (run_id, name, value) VALUES (?, ?, ?)",
                [(entry.run_id, name, value) for name, value in entry.metrics.items()],
            )

    def entries(self) -> tuple[RunEntry, ...]:
        """Todos os runs, do mais recente para o mais antigo."""
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT {_COLUMNS} FROM runs ORDER BY created_at DESC"
            ).fetchall()
            metrics = connection.execute(
                "SELECT run_id, name, value FROM metrics ORDER BY rowid"
            ).fetchall()
        values: dict[str, dict[str, float | None]] = {}
        for metric in metrics:
            values.setdefault(metric["run_id"], {})[metric["name"]] = metric["value"]
        return tuple(_entry(row, values.get(row["run_id"], {})) for row in rows)

    def entry(self, run_id: str) -> RunEntry:
        """A entrada de um run (``RunNotFoundError`` se não existir)."""
        with self._connect() as connection:
            row = connection.execute(
                f"SELECT {_COLUMNS} FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            metrics = connection.execute(
                "SELECT name, value FROM metrics WHERE run_id = ? ORDER BY rowid", (run_id,)
            ).fetchall()
        if row is None:
            raise RunNotFoundError(run_id)
        return _entry(row, {metric["name"]: metric["value"] for metric in metrics})

    def document(self, run_id: str) -> dict[str, Any]:
        """O resultado completo de um run (``RunNotFoundError`` se não existir)."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT document FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        if row is None:
            raise RunNotFoundError(run_id)
        document: dict[str, Any] = json.loads(row["document"])
        return document

    def update(
        self, run_id: str, *, name: str | None = None, saved: bool | None = None
    ) -> RunEntry:
        """Muda o nome ou a marca de guardado de um run; devolve a entrada nova."""
        current = self.entry(run_id)
        updated = replace(
            current,
            name=current.name if name is None else name,
            saved=current.saved if saved is None else saved,
        )
        with self._connect() as connection:
            connection.execute(
                "UPDATE runs SET name = ?, saved = ? WHERE run_id = ?",
                (updated.name, int(updated.saved), run_id),
            )
        return updated

    def delete(self, run_id: str) -> None:
        """Apaga um run e as suas métricas (não faz nada se ele já não existir)."""
        with self._connect() as connection:
            connection.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))

    @property
    def version(self) -> int:
        """A versão do esquema da base de dados."""
        with self._connect() as connection:
            version: int = connection.execute("PRAGMA user_version").fetchone()[0]
        return version

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Uma ligação nova, numa transação: confirma no fim, ou desfaz tudo se houver um erro.

        Raises:
            TrackingError: Se a base de dados falhar (ficheiro ilegível, disco cheio…).
        """
        try:
            connection = sqlite3.connect(self.path, timeout=10)
        except sqlite3.Error as error:
            raise TrackingError(f"Não foi possível abrir {self.path}: {error}") from error
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA foreign_keys = ON")  # o SQLite só as cumpre se se pedir
            with connection:
                yield connection
        except sqlite3.Error as error:
            raise TrackingError(f"A base de dados de experiências falhou: {error}") from error
        finally:
            connection.close()


def _migrate(connection: sqlite3.Connection) -> None:
    """Corre as migrações que faltam, pela ordem, e regista a versão a que se chegou."""
    version: int = connection.execute("PRAGMA user_version").fetchone()[0]
    if version > len(MIGRATIONS):
        raise TrackingError(
            f"A base de dados está na versão {version} do esquema, e este ML Lab só conhece até "
            f"à {len(MIGRATIONS)}: atualiza o ML Lab ou usa outra base de dados."
        )
    for number, script in enumerate(MIGRATIONS[version:], start=version + 1):
        connection.executescript(script)
        connection.execute(f"PRAGMA user_version = {number}")


def _entry(row: sqlite3.Row, metrics: Mapping[str, float | None]) -> RunEntry:
    return RunEntry(
        run_id=row["run_id"],
        created_at=datetime.fromisoformat(row["created_at"]),
        name=row["name"],
        dataset_name=row["dataset_name"],
        task_type=TaskType(row["task_type"]),
        model_name=row["model_name"],
        model_display_name=row["model_display_name"],
        seed=row["seed"],
        partition=Partition(row["partition"]),
        primary_metric=row["primary_metric"],
        metrics=dict(metrics),
        primary_label=row["primary_label"],
        primary_as_percent=bool(row["primary_as_percent"]),
        saved=bool(row["saved"]),
    )
