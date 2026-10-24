"""O registo de experiências: o contrato ``ExperimentStore`` e a versão em memória.

Cada run que termina fica registado (ADR 0020) com duas partes:

- uma **entrada** para o histórico (``RunEntry``): o que se mostra numa tabela, como o
  modelo, o dataset e as métricas;
- o **documento**: o resultado completo do run em JSON, o mesmo que os serviços devolvem
  (ADR 0019). O registo não sabe o que o documento tem: guarda-o e devolve-o.

Há duas implementações do mesmo contrato (regra 6 dos contratos): esta, em memória, para
os testes, e a ``SqliteStore``, que guarda num ficheiro e sobrevive a fechar a app. Os
testes de contrato (``tests/contract/test_stores.py``) correm nas duas.
"""

import json
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Protocol

from ml_lab.core.foundation.errors import RunNotFoundError
from ml_lab.core.foundation.types import Partition, TaskType


@dataclass(frozen=True, slots=True)
class RunEntry:
    """Um run do registo, como aparece no histórico.

    Attributes:
        run_id: O identificador do run.
        created_at: Quando terminou (UTC).
        name: Um nome para o reconhecer; por omissão, o modelo e o dataset.
        dataset_name: O dataset.
        task_type: O tipo de tarefa.
        model_name: O modelo, como está no registo ``MODELS``.
        model_display_name: O nome a mostrar do modelo.
        seed: A seed raiz (com ela e a configuração, o run repete-se).
        partition: A parte dos dados das métricas (o teste; o treino num modelo que só
            conhece os pontos com que treinou, como o t-SNE).
        primary_metric: A métrica principal da tarefa.
        metrics: O valor de cada métrica nessa parte (``None`` se ficou indefinida).
        primary_label: O nome a mostrar da métrica principal.
        primary_as_percent: Se a métrica principal se mostra como percentagem.
        saved: Se o modelo treinado e o treino passo a passo também foram guardados.
    """

    run_id: str
    created_at: datetime
    name: str
    dataset_name: str
    task_type: TaskType
    model_name: str
    model_display_name: str
    seed: int
    partition: Partition
    primary_metric: str | None
    metrics: Mapping[str, float | None] = field(default_factory=dict)
    primary_label: str | None = None
    primary_as_percent: bool = False
    saved: bool = False

    @property
    def primary_value(self) -> float | None:
        """O valor da métrica principal, se houver."""
        return None if self.primary_metric is None else self.metrics.get(self.primary_metric)


class ExperimentStore(Protocol):
    """O que um registo de experiências sabe fazer."""

    def add(self, entry: RunEntry, document: Mapping[str, Any]) -> None:
        """Regista um run (ou substitui o que tiver o mesmo ``run_id``)."""

    def entries(self) -> tuple[RunEntry, ...]:
        """Todos os runs, do mais recente para o mais antigo."""

    def entry(self, run_id: str) -> RunEntry:
        """A entrada de um run.

        Raises:
            RunNotFoundError: Se o run não estiver no registo.
        """

    def document(self, run_id: str) -> dict[str, Any]:
        """O resultado completo de um run, em valores de JSON.

        Raises:
            RunNotFoundError: Se o run não estiver no registo.
        """

    def update(
        self, run_id: str, *, name: str | None = None, saved: bool | None = None
    ) -> RunEntry:
        """Muda o nome ou a marca de guardado de um run; devolve a entrada nova.

        Raises:
            RunNotFoundError: Se o run não estiver no registo.
        """

    def delete(self, run_id: str) -> None:
        """Apaga um run (não faz nada se ele já não existir)."""


class InMemoryStore:
    """Registo em memória, para os testes: desaparece quando o processo acaba.

    Os documentos ficam em texto JSON, como na base de dados: quem os lê recebe sempre
    uma cópia nova, e um documento que não passe a JSON falha logo ao registar.
    """

    def __init__(self) -> None:
        self._runs: dict[str, tuple[RunEntry, str]] = {}
        self._lock = threading.Lock()

    def add(self, entry: RunEntry, document: Mapping[str, Any]) -> None:
        """Regista um run (ou substitui o que tiver o mesmo ``run_id``)."""
        text = json.dumps(document, allow_nan=False)
        with self._lock:
            self._runs[entry.run_id] = (entry, text)

    def entries(self) -> tuple[RunEntry, ...]:
        """Todos os runs, do mais recente para o mais antigo."""
        with self._lock:
            found = [entry for entry, _ in self._runs.values()]
        return tuple(sorted(found, key=lambda entry: entry.created_at, reverse=True))

    def entry(self, run_id: str) -> RunEntry:
        """A entrada de um run (``RunNotFoundError`` se não existir)."""
        return self._get(run_id)[0]

    def document(self, run_id: str) -> dict[str, Any]:
        """O resultado completo de um run (``RunNotFoundError`` se não existir)."""
        document: dict[str, Any] = json.loads(self._get(run_id)[1])
        return document

    def update(
        self, run_id: str, *, name: str | None = None, saved: bool | None = None
    ) -> RunEntry:
        """Muda o nome ou a marca de guardado de um run; devolve a entrada nova."""
        with self._lock:
            if run_id not in self._runs:
                raise RunNotFoundError(run_id)
            entry, text = self._runs[run_id]
            updated = replace(
                entry,
                name=entry.name if name is None else name,
                saved=entry.saved if saved is None else saved,
            )
            self._runs[run_id] = (updated, text)
        return updated

    def delete(self, run_id: str) -> None:
        """Apaga um run (não faz nada se ele já não existir)."""
        with self._lock:
            self._runs.pop(run_id, None)

    def _get(self, run_id: str) -> tuple[RunEntry, str]:
        with self._lock:
            found = self._runs.get(run_id)
        if found is None:
            raise RunNotFoundError(run_id)
        return found
