"""Gravador de snapshots: guarda o que um run anuncia, para o rever passo a passo.

Subscreve o ``EventBus`` e fica só com os eventos do seu run: todos os passos (a
curva da loss) e os snapshots amostrados (com o ``SnapshotSampler``, partilhado com o
contexto de treino, por isso nunca há mais do que o limite). É seguro entre fios: o
treino escreve num fio e a interface lê o progresso noutro.
"""

import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ml_lab.core.foundation.events import (
    Event,
    EventBus,
    SnapshotTaken,
    StepCompleted,
    Subscription,
)
from ml_lab.core.training.context import SnapshotSampler


@dataclass(frozen=True, slots=True)
class StepRecord:
    """Um passo do treino: o número, a loss e as métricas parciais."""

    step: int
    loss: float | None
    metrics: Mapping[str, float]


@dataclass(frozen=True, slots=True, eq=False)
class SnapshotRecord:
    """O estado do modelo num passo."""

    step: int
    state: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class Progress:
    """Onde vai o treino: último passo, última loss e se há snapshots."""

    steps: int
    last_loss: float | None
    snapshots: int


class SnapshotRecorder:
    """Guarda os passos e os snapshots de um run.

    Args:
        run_id: O run a gravar (os eventos de outros runs são ignorados).
        sampler: A mesma amostragem que o contexto de treino usa.
    """

    def __init__(self, run_id: str, sampler: SnapshotSampler | None = None) -> None:
        self.run_id = run_id
        self.sampler = sampler or SnapshotSampler()
        self._steps: list[StepRecord] = []
        self._snapshots: dict[int, SnapshotRecord] = {}
        self._lock = threading.Lock()
        self._subscription: Subscription[Event] | None = None

    def attach(self, bus: EventBus) -> None:
        """Começa a ouvir o ``bus``."""
        self._subscription = bus.subscribe(Event, self.receive)

    def detach(self, bus: EventBus) -> None:
        """Deixa de ouvir o ``bus``."""
        if self._subscription is not None:
            bus.unsubscribe(self._subscription)
            self._subscription = None

    def receive(self, event: Event) -> None:
        """Guarda o evento, se for deste run e for um passo ou um snapshot."""
        if event.run_id != self.run_id:
            return
        with self._lock:
            if isinstance(event, StepCompleted):
                self._steps.append(StepRecord(event.step, event.loss, dict(event.metrics)))
            elif isinstance(event, SnapshotTaken):
                self._snapshots[event.step] = SnapshotRecord(event.step, event.state)
                kept = set(self.sampler.thin(sorted(self._snapshots)))
                kept.add(event.step)  # o mais recente fica sempre (pode ser o do fim)
                self._snapshots = {s: r for s, r in self._snapshots.items() if s in kept}

    def steps(self) -> tuple[StepRecord, ...]:
        """Todos os passos, pela ordem."""
        with self._lock:
            return tuple(self._steps)

    def snapshots(self) -> tuple[SnapshotRecord, ...]:
        """Os snapshots guardados, por ordem de passo."""
        with self._lock:
            return tuple(self._snapshots[step] for step in sorted(self._snapshots))

    def progress(self) -> Progress:
        """O progresso até agora."""
        with self._lock:
            last = self._steps[-1] if self._steps else None
            return Progress(
                steps=last.step if last else 0,
                last_loss=last.loss if last else None,
                snapshots=len(self._snapshots),
            )
