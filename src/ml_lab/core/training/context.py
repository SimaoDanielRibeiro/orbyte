"""Contexto de treino: o canal entre o modelo e o exterior durante o treino.

O modelo nunca sabe quem o está a observar (regra 1 dos contratos). Durante o
``fit``, só fala com o contexto: anuncia cada passo (época, loss, métricas), entrega
fotografias do seu estado quando o contexto as quer e pergunta se deve parar.

Há três contextos:

- ``NullTrainingContext``: ignora tudo (quando ninguém está a ver);
- ``RecordingTrainingContext``: guarda tudo em listas (testes e notebooks);
- ``EventTrainingContext``: publica cada passo e cada snapshot no ``EventBus``, com
  amostragem dos snapshots e controlo de pausa e paragem. É o que o Trainer usa.

A **sonda** (``probe``) são pontos onde quem observa quer ver as previsões do modelo
a cada snapshot: por exemplo, uma grelha 2D para desenhar a fronteira de decisão a
mexer-se. Com ela, qualquer modelo mostra a fronteira, sem o contexto ter de saber
como o modelo funciona por dentro (ADR 0013).
"""

from collections.abc import Mapping
from typing import Any, Final, Protocol, override

import numpy as np
from numpy.typing import NDArray

from ml_lab.core.foundation.events import EventBus, SnapshotTaken, StepCompleted
from ml_lab.core.training.control import TrainingControl

DEFAULT_SNAPSHOT_LIMIT: Final = 60
"""Snapshots guardados, no máximo, por treino (mais um, o do fim)."""


class TrainingContext(Protocol):
    """O que um modelo pode fazer durante o treino.

    Attributes:
        probe: Pontos (no espaço das features) onde se querem as previsões em cada
            snapshot, ou ``None``.
    """

    probe: NDArray[np.float64] | None

    def report_step(
        self, step: int, *, loss: float | None = None, metrics: Mapping[str, float] | None = None
    ) -> None:
        """Anuncia que terminou o passo ``step`` (a começar em 1)."""

    def wants_snapshot(self, step: int) -> bool:
        """Se quer um snapshot neste passo (o estado custa a calcular; só quando serve)."""

    def report_snapshot(self, step: int, state: Mapping[str, Any]) -> None:
        """Entrega o estado do modelo no passo ``step`` (pesos, centróides, previsões na sonda)."""

    def should_stop(self) -> bool:
        """Se o treino deve parar já (por exemplo, porque o utilizador cancelou)."""


class NullTrainingContext:
    """Contexto que ignora tudo, não quer snapshots e nunca pede para parar."""

    probe: NDArray[np.float64] | None = None

    def report_step(
        self, step: int, *, loss: float | None = None, metrics: Mapping[str, float] | None = None
    ) -> None:
        """Ignora o passo."""

    def wants_snapshot(self, step: int) -> bool:
        """Nunca quer snapshots."""
        return False

    def report_snapshot(self, step: int, state: Mapping[str, Any]) -> None:
        """Ignora o snapshot."""

    def should_stop(self) -> bool:
        """Nunca pede para parar."""
        return False


class RecordingTrainingContext(NullTrainingContext):
    """Contexto que guarda tudo o que o modelo anuncia, em todos os passos.

    Args:
        probe: Pontos onde se querem as previsões em cada snapshot.
        stop_after: Pede para parar depois deste passo (para testar o cancelamento).
    """

    def __init__(
        self, probe: NDArray[np.float64] | None = None, *, stop_after: int | None = None
    ) -> None:
        self.probe = probe
        self.steps: list[tuple[int, float | None, dict[str, float]]] = []
        self.snapshots: list[tuple[int, dict[str, Any]]] = []
        self._stop_after = stop_after

    @override
    def report_step(
        self, step: int, *, loss: float | None = None, metrics: Mapping[str, float] | None = None
    ) -> None:
        self.steps.append((step, loss, dict(metrics or {})))

    @override
    def wants_snapshot(self, step: int) -> bool:
        return True

    @override
    def report_snapshot(self, step: int, state: Mapping[str, Any]) -> None:
        self.snapshots.append((step, dict(state)))

    @override
    def should_stop(self) -> bool:
        return self._stop_after is not None and len(self.steps) >= self._stop_after


class SnapshotSampler:
    """Decide em que passos se tiram snapshots: no máximo ``limit``, espalhados pelo treino.

    Não se sabe à partida quantos passos o treino vai ter. Por isso começa por querer
    todos os passos e, sempre que já guardou ``limit`` snapshots, duplica o intervalo e
    deita fora os que deixaram de calhar nele: no fim ficam espalhados por todo o treino
    (o passo 1 fica sempre, para se ver o início).

    Args:
        limit: Quantos snapshots guardar, no máximo.
    """

    def __init__(self, limit: int = DEFAULT_SNAPSHOT_LIMIT) -> None:
        self.limit = limit
        self.interval = 1

    def wants(self, step: int) -> bool:
        """Se o passo ``step`` calha no intervalo atual."""
        return step == 1 or step % self.interval == 0

    def thin(self, steps: list[int]) -> list[int]:
        """Os passos a manter quando já há demasiados (pode aumentar o intervalo)."""
        while len(steps) > self.limit:
            self.interval *= 2
            steps = [step for step in steps if self.wants(step)]
        return steps


class EventTrainingContext:
    """Contexto que publica os passos e os snapshots no ``EventBus``.

    Args:
        bus: Para onde publicar.
        run_id: O run a que os eventos pertencem.
        sampler: Decide os passos com snapshot (partilhado com o gravador).
        probe: Pontos onde se querem as previsões em cada snapshot.
        control: Pedidos de pausa e de paragem; sem ele, o treino nunca pára a meio.
    """

    def __init__(
        self,
        bus: EventBus,
        run_id: str,
        *,
        sampler: SnapshotSampler | None = None,
        probe: NDArray[np.float64] | None = None,
        control: TrainingControl | None = None,
    ) -> None:
        self.probe = probe
        self.last_step = 0
        self.last_snapshot_step: int | None = None
        self._bus = bus
        self._run_id = run_id
        self._sampler = sampler or SnapshotSampler()
        self._control = control

    def report_step(
        self, step: int, *, loss: float | None = None, metrics: Mapping[str, float] | None = None
    ) -> None:
        """Publica ``StepCompleted`` e, se houver uma pausa pedida, espera aqui."""
        self.last_step = step
        self._bus.publish(
            StepCompleted(run_id=self._run_id, step=step, loss=loss, metrics=dict(metrics or {}))
        )
        if self._control is not None:
            self._control.wait_while_paused()

    def wants_snapshot(self, step: int) -> bool:
        """Se o passo calha no intervalo de amostragem."""
        return self._sampler.wants(step)

    def report_snapshot(self, step: int, state: Mapping[str, Any]) -> None:
        """Publica ``SnapshotTaken``."""
        self.last_snapshot_step = step
        self._bus.publish(SnapshotTaken(run_id=self._run_id, step=step, state=state))

    def should_stop(self) -> bool:
        """Se foi pedido que o treino pare."""
        return self._control is not None and self._control.stop_requested
