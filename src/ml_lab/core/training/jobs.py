"""Treinos em segundo plano: correr, pausar, retomar e cancelar sem bloquear a interface.

Um ``TrainingJob`` corre um trabalho (por exemplo, uma experiência inteira) num fio
de execução próprio. A interface fica livre para mostrar o progresso, que lê do
gravador de snapshots, e para pedir pausas ou o cancelamento através do controlo
(ADR 0014). Cancelar não é um erro: o treino pára no fim do passo seguinte e o
resultado fica com o que o modelo já tinha aprendido.
"""

import threading
from collections.abc import Callable, Mapping
from enum import StrEnum
from types import MappingProxyType
from typing import Final, Self

from ml_lab.core.training.control import TrainingControl
from ml_lab.core.training.recorder import Progress, SnapshotRecorder


class JobStatus(StrEnum):
    """Em que estado está um treino em segundo plano."""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    FINISHED = "finished"
    FAILED = "failed"

    @property
    def label(self) -> str:
        """Nome a mostrar, em português."""
        return _STATUS_LABELS[self]


_STATUS_LABELS: Final[Mapping[JobStatus, str]] = MappingProxyType(
    {
        JobStatus.PENDING: "À espera",
        JobStatus.RUNNING: "A treinar",
        JobStatus.PAUSED: "Em pausa",
        JobStatus.FINISHED: "Terminado",
        JobStatus.FAILED: "Falhou",
    }
)


class TrainingJob[T]:
    """Um trabalho de treino num fio de execução próprio.

    Args:
        run_id: O run que o trabalho produz.
        work: O trabalho; recebe o controlo, para o passar ao treino.
        recorder: O gravador do run, de onde se lê o progresso.
        planned_steps: O máximo de passos, se se souber (para a barra de progresso); o
            treino pode acabar antes, quando o modelo converge.
    """

    def __init__(
        self,
        run_id: str,
        work: Callable[[TrainingControl], T],
        recorder: SnapshotRecorder,
        *,
        planned_steps: int | None = None,
    ) -> None:
        self.run_id = run_id
        self.control = TrainingControl()
        self.recorder = recorder
        self.planned_steps = planned_steps
        self._work = work
        self._result: T | None = None
        self._error: Exception | None = None
        self._thread = threading.Thread(target=self._run, name=f"treino-{run_id}", daemon=True)
        self._started = False

    def start(self) -> Self:
        """Começa o trabalho em segundo plano."""
        self._started = True
        self._thread.start()
        return self

    @property
    def status(self) -> JobStatus:
        """O estado atual."""
        if not self._started:
            return JobStatus.PENDING
        if self._thread.is_alive():
            return JobStatus.PAUSED if self.control.is_paused else JobStatus.RUNNING
        return JobStatus.FAILED if self._error is not None else JobStatus.FINISHED

    @property
    def done(self) -> bool:
        """Se já terminou (bem ou mal)."""
        return self.status in (JobStatus.FINISHED, JobStatus.FAILED)

    @property
    def progress(self) -> Progress:
        """O progresso até agora (passos, última loss, snapshots)."""
        return self.recorder.progress()

    @property
    def result(self) -> T | None:
        """O resultado, se o trabalho terminou bem."""
        return self._result

    @property
    def error(self) -> Exception | None:
        """O erro, se o trabalho falhou."""
        return self._error

    def pause(self) -> None:
        """Pede uma pausa no fim do passo atual."""
        self.control.pause()

    def resume(self) -> None:
        """Retoma depois de uma pausa."""
        self.control.resume()

    def cancel(self) -> None:
        """Pede ao treino que pare no fim do passo atual."""
        self.control.stop()

    def wait(self, timeout: float | None = None) -> T:
        """Espera pelo fim e devolve o resultado.

        Raises:
            TimeoutError: Se não terminar dentro de ``timeout`` segundos.
            Exception: O erro do trabalho, se falhou.
        """
        self._thread.join(timeout)
        if self._thread.is_alive():
            raise TimeoutError(f"O treino {self.run_id} ainda não terminou.")
        if self._error is not None:
            raise self._error
        assert self._result is not None  # o trabalho terminou sem erro
        return self._result

    def _run(self) -> None:
        try:
            self._result = self._work(self.control)
        except Exception as error:  # guardado aqui e relançado em wait(), no fio de quem espera
            self._error = error
