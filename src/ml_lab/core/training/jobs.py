"""Treinos em segundo plano: correr, pausar, retomar e cancelar sem bloquear a interface.

Um ``TrainingJob`` corre um trabalho (por exemplo, uma experiência inteira) num fio
de execução próprio. A interface fica livre para mostrar o progresso, que lê do
gravador de snapshots, e para pedir pausas ou o cancelamento através do controlo
(ADR 0014). Cancelar não é um erro: o treino pára no fim do passo seguinte e o
resultado fica com o que o modelo já tinha aprendido.

Quem lança um treino fica com o ``TrainingJob``. Quem só sabe o identificador do run (a
API, que recebe pedidos soltos) encontra-o no ``JobRegistry`` do serviço que o lançou.
"""

import threading
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Final, Self

from ml_lab.core.foundation.errors import (
    JobNotFoundError,
    TrainingFailedError,
    TrainingInProgressError,
)
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
        """Nome a mostrar, em inglês."""
        return _STATUS_LABELS[self]


_STATUS_LABELS: Final[Mapping[JobStatus, str]] = MappingProxyType(
    {
        JobStatus.PENDING: "Waiting",
        JobStatus.RUNNING: "Training",
        JobStatus.PAUSED: "Paused",
        JobStatus.FINISHED: "Finished",
        JobStatus.FAILED: "Failed",
    }
)


@dataclass(frozen=True, slots=True)
class JobState:
    """Um retrato do treino em segundo plano, em valores simples (o que a API devolve).

    Attributes:
        run_id: O run que o trabalho produz.
        status: O estado.
        progress: Os passos dados, a última loss e as fotografias tiradas.
        planned_steps: O máximo de passos, se se souber.
        error: Se falhou, a explicação do erro.
    """

    run_id: str
    status: JobStatus
    progress: Progress
    planned_steps: int | None = None
    error: str | None = None


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
        self._thread = threading.Thread(target=self._run, name=f"training-{run_id}", daemon=True)
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

    def state(self) -> JobState:
        """O estado agora, em valores simples: é o que se mostra enquanto o treino decorre."""
        return JobState(
            run_id=self.run_id,
            status=self.status,
            progress=self.progress,
            planned_steps=self.planned_steps,
            error=None if self._error is None else str(self._error),
        )

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
            raise TimeoutError(f"Training {self.run_id} hasn't finished yet.")
        if self._error is not None:
            raise self._error
        assert self._result is not None  # o trabalho terminou sem erro
        return self._result

    def _run(self) -> None:
        try:
            self._result = self._work(self.control)
        except Exception as error:  # guardado aqui e relançado em wait(), no fio de quem espera
            self._error = error


class JobRegistry[T]:
    """Os treinos em segundo plano de um serviço, encontrados pelo identificador do run.

    Os que ainda decorrem ficam sempre. Dos que já terminaram, quando há mais do que
    ``limit`` treinos, saem primeiro os mais antigos: cada um guarda o seu resultado, e
    os resultados ocupam memória.

    Args:
        limit: Quantos treinos lembrar, no máximo (sem contar os que decorrem).
    """

    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._jobs: OrderedDict[str, TrainingJob[T]] = OrderedDict()
        self._lock = threading.Lock()

    def add(self, job: TrainingJob[T]) -> TrainingJob[T]:
        """Junta um treino (e esquece os terminados que passem do limite)."""
        with self._lock:
            self._jobs[job.run_id] = job
            finished = [run_id for run_id, known in self._jobs.items() if known.done]
            for run_id in finished[: max(0, len(self._jobs) - self._limit)]:
                del self._jobs[run_id]
        return job

    def find(self, run_id: str) -> TrainingJob[T] | None:
        """O treino do run, ou ``None`` se não for um treino deste registo."""
        with self._lock:
            return self._jobs.get(run_id)

    def get(self, run_id: str) -> TrainingJob[T]:
        """O treino do run.

        Raises:
            JobNotFoundError: Se não houver nenhum treino com esse run.
        """
        job = self.find(run_id)
        if job is None:
            raise JobNotFoundError(run_id)
        return job

    def result(self, run_id: str) -> T:
        """O resultado de um treino que terminou bem.

        Raises:
            JobNotFoundError: Se não houver nenhum treino com esse run.
            TrainingInProgressError: Se ainda decorrer.
            TrainingFailedError: Se falhou.
        """
        job = self.get(run_id)
        if not job.done:
            raise TrainingInProgressError(run_id)
        if job.error is not None:
            raise TrainingFailedError(run_id, str(job.error)) from job.error
        assert job.result is not None  # terminou sem erro
        return job.result
