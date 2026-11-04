"""Treinos em segundo plano: estados, pausa, paragem e erros."""

import threading
import time
from collections.abc import Callable

import pytest

from ml_lab.core.foundation.events import StepCompleted
from ml_lab.core.training.control import TrainingControl
from ml_lab.core.training.jobs import JobStatus, TrainingJob
from ml_lab.core.training.recorder import SnapshotRecorder


def job(
    work: Callable[[TrainingControl], str], *, planned_steps: int | None = None
) -> TrainingJob[str]:
    return TrainingJob("run-1", work, SnapshotRecorder("run-1"), planned_steps=planned_steps)


def until_stopped(control: TrainingControl) -> str:
    """Um "treino" que dá passos até lhe pedirem para parar (e respeita as pausas)."""
    while not control.stop_requested:
        control.wait_while_paused()
        time.sleep(0.005)
    return "parado"


def wait_for(condition: Callable[[], bool]) -> None:
    deadline = time.monotonic() + 5
    while not condition():
        assert time.monotonic() < deadline, "o estado esperado nunca chegou"
        time.sleep(0.01)


def test_a_job_goes_from_waiting_to_finished() -> None:
    gate = threading.Event()
    training = job(lambda control: "feito" if gate.wait(5) else "atrasado")

    before = training.status
    training.start()
    during, done_during = training.status, training.done
    gate.set()

    assert training.wait(5) == "feito"
    assert (before, during, training.status) == (
        JobStatus.PENDING,
        JobStatus.RUNNING,
        JobStatus.FINISHED,
    )
    assert (done_during, training.done) == (False, True)
    assert training.result == "feito"
    assert training.error is None


def test_pausing_resuming_and_stopping_show_in_the_status() -> None:
    training = job(until_stopped).start()

    training.pause()
    wait_for(lambda: training.status is JobStatus.PAUSED)
    training.resume()
    wait_for(lambda: training.status is JobStatus.RUNNING)
    training.cancel()

    assert training.wait(5) == "parado"
    assert training.status is JobStatus.FINISHED


def test_stopping_a_paused_job_ends_it() -> None:
    training = job(until_stopped).start()
    training.pause()

    training.cancel()

    assert training.wait(5) == "parado"


def test_a_failure_is_kept_and_raised_to_whoever_waits() -> None:
    def broken(control: TrainingControl) -> str:
        raise ValueError("partiu")

    training = job(broken).start()

    with pytest.raises(ValueError, match="partiu"):
        training.wait(5)
    assert training.status is JobStatus.FAILED
    assert isinstance(training.error, ValueError)
    assert training.result is None


def test_waiting_too_little_raises_a_timeout() -> None:
    gate = threading.Event()
    training = job(lambda control: "feito" if gate.wait(5) else "atrasado").start()

    with pytest.raises(TimeoutError, match="run-1"):
        training.wait(0.05)
    gate.set()
    assert training.wait(5) == "feito"


def test_the_progress_comes_from_the_recorder() -> None:
    training = job(until_stopped, planned_steps=100)

    for step in (1, 2, 3):
        training.recorder.receive(StepCompleted(run_id="run-1", step=step, loss=1 / step))

    assert training.progress.steps == 3
    assert training.progress.last_loss == pytest.approx(1 / 3)
    assert training.planned_steps == 100


def test_the_state_is_one_plain_picture_of_the_job() -> None:
    def broken(control: TrainingControl) -> str:
        raise ValueError("partiu")

    running = job(until_stopped, planned_steps=10)
    running.recorder.receive(StepCompleted(run_id="run-1", step=1, loss=0.5))
    failed = job(broken).start()
    with pytest.raises(ValueError, match="partiu"):
        failed.wait(5)

    state = running.state()

    assert (state.run_id, state.status, state.planned_steps) == ("run-1", JobStatus.PENDING, 10)
    assert state.progress.steps == 1
    assert state.error is None
    assert failed.state().status is JobStatus.FAILED
    assert failed.state().error == "partiu"


def test_the_statuses_have_names_to_show() -> None:
    assert [status.label for status in JobStatus] == [
        "Waiting",
        "Training",
        "Paused",
        "Finished",
        "Failed",
    ]
