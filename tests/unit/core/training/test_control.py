"""Controlo do treino: pausa, retoma e paragem, pedidas de outro fio de execução."""

import threading

from ml_lab.core.training.control import TrainingControl


def test_a_new_control_lets_the_training_run() -> None:
    control = TrainingControl()

    control.wait_while_paused()  # não bloqueia

    assert not control.is_paused
    assert not control.stop_requested


def test_pause_and_resume() -> None:
    control = TrainingControl()

    control.pause()
    assert control.is_paused
    control.resume()

    assert not control.is_paused


def test_stopping_releases_a_paused_training() -> None:
    control = TrainingControl()
    released = threading.Event()

    def train() -> None:
        control.wait_while_paused()
        released.set()

    control.pause()
    worker = threading.Thread(target=train)
    worker.start()
    assert not released.wait(0.2)

    control.stop()

    assert released.wait(5)
    assert control.stop_requested
    assert not control.is_paused
    worker.join()
