"""Trainer: liga o modelo ao EventBus e garante a fotografia do último passo."""

import numpy as np
from numpy.typing import NDArray

from ml_lab.core.foundation.events import Event, EventBus, SnapshotTaken, StepCompleted
from ml_lab.core.models.base import MODELS
from ml_lab.core.training.context import SnapshotSampler
from ml_lab.core.training.control import TrainingControl
from ml_lab.core.training.recorder import SnapshotRecorder
from ml_lab.core.training.trainer import Trainer


def separable() -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 2))
    return X, (X[:, 0] + X[:, 1] > 0).astype(np.int64)


def test_every_step_is_announced_and_the_last_one_is_photographed() -> None:
    bus = EventBus()
    received: list[Event] = []
    bus.subscribe(Event, received.append)
    sampler = SnapshotSampler(limit=5)
    recorder = SnapshotRecorder("run-1", sampler)
    recorder.attach(bus)
    X, y = separable()

    outcome = Trainer(bus).train(
        MODELS.get("logistic_regression_gd")(),
        X,
        y,
        run_id="run-1",
        sampler=sampler,
        probe=X[:4],
    )

    steps = [event.step for event in received if isinstance(event, StepCompleted)]
    assert steps == list(range(1, outcome.steps + 1))
    assert not outcome.stopped_early
    kept = recorder.snapshots()
    assert len(kept) <= 6
    assert kept[0].step == 1
    assert kept[-1].step == outcome.steps  # mesmo que o último passo não calhe no intervalo
    assert np.asarray(kept[-1].state["probe"]).shape == (4, 2)


def test_a_stop_request_ends_the_training_early_with_what_was_learned() -> None:
    bus = EventBus()
    control = TrainingControl()
    bus.subscribe(StepCompleted, lambda event: control.stop() if event.step == 5 else None)
    model = MODELS.get("logistic_regression_gd")()
    X, y = separable()

    outcome = Trainer(bus).train(model, X, y, run_id="run-1", control=control)

    assert outcome.stopped_early
    assert outcome.steps == 5
    assert model.is_fitted
    assert model.state()["iterations"] == 5


def test_a_model_that_trains_at_once_announces_nothing() -> None:
    bus = EventBus()
    received: list[Event] = []
    bus.subscribe(Event, received.append)
    X, y = separable()

    outcome = Trainer(bus).train(
        MODELS.get("logistic_regression")(), X, y, run_id="run-1", probe=X[:4]
    )

    assert outcome.steps == 0
    assert not outcome.stopped_early
    assert not [event for event in received if isinstance(event, SnapshotTaken)]
