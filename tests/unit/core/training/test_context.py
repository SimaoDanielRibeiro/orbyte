"""Contextos de treino e amostragem dos snapshots."""

import threading
from itertools import pairwise

import numpy as np

from ml_lab.core.foundation.events import Event, EventBus, SnapshotTaken, StepCompleted
from ml_lab.core.training.context import (
    DEFAULT_SNAPSHOT_LIMIT,
    EventTrainingContext,
    NullTrainingContext,
    RecordingTrainingContext,
    SnapshotSampler,
)
from ml_lab.core.training.control import TrainingControl


def sample(sampler: SnapshotSampler, steps: int) -> list[int]:
    """Os passos com snapshot que ficam guardados num treino de ``steps`` passos."""
    kept: list[int] = []
    for step in range(1, steps + 1):
        if sampler.wants(step):
            kept = sampler.thin([*kept, step])
    return kept


def test_the_null_context_ignores_everything_and_never_stops() -> None:
    context = NullTrainingContext()

    context.report_step(1, loss=0.3, metrics={"accuracy": 0.9})
    context.report_snapshot(1, {"weights": [0.1, 0.2]})

    assert context.probe is None
    assert context.wants_snapshot(1) is False
    assert context.should_stop() is False


def test_the_recording_context_keeps_what_the_model_announced() -> None:
    context = RecordingTrainingContext()

    context.report_step(1, loss=0.5)
    context.report_step(2, loss=0.4, metrics={"accuracy": 0.8})
    context.report_snapshot(2, {"weights": [0.1]})

    assert context.steps == [(1, 0.5, {}), (2, 0.4, {"accuracy": 0.8})]
    assert context.snapshots == [(2, {"weights": [0.1]})]
    assert context.wants_snapshot(7) is True
    assert context.should_stop() is False


def test_the_recording_context_can_ask_to_stop_after_some_steps() -> None:
    probe = np.zeros((3, 2))
    context = RecordingTrainingContext(probe, stop_after=2)

    context.report_step(1)
    assert not context.should_stop()
    context.report_step(2)

    assert context.should_stop()
    assert context.probe is probe


def test_short_trainings_keep_a_snapshot_of_every_step() -> None:
    assert sample(SnapshotSampler(), 30) == list(range(1, 31))
    assert DEFAULT_SNAPSHOT_LIMIT == 60


def test_long_trainings_keep_at_most_the_limit_spread_evenly() -> None:
    sampler = SnapshotSampler(limit=10)

    kept = sample(sampler, 1000)

    assert len(kept) <= 10
    assert kept[0] == 1  # o início fica sempre
    assert kept[-1] > 500  # e chega ao fim do treino, não fica pelo início
    gaps = {after - before for before, after in pairwise(kept[1:])}
    assert gaps == {sampler.interval}


def test_the_event_context_publishes_the_steps_and_snapshots_of_its_run() -> None:
    bus = EventBus()
    received: list[Event] = []
    bus.subscribe(Event, received.append)
    context = EventTrainingContext(bus, "run-1")

    context.report_step(1, loss=0.5, metrics={"accuracy": 0.7})
    context.report_snapshot(1, {"weights": [0.1]})

    step, snapshot = received
    assert isinstance(step, StepCompleted)
    assert (step.run_id, step.step, step.loss) == ("run-1", 1, 0.5)
    assert dict(step.metrics) == {"accuracy": 0.7}
    assert isinstance(snapshot, SnapshotTaken)
    assert snapshot.state == {"weights": [0.1]}
    assert (context.last_step, context.last_snapshot_step) == (1, 1)
    assert not context.should_stop()


def test_the_event_context_wants_the_snapshots_its_sampler_wants() -> None:
    sampler = SnapshotSampler(limit=2)
    sampler.interval = 4
    context = EventTrainingContext(EventBus(), "run-1", sampler=sampler)

    assert [step for step in range(1, 10) if context.wants_snapshot(step)] == [1, 4, 8]


def test_a_pause_holds_the_training_at_the_end_of_the_step() -> None:
    control = TrainingControl()
    context = EventTrainingContext(EventBus(), "run-1", control=control)
    finished = threading.Event()

    def train() -> None:
        context.report_step(1)
        finished.set()

    control.pause()
    worker = threading.Thread(target=train)
    worker.start()

    assert not finished.wait(0.2)  # parado na pausa, no fim do passo 1
    control.resume()
    assert finished.wait(5)
    worker.join()


def test_stopping_is_asked_through_the_control() -> None:
    control = TrainingControl()
    context = EventTrainingContext(EventBus(), "run-1", control=control)

    assert not context.should_stop()
    control.stop()

    assert context.should_stop()
