"""Gravador de snapshots: fica só com o seu run e nunca guarda mais do que o limite."""

from ml_lab.core.foundation.events import EventBus, RunStarted, SnapshotTaken, StepCompleted
from ml_lab.core.training.context import SnapshotSampler
from ml_lab.core.training.recorder import Progress, SnapshotRecorder


def test_the_recorder_keeps_only_the_steps_and_snapshots_of_its_run() -> None:
    bus = EventBus()
    recorder = SnapshotRecorder("run-1")
    recorder.attach(bus)

    bus.publish(RunStarted(run_id="run-1", config={}, n_samples=10, n_features=2))
    bus.publish(StepCompleted(run_id="run-1", step=1, loss=0.9, metrics={"accuracy": 0.5}))
    bus.publish(StepCompleted(run_id="run-2", step=1, loss=0.1))
    bus.publish(SnapshotTaken(run_id="run-1", step=1, state={"weights": [1.0]}))
    recorder.detach(bus)
    bus.publish(StepCompleted(run_id="run-1", step=2, loss=0.5))

    (step,) = recorder.steps()
    assert (step.step, step.loss, dict(step.metrics)) == (1, 0.9, {"accuracy": 0.5})
    (snapshot,) = recorder.snapshots()
    assert (snapshot.step, snapshot.state) == (1, {"weights": [1.0]})
    assert recorder.progress() == Progress(steps=1, last_loss=0.9, snapshots=1)
    assert bus.subscription_count == 0


def test_without_steps_there_is_no_progress() -> None:
    assert SnapshotRecorder("run-1").progress() == Progress(steps=0, last_loss=None, snapshots=0)


def test_snapshots_are_thinned_but_the_first_and_the_latest_stay() -> None:
    sampler = SnapshotSampler(limit=8)
    recorder = SnapshotRecorder("run-1", sampler)

    for step in range(1, 201):
        recorder.receive(StepCompleted(run_id="run-1", step=step, loss=1 / step))
        if sampler.wants(step) or step == 200:  # o Trainer fotografa sempre o último passo
            recorder.receive(SnapshotTaken(run_id="run-1", step=step, state={"step": step}))

    kept = [record.step for record in recorder.snapshots()]
    assert len(kept) <= 9  # o limite, mais o do fim
    assert kept[0] == 1
    assert kept[-1] == 200
    assert kept == sorted(kept)
    assert len(recorder.steps()) == 200
    assert recorder.progress().steps == 200
