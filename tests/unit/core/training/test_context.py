from ml_lab.core.training.context import NullTrainingContext, RecordingTrainingContext


def test_the_null_context_ignores_everything_and_never_stops() -> None:
    context = NullTrainingContext()

    context.report_step(1, loss=0.3, metrics={"accuracy": 0.9})
    context.report_snapshot(1, {"weights": [0.1, 0.2]})

    assert context.should_stop() is False


def test_the_recording_context_keeps_what_the_model_announced() -> None:
    context = RecordingTrainingContext()

    context.report_step(1, loss=0.5)
    context.report_step(2, loss=0.4, metrics={"accuracy": 0.8})
    context.report_snapshot(2, {"weights": [0.1]})

    assert context.steps == [(1, 0.5, {}), (2, 0.4, {"accuracy": 0.8})]
    assert context.snapshots == [(2, {"weights": [0.1]})]
    assert context.should_stop() is False
