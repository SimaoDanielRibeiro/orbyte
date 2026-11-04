import threading

import pytest

from ml_lab.core.evaluation.report import EvaluationReport
from ml_lab.core.foundation.errors import (
    ComponentNotFoundError,
    HyperparameterError,
    ModelError,
)
from ml_lab.core.foundation.events import (
    Event,
    RunFailed,
    RunFinished,
    RunStarted,
    StepCompleted,
)
from ml_lab.core.services import (
    DatasetService,
    ExperimentConfig,
    ExperimentResult,
    ExperimentService,
    Lab,
    ModelSpec,
    Partition,
    PreparationConfig,
    SourceSpec,
    SplitConfig,
    SplitStrategy,
    TaskType,
    TrainingJob,
)
from ml_lab.core.services import experiments as experiments_module

GRADIENT_DESCENT = ModelSpec(name="logistic_regression_gd")

IRIS = SourceSpec("sample", {"name": "iris"})
REGRESSION = SourceSpec(
    "synthetic", {"generator": "regression", "seed": 0, "parameters": {"n_samples": 50}}
)


def config(**overrides: object) -> ExperimentConfig:
    values: dict[str, object] = {
        "dataset": IRIS,
        "preparation": PreparationConfig(
            target="species", split=SplitConfig(SplitStrategy.STRATIFIED)
        ),
        "model": ModelSpec(name="logistic_regression"),
        "seed": 42,
    }
    values.update(overrides)
    return ExperimentConfig(**values)


@pytest.fixture
def service() -> ExperimentService:
    return ExperimentService(DatasetService())


@pytest.fixture
def events(service: ExperimentService) -> list[Event]:
    received: list[Event] = []
    service.bus.subscribe(Event, received.append)
    return received


def test_a_run_trains_evaluates_and_describes_the_result(service: ExperimentService) -> None:
    result = service.run(config())

    assert result.dataset_name == "Iris"
    assert result.task_type is TaskType.MULTICLASS_CLASSIFICATION
    assert result.model_display_name == "Logistic regression"
    assert result.class_labels == ("setosa", "versicolor", "virginica")
    assert result.feature_names == ("sepal_length", "sepal_width", "petal_length", "petal_width")
    assert list(result.reports) == [Partition.TRAIN, Partition.TEST]
    assert all(isinstance(report, EvaluationReport) for report in result.reports.values())
    assert result.primary is not None
    assert result.primary.name == "accuracy"
    assert result.primary.value is not None
    assert result.primary.value >= 0.9
    assert result.hyperparameters["C"] == 1.0
    assert len(result.model_state["coefficients"]) == 3
    assert "no_overfitting" in [notice.code for notice in result.notices]


def test_the_run_is_announced_on_the_event_bus(
    service: ExperimentService, events: list[Event]
) -> None:
    result = service.run(config())

    started, _, finished = events
    assert [event.kind for event in events] == ["run_started", "evaluation", "run_finished"]
    assert {event.run_id for event in events} == {result.run_id}
    assert isinstance(started, RunStarted)
    assert started.config["model"]["name"] == "logistic_regression"
    assert isinstance(finished, RunFinished)
    assert result.primary is not None
    assert finished.summary == {"accuracy": result.primary.value, "steps": 0}


def test_the_same_configuration_gives_the_same_result(service: ExperimentService) -> None:
    first = service.run(config())
    second = service.run(config())

    assert first.run_id != second.run_id
    assert first.reports[Partition.TEST].values() == second.reports[Partition.TEST].values()
    assert first.model_state == second.model_state


def test_without_a_seed_one_is_generated_and_recorded(service: ExperimentService) -> None:
    result = service.run(config(seed=None))

    assert result.config.seed is not None
    assert result.seed == result.config.seed
    repeated = service.run(result.config)
    assert repeated.reports[Partition.TEST].values() == result.reports[Partition.TEST].values()


def test_a_validation_split_is_evaluated_too(service: ExperimentService) -> None:
    preparation = PreparationConfig(
        target="species", split=SplitConfig(SplitStrategy.STRATIFIED, validation_size=0.2)
    )

    result = service.run(config(preparation=preparation))

    assert list(result.reports) == [Partition.TRAIN, Partition.VALIDATION, Partition.TEST]


def test_problems_found_before_training_publish_nothing(
    service: ExperimentService, events: list[Event]
) -> None:
    with pytest.raises(ComponentNotFoundError):
        service.run(config(model=ModelSpec(name="logistic")))
    with pytest.raises(HyperparameterError, match="C"):
        service.run(config(model=ModelSpec(name="logistic_regression", hyperparameters={"C": 0})))

    assert events == []


def test_a_model_that_does_not_solve_the_task_is_refused(service: ExperimentService) -> None:
    with pytest.raises(ModelError, match="doesn't solve regression"):
        service.run(config(dataset=REGRESSION, preparation=PreparationConfig(target="target")))


def test_failures_during_the_run_are_announced(
    service: ExperimentService, events: list[Event], monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(task_type: TaskType) -> None:
        raise ModelError("avaliador partido")

    monkeypatch.setattr(experiments_module, "evaluator_for", broken)

    with pytest.raises(ModelError, match="avaliador partido"):
        service.run(config())

    assert [event.kind for event in events] == ["run_started", "run_failed"]
    failed = events[-1]
    assert isinstance(failed, RunFailed)
    assert failed.message == "avaliador partido"
    assert failed.error_type == "ModelError"


def test_the_lab_wires_the_experiments_to_its_datasets() -> None:
    lab = Lab.create()

    lab.experiments.run(config())

    assert [summary.name for summary in lab.datasets.loaded()] == ["Iris"]


# --- fase 3: treino passo a passo -------------------------------------------------------


def test_a_step_by_step_model_leaves_its_history_and_animation(
    service: ExperimentService,
) -> None:
    result = service.run(config(model=GRADIENT_DESCENT))

    assert result.steps > 0
    assert [record.step for record in result.history] == list(range(1, result.steps + 1))
    assert result.model_state["iterations"] == result.steps
    animation = result.animation
    assert animation is not None
    assert animation.animated
    assert animation.value_kind == "class"  # o Iris tem três classes
    assert len(animation.frames) <= 61
    assert animation.frames[0].step == 1
    assert animation.frames[-1].step == result.steps
    assert set(result.training_notices) <= set(result.notices)


def test_steps_and_snapshots_come_between_the_start_and_the_evaluation(
    service: ExperimentService, events: list[Event]
) -> None:
    result = service.run(config(model=GRADIENT_DESCENT))

    kinds = [event.kind for event in events]
    assert kinds[0] == "run_started"
    assert kinds[-2:] == ["evaluation", "run_finished"]
    assert set(kinds[1:-2]) == {"step", "snapshot"}
    assert kinds.count("step") == result.steps
    finished = events[-1]
    assert isinstance(finished, RunFinished)
    assert finished.summary["steps"] == result.steps


def test_a_model_that_trains_at_once_shows_only_its_final_picture(
    service: ExperimentService,
) -> None:
    result = service.run(config())

    assert result.steps == 0
    assert result.history == ()
    assert result.animation is not None
    assert not result.animation.animated
    assert result.animation.frames[0].step == 0


def test_a_regression_with_one_feature_is_animated_along_a_line(
    service: ExperimentService,
) -> None:
    result = service.run(
        config(
            dataset=REGRESSION,
            preparation=PreparationConfig(target="target"),
            model=ModelSpec(name="linear_regression_gd"),
        )
    )

    assert result.task_type is TaskType.REGRESSION
    assert result.primary is not None
    assert result.primary.name == "r2"
    assert result.animation is not None
    assert result.animation.dimensions == 1
    assert result.animation.value_kind == "value"


def test_start_trains_in_the_background_and_gives_the_same_result(
    service: ExperimentService,
) -> None:
    job = service.start(config(model=GRADIENT_DESCENT))

    result = job.wait(60)

    assert isinstance(job, TrainingJob)
    assert job.run_id == result.run_id
    assert job.planned_steps == 2_000  # o máximo de épocas por omissão
    assert result.model_state == service.run(config(model=GRADIENT_DESCENT)).model_state


def test_start_refuses_a_bad_configuration_before_training(
    service: ExperimentService, events: list[Event]
) -> None:
    bad = ModelSpec(name="logistic_regression_gd", hyperparameters={"learning_rate": 0})

    with pytest.raises(HyperparameterError, match="Learning rate"):
        service.start(config(model=bad))

    assert events == []


def stop_at(
    service: ExperimentService, step: int
) -> tuple[threading.Event, list[TrainingJob[ExperimentResult]]]:
    """No passo ``step``, o treino espera até lhe darem o trabalho e manda-o parar."""
    ready = threading.Event()
    jobs: list[TrainingJob[ExperimentResult]] = []

    def at_step(event: StepCompleted) -> None:
        if event.step == step:
            assert ready.wait(10)
            jobs[0].cancel()

    service.bus.subscribe(StepCompleted, at_step)
    return ready, jobs


def test_a_stopped_training_is_evaluated_with_what_it_learned(
    service: ExperimentService,
) -> None:
    ready, jobs = stop_at(service, 10)
    jobs.append(service.start(config(model=GRADIENT_DESCENT)))
    ready.set()

    result = jobs[0].wait(60)

    assert result.stopped_early
    assert result.steps == 10
    assert result.training_notices[0].code == "stopped_early"
    assert "step 10" in result.training_notices[0].message
    assert result.reports[Partition.TEST].metric("accuracy").value is not None


def test_the_preview_shows_the_latest_snapshot_while_training(
    service: ExperimentService,
) -> None:
    reached, release = threading.Event(), threading.Event()

    def hold(event: StepCompleted) -> None:
        if event.step == 50:
            reached.set()
            assert release.wait(10)

    service.bus.subscribe(StepCompleted, hold)
    job = service.start(config(model=GRADIENT_DESCENT))
    assert reached.wait(10)

    preview = service.preview(job.run_id)
    release.set()
    job.wait(60)

    assert preview is not None
    assert len(preview.frames) == 1
    assert 1 <= preview.frames[0].step < 50
    assert service.preview(job.run_id) is None  # acabou: a animação está no resultado
    assert service.preview("run-desconhecido") is None


def test_recent_runs_keep_their_model_and_data(
    service: ExperimentService, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(experiments_module, "KEPT_RUNS", 2)
    first, second, third = (service.run(config()) for _ in range(3))

    artifacts = service.artifacts(third.run_id)
    service.artifacts(second.run_id)

    assert artifacts.model.is_fitted
    assert artifacts.dataset.name == "Iris"
    assert artifacts.config == third.config
    with pytest.raises(ModelError, match="no longer in memory"):
        service.artifacts(first.run_id)
