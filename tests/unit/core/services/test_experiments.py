import pytest

from ml_lab.core.evaluation.report import EvaluationReport
from ml_lab.core.foundation.errors import (
    ComponentNotFoundError,
    HyperparameterError,
    ModelError,
)
from ml_lab.core.foundation.events import Event, RunFailed, RunFinished, RunStarted
from ml_lab.core.services import (
    DatasetService,
    ExperimentConfig,
    ExperimentService,
    Lab,
    ModelSpec,
    Partition,
    PreparationConfig,
    SourceSpec,
    SplitConfig,
    SplitStrategy,
    TaskType,
)
from ml_lab.core.services import experiments as experiments_module

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
    assert result.model_display_name == "Regressão logística"
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
    assert finished.summary == {"accuracy": result.primary.value}


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
    with pytest.raises(ModelError, match="não resolve regressão"):
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
