import pytest

from ml_lab.core.foundation.types import Paradigm, TaskType


def test_task_types_are_the_closed_list_from_the_plan() -> None:
    assert {task_type.value for task_type in TaskType} == {
        "binary_classification",
        "multiclass_classification",
        "regression",
        "clustering",
        "dimensionality_reduction",
        "anomaly_detection",
    }


@pytest.mark.parametrize("task_type", list(TaskType))
def test_every_task_type_has_a_label_and_a_paradigm(task_type: TaskType) -> None:
    assert task_type.label
    assert isinstance(task_type.paradigm, Paradigm)


def test_only_supervised_tasks_require_a_target() -> None:
    # Regra 1 da deteção automática: sem coluna-alvo só sobram as não supervisionadas.
    assert {task_type for task_type in TaskType if not task_type.requires_target} == {
        TaskType.CLUSTERING,
        TaskType.DIMENSIONALITY_REDUCTION,
        TaskType.ANOMALY_DETECTION,
    }


def test_classification_tasks() -> None:
    assert {task_type for task_type in TaskType if task_type.is_classification} == {
        TaskType.BINARY_CLASSIFICATION,
        TaskType.MULTICLASS_CLASSIFICATION,
    }


def test_task_type_round_trips_through_its_text_value() -> None:
    # O valor em texto é o que fica guardado em JSON e na base de dados.
    assert TaskType("regression") is TaskType.REGRESSION
    assert str(TaskType.REGRESSION) == "regression"
