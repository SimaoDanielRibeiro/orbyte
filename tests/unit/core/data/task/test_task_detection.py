from collections.abc import Sequence

import numpy as np
import pandas as pd
import pytest

from ml_lab.core.data.task.task_detection import UNSUPERVISED_TASKS, suggest_task
from ml_lab.core.foundation.errors import ColumnNotFoundError
from ml_lab.core.foundation.types import TaskType


def suggest(values: Sequence[object]) -> tuple[TaskType | None, tuple[TaskType, ...], list[str]]:
    suggestion = suggest_task(pd.DataFrame({"alvo": values}), "alvo")
    return suggestion.task_type, suggestion.allowed, [n.code for n in suggestion.notices]


def test_without_a_target_only_unsupervised_tasks_make_sense() -> None:
    suggestion = suggest_task(pd.DataFrame({"x": [1, 2]}), None)

    assert suggestion.task_type is TaskType.CLUSTERING
    assert suggestion.allowed == UNSUPERVISED_TASKS
    assert set(suggestion.allowed) == {
        TaskType.CLUSTERING,
        TaskType.DIMENSIONALITY_REDUCTION,
        TaskType.ANOMALY_DETECTION,
    }


def test_titanic_survived_suggests_binary_classification(passengers: pd.DataFrame) -> None:
    suggestion = suggest_task(passengers, "Survived")

    assert suggestion.task_type is TaskType.BINARY_CLASSIFICATION
    assert suggestion.n_classes == 2
    assert "dois valores (0 e 1)" in suggestion.reason


@pytest.mark.parametrize(
    ("values", "task", "allowed"),
    [
        pytest.param(
            ["setosa", "versicolor", "virginica"] * 3,
            TaskType.MULTICLASS_CLASSIFICATION,
            (TaskType.MULTICLASS_CLASSIFICATION,),
            id="text",
        ),
        pytest.param(
            [True, False] * 3,
            TaskType.BINARY_CLASSIFICATION,
            (TaskType.BINARY_CLASSIFICATION,),
            id="bool",
        ),
        pytest.param(
            [1, 2, 3] * 3,
            TaskType.MULTICLASS_CLASSIFICATION,
            (TaskType.MULTICLASS_CLASSIFICATION, TaskType.REGRESSION),
            id="few-ints",
        ),
        pytest.param(
            list(np.linspace(0, 1, 30)), TaskType.REGRESSION, (TaskType.REGRESSION,), id="floats"
        ),
        pytest.param(
            list(range(30)) * 2,  # sem repetições, 0…29 seria um identificador
            TaskType.REGRESSION,
            (TaskType.REGRESSION, TaskType.MULTICLASS_CLASSIFICATION),
            id="many-ints",
        ),
    ],
)
def test_the_kind_of_target_decides_the_task(
    values: list[object], task: TaskType, allowed: tuple[TaskType, ...]
) -> None:
    suggested, options, _ = suggest(values)

    assert suggested is task
    assert options == allowed


def test_missing_target_values_are_announced() -> None:
    _, _, codes = suggest(["a", "b", None, "a", "b"])

    assert codes == ["missing_target"]


@pytest.mark.parametrize(
    "values",
    [
        pytest.param([None, None], id="empty"),
        pytest.param([5, 5, 5], id="constant"),
        pytest.param(list(pd.to_datetime(["2024-01-01", "2024-02-01"])), id="dates"),
    ],
)
def test_unusable_targets_get_an_error_and_no_task(values: list[object]) -> None:
    task, allowed, codes = suggest(values)

    assert task is None
    assert allowed == ()
    assert codes[-1] == "unusable_target"


def test_numbers_stored_as_text_must_be_converted_first() -> None:
    task, _, codes = suggest(["1,5", "2,7", "3,1"])

    assert task is TaskType.REGRESSION
    assert codes == ["target_as_text"]


def test_free_text_targets_are_flagged() -> None:
    task, _, codes = suggest([f"nome {i}" for i in range(30)])

    assert task is TaskType.MULTICLASS_CLASSIFICATION
    assert codes == ["many_classes"]


def test_classes_with_a_single_row_are_flagged() -> None:
    _, _, codes = suggest(["a"] * 5 + ["b"] * 5 + ["c"])

    assert codes == ["tiny_class"]


def test_unknown_target_column() -> None:
    with pytest.raises(ColumnNotFoundError):
        suggest_task(pd.DataFrame({"x": [1]}), "y")
