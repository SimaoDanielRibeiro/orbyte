"""O objetivo da procura: que métricas os agentes otimizam, e em que sentido as ordenam."""

import math

import pytest

from ml_lab.core.automl.objective import OBJECTIVES, objective_for, supports
from ml_lab.core.evaluation.base import PRIMARY_METRIC
from ml_lab.core.evaluation.metrics.base import METRICS
from ml_lab.core.foundation.errors import InvalidParameterError
from ml_lab.core.foundation.types import TaskType


@pytest.mark.parametrize("task_type", list(OBJECTIVES), ids=str)
def test_every_objective_is_a_metric_of_its_task_with_a_direction(task_type: TaskType) -> None:
    names = OBJECTIVES[task_type]

    assert names[0] == PRIMARY_METRIC[task_type]  # a primeira é a principal
    for name in names:
        metric = METRICS.get(name)
        assert task_type in metric.task_types
        assert metric.higher_is_better is not None  # uma métrica que só descreve não se otimiza


def test_precision_and_recall_alone_are_not_objectives() -> None:
    # Ganham-se com um modelo inútil: dizer sempre "positivo" dá um recall de 100%.
    assert "precision" not in OBJECTIVES[TaskType.BINARY_CLASSIFICATION]
    assert "recall" not in OBJECTIVES[TaskType.BINARY_CLASSIFICATION]


def test_the_agents_only_work_with_a_target() -> None:
    assert supports(TaskType.REGRESSION)
    assert not supports(TaskType.CLUSTERING)
    with pytest.raises(InvalidParameterError, match="modo manual"):
        objective_for(TaskType.CLUSTERING)


def test_the_default_objective_is_the_primary_metric() -> None:
    assert objective_for(TaskType.BINARY_CLASSIFICATION).name == "accuracy"
    assert objective_for(TaskType.REGRESSION).name == "r2"


def test_an_unknown_objective_lists_the_ones_that_exist() -> None:
    with pytest.raises(InvalidParameterError, match="Exatidão equilibrada"):
        objective_for(TaskType.BINARY_CLASSIFICATION, "precision")


def test_the_order_follows_the_direction_of_the_metric() -> None:
    accuracy = objective_for(TaskType.BINARY_CLASSIFICATION)
    rmse = objective_for(TaskType.REGRESSION, "rmse")

    assert accuracy.is_better(0.9, 0.8)
    assert rmse.is_better(0.8, 0.9)  # na RMSE, menos é melhor
    assert rmse.direction == "quanto menor, melhor"
    assert accuracy.key(None) == math.inf  # quem falhou vai para o fim
    assert not accuracy.is_better(None, 0.1)


def test_values_and_gaps_are_written_in_portuguese() -> None:
    accuracy = objective_for(TaskType.BINARY_CLASSIFICATION)
    r2 = objective_for(TaskType.REGRESSION)

    assert accuracy.written(0.8125) == "81,2%"
    assert accuracy.gap(0.83, 0.8) == "3,0 pontos"
    assert r2.written(0.6123) == "0,612"
    assert r2.gap(0.7, 0.65) == "0,050"
