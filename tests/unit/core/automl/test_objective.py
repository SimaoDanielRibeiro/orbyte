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
    with pytest.raises(InvalidParameterError, match="manual mode"):
        objective_for(TaskType.CLUSTERING)


def test_the_default_objective_is_the_primary_metric() -> None:
    assert objective_for(TaskType.BINARY_CLASSIFICATION).name == "accuracy"
    assert objective_for(TaskType.REGRESSION).name == "r2"


def test_an_unknown_objective_lists_the_ones_that_exist() -> None:
    with pytest.raises(InvalidParameterError, match="Balanced accuracy"):
        objective_for(TaskType.BINARY_CLASSIFICATION, "precision")


def test_the_order_follows_the_direction_of_the_metric() -> None:
    accuracy = objective_for(TaskType.BINARY_CLASSIFICATION)
    rmse = objective_for(TaskType.REGRESSION, "rmse")

    assert accuracy.is_better(0.9, 0.8)
    assert rmse.is_better(0.8, 0.9)  # na RMSE, menos é melhor
    assert rmse.direction == "lower is better"
    assert accuracy.key(None) == math.inf  # quem falhou vai para o fim
    assert not accuracy.is_better(None, 0.1)


def test_values_and_gaps_are_written_as_on_the_site() -> None:
    accuracy = objective_for(TaskType.BINARY_CLASSIFICATION)
    r2 = objective_for(TaskType.REGRESSION)

    assert accuracy.written(0.8125) == "81.2%"
    assert accuracy.gap(0.83, 0.8) == "3.0 points"
    assert r2.written(0.6123) == "0.612"
    assert r2.gap(0.7, 0.65) == "0.050"


@pytest.mark.parametrize("task_type", list(OBJECTIVES))
def test_every_objective_reads_its_value_in_words(task_type: TaskType) -> None:
    for name in OBJECTIVES[task_type]:
        reading = objective_for(task_type, name).reading(0.75)
        assert reading
        assert reading[0].isupper()
        assert not reading.endswith(".")  # quem a usa põe o ponto final


def test_the_reading_says_what_the_value_means() -> None:
    accuracy = objective_for(TaskType.BINARY_CLASSIFICATION)
    r2 = objective_for(TaskType.REGRESSION)

    assert accuracy.reading(0.812) == "It gets about 81 in every 100 rows right"
    assert r2.reading(0.752) == "It explains about 75% of the variation in the target"
    assert r2.reading(-0.2) == "It does worse than always predicting the mean of the target"
    assert accuracy.reading(None) == ""


def test_overfitting_is_a_training_score_well_above_the_validation() -> None:
    accuracy = objective_for(TaskType.BINARY_CLASSIFICATION)
    rmse = objective_for(TaskType.REGRESSION, "rmse")

    assert accuracy.overfits(0.99, 0.8)
    assert not accuracy.overfits(0.85, 0.8)
    assert rmse.overfits(1.0, 2.0)  # o erro na validação é o dobro do do treino
    assert not rmse.overfits(1.0, 1.2)
