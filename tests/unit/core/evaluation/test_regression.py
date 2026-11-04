"""Métricas e avaliador de regressão, com valores calculados à mão."""

import math

import numpy as np
import pytest
from numpy.typing import NDArray

from ml_lab.core.evaluation.metrics.base import METRICS, MetricInputs
from ml_lab.core.evaluation.regression import DISPLAY_ORDER, RegressionEvaluator
from ml_lab.core.foundation.errors import ModelError
from ml_lab.core.foundation.types import Partition, TaskType
from ml_lab.core.models.supervised.regression.linear_regression_gd import LinearRegressionGD


def value(name: str, y_true: list[float], y_pred: list[float]) -> float:
    return METRICS.get(name)(MetricInputs(np.array(y_true), np.array(y_pred)))


def test_the_regression_metrics_match_the_hand_computed_values() -> None:
    # Erros: 0,5; 0; -1; 1. A média do alvo é 2,5 e a soma dos desvios ao quadrado é 5.
    y_true, y_pred = [1.0, 2.0, 3.0, 4.0], [1.5, 2.0, 2.0, 5.0]

    assert value("mae", y_true, y_pred) == pytest.approx(2.5 / 4)
    assert value("mse", y_true, y_pred) == pytest.approx(2.25 / 4)
    assert value("rmse", y_true, y_pred) == pytest.approx(0.75)
    assert value("r2", y_true, y_pred) == pytest.approx(1 - 2.25 / 5)


def test_r2_is_zero_for_the_mean_and_undefined_without_variation() -> None:
    assert value("r2", [1.0, 2.0, 3.0], [2.0, 2.0, 2.0]) == pytest.approx(0.0)
    assert math.isnan(value("r2", [3.0], [2.0]))
    assert math.isnan(value("r2", [2.0, 2.0], [1.0, 3.0]))


def fitted() -> tuple[LinearRegressionGD, NDArray[np.float64], NDArray[np.float64]]:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 2))
    y = X @ np.array([2.0, -1.0]) + rng.normal(scale=0.5, size=100)
    return LinearRegressionGD().fit(X, y), X, y


def test_a_good_model_is_compared_with_predicting_the_mean() -> None:
    model, X, y = fitted()

    report = RegressionEvaluator().evaluate(
        model, X, y, partition=Partition.TEST, task_type=TaskType.REGRESSION
    )

    assert tuple(metric.name for metric in report.metrics) == DISPLAY_ORDER
    assert report.n_samples == 100
    assert report.confusion_matrix is None
    r2 = report.metric("r2").value
    assert r2 is not None
    assert r2 > 0.9
    (notice,) = report.notices
    assert notice.code == "baseline"
    assert "always predicting the mean would give R² = 0" in notice.message


def test_a_model_worse_than_the_mean_gets_a_warning() -> None:
    model, X, y = fitted()

    report = RegressionEvaluator().evaluate(
        model, X, -y, partition=Partition.TEST, task_type=TaskType.REGRESSION
    )

    assert [notice.code for notice in report.notices] == ["below_baseline"]
    assert "worse than always predicting the mean" in report.notices[0].message


def test_a_single_row_leaves_r2_undefined_and_explains_why() -> None:
    model, X, y = fitted()

    report = RegressionEvaluator().evaluate(
        model, X[:1], y[:1], partition=Partition.TEST, task_type=TaskType.REGRESSION
    )

    assert report.metric("r2").formatted == "—"
    assert [notice.code for notice in report.notices] == ["undefined_metric"]
    assert report.notices[0].message.startswith("R²")


def test_a_regression_needs_the_target_to_be_evaluated() -> None:
    model, X, _ = fitted()

    with pytest.raises(ModelError, match="needs the target"):
        RegressionEvaluator().evaluate(
            model, X, None, partition=Partition.TEST, task_type=TaskType.REGRESSION
        )
