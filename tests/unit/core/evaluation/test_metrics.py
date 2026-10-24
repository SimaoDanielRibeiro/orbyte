"""Métricas de classificação com valores calculados à mão (ou do exemplo da documentação)."""

import math

import numpy as np
import pytest
from sklearn.metrics import balanced_accuracy_score

from ml_lab.core.evaluation.metrics.base import METRICS, MetricInputs
from ml_lab.core.foundation.types import TaskType


def binary() -> MetricInputs:
    # 2 verdadeiros positivos, 1 falso positivo, 1 falso negativo, 1 verdadeiro negativo.
    return MetricInputs(np.array([0, 0, 1, 1, 1]), np.array([0, 1, 1, 1, 0]), n_classes=2)


def multiclass() -> MetricInputs:
    # O exemplo da documentação do scikit-learn para o f1_score.
    return MetricInputs(np.array([0, 1, 2, 0, 1, 2]), np.array([0, 2, 1, 0, 0, 1]), n_classes=3)


def value(name: str, inputs: MetricInputs) -> float:
    return METRICS.get(name)(inputs)


def test_the_classification_metrics_are_registered() -> None:
    names = set(METRICS.for_task(TaskType.BINARY_CLASSIFICATION))

    assert names == {
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "log_loss",
        "roc_auc",
    }
    assert set(METRICS.for_task(TaskType.REGRESSION)) == {"r2", "rmse", "mae", "mse"}


def test_binary_metrics_measure_the_positive_class() -> None:
    inputs = binary()

    assert value("accuracy", inputs) == pytest.approx(3 / 5)
    assert value("precision", inputs) == pytest.approx(2 / 3)
    assert value("recall", inputs) == pytest.approx(2 / 3)
    assert value("f1", inputs) == pytest.approx(2 / 3)
    assert value("balanced_accuracy", inputs) == pytest.approx((1 / 2 + 2 / 3) / 2)


def test_multiclass_metrics_use_the_simple_average_of_the_classes() -> None:
    inputs = multiclass()

    assert value("precision", inputs) == pytest.approx(2 / 9)
    assert value("recall", inputs) == pytest.approx(1 / 3)
    assert value("f1", inputs) == pytest.approx(4 / 15)


def test_balanced_accuracy_is_the_scikit_learn_definition() -> None:
    inputs = multiclass()

    assert value("balanced_accuracy", inputs) == pytest.approx(
        balanced_accuracy_score(inputs.y_true, inputs.y_pred)
    )


def test_balanced_accuracy_ignores_classes_missing_from_the_evaluated_rows() -> None:
    # A classe 2 só aparece nas previsões: o scikit-learn avisaria, esta versão não.
    inputs = MetricInputs(np.array([0, 0, 1, 1]), np.array([0, 2, 1, 1]), n_classes=3)

    assert value("balanced_accuracy", inputs) == pytest.approx((1 / 2 + 1) / 2)


def test_probability_metrics() -> None:
    proba = np.array([[0.9, 0.1], [0.6, 0.4], [0.65, 0.35], [0.2, 0.8]])
    inputs = MetricInputs(np.array([0, 0, 1, 1]), np.array([0, 0, 0, 1]), proba, n_classes=2)

    # Dos 4 pares (positivo, negativo), 3 estão bem ordenados: AUC = 0,75.
    assert value("roc_auc", inputs) == pytest.approx(0.75)
    expected = -np.mean(np.log([0.9, 0.6, 0.35, 0.8]))
    assert value("log_loss", inputs) == pytest.approx(expected)


def test_undefined_values_are_nan_instead_of_errors() -> None:
    without_proba = binary()
    one_class = MetricInputs(
        np.array([1, 1]), np.array([1, 0]), np.array([[0.2, 0.8], [0.6, 0.4]]), n_classes=2
    )

    assert math.isnan(value("log_loss", without_proba))
    assert math.isnan(value("roc_auc", without_proba))
    assert math.isnan(value("roc_auc", one_class))


def test_labels_cover_every_class() -> None:
    assert multiclass().labels == [0, 1, 2]
    assert MetricInputs(np.array([1.0]), np.array([1.0])).labels == []
