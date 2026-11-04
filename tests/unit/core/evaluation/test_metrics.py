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


# --- os termos das fórmulas: dão o valor da métrica -----------------------------------------

TP, FP, FN = r"\mathrm{TP}", r"\mathrm{FP}", r"\mathrm{FN}"


def terms(name: str, inputs: MetricInputs) -> dict[str, float]:
    """Os termos de uma métrica sem grupos (a binária, a regressão), pelo símbolo."""
    found = METRICS.get(name).terms(inputs)
    assert all(term.group is None for term in found)
    return {term.symbol: term.value for term in found}


def by_class(name: str, inputs: MetricInputs) -> dict[int, dict[str, float]]:
    """Os termos de uma métrica feita por classe, pela classe e pelo símbolo."""
    found: dict[int, dict[str, float]] = {}
    for term in METRICS.get(name).terms(inputs):
        assert term.group is not None
        found.setdefault(term.group, {})[term.symbol] = term.value
    return found


def test_binary_terms_count_the_positive_class() -> None:
    inputs = binary()

    precision, recall = terms("precision", inputs), terms("recall", inputs)

    assert precision == {TP: 2, FP: 1}
    assert recall == {TP: 2, FN: 1}
    assert precision[TP] / (precision[TP] + precision[FP]) == pytest.approx(
        value("precision", inputs)
    )
    f1 = terms("f1", inputs)
    assert 2 * f1["P"] * f1["R"] / (f1["P"] + f1["R"]) == pytest.approx(value("f1", inputs))
    accuracy = terms("accuracy", inputs)
    assert accuracy[r"\text{correct}"] / accuracy["n"] == pytest.approx(value("accuracy", inputs))


def test_multiclass_terms_come_per_class_and_average_to_the_metric() -> None:
    inputs = multiclass()

    precision = by_class("precision", inputs)
    recalls = by_class("balanced_accuracy", inputs)

    assert set(precision) == {0, 1, 2}
    per_class = [c[TP] / (c[TP] + c[FP]) if c[TP] + c[FP] else 0.0 for c in precision.values()]
    assert np.mean(per_class) == pytest.approx(value("precision", inputs))
    mean_recall = np.mean([c[r"\text{recall}_c"] for c in recalls.values()])
    assert mean_recall == pytest.approx(value("balanced_accuracy", inputs))


def test_the_log_loss_terms_add_up_the_penalties() -> None:
    proba = np.array([[0.9, 0.1], [0.6, 0.4], [0.65, 0.35], [0.2, 0.8]])
    inputs = MetricInputs(np.array([0, 0, 1, 1]), np.array([0, 0, 0, 1]), proba, n_classes=2)

    found = terms("log_loss", inputs)

    total = found[r"\sum_{i=1}^{n} -\log p_i"]
    assert total / found["n"] == pytest.approx(value("log_loss", inputs))


def test_regression_terms_rebuild_each_error() -> None:
    inputs = MetricInputs(np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.5, 2.0, 2.0, 5.0]))
    squares = r"\sum_{i=1}^{n} (y_i - \hat{y}_i)^2"

    mae, mse = terms("mae", inputs), terms("mse", inputs)
    r2 = terms("r2", inputs)

    assert mae[r"\sum_{i=1}^{n} |y_i - \hat{y}_i|"] / mae["n"] == pytest.approx(
        value("mae", inputs)
    )
    assert mse[squares] / mse["n"] == pytest.approx(value("mse", inputs))
    assert math.sqrt(mse[squares] / mse["n"]) == pytest.approx(value("rmse", inputs))
    residual, total = r2[r"\mathrm{SS}_{\text{res}}"], r2[r"\mathrm{SS}_{\text{tot}}"]
    assert 1 - residual / total == pytest.approx(value("r2", inputs))


def test_anomaly_and_noise_terms_count_the_points() -> None:
    anomalies = MetricInputs(np.array([1, 1, 0, 0, 0]), np.array([1, 0, 1, 0, 0]))
    clusters = MetricInputs(y_pred=np.array([0, 0, 1, -1, -1]))

    assert terms("anomaly_precision", anomalies) == {TP: 1, FP: 1}
    assert terms("anomaly_recall", anomalies) == {TP: 1, FN: 1}
    assert terms("anomaly_rate", anomalies) == {r"n_{\text{flagged}}": 2, "n": 5}
    assert terms("noise_ratio", clusters) == {r"n_{\text{noise}}": 2, "n": 5}


def test_metrics_without_counts_have_no_terms() -> None:
    proba = np.array([[0.9, 0.1], [0.2, 0.8]])
    inputs = MetricInputs(np.array([0, 1]), np.array([0, 1]), proba, n_classes=2)

    assert METRICS.get("roc_auc").terms(inputs) == ()
