"""Contratos das métricas e dos avaliadores: aplicados a TODAS as métricas registadas.

Uma métrica nova fica testada aqui sem escrever testes novos (ver "Adicionar uma
métrica" em docs/architecture.md).
"""

import math

import numpy as np
import pytest

from ml_lab.core.evaluation.base import evaluator_for
from ml_lab.core.evaluation.explanations import METRIC_EXPLANATIONS
from ml_lab.core.evaluation.metrics.base import METRICS, MetricInputs
from ml_lab.core.foundation.types import TaskType
from ml_lab.core.models.base import MODELS


def sample_inputs(task_type: TaskType, *, perfect: bool = False) -> MetricInputs:
    """Valores reais e previstos de exemplo, com todas as classes presentes."""
    rng = np.random.default_rng(0)
    if task_type.is_classification:
        n_classes = 2 if task_type is TaskType.BINARY_CLASSIFICATION else 3
        y_true = np.repeat(np.arange(n_classes, dtype=np.int64), 10)
        y_pred = y_true.copy()
        if not perfect:
            y_pred[::4] = (y_pred[::4] + 1) % n_classes
        logits = np.eye(n_classes)[y_pred] * (8.0 if perfect else 2.0)
        logits += rng.normal(scale=0.0 if perfect else 0.5, size=logits.shape)
        proba = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        if perfect:
            proba = np.eye(n_classes)[y_true] * 0.999 + 0.001 / n_classes
        return MetricInputs(y_true, y_pred, proba, n_classes)
    values = rng.normal(size=30)
    noise = 0.0 if perfect else rng.normal(scale=0.3, size=30)
    return MetricInputs(values, values + noise)


CASES = [
    pytest.param(name, task_type, id=f"{name}-{task_type}")
    for name, metric in METRICS.items()
    for task_type in sorted(metric.task_types)
]


def test_there_are_classification_metrics() -> None:
    assert METRICS.for_task(TaskType.BINARY_CLASSIFICATION)
    assert METRICS.for_task(TaskType.MULTICLASS_CLASSIFICATION)


@pytest.mark.parametrize(("name", "task_type"), CASES)
def test_every_metric_returns_a_finite_value_on_sample_data(name: str, task_type: TaskType) -> None:
    value = METRICS.get(name)(sample_inputs(task_type))

    assert math.isfinite(value)


@pytest.mark.parametrize(("name", "task_type"), CASES)
def test_metrics_are_pure_functions(name: str, task_type: TaskType) -> None:
    inputs = sample_inputs(task_type)
    y_pred = inputs.y_pred.copy()

    first = METRICS.get(name)(inputs)
    second = METRICS.get(name)(inputs)

    assert first == second
    assert np.array_equal(inputs.y_pred, y_pred)  # não altera o que recebe


@pytest.mark.parametrize(("name", "task_type"), CASES)
def test_perfect_predictions_score_at_least_as_well(name: str, task_type: TaskType) -> None:
    metric = METRICS.get(name)

    perfect = metric(sample_inputs(task_type, perfect=True))
    imperfect = metric(sample_inputs(task_type))

    assert perfect >= imperfect if metric.higher_is_better else perfect <= imperfect


@pytest.mark.parametrize("name", METRICS.names())
def test_every_metric_is_explained(name: str) -> None:
    metric = METRICS.get(name)
    explanation = METRIC_EXPLANATIONS.get(name)

    assert metric.name == name
    assert metric.display_name
    assert explanation is not None, f"falta a explicação de {name} em evaluation/explanations.py"
    assert explanation.what
    assert explanation.how_to_read


def test_every_explanation_belongs_to_a_metric() -> None:
    assert set(METRIC_EXPLANATIONS) == set(METRICS.names())


@pytest.mark.parametrize(
    "task_type",
    sorted({task for _, model in MODELS.items() for task in model.info.task_types}),
)
def test_every_task_with_models_has_an_evaluator(task_type: TaskType) -> None:
    evaluator = evaluator_for(task_type)

    assert task_type in evaluator.task_types
