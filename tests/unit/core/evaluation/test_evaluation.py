"""Relatório, avaliador de classificação e diagnóstico do overfitting."""

import json
from typing import Any, override

import numpy as np
import pytest
from numpy.typing import NDArray

from ml_lab.core.evaluation.base import PRIMARY_METRIC, evaluator_for
from ml_lab.core.evaluation.classification import DISPLAY_ORDER, ClassificationEvaluator
from ml_lab.core.evaluation.diagnostics import compare_train_and_test
from ml_lab.core.evaluation.regression import RegressionEvaluator
from ml_lab.core.evaluation.report import EvaluationReport, MetricResult
from ml_lab.core.foundation.errors import ComponentNotFoundError, ModelError
from ml_lab.core.foundation.notices import NoticeLevel
from ml_lab.core.foundation.types import Partition, TaskType
from ml_lab.core.models.base import (
    Features,
    Implementation,
    LabModel,
    ModelFamily,
    ModelInfo,
    Target,
)
from ml_lab.core.schemas.hyperparameters import Hyperparameters
from ml_lab.core.schemas.serialization import to_json
from ml_lab.core.training.context import TrainingContext


class NoParams(Hyperparameters):
    """Sem hiperparâmetros."""


class AlwaysFirst(LabModel[NoParams]):
    """Prevê sempre a primeira classe vista no treino, com toda a certeza."""

    info = ModelInfo(
        name="always_first",
        display_name="Sempre a primeira",
        summary="Para testes.",
        task_types=frozenset({TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION}),
        family=ModelFamily.LINEAR,
        implementation=Implementation.DIDACTIC,
        iterative=False,
        supports_proba=True,
        hyperparameters=NoParams,
        doc="nenhuma.md",
    )

    @override
    def _fit(self, X: Features, y: Target | None, context: TrainingContext) -> None:
        assert y is not None
        self._seen = np.unique(y).astype(np.int64)

    @override
    def _predict(self, X: Features) -> NDArray[Any]:
        return np.full(len(X), self._seen[0])

    @override
    def _predict_proba(self, X: Features) -> NDArray[np.float64]:
        proba = np.zeros((len(X), len(self._seen)))
        proba[:, 0] = 1.0
        return proba

    @override
    def classes(self) -> NDArray[np.int64] | None:
        return self._seen


def evaluate(
    model: LabModel[Any], X: Features, y: Target, labels: tuple[str, ...]
) -> EvaluationReport:
    task = (
        TaskType.BINARY_CLASSIFICATION if len(labels) == 2 else TaskType.MULTICLASS_CLASSIFICATION
    )
    return ClassificationEvaluator().evaluate(
        model, X, y, partition=Partition.TEST, task_type=task, class_labels=labels
    )


def report(partition: Partition, accuracy: float | None, rows: int = 10) -> EvaluationReport:
    result = MetricResult("accuracy", "Accuracy", accuracy, True, True, "what", "how")
    return EvaluationReport(partition, TaskType.BINARY_CLASSIFICATION, rows, (result,))


# --- relatório ------------------------------------------------------------------------


def test_results_are_formatted_as_on_the_site() -> None:
    assert MetricResult("a", "A", 0.9667, True, True, "", "").formatted == "96.7%"
    assert MetricResult("b", "B", 0.1234, False, False, "", "").formatted == "0.123"
    assert MetricResult("c", "C", None, True, True, "", "").formatted == "—"
    assert MetricResult("d", "D", float("nan"), True, False, "", "").formatted == "—"


def test_counts_are_shown_without_decimals() -> None:
    # Uma métrica que só descreve (o número de grupos) é uma contagem: "3", não "3.000".
    assert MetricResult("n", "N", 3.0, None, False, "", "").formatted == "3"
    assert MetricResult("s", "S", 3.0, True, False, "", "").formatted == "3.000"
    assert MetricResult("r", "R", 0.25, None, False, "", "").formatted == "0.250"


def test_reports_find_metrics_and_become_plain_json() -> None:
    result = report(Partition.TRAIN, 0.8)

    assert result.metric("accuracy").value == 0.8
    assert result.values() == {"accuracy": 0.8}
    assert to_json(result)["partition"] == "train"
    json.dumps(to_json(result), allow_nan=False)
    with pytest.raises(ComponentNotFoundError, match="Did you mean 'accuracy'"):
        result.metric("acuracy")


# --- avaliador ------------------------------------------------------------------------


def test_the_evaluator_collects_every_metric_in_display_order() -> None:
    X = np.zeros((6, 2))
    y = np.array([0, 0, 0, 0, 1, 1])
    model = AlwaysFirst().fit(X, y)

    result = evaluate(model, X, y, ("não", "sim"))

    assert [metric.name for metric in result.metrics] == list(DISPLAY_ORDER)
    assert result.metric("accuracy").value == pytest.approx(4 / 6)
    assert result.metric("accuracy").what  # cada métrica leva a sua explicação
    assert result.confusion_matrix is not None
    assert result.confusion_matrix.counts == ((4, 0), (2, 0))
    assert result.n_samples == 6


def test_a_model_no_better_than_the_majority_class_is_flagged() -> None:
    X = np.zeros((6, 2))
    y = np.array([0, 0, 0, 0, 1, 1])

    result = evaluate(AlwaysFirst().fit(X, y), X, y, ("não", "sim"))

    notice = result.notices[0]
    assert notice.code == "below_baseline"
    assert notice.level is NoticeLevel.WARNING
    assert "Always predicting “não” would get 66.7% right" in notice.message


def test_probabilities_cover_classes_missing_from_training() -> None:
    # O modelo só viu as classes 0 e 2; a 1 tem de aparecer com probabilidade zero.
    X = np.zeros((4, 2))
    model = AlwaysFirst().fit(X, np.array([0, 0, 2, 2]))

    result = evaluate(model, X, np.array([0, 1, 2, 2]), ("a", "b", "c"))

    assert result.confusion_matrix is not None
    assert len(result.confusion_matrix.counts) == 3
    assert result.metric("log_loss").value is not None


def test_metrics_that_cannot_be_computed_are_explained() -> None:
    X = np.zeros((3, 2))
    model = AlwaysFirst().fit(X, np.array([0, 1, 1]))

    result = evaluate(model, X, np.array([0, 0, 0]), ("a", "b"))

    assert result.metric("roc_auc").value is None
    assert "undefined_metric" in [notice.code for notice in result.notices]


def test_classification_needs_the_target_and_the_classes() -> None:
    model = AlwaysFirst().fit(np.zeros((2, 1)), np.array([0, 1]))

    with pytest.raises(ModelError):
        ClassificationEvaluator().evaluate(
            model,
            np.zeros((2, 1)),
            None,
            partition=Partition.TEST,
            task_type=TaskType.BINARY_CLASSIFICATION,
            class_labels=("a", "b"),
        )


def test_each_task_type_gets_its_evaluator() -> None:
    assert isinstance(evaluator_for(TaskType.BINARY_CLASSIFICATION), ClassificationEvaluator)
    assert isinstance(evaluator_for(TaskType.REGRESSION), RegressionEvaluator)
    assert PRIMARY_METRIC[TaskType.MULTICLASS_CLASSIFICATION] == "accuracy"
    assert PRIMARY_METRIC[TaskType.REGRESSION] == "r2"
    for task_type in TaskType:
        assert task_type in PRIMARY_METRIC
        # Desde a fase 4, todas as tarefas que aprendem com dados têm avaliador. O controlo
        # (fase 8) avalia-se a jogar episódios, sem dados: core/rl/evaluation.py.
        if task_type.learns_from_data:
            assert task_type in evaluator_for(task_type).task_types


# --- diagnóstico ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("train", "test", "rows", "code"),
    [
        (0.99, 0.80, 10, "overfitting"),
        (0.85, 0.83, 10, "no_overfitting"),
        (0.80, 0.90, 10, "lucky_test"),
        # Perto em pontos, mas o que falha passa de 0 para 7,4%, e não é ruído com 30 000
        # linhas de teste (ADR 0051): decorou o treino.
        (1.00, 0.926, 30_000, "overfitting"),
        # O Iris: 3 erros em 30 linhas de teste ainda cabem no ruído (1 erro já dava 96,7%).
        (0.975, 0.90, 30, "no_overfitting"),
        # Cinco pontos com o que falha a subir pouco (de 15% para 20%) não é decorar.
        (0.85, 0.80, 30_000, "no_overfitting"),
    ],
)
def test_train_and_test_are_compared(train: float, test: float, rows: int, code: str) -> None:
    notices = compare_train_and_test(
        {
            Partition.TRAIN: report(Partition.TRAIN, train),
            Partition.TEST: report(Partition.TEST, test, rows),
        }
    )

    assert [notice.code for notice in notices] == [code]


def test_the_overfitting_warning_quotes_both_values() -> None:
    reports = {
        Partition.TRAIN: report(Partition.TRAIN, 0.99),
        Partition.TEST: report(Partition.TEST, 0.80),
    }

    (notice,) = compare_train_and_test(reports)

    assert notice.level is NoticeLevel.WARNING
    assert notice.message.startswith(
        "Accuracy: 99.0% on the training data and 80.0% on the test data, 19.0 points"
    )


def test_a_small_gap_with_a_much_bigger_error_says_why_it_is_overfitting() -> None:
    reports = {
        Partition.TRAIN: report(Partition.TRAIN, 1.0),
        Partition.TEST: report(Partition.TEST, 0.926, 30_000),
    }

    (notice,) = compare_train_and_test(reports)

    assert notice.level is NoticeLevel.WARNING
    assert "only 7.4 points apart" in notice.message
    assert "(7.4%) is more than twice what it misses on the training data (0.0%)" in (
        notice.message
    )


def test_the_comparison_reads_well_for_any_primary_metric() -> None:
    def r2(partition: Partition, value: float) -> EvaluationReport:
        result = MetricResult("r2", "R²", value, True, False, "what", "how")
        return EvaluationReport(partition, TaskType.REGRESSION, 10, (result,))

    reports = {Partition.TRAIN: r2(Partition.TRAIN, 0.61), Partition.TEST: r2(Partition.TEST, 0.6)}

    (notice,) = compare_train_and_test(reports)

    assert notice.code == "no_overfitting"
    assert notice.message.startswith(
        "R²: 0.610 on the training data and 0.600 on the test data: close values"
    )


def test_nothing_to_compare_gives_no_notices() -> None:
    assert compare_train_and_test({Partition.TRAIN: report(Partition.TRAIN, 0.9)}) == ()
    reports = {
        Partition.TRAIN: report(Partition.TRAIN, 0.9),
        Partition.TEST: report(Partition.TEST, None),
    }
    assert compare_train_and_test(reports) == ()
