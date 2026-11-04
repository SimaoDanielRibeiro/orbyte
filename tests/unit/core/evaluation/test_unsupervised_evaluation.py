"""Métricas e avaliadores de clustering, redução de dimensionalidade e anomalias."""

import math
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from ml_lab.core.evaluation.anomaly import AnomalyEvaluator
from ml_lab.core.evaluation.clustering import ClusteringEvaluator, group_names
from ml_lab.core.evaluation.dimensionality import DimensionalityEvaluator
from ml_lab.core.evaluation.metrics.base import METRICS, MetricInputs
from ml_lab.core.foundation.notices import Notice, NoticeLevel
from ml_lab.core.foundation.types import Partition, TaskType
from ml_lab.core.models.base import MODELS, LabModel


def make(name: str, **values: Any) -> LabModel[Any]:
    model_class = MODELS.get(name)
    return model_class(model_class.info.hyperparameters.parse(values))


def clouds() -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    rng = np.random.default_rng(0)
    truth = np.repeat(np.arange(3), 40)
    centers = np.array([[0.0, 0.0], [6.0, 0.0], [3.0, 5.0]])
    return centers[truth] + rng.normal(scale=0.5, size=(120, 2)), truth


def value(name: str, **inputs: Any) -> float:
    return METRICS.get(name)(MetricInputs(**inputs))


def test_clustering_metrics_ignore_the_noise() -> None:
    X, truth = clouds()
    noisy = truth.copy()
    noisy[:10] = -1

    assert value("silhouette", X=X, y_pred=truth) > 0.7
    assert value("silhouette", X=X, y_pred=noisy) == pytest.approx(
        value("silhouette", X=X[10:], y_pred=truth[10:])
    )
    assert value("n_clusters", y_pred=noisy) == 3
    assert value("noise_ratio", y_pred=noisy) == pytest.approx(10 / 120)
    assert value("adjusted_rand", y_true=truth, y_pred=truth[::-1].copy() * 0 + truth) == 1.0
    assert math.isnan(value("silhouette", X=X, y_pred=np.zeros(120, dtype=np.int64)))


def test_trustworthiness_is_perfect_for_the_data_itself_and_repeatable() -> None:
    rng = np.random.default_rng(1)
    X = rng.normal(size=(1_500, 2))

    assert value("trustworthiness", X=X, embedding=X) == pytest.approx(1.0)
    shuffled = rng.normal(size=(1_500, 2))
    first = value("trustworthiness", X=X, embedding=shuffled)
    assert first < 0.7
    assert value("trustworthiness", X=X, embedding=shuffled) == first  # a amostra é sempre a mesma


def test_anomaly_metrics_compare_the_flags_with_the_true_anomalies() -> None:
    truth = np.array([0, 0, 0, 0, 1, 1])
    flags = np.array([0, 0, 0, 1, 1, 0])
    scores = np.array([0.1, 0.2, 0.3, 0.9, 0.8, 0.4])

    assert value("anomaly_rate", y_pred=flags) == pytest.approx(2 / 6)
    assert value("anomaly_precision", y_true=truth, y_pred=flags) == pytest.approx(1 / 2)
    assert value("anomaly_recall", y_true=truth, y_pred=flags) == pytest.approx(1 / 2)
    assert value("anomaly_roc_auc", y_true=truth, scores=scores) == pytest.approx(6 / 8)


def test_anomaly_precision_and_recall_are_undefined_without_anything_to_count() -> None:
    # Sem anomalias verdadeiras não há recall (e não 0%); sem pontos marcados, não há precisão.
    nothing = np.zeros(4, dtype=np.int64)
    some = np.array([0, 1, 0, 1])

    assert math.isnan(value("anomaly_recall", y_true=nothing, y_pred=some))
    assert math.isnan(value("anomaly_precision", y_true=some, y_pred=nothing))
    assert value("anomaly_precision", y_true=nothing, y_pred=some) == 0.0


def test_the_clustering_report_crosses_the_groups_with_the_reference() -> None:
    X, truth = clouds()
    model = make("kmeans", n_clusters=3).fit(X)

    report = ClusteringEvaluator().evaluate(
        model,
        X,
        truth,
        partition=Partition.TEST,
        task_type=TaskType.CLUSTERING,
        class_labels=("a", "b", "c"),
    )

    assert report.metric("adjusted_rand").value == pytest.approx(1.0)
    matrix = report.confusion_matrix
    assert matrix is not None
    assert matrix.columns == ("Group 1", "Group 2", "Group 3")
    assert sorted(max(row) for row in matrix.counts) == [40, 40, 40]  # cada classe num só grupo


def test_without_a_reference_the_clustering_report_only_measures_the_shape() -> None:
    X, _ = clouds()
    model = make("dbscan", eps=0.05, min_samples=20).fit(X)  # raio minúsculo: tudo é ruído

    report = ClusteringEvaluator().evaluate(
        model, X, None, partition=Partition.TEST, task_type=TaskType.CLUSTERING
    )

    assert report.confusion_matrix is None
    assert report.find("adjusted_rand") is None
    assert {notice.code for notice in report.notices} >= {"single_cluster", "mostly_noise"}


def test_group_names_put_the_noise_last() -> None:
    assert group_names(np.array([1, 0, -1, 1])) == ("Group 1", "Group 2", "Noise")


def test_a_model_that_only_maps_its_training_points_has_no_test_report() -> None:
    X, _ = clouds()
    model = make("tsne", perplexity=10).fit(X)
    evaluator = DimensionalityEvaluator()

    train = evaluator.evaluate(
        model, X, None, partition=Partition.TRAIN, task_type=TaskType.DIMENSIONALITY_REDUCTION
    )
    test = evaluator.evaluate(
        model, X[:20], None, partition=Partition.TEST, task_type=TaskType.DIMENSIONALITY_REDUCTION
    )

    assert train.metric("trustworthiness").value is not None
    assert test.metrics == ()
    assert [notice.code for notice in test.notices] == ["train_only"]


def test_the_anomaly_report_uses_a_binary_reference_and_explains_a_wrong_one() -> None:
    X, _ = clouds()
    X = np.vstack([X, [[20.0, 20.0], [-15.0, 10.0]]])
    reference = np.array([0] * 120 + [1, 1])
    model = make("isolation_forest", contamination=0.02).fit(X)
    evaluator = AnomalyEvaluator()

    report = evaluator.evaluate(
        model,
        X,
        reference,
        partition=Partition.TEST,
        task_type=TaskType.ANOMALY_DETECTION,
        class_labels=("normal", "anomalia"),
    )
    three = evaluator.evaluate(
        model,
        X,
        np.arange(len(X)) % 3,
        partition=Partition.TEST,
        task_type=TaskType.ANOMALY_DETECTION,
        class_labels=("a", "b", "c"),
    )

    assert report.metric("anomaly_roc_auc").value == pytest.approx(1.0)
    assert report.metric("anomaly_recall").value == pytest.approx(1.0)
    assert report.confusion_matrix is not None
    assert "reference" in [notice.code for notice in report.notices]
    assert report.find("anomaly_precision") is not None
    assert three.find("anomaly_precision") is None
    assert "reference_not_binary" in [notice.code for notice in three.notices]


def test_a_detector_that_flags_nothing_is_warned_only_if_it_missed_real_anomalies() -> None:
    X, _ = clouds()
    model = make("isolation_forest", contamination=0.001).fit(X)
    # Pontos no meio das nuvens: com um limiar tão alto, nenhum parece anómalo.
    centers = np.repeat([[0.0, 0.0], [6.0, 0.0], [3.0, 5.0]], 10, axis=0)
    test = centers + np.random.default_rng(1).normal(scale=0.1, size=centers.shape)
    assert not model.predict(test).any()

    def nothing_flagged(reference: NDArray[np.int64] | None) -> Notice | None:
        report = AnomalyEvaluator().evaluate(
            model,
            test,
            reference,
            partition=Partition.TEST,
            task_type=TaskType.ANOMALY_DETECTION,
            class_labels=("normal", "anomaly") if reference is not None else (),
        )
        return next((n for n in report.notices if n.code == "nothing_flagged"), None)

    two = np.zeros(len(test), dtype=np.int64)
    two[:2] = 1
    missed = nothing_flagged(two)
    assert missed is not None
    assert missed.level is NoticeLevel.WARNING
    assert "missed both real anomalies" in missed.message
    # Sem anomalias verdadeiras no teste, não marcar nada é acertar.
    assert nothing_flagged(np.zeros(len(test), dtype=np.int64)) is None
    # Sem referência não se sabe se havia o que marcar: fica uma nota, não um aviso.
    unknown = nothing_flagged(None)
    assert unknown is not None
    assert unknown.level is NoticeLevel.INFO


def test_a_detector_that_only_flags_normal_rows_is_warned() -> None:
    X, _ = clouds()
    X = np.vstack([X, [[20.0, 20.0], [-15.0, 10.0]]])
    model = make("isolation_forest", contamination=0.02).fit(X)
    flags = model.predict(X)
    reference = np.zeros(len(X), dtype=np.int64)
    reference[np.flatnonzero(flags == 0)[:2]] = 1  # duas linhas que ele não marcou

    report = AnomalyEvaluator().evaluate(
        model,
        X,
        reference,
        partition=Partition.TEST,
        task_type=TaskType.ANOMALY_DETECTION,
        class_labels=("normal", "anomaly"),
    )

    caught = [n for n in report.notices if n.code == "no_anomaly_caught"]
    assert len(caught) == 1
    assert caught[0].level is NoticeLevel.WARNING
    assert "it missed both real anomalies" in caught[0].message
