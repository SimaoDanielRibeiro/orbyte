"""Dados dos gráficos da avaliação, para cada tipo de tarefa."""

from typing import Any

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from ml_lab.core.data.cleaning.base import CleaningStep
from ml_lab.core.data.dataset import Dataset
from ml_lab.core.data.io.base import SourceSpec
from ml_lab.core.data.preparation import PreparationConfig
from ml_lab.core.data.splitting.base import SplitConfig, SplitStrategy
from ml_lab.core.foundation.types import TaskType
from ml_lab.core.models.base import MODELS, LabModel
from ml_lab.core.services.datasets import DatasetService
from ml_lab.core.visualization_data.evaluation import (
    dendrogram,
    evaluation_charts,
    silhouette_profile,
)

IRIS = SourceSpec("sample", {"name": "iris"})


def prepared(source: SourceSpec, config: PreparationConfig) -> Dataset:
    return DatasetService().prepared(source, config, 0).dataset


def trained(name: str, dataset: Dataset, **values: Any) -> LabModel[Any]:
    model_class = MODELS.get(name)
    model = model_class(model_class.info.hyperparameters.parse(values))
    target = dataset.train.y if dataset.task_type.requires_target else None
    return model.fit(dataset.train.X, target)


def test_classification_charts_have_one_roc_curve_per_class() -> None:
    iris = prepared(IRIS, PreparationConfig(target="species"))
    model = trained("logistic_regression", iris)

    charts = evaluation_charts(model, iris)

    assert [curve.label for curve in charts.roc] == ["setosa", "versicolor", "virginica"]
    assert len(charts.pr) == 3
    assert iris.test.y is not None
    truth = (iris.test.y == 1).astype(int)
    expected = roc_auc_score(truth, model.predict_proba(iris.test.X)[:, 1])
    assert charts.roc[1].area == pytest.approx(expected)
    assert charts.test_points is not None
    assert set(charts.test_points.labels) <= {"Certo", "Errado"}


def test_regression_charts_show_the_residuals() -> None:
    line = prepared(
        SourceSpec("synthetic", {"generator": "regression", "seed": 0, "parameters": {}}),
        PreparationConfig(target="target"),
    )
    model = trained("linear_regression", line)

    residuals = evaluation_charts(model, line).residuals

    assert residuals is not None
    assert np.allclose(np.array(residuals.actual) - np.array(residuals.predicted), residuals.errors)
    assert sum(residuals.histogram.counts[0]) == len(line.test)


def test_clustering_charts_have_the_silhouette_and_the_dendrogram() -> None:
    iris = prepared(IRIS, PreparationConfig(target="species", task_type=TaskType.CLUSTERING))
    model = trained("agglomerative", iris, n_clusters=3)

    charts = evaluation_charts(model, iris)

    assert charts.silhouette is not None
    assert len(charts.silhouette.groups) == 3
    assert charts.dendrogram is not None
    assert sum(count for _, count in charts.cluster_sizes) == len(iris.test)


def test_the_dendrogram_draws_the_last_merges_and_the_cut() -> None:
    # Quatro pontos: 0 e 1 juntam-se a 1, 2 e 3 a 2, e os dois grupos a 5.
    linkage = [[0, 1, 1.0, 2], [2, 3, 2.0, 2], [4, 5, 5.0, 4]]

    full = dendrogram(linkage, n_clusters=2)
    truncated = dendrogram(linkage, n_clusters=2, leaves=2)

    assert len(full.segments) == 3
    assert full.leaf_labels == ("1", "1", "1", "1")
    assert full.cut_height == pytest.approx(3.5)  # entre a junção a 2 e a junção a 5
    assert len(truncated.segments) == 1
    assert truncated.leaf_labels == ("(2)", "(2)")


def test_the_silhouette_profile_sorts_each_group() -> None:
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(loc=center, size=(30, 2)) for center in ((0, 0), (8, 0))])
    labels = np.repeat([0, 1], 30)

    profile = silhouette_profile(X, labels)

    assert profile is not None
    for group in profile.groups:
        assert list(group.values) == sorted(group.values, reverse=True)
    assert silhouette_profile(X, np.zeros(60, dtype=np.int64)) is None


ANOMALIES = SourceSpec("synthetic", {"generator": "anomalies", "seed": 0, "parameters": {}})


def test_anomaly_scores_are_split_by_the_reference_when_there_is_one() -> None:
    anomalies = prepared(
        ANOMALIES,
        PreparationConfig(
            target="target",
            task_type=TaskType.ANOMALY_DETECTION,
            split=SplitConfig(SplitStrategy.STRATIFIED),  # o teste fica com 2 das 10 anomalias
        ),
    )
    model = trained("isolation_forest", anomalies)

    charts = evaluation_charts(model, anomalies)

    assert charts.scores is not None
    assert charts.scores.series == ("Normal", "Anomalia")
    assert charts.scores_reference
    assert charts.threshold is not None
    assert anomalies.test.y is not None
    assert sum(charts.scores.counts[1]) == int(anomalies.test.y.sum())  # o 1 é a anomalia
    assert len(charts.roc) == 1


def test_without_a_reference_the_scores_are_split_at_the_threshold() -> None:
    anomalies = prepared(
        ANOMALIES,
        PreparationConfig(
            task_type=TaskType.ANOMALY_DETECTION,
            cleaning=(CleaningStep("drop_columns", {"columns": ["target"]}),),
        ),
    )
    model = trained("isolation_forest", anomalies)

    charts = evaluation_charts(model, anomalies)

    assert charts.scores is not None
    assert not charts.scores_reference
    assert sum(charts.scores.counts[1]) == int(model.predict(anomalies.test.X).sum())
    assert charts.roc == ()


def test_pca_charts_show_the_explained_variance() -> None:
    iris = prepared(
        IRIS, PreparationConfig(target="species", task_type=TaskType.DIMENSIONALITY_REDUCTION)
    )

    charts = evaluation_charts(trained("pca", iris, n_components=3), iris)

    assert len(charts.explained_variance) == 3
    assert list(charts.explained_variance) == sorted(charts.explained_variance, reverse=True)
