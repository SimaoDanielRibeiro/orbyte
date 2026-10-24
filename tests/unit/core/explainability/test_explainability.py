"""Importância por permutação e dependência parcial, em dados onde se sabe a resposta."""

from typing import Any

import numpy as np
import pytest

from ml_lab.core.data.dataset import Dataset
from ml_lab.core.data.io.base import SourceSpec
from ml_lab.core.data.preparation import PreparationConfig
from ml_lab.core.explainability.partial_dependence import partial_dependence
from ml_lab.core.explainability.permutation import permutation_importance
from ml_lab.core.foundation.errors import ModelError
from ml_lab.core.foundation.types import TaskType
from ml_lab.core.models.base import MODELS, LabModel
from ml_lab.core.services.datasets import DatasetService

# target = 4,4·x1 − 4,1·x2 + 0,6·x3 + … (os coeficientes da regra sintética, com a seed 3).
REGRESSION = SourceSpec(
    "synthetic", {"generator": "regression", "seed": 3, "parameters": {"n_features": 3}}
)


def prepared(source: SourceSpec, config: PreparationConfig) -> Dataset:
    return DatasetService().prepared(source, config, 0).dataset


def trained(name: str, dataset: Dataset, **values: Any) -> LabModel[Any]:
    model_class = MODELS.get(name)
    model = model_class(model_class.info.hyperparameters.parse(values))
    return model.fit(dataset.train.X, dataset.train.y)


@pytest.fixture(scope="module")
def line() -> Dataset:
    return prepared(REGRESSION, PreparationConfig(target="target"))


def coefficients(dataset: Dataset) -> list[float]:
    return list(trained("linear_regression", dataset).state()["coefficients"])


def test_the_feature_with_the_largest_weight_matters_most(line: Dataset) -> None:
    model = trained("linear_regression", line)

    importance = permutation_importance(model, line, seed=0)

    weights = dict(zip(line.feature_names, np.abs(coefficients(line)), strict=True))
    expected = sorted(weights, key=lambda name: -weights[name])
    assert [item.feature for item in importance.importances] == expected
    assert importance.metric_label == "R²"
    assert importance.baseline > 0.9


def test_a_useless_feature_has_no_importance() -> None:
    iris = prepared(SourceSpec("sample", {"name": "iris"}), PreparationConfig(target="species"))
    model = trained("decision_tree", iris, max_depth=2)  # só usa as pétalas

    importance = permutation_importance(model, iris, seed=0)

    by_name = {item.feature: item.mean for item in importance.importances}
    assert by_name["sepal_width"] == pytest.approx(0.0)
    assert importance.importances[0].feature.startswith("petal")


def test_the_partial_dependence_of_a_linear_model_is_a_straight_line(line: Dataset) -> None:
    model = trained("linear_regression", line)
    feature = line.feature_names[0]

    dependence = partial_dependence(model, line, feature, seed=0)

    grid, (average,) = np.array(dependence.grid), np.array(dependence.average)
    slope = np.polyfit(grid, average, 1)[0]
    assert slope == pytest.approx(coefficients(line)[0])
    assert np.allclose(np.diff(average) / np.diff(grid), slope)
    assert len(dependence.individual) > 0
    assert dependence.output_labels == ("Previsão de target",)


def test_classification_shows_the_probability_of_each_class() -> None:
    iris = prepared(SourceSpec("sample", {"name": "iris"}), PreparationConfig(target="species"))
    model = trained("logistic_regression", iris)

    dependence = partial_dependence(model, iris, "petal_length", seed=0)

    assert len(dependence.average) == 3
    first_point = [curve[0] for curve in dependence.average]
    assert sum(first_point) == pytest.approx(1.0)  # as probabilidades das três classes


def test_explanations_need_a_target_and_an_existing_feature(line: Dataset) -> None:
    model = trained("linear_regression", line)
    clustering = prepared(REGRESSION, PreparationConfig(task_type=TaskType.CLUSTERING))
    kmeans = MODELS.get("kmeans")().fit(clustering.train.X)

    with pytest.raises(ModelError, match="não existe"):
        partial_dependence(model, line, "x99", seed=0)
    with pytest.raises(ModelError, match="alvo"):
        permutation_importance(kmeans, clustering, seed=0)
