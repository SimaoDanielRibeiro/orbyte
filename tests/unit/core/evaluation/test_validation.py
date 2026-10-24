"""Validação cruzada e curva de aprendizagem."""

import pytest

from ml_lab.core.data.dataset import Dataset
from ml_lab.core.data.io.base import SourceSpec
from ml_lab.core.data.preparation import PreparationConfig
from ml_lab.core.evaluation.validation import cross_validate, learning_curve
from ml_lab.core.foundation.errors import ModelError
from ml_lab.core.foundation.types import TaskType
from ml_lab.core.models.base import MODELS
from ml_lab.core.services.datasets import DatasetService

IRIS = SourceSpec("sample", {"name": "iris"})


def folds(config: PreparationConfig, k: int = 5) -> tuple[Dataset, ...]:
    return DatasetService().folds(IRIS, config, 7, k)


def test_cross_validation_trains_and_scores_every_fold() -> None:
    model = MODELS.get("logistic_regression")

    result = cross_validate(
        model, model.info.hyperparameters(), 0, folds(PreparationConfig("species"))
    )

    assert result.metric == "accuracy"
    assert result.as_percent
    assert [fold.fold for fold in result.folds] == [1, 2, 3, 4, 5]
    assert all(fold.test is not None and fold.test > 0.8 for fold in result.folds)
    assert result.mean is not None
    assert result.mean > 0.9
    assert result.std is not None
    assert "cross_validation" in [notice.code for notice in result.notices]


def test_the_learning_curve_grows_the_training_set() -> None:
    model = MODELS.get("knn")

    curve = learning_curve(
        model, model.info.hyperparameters(), 0, folds(PreparationConfig("species"))
    )

    sizes = [point.size for point in curve.points]
    assert sizes == sorted(sizes)
    assert sizes[-1] > 4 * sizes[0]
    assert curve.points[-1].test_mean > 0.85
    assert len(curve.notices) == 1


def test_a_model_that_decorates_the_training_set_is_flagged() -> None:
    model = MODELS.get("decision_tree_regression")
    params = model.info.hyperparameters.parse({"max_depth": 0})
    regression = PreparationConfig("sepal_length")

    curve = learning_curve(model, params, 0, folds(regression))

    assert curve.metric == "r2"
    # Uma árvore sem limite decora o treino (só não acerta nas flores repetidas com alvos
    # diferentes, que nenhuma árvore separa).
    assert curve.points[-1].train_mean > 0.99
    assert curve.notices[0].code == "high_variance"


def test_tasks_without_a_target_have_no_cross_validation() -> None:
    model = MODELS.get("kmeans")
    clustering = PreparationConfig(task_type=TaskType.CLUSTERING)

    with pytest.raises(ModelError, match="alvo"):
        cross_validate(model, model.info.hyperparameters(), 0, folds(clustering))
