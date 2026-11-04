"""Validação cruzada e curva de aprendizagem."""

import pytest

from ml_lab.core.data.dataset import Dataset
from ml_lab.core.data.io.base import SourceSpec
from ml_lab.core.data.preparation import PreparationConfig
from ml_lab.core.evaluation.metrics.base import METRICS
from ml_lab.core.evaluation.validation import cross_validate, learning_curve
from ml_lab.core.foundation.errors import CapabilityError, ModelError
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


def test_whoever_follows_hears_each_fold_before_it_trains() -> None:
    model = MODELS.get("knn")
    heard: list[int] = []

    cross_validate(
        model,
        model.info.hyperparameters(),
        0,
        folds(PreparationConfig("species"), k=3),
        on_fold=heard.append,
    )

    assert heard == [1, 2, 3]


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

    with pytest.raises(ModelError, match="against a target"):
        cross_validate(model, model.info.hyperparameters(), 0, folds(clustering))


def test_cross_validation_can_measure_another_metric() -> None:
    # Os agentes da fase 9 otimizam a métrica que se lhes pede: aqui, a ROC AUC, que mede as
    # probabilidades (uma coluna por classe, mesmo que um fold não tenha alguma).
    model = MODELS.get("logistic_regression")

    result = cross_validate(
        model,
        model.info.hyperparameters(),
        0,
        folds(PreparationConfig("species")),
        METRICS.get("roc_auc"),
    )

    assert result.metric == "roc_auc"
    assert not result.as_percent
    assert result.mean is not None
    assert result.mean > 0.95


def test_a_lower_is_better_metric_is_measured_as_it_is() -> None:
    model = MODELS.get("linear_regression")
    regression = PreparationConfig("sepal_length")

    rmse = cross_validate(
        model, model.info.hyperparameters(), 0, folds(regression), METRICS.get("rmse")
    )

    assert rmse.metric == "rmse"
    assert rmse.mean is not None
    assert 0.2 < rmse.mean < 0.6  # em centímetros, a escala do alvo


def test_a_metric_that_needs_probabilities_refuses_a_model_without_them() -> None:
    model = MODELS.get("perceptron")

    with pytest.raises(CapabilityError, match="probabilities"):
        cross_validate(
            model,
            model.info.hyperparameters(),
            0,
            folds(PreparationConfig("species")),
            METRICS.get("log_loss"),
        )


def test_a_metric_from_another_task_is_refused() -> None:
    model = MODELS.get("logistic_regression")

    with pytest.raises(ModelError, match="isn't a multiclass classification metric"):
        cross_validate(
            model,
            model.info.hyperparameters(),
            0,
            folds(PreparationConfig("species")),
            METRICS.get("r2"),
        )
