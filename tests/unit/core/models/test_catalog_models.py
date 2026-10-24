"""Comportamentos próprios dos modelos do catálogo da fase 4 (o geral está nos contratos)."""

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from ml_lab.core.foundation.errors import CapabilityError, ModelError
from ml_lab.core.models.adapters.sklearn_adapter import SklearnModel
from ml_lab.core.models.base import MODELS, LabModel
from ml_lab.core.models.unsupervised.clustering import agglomerative as agglomerative_module
from ml_lab.core.models.unsupervised.clustering.agglomerative import linkage_matrix
from ml_lab.core.training.context import RecordingTrainingContext


def make(name: str, **values: Any) -> LabModel[Any]:
    model_class = MODELS.get(name)
    return model_class(model_class.info.hyperparameters.parse(values))


def estimator(model: LabModel[Any]) -> Any:
    """O estimador do scikit-learn por trás de um modelo do catálogo."""
    assert isinstance(model, SklearnModel)
    return model.estimator_


def classes(seed: int = 0) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(120, 3))
    return X, (X[:, 0] + X[:, 1] * X[:, 2] > 0).astype(np.int64)


def clouds(seed: int = 0) -> NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    return np.vstack(
        [rng.normal(loc=center, scale=0.5, size=(40, 2)) for center in ((0, 0), (6, 0), (3, 5))]
    )


@pytest.mark.parametrize("name", ["gradient_boosting", "adaboost"])
def test_stopping_a_boosting_keeps_exactly_the_first_stages(name: str) -> None:
    X, y = classes()

    stopped = make(name).fit(X, y, RecordingTrainingContext(stop_after=5))
    ten = make(name, n_estimators=10).fit(X, y)  # o mínimo permitido; as 5 primeiras são iguais
    replay = list(estimator(ten).staged_predict_proba(X))[4]

    assert stopped.state()["n_stages"] == 5
    assert np.allclose(stopped.predict_proba(X), replay)


def test_a_tree_grows_one_level_per_step_until_it_stops_growing() -> None:
    X, y = classes()
    context = RecordingTrainingContext()

    tree = make("decision_tree", max_depth=0, min_samples_leaf=5).fit(X, y, context)

    steps = [step for step, _, _ in context.steps]
    assert steps == list(range(1, tree.state()["depth"] + 1))
    # Cada nível acerta pelo menos tanto como o anterior.
    losses = [loss for _, loss, _ in context.steps if loss is not None]
    assert losses == sorted(losses, reverse=True)


def test_a_forest_announces_its_out_of_bag_error_as_it_grows() -> None:
    X, y = classes()
    context = RecordingTrainingContext()

    make("random_forest", n_estimators=40).fit(X, y, context)

    assert len(context.steps) == 20
    for _, loss, metrics in context.steps:
        assert loss == pytest.approx(1.0 - metrics["oob_score"])


def test_dbscan_keeps_the_training_groups_and_sends_faraway_points_to_noise() -> None:
    X = clouds()
    model = make("dbscan", eps=0.8, min_samples=5).fit(X)

    assert np.array_equal(model.predict(X), estimator(model).labels_)
    assert model.predict(np.array([[40.0, 40.0]]))[0] == -1
    assert model.state()["n_clusters"] == 3


def test_agglomerative_keeps_the_dendrogram_and_refuses_too_many_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    X = clouds()

    model = make("agglomerative", n_clusters=3).fit(X)
    linkage = np.asarray(model.state()["linkage"])

    assert linkage.shape == (len(X) - 1, 4)
    assert linkage[-1, 3] == len(X)  # a última junção junta todos os pontos
    assert np.all(np.diff(linkage[:, 2]) >= -1e-12)  # com Ward, as distâncias só sobem
    monkeypatch.setattr(agglomerative_module, "MAX_ROWS", 50)
    with pytest.raises(ModelError, match="pares"):
        make("agglomerative").fit(X)


def test_the_linkage_matrix_counts_the_points_of_each_merge() -> None:
    children = np.array([[0, 1], [2, 3], [4, 5]])

    assert linkage_matrix(children, np.array([1.0, 2.0, 3.0])) == [
        [0.0, 1.0, 1.0, 2.0],
        [2.0, 3.0, 2.0, 2.0],
        [4.0, 5.0, 3.0, 4.0],
    ]


def test_tsne_only_maps_its_training_points() -> None:
    X = clouds()
    model = make("tsne", perplexity=10).fit(X)

    assert model.transform(X).shape == (len(X), 2)
    with pytest.raises(CapabilityError, match="treinou"):
        model.transform(X[:5])
    with pytest.raises(ModelError, match="perplexidade"):
        make("tsne", perplexity=50).fit(X[:30])


def test_lof_scores_its_training_points_with_the_training_factor() -> None:
    X = np.vstack([clouds(), [[30.0, 30.0]]])
    model = make("lof").fit(X)

    scores = model.anomaly_scores(X)

    assert np.allclose(scores, -estimator(model).negative_outlier_factor_)
    assert np.argmax(scores) == len(X) - 1
    assert model.predict(X)[-1] == 1


@pytest.mark.parametrize("covariance_type", ["full", "tied", "diag", "spherical"])
def test_the_mixture_reports_full_covariance_matrices(covariance_type: str) -> None:
    X = clouds()
    model = make("gaussian_mixture", n_components=3, covariance_type=covariance_type).fit(X)

    covariances = np.asarray(model.state()["covariances"])

    assert covariances.shape == (3, 2, 2)
    assert np.allclose(covariances, covariances.transpose(0, 2, 1))  # simétricas
    assert np.allclose(np.asarray(model.predict_proba(X)).sum(axis=1), 1.0)


def test_misconfigured_models_explain_what_to_change() -> None:
    X, y = classes()

    with pytest.raises(ModelError, match="vizinhos"):
        make("knn", n_neighbors=100).fit(X[:20], y[:20])
    with pytest.raises(ModelError, match="termos"):
        make("polynomial_regression", degree=15).fit(np.zeros((10, 12)), np.zeros(10))
    with pytest.raises(ModelError, match="componentes"):
        make("pca", n_components=5).fit(X)
    with pytest.raises(ModelError, match="dois exemplos"):
        make("svm").fit(X[:11], np.array([0] * 10 + [1]))
    with pytest.raises(ModelError, match="grupos"):
        make("kmeans_numpy", n_clusters=20).fit(X[:10])


def test_the_svm_prediction_agrees_with_its_probabilities() -> None:
    X, y = classes()
    model = make("svm").fit(X, y)

    proba = model.predict_proba(X)

    assert np.array_equal(model.predict(X), np.argmax(proba, axis=1))
    assert sum(model.state()["n_support"]) == len(model.state()["support_indices"])
