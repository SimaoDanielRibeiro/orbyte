"""As implementações didáticas da fase 4 dão o mesmo que o scikit-learn.

O perceptrão (sem baralhar), o KNN, o k-means (com os mesmos pontos de partida) e a
PCA fazem as mesmas contas que as versões do scikit-learn, por isso os resultados
coincidem até à precisão da máquina.
"""

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier

from ml_lab.core.models.base import MODELS, LabModel


def make(name: str, **values: Any) -> LabModel[Any]:
    model_class = MODELS.get(name)
    return model_class(model_class.info.hyperparameters.parse(values))


def blobs(
    rng: np.random.Generator, n_classes: int = 3
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    centers = rng.uniform(-4, 4, size=(n_classes, 2))
    y = np.repeat(np.arange(n_classes), 30)
    return centers[y] + rng.normal(size=(len(y), 2)), y


def test_the_perceptron_learns_the_scikit_learn_weights_when_not_shuffling() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 3))
    y = (X @ np.array([1.0, -2.0, 0.5]) + 0.3 > 0).astype(np.int64)

    ours = make("perceptron", shuffle=False, max_epochs=50).fit(X, y)
    theirs = Perceptron(shuffle=False, eta0=1.0, max_iter=50, tol=None).fit(X, y)

    assert np.array_equal(ours.state()["coefficients"], theirs.coef_)
    assert np.array_equal(ours.state()["intercept"], theirs.intercept_)
    assert ours.state()["converged"]  # separáveis: pára quando deixa de errar


def test_the_multiclass_perceptron_is_one_against_the_others() -> None:
    X, y = blobs(np.random.default_rng(1))

    ours = make("perceptron", shuffle=False, max_epochs=30).fit(X, y)
    theirs = Perceptron(shuffle=False, eta0=1.0, max_iter=30, tol=None).fit(X, y)

    assert np.allclose(ours.state()["coefficients"], theirs.coef_)
    assert np.array_equal(ours.predict(X), theirs.predict(X))


@pytest.mark.parametrize("weights", ["uniform", "distance"])
@pytest.mark.parametrize("metric", ["euclidean", "manhattan"])
def test_the_numpy_knn_votes_like_scikit_learn(weights: str, metric: str) -> None:
    rng = np.random.default_rng(2)
    X = rng.normal(size=(200, 4))
    y = (X[:, 0] + X[:, 1] ** 2 > 1).astype(np.int64)
    new = rng.normal(size=(100, 4))

    ours = make("knn_numpy", n_neighbors=7, weights=weights, metric=metric).fit(X, y)
    theirs = KNeighborsClassifier(n_neighbors=7, weights=weights, metric=metric).fit(X, y)

    assert np.array_equal(ours.predict(new), theirs.predict(new))
    assert np.allclose(ours.predict_proba(new), theirs.predict_proba(new))


def test_numpy_kmeans_from_the_same_start_ends_where_scikit_learn_ends() -> None:
    rng = np.random.default_rng(3)
    X = np.vstack([rng.normal(loc=center, size=(50, 2)) for center in ((0, 0), (5, 5), (0, 6))])

    ours = make("kmeans_numpy", n_clusters=3, init="random").fit(X)
    start = np.asarray(ours.state()["initial_centers"])
    theirs = KMeans(n_clusters=3, init=start, n_init=1, algorithm="lloyd", tol=1e-4).fit(X)

    assert np.allclose(ours.state()["centers"], theirs.cluster_centers_)
    assert ours.state()["inertia"] == pytest.approx(theirs.inertia_)
    assert ours.state()["iterations"] == theirs.n_iter_
    assert np.array_equal(ours.predict(X), theirs.predict(X))


def test_kmeans_plus_plus_spreads_the_starting_centers() -> None:
    rng = np.random.default_rng(4)
    X = np.vstack(
        [rng.normal(loc=center, scale=0.3, size=(50, 2)) for center in ((0, 0), (8, 8), (0, 8))]
    )

    start = np.asarray(make("kmeans_numpy", n_clusters=3).fit(X).state()["initial_centers"])

    nearest_group = {
        int(np.argmin([np.linalg.norm(s - c) for c in ((0, 0), (8, 8), (0, 8))])) for s in start
    }
    assert nearest_group == {0, 1, 2}  # um ponto de partida em cada nuvem


def test_numpy_pca_finds_the_scikit_learn_components_with_the_same_signs() -> None:
    rng = np.random.default_rng(5)
    X = rng.normal(size=(100, 5)) @ rng.normal(size=(5, 5))

    ours = make("pca_numpy", n_components=3).fit(X)
    theirs = PCA(n_components=3).fit(X)

    assert np.allclose(ours.state()["components"], theirs.components_)
    assert np.allclose(ours.state()["explained_variance_ratio"], theirs.explained_variance_ratio_)
    assert np.allclose(ours.transform(X), theirs.transform(X))
