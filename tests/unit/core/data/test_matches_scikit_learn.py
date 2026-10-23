"""As implementações didáticas da camada de dados dão o mesmo que o scikit-learn.

A camada de dados não usa o scikit-learn (ADR 0008): imputação, codificação, escala,
PCA, divisão e dados sintéticos estão escritos em pandas e NumPy. Estes testes são o
"teste extra" das implementações didáticas (secção "Testes" de docs/architecture.md):
no mesmo dataset, o resultado tem de coincidir com o da versão oficial.
"""

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn import decomposition
from sklearn.datasets import make_circles, make_moons
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler as SklearnMinMaxScaler
from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder
from sklearn.preprocessing import StandardScaler as SklearnStandardScaler

from ml_lab.core.data.io.synthetic_source import TARGET, SyntheticSource
from ml_lab.core.data.projection.pca import fit_pca
from ml_lab.core.data.splitting.base import SplitConfig, partition_sizes
from ml_lab.core.data.transforms.encoding import OneHotEncoder
from ml_lab.core.data.transforms.imputation import ImputationStrategy, Imputer
from ml_lab.core.data.transforms.scaling import MinMaxScaler, StandardScaler

RNG = np.random.default_rng(42)
TRAIN = pd.DataFrame({"a": RNG.normal(10, 3, 50), "b": RNG.exponential(2, 50)})
TEST = pd.DataFrame({"a": RNG.normal(10, 3, 20), "b": RNG.exponential(2, 20)})


def with_gaps(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.mask(pd.DataFrame(RNG.random(frame.shape) < 0.2, columns=frame.columns))


def test_standard_scaling() -> None:
    ours = StandardScaler(["a", "b"]).fit(TRAIN)
    theirs = SklearnStandardScaler().fit(TRAIN.to_numpy())

    assert np.allclose(ours.transform(TEST).to_numpy(), theirs.transform(TEST.to_numpy()))


def test_minmax_scaling() -> None:
    ours = MinMaxScaler(["a", "b"]).fit(TRAIN)
    theirs = SklearnMinMaxScaler().fit(TRAIN.to_numpy())

    assert np.allclose(ours.transform(TEST).to_numpy(), theirs.transform(TEST.to_numpy()))


@pytest.mark.parametrize("strategy", [ImputationStrategy.MEAN, ImputationStrategy.MEDIAN])
def test_numeric_imputation(strategy: ImputationStrategy) -> None:
    train, test = with_gaps(TRAIN), with_gaps(TEST)

    ours = Imputer(["a", "b"], strategy).fit(train)
    theirs = SimpleImputer(strategy=strategy.value).fit(train.to_numpy())

    assert np.allclose(ours.transform(test).to_numpy(), theirs.transform(test.to_numpy()))


def test_most_frequent_imputation_of_text() -> None:
    train = pd.DataFrame({"city": ["Porto", None, "Lisboa", "Porto", None]})
    test = pd.DataFrame({"city": [None, "Faro"]})

    ours = Imputer(["city"], ImputationStrategy.MOST_FREQUENT).fit(train)
    theirs = SimpleImputer(strategy="most_frequent").fit(train.astype(object).to_numpy())

    assert (
        ours.transform(test)["city"].tolist()
        == theirs.transform(test.astype(object).to_numpy()).ravel().tolist()
    )


def test_one_hot_encoding_with_unknown_categories() -> None:
    train = pd.DataFrame({"port": ["S", "C", "Q", "S"], "sex": ["m", "f", "f", "m"]})
    test = pd.DataFrame({"port": ["C", "X"], "sex": ["f", "m"]})

    ours = OneHotEncoder(["port", "sex"]).fit(train)
    theirs = SklearnOneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(train)

    assert ours.transform(test).columns.tolist() == theirs.get_feature_names_out().tolist()
    assert np.array_equal(ours.transform(test).to_numpy(), theirs.transform(test))


def test_pca_including_the_sign_of_each_component() -> None:
    values = RNG.normal(size=(80, 4)) @ RNG.normal(size=(4, 4))

    ours = fit_pca(values, n_components=3)
    theirs = decomposition.PCA(n_components=3, svd_solver="full").fit(values)

    assert np.allclose(ours.explained_variance_ratio, theirs.explained_variance_ratio_)
    assert np.allclose(ours.components, theirs.components_)
    assert np.allclose(ours.transform(values), theirs.transform(values))


@pytest.mark.parametrize(("n_rows", "test_size"), [(891, 0.2), (150, 0.25), (101, 0.33), (7, 0.5)])
def test_split_sizes(n_rows: int, test_size: float) -> None:
    n_train, _, n_test = partition_sizes(n_rows, SplitConfig(test_size=test_size))

    train, test = train_test_split(np.arange(n_rows), test_size=test_size, random_state=0)

    assert (n_train, n_test) == (len(train), len(test))


@pytest.mark.parametrize(
    ("generator", "make", "extra"),
    [("moons", make_moons, {}), ("circles", make_circles, {"factor": 0.5})],
)
def test_synthetic_shapes_without_noise(
    generator: str, make: Callable[..., tuple[Any, Any]], extra: dict[str, float]
) -> None:
    frame = (
        SyntheticSource(generator, seed=0, parameters={"n_samples": 100, "noise": 0.0, **extra})
        .load()
        .frame
    )
    X, y = make(n_samples=100, noise=0.0, shuffle=False, **extra)

    ours = np.column_stack([frame["x1"], frame["x2"], frame[TARGET]])
    theirs = np.column_stack([X, y])

    def ordered(points: np.ndarray) -> np.ndarray:
        return points[np.lexsort(np.round(points, 9).T[::-1])]

    # Os nossos pontos vêm baralhados; ordenados, são exatamente os mesmos.
    assert np.allclose(ordered(ours), ordered(theirs))
    assert math.isclose(frame[TARGET].mean(), 0.5)
