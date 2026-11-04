"""O espaço de procura: o que os agentes sorteiam em cada modelo, lido do ``x-search``."""

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from pydantic import Field

from ml_lab.core.automl.search_space import search_space, written_value
from ml_lab.core.foundation.errors import ModelError
from ml_lab.core.models.base import MODELS, LabModel
from ml_lab.core.schemas.hyperparameters import Hyperparameters


def test_the_space_comes_from_the_hyperparameters_of_the_model() -> None:
    space = search_space(MODELS.get("knn"))

    assert [dimension.name for dimension in space.dimensions] == [
        "n_neighbors",
        "weights",
        "metric",
    ]
    neighbors = space.dimensions[0]
    assert (neighbors.low, neighbors.high, neighbors.integer) == (1.0, 30.0, True)
    assert space.dimensions[1].values == ("uniform", "distance")


def test_integers_are_drawn_whole_and_within_the_range() -> None:
    neighbors = search_space(MODELS.get("knn")).dimensions[0]
    rng = np.random.default_rng(0)

    drawn = [neighbors.sample(rng) for _ in range(500)]

    assert all(isinstance(value, int) and 1 <= value <= 30 for value in drawn)
    assert {1, 30} <= set(drawn)  # os extremos também saem


def test_a_logarithmic_range_is_drawn_in_the_exponents() -> None:
    c = search_space(MODELS.get("svm")).dimensions[0]  # o C: de 0,1 a 100, em escala log
    rng = np.random.default_rng(0)

    drawn = np.array([c.sample(rng) for _ in range(3000)])

    assert c.log
    assert drawn.min() >= 0.1
    assert drawn.max() <= 100
    # Cada década (0,1-1, 1-10, 10-100) sai um terço das vezes: numa escala linear, 90% dos
    # sorteios cairiam entre 10 e 100.
    assert np.mean(drawn < 1) == pytest.approx(1 / 3, abs=0.04)
    assert np.mean(drawn > 10) == pytest.approx(1 / 3, abs=0.04)


def test_drawn_numbers_keep_three_significant_digits() -> None:
    gamma = search_space(MODELS.get("svm")).dimensions[2]
    rng = np.random.default_rng(1)

    for _ in range(50):
        value = gamma.sample(rng)
        assert value == float(f"{value:.3g}")


def test_a_list_of_values_counts_its_combinations() -> None:
    space = search_space(MODELS.get("decision_tree"))

    assert space.size == 8 * 5 * 2  # profundidades × mínimo por folha × impureza
    assert search_space(MODELS.get("svm")).size is None  # com números contínuos, não há fim


def test_a_model_with_nothing_to_search_has_an_empty_space() -> None:
    space = search_space(MODELS.get("linear_regression"))

    assert space.is_empty
    assert space.sample(np.random.default_rng(0)) == {}


def test_the_space_describes_itself_in_plain_words() -> None:
    descriptions = [
        dimension.description for dimension in search_space(MODELS.get("svm")).dimensions
    ]

    assert descriptions[0] == "C (cost of errors): from 0.1 to 100, on a logarithmic scale"
    assert descriptions[1] == "Kernel: RBF (Gaussian), Linear"  # os nomes do x-labels
    assert written_value(True) == "yes"
    assert written_value(0.0128) == "0.0128"
    assert written_value(0.00001) == "1 × 10⁻⁵"


class _BadParams(Hyperparameters):
    """Um x-search mal escrito: o mínimo a seguir ao máximo."""

    alpha: float = Field(default=1.0, json_schema_extra={"x-search": {"range": [10, 1]}})


class _EmptyParams(Hyperparameters):
    """Um x-search sem valores."""

    kind: str = Field(default="a", json_schema_extra={"x-search": {"values": []}})


def _fake(params: type[Hyperparameters]) -> type[LabModel[Any]]:
    """Um "modelo" com só o que o espaço de procura lê: o nome e os hiperparâmetros."""
    info = SimpleNamespace(name="falso", hyperparameters=params)
    return cast("type[LabModel[Any]]", SimpleNamespace(info=info))


@pytest.mark.parametrize("params", [_BadParams, _EmptyParams], ids=["range", "values"])
def test_a_badly_written_search_is_a_model_error(params: type[Hyperparameters]) -> None:
    with pytest.raises(ModelError, match="x-search of falso"):
        search_space(_fake(params))
