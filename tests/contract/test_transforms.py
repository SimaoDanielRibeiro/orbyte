"""Contrato das transformações: cada uma volta a existir, já treinada, a partir de JSON.

O pacote de um modelo leva a preparação em JSON, sem pickle (ADR 0039): o que cada
transformação aprendeu (``learned``) e a sua configuração (``params``). Aqui confirma-se, para
TODAS as transformações conhecidas, que a restaurada transforma exatamente como a original,
mesmo depois de passar por texto JSON. Uma transformação nova sem caso aqui faz falhar o
primeiro teste.
"""

import json
from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from ml_lab.core.data.transforms.base import (
    DropColumns,
    Transform,
    transform_class,
    transform_names,
)
from ml_lab.core.data.transforms.encoding import BooleanEncoder, OneHotEncoder
from ml_lab.core.data.transforms.feature_engineering import (
    Binning,
    DateFeatures,
    PolynomialFeatures,
)
from ml_lab.core.data.transforms.imputation import ImputationStrategy, Imputer
from ml_lab.core.data.transforms.scaling import MinMaxScaler, StandardScaler
from ml_lab.core.foundation.errors import InvalidParameterError

TRAIN = pd.DataFrame(
    {
        "age": [10.0, 20.0, np.nan, 40.0, 33.0],
        "fare": [7.25, 71.3, 8.05, 53.1, 13.0],
        "city": ["Porto", "Lisboa", "Porto", None, "Braga"],
        "member": [True, False, True, True, False],
        "when": pd.to_datetime(
            pd.Series(["2024-01-05", "2024-03-10", None, "2023-12-24", "2024-07-01"])
        ),
    }
)
NEW = pd.DataFrame(
    {
        "age": [np.nan, 1000.0, 18.0],
        "fare": [0.0, 512.3, 20.0],
        "city": ["Faro", "Lisboa", None],
        "member": [False, True, True],
        "when": pd.to_datetime(pd.Series(["2025-02-28", None, "2020-01-01"])),
    }
)

CASES: dict[str, Callable[[], Transform]] = {
    "drop_columns": lambda: DropColumns(["when"], "as datas não entram."),
    "date_features": lambda: DateFeatures(["when"]),
    "impute": lambda: Imputer(["age", "fare"], ImputationStrategy.MEDIAN),
    "binning": lambda: Binning(["fare"], n_bins=3),
    "boolean_to_number": lambda: BooleanEncoder(["member"]),
    "one_hot": lambda: OneHotEncoder(["city"]),
    "polynomial_features": lambda: PolynomialFeatures(["age", "fare"], degree=2),
    "standard_scaler": lambda: StandardScaler(["fare"]),
    "minmax_scaler": lambda: MinMaxScaler(["fare"]),
}
"""Uma transformação de cada tipo, com colunas que ela sabe tratar."""


def test_every_transform_has_a_case_here() -> None:
    assert set(transform_names()) == set(CASES)


@pytest.mark.parametrize("name", sorted(CASES))
def test_a_restored_transform_transforms_like_the_original(name: str) -> None:
    original = CASES[name]().fit(TRAIN)
    # Pela mesma estrada do pacote: texto JSON, e de volta.
    params = json.loads(json.dumps(original.params()))
    learned = json.loads(json.dumps(original.learned()))

    restored = transform_class(name).restore(params, learned)

    assert restored.is_fitted
    pd.testing.assert_frame_equal(restored.transform(NEW), original.transform(NEW))
    assert restored.describe() == original.describe()
    assert restored.lineage_step() == original.lineage_step()


@pytest.mark.parametrize("name", sorted(CASES))
def test_params_are_the_constructor_arguments(name: str) -> None:
    transform = CASES[name]()

    again = type(transform)(**transform.params())

    assert again.params() == transform.params()


def test_an_imputer_keeps_the_fill_values_as_plain_json_values() -> None:
    imputer = Imputer(["city", "member"], ImputationStrategy.MOST_FREQUENT).fit(TRAIN)

    learned = imputer.learned()

    assert learned["fill_values"] == {"city": "Porto", "member": True}
    assert type(learned["fill_values"]["member"]) is bool
    assert learned["missing_in_train"] == {"city": 1, "member": 0}


def test_an_unfitted_transform_has_nothing_learned_yet() -> None:
    with pytest.raises(Exception, match="ainda não aprendeu"):
        StandardScaler(["fare"]).learned()


def test_an_unknown_or_incomplete_transform_is_explained() -> None:
    with pytest.raises(InvalidParameterError, match="desconhecida: 'magia'"):
        transform_class("magia")
    with pytest.raises(InvalidParameterError, match="Não foi possível restaurar"):
        StandardScaler.restore({"columns": ["fare"]}, {"means": {"fare": 1.0}})
