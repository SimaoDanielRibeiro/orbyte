"""A preparação em ONNX dá as mesmas features que a preparação do laboratório.

Cada transformação (imputar, intervalos, booleanos, one-hot, polinomiais, escalar) e a limpeza
que se repete em linhas novas (limitar os extremos) passam para ONNX: com as mesmas linhas, o ONNX
Runtime tem de dar as features que ``FittedPreparation.transform`` dá, incluindo os valores em
falta e as categorias que o treino nunca viu.
"""

import numpy as np
import pandas as pd
import pytest

from ml_lab.core.data.cleaning.base import CleaningStep
from ml_lab.core.data.io.base import LoadedTable, SourceSpec
from ml_lab.core.data.preparation import PreparationConfig, prepare
from ml_lab.core.data.serving import FittedPreparation
from ml_lab.core.data.splitting.base import SplitConfig, SplitStrategy
from ml_lab.core.data.transforms.imputation import ImputationStrategy
from ml_lab.core.data.transforms.pipeline_builder import TransformConfig
from ml_lab.core.data.transforms.scaling import ScalingMethod
from ml_lab.core.export import preparation_onnx, run_onnx
from ml_lab.core.foundation.errors import CapabilityError
from ml_lab.core.foundation.random import SeedManager

DROP_TEXT = CleaningStep("drop_columns", {"columns": ["PassengerId", "Name"]})


def passengers() -> pd.DataFrame:
    """Um "mini Titanic": números com nulos, categorias com um nulo e um booleano com um nulo."""
    return pd.DataFrame(
        {
            "PassengerId": list(range(1, 21)),
            "Name": [f"Passageiro número {i}" for i in range(1, 21)],
            "Pclass": [1, 2, 3, 3, 1, 2, 3, 3, 2, 1] * 2,
            "Sex": ["male", "female"] * 10,
            "Age": [
                *(22.0, 38.0, np.nan, 35.0, 54.0, np.nan, 2.0, 27.0, 14.0, 4.0),
                *(58.0, 20.0, 39.0, np.nan, 55.0, 31.0, 34.0, 15.0, 28.0, 8.0),
            ],
            "Embarked": ["S", "C", "S", "Q", "S", "C", "S", "S", None, "Q"] * 2,
            "Fare": [7.25, 71.28, 7.92, 53.1, 8.05, 8.46, 51.86, 21.07, 11.13, 300.0] * 2,
            "Alone": pd.array(
                [True, False, True, True, None, False, True, False, True, False] * 2,
                dtype="boolean",
            ),
            "Survived": [0, 1, 1, 1, 0, 0, 0, 1, 1, 0] * 2,
        }
    )


def fitted(**overrides: object) -> FittedPreparation:
    settings: dict[str, object] = {
        "target": "Survived",
        "cleaning": (DROP_TEXT,),
        "split": SplitConfig(SplitStrategy.STRATIFIED, test_size=0.25),
    }
    settings.update(overrides)
    data = prepare(
        LoadedTable("Passageiros", passengers(), SourceSpec("test", {})),
        PreparationConfig(**settings),  # type: ignore[arg-type]
        seeds=SeedManager(7),
    )
    assert data.dataset.preparation is not None
    return data.dataset.preparation


def new_rows(preparation: FittedPreparation) -> pd.DataFrame:
    """Linhas novas como chegam: com valores em falta e uma categoria que o treino não viu."""
    rows = passengers().drop(columns=["PassengerId", "Name", "Survived"]).head(6).copy()
    rows.loc[0, "Age"] = np.nan
    rows.loc[1, "Embarked"] = "Z"
    rows.loc[2, "Fare"] = 900.0
    rows.loc[3, "Sex"] = None
    return rows[list(preparation.input_names)]


def onnx_features(preparation: FittedPreparation, rows: pd.DataFrame) -> np.ndarray:
    return run_onnx(preparation_onnx(preparation), preparation.input_arrays(rows))["features"]


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"transforms": TransformConfig(scaling=ScalingMethod.MINMAX, polynomial_degree=2)},
        {"transforms": TransformConfig(binned_columns=("Fare",), n_bins=3)},
        {"transforms": TransformConfig(numeric_imputation=ImputationStrategy.MEAN, scaling=None)},
        {
            "cleaning": (
                DROP_TEXT,
                CleaningStep("treat_outliers", {"column": "Fare", "method": "clip"}),
            )
        },
    ],
    ids=["por-omissao", "minmax-polinomiais", "intervalos", "media-sem-escala", "extremos"],
)
def test_the_onnx_preparation_gives_the_lab_features(overrides: dict[str, object]) -> None:
    preparation = fitted(**overrides)
    rows = new_rows(preparation)

    np.testing.assert_allclose(
        onnx_features(preparation, rows), preparation.transform(rows), rtol=1e-12, atol=1e-12
    )


def test_categories_and_booleans_enter_as_text_and_as_zero_or_one() -> None:
    preparation = fitted()
    arrays = preparation.input_arrays(new_rows(preparation))

    assert arrays["Sex"].dtype == object
    assert arrays["Sex"][3, 0] == ""  # em falta: o texto vazio
    assert arrays["Embarked"][1, 0] == "Z"
    assert arrays["Age"].dtype == np.float64
    assert np.isnan(arrays["Age"][0, 0])
    assert set(np.unique(arrays["Alone"][~np.isnan(arrays["Alone"])])) <= {0.0, 1.0}


def test_dates_do_not_pass_to_onnx_yet() -> None:
    frame = passengers().assign(Boarded=pd.date_range("1912-04-10", periods=20, freq="h"))
    data = prepare(
        LoadedTable("Passageiros", frame, SourceSpec("test", {})),
        PreparationConfig(
            target="Survived",
            cleaning=(DROP_TEXT,),
            split=SplitConfig(SplitStrategy.STRATIFIED, test_size=0.25),
        ),
        seeds=SeedManager(7),
    )
    assert data.dataset.preparation is not None

    with pytest.raises(CapabilityError, match="Dates can't go to ONNX yet"):
        preparation_onnx(data.dataset.preparation)
