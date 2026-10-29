import json

import numpy as np
import pandas as pd
import pytest

from ml_lab.core.data.cleaning.base import CleaningStep
from ml_lab.core.data.io.base import LoadedTable, SourceSpec
from ml_lab.core.data.preparation import PreparationConfig, PreparedData, prepare
from ml_lab.core.data.schema import ColumnKind
from ml_lab.core.data.serving import MAX_EXAMPLES, FittedPreparation
from ml_lab.core.data.splitting.base import SplitConfig, SplitStrategy
from ml_lab.core.data.transforms.imputation import ImputationStrategy
from ml_lab.core.data.transforms.pipeline_builder import TransformConfig
from ml_lab.core.data.transforms.scaling import ScalingMethod
from ml_lab.core.foundation.errors import DataError
from ml_lab.core.foundation.random import SeedManager

DROP_TEXT = CleaningStep("drop_columns", {"columns": ["PassengerId", "Name"]})


def prepared(frame: pd.DataFrame, **overrides: object) -> PreparedData:
    settings: dict[str, object] = {
        "target": "Survived",
        "cleaning": (DROP_TEXT,),
        "split": SplitConfig(SplitStrategy.STRATIFIED, test_size=0.25),
    }
    settings.update(overrides)
    config = PreparationConfig(**settings)  # type: ignore[arg-type]
    return prepare(
        LoadedTable("Passageiros", frame, SourceSpec("test", {})), config, seeds=SeedManager(7)
    )


def preparation_of(data: PreparedData) -> FittedPreparation:
    assert data.dataset.preparation is not None
    return data.dataset.preparation


def raw_test(data: PreparedData) -> pd.DataFrame:
    """As linhas de teste como estavam depois da limpeza, antes das transformações."""
    return data.cleaned.loc[data.dataset.test.rows]


def round_trip(preparation: FittedPreparation) -> FittedPreparation:
    """A preparação pelo caminho do pacote: dois documentos em texto JSON, e de volta."""
    inputs = json.loads(json.dumps(preparation.input_document(), allow_nan=False))
    steps = json.loads(json.dumps(preparation.preparation_document(), allow_nan=False))
    return FittedPreparation.from_documents(inputs, steps)


def test_new_rows_are_transformed_exactly_like_the_test_rows(passengers: pd.DataFrame) -> None:
    data = prepared(passengers)

    features = preparation_of(data).transform(raw_test(data))

    np.testing.assert_array_equal(features, data.dataset.test.X)


@pytest.mark.parametrize(
    "transforms",
    [
        TransformConfig(),
        TransformConfig(scaling=ScalingMethod.MINMAX, polynomial_degree=2),
        TransformConfig(binned_columns=("Fare",), n_bins=3),
        TransformConfig(numeric_imputation=ImputationStrategy.MEAN, scaling=None),
    ],
)
def test_the_preparation_survives_json_with_every_option(
    passengers: pd.DataFrame, transforms: TransformConfig
) -> None:
    data = prepared(passengers, transforms=transforms)

    restored = round_trip(preparation_of(data))

    np.testing.assert_allclose(restored.transform(raw_test(data)), data.dataset.test.X)
    assert restored.feature_names == data.dataset.feature_names


def test_cleaning_that_makes_sense_for_one_row_is_repeated(passengers: pd.DataFrame) -> None:
    clip = CleaningStep("treat_outliers", {"column": "Fare", "method": "clip", "factor": 0.5})
    data = prepared(passengers, cleaning=(DROP_TEXT, clip))
    preparation = round_trip(preparation_of(data))
    row = raw_test(data).head(1).assign(Fare=10_000.0)

    # A tarifa enorme é limitada ao limite que se calculou com a tabela, antes de escalar.
    clipped = preparation.transform(row)
    at_the_limit = preparation.transform(row.assign(Fare=preparation.cleaning[1].learned["upper"]))

    np.testing.assert_allclose(clipped, at_the_limit)
    document = preparation_of(data).preparation_document()
    assert [step["on_new_rows"] for step in document["cleaning"]] == [True, True]


def test_the_input_profile_describes_each_column(passengers: pd.DataFrame) -> None:
    columns = {column.name: column for column in preparation_of(prepared(passengers)).columns}

    assert set(columns) == {"Pclass", "Sex", "Age", "Embarked", "Fare"}
    age = columns["Age"]
    assert age.kind is ColumnKind.NUMERIC
    assert age.minimum is not None
    assert age.maximum is not None
    assert age.minimum <= 28 <= age.maximum
    assert age.can_be_missing
    assert columns["Sex"].categories == ("female", "male")
    assert columns["Sex"].typical in {"female", "male"}


def test_the_input_document_says_what_the_model_expects(passengers: pd.DataFrame) -> None:
    document = preparation_of(prepared(passengers)).input_document()

    assert document["target"] == {"name": "Survived", "classes": ["0", "1"]}
    assert document["left_out"] == ["PassengerId", "Name"]
    age = next(column for column in document["columns"] if column["name"] == "Age")
    assert set(age) >= {"min", "max", "typical", "can_be_missing", "missing_in_train"}
    sex = next(column for column in document["columns"] if column["name"] == "Sex")
    assert sex["categories"] == ["female", "male"]


def test_a_missing_column_or_value_is_explained(passengers: pd.DataFrame) -> None:
    # Sem imputação das numéricas, as linhas sem idade saem na limpeza; uma linha nova sem
    # idade já não tem quem a preencha.
    no_age = CleaningStep("drop_missing_rows", {"columns": ["Age"]})
    data = prepared(
        passengers,
        cleaning=(DROP_TEXT, no_age),
        transforms=TransformConfig(numeric_imputation=None),
    )
    preparation = preparation_of(data)
    rows = raw_test(data)

    with pytest.raises(DataError, match="Faltam colunas para o modelo: Fare"):
        preparation.transform(rows.drop(columns=["Fare"]))
    with pytest.raises(DataError, match="Ficaram valores em falta em Age"):
        preparation.transform(rows.assign(Age=np.nan))
    assert not next(c for c in preparation.columns if c.name == "Age").can_be_missing
    assert [step["on_new_rows"] for step in preparation.preparation_document()["cleaning"]] == [
        True,
        False,
    ]


def test_the_example_rows_are_the_first_test_rows_with_the_prediction(
    passengers: pd.DataFrame,
) -> None:
    data = prepared(passengers)
    preparation = preparation_of(data)

    text = preparation.examples_csv(["1", "0", "1"], n_rows=3)

    lines = text.strip().split("\n")
    assert lines[0] == "Pclass,Sex,Age,Embarked,Fare,Survived,previsto"
    assert len(lines) == 4
    assert lines[1].endswith(",1")
    assert len(preparation.examples) == min(MAX_EXAMPLES, len(data.dataset.test))


def test_dates_travel_with_the_preparation() -> None:
    rng = np.random.default_rng(3)
    frame = pd.DataFrame(
        {
            "when": pd.date_range("2024-01-01", periods=60, freq="7D"),
            "x": rng.normal(size=60),
            "y": rng.normal(size=60),
        }
    )
    data = prepare(
        LoadedTable("Datas", frame, SourceSpec("test", {})),
        PreparationConfig(target="y"),
        seeds=SeedManager(1),
    )
    preparation = round_trip(preparation_of(data))

    np.testing.assert_allclose(preparation.transform(raw_test(data)), data.dataset.test.X)
    when = next(column for column in preparation.columns if column.name == "when")
    assert when.kind is ColumnKind.DATETIME
    assert isinstance(when.minimum, str)
    assert when.minimum.startswith("2024-01")
