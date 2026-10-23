import numpy as np
import pandas as pd
import pytest

from ml_lab.core.data.schema import ColumnKind, TableSchema
from ml_lab.core.data.transforms.base import DropColumns
from ml_lab.core.data.transforms.encoding import BooleanEncoder, OneHotEncoder, to_zero_one
from ml_lab.core.data.transforms.feature_engineering import (
    Binning,
    DateFeatures,
    PolynomialFeatures,
)
from ml_lab.core.data.transforms.imputation import ImputationStrategy, Imputer
from ml_lab.core.data.transforms.pipeline_builder import TransformConfig, build_pipeline
from ml_lab.core.data.transforms.scaling import MinMaxScaler, ScalingMethod, StandardScaler
from ml_lab.core.foundation.errors import (
    ColumnNotFoundError,
    DataError,
    InvalidParameterError,
    NotFittedError,
)

TRAIN = pd.DataFrame(
    {
        "age": [10.0, 20.0, np.nan, 40.0],
        "city": ["Porto", "Lisboa", "Porto", None],
        "member": [True, False, True, True],
    }
)
TEST = pd.DataFrame({"age": [np.nan, 1000.0], "city": ["Faro", "Lisboa"], "member": [False, True]})


# --- o contrato -------------------------------------------------------------------------


def test_transforms_learn_only_from_the_training_data() -> None:
    # O teste tem um valor enorme; se a mediana o visse, mudava. Não pode mudar.
    imputer = Imputer(["age"], ImputationStrategy.MEDIAN).fit(TRAIN)

    result = imputer.transform(TEST)

    assert imputer.fill_values == {"age": 20.0}
    assert result["age"].tolist() == [20.0, 1000.0]


def test_transforms_never_change_their_input() -> None:
    original = TRAIN.copy()

    StandardScaler(["age"]).fit_transform(TRAIN.fillna({"age": 0.0}))
    OneHotEncoder(["city"]).fit_transform(TRAIN)

    assert TRAIN.equals(original)


def test_using_a_transform_before_fit_is_explained() -> None:
    with pytest.raises(NotFittedError, match="ainda não aprendeu"):
        Imputer(["age"], ImputationStrategy.MEAN).transform(TRAIN)
    with pytest.raises(NotFittedError):
        StandardScaler(["age"]).lineage_step()


def test_missing_columns_are_reported() -> None:
    with pytest.raises(ColumnNotFoundError, match="Querias dizer 'age'"):
        Imputer(["agee"], ImputationStrategy.MEAN).fit(TRAIN)


def test_the_lineage_step_lists_the_created_and_removed_columns() -> None:
    encoder = OneHotEncoder(["city"]).fit(TRAIN)

    step = encoder.lineage_step()

    assert step.columns_added == ("city_Lisboa", "city_Porto")
    assert step.columns_removed == ("city",)
    assert step.rows_before == step.rows_after == 4
    assert "Porto" in step.description


# --- cada transformação -----------------------------------------------------------------


def test_imputation_strategies() -> None:
    mean = Imputer(["age"], ImputationStrategy.MEAN).fit(TRAIN)
    frequent = Imputer(["city"], ImputationStrategy.MOST_FREQUENT).fit(TRAIN)

    assert mean.fill_values["age"] == pytest.approx(70 / 3)
    assert frequent.fill_values["city"] == "Porto"
    assert frequent.transform(TRAIN)["city"].tolist() == ["Porto", "Lisboa", "Porto", "Porto"]
    assert "1 nulo" in mean.describe()


def test_columns_without_missing_values_in_training_are_explained() -> None:
    imputer = Imputer(["member"], ImputationStrategy.MOST_FREQUENT).fit(TRAIN)

    assert imputer.describe().startswith("Nenhuma das 1 coluna tinha nulos no treino")


def test_one_hot_uses_the_training_categories_and_zeros_for_new_ones() -> None:
    encoder = OneHotEncoder(["city"]).fit(TRAIN)

    encoded = encoder.transform(TEST)

    assert encoded.columns.tolist() == ["age", "city_Lisboa", "city_Porto", "member"]
    assert encoded[["city_Lisboa", "city_Porto"]].to_numpy().tolist() == [[0, 0], [1, 0]]
    assert encoder.unknown_counts(TEST) == {"city": 1}


def test_booleans_become_zero_and_one() -> None:
    assert BooleanEncoder(["member"]).fit_transform(TRAIN)["member"].tolist() == [1, 0, 1, 1]
    assert to_zero_one(pd.Series(["Sim", "não", None])).tolist()[:2] == [1.0, 0.0]


def test_standard_scaling_uses_the_population_deviation_of_the_training_data() -> None:
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "constante": [5.0] * 4})

    scaled = StandardScaler(["x", "constante"]).fit_transform(frame)

    assert scaled["x"].mean() == pytest.approx(0.0)
    assert scaled["x"].std(ddof=0) == pytest.approx(1.0)
    assert scaled["constante"].tolist() == [0.0] * 4


def test_minmax_scaling_maps_the_training_range_to_zero_one() -> None:
    scaler = MinMaxScaler(["x"]).fit(pd.DataFrame({"x": [10.0, 20.0, 30.0]}))

    assert scaler.transform(pd.DataFrame({"x": [10.0, 30.0, 40.0]}))["x"].tolist() == [0, 1, 1.5]


def test_dates_become_numeric_parts() -> None:
    frame = pd.DataFrame({"dia": pd.to_datetime(["2024-12-25", "2023-02-01"])})

    parts = DateFeatures(["dia"]).fit_transform(frame)

    assert parts.columns.tolist() == ["dia_year", "dia_month", "dia_day", "dia_day_of_week"]
    assert parts.iloc[0].tolist() == [2024, 12, 25, 2]  # 25/12/2024 foi uma quarta-feira
    with pytest.raises(DataError, match="não é do tipo data"):
        DateFeatures(["dia"]).fit(pd.DataFrame({"dia": ["2024-12-25"]}))


def test_polynomial_features_add_powers_and_products() -> None:
    frame = pd.DataFrame({"a": [2.0, 3.0], "b": [5.0, 7.0]})

    expanded = PolynomialFeatures(["a", "b"], 2).fit_transform(frame)

    assert expanded.columns.tolist() == ["a", "b", "a^2", "a·b", "b^2"]
    assert expanded.iloc[1].tolist() == [3, 7, 9, 21, 49]
    assert len(PolynomialFeatures(["a", "b"], 3).output_names()) == 7
    with pytest.raises(InvalidParameterError):
        PolynomialFeatures(["a"], 5)


def test_binning_uses_training_quantiles_and_keeps_missing_values() -> None:
    binning = Binning(["x"], n_bins=4).fit(pd.DataFrame({"x": [float(v) for v in range(1, 9)]}))

    result = binning.transform(pd.DataFrame({"x": [0.0, 4.0, 100.0, np.nan]}))

    assert binning.edges["x"] == (1.0, 2.75, 4.5, 6.25, 8.0)
    assert result.columns.tolist() == ["x_bin"]
    assert result["x_bin"].tolist()[:3] == [0.0, 1.0, 3.0]
    assert np.isnan(result["x_bin"].iloc[3])


def test_drop_columns_explains_why() -> None:
    dropper = DropColumns(["city"], "não interessa.").fit(TRAIN)

    assert "city" not in dropper.transform(TRAIN).columns
    assert dropper.describe() == "Retiradas city: não interessa."


# --- pipeline -----------------------------------------------------------------------------


def schema() -> TableSchema:
    return TableSchema.of(
        {
            "age": ColumnKind.NUMERIC,
            "city": ColumnKind.CATEGORICAL,
            "member": ColumnKind.BOOLEAN,
            "when": ColumnKind.DATETIME,
        }
    )


def frame_with_dates(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.assign(when=pd.date_range("2024-01-01", periods=len(frame)))


def test_the_default_pipeline_turns_every_column_into_numbers() -> None:
    pipeline = build_pipeline(schema(), TransformConfig())

    train = pipeline.fit_transform(frame_with_dates(TRAIN))
    test = pipeline.transform(frame_with_dates(TEST))

    assert [type(step).__name__ for step in pipeline.steps] == [
        "DateFeatures",
        "Imputer",
        "Imputer",
        "BooleanEncoder",
        "OneHotEncoder",
        "StandardScaler",
    ]
    assert all(pd.api.types.is_numeric_dtype(train[c]) for c in train.columns)
    assert not train.isna().any().any()
    assert train.columns.tolist() == test.columns.tolist()
    assert pipeline.unknown_categories(frame_with_dates(TEST)) == {"city": 1}
    assert len(pipeline.lineage_steps()) == 6


def test_options_change_the_pipeline() -> None:
    config = TransformConfig(
        numeric_imputation=None,
        scaling=ScalingMethod.MINMAX,
        date_features=False,
        polynomial_degree=2,
        binned_columns=("age",),
    )

    steps = [type(step).__name__ for step in build_pipeline(schema(), config).steps]

    assert steps == ["DropColumns", "Imputer", "Binning", "BooleanEncoder", "OneHotEncoder"]


def test_transform_config_accepts_json_values_and_validates_them() -> None:
    config = TransformConfig(numeric_imputation="mean", scaling="minmax")  # type: ignore[arg-type]

    assert config.numeric_imputation is ImputationStrategy.MEAN
    assert config.scaling is ScalingMethod.MINMAX
    with pytest.raises(InvalidParameterError, match="média ou a mediana"):
        TransformConfig(numeric_imputation=ImputationStrategy.MOST_FREQUENT)
    with pytest.raises(InvalidParameterError):
        TransformConfig(polynomial_degree=4)
    with pytest.raises(InvalidParameterError, match="Só as features numéricas"):
        build_pipeline(schema(), TransformConfig(binned_columns=("city",)))
