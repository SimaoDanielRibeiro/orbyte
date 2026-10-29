import numpy as np
import pandas as pd
import pytest

from ml_lab.core.data.cleaning.base import CLEANING, CleaningStep, apply_cleaning
from ml_lab.core.data.cleaning.outliers import iqr_bounds
from ml_lab.core.data.cleaning.suggestions import suggest_cleaning
from ml_lab.core.data.lineage import Stage
from ml_lab.core.data.profiling.dataset_profile import profile_dataset
from ml_lab.core.foundation.errors import (
    ColumnNotFoundError,
    ComponentNotFoundError,
    DataError,
    InvalidParameterError,
)


def clean(frame: pd.DataFrame, operation: str, **params: object) -> pd.DataFrame:
    return apply_cleaning(frame, [CleaningStep(operation, params)]).frame


def test_every_operation_is_registered_with_a_title() -> None:
    assert CLEANING.names() == [
        "convert_type",
        "drop_columns",
        "drop_duplicates",
        "drop_missing_rows",
        "treat_outliers",
    ]
    assert all(operation.title and operation.description for _, operation in CLEANING.items())


def test_the_recipe_is_applied_in_order_and_recorded(passengers: pd.DataFrame) -> None:
    original = passengers.copy()
    steps = [
        CleaningStep("drop_columns", {"columns": ["PassengerId", "Name", "Age"]}),
        CleaningStep("drop_duplicates"),
        CleaningStep("drop_missing_rows", {"columns": ["Embarked"]}),
    ]

    cleaned = apply_cleaning(passengers, steps)

    assert passengers.equals(original)  # o original nunca muda
    assert [step.operation for step in cleaned.lineage] == [s.operation for s in steps]
    assert all(step.stage is Stage.CLEANING for step in cleaned.lineage)
    assert cleaned.lineage[0].columns_removed == ("PassengerId", "Name", "Age")
    # Sem estas três colunas, a segunda metade da tabela repete a primeira: é por isso que
    # os duplicados se procuram antes de remover os identificadores.
    assert cleaned.lineage[1].rows_removed == 10
    assert cleaned.lineage[2].rows_removed == 1
    assert len(cleaned.frame) == 9


def test_bad_recipes_are_explained(passengers: pd.DataFrame) -> None:
    with pytest.raises(ComponentNotFoundError, match="drop_duplicates"):
        clean(passengers, "drop_duplicate")
    with pytest.raises(InvalidParameterError, match="drop_columns"):
        clean(passengers, "drop_columns", cols=["Age"])
    with pytest.raises(ColumnNotFoundError, match="Cabin"):
        clean(passengers, "drop_columns", columns=["Cabin"])
    with pytest.raises(InvalidParameterError, match="pelo menos uma"):
        clean(passengers, "drop_columns", columns=[])


def test_duplicates_can_be_compared_on_some_columns(passengers: pd.DataFrame) -> None:
    assert len(clean(passengers, "drop_duplicates")) == 20
    assert len(clean(passengers, "drop_duplicates", columns=["Sex"])) == 2


def test_lineage_uses_singular_and_plural() -> None:
    frame = pd.DataFrame({"a": [1, 1, 2]})

    description = apply_cleaning(frame, [CleaningStep("drop_duplicates")]).lineage[0].description

    assert description == "Removida 1 linha duplicada."


def test_rows_with_missing_values_in_every_column_only() -> None:
    frame = pd.DataFrame({"a": [1.0, np.nan, np.nan], "b": [np.nan, np.nan, 3.0]})

    assert len(clean(frame, "drop_missing_rows", how="all")) == 2
    assert len(clean(frame, "drop_missing_rows")) == 0


# --- conversões de tipo ------------------------------------------------------------------


def test_text_with_decimal_commas_becomes_numbers_and_failures_are_counted() -> None:
    frame = pd.DataFrame({"preço": ["1,5", "2", "n/d", None]})

    result = apply_cleaning(
        frame, [CleaningStep("convert_type", {"column": "preço", "to": "numeric", "decimal": ","})]
    )

    assert result.frame["preço"].tolist()[:2] == [1.5, 2.0]
    assert result.frame["preço"].isna().sum() == 2
    assert "1 valor não se converteu" in result.lineage[0].description
    assert result.lineage[0].columns_changed == ("preço",)
    assert [notice.code for notice in result.notices] == ["conversion_failures"]


def test_dates_booleans_categories_and_text() -> None:
    frame = pd.DataFrame(
        {"data": ["25/12/2024", "01/02/2023"], "ativo": ["sim", "Não"], "cor": ["azul", "verde"]}
    )

    dates = clean(frame, "convert_type", column="data", to="datetime", date_format="%d/%m/%Y")
    booleans = clean(frame, "convert_type", column="ativo", to="boolean")
    categories = clean(frame, "convert_type", column="cor", to="category")
    numbers_as_text = clean(pd.DataFrame({"n": [1, 2]}), "convert_type", column="n", to="text")

    assert dates["data"].dt.year.tolist() == [2024, 2023]
    assert booleans["ativo"].tolist() == [True, False]
    assert isinstance(categories["cor"].dtype, pd.CategoricalDtype)
    assert numbers_as_text["n"].tolist() == ["1", "2"]


def test_unknown_target_types_are_rejected() -> None:
    with pytest.raises(InvalidParameterError, match="desconhecido"):
        clean(pd.DataFrame({"a": [1]}), "convert_type", column="a", to="complex")


# --- valores extremos ----------------------------------------------------------------


def test_iqr_bounds() -> None:
    bounds = iqr_bounds(pd.Series([1.0, 2.0, 3.0, 4.0, 100.0]))

    assert bounds is not None
    assert (bounds.lower, bounds.upper) == (-1.0, 7.0)
    assert (bounds.n_below, bounds.n_above) == (0, 1)
    assert iqr_bounds(pd.Series([np.nan])) is None


def test_outliers_are_clipped_or_removed() -> None:
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 100.0]})

    clipped = apply_cleaning(frame, [CleaningStep("treat_outliers", {"column": "x"})])
    removed = clean(frame, "treat_outliers", column="x", method="remove")

    assert clipped.frame["x"].max() == 7.0
    assert "1 valor extremo limitado" in clipped.lineage[0].description
    assert removed["x"].tolist() == [1.0, 2.0, 3.0, 4.0]


def test_outliers_need_a_numeric_column(passengers: pd.DataFrame) -> None:
    with pytest.raises(DataError, match="não é numérica"):
        clean(passengers, "treat_outliers", column="Sex")
    with pytest.raises(InvalidParameterError, match="Método"):
        clean(passengers, "treat_outliers", column="Fare", method="cut")


# --- sugestões -----------------------------------------------------------------------------


def test_suggestions_put_duplicates_first_then_conversions_then_columns() -> None:
    frame = pd.DataFrame(
        {
            "id": list(range(1, 13)),
            "preço": ["1,5", "2,5"] * 6,
            "vazia": [np.nan] * 12,
            "grupo": ["a", "b", "c"] * 4,
            "alvo": [0, 1] * 6,
        }
    )
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)

    suggestions = suggest_cleaning(profile_dataset(frame), target="alvo")

    assert [(s.step.operation, s.column) for s in suggestions] == [
        ("drop_duplicates", None),
        ("convert_type", "preço"),
        ("drop_columns", "vazia"),
    ]
    assert suggestions[1].step.params == {"column": "preço", "to": "numeric", "decimal": ","}


def test_titanic_like_suggestions(passengers: pd.DataFrame) -> None:
    suggestions = suggest_cleaning(profile_dataset(passengers), target="Survived")

    assert [s.column for s in suggestions] == ["PassengerId", "Name"]


def test_the_target_is_never_suggested_for_removal() -> None:
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0], "alvo": [np.nan, np.nan, 1.0]})

    assert suggest_cleaning(profile_dataset(frame), target="alvo") == ()


# --- linhas novas (o pacote de um modelo, ADR 0039) ----------------------------------------


def test_the_recipe_keeps_what_each_step_learned() -> None:
    frame = pd.DataFrame({"fare": [1.0, 2.0, 2.5, 3.0, 100.0], "city": ["a", "b", "a", "b", "a"]})
    clip = CleaningStep("treat_outliers", {"column": "fare", "method": "clip"})
    dropped = CleaningStep("drop_columns", {"columns": ["city"]})

    cleaned = apply_cleaning(frame, [clip, dropped])

    learned = cleaned.applied[0].learned
    assert learned["lower"] < 1.0
    assert 3.0 < learned["upper"] < 100.0
    assert cleaned.applied[1].learned == {}


def test_new_rows_repeat_only_the_steps_that_make_sense_for_one_row() -> None:
    frame = pd.DataFrame(
        {
            "fare": [1.0, 2.0, 2.5, 3.0, 100.0],
            "price": ["1,5", "2,0", "3,5", "4,0", "5,5"],
            "city": ["a", "b", "a", "b", "a"],
        }
    )
    steps = [
        CleaningStep("treat_outliers", {"column": "fare", "method": "clip"}),
        CleaningStep("convert_type", {"column": "price", "to": "numeric", "decimal": ","}),
        CleaningStep("drop_columns", {"columns": ["city"]}),
        CleaningStep("drop_duplicates"),
    ]
    applied = apply_cleaning(frame, steps).applied
    new = pd.DataFrame({"fare": [500.0], "price": ["7,25"]})

    for step in applied:
        new = step.replay(new)

    assert new["fare"].iloc[0] == applied[0].learned["upper"]
    assert new["price"].iloc[0] == 7.25
    assert [step.replayable for step in applied] == [True, True, True, False]
