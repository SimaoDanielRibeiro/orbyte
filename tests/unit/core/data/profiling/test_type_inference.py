import numpy as np
import pandas as pd
import pytest

from ml_lab.core.data.profiling.type_inference import (
    infer_column,
    infer_schema,
    looks_like_an_identifier_name,
    looks_like_classes,
)
from ml_lab.core.data.schema import ColumnKind, ColumnRole


@pytest.mark.parametrize(
    ("values", "name", "kind"),
    [
        pytest.param([True, False, True], "flag", ColumnKind.BOOLEAN, id="bool"),
        pytest.param([0, 1, 1, 0], "survived", ColumnKind.BOOLEAN, id="zero-one"),
        pytest.param(list(range(1, 31)), "linha", ColumnKind.IDENTIFIER, id="consecutive-ints"),
        pytest.param([7, 3, 99, 42, 15], "user_id", ColumnKind.IDENTIFIER, id="id-name"),
        pytest.param([7, 3, 99, 42, 15], "pontos", ColumnKind.NUMERIC, id="distinct-ints"),
        pytest.param([1, 2, 3, 3, 2, 1], "classe", ColumnKind.NUMERIC, id="few-ints"),
        pytest.param([1.5, 2.25, np.nan, 3.0], "preço", ColumnKind.NUMERIC, id="floats"),
        pytest.param(["sim", "não", "Sim", None], "ativo", ColumnKind.BOOLEAN, id="sim-nao"),
        pytest.param(["a", "b", "a", "b", "a", "c"], "letra", ColumnKind.CATEGORICAL, id="categ"),
        pytest.param(
            [f"AB{i:04d}" for i in range(30)], "código", ColumnKind.IDENTIFIER, id="codes"
        ),
        pytest.param(
            [f"uma frase número {i}" for i in range(30)], "nota", ColumnKind.TEXT, id="free-text"
        ),
        pytest.param([np.nan, np.nan], "vazia", ColumnKind.NUMERIC, id="empty-float"),
        pytest.param([None, None], "vazia", ColumnKind.TEXT, id="empty-object"),
    ],
)
def test_kinds_are_inferred_from_the_values(
    values: list[object], name: str, kind: ColumnKind
) -> None:
    inference = infer_column(pd.Series(values, name=name))

    assert inference.kind is kind
    assert inference.reason
    assert inference.column == name


def test_pandas_categories_and_dates_keep_their_meaning() -> None:
    category = pd.Series(["a", "b"], name="c").astype("category")
    dates = pd.Series(pd.to_datetime(["2024-01-02", "2024-03-04"]), name="d")

    assert infer_column(category).kind is ColumnKind.CATEGORICAL
    assert infer_column(dates).kind is ColumnKind.DATETIME


@pytest.mark.parametrize(
    ("values", "kind", "text_format"),
    [
        (["1.5", "2", "3.25", "-4"], ColumnKind.NUMERIC, "."),
        (["1,5", "2", "3,25", "-4"], ColumnKind.NUMERIC, ","),
        (["25/12/2024", "01/02/2023", "31/01/2022"], ColumnKind.DATETIME, "%d/%m/%Y"),
        (["2024-12-25", "2023-02-01"], ColumnKind.DATETIME, "ISO8601"),
    ],
)
def test_numbers_and_dates_stored_as_text_are_detected(
    values: list[str], kind: ColumnKind, text_format: str
) -> None:
    inference = infer_column(pd.Series(values, name="texto"))

    assert inference.kind is kind
    assert inference.stored_as_text
    assert inference.text_format == text_format


def test_ambiguous_dates_are_read_day_first_as_in_portugal() -> None:
    inference = infer_column(pd.Series(["01/02/2024", "03/04/2024"], name="data"))

    assert inference.text_format == "%d/%m/%Y"


def test_a_few_bad_values_do_not_stop_a_column_from_being_numeric() -> None:
    values = [str(i) for i in range(99)] + ["n/d"]

    assert infer_column(pd.Series(values, name="n")).kind is ColumnKind.NUMERIC


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("PassengerId", True),
        ("user_id", True),
        ("ID", True),
        ("order-code", True),
        ("idade", False),
        ("Paid", False),
        ("width", False),
    ],
)
def test_identifier_names_are_recognized_word_by_word(name: str, expected: bool) -> None:
    assert looks_like_an_identifier_name(name) is expected


def test_the_schema_ignores_text_and_identifiers(passengers: pd.DataFrame) -> None:
    schema = infer_schema(passengers, target="Survived")

    assert schema["PassengerId"].role is ColumnRole.IGNORED
    assert schema["Name"].role is ColumnRole.IGNORED
    assert schema["Survived"].role is ColumnRole.TARGET
    assert schema.feature_names() == ("Pclass", "Sex", "Age", "Embarked", "Fare")


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (["a", "b", "a"], True),
        ([0, 1, 1], True),
        ([1, 2, 3, 2], True),
        ([1.0, 2.0, 3.0], True),
        (list(range(30)), False),
        ([1.5, 2.5, 3.5], False),
    ],
)
def test_classes_are_text_booleans_or_few_integers(values: list[object], expected: bool) -> None:
    assert looks_like_classes(pd.Series(values, name="alvo")) is expected
