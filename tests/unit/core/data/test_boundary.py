from datetime import date

import numpy as np
import pandas as pd
import pytest

from ml_lab.core.data.boundary import (
    class_labels,
    encode_classes,
    plain_value,
    preview,
    to_matrix,
    to_target,
)
from ml_lab.core.foundation.errors import DataError
from ml_lab.core.foundation.types import Partition


def test_numbers_become_a_float64_matrix() -> None:
    matrix = to_matrix(pd.DataFrame({"a": [1, 2], "b": [0.5, 1.5]}), Partition.TRAIN)

    assert matrix.dtype == np.float64
    assert matrix.tolist() == [[1.0, 0.5], [2.0, 1.5]]


@pytest.mark.parametrize(
    ("frame", "message"),
    [
        (pd.DataFrame({"a": ["x", "y"]}), "ainda não são números"),
        (pd.DataFrame({"a": [True, False]}), "ainda não são números"),
        (pd.DataFrame({"a": [1.0, np.nan]}), r"a \(1 valor\)"),
        (pd.DataFrame({"a": [1.0, np.inf]}), "infinitos"),
    ],
)
def test_text_missing_and_infinite_values_cannot_reach_the_models(
    frame: pd.DataFrame, message: str
) -> None:
    with pytest.raises(DataError, match=message):
        to_matrix(frame, Partition.TEST)


def test_class_labels_are_sorted_numerically_when_they_are_numbers() -> None:
    assert class_labels(pd.Series([10, 2, 1, 2])) == ("1", "2", "10")
    assert class_labels(pd.Series([1.0, 0.0, np.nan])) == ("0", "1")
    assert class_labels(pd.Series(["virginica", "setosa"])) == ("setosa", "virginica")
    assert class_labels(pd.Series([True, False])) == ("False", "True")


def test_classes_become_their_position_in_the_labels() -> None:
    codes = encode_classes(pd.Series(["b", "a", "b"]), ("a", "b"))

    assert codes.dtype == np.int64
    assert codes.tolist() == [1, 0, 1]
    with pytest.raises(DataError, match="desconhecidas: c"):
        encode_classes(pd.Series(["a", "c"]), ("a", "b"))


def test_regression_targets_must_be_complete_numbers() -> None:
    assert to_target(pd.Series([1, 2])).tolist() == [1.0, 2.0]
    with pytest.raises(DataError, match="não é numérico"):
        to_target(pd.Series(["1", "2"], name="preço"))
    with pytest.raises(DataError, match="valores em falta"):
        to_target(pd.Series([1.0, np.nan]))


def test_previews_contain_only_plain_python_values() -> None:
    frame = pd.DataFrame(
        {
            "n": [1, 2, 3],
            "x": [0.5, np.nan, 2.0],
            "t": ["a", None, "c"],
            "d": pd.to_datetime(pd.Series(["2024-01-02", None, "2024-03-04"])),
        }
    )

    result = preview(frame, n_rows=2)

    assert result.n_rows == 3
    assert result.rows == ((1, 0.5, "a", "2024-01-02T00:00:00"), (2, None, None, None))
    assert result.to_records()[0] == {"n": 1, "x": 0.5, "t": "a", "d": "2024-01-02T00:00:00"}
    assert all(type(value) in (int, float, str, type(None)) for row in result.rows for value in row)


def test_plain_values() -> None:
    assert plain_value(np.int64(3)) == 3
    assert plain_value(pd.NA) is None
    assert plain_value(date(2024, 1, 2)) == "2024-01-02"
    assert plain_value(pd.Interval(0, 1)) == "(0, 1]"
