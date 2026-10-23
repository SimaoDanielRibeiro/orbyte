import doctest
import math

import pytest

import ml_lab.formatting as formatting_module
from ml_lab.formatting import format_count, format_number, format_percent, format_quantity


@pytest.mark.parametrize(
    ("value", "decimals", "expected"),
    [
        (3.14159, 2, "3,14"),
        (0.5, 1, "0,5"),
        (-2.25, 2, "-2,25"),
        (1234.5, 1, "1234,5"),
        (12345.678, 2, "12 345,68"),
        (-1_000_000, 0, "-1 000 000"),
    ],
)
def test_numbers_use_a_decimal_comma_and_spaces_for_thousands(
    value: float, decimals: int, expected: str
) -> None:
    assert format_number(value, decimals) == expected


def test_non_finite_numbers_have_their_own_symbols() -> None:
    assert format_number(math.nan) == "—"
    assert format_number(math.inf) == "∞"
    assert format_number(-math.inf) == "-∞"


def test_counts_and_percentages() -> None:
    assert format_count(20640) == "20 640"
    assert format_count(891) == "891"
    assert format_percent(0.1987) == "19,9%"
    assert format_percent(1) == "100,0%"


def test_quantities_agree_in_number() -> None:
    assert format_quantity(1, "linha", "linhas") == "1 linha"
    assert format_quantity(0, "linha", "linhas") == "0 linhas"
    assert format_quantity(20640, "linha", "linhas") == "20 640 linhas"


def test_docstring_examples_run_as_documented() -> None:
    result = doctest.testmod(formatting_module)

    assert result.attempted > 0
    assert result.failed == 0
