import doctest
import math

import pytest

import ml_lab.formatting as formatting_module
from ml_lab.formatting import (
    format_count,
    format_list,
    format_number,
    format_percent,
    format_quantity,
    format_scientific,
    format_significant,
    format_value,
)


@pytest.mark.parametrize(
    ("value", "decimals", "expected"),
    [
        (3.14159, 2, "3.14"),
        (0.5, 1, "0.5"),
        (-2.25, 2, "-2.25"),
        (999.5, 1, "999.5"),
        (1234.5, 1, "1,234.5"),
        (12345.678, 2, "12,345.68"),
        (-1_000_000, 0, "-1,000,000"),
    ],
)
def test_numbers_use_a_decimal_point_and_commas_for_thousands(
    value: float, decimals: int, expected: str
) -> None:
    assert format_number(value, decimals) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.000001, "1 × 10⁻⁶"),
        (0.000025, "2.5 × 10⁻⁵"),
        (1e-10, "1 × 10⁻¹⁰"),
        (-0.00012345, "-1.23 × 10⁻⁴"),
        (5e12, "5 × 10¹²"),
        (0.0, "0"),
    ],
)
def test_tiny_and_huge_numbers_are_written_as_powers_of_ten(value: float, expected: str) -> None:
    assert format_scientific(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.320912, "0.3209"),
        (12.3456, "12.35"),
        (12345.678, "12,346"),
        (4661704537.05, "4.662 × 10⁹"),
        (0.000123456, "1.235 × 10⁻⁴"),
        (-0.5, "-0.5000"),
        (0.0, "0"),
    ],
)
def test_significant_digits_keep_any_scale_readable(value: float, expected: str) -> None:
    assert format_significant(value) == expected


def test_non_finite_numbers_have_their_own_symbols() -> None:
    assert format_number(math.nan) == "—"
    assert format_number(math.inf) == "∞"
    assert format_number(-math.inf) == "-∞"


def test_counts_and_percentages() -> None:
    assert format_count(20640) == "20,640"
    assert format_count(891) == "891"
    assert format_percent(0.1987) == "19.9%"
    assert format_percent(1) == "100.0%"


def test_quantities_agree_in_number() -> None:
    assert format_quantity(1, "row", "rows") == "1 row"
    assert format_quantity(0, "row", "rows") == "0 rows"
    assert format_quantity(20640, "row", "rows") == "20,640 rows"


def test_values_and_lists_read_like_english() -> None:
    assert format_value(0.000714) == "0.000714"
    assert format_value(3.0) == "3"
    assert format_value(25_000) == "25,000"
    assert format_value(False) == "no"
    assert format_value("rbf") == "rbf"
    assert format_list([]) == ""
    assert format_list(["Sex", "Embarked"]) == "Sex and Embarked"


def test_docstring_examples_run_as_documented() -> None:
    result = doctest.testmod(formatting_module)

    assert result.attempted > 0
    assert result.failed == 0
