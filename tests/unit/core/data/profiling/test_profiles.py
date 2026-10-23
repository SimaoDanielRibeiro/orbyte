import numpy as np
import pandas as pd
import pytest

from ml_lab.core.data.profiling.column_profile import profile_column
from ml_lab.core.data.profiling.dataset_profile import profile_dataset
from ml_lab.core.data.schema import ColumnKind
from ml_lab.core.foundation.errors import ColumnNotFoundError


def codes(column: pd.Series) -> list[str]:
    return [notice.code for notice in profile_column(column).notices]


# --- colunas -----------------------------------------------------------------------------


def test_numeric_summary_matches_pandas(passengers: pd.DataFrame) -> None:
    profile = profile_column(passengers["Age"])
    age = passengers["Age"].dropna()

    assert profile.kind is ColumnKind.NUMERIC
    assert profile.n_missing == 3
    assert profile.numeric is not None
    assert profile.numeric.mean == pytest.approx(age.mean())
    assert profile.numeric.median == pytest.approx(age.median())
    assert profile.numeric.std == pytest.approx(age.std())
    assert profile.histogram is not None
    assert sum(profile.histogram.counts) == len(age)
    assert profile.top_values == ()


def test_categorical_columns_get_their_most_frequent_values(passengers: pd.DataFrame) -> None:
    profile = profile_column(passengers["Embarked"])

    assert profile.kind is ColumnKind.CATEGORICAL
    assert [(v.value, v.count) for v in profile.top_values] == [("S", 10), ("C", 4), ("Q", 4)]
    assert profile.top_values[0].ratio == pytest.approx(10 / 18)
    assert profile.numeric is None


def test_dates_get_their_range() -> None:
    dates = pd.Series(pd.to_datetime(pd.Series(["2024-03-01", "2023-01-15", None])), name="data")

    assert profile_column(dates).date_range == ("2023-01-15T00:00:00", "2024-03-01T00:00:00")


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        pytest.param([1.0, 2.0, np.nan, np.nan, np.nan], ["missing_values"], id="many-missing"),
        pytest.param([7, 7, 7, 7], ["constant_column"], id="constant"),
        pytest.param(list(range(1, 11)), ["identifier_column"], id="identifier"),
        pytest.param([f"texto número {i}" for i in range(10)], ["free_text"], id="free-text"),
        pytest.param(["1,5", "2,5", "3"], ["stored_as_text"], id="numbers-as-text"),
        pytest.param([np.nan, np.nan], ["empty_column"], id="empty"),
    ],
)
def test_columns_get_didactic_notices(values: list[object], expected: list[str]) -> None:
    assert codes(pd.Series(values, name="coluna")) == expected


def test_missing_values_are_graded_by_proportion() -> None:
    few = profile_column(pd.Series([1.0] * 99 + [np.nan], name="a")).notices
    some = profile_column(pd.Series([1.0, 2.0] * 9 + [np.nan, np.nan], name="b")).notices
    many = profile_column(pd.Series([1.0, np.nan, np.nan], name="c")).notices

    assert few[0].level == "info"
    assert some[0].level == "warning"
    assert "considera removê-la" in many[0].message


def test_skewed_columns_and_outliers_are_noted() -> None:
    values = pd.Series([1.0] * 30 + [2.0] * 10 + [3.0] * 5 + [50.0, 80.0], name="tarifa")

    assert codes(values) == ["skewed", "outliers"]


def test_outliers_are_not_noted_when_the_spread_is_zero() -> None:
    # Como o Parch do Titanic: quase tudo 0, e um IQR de zero tornaria "extremo" qualquer 1.
    values = pd.Series([0] * 40 + [1, 2, 1], name="parch")

    assert "outliers" not in codes(values)


def test_rare_categories_are_noted() -> None:
    values = pd.Series(["comum"] * 20 + ["rara"] * 2, name="cat")

    assert codes(values) == ["rare_categories"]


def test_counts_can_come_from_the_full_column_when_the_rest_uses_a_sample() -> None:
    full = pd.Series([1.0, np.nan] * 50, name="x")

    profile = profile_column(full.iloc[:10], full=full)

    assert profile.n_rows == 100
    assert profile.n_missing == 50


# --- dataset ------------------------------------------------------------------------------


def test_dataset_profile_summarizes_the_table(passengers: pd.DataFrame) -> None:
    original = passengers.copy()

    profile = profile_dataset(passengers, target="Survived")

    assert passengers.equals(original)  # o perfil só descreve, nunca altera
    assert (profile.n_rows, profile.n_columns) == (20, 8)
    assert profile.n_missing_cells == 5
    assert profile.n_duplicate_rows == 0
    assert not profile.sampled
    assert profile.kind_counts[ColumnKind.NUMERIC] == 3
    assert profile.column("Name").kind is ColumnKind.TEXT


def test_correlations_are_symmetric_with_ones_on_the_diagonal(numbers: pd.DataFrame) -> None:
    correlations = profile_dataset(numbers).correlations

    assert correlations is not None
    assert correlations.columns == ("x1", "x2", "y")
    assert correlations.get("x1", "x1") == pytest.approx(1.0)
    assert correlations.get("x1", "y") == pytest.approx(correlations.get("y", "x1"))


def test_class_balance_and_group_means(passengers: pd.DataFrame) -> None:
    profile = profile_dataset(passengers, target="Survived")

    assert profile.class_balance is not None
    assert [(c.value, c.count) for c in profile.class_balance.classes] == [("0", 10), ("1", 10)]
    assert profile.class_balance.imbalance_ratio == 1.0
    assert profile.group_means is not None
    assert profile.group_means.groups == ("0", "1")
    fare = profile.group_means.columns.index("Fare")
    survivors = passengers[passengers["Survived"] == 1]["Fare"].mean()
    assert profile.group_means.means[1][fare] == pytest.approx(survivors)


def test_imbalanced_targets_and_leaks_are_flagged() -> None:
    frame = pd.DataFrame({"x": np.arange(40.0), "y": [1] * 2 + [0] * 38})
    frame["leak"] = frame["y"] * 10.0

    codes = [notice.code for notice in profile_dataset(frame, target="y").notices]

    assert "class_imbalance" in codes
    assert "possible_leakage" in codes


def test_regression_targets_have_no_class_balance(numbers: pd.DataFrame) -> None:
    profile = profile_dataset(numbers, target="y")

    assert profile.class_balance is None
    assert profile.group_means is None


def test_large_tables_use_a_sample_for_distributions_but_exact_counts() -> None:
    frame = pd.DataFrame({"x": np.arange(1000.0), "y": [np.nan, 1.0] * 500})

    profile = profile_dataset(frame, max_rows=100)

    assert profile.sampled
    assert profile.profiled_rows == 100
    assert profile.column("y").n_missing == 500
    assert "sampled_profile" in [notice.code for notice in profile.notices]
    assert len(profile.missing_map.rows) == 200


def test_the_profile_is_the_same_every_time() -> None:
    frame = pd.DataFrame({"x": np.random.default_rng(0).normal(size=500)})

    assert profile_dataset(frame, max_rows=50) == profile_dataset(frame, max_rows=50)


def test_unknown_target_is_reported(passengers: pd.DataFrame) -> None:
    with pytest.raises(ColumnNotFoundError):
        profile_dataset(passengers, target="Cabin")


def test_all_notices_put_the_most_severe_first(passengers: pd.DataFrame) -> None:
    levels = [notice.level for notice in profile_dataset(passengers).all_notices]

    assert levels == sorted(levels, key=["error", "warning", "info"].index)
