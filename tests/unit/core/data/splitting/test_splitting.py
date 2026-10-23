import numpy as np
import pandas as pd
import pytest

from ml_lab.core.data.splitting.base import (
    SPLITTERS,
    SplitConfig,
    SplitIndices,
    SplitStrategy,
    partition_sizes,
    split_rows,
)
from ml_lab.core.data.splitting.stratified_split import allocate
from ml_lab.core.foundation.errors import DataError, InvalidParameterError
from ml_lab.core.foundation.types import Partition


def split(
    frame: pd.DataFrame, strategy: str, *, seed: int = 0, target: str | None = None, **kw: object
) -> SplitIndices:
    config = SplitConfig(SplitStrategy(strategy), **kw)  # type: ignore[arg-type]
    return split_rows(frame, config, seed=seed, target=target)[0]


def assert_is_a_partition(indices: SplitIndices, n_rows: int) -> None:
    together = np.concatenate([indices.train, indices.validation, indices.test])
    assert sorted(together.tolist()) == list(range(n_rows))


def labels(n_minority: int = 20, n_majority: int = 80) -> pd.DataFrame:
    return pd.DataFrame({"y": ["raro"] * n_minority + ["comum"] * n_majority})


# --- configuração ------------------------------------------------------------------------


def test_strategies_are_registered() -> None:
    assert SPLITTERS.names() == [strategy.value for strategy in sorted(SplitStrategy)]


@pytest.mark.parametrize(
    ("test_size", "validation_size"), [(0.0, 0.0), (1.0, 0.0), (0.2, -0.1), (0.5, 0.5)]
)
def test_invalid_proportions_are_rejected(test_size: float, validation_size: float) -> None:
    with pytest.raises(InvalidParameterError):
        SplitConfig(test_size=test_size, validation_size=validation_size)


def test_strategy_names_from_json_are_accepted() -> None:
    assert SplitConfig("stratified").strategy is SplitStrategy.STRATIFIED  # type: ignore[arg-type]


def test_sizes_round_the_test_up_like_scikit_learn() -> None:
    assert partition_sizes(891, SplitConfig()) == (712, 0, 179)
    assert partition_sizes(100, SplitConfig(test_size=0.2, validation_size=0.1)) == (70, 10, 20)
    with pytest.raises(DataError, match="não há linhas suficientes"):
        partition_sizes(2, SplitConfig(test_size=0.5, validation_size=0.4))


# --- estratégias ---------------------------------------------------------------------------


def test_random_split_is_a_reproducible_partition() -> None:
    frame = labels()

    first = split(frame, "random", seed=1, validation_size=0.1)
    again = split(frame, "random", seed=1, validation_size=0.1)
    other = split(frame, "random", seed=2, validation_size=0.1)

    assert_is_a_partition(first, 100)
    assert first.sizes == {Partition.TRAIN: 70, Partition.VALIDATION: 10, Partition.TEST: 20}
    assert np.array_equal(first.test, again.test)
    assert not np.array_equal(first.test, other.test)


def test_stratified_split_keeps_the_class_proportions() -> None:
    frame = labels(n_minority=20, n_majority=80)

    indices = split(frame, "stratified", target="y", validation_size=0.1)

    assert_is_a_partition(indices, 100)
    for part in Partition:
        rows = frame.iloc[indices.of(part)]
        assert (rows["y"] == "raro").mean() == pytest.approx(0.2)


def test_stratified_split_needs_a_target_without_tiny_classes() -> None:
    with pytest.raises(InvalidParameterError, match="precisa de um alvo"):
        split(labels(), "stratified")
    with pytest.raises(DataError, match="só têm uma linha"):
        split(pd.DataFrame({"y": ["a", "b", "b", "b"]}), "stratified", target="y")
    with pytest.raises(DataError, match="valores em falta"):
        split(pd.DataFrame({"y": ["a", None, "b", "b"]}), "stratified", target="y")


def test_allocation_hits_the_total_and_respects_the_capacity() -> None:
    counts = np.array([7, 2, 1])

    allocation = allocate(counts, 4, capacity=counts - 1)

    assert allocation.sum() == 4
    assert (allocation <= counts - 1).all()


def test_time_split_trains_on_the_past_and_tests_on_the_future() -> None:
    frame = pd.DataFrame({"t": pd.to_datetime(["2024-03-01", "2024-01-01", "2024-02-01"] * 10)})

    indices = split(frame, "time", test_size=0.3, time_column="t")

    last_train = frame["t"].iloc[indices.train].max()
    first_test = frame["t"].iloc[indices.test].min()
    assert last_train <= first_test
    assert_is_a_partition(indices, 30)


def test_time_split_without_a_column_uses_the_row_order() -> None:
    indices = split(pd.DataFrame({"x": range(10)}), "time", test_size=0.2)

    assert indices.test.tolist() == [8, 9]


def test_time_split_needs_a_complete_time_column() -> None:
    with pytest.raises(DataError, match="não é de datas"):
        split(pd.DataFrame({"t": ["a", "b"] * 5}), "time", time_column="t")
    with pytest.raises(DataError, match="valores em falta"):
        split(pd.DataFrame({"t": [1.0, np.nan] * 5}), "time", time_column="t")


def test_the_lineage_explains_the_split() -> None:
    _, step = split_rows(labels(), SplitConfig(SplitStrategy.RANDOM), seed=4, root_seed=42)

    assert step.description.startswith("Divisão aleatória com a seed 4, derivada da seed 42")
    assert "treino 80 linhas (80,0%)" in step.description
    assert step.params["seed"] == 4
