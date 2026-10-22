import pytest

from ml_lab.core.foundation.errors import InvalidSeedError
from ml_lab.core.foundation.random import (
    MAX_SEED,
    SeedManager,
    derive_seed,
    generate_seed,
    make_rng,
    validate_seed,
)


def test_derived_seeds_are_stable_across_runs_and_versions() -> None:
    # Valores fixos de propósito: se este teste falhar, a derivação mudou e as
    # experiências já guardadas deixam de ser reproduzíveis.
    assert derive_seed(42, "split") == 4025458178
    assert derive_seed(42, "model") == 1177043635
    assert derive_seed(0, "split") == 450945530


def test_different_names_give_independent_seeds() -> None:
    assert derive_seed(42, "split") != derive_seed(42, "model")


def test_different_roots_give_different_seeds() -> None:
    assert derive_seed(1, "split") != derive_seed(2, "split")


def test_derived_seeds_spread_over_the_whole_range() -> None:
    seeds = {derive_seed(7, f"component_{index}") for index in range(1_000)}

    assert len(seeds) == 1_000
    assert all(0 <= seed <= MAX_SEED for seed in seeds)


def test_derive_seed_rejects_an_empty_name() -> None:
    with pytest.raises(ValueError, match="vazio"):
        derive_seed(42, "")


@pytest.mark.parametrize("seed", [0, 1, 42, MAX_SEED])
def test_valid_seeds_are_accepted(seed: int) -> None:
    assert validate_seed(seed) == seed


@pytest.mark.parametrize("seed", [-1, MAX_SEED + 1, 1.5, "42", None, True])
def test_invalid_seeds_are_rejected(seed: object) -> None:
    with pytest.raises(InvalidSeedError):
        validate_seed(seed)


def test_generated_seeds_are_valid() -> None:
    assert all(validate_seed(generate_seed()) >= 0 for _ in range(100))


def test_rngs_with_the_same_seed_produce_the_same_numbers() -> None:
    first, second = make_rng(3), make_rng(3)

    assert [first.random() for _ in range(5)] == [second.random() for _ in range(5)]


def test_seed_manager_gives_each_component_a_reproducible_seed() -> None:
    seeds = SeedManager(42)

    assert seeds.seed_for("split") == derive_seed(42, "split")
    assert SeedManager(42).seed_for("split") == seeds.seed_for("split")


def test_seed_manager_rngs_are_reproducible() -> None:
    first, second = SeedManager(7).rng("noise"), SeedManager(7).rng("noise")

    assert [first.random() for _ in range(5)] == [second.random() for _ in range(5)]


def test_child_managers_are_deterministic_and_independent() -> None:
    folds = SeedManager(7).child("cv")

    assert folds.seed_for("fold_0") == SeedManager(7).child("cv").seed_for("fold_0")
    assert folds.seed_for("fold_0") != folds.seed_for("fold_1")
    assert folds.seed != SeedManager(7).child("bootstrap").seed


def test_create_without_a_seed_generates_one() -> None:
    assert 0 <= SeedManager.create().seed <= MAX_SEED


def test_create_with_a_seed_uses_it() -> None:
    assert SeedManager.create(123).seed == 123


def test_seed_manager_validates_its_seed() -> None:
    with pytest.raises(InvalidSeedError):
        SeedManager(-5)
