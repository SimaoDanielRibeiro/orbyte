"""Contratos dos componentes da camada de dados: aplicados a TODOS os registados.

Um exemplo, um gerador, uma operação de limpeza ou uma estratégia de divisão nova
fica testada aqui sem escrever testes novos.
"""

import numpy as np
import pandas as pd
import pytest

from ml_lab.core.data.cleaning.base import CLEANING
from ml_lab.core.data.io.builtin_source import SAMPLES, SampleSource
from ml_lab.core.data.io.synthetic_source import GENERATORS, TARGET, SyntheticSource
from ml_lab.core.data.splitting.base import SPLITTERS, SplitConfig, SplitStrategy
from ml_lab.core.data.task.task_detection import suggest_task
from ml_lab.core.data.validation.checks import validate_table


@pytest.mark.parametrize("name", SAMPLES.names())
def test_samples_are_usable_and_their_target_matches_their_task(name: str) -> None:
    info = SAMPLES.get(name)
    table = SampleSource(name).load()

    assert info.title
    assert info.description
    assert info.source
    assert validate_table(table.frame).is_usable
    assert suggest_task(table.frame, info.target).task_type is info.task_type


@pytest.mark.parametrize("name", GENERATORS.names())
def test_generators_produce_usable_data_with_their_defaults(name: str) -> None:
    info = GENERATORS.get(name)
    table = SyntheticSource(name, seed=0).load()

    assert table.frame.notna().all().all()
    assert validate_table(table.frame).is_usable
    suggestion = suggest_task(table.frame, TARGET)
    # A tarefa para que os dados foram pensados é a sugerida (nas supervisionadas) ou uma
    # das permitidas, com a coluna target como referência (nas anomalias, por exemplo).
    if info.task_type.requires_target:
        assert suggestion.task_type is info.task_type
    else:
        assert info.task_type in suggestion.allowed
    defaults = info.resolve({})
    assert len(table.frame) == defaults["n_samples"]


@pytest.mark.parametrize("name", GENERATORS.names())
def test_generator_parameters_have_sensible_limits(name: str) -> None:
    for parameter in GENERATORS.get(name).parameters:
        assert parameter.label
        assert parameter.minimum <= parameter.default <= parameter.maximum


@pytest.mark.parametrize("name", CLEANING.names())
def test_cleaning_operations_are_documented(name: str) -> None:
    operation = CLEANING.get(name)

    assert operation.name == name
    assert operation.title
    assert len(operation.description) > 40


@pytest.mark.parametrize("strategy", [strategy.value for strategy in SplitStrategy])
def test_every_strategy_is_a_reproducible_partition_of_the_rows(strategy: str) -> None:
    assert strategy in SPLITTERS
    frame = pd.DataFrame({"x": np.arange(60), "y": ["a", "b", "c"] * 20})
    config = SplitConfig(SplitStrategy(strategy), test_size=0.2, validation_size=0.1)
    split = SPLITTERS.get(strategy)

    first = split(frame, config, 11, "y")
    again = split(frame, config, 11, "y")

    parts = [first.train, first.validation, first.test]
    assert sorted(np.concatenate(parts).tolist()) == list(range(60))
    assert [len(part) for part in parts] == [42, 6, 12]
    assert all(
        np.array_equal(a, b)
        for a, b in zip(parts, [again.train, again.validation, again.test], strict=True)
    )
