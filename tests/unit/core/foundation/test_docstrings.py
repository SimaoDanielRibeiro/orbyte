"""Os exemplos nas docstrings da fundação são executados, para nunca ficarem errados."""

import doctest
from types import ModuleType

import pytest

import ml_lab.core.foundation.events as events_module
import ml_lab.core.foundation.random as random_module
import ml_lab.core.foundation.registry as registry_module


@pytest.mark.parametrize(
    "module",
    [events_module, random_module, registry_module],
    ids=lambda module: module.__name__,
)
def test_docstring_examples_run_as_documented(module: ModuleType) -> None:
    result = doctest.testmod(module)

    assert result.attempted > 0
    assert result.failed == 0
