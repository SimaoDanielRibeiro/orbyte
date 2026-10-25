"""Contrato do vocabulário: cada enum que os serviços devolvem tem um nome para cada valor.

Os enums viajam em JSON pelo valor; o frontend mostra-os com os nomes de ``GET /vocabulary``
(ADR 0026). Um enum novo num resultado, ou um valor novo num enum, falha aqui até ter nome:
assim nunca chega ao frontend um ``"binary_classification"`` cru.
"""

import enum
import inspect
import typing
from dataclasses import is_dataclass
from typing import Any, get_args, get_origin, get_type_hints

import pytest
from pydantic import BaseModel

from ml_lab.core.evaluation.metrics.base import METRICS
from ml_lab.core.foundation.types import TaskType
from ml_lab.core.models.base import MODELS
from ml_lab.core.schemas.serialization import is_internal
from ml_lab.core.services import Lab, vocabulary
from ml_lab.core.training.jobs import JobState, TrainingJob


def returned_enums() -> set[type[enum.Enum]]:
    """Os enums a que se chega a partir do que os métodos públicos dos serviços devolvem."""
    seen: set[Any] = set()
    found: set[type[enum.Enum]] = set()

    def visit(kind: Any) -> None:
        if get_origin(kind) is TrainingJob:
            visit(JobState)
        elif get_origin(kind) is not None:
            for argument in get_args(kind):
                visit(argument)
        elif isinstance(kind, typing.TypeAliasType):
            visit(kind.__value__)
        elif isinstance(kind, type) and kind not in seen:
            seen.add(kind)
            if issubclass(kind, enum.Enum):
                found.add(kind)
            elif is_dataclass(kind) or (issubclass(kind, BaseModel) and kind is not BaseModel):
                for hint in get_type_hints(kind).values():
                    visit(hint)

    for service_class in get_type_hints(Lab).values():
        for name, method in inspect.getmembers(service_class, inspect.isfunction):
            if not name.startswith("_") and not is_internal(method):
                visit(get_type_hints(method)["return"])
    return found


ENUMS = sorted(returned_enums(), key=lambda kind: kind.__name__)


def test_the_discovery_finds_the_enums() -> None:
    assert TaskType in ENUMS
    assert len(ENUMS) >= 10


@pytest.mark.parametrize("kind", ENUMS, ids=lambda kind: kind.__name__)
def test_every_returned_enum_has_a_name_for_every_value(kind: type[enum.Enum]) -> None:
    labels = vocabulary().labels

    assert kind.__name__ in labels, f"Junta {kind.__name__} ao vocabulário (services/vocabulary.py)"
    names = labels[kind.__name__]
    assert set(names) == {member.value for member in kind}
    assert all(name.strip() for name in names.values())
    assert len(set(names.values())) == len(names)  # dois valores com o mesmo nome confundem


def test_every_task_says_what_changes_with_it() -> None:
    tasks = {info.task_type: info for info in vocabulary().tasks}

    assert set(tasks) == set(TaskType)
    metrics, models = dict(METRICS.items()), dict(MODELS.items())
    for task_type, info in tasks.items():
        assert info.label == task_type.label
        # O controlo mede-se com as métricas dos episódios (core/rl/evaluation.py), que não
        # estão no registo METRICS: o agente não prevê, joga.
        if task_type.learns_from_data:
            assert info.primary_metric in metrics
        assert info.suggested_model is None or info.suggested_model in models
    assert tasks[TaskType.CONTROL].primary_metric == "success_rate"
    assert tasks[TaskType.REGRESSION].primary_metric == "r2"
    assert tasks[TaskType.BINARY_CLASSIFICATION].agents
    assert not tasks[TaskType.CLUSTERING].agents
