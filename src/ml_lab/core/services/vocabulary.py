"""O vocabulário da interface: o nome a mostrar de cada valor, e o que cada tarefa é.

Os enums viajam em JSON pelo valor (``"binary_classification"``), que é estável; o nome a
mostrar (``"Classificação binária"``) está no ``label`` de cada um. A interface simples lê-o
diretamente; o frontend pede o vocabulário uma vez (``GET /vocabulary``) e usa-o em todos os
ecrãs, sem escrever os nomes outra vez (regra 5 do ``CONTRIBUTING.md``, ADR 0026).

As tarefas vêm com o que muda com elas: o paradigma, se precisam de alvo, a métrica que
resume um run (a do "último resultado"), o modelo sugerido e se os agentes do modo
automático trabalham nelas. São regras de domínio: o frontend não as decide, lê-as daqui.

Um teste de contrato garante que cada enum que os serviços devolvem está no vocabulário, com
um nome para cada valor: um enum novo não chega ao frontend sem nome.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

from ml_lab.core.automl.objective import supports
from ml_lab.core.automl.search_log import AgentRole, SearchPhase
from ml_lab.core.data.lineage import Stage
from ml_lab.core.data.schema import ColumnKind
from ml_lab.core.data.splitting.base import SplitStrategy
from ml_lab.core.data.transforms.imputation import ImputationStrategy
from ml_lab.core.data.transforms.scaling import ScalingMethod
from ml_lab.core.evaluation.base import PRIMARY_METRIC
from ml_lab.core.foundation.notices import NoticeLevel
from ml_lab.core.foundation.types import Paradigm, Partition, TaskType
from ml_lab.core.models.base import Implementation, ModelFamily
from ml_lab.core.rl.environments.base import CellKind
from ml_lab.core.services.catalog import SUGGESTED_MODELS
from ml_lab.core.services.laboratory import LaboratoryDemo
from ml_lab.core.training.jobs import JobStatus


class _Labeled(Protocol):
    """Um valor de um enum com nome a mostrar."""

    @property
    def value(self) -> str:
        """O valor, como viaja em JSON."""

    @property
    def label(self) -> str:
        """O nome a mostrar."""


@dataclass(frozen=True, slots=True)
class TaskInfo:
    """Uma tarefa e o que muda com ela.

    Attributes:
        task_type: A tarefa.
        label: O nome a mostrar.
        paradigm: O paradigma (supervisionada, não supervisionada, reinforcement learning).
        requires_target: Se precisa de uma coluna-alvo.
        learns_from_data: Se aprende com um dataset (o controlo aprende num ambiente).
        primary_metric: A métrica que resume um run desta tarefa, no "último resultado" e no
            histórico (o nome no registo de métricas).
        suggested_model: O primeiro modelo a experimentar, a referência dos outros.
        agents: Se os agentes do modo automático trabalham nesta tarefa.
    """

    task_type: TaskType
    label: str
    paradigm: Paradigm
    requires_target: bool
    learns_from_data: bool
    primary_metric: str | None
    suggested_model: str | None
    agents: bool


@dataclass(frozen=True, slots=True)
class Vocabulary:
    """O vocabulário da interface.

    Attributes:
        labels: Para cada enum, pelo nome do seu tipo (o mesmo do OpenAPI, como
            ``TaskType``), o nome a mostrar de cada valor.
        tasks: As tarefas, pela ordem do enum.
    """

    labels: dict[str, dict[str, str]]
    tasks: tuple[TaskInfo, ...]


def vocabulary() -> Vocabulary:
    """O nome a mostrar de cada valor de cada enum, e o que muda com cada tarefa."""
    labels = {
        "AgentRole": _labels(AgentRole),
        "CellKind": _labels(CellKind),
        "ColumnKind": _labels(ColumnKind),
        "Implementation": _labels(Implementation),
        "ImputationStrategy": _labels(ImputationStrategy),
        "JobStatus": _labels(JobStatus),
        "LaboratoryDemo": _labels(LaboratoryDemo),
        "ModelFamily": _labels(ModelFamily),
        "NoticeLevel": _labels(NoticeLevel),
        "Paradigm": _labels(Paradigm),
        "Partition": _labels(Partition),
        "ScalingMethod": _labels(ScalingMethod),
        "SearchPhase": _labels(SearchPhase),
        "SplitStrategy": _labels(SplitStrategy),
        "Stage": _labels(Stage),
        "TaskType": _labels(TaskType),
    }
    return Vocabulary(labels=labels, tasks=tuple(_task(task_type) for task_type in TaskType))


def _labels(members: Iterable[_Labeled]) -> dict[str, str]:
    return {member.value: member.label for member in members}


def _task(task_type: TaskType) -> TaskInfo:
    return TaskInfo(
        task_type=task_type,
        label=task_type.label,
        paradigm=task_type.paradigm,
        requires_target=task_type.requires_target,
        learns_from_data=task_type.learns_from_data,
        primary_metric=PRIMARY_METRIC.get(task_type),
        suggested_model=SUGGESTED_MODELS.get(task_type),
        agents=supports(task_type),
    )
