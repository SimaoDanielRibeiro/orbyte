"""O objetivo da procura: a métrica que os agentes otimizam, e em que sentido.

Por omissão é a métrica principal da tarefa (a exatidão, o R²), mas nem sempre é a certa:
com classes desequilibradas, um modelo que diz sempre a classe mais frequente tem uma
exatidão alta e não serve para nada. Por isso o objetivo escolhe-se, e o agente de dados
sugere a exatidão equilibrada quando as classes estão desequilibradas.

Nem todas as métricas servem de objetivo:

- a **precisão** e o **recall**, sozinhos, ganham-se com um modelo inútil (dizer sempre
  "positivo" dá um recall de 100%). Quem os quer equilibrar otimiza a F1;
- o **MSE** ordena os modelos como o RMSE (é o seu quadrado), por isso basta um.

A ROC AUC e a log-loss medem as probabilidades: com elas, só entram os modelos que as dão.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from ml_lab.core.evaluation.base import PRIMARY_METRIC
from ml_lab.core.evaluation.metrics.base import METRICS, Metric
from ml_lab.core.foundation.errors import InvalidParameterError
from ml_lab.core.foundation.types import TaskType
from ml_lab.formatting import format_number, format_percent

_CLASSIFICATION: Final = ("accuracy", "balanced_accuracy", "f1", "roc_auc", "log_loss")

OBJECTIVES: Final[Mapping[TaskType, tuple[str, ...]]] = MappingProxyType(
    {
        TaskType.BINARY_CLASSIFICATION: _CLASSIFICATION,
        TaskType.MULTICLASS_CLASSIFICATION: _CLASSIFICATION,
        TaskType.REGRESSION: ("r2", "rmse", "mae"),
    }
)
"""As métricas que os agentes sabem otimizar em cada tarefa; a primeira é a principal."""


@dataclass(frozen=True, slots=True)
class Objective:
    """A métrica a otimizar.

    Attributes:
        metric: A métrica registada (em ``METRICS``).
    """

    metric: Metric

    @property
    def name(self) -> str:
        """O nome da métrica no registo."""
        return self.metric.name

    @property
    def label(self) -> str:
        """O nome a mostrar."""
        return self.metric.display_name

    @property
    def higher_is_better(self) -> bool:
        """Se um valor maior é melhor (na RMSE e na log-loss, é ao contrário)."""
        return self.metric.higher_is_better is not False

    @property
    def needs_proba(self) -> bool:
        """Se precisa das probabilidades (só entram os modelos que as dão)."""
        return self.metric.needs_proba

    @property
    def direction(self) -> str:
        """O sentido, em português: "quanto maior, melhor" ou "quanto menor, melhor"."""
        return "quanto maior, melhor" if self.higher_is_better else "quanto menor, melhor"

    def key(self, value: float | None) -> float:
        """A chave para ordenar do melhor para o pior (os valores em falta vão para o fim)."""
        if value is None or not math.isfinite(value):
            return math.inf
        return -value if self.higher_is_better else value

    def is_better(self, value: float | None, than: float | None) -> bool:
        """Se ``value`` é melhor do que ``than`` (um valor em falta nunca é melhor)."""
        return self.key(value) < self.key(than)

    def written(self, value: float | None) -> str:
        """O valor escrito como se mostra: em percentagem, ou com três casas decimais."""
        if value is None:
            return "sem valor"
        return format_percent(value) if self.metric.as_percent else format_number(value, 3)

    def gap(self, better: float, worse: float) -> str:
        """Quanto ``better`` ganha a ``worse``: em pontos percentuais, ou na escala da métrica."""
        difference = abs(better - worse)
        if self.metric.as_percent:
            return f"{format_number(difference * 100, 1)} pontos"
        return format_number(difference, 3)


def supports(task_type: TaskType) -> bool:
    """Se os agentes sabem procurar um modelo para ``task_type``."""
    return task_type in OBJECTIVES


def unsupported_reason(task_type: TaskType) -> str:
    """Porque é que os agentes não trabalham em ``task_type`` (para mostrar)."""
    return (
        "Os agentes escolhem o modelo medindo as previsões contra um alvo, com validação "
        f"cruzada: em {task_type.label.lower()} não há previsões certas para comparar. "
        "Experimenta os modelos no modo manual (ecrãs Preparar e Modelo)."
    )


def objective_for(task_type: TaskType, name: str | None = None) -> Objective:
    """O objetivo ``name`` para ``task_type`` (por omissão, a métrica principal da tarefa).

    Raises:
        InvalidParameterError: Se os agentes não trabalharem nesta tarefa, ou ``name`` não
            for um dos objetivos dela.
    """
    if not supports(task_type):
        raise InvalidParameterError(unsupported_reason(task_type))
    allowed = OBJECTIVES[task_type]
    chosen = name or PRIMARY_METRIC[task_type]
    if chosen not in allowed:
        names = ", ".join(METRICS.get(option).display_name for option in allowed)
        raise InvalidParameterError(
            f"Os agentes não otimizam «{chosen}» em {task_type.label.lower()}. Escolhe uma "
            f"destas métricas: {names}."
        )
    return Objective(METRICS.get(chosen))
