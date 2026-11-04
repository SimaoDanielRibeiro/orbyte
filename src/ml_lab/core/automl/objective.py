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
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from ml_lab.core.evaluation.base import PRIMARY_METRIC
from ml_lab.core.evaluation.diagnostics import memorised
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
        """O sentido, por extenso: "higher is better" ou "lower is better"."""
        return "higher is better" if self.higher_is_better else "lower is better"

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
            return "no value"
        return format_percent(value) if self.metric.as_percent else format_number(value, 3)

    def gap(self, better: float, worse: float) -> str:
        """Quanto ``better`` ganha a ``worse``: em pontos percentuais, ou na escala da métrica."""
        difference = abs(better - worse)
        if self.metric.as_percent:
            return f"{format_number(difference * 100, 1)} points"
        return format_number(difference, 3)

    def reading(self, value: float | None) -> str:
        """O que um valor quer dizer, numa frase curta, sem ponto final (ADR 0049).

        A conversa dos agentes diz "R² 0.752" e logo a seguir "It explains about 75% of the
        variation in the target": quem não sabe o que é o R² percebe na mesma o resultado. Uma
        métrica sem leitura devolve texto vazio.
        """
        if value is None or not math.isfinite(value):
            return ""
        write = _READINGS.get(self.name)
        return "" if write is None else write(value)

    def overfits(self, train: float, validation: float) -> bool:
        """Se o treino ficou muito acima da validação cruzada (o modelo decora o treino).

        Nas métricas de 0 a 1 em que maior é melhor, é a regra do aviso do treino
        (``memorised``: dez pontos de diferença, ou o que falha a mais do dobro); nos erros, a
        validação mais de 50% acima do treino.
        """
        if not (math.isfinite(train) and math.isfinite(validation)):
            return False
        if self.higher_is_better:
            return memorised(train, validation)
        return validation > train * OVERFIT_RATIO and validation - train > 1e-9


OVERFIT_RATIO: Final = 1.5
"""Num erro, quantas vezes maior na validação do que no treino já é decorar o treino."""


def _in_100(value: float) -> str:
    """Um valor de 0 a 1 em "N in every 100" (o que se lê melhor do que uma percentagem)."""
    return format_number(round(max(0.0, min(1.0, value)) * 100), 0)


def _r2(value: float) -> str:
    if value < 0:
        return "It does worse than always predicting the mean of the target"
    return f"It explains about {_in_100(value)}% of the variation in the target"


_READINGS: Final[Mapping[str, Callable[[float], str]]] = MappingProxyType(
    {
        "accuracy": lambda v: f"It gets about {_in_100(v)} in every 100 rows right",
        "balanced_accuracy": lambda v: (
            f"It gets about {_in_100(v)} in every 100 rows of each class right, on average"
        ),
        "f1": lambda v: (
            f"It scores about {_in_100(v)} out of 100 at finding the positive rows while "
            "being right about them"
        ),
        "roc_auc": lambda v: (
            f"It ranks a positive row above a negative one about {_in_100(v)} times in 100"
        ),
        "log_loss": lambda _: (
            "The lower, the better: 0 would be perfect probabilities, and confident mistakes "
            "cost the most"
        ),
        "r2": _r2,
        "rmse": lambda _: (
            "That's the size of a typical miss, in the target's units, with the big misses "
            "weighing more"
        ),
        "mae": lambda _: "That's how much it misses by on average, in the target's units",
    }
)
"""A leitura de cada objetivo (``Objective.reading``)."""


def supports(task_type: TaskType) -> bool:
    """Se os agentes sabem procurar um modelo para ``task_type``."""
    return task_type in OBJECTIVES


def unsupported_reason(task_type: TaskType) -> str:
    """Porque é que os agentes não trabalham em ``task_type`` (para mostrar)."""
    return (
        "The agents choose the model by measuring predictions against a target, with "
        f"cross-validation: in {task_type.label.lower()} there are no right predictions to "
        "compare with. Try the models in the manual mode (the Prepare and Model screens)."
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
            f"The agents don't optimise “{chosen}” in {task_type.label.lower()}. Choose one of "
            f"these metrics: {names}."
        )
    return Objective(METRICS.get(chosen))
