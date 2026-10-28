"""As métricas que um modelo iterativo reporta em cada passo do treino, além da loss.

Os modelos guardam-nas em ``StepRecord.metrics`` pelo nome (``"accuracy"``, ``"val_loss"``…),
e os agentes de reinforcement learning também, em cada episódio (``"reward"``, ``"length"``…).
O nome a mostrar e se se escrevem em percentagem estão aqui, para a interface as desenhar
sem os escrever outra vez; o vocabulário (``GET /vocabulary``) leva-as ao frontend.
Os testes de contrato garantem que cada métrica que um modelo ou um agente reporta está aqui.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final


@dataclass(frozen=True, slots=True)
class StepMetric:
    """Uma métrica de cada passo do treino.

    Attributes:
        name: O nome em ``StepRecord.metrics``.
        label: O nome a mostrar.
        as_percent: Se se mostra em percentagem (as que vão de 0 a 1).
        is_loss: Se é uma loss (desenha-se junto da loss do treino, na mesma escala).
    """

    name: str
    label: str
    as_percent: bool
    is_loss: bool


STEP_METRICS: Final[Mapping[str, StepMetric]] = MappingProxyType(
    {
        metric.name: metric
        for metric in (
            StepMetric("accuracy", "Exatidão no treino", as_percent=True, is_loss=False),
            StepMetric("val_loss", "Loss na validação", as_percent=False, is_loss=True),
            StepMetric("oob_score", "Pontuação fora do saco (OOB)", as_percent=True, is_loss=False),
            # As de cada episódio de um agente de reinforcement learning.
            StepMetric("reward", "Recompensa do episódio", as_percent=False, is_loss=False),
            StepMetric("length", "Passos do episódio", as_percent=False, is_loss=False),
            StepMetric("success", "Chegou ao objetivo", as_percent=True, is_loss=False),
            StepMetric("epsilon", "Exploração (ε)", as_percent=False, is_loss=False),
        )
    }
)
