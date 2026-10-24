"""Trainer: treina um modelo com um contexto que publica cada passo e cada snapshot.

O Trainer é quem liga o modelo ao mundo durante o treino: cria o contexto (com a
sonda, a amostragem e o controlo), manda o modelo treinar e, no fim, garante que o
último estado fica fotografado, mesmo que o último passo não calhasse no intervalo
de amostragem. Quem quiser acompanhar o treino subscreve o ``EventBus``.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ml_lab.core.foundation.events import EventBus
from ml_lab.core.models.base import Features, LabModel, Target
from ml_lab.core.training.context import EventTrainingContext, SnapshotSampler
from ml_lab.core.training.control import TrainingControl


@dataclass(frozen=True, slots=True)
class TrainingOutcome:
    """Como correu o treino.

    Attributes:
        steps: Quantos passos o modelo anunciou (0 nos modelos que treinam de uma vez).
        stopped_early: Se parou antes do fim porque alguém pediu.
    """

    steps: int
    stopped_early: bool


class Trainer:
    """Treina modelos, publicando o que acontece no ``EventBus``.

    Args:
        bus: Para onde vão os eventos dos passos e dos snapshots.
    """

    def __init__(self, bus: EventBus) -> None:
        self._bus = bus

    def train(
        self,
        model: LabModel[Any],
        X: Features,
        y: Target | None,
        *,
        run_id: str,
        sampler: SnapshotSampler | None = None,
        probe: NDArray[np.float64] | None = None,
        control: TrainingControl | None = None,
    ) -> TrainingOutcome:
        """Treina ``model`` com ``X`` e ``y``.

        Args:
            model: O modelo, ainda por treinar.
            X: As features de treino.
            y: O alvo de treino (``None`` sem alvo).
            run_id: O run a que os eventos pertencem.
            sampler: A amostragem dos snapshots (partilhada com o gravador).
            probe: Pontos onde se querem as previsões em cada snapshot.
            control: Pedidos de pausa e de paragem.
        """
        context = EventTrainingContext(
            self._bus, run_id, sampler=sampler, probe=probe, control=control
        )
        model.fit(X, y, context)
        if context.last_step and context.last_snapshot_step != context.last_step:
            context.report_snapshot(context.last_step, model.snapshot_state(probe))
        return TrainingOutcome(
            steps=context.last_step,
            stopped_early=control is not None and control.stop_requested,
        )
