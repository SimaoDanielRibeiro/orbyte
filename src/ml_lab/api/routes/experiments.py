"""Rotas do treino (ecrã Treinar e avaliar): lançar em segundo plano, acompanhar e controlar.

O treino é sempre em segundo plano: ``POST /experiments`` valida a configuração, começa o
treino e responde logo (202) com o estado. Quem quer acompanhar pode:

- perguntar de vez em quando (``GET .../updates``, dizendo o que já tem);
- abrir o WebSocket ``.../stream``, que envia as mesmas mudanças sozinho.

Quando o estado diz ``finished``, o resultado pede-se com ``GET .../result``.
"""

from typing import Annotated

from fastapi import Query, WebSocket

from ml_lab.api.errors import ERROR_RESPONSES
from ml_lab.api.routing import LabDep, ServiceRouter
from ml_lab.api.stream import stream_updates
from ml_lab.core.services import (
    ExperimentConfig,
    ExperimentResult,
    JobState,
    TrainingAnimation,
    TrainingUpdate,
)

router = ServiceRouter(prefix="/experiments", tags=["experiments"], responses=ERROR_RESPONSES)


@router.post("", status_code=202)
def start(lab: LabDep, config: ExperimentConfig) -> JobState:
    """Valida a experiência e começa o treino em segundo plano.

    Um problema na configuração (o modelo não existe, um hiperparâmetro fora dos limites, o
    modelo não resolve a tarefa dos dados) responde logo com o erro, antes de treinar.
    """
    return lab.experiments.start(config).state()


@router.get("/{run_id}/job")
def job(lab: LabDep, run_id: str) -> JobState:
    """O estado do treino: em curso, em pausa, terminado ou falhado, e o progresso."""
    return lab.experiments.job(run_id)


@router.post("/{run_id}/pause")
def pause(lab: LabDep, run_id: str) -> JobState:
    """Pede uma pausa no fim do passo atual."""
    return lab.experiments.pause(run_id)


@router.post("/{run_id}/resume")
def resume(lab: LabDep, run_id: str) -> JobState:
    """Retoma um treino em pausa."""
    return lab.experiments.resume(run_id)


@router.post("/{run_id}/cancel")
def cancel(lab: LabDep, run_id: str) -> JobState:
    """Pede ao treino que pare no fim do passo atual (o modelo fica com o que aprendeu)."""
    return lab.experiments.cancel(run_id)


@router.get("/{run_id}/updates")
def updates(
    lab: LabDep,
    run_id: str,
    after_step: Annotated[int, Query(ge=0, description="O último passo que já tens.")] = 0,
    after_snapshot: Annotated[
        int, Query(ge=0, description="O passo do último snapshot que já tens.")
    ] = 0,
) -> TrainingUpdate:
    """O que mudou no treino: o estado, os passos novos e a fronteira do snapshot novo."""
    return lab.experiments.updates(run_id, after_step, after_snapshot)


@router.get("/{run_id}/preview")
def preview(lab: LabDep, run_id: str) -> TrainingAnimation | None:
    """A fronteira no último snapshot de um treino que ainda decorre."""
    return lab.experiments.preview(run_id)


@router.get("/{run_id}/result")
def result(lab: LabDep, run_id: str) -> ExperimentResult:
    """O resultado de um treino terminado: as métricas, a animação e o que o modelo aprendeu."""
    return lab.experiments.result(run_id)


@router.websocket("/{run_id}/stream")
async def stream(websocket: WebSocket, lab: LabDep, run_id: str) -> None:
    """Envia as mudanças do treino (o mesmo que ``updates``) até ele terminar."""

    def poll(after_step: int, after_snapshot: int) -> TrainingUpdate:
        return lab.experiments.updates(run_id, after_step, after_snapshot)

    await stream_updates(websocket, poll)
