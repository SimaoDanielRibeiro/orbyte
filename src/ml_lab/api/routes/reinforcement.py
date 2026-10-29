"""Rotas do reinforcement learning (ecrã Reinforcement learning): treinar agentes num ambiente.

Funciona como o treino dos modelos: ``POST /reinforcement`` valida a configuração, começa o
treino em segundo plano e responde logo (202) com o estado. Acompanha-se com
``GET .../updates`` ou pelo WebSocket ``.../stream``, e o resultado pede-se com
``GET .../result``. Cada episódio é um passo do treino.
"""

from typing import Annotated

from fastapi import Query, WebSocket

from ml_lab.api.errors import ERROR_RESPONSES
from ml_lab.api.routing import LabDep, ServiceRouter
from ml_lab.api.stream import stream_updates
from ml_lab.core.services import (
    AgentCard,
    EnvironmentCard,
    JobState,
    ReinforcementConfig,
    ReinforcementLesson,
    ReinforcementResult,
    ReinforcementUpdate,
)

router = ServiceRouter(prefix="/reinforcement", tags=["reinforcement"], responses=ERROR_RESPONSES)


@router.get("/environments")
def environments(lab: LabDep) -> list[EnvironmentCard]:
    """Os ambientes, com o formulário dos parâmetros de cada um."""
    return lab.reinforcement.environments()


@router.get("/agents")
def agents(lab: LabDep) -> list[AgentCard]:
    """Os agentes, com o formulário dos hiperparâmetros e a ficha didática de cada um."""
    return lab.reinforcement.agents()


@router.get("/lesson")
def lesson(lab: LabDep) -> ReinforcementLesson:
    """A lição que abre o ecrã: um Q-learning pequeno, passo a passo (não é um run)."""
    return lab.reinforcement.lesson()


@router.post("", status_code=202)
def start(lab: LabDep, config: ReinforcementConfig) -> JobState:
    """Valida o treino e corre-o em segundo plano (um episódio é um passo)."""
    return lab.reinforcement.start(config).state()


@router.get("/{run_id}/job")
def job(lab: LabDep, run_id: str) -> JobState:
    """O estado do treino: em curso, em pausa, terminado ou falhado, e o progresso."""
    return lab.reinforcement.job(run_id)


@router.post("/{run_id}/pause")
def pause(lab: LabDep, run_id: str) -> JobState:
    """Pede uma pausa no fim do episódio atual."""
    return lab.reinforcement.pause(run_id)


@router.post("/{run_id}/resume")
def resume(lab: LabDep, run_id: str) -> JobState:
    """Retoma um treino em pausa."""
    return lab.reinforcement.resume(run_id)


@router.post("/{run_id}/cancel")
def cancel(lab: LabDep, run_id: str) -> JobState:
    """Pede ao treino que pare no fim do episódio atual (o agente fica com o que aprendeu)."""
    return lab.reinforcement.cancel(run_id)


@router.get("/{run_id}/updates")
def updates(
    lab: LabDep,
    run_id: str,
    after_step: Annotated[int, Query(ge=0, description="O último episódio que já tens.")] = 0,
    after_snapshot: Annotated[
        int, Query(ge=0, description="O episódio do último snapshot que já tens.")
    ] = 0,
) -> ReinforcementUpdate:
    """O que mudou no treino: o estado, os episódios novos e a política do snapshot novo."""
    return lab.reinforcement.updates(run_id, after_step, after_snapshot)


@router.get("/{run_id}/result")
def result(lab: LabDep, run_id: str) -> ReinforcementResult:
    """O resultado de um treino terminado: as métricas, a política e a recompensa por episódio."""
    return lab.reinforcement.result(run_id)


@router.websocket("/{run_id}/stream")
async def stream(websocket: WebSocket, lab: LabDep, run_id: str) -> None:
    """Envia as mudanças do treino (o mesmo que ``updates``) até ele terminar."""

    def poll(after_step: int, after_snapshot: int) -> ReinforcementUpdate:
        return lab.reinforcement.updates(run_id, after_step, after_snapshot)

    await stream_updates(websocket, poll)
