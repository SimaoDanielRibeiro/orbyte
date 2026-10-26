"""Transmitir um treino pelo WebSocket: o mesmo ``updates`` dos serviços, de 0,25 em 0,25 s.

O treino corre num fio de execução próprio, e o WebSocket no ciclo assíncrono do servidor.
Em vez de passar cada evento do ``EventBus`` de um lado para o outro (o que pede filas e
cuidado com os fios), o WebSocket pergunta ao serviço o que mudou, de ``POLL_SECONDS`` em
``POLL_SECONDS``. Assim:

- os passos que chegam entretanto vão juntos numa só mensagem (um modelo rápido dá
  milhares de passos por segundo, e o navegador não precisa de os receber um a um);
- quem acompanha por HTTP (``GET .../updates``) e quem acompanha pelo WebSocket recebem
  exatamente a mesma coisa (ADR 0022).

Cada mensagem é um objeto ``{"kind": ..., "data": ...}``:

- ``{"kind": "update", "data": <update>}`` quando há passos novos, um snapshot novo, o
  estado mudou ou mudou o trabalho a decorrer (na procura dos agentes, o candidato que um
  agente está a medir e o fold em que vai, e a fase: o fluxograma acompanha-os);
- ``{"kind": "error", "data": <ApiError>}`` se o treino não existir.

A ligação fecha quando o treino termina (a última mensagem tem o estado final) e o
resultado pede-se por HTTP.
"""

import asyncio
from collections.abc import Callable
from typing import Final, Protocol

from fastapi import WebSocket, WebSocketDisconnect
from starlette.concurrency import run_in_threadpool

from ml_lab.api.errors import api_error
from ml_lab.core.schemas.serialization import to_json
from ml_lab.core.services import JobState, JobStatus, MLLabError, StepRecord

POLL_SECONDS: Final = 0.25

_DONE: Final = frozenset({JobStatus.FINISHED, JobStatus.FAILED})


class Update(Protocol):
    """O que os ``updates`` dos serviços devolvem (o do treino, o do RL e o dos agentes).

    São dois cursores: o último passo e o segundo, que no treino é o passo do último
    snapshot e na procura dos agentes é o número da última mensagem da conversa.
    """

    @property
    def state(self) -> JobState:
        """O retrato do treino agora."""

    @property
    def steps(self) -> tuple[StepRecord, ...]:
        """Os passos novos."""

    @property
    def snapshot_step(self) -> int | None:
        """O segundo cursor: o passo do snapshot novo (ou a última mensagem nova), se houver."""


async def stream_updates(websocket: WebSocket, poll: Callable[[int, int], Update]) -> None:
    """Envia as mudanças de um treino até ele terminar (ou até quem ouve desligar).

    Args:
        websocket: A ligação, ainda por aceitar.
        poll: O ``updates`` do serviço, com o run já escolhido: recebe o último passo e o
            passo do último snapshot que já foram enviados.
    """
    await websocket.accept()
    after_step = after_snapshot = 0
    last_status: JobStatus | None = None
    last_work: object = None
    try:
        while True:
            try:
                # Os serviços são síncronos: correm fora do ciclo assíncrono, para não o parar.
                update = await run_in_threadpool(poll, after_step, after_snapshot)
            except MLLabError as error:
                await websocket.send_json({"kind": "error", "data": to_json(api_error(error))})
                break
            status = update.state.status
            work = _work(update)
            if (
                update.steps
                or update.snapshot_step is not None
                or status != last_status
                or work != last_work
            ):
                await websocket.send_json({"kind": "update", "data": to_json(update)})
            if update.steps:
                after_step = update.steps[-1].step
            after_snapshot = update.snapshot_step or after_snapshot
            last_status = status
            last_work = work
            if status in _DONE:
                break
            await asyncio.sleep(POLL_SECONDS)
    except WebSocketDisconnect:
        return
    await websocket.close()


def _work(update: Update) -> object:
    """O trabalho a decorrer que não conta como passo nem como mensagem.

    Na procura dos agentes, um candidato só entra no quadro depois de medido: até lá, o que
    muda é o candidato que um agente mede, o fold em que vai e a fase. Os outros trabalhos (o
    treino, o RL) não têm nada disto.
    """
    return (getattr(update, "working", None), getattr(update, "phase", None))
