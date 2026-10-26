"""O WebSocket manda só o que mudou: os passos, as mensagens, o estado e o trabalho a decorrer.

Testa-se o ``stream_updates`` sozinho, com uma ligação a fingir e uma sequência de respostas do
serviço: assim cada caso fica explícito, sem treinos nem procuras a sério.
"""

import asyncio
from dataclasses import dataclass
from typing import Any, cast

import pytest
from fastapi import WebSocket

from ml_lab.api import stream
from ml_lab.core.services import JobState, JobStatus, Progress, StepRecord


@dataclass(frozen=True, slots=True)
class _Update:
    """Uma resposta do serviço; ``working`` faz de candidato a ser medido pelos agentes."""

    state: JobState
    steps: tuple[StepRecord, ...] = ()
    working: str | None = None

    @property
    def snapshot_step(self) -> int | None:
        return None


class _Socket:
    """Uma ligação a fingir, que guarda o que lhe mandam."""

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    async def accept(self) -> None:
        return None

    async def send_json(self, data: dict[str, Any]) -> None:
        self.sent.append(data)

    async def close(self) -> None:
        return None


def _state(status: JobStatus) -> JobState:
    return JobState("run-1", status, Progress(0, None, 0))


def _streamed(updates: list[_Update], monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    monkeypatch.setattr(stream, "POLL_SECONDS", 0)
    replies = iter(updates)
    socket = _Socket()

    asyncio.run(
        stream.stream_updates(cast(WebSocket, socket), lambda _step, _snapshot: next(replies))
    )

    return [message["data"] for message in socket.sent]


def test_the_work_in_progress_alone_is_worth_a_message(monkeypatch: pytest.MonkeyPatch) -> None:
    sent = _streamed(
        [
            _Update(_state(JobStatus.RUNNING), working="candidato 3, fold 1"),
            _Update(_state(JobStatus.RUNNING), working="candidato 3, fold 2"),
            _Update(_state(JobStatus.RUNNING), working="candidato 3, fold 2"),
            _Update(_state(JobStatus.FINISHED)),
        ],
        monkeypatch,
    )

    # O fold novo vai sozinho (o fluxograma mostra-o); a repetição não vai.
    assert [data["working"] for data in sent] == [
        "candidato 3, fold 1",
        "candidato 3, fold 2",
        None,
    ]


def test_nothing_new_sends_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    sent = _streamed(
        [
            _Update(_state(JobStatus.RUNNING)),
            _Update(_state(JobStatus.RUNNING)),
            _Update(_state(JobStatus.RUNNING), steps=(StepRecord(1, 0.5, {}),)),
            _Update(_state(JobStatus.FINISHED)),
        ],
        monkeypatch,
    )

    assert [data["state"]["status"] for data in sent] == ["running", "running", "finished"]
    assert [len(data["steps"]) for data in sent] == [0, 1, 0]
