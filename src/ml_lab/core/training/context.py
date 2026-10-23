"""Contexto de treino: o canal entre o modelo e o exterior durante o treino.

O modelo nunca sabe quem o está a observar (regra 1 dos contratos). Durante o
``fit``, só fala com o contexto: anuncia cada passo (época, loss, métricas), entrega
fotografias do seu estado e pergunta se deve parar.

Na fase 2 o contrato fica fixo, com um contexto que não faz nada
(``NullTrainingContext``), porque os primeiros modelos não são iterativos. Na fase 3
chega o contexto que publica eventos no ``EventBus`` e grava snapshots.
"""

from collections.abc import Mapping
from typing import Any, Protocol, override


class TrainingContext(Protocol):
    """O que um modelo pode fazer durante o treino."""

    def report_step(
        self, step: int, *, loss: float | None = None, metrics: Mapping[str, float] | None = None
    ) -> None:
        """Anuncia que terminou o passo ``step`` (a começar em 1)."""

    def report_snapshot(self, step: int, state: Mapping[str, Any]) -> None:
        """Entrega o estado do modelo no passo ``step`` (pesos, centróides...)."""

    def should_stop(self) -> bool:
        """Se o treino deve parar já (por exemplo, porque o utilizador cancelou)."""


class NullTrainingContext:
    """Contexto que ignora tudo e nunca pede para parar."""

    def report_step(
        self, step: int, *, loss: float | None = None, metrics: Mapping[str, float] | None = None
    ) -> None:
        """Ignora o passo."""

    def report_snapshot(self, step: int, state: Mapping[str, Any]) -> None:
        """Ignora o snapshot."""

    def should_stop(self) -> bool:
        """Nunca pede para parar."""
        return False


class RecordingTrainingContext(NullTrainingContext):
    """Contexto que guarda tudo o que o modelo anuncia (útil em testes e notebooks)."""

    def __init__(self) -> None:
        self.steps: list[tuple[int, float | None, dict[str, float]]] = []
        self.snapshots: list[tuple[int, dict[str, Any]]] = []

    @override
    def report_step(
        self, step: int, *, loss: float | None = None, metrics: Mapping[str, float] | None = None
    ) -> None:
        self.steps.append((step, loss, dict(metrics or {})))

    @override
    def report_snapshot(self, step: int, state: Mapping[str, Any]) -> None:
        self.snapshots.append((step, dict(state)))
