"""Eventos de treino e EventBus (padrão observer).

Durante o treino, o modelo não sabe quem o está a observar: limita-se a anunciar
o que acontece (início, cada passo, snapshots, avaliação, fim ou falha). O
EventBus entrega cada evento a quem o subscreveu: gravador de snapshots, registo
de experiências, interface em tempo real ou um teste.

Exemplo:
    >>> bus = EventBus()
    >>> received = []
    >>> subscription = bus.subscribe(StepCompleted, received.append)
    >>> bus.publish(StepCompleted(run_id="run-1", step=1, loss=0.5))
    >>> [event.loss for event in received]
    [0.5]
    >>> bus.unsubscribe(subscription)
    >>> bus.publish(StepCompleted(run_id="run-1", step=2, loss=0.4))
    >>> len(received)
    1
"""

import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar


def _now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True, kw_only=True)
class Event:
    """Base de todos os eventos de treino. Os eventos são imutáveis.

    Attributes:
        kind: Nome estável do tipo de evento, usado em JSON e na base de dados.
        run_id: Identificador do run a que o evento pertence.
        timestamp: Momento em que o evento foi criado, em UTC.
    """

    kind: ClassVar[str] = "event"
    run_id: str
    timestamp: datetime = field(default_factory=_now)


@dataclass(frozen=True, slots=True, kw_only=True)
class RunStarted(Event):
    """O run começou; é anunciado antes do primeiro passo.

    Attributes:
        config: Configuração da experiência (dataset, modelo, hiperparâmetros, seed).
        n_samples: Número de exemplos de treino.
        n_features: Número de features.
    """

    kind: ClassVar[str] = "run_started"
    config: Mapping[str, Any] = field(default_factory=dict)
    n_samples: int
    n_features: int


@dataclass(frozen=True, slots=True, kw_only=True)
class StepCompleted(Event):
    """Terminou um passo do treino (uma época ou uma iteração).

    Attributes:
        step: Número do passo, a começar em 1.
        loss: Valor da função de perda, se o modelo a tiver.
        metrics: Métricas parciais calculadas neste passo.
    """

    kind: ClassVar[str] = "step"
    step: int
    loss: float | None = None
    metrics: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True, kw_only=True)
class SnapshotTaken(Event):
    """Fotografia do estado do modelo, tirada a cada N passos.

    Attributes:
        step: Passo em que o snapshot foi tirado.
        state: Estado do modelo: pesos, centróides, árvore, fronteira...
    """

    kind: ClassVar[str] = "snapshot"
    step: int
    state: Mapping[str, Any]


@dataclass(frozen=True, slots=True, kw_only=True)
class EvaluationCompleted(Event):
    """O modelo treinado foi avaliado. A diferença entre os dois relatórios revela overfitting.

    Attributes:
        train_report: Relatório da avaliação nos dados de treino.
        test_report: Relatório da avaliação nos dados de teste.
    """

    kind: ClassVar[str] = "evaluation"
    train_report: Mapping[str, Any]
    test_report: Mapping[str, Any]


@dataclass(frozen=True, slots=True, kw_only=True)
class RunFinished(Event):
    """O run terminou com sucesso.

    Attributes:
        duration_seconds: Duração total do run.
        summary: Resultado resumido (por exemplo, as métricas principais).
        model_path: Onde ficou guardado o modelo treinado, se foi guardado.
    """

    kind: ClassVar[str] = "run_finished"
    duration_seconds: float
    summary: Mapping[str, Any] = field(default_factory=dict)
    model_path: Path | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class RunFailed(Event):
    """O run falhou.

    Attributes:
        message: Explicação do que correu mal e, se possível, do que fazer.
        error_type: Nome da exceção que causou a falha.
    """

    kind: ClassVar[str] = "run_failed"
    message: str
    error_type: str | None = None


type EventHandler[E: Event] = Callable[[E], object]
"""Função chamada com cada evento; o valor que devolve é ignorado."""


@dataclass(frozen=True, slots=True, eq=False)
class Subscription[E: Event]:
    """Subscrição ativa. Guarda-a para a cancelar com ``EventBus.unsubscribe``.

    Attributes:
        event_type: Tipo de evento subscrito; inclui as suas subclasses.
        handler: Função chamada com cada evento.
    """

    event_type: type[E]
    handler: EventHandler[E]


class EventBus:
    """Distribui eventos de treino a quem os subscreveu.

    Os subscritores são chamados pela ordem em que subscreveram, no mesmo fio de
    execução de quem publica. Se um subscritor lançar uma exceção, a publicação
    pára e a exceção chega a quem publicou: os erros não ficam escondidos.

    Pode-se subscrever e cancelar a partir de outros fios (threads), por exemplo
    a interface enquanto o treino corre em segundo plano.
    """

    def __init__(self) -> None:
        # Um dict serve de conjunto ordenado: mantém a ordem de subscrição.
        self._subscriptions: dict[Subscription[Any], None] = {}
        self._lock = threading.Lock()

    @property
    def subscription_count(self) -> int:
        """Número de subscrições ativas."""
        with self._lock:
            return len(self._subscriptions)

    def subscribe[E: Event](self, event_type: type[E], handler: EventHandler[E]) -> Subscription[E]:
        """Passa a chamar ``handler`` com cada evento do tipo ``event_type``.

        Subscrever ``Event`` recebe todos os eventos.

        Args:
            event_type: Tipo de evento a receber (inclui as subclasses).
            handler: Função chamada com cada evento.

        Returns:
            A subscrição, para a poder cancelar.
        """
        subscription = Subscription(event_type, handler)
        with self._lock:
            self._subscriptions[subscription] = None
        return subscription

    def unsubscribe(self, subscription: Subscription[Any]) -> None:
        """Cancela uma subscrição. Cancelar duas vezes não tem efeito."""
        with self._lock:
            self._subscriptions.pop(subscription, None)

    def publish(self, event: Event) -> None:
        """Entrega ``event`` a todos os subscritores do seu tipo."""
        with self._lock:
            subscriptions = list(self._subscriptions)
        for subscription in subscriptions:
            # Um subscritor pode ter cancelado outro durante esta mesma publicação.
            if subscription in self._subscriptions and isinstance(event, subscription.event_type):
                subscription.handler(event)
