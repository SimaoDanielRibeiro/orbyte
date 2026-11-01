"""A conversa da procura: o que cada agente decide e encontra, pela ordem em que acontece.

A procura corre num fio de execução próprio e quem a acompanha (a interface, a API) lê
noutro. O ``SearchLog`` é o sítio onde os agentes escrevem e de onde os outros leem: as
mensagens (com a razão de cada decisão), os candidatos já medidos, o que está a ser medido
agora (e em que fold) e a fase em que vai. Cada mensagem e cada candidato têm um número,
para quem acompanha pedir só o que é novo.
"""

import threading
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Final

from ml_lab.core.models.base import ModelFamily


class SearchPhase(StrEnum):
    """Em que parte da procura se está."""

    DATA = "data"
    SCREENING = "screening"
    TUNING = "tuning"
    FINAL = "final"
    DONE = "done"

    @property
    def label(self) -> str:
        """Nome a mostrar, em português."""
        return _PHASE_LABELS[self]


_PHASE_LABELS: Final[Mapping[SearchPhase, str]] = MappingProxyType(
    {
        SearchPhase.DATA: "Preparar os dados",
        SearchPhase.SCREENING: "Ronda 1 · triagem",
        SearchPhase.TUNING: "Ronda 2 · afinação",
        SearchPhase.FINAL: "Treinar e testar o vencedor",
        SearchPhase.DONE: "Terminada",
    }
)


class AgentRole(StrEnum):
    """Quem fala na conversa."""

    DATA = "data"
    COORDINATOR = "coordinator"
    MODEL = "model"
    AI = "ai"

    @property
    def label(self) -> str:
        """Nome a mostrar na interface, em português."""
        return _ROLE_LABELS[self]


_ROLE_LABELS: Final[Mapping[AgentRole, str]] = MappingProxyType(
    {
        AgentRole.DATA: "Agente de dados",
        AgentRole.COORDINATOR: "Coordenador",
        AgentRole.MODEL: "Agente de modelo",
        AgentRole.AI: "Agente de IA",
    }
)


DATA_AGENT: Final = "Agente de dados"
COORDINATOR: Final = "Coordenador"
AI_AGENT: Final = "Agente · IA"


def model_agent_name(family: ModelFamily) -> str:
    """O nome do agente de uma família ("Agente · Árvores de decisão")."""
    return f"Agente · {family.label}"


class ParticipantKind(StrEnum):
    """Quem pode receber uma mensagem: os agentes e as duas etapas do fim da procura."""

    DATA = "data"
    COORDINATOR = "coordinator"
    MODEL = "model"
    AI = "ai"
    WINNER = "winner"
    TEST = "test"

    @property
    def label(self) -> str:
        """Nome a mostrar, em português."""
        return _PARTICIPANT_LABELS[self]


_PARTICIPANT_LABELS: Final[Mapping[ParticipantKind, str]] = MappingProxyType(
    {
        ParticipantKind.DATA: "Agente de dados",
        ParticipantKind.COORDINATOR: "Coordenador",
        ParticipantKind.MODEL: "Agente de modelo",
        ParticipantKind.AI: "Agente de IA",
        ParticipantKind.WINNER: "Vencedor",
        ParticipantKind.TEST: "Teste",
    }
)


@dataclass(frozen=True, slots=True)
class Participant:
    """Um destinatário de uma mensagem.

    Attributes:
        kind: Quem é.
        family: Num agente de modelo, a sua família.
    """

    kind: ParticipantKind
    family: ModelFamily | None = None


TO_COORDINATOR: Final = (Participant(ParticipantKind.COORDINATOR),)
TO_WINNER: Final = (Participant(ParticipantKind.WINNER),)
TO_TEST: Final = (Participant(ParticipantKind.TEST),)
TO_AI: Final = (Participant(ParticipantKind.AI),)


def to_agents(families: Iterable[ModelFamily]) -> tuple[Participant, ...]:
    """Os agentes das famílias ``families`` (cada um uma vez, pela ordem)."""
    return tuple(Participant(ParticipantKind.MODEL, family) for family in dict.fromkeys(families))


class MessageKind(StrEnum):
    """O que uma mensagem é na troca entre os agentes (o histórico das trocas desenha-a)."""

    REPORT = "report"
    ANNOUNCEMENT = "announcement"
    REQUEST = "request"
    REPLY = "reply"
    DECISION = "decision"
    TEST = "test"
    THOUGHT = "thought"

    @property
    def label(self) -> str:
        """Nome a mostrar, em português."""
        return _MESSAGE_KIND_LABELS[self]


_MESSAGE_KIND_LABELS: Final[Mapping[MessageKind, str]] = MappingProxyType(
    {
        MessageKind.REPORT: "Relatório",
        MessageKind.ANNOUNCEMENT: "Anúncio",
        MessageKind.REQUEST: "Pedido",
        MessageKind.REPLY: "Resposta",
        MessageKind.DECISION: "Decisão",
        MessageKind.TEST: "Teste",
        MessageKind.THOUGHT: "Pensamento",
    }
)


@dataclass(frozen=True, slots=True)
class AgentMessage:
    """Uma mensagem da conversa: quem a diz, a quem, e o quê.

    As mensagens contam a procura como uma troca: o agente de dados entrega o plano ao
    coordenador (``REPORT``); o coordenador anuncia as rondas aos agentes (``ANNOUNCEMENT``),
    pede a cada um que meça um candidato (``REQUEST``) e recebe a resposta com o resultado
    (``REPLY``); no fim, passa o melhor ao vencedor (``DECISION``) e o vencedor vai ao teste
    (``TEST``). O agente de IA, antes de responder, pensa em voz alta (``THOUGHT``, sem
    destinatários: é para quem acompanha). É com ``kind`` e ``recipients`` que o histórico
    visual desenha cada troca.

    Attributes:
        number: A ordem na conversa (1, 2, 3…).
        role: Quem fala: o agente de dados, o coordenador ou um agente de modelo.
        author: O nome a mostrar de quem fala.
        text: O que diz, em português.
        phase: A fase da procura em que o disse.
        family: Num agente de modelo, a sua família.
        candidate: O candidato de que fala, se for o caso.
        kind: O que a mensagem é na troca.
        recipients: A quem se dirige (vazio: a quem acompanha a procura).
    """

    number: int
    role: AgentRole
    author: str
    text: str
    phase: SearchPhase
    family: ModelFamily | None = None
    candidate: int | None = None
    kind: MessageKind = MessageKind.ANNOUNCEMENT
    recipients: tuple[Participant, ...] = ()


@dataclass(frozen=True, slots=True)
class Candidate:
    """Um modelo com uns hiperparâmetros, medido com validação cruzada nos folds do treino.

    Attributes:
        number: A ordem em que foi experimentado (1, 2, 3…): identifica-o na procura.
        phase: A ronda em que entrou: a triagem ou a afinação.
        family: A família do modelo (o agente que o propôs).
        model: O modelo, no registo.
        model_display_name: O nome a mostrar.
        hyperparameters: Os hiperparâmetros, completos.
        changes: Os que mudaram em relação aos valores por omissão.
        mean: A média do objetivo nos folds de avaliação (``None`` se falhou).
        std: O desvio padrão do objetivo nos folds de avaliação.
        train_mean: A média do objetivo nos folds de treino: a distância a ``mean`` mostra
            quanto o modelo decora.
        folds: O objetivo em cada fold de avaliação.
        seconds: Quanto tempo demoraram todos os folds.
        error: Se falhou, porquê.
        ai: Se foi o agente de IA a propô-lo (e não o sorteio do agente da família).
        reason: Porque é que o agente de IA o propôs, nas palavras dele.
    """

    number: int
    phase: SearchPhase
    family: ModelFamily
    model: str
    model_display_name: str
    hyperparameters: dict[str, Any]
    changes: dict[str, Any]
    mean: float | None = None
    std: float | None = None
    train_mean: float | None = None
    folds: tuple[float | None, ...] = ()
    seconds: float = 0.0
    error: str | None = None
    ai: bool = False
    reason: str | None = None

    @property
    def failed(self) -> bool:
        """Se a validação cruzada falhou (o candidato fica no quadro, com a razão)."""
        return self.error is not None


@dataclass(frozen=True, slots=True)
class WorkInProgress:
    """O que está a ser feito agora: um agente a medir um candidato, fold a fold.

    Um candidato só entra no quadro depois de medido; até lá, quem acompanha a procura sabe
    por aqui que agente trabalha, em quê, e em que fold vai (o fluxograma dos agentes).

    Attributes:
        candidate: O número que o candidato vai ter.
        phase: A ronda: a triagem ou a afinação.
        family: A família do modelo (o agente que trabalha).
        author: O nome a mostrar do agente que trabalha.
        model: O modelo, no registo.
        model_display_name: O nome a mostrar do modelo.
        changes: Os hiperparâmetros que mudam em relação aos valores por omissão.
        fold: O fold que está a treinar (1, 2…), ou 0 antes do primeiro.
        folds: Quantos folds há.
        ai: Se o candidato é uma proposta do agente de IA (o fluxograma acende o nó dele).
    """

    candidate: int
    phase: SearchPhase
    family: ModelFamily
    author: str
    model: str
    model_display_name: str
    changes: dict[str, Any]
    fold: int
    folds: int
    ai: bool = False


@dataclass(frozen=True, slots=True)
class Thought:
    """O agente de IA a pensar, agora: o raciocínio até aqui, que vai crescendo aos bocados.

    Attributes:
        author: Quem pensa.
        candidate: O número que o candidato proposto vai ter.
        text: O raciocínio até aqui.
    """

    author: str
    candidate: int
    text: str = ""


class SearchLog:
    """O que a procura vai dizendo e encontrando (seguro entre fios de execução).

    Args:
        on_candidate: Chamado com cada candidato medido (o serviço conta-o como um passo).
    """

    def __init__(self, on_candidate: Callable[[Candidate], None] | None = None) -> None:
        self._on_candidate = on_candidate
        self._messages: list[AgentMessage] = []
        self._candidates: list[Candidate] = []
        self._phase = SearchPhase.DATA
        self._working: WorkInProgress | None = None
        self._thought: Thought | None = None
        self._lock = threading.Lock()

    @property
    def phase(self) -> SearchPhase:
        """A fase em que a procura vai."""
        with self._lock:
            return self._phase

    def enter(self, phase: SearchPhase) -> None:
        """Passa à fase ``phase``."""
        with self._lock:
            self._phase = phase

    def say(
        self,
        role: AgentRole,
        author: str,
        text: str,
        *,
        family: ModelFamily | None = None,
        candidate: int | None = None,
        kind: MessageKind = MessageKind.ANNOUNCEMENT,
        recipients: tuple[Participant, ...] = (),
    ) -> AgentMessage:
        """Junta uma mensagem à conversa."""
        with self._lock:
            message = AgentMessage(
                number=len(self._messages) + 1,
                role=role,
                author=author,
                text=text,
                phase=self._phase,
                family=family,
                candidate=candidate,
                kind=kind,
                recipients=recipients,
            )
            self._messages.append(message)
        return message

    def start(self, work: WorkInProgress) -> None:
        """Anuncia o candidato que um agente começa a medir."""
        with self._lock:
            self._working = work

    def progress(self, fold: int) -> None:
        """O candidato em curso passou ao fold ``fold``."""
        with self._lock:
            if self._working is not None:
                self._working = replace(self._working, fold=fold)

    @property
    def working(self) -> WorkInProgress | None:
        """O que está a ser medido agora (``None`` entre candidatos)."""
        with self._lock:
            return self._working

    def start_thinking(self, author: str, candidate: int) -> None:
        """O agente de IA começa a pensar no candidato ``candidate``."""
        with self._lock:
            self._thought = Thought(author, candidate)

    def think(self, piece: str) -> None:
        """Mais um bocado do raciocínio (chega à medida que o modelo o escreve)."""
        if not piece:
            return
        with self._lock:
            if self._thought is not None:
                self._thought = replace(self._thought, text=self._thought.text + piece)

    def stop_thinking(self) -> None:
        """O agente de IA acabou de pensar (o raciocínio inteiro fica numa mensagem)."""
        with self._lock:
            self._thought = None

    @property
    def thought(self) -> Thought | None:
        """O que o agente de IA está a pensar agora (``None`` quando não está a pensar)."""
        with self._lock:
            return self._thought

    def add(self, candidate: Candidate) -> None:
        """Junta um candidato medido (e o trabalho em curso acaba)."""
        with self._lock:
            self._candidates.append(candidate)
            self._working = None
        if self._on_candidate is not None:
            self._on_candidate(candidate)

    @property
    def next_number(self) -> int:
        """O número do próximo candidato."""
        with self._lock:
            return len(self._candidates) + 1

    def messages(self, after: int = 0) -> tuple[AgentMessage, ...]:
        """As mensagens depois da número ``after``."""
        with self._lock:
            return tuple(self._messages[after:])

    def candidates(self, after: int = 0) -> tuple[Candidate, ...]:
        """Os candidatos depois do número ``after``."""
        with self._lock:
            return tuple(self._candidates[after:])
