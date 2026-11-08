"""O analista no serviço dos agentes: as perguntas sobre uma procura, em segundo plano (ADR 0053).

O ``AutoMLService`` é a porta (``ask``, ``questions``, ``question_updates``…), e este módulo
guarda as perguntas. Cada pergunta corre num ``TrainingJob`` próprio, como as procuras (ADR
0022): o modelo de linguagem demora segundos em cada vez, e o ecrã não pode ficar à espera. Quem
acompanha pede, pelo id da pergunta, os passos novos do analista e o que ele está a pensar agora.

Com as respostas guardadas no pedido (``AnalystRequest.replies``), o analista conversa com o
``ScriptedModel``, que as dá outra vez pela ordem: a pergunta repete-se tal e qual, sem o Ollama.
A resposta traz sempre o pedido com as respostas que usou.
"""

import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from typing import Final

from ml_lab.core.automl.analyst import (
    MAX_TURNS,
    Analyst,
    AnalystLog,
    AnalystOutcome,
    AnalystStep,
    LiveTurn,
    answer_notices,
    opening,
)
from ml_lab.core.automl.analyst_tools import TOOLS, SearchFacts
from ml_lab.core.foundation.errors import LanguageModelError, QuestionNotFoundError
from ml_lab.core.foundation.events import StepCompleted
from ml_lab.core.foundation.notices import Notice
from ml_lab.core.llm import LanguageModel, Reply, ScriptedModel, ToolCall
from ml_lab.core.schemas.automl import AnalystRequest, RecordedReply, RecordedToolCall
from ml_lab.core.services.llm import LanguageModelService
from ml_lab.core.training.control import TrainingControl
from ml_lab.core.training.jobs import JobRegistry, JobState, TrainingJob
from ml_lab.core.training.recorder import SnapshotRecorder, StepRecord
from ml_lab.logging import get_logger

logger = get_logger(__name__)

KEPT_QUESTIONS: Final = 20
"""Quantas perguntas ficam em memória, com os passos e a resposta."""

SAVED_ANSWERS: Final = "saved answers"
"""O nome do "modelo" que responde com as respostas guardadas (o ``ScriptedModel``)."""


@dataclass(frozen=True, slots=True)
class AnalystToolInfo:
    """Uma ferramenta do analista, pronta a mostrar.

    Attributes:
        name: O nome, como o modelo de linguagem a pede (``"get_candidate"``).
        label: O nome a mostrar.
        purpose: O que faz, numa frase.
    """

    name: str
    label: str
    purpose: str


@dataclass(frozen=True, slots=True)
class AnalystOptions:
    """O que se pode perguntar ao analista sobre uma procura acabada.

    Attributes:
        search_id: A procura.
        suggestions: Perguntas que esta procura sabe responder ("Why did the winner win?"):
            dependem do que ela tem (um vencedor, a referência, o agente de IA…).
        tools: As ferramentas que o analista pode usar, pela ordem.
        model: O modelo de linguagem configurado ("qwen3:8b"), se houver um ligado.
        max_turns: Quantas vezes o analista pode pedir ferramentas antes de ter de responder.
    """

    search_id: str
    suggestions: tuple[str, ...]
    tools: tuple[AnalystToolInfo, ...]
    model: str | None
    max_turns: int


@dataclass(frozen=True, slots=True)
class AnalystAnswer:
    """A resposta do analista, com cada passo que deu até ela.

    Attributes:
        question_id: O identificador da pergunta.
        search_id: A procura de que fala.
        request: O pedido, com as respostas do modelo de linguagem: repete a resposta tal e qual,
            sem o Ollama.
        question: A pergunta.
        model: O modelo de linguagem que respondeu ("qwen3:8b", ou "saved answers").
        trace: Cada passo, pela ordem: o raciocínio, as ferramentas (com os argumentos e o que
            devolveram) e a resposta.
        answer: A resposta (vazia se parou antes de responder).
        tools_used: Quantas ferramentas usou.
        notices: A leitura da resposta (os números verificados, uma resposta sem ferramentas…).
        duration_seconds: Quanto tempo demorou.
        stopped_early: Se parou a pedido, antes de responder.
    """

    question_id: str
    search_id: str
    request: AnalystRequest
    question: str
    model: str
    trace: tuple[AnalystStep, ...]
    answer: str
    tools_used: int
    notices: tuple[Notice, ...]
    duration_seconds: float
    stopped_early: bool = False


@dataclass(frozen=True, slots=True)
class AnalystUpdate:
    """O que mudou numa pergunta desde a última vez (o que a API transmite).

    Attributes:
        state: O retrato do trabalho agora (o progresso conta os passos).
        search_id: A procura de que fala.
        question: A pergunta.
        trace: Os passos novos (depois de ``after_step``), pela ordem.
        live: O que o modelo de linguagem está a pensar e a escrever agora, até aqui (``None``
            entre vezes e no fim).
    """

    state: JobState
    search_id: str
    question: str
    trace: tuple[AnalystStep, ...] = ()
    live: LiveTurn | None = None

    @property
    def steps(self) -> tuple[StepRecord, ...]:
        """Os passos novos como passos do trabalho (o WebSocket usa-os para saber onde vai)."""
        return tuple(StepRecord(step.number, None, {}) for step in self.trace)

    @property
    def snapshot_step(self) -> int | None:
        """O segundo cursor do WebSocket: a pergunta não tem snapshots."""
        return None


@dataclass(frozen=True, slots=True, eq=False)
class _Question:
    """Uma pergunta feita nesta sessão: o pedido e os passos, enquanto decorre e depois."""

    question_id: str
    search_id: str
    request: AnalystRequest
    log: AnalystLog


class AnalystDesk:
    """As perguntas ao analista: lança-as em segundo plano e encontra-as pelo id.

    Args:
        llm: O modelo de linguagem; sem ele, só se responde com respostas guardadas.
    """

    def __init__(self, llm: LanguageModelService | None) -> None:
        self._llm = llm
        self._jobs: JobRegistry[AnalystAnswer] = JobRegistry(KEPT_QUESTIONS)
        self._questions: OrderedDict[str, _Question] = OrderedDict()
        self._answers: OrderedDict[str, AnalystAnswer] = OrderedDict()
        self._lock = threading.Lock()

    def options(self, search_id: str, facts: SearchFacts) -> AnalystOptions:
        """As sugestões de perguntas para esta procura e as ferramentas do analista."""
        return AnalystOptions(
            search_id=search_id,
            suggestions=suggestions(facts),
            tools=tuple(AnalystToolInfo(tool.name, tool.label, tool.purpose) for tool in TOOLS),
            model=None if self._llm is None else self._llm.model_name,
            max_turns=MAX_TURNS,
        )

    def ask(
        self, search_id: str, facts: SearchFacts, request: AnalystRequest, *, seed: int
    ) -> TrainingJob[AnalystAnswer]:
        """Faz a pergunta em segundo plano (cada passo do analista é um passo do trabalho).

        Raises:
            LanguageModelError: Se for preciso o Ollama e ele não estiver pronto.
        """
        analyst = Analyst(self._language_model(request), facts, seed=seed)
        question_id = f"question-{uuid.uuid4().hex[:8]}"
        recorder = SnapshotRecorder(question_id)

        def counted(step: AnalystStep) -> None:
            recorder.receive(StepCompleted(run_id=question_id, step=step.number))

        entry = _Question(question_id, search_id, request, AnalystLog(counted))

        def work(control: TrainingControl) -> AnalystAnswer:
            return self._answer(entry, analyst, facts, control)

        job: TrainingJob[AnalystAnswer] = TrainingJob(question_id, work, recorder)
        with self._lock:
            self._questions[question_id] = entry
            self._forget_old_questions(keep=question_id)
        self._jobs.add(job.start())
        return job

    def answers(self, search_id: str) -> tuple[AnalystAnswer, ...]:
        """As perguntas respondidas sobre uma procura, pela ordem em que se fizeram."""
        with self._lock:
            return tuple(a for a in self._answers.values() if a.search_id == search_id)

    def job(self, question_id: str) -> JobState:
        """O estado de uma pergunta.

        Raises:
            QuestionNotFoundError: Se não houver nenhuma pergunta com esse identificador.
        """
        return self._job(question_id).state()

    def cancel(self, question_id: str) -> JobState:
        """Pede ao analista que pare (a meio da resposta do modelo de linguagem, se for preciso).

        Raises:
            QuestionNotFoundError: Se não houver nenhuma pergunta com esse identificador.
        """
        job = self._job(question_id)
        job.cancel()
        return job.state()

    def updates(self, question_id: str, after_step: int = 0) -> AnalystUpdate:
        """O que mudou numa pergunta desde o passo ``after_step``.

        Raises:
            QuestionNotFoundError: Se não houver nenhuma pergunta com esse identificador.
        """
        state = self._job(question_id).state()  # primeiro: se acabou, os passos já estão todos
        with self._lock:
            entry = self._questions.get(question_id)
        if entry is None:
            raise QuestionNotFoundError(question_id)
        return AnalystUpdate(
            state=state,
            search_id=entry.search_id,
            question=entry.request.question,
            trace=entry.log.steps(after_step),
            live=entry.log.live,
        )

    def result(self, question_id: str) -> AnalystAnswer:
        """A resposta a uma pergunta.

        Raises:
            TrainingInProgressError: Se o analista ainda estiver a responder.
            TrainingFailedError: Se a pergunta falhou (o modelo de linguagem deixou de responder).
            QuestionNotFoundError: Se a pergunta já não estiver em memória.
        """
        if self._jobs.find(question_id) is not None:
            return self._jobs.result(question_id)
        with self._lock:
            answer = self._answers.get(question_id)
        if answer is None:
            raise QuestionNotFoundError(question_id)
        return answer

    # --- por dentro --------------------------------------------------------------------

    def _language_model(self, request: AnalystRequest) -> LanguageModel:
        """O modelo que responde: as respostas guardadas, se as houver; senão, o Ollama.

        Raises:
            LanguageModelError: Se for preciso o Ollama e ele não estiver pronto.
        """
        if request.replies:
            return ScriptedModel([_reply(saved) for saved in request.replies], name=SAVED_ANSWERS)
        if self._llm is None:
            raise LanguageModelError(
                "This lab has no language model connected: the analyst can't answer. Open "
                "Ollama, or ask with saved replies."
            )
        return self._llm.model()

    def _answer(
        self,
        entry: _Question,
        analyst: Analyst,
        facts: SearchFacts,
        control: TrainingControl,
    ) -> AnalystAnswer:
        """Responde à pergunta (no fio de execução da pergunta) e guarda a resposta."""
        start = time.perf_counter()
        question = entry.request.question
        outcome = analyst.answer(question, entry.log, stop_requested=lambda: control.stop_requested)
        trace = entry.log.steps()
        answer = AnalystAnswer(
            question_id=entry.question_id,
            search_id=entry.search_id,
            request=entry.request.model_copy(update={"replies": _recorded(outcome)}),
            question=question,
            model=analyst.model_name,
            trace=trace,
            answer=outcome.answer,
            tools_used=sum(step.tool is not None for step in trace),
            notices=answer_notices(question, outcome, trace, opening(facts, question)),
            duration_seconds=time.perf_counter() - start,
            stopped_early=outcome.stopped,
        )
        with self._lock:
            self._answers[entry.question_id] = answer
            while len(self._answers) > KEPT_QUESTIONS:
                self._answers.popitem(last=False)
        logger.info(
            "o analista respondeu",
            extra={
                "question_id": entry.question_id,
                "search_id": entry.search_id,
                "tools": answer.tools_used,
                "stopped": outcome.stopped,
            },
        )
        return answer

    def _forget_old_questions(self, *, keep: str) -> None:
        """Esquece as perguntas mais antigas, mas nunca uma que ainda decorre."""
        excess = len(self._questions) - 2 * KEPT_QUESTIONS
        finished = [
            question_id
            for question_id in self._questions
            if question_id != keep and ((job := self._jobs.find(question_id)) is None or job.done)
        ]
        for question_id in finished[: max(0, excess)]:
            del self._questions[question_id]

    def _job(self, question_id: str) -> TrainingJob[AnalystAnswer]:
        job = self._jobs.find(question_id)
        if job is None:
            raise QuestionNotFoundError(question_id)
        return job


def suggestions(facts: SearchFacts) -> tuple[str, ...]:
    """As perguntas que esta procura sabe responder, pela ordem em que se mostram.

    São regras do domínio (ADR 0019): a pergunta sobre o agente de IA só aparece se ele entrou, a
    da referência só se o vencedor não for a própria referência, e assim por diante.
    """
    asks: list[str] = []
    if facts.winner is not None:
        asks += ["Why did the winner win?", "Does the winner memorise the training data?"]
    elif facts.stopped_early and facts.ranking:
        asks.append("Which candidate was best when the search stopped?")
    if len(facts.ranking) > 1:
        asks.append("How close was the second best?")
    if facts.baseline is not None and facts.winner not in (None, facts.baseline):
        asks.append("How much better than the baseline is the winner?")
    if facts.ai_model is not None:
        asks.append("Did the AI agent beat the random search?")
    if any(candidate.failed for candidate in facts.candidates):
        asks.append("Why did some candidates fail?")
    asks.append("What did the data agent do to the data?")
    return tuple(asks)


def _reply(saved: RecordedReply) -> Reply:
    """Uma resposta guardada, de volta ao tipo da conversa."""
    calls = tuple(ToolCall(call.name, dict(call.arguments)) for call in saved.tool_calls)
    return Reply(saved.thinking, saved.content, calls)


def _recorded(outcome: AnalystOutcome) -> tuple[RecordedReply, ...]:
    """As respostas do modelo de linguagem, para o pedido da resposta as repetir."""
    return tuple(
        RecordedReply(
            thinking=reply.thinking,
            content=reply.content,
            tool_calls=tuple(
                RecordedToolCall(name=call.name, arguments=dict(call.arguments))
                for call in reply.tool_calls
            ),
        )
        for reply in outcome.replies
    )
