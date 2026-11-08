"""O analista: responde a perguntas sobre uma procura acabada, usando ferramentas (ADR 0053).

O agente de IA da afinação (``ai_agent.py``) recebe tudo de uma vez, num texto, e responde num
formato fixo. O analista trabalha de outra maneira, a de um agente com ferramentas: começa só
com a pergunta e decide o que precisa de ler. O ciclo é sempre o mesmo:

1. **pensa** (o raciocínio chega aos bocados, e a interface mostra-o ao vivo);
2. **pede uma ferramenta** (ou várias): o quadro, um candidato, dois lado a lado, o plano dos
   dados (``analyst_tools.py``);
3. **lê o resultado**, que volta à conversa como uma mensagem ``tool``;
4. volta a pensar e, quando já sabe o suficiente, **responde** por palavras.

Cada passo fica no ``AnalystLog``: o raciocínio, cada ferramenta com os argumentos e o que
devolveu, e a resposta. O que o mantém honesto é o mesmo da ADR 0042: **a IA explica; as regras
medem**. O analista não mede nada nem vê o teste para lá do que o resultado já mostra, porque as
ferramentas só leem o que a procura mediu. E a resposta passa por uma verificação: cada número
que ela diz tem de estar no que as ferramentas devolveram (``answer_notices``). Um modelo de
linguagem pode inventar um número que soa bem; aqui, isso vê-se.

Ao fim de ``MAX_TURNS`` voltas com ferramentas, o analista tem de responder com o que tem: um
modelo de linguagem pode ficar a pedir ferramentas sem fim.
"""

import re
import threading
from collections.abc import Callable, Generator, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from types import MappingProxyType
from typing import Final

from ml_lab.core.automl.analyst_tools import TOOLS, SearchFacts, ToolUse, use_tool
from ml_lab.core.foundation.notices import Notice, sort_notices
from ml_lab.core.llm import LanguageModel, Message, Reply, Tool, ToolCall
from ml_lab.formatting import format_list

MAX_TURNS: Final = 6
"""Quantas vezes o analista pode pedir ferramentas antes de ter de responder."""

INSTRUCTIONS: Final = (
    "You are the analyst of a machine learning model search, in a lab for learning. People ask "
    "you about a search that has finished: why the winner won, how two candidates compare, "
    "whether a model memorises the training data, what the data agent did. You don't see the "
    "search: you only know what your tools return, so use them before you answer.\n\n"
    "How to work:\n"
    "1. Read the board first (list_candidates). It's an index of the whole search: who won, "
    "every model that took part with its best candidate, and every candidate with its score and "
    "how far behind the first it is. The details are in the other tools.\n"
    "2. Then read the details the question needs: compare_candidates for why one candidate beat "
    "another (the folds each one won, the settings that differ), get_candidate for one "
    "candidate (each fold, who proposed it and why), data_plan for what the data agent did. To "
    "explain why the winner won, always compare it with the second.\n"
    "3. Answer when the tools have told you enough.\n\n"
    "Rules:\n"
    "- Base every statement on what the tools returned, and name the candidates by their number "
    "and model (#7, Random forest). If the tools don't say something, say you don't know.\n"
    "- Quote the numbers exactly as the tools write them. Don't compute new numbers: the tools "
    "already give the gaps, the ties, the folds each candidate won and whether a model memorises "
    "the training data.\n"
    "- Explain with what this search measured, not with what models usually do in general.\n"
    "- The cross-validation used the training data only. Only the winner went to the test, once: "
    "no other candidate has a test score, so never say how another one did on the test.\n"
    "- Think in English. Answer in English, in a short paragraph or a few bullet points, for "
    "someone learning machine learning."
)
"""As instruções do analista (a mensagem de sistema): o papel, como trabalhar e as regras. Em
inglês, como o site (ADR 0043): o raciocínio e a resposta aparecem no ecrã. Cada regra nasceu de
um erro do ``qwen3:8b`` a sério (as contas feitas à mão, o teste comparado com a validação dos
outros, a resposta só com o quadro): ver as notas da etapa 13.3 em ``docs/architecture.md``."""

ANSWER_NOW: Final = (
    "You've used all your tool calls. Answer the question now, with what the tools returned."
)
"""O que se diz ao analista quando acaba as voltas com ferramentas (a última vai sem elas)."""


class AnalystStepKind(StrEnum):
    """O que é cada passo do analista."""

    THOUGHT = "thought"
    TOOL = "tool"
    ANSWER = "answer"

    @property
    def label(self) -> str:
        """Nome a mostrar, em inglês."""
        return _STEP_LABELS[self]


_STEP_LABELS: Final[Mapping[AnalystStepKind, str]] = MappingProxyType(
    {
        AnalystStepKind.THOUGHT: "Thought",
        AnalystStepKind.TOOL: "Tool",
        AnalystStepKind.ANSWER: "Answer",
    }
)


@dataclass(frozen=True, slots=True)
class AnalystStep:
    """Um passo do analista: um raciocínio, uma ferramenta usada ou a resposta.

    Attributes:
        number: A ordem (1, 2, 3…): quem acompanha pede só os passos novos.
        kind: O que o passo é.
        text: O raciocínio, o resumo do que a ferramenta devolveu, ou a resposta.
        tool: Numa ferramenta, o que se pediu e o que se leu.
    """

    number: int
    kind: AnalystStepKind
    text: str = ""
    tool: ToolUse | None = None


@dataclass(frozen=True, slots=True)
class LiveTurn:
    """A vez do modelo de linguagem a decorrer: o que já pensou e escreveu, a crescer.

    Attributes:
        thinking: O raciocínio até aqui.
        answer: O texto até aqui (a resposta, se for a última vez).
    """

    thinking: str = ""
    answer: str = ""


class AnalystLog:
    """O que o analista vai fazendo, passo a passo (seguro entre fios de execução).

    O analista escreve num fio de execução, e quem acompanha (a API) lê noutro, como no
    ``SearchLog`` da procura.

    Args:
        on_step: Chamado com cada passo novo (o serviço conta-o como um passo do trabalho).
    """

    def __init__(self, on_step: Callable[[AnalystStep], None] | None = None) -> None:
        self._on_step = on_step
        self._steps: list[AnalystStep] = []
        self._live: LiveTurn | None = None
        self._lock = threading.Lock()

    def start_turn(self) -> None:
        """O modelo de linguagem começa a responder."""
        with self._lock:
            self._live = LiveTurn()

    def think(self, piece: str) -> None:
        """Mais um bocado do raciocínio."""
        if not piece:
            return
        with self._lock:
            if self._live is not None:
                self._live = replace(self._live, thinking=self._live.thinking + piece)

    def write(self, piece: str) -> None:
        """Mais um bocado do texto."""
        if not piece:
            return
        with self._lock:
            if self._live is not None:
                self._live = replace(self._live, answer=self._live.answer + piece)

    def end_turn(self) -> None:
        """O modelo de linguagem acabou de responder (o que disse fica nos passos)."""
        with self._lock:
            self._live = None

    @property
    def live(self) -> LiveTurn | None:
        """A vez a decorrer (``None`` entre vezes e no fim)."""
        with self._lock:
            return self._live

    def add(self, kind: AnalystStepKind, text: str, tool: ToolUse | None = None) -> AnalystStep:
        """Junta um passo."""
        with self._lock:
            step = AnalystStep(len(self._steps) + 1, kind, text, tool)
            self._steps.append(step)
        if self._on_step is not None:
            self._on_step(step)
        return step

    def steps(self, after: int = 0) -> tuple[AnalystStep, ...]:
        """Os passos depois do número ``after``."""
        with self._lock:
            return tuple(self._steps[after:])


@dataclass(frozen=True, slots=True)
class AnalystOutcome:
    """Como acabou uma pergunta.

    Attributes:
        answer: A resposta (vazia se parou antes de responder).
        replies: Cada resposta do modelo de linguagem, pela ordem: com elas, a pergunta repete-se
            sem o modelo (o ``ScriptedModel`` dá-as outra vez).
        stopped: Se parou a pedido, antes de responder.
        forced: Se gastou as voltas com ferramentas e teve de responder com o que tinha.
    """

    answer: str
    replies: tuple[Reply, ...]
    stopped: bool = False
    forced: bool = False


class Analyst:
    """O analista de uma procura acabada.

    Args:
        llm: O modelo de linguagem (o Ollama, ou o ``ScriptedModel`` nos testes e ao repetir).
        facts: O que a procura mediu e disse, para as ferramentas lerem.
        seed: A seed das respostas (cada vez usa a seguinte).
    """

    def __init__(self, llm: LanguageModel, facts: SearchFacts, *, seed: int) -> None:
        self._llm = llm
        self._facts = facts
        self._seed = seed

    @property
    def model_name(self) -> str:
        """O modelo de linguagem que pensa por ele ("qwen3:8b")."""
        return self._llm.name

    def answer(
        self,
        question: str,
        log: AnalystLog,
        *,
        stop_requested: Callable[[], bool] | None = None,
    ) -> AnalystOutcome:
        """Responde à pergunta: pensa, usa as ferramentas que quiser e responde.

        Args:
            question: A pergunta.
            log: Onde fica cada passo (e o que está a pensar agora).
            stop_requested: Diz se é para parar; pergunta-se a cada bocado da resposta.

        Raises:
            LanguageModelError: Se o modelo de linguagem não responder.
        """
        messages = [
            Message("system", INSTRUCTIONS),
            Message("user", opening(self._facts, question)),
        ]
        tools = tuple(tool.tool for tool in TOOLS)
        replies: list[Reply] = []
        turn = 0
        while True:
            last = turn >= MAX_TURNS
            if last:
                messages.append(Message("user", ANSWER_NOW))
            reply = self._turn(
                messages, () if last else tools, self._seed + turn, log, stop_requested
            )
            if reply is None:
                return AnalystOutcome("", tuple(replies), stopped=True)
            replies.append(reply)
            # Na última vez não há ferramentas: o que pedir não se usa, e o texto é a resposta.
            calls = () if last else reply.tool_calls
            note = _note(reply, calls)
            if note:
                log.add(AnalystStepKind.THOUGHT, note)
            if not calls:
                answer = reply.content.strip()
                if answer:
                    log.add(AnalystStepKind.ANSWER, answer)
                return AnalystOutcome(answer, tuple(replies), forced=last)
            messages.append(Message("assistant", reply.content, calls))
            for call in calls:
                use = use_tool(self._facts, call)
                log.add(AnalystStepKind.TOOL, use.summary, use)
                messages.append(Message("tool", use.result, tool_name=use.name))
            turn += 1

    def _turn(
        self,
        messages: Sequence[Message],
        tools: Sequence[Tool],
        seed: int,
        log: AnalystLog,
        stop_requested: Callable[[], bool] | None,
    ) -> Reply | None:
        """Uma vez do modelo de linguagem, aos bocados; ``None`` se for para parar a meio."""
        if stop_requested is not None and stop_requested():
            return None
        thinking: list[str] = []
        content: list[str] = []
        calls: list[ToolCall] = []
        log.start_turn()
        stream = self._llm.chat(messages, think=True, tools=tools, seed=seed)
        try:
            for chunk in stream:
                if stop_requested is not None and stop_requested():
                    return None
                thinking.append(chunk.thinking)
                content.append(chunk.content)
                calls.extend(chunk.tool_calls)
                log.think(chunk.thinking)
                log.write(chunk.content)
        finally:
            # Parar a meio fecha a resposta: o Ollama deixa de escrever quando a ligação fecha.
            if isinstance(stream, Generator):
                stream.close()
            log.end_turn()
        return Reply("".join(thinking), "".join(content), tuple(calls))


def opening(facts: SearchFacts, question: str) -> str:
    """O pedido ao analista: a pergunta, com o que é a procura (os números vêm das ferramentas)."""
    objective = facts.objective
    return (
        f"The search: predict {facts.target} in {facts.dataset} "
        f"({facts.task_type.label.lower()}), optimising {objective.label} "
        f"({objective.direction}). Use the tools to see what the search measured, then answer."
        f"\n\nQuestion: {question}"
    )


def answer_notices(
    question: str, outcome: AnalystOutcome, trace: Sequence[AnalystStep], context: str = ""
) -> tuple[Notice, ...]:
    """A leitura da resposta: as regras que verificam o que o analista disse.

    - sem nenhuma ferramenta lida, a resposta não vem da procura (``no_tools_used``);
    - cada número da resposta tem de estar no que as ferramentas devolveram, na pergunta ou no
      pedido (``unsupported_numbers``, ou ``numbers_checked`` quando estão todos);
    - gastar as voltas (``forced_answer``), não responder (``empty_answer``) e parar a pedido
      (``stopped_early``) também se dizem.

    Args:
        question: A pergunta.
        outcome: Como acabou.
        trace: Os passos do analista.
        context: O resto do pedido que o analista leu (``opening``).
    """
    if outcome.stopped:
        return (
            Notice.info(
                "stopped_early",
                "You stopped the analyst before it answered: the steps it took stay above.",
            ),
        )
    notices: list[Notice] = []
    if not outcome.answer:
        notices.append(
            Notice.warning(
                "empty_answer",
                "The analyst didn't write an answer. Ask again, perhaps in other words.",
            )
        )
    if outcome.forced:
        notices.append(
            Notice.info(
                "forced_answer",
                f"The analyst used its {MAX_TURNS} rounds of tools and had to answer with what "
                "it had read.",
            )
        )
    read = [step.tool for step in trace if step.tool is not None and step.tool.error is None]
    if outcome.answer and not read:
        notices.append(
            Notice.warning(
                "no_tools_used",
                "The analyst answered without reading any tool: nothing in its answer comes from "
                "this search. Ask again, or read the board yourself.",
            )
        )
    elif outcome.answer:
        sources = [use.result for use in read] + [question, context]
        unsupported = unsupported_numbers(outcome.answer, sources)
        if unsupported:
            notices.append(
                Notice.warning(
                    "unsupported_numbers",
                    f"The answer has numbers that none of the tools returned: "
                    f"{format_list(list(unsupported))}. Check them against what the tools "
                    "returned, above.",
                )
            )
        elif numbers_in(outcome.answer):
            notices.append(
                Notice.info(
                    "numbers_checked",
                    "Every number in the answer is in what the tools returned.",
                )
            )
    return sort_notices(notices)


_NUMBER: Final = re.compile(r"\d[\d,]*(?:\.\d+)?")
"""Um número escrito: os algarismos, a vírgula dos milhares e as casas decimais."""


def numbers_in(text: str) -> dict[float, str]:
    """Os números de um texto: o valor e a primeira forma em que aparece ("1,234" vale 1234).

    Comparam-se pelo valor, para "100%" e "100.0%" serem o mesmo número.
    """
    found: dict[float, str] = {}
    for match in _NUMBER.finditer(text):
        written = match.group().rstrip(",")
        try:
            value = float(written.replace(",", ""))
        except ValueError:  # uma vírgula sozinha entre algarismos ("3,,4"): não é um número
            continue
        found.setdefault(round(value, 9), written)
    return found


def unsupported_numbers(answer: str, sources: Iterable[str]) -> tuple[str, ...]:
    """Os números da resposta que não aparecem em nenhuma das fontes, pela ordem da resposta."""
    known: set[float] = set()
    for source in sources:
        known.update(numbers_in(source))
    return tuple(written for value, written in numbers_in(answer).items() if value not in known)


def _note(reply: Reply, calls: Sequence[ToolCall]) -> str:
    """O raciocínio de uma vez; com ferramentas, o texto que vier com elas também conta."""
    parts = [reply.thinking.strip()]
    if calls:
        parts.append(reply.content.strip())
    return "\n\n".join(part for part in parts if part)
