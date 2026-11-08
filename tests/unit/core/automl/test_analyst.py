"""O analista: o ciclo pensar → ferramenta → ler → responder, o registo e a leitura da resposta.

O modelo de linguagem é o ``ScriptedModel``, com as respostas escritas à mão (com as ferramentas
que pede), e a procura é a escrita à mão do ``conftest.py`` desta pasta: os testes correm sem o
Ollama.
"""

import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any

import pytest

from ml_lab.core.automl.analyst import (
    ANSWER_NOW,
    INSTRUCTIONS,
    MAX_TURNS,
    Analyst,
    AnalystLog,
    AnalystOutcome,
    AnalystStep,
    AnalystStepKind,
    answer_notices,
    numbers_in,
    opening,
    unsupported_numbers,
)
from ml_lab.core.automl.analyst_tools import TOOLS, SearchFacts, ToolUse
from ml_lab.core.llm import Chunk, LanguageModel, Message, Reply, ScriptedModel, Tool, ToolCall

QUESTION = "Why did the winner win?"
BOARD = ToolCall("list_candidates", {})


@pytest.fixture
def search(facts: Callable[..., SearchFacts]) -> SearchFacts:
    return facts()


def reply(thinking: str, *calls: ToolCall, content: str = "") -> Reply:
    return Reply(thinking, content, calls)


def ask(
    llm: LanguageModel, search: SearchFacts, **options: Any
) -> tuple[AnalystOutcome, AnalystLog]:
    log = AnalystLog()
    outcome = Analyst(llm, search, seed=40).answer(QUESTION, log, **options)
    return outcome, log


# --- o ciclo -------------------------------------------------------------------------


def test_it_thinks_uses_the_tools_reads_them_and_answers(search: SearchFacts) -> None:
    llm = ScriptedModel(
        [
            reply("The board first.", BOARD),
            reply(
                "Now the winner against the second.",
                ToolCall("get_candidate", {"number": 4}),
                ToolCall("compare_candidates", {"first": 4, "second": 3}),
            ),
            reply("I know enough.", content="#4 won by 1.0 points: a tie in practice."),
        ]
    )

    outcome, log = ask(llm, search)

    assert [step.kind for step in log.steps()] == [
        AnalystStepKind.THOUGHT,
        AnalystStepKind.TOOL,
        AnalystStepKind.THOUGHT,
        AnalystStepKind.TOOL,
        AnalystStepKind.TOOL,
        AnalystStepKind.THOUGHT,
        AnalystStepKind.ANSWER,
    ]
    assert [step.number for step in log.steps()] == list(range(1, 8))
    tools = [step.tool.name for step in log.steps() if step.tool is not None]
    assert tools == ["list_candidates", "get_candidate", "compare_candidates"]
    assert outcome.answer == "#4 won by 1.0 points: a tie in practice."
    assert (outcome.stopped, outcome.forced, len(outcome.replies)) == (False, False, 3)
    assert log.live is None  # acabou de responder


def test_every_call_thinks_offers_the_tools_and_moves_the_seed(search: SearchFacts) -> None:
    llm = ScriptedModel([reply("Board.", BOARD), reply("Done.", content="#4 won.")])

    ask(llm, search)

    assert [(call.think, call.seed) for call in llm.calls] == [(True, 40), (True, 41)]
    offered = [[tool.name for tool in call.tools] for call in llm.calls]
    assert offered == [[tool.name for tool in TOOLS]] * 2
    first = llm.calls[0].messages
    assert (first[0].role, first[0].content) == ("system", INSTRUCTIONS)
    assert first[1].content == opening(search, QUESTION)
    assert first[1].content.endswith(f"Question: {QUESTION}")


def test_what_a_tool_returns_goes_back_to_the_model_as_a_tool_message(
    search: SearchFacts,
) -> None:
    llm = ScriptedModel([reply("Board.", BOARD), reply("Done.", content="#4 won.")])

    _, log = ask(llm, search)

    assistant, tool = llm.calls[1].messages[-2:]
    assert (assistant.role, assistant.tool_calls) == ("assistant", (BOARD,))
    assert (tool.role, tool.tool_name) == ("tool", "list_candidates")
    used = log.steps()[1].tool
    assert used is not None
    assert tool.content == used.result
    assert json.loads(tool.content)["winner"]["number"] == 4


def test_a_refused_request_goes_back_so_the_model_can_correct_itself(
    search: SearchFacts,
) -> None:
    llm = ScriptedModel(
        [
            reply("Candidate 9?", ToolCall("get_candidate", {"number": 9})),
            reply("There are only five.", ToolCall("get_candidate", {"number": 4})),
            reply("Done.", content="#4 won."),
        ]
    )

    _, log = ask(llm, search)

    refused, read = (step.tool for step in log.steps() if step.tool is not None)
    assert refused is not None
    assert read is not None
    assert refused.error is not None
    assert "There's no candidate #9" in llm.calls[1].messages[-1].content
    assert read.error is None


def test_after_the_last_round_of_tools_it_has_to_answer(search: SearchFacts) -> None:
    # O modelo pede sempre o quadro: na última vez vai sem ferramentas e com o pedido de responder.
    llm = ScriptedModel(
        [reply("Again.", BOARD) for _ in range(MAX_TURNS)]
        + [reply("Fine.", BOARD, content="#4 won, by what the board says.")]
    )

    outcome, log = ask(llm, search)

    assert len(llm.calls) == MAX_TURNS + 1
    last = llm.calls[-1]
    assert last.tools == ()
    assert last.messages[-1] == Message("user", ANSWER_NOW)
    assert outcome.forced
    assert outcome.answer == "#4 won, by what the board says."
    # O que pediu na última vez não se usa: há uma ferramenta por cada uma das outras vezes.
    assert sum(step.tool is not None for step in log.steps()) == MAX_TURNS


def test_the_text_that_comes_with_the_tools_stays_with_the_thought(search: SearchFacts) -> None:
    llm = ScriptedModel(
        [
            reply("Board first.", BOARD, content="Let me look at the board."),
            reply("", content="#4 won."),
        ]
    )

    _, log = ask(llm, search)

    thought, _, answer = log.steps()
    assert thought.text == "Board first.\n\nLet me look at the board."
    assert (answer.kind, answer.text) == (AnalystStepKind.ANSWER, "#4 won.")


# --- ao vivo e a parar ---------------------------------------------------------------


class SlowModel:
    """Um modelo que pensa sem fim, uma palavra de cada vez, até lhe fecharem a resposta."""

    name = "slow"

    def __init__(self) -> None:
        self.closed = False
        self.pieces = 0

    def chat(
        self,
        messages: Sequence[Message],
        *,
        think: bool = False,
        schema: Mapping[str, Any] | None = None,
        tools: Sequence[Tool] = (),
        seed: int | None = None,
    ) -> Iterator[Chunk]:
        try:
            while True:
                self.pieces += 1
                yield Chunk(thinking="hmm ")
        finally:
            self.closed = True


def test_it_stops_when_asked_and_closes_the_answer_halfway(search: SearchFacts) -> None:
    llm = SlowModel()

    outcome, log = ask(llm, search, stop_requested=lambda: llm.pieces > 3)

    assert outcome.stopped
    assert outcome.answer == ""
    assert llm.closed  # parar fecha a ligação: o Ollama deixa de escrever
    assert log.live is None
    assert log.steps() == ()


def test_the_live_turn_grows_while_the_model_writes() -> None:
    log = AnalystLog()

    log.start_turn()
    log.think("Look at ")
    log.think("")
    log.think("the board.")
    log.write("The winner ")
    log.write("won.")
    live = log.live
    log.end_turn()

    assert live is not None
    assert (live.thinking, live.answer) == ("Look at the board.", "The winner won.")
    assert log.live is None


def test_each_step_is_counted_as_it_arrives() -> None:
    seen: list[AnalystStep] = []
    log = AnalystLog(on_step=seen.append)

    log.add(AnalystStepKind.THOUGHT, "First.")
    log.add(AnalystStepKind.ANSWER, "Then.")

    assert [step.number for step in seen] == [1, 2]
    assert [step.number for step in log.steps(after=1)] == [2]


# --- a leitura da resposta -----------------------------------------------------------


def notices(answer: str, *results: str, forced: bool = False) -> list[str]:
    """Os códigos dos avisos de uma resposta, com uma ferramenta lida por cada resultado."""
    trace = tuple(
        AnalystStep(
            number,
            AnalystStepKind.TOOL,
            "The board.",
            ToolUse("list_candidates", "The candidate board", {}, "The board.", result),
        )
        for number, result in enumerate(results, start=1)
    )
    outcome = AnalystOutcome(answer, (), forced=forced)
    return [notice.code for notice in answer_notices(QUESTION, outcome, trace)]


def test_every_number_of_the_answer_has_to_come_from_the_tools() -> None:
    board = '{"lead": "1.0 points", "number": 4, "validation": "93.0%"}'

    assert notices("#4 won with 93.0%, 1.0 points ahead.", board) == ["numbers_checked"]
    assert notices("#4 won with 95.5%.", board) == ["unsupported_numbers"]
    assert notices("The winner won.", board) == []  # sem números, nada a verificar


def test_an_answer_without_tools_does_not_come_from_the_search() -> None:
    assert notices("Trees are always the best.") == ["no_tools_used"]


def test_a_forced_answer_an_empty_one_and_a_stopped_one_are_said() -> None:
    assert notices("The winner won.", '{"number": 4}', forced=True) == ["forced_answer"]
    assert notices("", '{"number": 4}') == ["empty_answer"]
    stopped = answer_notices(QUESTION, AnalystOutcome("", (), stopped=True), ())
    assert [notice.code for notice in stopped] == ["stopped_early"]


def test_the_unsupported_numbers_are_listed_as_the_answer_writes_them() -> None:
    found = unsupported_numbers(
        "From 1,234 rows: 82.4% and 100%, a gap of 2.2", ["1234", "82.40", "100.0%"]
    )

    assert found == ("2.2",)


def test_numbers_are_compared_by_their_value() -> None:
    assert numbers_in("1,234 rows, 82.4% and 100%, then 3, 4") == {
        1234.0: "1,234",
        82.4: "82.4",
        100.0: "100",
        3.0: "3",
        4.0: "4",
    }
