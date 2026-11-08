"""As ferramentas do analista: o que cada uma lê de uma procura acabada, e o que recusa.

A procura é a escrita à mão do ``conftest.py`` desta pasta (a fixture ``facts``): cinco
candidatos com notas redondas, para cada leitura se conferir de cabeça.
"""

import json
from collections.abc import Callable
from typing import Any

import pytest

from ml_lab.core.automl import analyst_tools
from ml_lab.core.automl.analyst_tools import (
    NOT_TESTED,
    TOOLS,
    SearchFacts,
    ToolUse,
    use_tool,
)
from ml_lab.core.automl.search_log import Candidate
from ml_lab.core.llm import ToolCall

type Facts = Callable[..., SearchFacts]


def read(name: str, search: SearchFacts, **arguments: Any) -> tuple[ToolUse, Any]:
    use = use_tool(search, ToolCall(name, arguments))
    return use, json.loads(use.result)


# --- o quadro ------------------------------------------------------------------------


def test_the_board_puts_the_winner_apart_with_the_calls_to_know_more(facts: Facts) -> None:
    use, board = read("list_candidates", facts())

    assert use.error is None
    assert use.label == "The candidate board"
    assert use.summary == "All 4 candidates, best first, from 3 models."
    winner = board["winner"]
    assert (winner["number"], winner["model"], winner["validation"]) == (
        4,
        "Decision tree",
        "93.0% ± 2.0%",
    )
    assert winner["settings"] == "maximum depth 3 (default 5)"
    second = winner["second"]
    assert (second["number"], second["behind_the_first"]) == (3, "1.0 points")
    # Só o vencedor vai ao teste: dizê-lo evita que o modelo invente a nota do segundo.
    assert second["test"] == NOT_TESTED
    assert board["baseline"]["test"] == NOT_TESTED
    assert "can't be compared with the other candidates" in winner["test"]
    assert winner["test_reading"].startswith("On the test: 94.0%.")
    # O quadro é um índice: o empate e o treino ficam nas ferramentas dos pormenores, e o
    # vencedor diz quais chamar, com os números já escritos.
    assert "tied_in_practice" not in winner
    assert "training" not in winner
    assert [call.split("(")[0] for call in winner["see_also"]] == [
        "compare_candidates",
        "get_candidate",
    ]
    assert winner["see_also"][0].startswith("compare_candidates(first=4, second=3)")


def test_each_candidate_is_one_line_with_its_number_score_and_gap_together(facts: Facts) -> None:
    _, board = read("list_candidates", facts())

    # Uma linha por candidato: o número, o modelo e a nota juntos, para um modelo de linguagem
    # pequeno não trocar os números de um com os de outro.
    assert board["board"] == [
        "1st: #4 Decision tree; 93.0% ± 2.0%; maximum depth 3 (default 5); Round 2 · tuning; "
        "proposed by the AI agent",
        "2nd: #3 Decision tree; 92.0% ± 2.0%, 1.0 points behind the first; its default "
        "settings; Round 1 · screening",
        "3rd: #1 Logistic regression; 90.0% ± 3.0%, 3.0 points behind the first; its default "
        "settings; Round 1 · screening",
        "4th: #2 K-nearest neighbours (KNN); 88.0% ± 5.0%, 5.0 points behind the first; its "
        "default settings; Round 1 · screening",
    ]
    assert "not_shown" not in board
    assert board["baseline"]["behind_the_first"] == "3.0 points"
    assert board["failed"] == [{"number": 5, "model": "SVM", "error": "It ran out of memory."}]
    assert board["readings"] == ["The winner beats the baseline by 3.0 points."]


def test_the_board_says_every_model_once_with_its_best_candidate(facts: Facts) -> None:
    _, board = read("list_candidates", facts())

    # Pela ordem do melhor de cada um; o modelo em que falharam todos fica no fim.
    assert board["models"] == [
        "Decision tree: 2 candidates, the best is #4 with 93.0% ± 2.0%, the winner",
        "Logistic regression: 1 candidate, #1 with 90.0% ± 3.0%, 3.0 points behind the first",
        "K-nearest neighbours (KNN): 1 candidate, #2 with 88.0% ± 5.0%, 5.0 points behind the "
        "first",
        "SVM: 1 candidate, it failed",
    ]


def test_the_board_always_has_every_candidate_even_if_a_limit_is_asked(facts: Facts) -> None:
    # Uma resposta guardada da etapa 13.3 ainda pode pedir um limite: o quadro vai inteiro.
    use, board = read("list_candidates", facts(), limit=2)

    assert use.error is None
    assert len(board["board"]) == 4
    assert "limit" not in TOOLS[0].tool.parameters.get("properties", {})


def test_a_board_too_long_for_the_context_keeps_every_models_best(
    facts: Facts, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(analyst_tools, "MAX_ROWS", 2)

    use, board = read("list_candidates", facts())

    assert [row.split(";")[0] for row in board["board"]] == [
        "1st: #4 Decision tree",
        "2nd: #3 Decision tree",
    ]
    assert board["not_shown"].startswith("2 more candidates below these, the weakest ones")
    # Os candidatos de fora das linhas continuam nos modelos, com o melhor de cada um.
    assert any(line.startswith("K-nearest neighbours (KNN)") for line in board["models"])
    assert use.summary == "The best 2 of 4 candidates, and the best of each of the 3 models."


def test_a_baseline_that_wins_is_said_in_words(facts: Facts) -> None:
    _, board = read("list_candidates", facts(ranking=(1, 3, 4, 2), winner=1))

    assert board["baseline"].startswith("#1 Logistic regression, with its default settings, is ")


def test_a_stopped_search_has_the_best_so_far_and_no_winner(facts: Facts) -> None:
    _, board = read("list_candidates", facts(winner=None, test=None, stopped_early=True))

    assert "winner" not in board
    assert board["best_so_far"]["number"] == 4
    assert "stopped before a winner" in board["best_so_far"]["note"]


# --- um candidato --------------------------------------------------------------------


def test_a_candidate_in_detail_says_who_proposed_it_and_what_its_agent_said(
    facts: Facts,
) -> None:
    use, found = read("get_candidate", facts(), number=4)

    assert use.summary == "#4 Decision tree: 93.0% ± 2.0%."
    assert found["rank"] == "1st of 4"
    assert found["winner"] is True
    assert found["folds"] == ["92.0%", "93.0%", "94.0%"]
    assert (found["training"], found["memorises_training"]) == ("95.0%", False)
    assert found["meaning"] == "It gets about 93 in every 100 rows right."
    assert found["proposed_by"] == "the AI agent"
    assert found["reason_given"].startswith("The tree memorises")
    assert found["what_its_agent_said"].startswith("I measured Decision tree")
    assert found["test"].startswith("94.0%. Only the winner goes to the test")
    assert found["time"] == "1.2 s for all the folds"


def test_a_candidate_says_whether_it_memorises_the_training_data(facts: Facts) -> None:
    _, tree = read("get_candidate", facts(), number=3)

    # 100% no treino e 92% nos folds que não viu: decora (a regra do Objective).
    assert (tree["training"], tree["memorises_training"]) == ("100.0%", True)
    assert "test" not in tree  # não é o vencedor: não foi ao teste


def test_a_failed_candidate_says_why(facts: Facts) -> None:
    use, found = read("get_candidate", facts(), number=5)

    assert use.error is None
    assert (found["rank"], found["failed"]) == ("failed", "It ran out of memory.")
    assert use.summary == "#5 SVM: it failed."
    assert found["proposed_by"] == "Agent · Kernel methods"


@pytest.mark.parametrize("number", ["#4", "4", 4.0])
def test_the_number_is_read_as_the_models_write_it(facts: Facts, number: object) -> None:
    use, found = read("get_candidate", facts(), number=number)

    assert use.error is None
    assert found["number"] == 4


@pytest.mark.parametrize(
    ("arguments", "problem"),
    [
        ({}, "“number” is missing"),
        ({"number": "four"}, "must be a whole number"),
        ({"number": 9}, "There's no candidate #9 in this search: it has candidates #1 to #5."),
    ],
)
def test_a_request_that_does_not_fit_is_refused_with_the_reason(
    facts: Facts, arguments: dict[str, Any], problem: str
) -> None:
    use, found = read("get_candidate", facts(), **arguments)

    assert use.error is not None
    assert problem in use.error
    assert found == {"error": use.error}  # o modelo lê a razão como o resultado
    assert use.summary == use.error


# --- dois candidatos -----------------------------------------------------------------


def test_two_candidates_side_by_side_with_the_folds_each_one_won(facts: Facts) -> None:
    use, found = read("compare_candidates", facts(), first=3, second=4)

    assert found["better"] == "#4 (Decision tree)"
    assert (found["gap"], found["tied_in_practice"]) == ("1.0 points", True)
    # 92% contra 90% e 93% contra 92% para o #4; 94% contra 94% é um empate.
    assert found["folds_won"] == {"#3": 0, "#4": 2, "level": 1}
    assert found["settings_that_differ"] == {"max_depth": {"#3": 5, "#4": 3}}
    assert found["first"]["memorises_training"] is True
    assert use.summary == (
        "#3 Decision tree against #4 Decision tree: level in practice (1.0 points apart, less "
        "than the spread between folds)."
    )


def test_a_clear_gap_is_not_a_tie(facts: Facts) -> None:
    use, found = read("compare_candidates", facts(), first=4, second=2)

    assert (found["gap"], found["tied_in_practice"]) == ("5.0 points", False)
    assert found["models"].startswith("Two different models")
    assert use.summary.endswith("#4 is 5.0 points better.")


@pytest.mark.parametrize(
    ("mean", "std", "why"),
    [
        (0.9301, 0.02, "its mean is a little better, in decimals the board doesn't show"),
        (0.93, 0.01, "its spread between folds is smaller"),
        (0.93, 0.02, "it was measured first"),
    ],
)
def test_when_the_scores_look_the_same_it_says_why_one_is_ahead(
    facts: Facts,
    make_candidate: Callable[..., Candidate],
    mean: float,
    std: float,
    why: str,
) -> None:
    # Dois candidatos que o quadro escreve iguais (93.0%): o desempate do coordenador.
    ahead = make_candidate(1, "decision_tree", mean, std, folds=(0.93,))
    behind = make_candidate(2, "decision_tree", 0.93, 0.02, folds=(0.93,))
    search = facts(candidates=(ahead, behind), ranking=(1, 2), winner=1, baseline=None)

    _, found = read("compare_candidates", search, first=2, second=1)

    assert found["why_ahead"].startswith("#1 is ahead because")
    assert why in found["why_ahead"]


@pytest.mark.parametrize(
    ("first", "second", "problem"),
    [
        (4, 4, "Choose two different candidates"),
        (4, 5, "Candidate #5 (SVM) failed, so it has no score to compare"),
        (4, 9, "There's no candidate #9"),
    ],
)
def test_a_comparison_that_does_not_fit_is_refused(
    facts: Facts, first: int, second: int, problem: str
) -> None:
    use, _ = read("compare_candidates", facts(), first=first, second=second)

    assert use.error is not None
    assert problem in use.error


# --- o plano dos dados e as ferramentas ----------------------------------------------


def test_the_data_plan_has_the_decisions_the_rows_and_who_took_part(facts: Facts) -> None:
    use, plan = read("data_plan", facts())

    assert plan["decisions"][0] == {
        "decision": "Drop duplicate rows",
        "reason": "One row repeats another.",
    }
    assert plan["rows"] == {"training": 119, "test": 30, "used_by_the_cross_validation": 100}
    assert "sample of 100 of the 119 training rows" in plan["sample"]
    assert plan["ai_agent"] == "Took part in the tuning, thinking with qwen3:8b."
    assert plan["left_out"] == [{"models": ["LOF", "t-SNE"], "reason": "it only knows its rows"}]
    assert use.summary == "2 decisions of the data agent, 119 training rows and 30 test rows."


def test_an_unknown_tool_lists_the_ones_there_are(facts: Facts) -> None:
    use, found = read("delete_everything", facts())

    assert use.error is not None
    assert "There's no tool called “delete_everything”" in use.error
    assert all(tool.name in use.error for tool in TOOLS)
    assert found == {"error": use.error}


def test_every_tool_tells_the_model_what_it_does_and_the_person_too() -> None:
    names = [tool.name for tool in TOOLS]

    assert names == ["list_candidates", "get_candidate", "compare_candidates", "data_plan"]
    for tool in TOOLS:
        assert tool.tool.description
        assert tool.tool.parameters["type"] == "object"
        assert tool.label
        assert tool.purpose
