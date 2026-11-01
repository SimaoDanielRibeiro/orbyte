"""O agente de IA: o que pergunta ao modelo de linguagem, o que aceita, o que recusa e a sua vez.

O modelo de linguagem é o ``ScriptedModel``, com as respostas escritas à mão: os testes correm
sem o Ollama e sabem o que o "modelo" vai responder. As procuras usam o Iris com três folds e
as três famílias rápidas, como os testes do coordenador.
"""

import json
from typing import Any

import pytest

from ml_lab.core.automl.ai_agent import (
    MAX_REFUSALS,
    AiAgent,
    Finalist,
    answer_schema,
    briefing,
    read_answer,
)
from ml_lab.core.automl.coordinator import Coordinator, SearchOutcome
from ml_lab.core.automl.model_agent import model_agents
from ml_lab.core.automl.objective import objective_for
from ml_lab.core.automl.search_log import (
    AI_AGENT,
    AgentRole,
    Candidate,
    MessageKind,
    SearchLog,
    SearchPhase,
)
from ml_lab.core.data.io.base import SourceSpec
from ml_lab.core.data.preparation import PreparationConfig
from ml_lab.core.data.splitting.base import SplitConfig, SplitStrategy
from ml_lab.core.foundation.types import TaskType
from ml_lab.core.llm import Reply, ScriptedModel
from ml_lab.core.models.base import ModelFamily
from ml_lab.core.services.datasets import DatasetService

IRIS = SourceSpec("sample", {"name": "iris"})
SPECIES = PreparationConfig("species", split=SplitConfig(SplitStrategy.STRATIFIED))
FAST = (ModelFamily.LINEAR, ModelFamily.INSTANCE_BASED, ModelFamily.TREE)
MULTICLASS = TaskType.MULTICLASS_CLASSIFICATION
OBJECTIVE = objective_for(MULTICLASS)


def finalists() -> list[Finalist]:
    agents, _ = model_agents(MULTICLASS, OBJECTIVE, families=FAST)
    return [(agent, model) for agent in agents for model in agent.models]


def answer(model: str, values: dict[str, Any], reason: str = "Árvores rasas decoram menos.") -> str:
    return json.dumps({"model": model, "values": values, "reason": reason})


def thinking_reply(content: str) -> Reply:
    return Reply("A árvore decora o treino: vou baixar a profundidade.", content)


# --- ler a resposta ------------------------------------------------------------------


def test_a_good_answer_becomes_a_complete_proposal() -> None:
    proposal, problem = read_answer(answer("decision_tree", {"max_depth": 3.0}), finalists())

    assert problem is None
    assert proposal is not None
    assert proposal.model.info.name == "decision_tree"
    assert proposal.values["max_depth"] == 3  # o 3.0 do modelo de linguagem passa a inteiro
    assert proposal.values["criterion"] == proposal.agent.defaults(proposal.model)["criterion"]
    assert proposal.reason == "Árvores rasas decoram menos."


@pytest.mark.parametrize(
    ("content", "problem"),
    [
        ("isto não é JSON", "não é o JSON pedido"),
        (answer("svm", {}), "não é um dos finalistas"),
        (answer("decision_tree", {"max_depth": 7}), "não é um dos valores procurados"),
        (answer("knn", {"n_neighbors": 99}), "sai do espaço procurado"),
        (answer("knn", {"n_neighbors": 2.5}), "tem de ser um número inteiro"),
        (answer("decision_tree", {"splitter": "random"}), "não é um hiperparâmetro procurado"),
    ],
)
def test_an_answer_that_does_not_fit_is_refused_with_the_reason(content: str, problem: str) -> None:
    proposal, found = read_answer(content, finalists())

    assert proposal is None
    assert found is not None
    assert problem in found


def test_a_combination_already_measured_is_refused() -> None:
    chosen = finalists()
    agent, model = next(entry for entry in chosen if entry[1].info.name == "decision_tree")
    agent.remember(model, {**agent.defaults(model), "max_depth": 3})

    proposal, problem = read_answer(answer("decision_tree", {"max_depth": 3}), chosen)

    assert proposal is None
    assert problem is not None
    assert "já foi medida" in problem


# --- o que pergunta ------------------------------------------------------------------


def test_the_answer_format_only_allows_the_finalists_and_their_spaces() -> None:
    schema = answer_schema(finalists())

    assert schema["required"] == ["model", "values", "reason"]
    assert schema["properties"]["model"]["enum"] == ["logistic_regression", "knn", "decision_tree"]
    values = schema["properties"]["values"]["properties"]
    assert values["max_depth"] == {"enum": [2, 3, 4, 5, 6, 8, 10, 0]}
    assert values["n_neighbors"] == {"type": "integer", "minimum": 1, "maximum": 30}


def test_the_briefing_has_the_spaces_and_every_result_so_far() -> None:
    measured = Candidate(
        number=1,
        phase=SearchPhase.SCREENING,
        family=ModelFamily.TREE,
        model="decision_tree",
        model_display_name="Árvore de decisão",
        hyperparameters={},
        changes={},
        mean=0.9123,
        std=0.0201,
        train_mean=1.0,
    )

    text = briefing(OBJECTIVE, MULTICLASS, finalists(), [measured])

    assert "decision_tree (Árvore de decisão)" in text
    assert "max_depth: um destes: [2, 3, 4, 5, 6, 8, 10, 0]" in text
    assert "#1 decision_tree valores por omissão: 0.9123 ± 0.0201, no treino 1.0000" in text


def test_it_thinks_out_loud_and_keeps_every_reply() -> None:
    llm = ScriptedModel([thinking_reply(answer("decision_tree", {"max_depth": 3}))])
    ai = AiAgent(llm, OBJECTIVE, MULTICLASS, seed=11)
    pieces: list[str] = []

    turn = ai.propose(finalists(), [], on_thinking=pieces.append)

    assert turn.proposal is not None
    assert "".join(pieces) == "A árvore decora o treino: vou baixar a profundidade."
    assert len([piece for piece in pieces if piece]) > 1  # chegou aos bocados
    assert ai.replies == [turn.reply]
    call = llm.calls[0]
    assert (call.think, call.seed) == (True, 11)
    assert call.schema == answer_schema(finalists())
    assert call.messages[0].role == "system"


# --- a vez dele, na procura ----------------------------------------------------------


def search(llm: ScriptedModel, *, budget: int) -> tuple[SearchOutcome, SearchLog]:
    agents, _ = model_agents(MULTICLASS, OBJECTIVE, families=FAST)
    log = SearchLog()
    coordinator = Coordinator(
        OBJECTIVE,
        agents,
        DatasetService().folds(IRIS, SPECIES, 7, 3),
        budget=budget,
        finalists=3,
        model_seed=0,
        search_seed=1,
        log=log,
        baseline="logistic_regression",
        ai=AiAgent(llm, OBJECTIVE, MULTICLASS, seed=0),
    )
    return coordinator.search(), log


def test_its_proposal_is_measured_like_the_others_and_says_why() -> None:
    reply = thinking_reply(answer("decision_tree", {"max_depth": 2}))

    outcome, log = search(ScriptedModel([reply]), budget=7)

    # Triagem (3), uma vez de cada finalista (3) e a vez do agente de IA (o 7.º).
    proposed = outcome.candidates[-1]
    assert proposed.ai
    assert proposed.phase is SearchPhase.TUNING
    assert proposed.hyperparameters["max_depth"] == 2
    assert proposed.reason == "Árvores rasas decoram menos."
    assert proposed.mean is not None
    assert not any(candidate.ai for candidate in outcome.candidates[:-1])
    thoughts = [m for m in log.messages() if m.kind is MessageKind.THOUGHT]
    assert [(m.role, m.author, m.text, m.candidate) for m in thoughts] == [
        (AgentRole.AI, AI_AGENT, reply.thinking, 7)
    ]
    assert log.thought is None  # acabou de pensar


def test_three_refusals_in_a_row_take_it_out_of_the_tuning() -> None:
    wrong = Reply("", answer("svm", {}))
    llm = ScriptedModel([wrong] * MAX_REFUSALS)

    outcome, log = search(llm, budget=15)

    assert len(llm.calls) == MAX_REFUSALS
    assert not any(candidate.ai for candidate in outcome.candidates)
    assert len(outcome.candidates) == 15  # as recusas não gastaram orçamento
    assert any("sais da afinação" in message.text for message in log.messages())


def test_a_language_model_that_stops_answering_leaves_the_search_going() -> None:
    outcome, log = search(ScriptedModel([]), budget=8)

    assert len(outcome.candidates) == 8
    assert outcome.winner is not None
    said = [m.text for m in log.messages() if m.role is AgentRole.AI]
    assert said
    assert said[0].startswith("Não consegui responder")


def test_the_thought_grows_while_it_thinks() -> None:
    log = SearchLog()

    log.start_thinking(AI_AGENT, 4)
    log.think("Primeiro ")
    log.think("")
    log.think("isto.")
    live = log.thought
    log.stop_thinking()

    assert live is not None
    assert (live.author, live.candidate, live.text) == (AI_AGENT, 4, "Primeiro isto.")
    assert log.thought is None
