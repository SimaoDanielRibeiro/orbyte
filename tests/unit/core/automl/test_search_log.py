"""A conversa da procura: números seguidos, só o que é novo, e o passo de cada candidato."""

from ml_lab.core.automl.search_log import (
    AgentRole,
    Candidate,
    SearchLog,
    SearchPhase,
    model_agent_name,
)
from ml_lab.core.models.base import ModelFamily


def candidate(number: int) -> Candidate:
    return Candidate(
        number=number,
        phase=SearchPhase.SCREENING,
        family=ModelFamily.TREE,
        model="decision_tree",
        model_display_name="Árvore de decisão",
        hyperparameters={},
        changes={},
        mean=0.8,
        std=0.01,
    )


def test_messages_are_numbered_and_carry_the_phase() -> None:
    log = SearchLog()

    log.say(AgentRole.DATA, "Agente de dados", "Remover PassengerId.")
    log.enter(SearchPhase.SCREENING)
    second = log.say(AgentRole.COORDINATOR, "Coordenador", "Ronda 1.")

    assert second.number == 2
    assert [m.phase for m in log.messages()] == [SearchPhase.DATA, SearchPhase.SCREENING]
    assert log.phase is SearchPhase.SCREENING


def test_whoever_follows_asks_only_for_what_is_new() -> None:
    log = SearchLog()
    for number in range(1, 4):
        log.say(AgentRole.COORDINATOR, "Coordenador", f"mensagem {number}")
        log.add(candidate(number))

    assert [m.number for m in log.messages(after=1)] == [2, 3]
    assert [c.number for c in log.candidates(after=2)] == [3]
    assert log.next_number == 4


def test_each_candidate_is_announced_as_it_arrives() -> None:
    seen: list[int] = []
    log = SearchLog(lambda found: seen.append(found.number))

    log.add(candidate(1))
    log.add(candidate(2))

    assert seen == [1, 2]


def test_a_model_agent_is_named_after_its_family() -> None:
    assert model_agent_name(ModelFamily.NEURAL) == "Agente · Redes neuronais"
