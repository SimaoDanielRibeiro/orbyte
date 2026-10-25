"""O serviço dos agentes: o pedido, a procura em segundo plano e o vencedor no histórico."""

import json

import pytest

from ml_lab.core.foundation.errors import (
    CandidateNotFoundError,
    ColumnNotFoundError,
    InvalidParameterError,
    SearchNotFoundError,
)
from ml_lab.core.schemas.serialization import to_json
from ml_lab.core.services import (
    AgentRole,
    AutoMLRequest,
    JobStatus,
    Lab,
    ModelFamily,
    SearchPhase,
    SearchSettings,
    SourceSpec,
    TaskType,
)
from ml_lab.core.tracking.store import InMemoryStore

IRIS = SourceSpec("sample", {"name": "iris"})
FAST = (ModelFamily.LINEAR, ModelFamily.INSTANCE_BASED, ModelFamily.TREE)
REQUEST = AutoMLRequest(
    dataset=IRIS,
    target="species",
    families=FAST,
    settings=SearchSettings(budget=6, folds=3, finalists=2),
    seed=3,
)


@pytest.fixture(scope="module")
def lab() -> Lab:
    return Lab.create(store=InMemoryStore())


def test_the_options_say_what_can_be_asked(lab: Lab) -> None:
    options = lab.automl.options(TaskType.BINARY_CLASSIFICATION)

    assert options.supported
    assert [o.name for o in options.objectives][:2] == ["accuracy", "balanced_accuracy"]
    assert len(options.agents) == 7
    assert options.settings.budget == 24
    assert set(options.settings_schema["properties"]) == {
        "budget",
        "folds",
        "finalists",
        "max_rows",
    }
    assert options.doc.startswith("# Agentes de AutoML")


def test_tasks_without_a_target_are_not_for_the_agents(lab: Lab) -> None:
    options = lab.automl.options(TaskType.CLUSTERING)

    assert not options.supported
    assert options.reason is not None
    assert "modo manual" in options.reason
    assert options.agents == ()


def test_the_plan_shows_the_decisions_before_anything_starts(lab: Lab) -> None:
    plan = lab.automl.plan(REQUEST)

    assert plan.task_type is TaskType.MULTICLASS_CLASSIFICATION
    assert plan.objective.name == "accuracy"
    assert plan.baseline == "logistic_regression"
    # O Iris tem uma linha repetida: o agente de dados tira-a, e sobram 149 (119 + 30).
    titles = [decision.title for decision in plan.decisions]
    assert titles[0] == "Remover linhas duplicadas"
    assert (plan.train_rows, plan.test_rows, plan.search_rows) == (119, 30, 119)
    assert not plan.sampled
    assert "Tarefa: classificação multiclasse" in titles
    assert [card.family for card in plan.agents] == list(FAST)


def test_a_missing_target_column_suggests_the_closest(lab: Lab) -> None:
    with pytest.raises(ColumnNotFoundError, match="species"):
        lab.automl.plan(REQUEST.model_copy(update={"target": "specie"}))


def test_a_large_training_set_is_searched_on_a_sample(lab: Lab) -> None:
    sampled = REQUEST.model_copy(update={"settings": SearchSettings(max_rows=100, folds=3)})

    plan = lab.automl.plan(sampled)

    assert (plan.train_rows, plan.search_rows) == (119, 100)
    assert plan.sampled


def test_an_unsupervised_task_is_refused_before_starting(lab: Lab) -> None:
    clustering = REQUEST.model_copy(update={"task_type": TaskType.CLUSTERING})

    with pytest.raises(InvalidParameterError, match="modo manual"):
        lab.automl.start(clustering)


def test_a_search_runs_in_the_background_and_the_winner_lands_in_the_history(lab: Lab) -> None:
    job = lab.automl.start(REQUEST)
    result = job.wait(timeout=120)

    state = job.state()
    assert state.status is JobStatus.FINISHED
    assert state.planned_steps == 6
    assert state.progress.steps == len(result.candidates) == 6  # cada candidato é um passo
    assert result.winner == result.ranking[0]
    assert result.run_id is not None
    entry = lab.history.entry(result.run_id)
    winner = result.winner_candidate
    assert winner is not None
    assert entry.model_name == winner.model
    assert entry.name == "Agentes · " + winner.model_display_name + " · Iris (species)"
    assert result.test is not None
    assert result.test.name == "accuracy"
    assert result.messages[0].role is AgentRole.DATA
    assert result.messages[-1].text.startswith("No teste: ")


def test_updates_send_only_what_is_new(lab: Lab) -> None:
    job = lab.automl.start(REQUEST)
    job.wait(timeout=120)

    everything = lab.automl.updates(job.run_id)
    later = lab.automl.updates(job.run_id, after_step=4, after_message=10)

    assert everything.phase is SearchPhase.DONE
    assert [c.number for c in everything.candidates] == [1, 2, 3, 4, 5, 6]
    assert [c.number for c in later.candidates] == [5, 6]
    assert later.messages[0].number == 11
    assert later.snapshot_step == everything.messages[-1].number  # o cursor do WebSocket
    assert [step.step for step in later.steps] == [5, 6]
    assert everything.leader is not None
    assert everything.objective.name == "accuracy"


def test_a_stopped_search_still_names_a_winner(lab: Lab) -> None:
    longer = REQUEST.model_copy(update={"settings": SearchSettings(budget=200, folds=3)})
    job = lab.automl.start(longer)
    lab.automl.pause(job.run_id)
    lab.automl.resume(job.run_id)

    lab.automl.cancel(job.run_id)
    result = job.wait(timeout=120)

    assert result.stopped_early
    assert len(result.candidates) < 200
    assert "stopped_early" in [notice.code for notice in result.notices]


def test_a_running_search_is_never_forgotten(lab: Lab) -> None:
    # O serviço esquece os planos mais antigos, mas nunca o de uma procura que decorre:
    # sem ele, quem a acompanha deixava de ver a conversa e os candidatos.
    longer = REQUEST.model_copy(update={"settings": SearchSettings(budget=200, folds=3)})
    tiny = REQUEST.model_copy(
        update={
            "families": (ModelFamily.TREE,),
            "settings": SearchSettings(budget=1, folds=2),
        }
    )
    running = lab.automl.start(longer)
    lab.automl.pause(running.run_id)

    for _ in range(12):  # mais do que os planos que o serviço guarda
        lab.automl.run(tiny)

    assert lab.automl.updates(running.run_id).state.status is JobStatus.PAUSED
    lab.automl.cancel(running.run_id)
    assert running.wait(timeout=120).stopped_early


def test_an_unknown_search_is_not_found(lab: Lab) -> None:
    with pytest.raises(SearchNotFoundError, match="search-nada"):
        lab.automl.job("search-nada")
    with pytest.raises(SearchNotFoundError):
        lab.automl.result("search-nada")


def test_a_candidate_can_be_trained_by_hand(lab: Lab) -> None:
    result = lab.automl.run(REQUEST)
    last = result.candidates[-1]

    experiment = lab.automl.train_candidate(result.search_id, last.number).wait(timeout=60)

    assert experiment.model_name == last.model
    assert experiment.hyperparameters == last.hyperparameters
    assert experiment.seed == result.seed
    with pytest.raises(CandidateNotFoundError, match="de 1 a 6"):
        lab.automl.train_candidate(result.search_id, 99)


def test_the_same_request_gives_the_same_search(lab: Lab) -> None:
    first = lab.automl.run(REQUEST)
    second = lab.automl.run(REQUEST)

    assert [(c.model, c.hyperparameters, c.mean) for c in first.candidates] == [
        (c.model, c.hyperparameters, c.mean) for c in second.candidates
    ]
    assert first.winner == second.winner


def test_the_result_becomes_strict_json(lab: Lab) -> None:
    result = lab.automl.run(REQUEST)

    text = json.dumps(to_json(result), allow_nan=False)

    assert '"search_id"' in text
    assert AutoMLRequest.model_validate(to_json(result)["request"]) == result.request
