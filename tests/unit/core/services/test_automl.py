"""O serviço dos agentes: o pedido, a procura em segundo plano, o vencedor no histórico e o
analista que responde a perguntas sobre a procura acabada."""

import json
import socket
from pathlib import Path

import pytest

from ml_lab.config import Settings
from ml_lab.core.foundation.errors import (
    CandidateNotFoundError,
    ColumnNotFoundError,
    InvalidParameterError,
    LanguageModelError,
    QuestionNotFoundError,
    SearchNotFoundError,
    TrainingInProgressError,
)
from ml_lab.core.schemas.automl import RecordedReply, RecordedToolCall
from ml_lab.core.schemas.serialization import to_json
from ml_lab.core.services import (
    AgentRole,
    AnalystRequest,
    AnalystStepKind,
    AutoMLRequest,
    AutoMLService,
    JobStatus,
    Lab,
    LanguageModelService,
    MessageKind,
    ModelFamily,
    Participant,
    ParticipantKind,
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
    # Das três famílias, um modelo rápido de cada (as implementações didáticas também entram).
    models=("logistic_regression", "knn", "decision_tree"),
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
    assert options.doc.startswith("# AutoML agents")


def test_tasks_without_a_target_are_not_for_the_agents(lab: Lab) -> None:
    options = lab.automl.options(TaskType.CLUSTERING)

    assert not options.supported
    assert options.reason is not None
    assert "manual mode" in options.reason
    assert options.agents == ()


def test_the_plan_shows_the_decisions_before_anything_starts(lab: Lab) -> None:
    plan = lab.automl.plan(REQUEST)

    assert plan.task_type is TaskType.MULTICLASS_CLASSIFICATION
    assert plan.objective.name == "accuracy"
    assert plan.baseline == "logistic_regression"
    # O Iris tem uma linha repetida: o agente de dados tira-a, e sobram 149 (119 + 30).
    titles = [decision.title for decision in plan.decisions]
    assert titles[0] == "Drop duplicate rows"
    assert (plan.train_rows, plan.test_rows, plan.search_rows) == (119, 30, 119)
    assert not plan.sampled
    assert "Task: multiclass classification" in titles
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

    with pytest.raises(InvalidParameterError, match="manual mode"):
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
    assert entry.name == "Agents · " + winner.model_display_name + " · Iris (species)"
    assert result.test is not None
    assert result.test.name == "accuracy"
    assert result.messages[0].role is AgentRole.DATA
    assert result.messages[-1].text.startswith("On the test: ")
    # A troca: o plano vai ao coordenador; no fim, o vencedor vai ao teste, e a leitura do
    # teste é para quem acompanha (sem destinatários).
    first, training, reading = result.messages[0], result.messages[-2], result.messages[-1]
    assert first.kind is MessageKind.REPORT
    assert first.recipients == (Participant(ParticipantKind.COORDINATOR),)
    assert training.kind is MessageKind.TEST
    assert training.recipients == (Participant(ParticipantKind.TEST),)
    assert training.candidate == reading.candidate == result.winner
    assert reading.kind is MessageKind.TEST
    assert reading.recipients == ()


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


def test_a_stopped_search_stops_everything(lab: Lab) -> None:
    longer = REQUEST.model_copy(update={"settings": SearchSettings(budget=200, folds=3)})
    job = lab.automl.start(longer)
    lab.automl.pause(job.run_id)
    lab.automl.resume(job.run_id)

    lab.automl.cancel(job.run_id)
    result = job.wait(timeout=120)

    assert result.stopped_early
    assert len(result.candidates) < 200
    assert "stopped_early" in [notice.code for notice in result.notices]
    # Parar é parar tudo (ADR 0051): o melhor até aí não treina nem vai ao teste.
    assert result.run_id is None
    assert result.test is None
    assert not any(message.kind is MessageKind.TEST for message in result.messages)


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
    with pytest.raises(CandidateNotFoundError, match="1 to 6"):
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


# --- o agente de IA (fase 13) --------------------------------------------------------

PROPOSAL = json.dumps(
    {"model": "decision_tree", "values": {"max_depth": 2}, "reason": "Rasa decora menos."}
)
WITH_AI = REQUEST.model_copy(
    update={
        "ai_agent": True,
        "settings": SearchSettings(budget=7, folds=3, finalists=3),
        "ai_replies": (RecordedReply(thinking="Vou baixar a profundidade.", content=PROPOSAL),),
    }
)


def test_with_saved_replies_the_ai_agent_takes_part_without_the_ollama(lab: Lab) -> None:
    result = lab.automl.run(WITH_AI)

    proposed = [candidate for candidate in result.candidates if candidate.ai]
    assert [(c.model, c.hyperparameters["max_depth"]) for c in proposed] == [("decision_tree", 2)]
    assert result.plan.ai_agent
    assert "ai_comparison" in {notice.code for notice in result.notices}
    assert any(m.kind is MessageKind.THOUGHT for m in result.messages)
    assert result.request.ai_replies == WITH_AI.ai_replies
    json.dumps(to_json(result), allow_nan=False)


def test_the_result_request_repeats_the_ai_search_as_it_was(lab: Lab) -> None:
    first = lab.automl.run(WITH_AI)

    again = lab.automl.run(first.request)

    assert [(c.model, c.hyperparameters, c.mean, c.ai) for c in first.candidates] == [
        (c.model, c.hyperparameters, c.mean, c.ai) for c in again.candidates
    ]


def test_without_the_ollama_the_ai_agent_is_refused_saying_how_to_start_it(
    lab: Lab, tmp_path: Path
) -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        closed = f"http://127.0.0.1:{probe.getsockname()[1]}"
    settings = Settings.from_env({"ML_LAB_OLLAMA_URL": closed}, home=tmp_path)
    service = AutoMLService(lab.datasets, lab.experiments, llm=LanguageModelService(settings))
    alone = AutoMLService(lab.datasets, lab.experiments)
    request = WITH_AI.model_copy(update={"ai_replies": ()})

    with pytest.raises(LanguageModelError, match="ollama serve"):
        service.start(request)
    with pytest.raises(LanguageModelError, match="has no language model connected"):
        alone.start(request)
    assert service.plan(request).ai_agent  # o plano mostra-se na mesma, antes de pedir


# --- o analista (fase 13, etapa 13.3) ------------------------------------------------


def saved_replies(winner: int, second: int) -> tuple[RecordedReply, ...]:
    """As respostas de um modelo: o quadro, o vencedor contra o segundo e a resposta."""
    return (
        RecordedReply(
            thinking="The board first.", tool_calls=(RecordedToolCall(name="list_candidates"),)
        ),
        RecordedReply(
            thinking="Now the winner against the second.",
            tool_calls=(
                RecordedToolCall(
                    name="compare_candidates", arguments={"first": winner, "second": second}
                ),
            ),
        ),
        RecordedReply(thinking="I know enough.", content=f"#{winner} is first on the board."),
    )


def test_the_analyst_answers_about_a_finished_search_with_its_tools(lab: Lab) -> None:
    result = lab.automl.run(REQUEST)
    winner, second = result.ranking[:2]
    request = AnalystRequest(
        question="Why did the winner win?", replies=saved_replies(winner, second)
    )

    job = lab.automl.ask(result.search_id, request)
    answer = job.wait(timeout=60)

    assert job.state().status is JobStatus.FINISHED
    assert answer.question_id.startswith("question-")
    assert answer.search_id == result.search_id
    assert [step.kind for step in answer.trace] == [
        AnalystStepKind.THOUGHT,
        AnalystStepKind.TOOL,
        AnalystStepKind.THOUGHT,
        AnalystStepKind.TOOL,
        AnalystStepKind.THOUGHT,
        AnalystStepKind.ANSWER,
    ]
    tools = [step.tool for step in answer.trace if step.tool is not None]
    assert [tool.label for tool in tools] == ["The candidate board", "Compare two candidates"]
    assert json.loads(tools[0].result)["winner"]["number"] == result.winner
    assert answer.answer == f"#{winner} is first on the board."
    assert (answer.tools_used, answer.model) == (2, "saved answers")
    assert [notice.code for notice in answer.notices] == ["numbers_checked"]
    # Cada passo do analista é um passo do trabalho, e o pedido traz as respostas que usou.
    assert job.state().progress.steps == len(answer.trace)
    assert answer.request == request
    assert lab.automl.questions(result.search_id) == (answer,)
    json.dumps(to_json(answer), allow_nan=False)


def test_the_answer_s_request_repeats_it_without_the_ollama(lab: Lab) -> None:
    result = lab.automl.run(REQUEST)
    first = lab.automl.ask(
        result.search_id,
        AnalystRequest(
            question="Why did the winner win?", replies=saved_replies(*result.ranking[:2])
        ),
    ).wait(timeout=60)

    again = lab.automl.ask(result.search_id, first.request).wait(timeout=60)

    assert again.question_id != first.question_id
    assert [(s.kind, s.text, s.tool) for s in again.trace] == [
        (s.kind, s.text, s.tool) for s in first.trace
    ]


def test_updates_bring_the_new_steps_of_a_question(lab: Lab) -> None:
    result = lab.automl.run(REQUEST)
    job = lab.automl.ask(
        result.search_id,
        AnalystRequest(question="Why?", replies=saved_replies(*result.ranking[:2])),
    )
    job.wait(timeout=60)

    everything = lab.automl.question_updates(job.run_id)
    later = lab.automl.question_updates(job.run_id, after_step=4)

    assert everything.question == "Why?"
    assert everything.search_id == result.search_id
    assert [step.number for step in everything.trace] == [1, 2, 3, 4, 5, 6]
    assert [step.number for step in later.trace] == [5, 6]
    assert [record.step for record in later.steps] == [5, 6]  # o cursor do WebSocket
    assert later.live is None  # acabou: não está a pensar
    assert lab.automl.question_job(job.run_id).status is JobStatus.FINISHED
    assert lab.automl.question_result(job.run_id).trace == everything.trace
    assert lab.automl.cancel_question(job.run_id).status is JobStatus.FINISHED  # já acabou


def test_the_analyst_only_reads_finished_searches(lab: Lab) -> None:
    longer = REQUEST.model_copy(update={"settings": SearchSettings(budget=200, folds=3)})
    running = lab.automl.start(longer)
    lab.automl.pause(running.run_id)

    with pytest.raises(TrainingInProgressError):
        lab.automl.ask(running.run_id, AnalystRequest(question="Why?"))
    with pytest.raises(TrainingInProgressError):
        lab.automl.analyst_options(running.run_id)
    lab.automl.cancel(running.run_id)
    running.wait(timeout=120)
    with pytest.raises(SearchNotFoundError):
        lab.automl.ask("search-nada", AnalystRequest(question="Why?"))
    with pytest.raises(SearchNotFoundError):
        lab.automl.questions("search-nada")


def test_an_unknown_question_is_not_found(lab: Lab) -> None:
    for method in (
        lab.automl.question_job,
        lab.automl.question_updates,
        lab.automl.question_result,
        lab.automl.cancel_question,
    ):
        with pytest.raises(QuestionNotFoundError, match="question-nada"):
            method("question-nada")


def test_the_suggestions_depend_on_what_the_search_has(lab: Lab) -> None:
    plain = lab.automl.run(REQUEST)
    with_ai = lab.automl.run(WITH_AI)

    options = lab.automl.analyst_options(plain.search_id)
    ai_options = lab.automl.analyst_options(with_ai.search_id)

    assert options.suggestions[0] == "Why did the winner win?"
    assert "Did the AI agent beat the random search?" not in options.suggestions
    assert "Did the AI agent beat the random search?" in ai_options.suggestions
    assert [tool.name for tool in options.tools] == [
        "list_candidates",
        "get_candidate",
        "compare_candidates",
        "data_plan",
    ]
    assert options.max_turns > 1
    json.dumps(to_json(options), allow_nan=False)


def test_without_the_ollama_the_analyst_is_refused_saying_how_to_start_it(
    lab: Lab, tmp_path: Path
) -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        closed = f"http://127.0.0.1:{probe.getsockname()[1]}"
    settings = Settings.from_env({"ML_LAB_OLLAMA_URL": closed}, home=tmp_path)
    service = AutoMLService(lab.datasets, lab.experiments, llm=LanguageModelService(settings))
    alone = AutoMLService(lab.datasets, lab.experiments)
    searched = service.run(REQUEST)
    alone_searched = alone.run(REQUEST)

    with pytest.raises(LanguageModelError, match="ollama serve"):
        service.ask(searched.search_id, AnalystRequest(question="Why?"))
    with pytest.raises(LanguageModelError, match="has no language model connected"):
        alone.ask(alone_searched.search_id, AnalystRequest(question="Why?"))
    # Com as respostas guardadas, responde na mesma: não precisa do Ollama.
    replies = saved_replies(*searched.ranking[:2])
    answer = service.ask(searched.search_id, AnalystRequest(question="Why?", replies=replies))
    assert answer.wait(timeout=60).answer
    assert alone.analyst_options(alone_searched.search_id).model is None
