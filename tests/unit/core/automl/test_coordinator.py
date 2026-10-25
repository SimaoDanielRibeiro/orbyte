"""O coordenador: as duas rondas, o orçamento, a escolha do vencedor e o teste que nunca vê.

Os testes usam o Iris com três folds e três famílias rápidas (lineares, vizinhos e árvores):
cada procura demora décimas de segundo.
"""

import threading
import time
from collections.abc import Sequence
from typing import Any

import pytest

from ml_lab.core.automl.coordinator import Coordinator, SearchOutcome
from ml_lab.core.automl.model_agent import ModelAgent, model_agents
from ml_lab.core.automl.objective import objective_for
from ml_lab.core.automl.search_log import (
    AgentRole,
    Candidate,
    SearchLog,
    SearchPhase,
    WorkInProgress,
)
from ml_lab.core.data.dataset import Dataset
from ml_lab.core.data.io.base import SourceSpec
from ml_lab.core.data.preparation import PreparationConfig
from ml_lab.core.data.splitting.base import SplitConfig, SplitStrategy
from ml_lab.core.foundation.types import TaskType
from ml_lab.core.models.base import LabModel, ModelFamily
from ml_lab.core.services.datasets import DatasetService
from ml_lab.core.training.control import TrainingControl

IRIS = SourceSpec("sample", {"name": "iris"})
SPECIES = PreparationConfig("species", split=SplitConfig(SplitStrategy.STRATIFIED))
FAST = (ModelFamily.LINEAR, ModelFamily.INSTANCE_BASED, ModelFamily.TREE)
MULTICLASS = TaskType.MULTICLASS_CLASSIFICATION


def folds(config: PreparationConfig = SPECIES) -> tuple[Dataset, ...]:
    return DatasetService().folds(IRIS, config, 7, 3)


def search(
    *,
    budget: int = 8,
    finalists: int = 2,
    families: Sequence[ModelFamily] = FAST,
    task_type: TaskType = MULTICLASS,
    objective: str | None = None,
    config: PreparationConfig = SPECIES,
    seed: int = 1,
    log: SearchLog | None = None,
    control: TrainingControl | None = None,
) -> tuple[SearchOutcome, SearchLog]:
    chosen = objective_for(task_type, objective)
    agents, _ = model_agents(task_type, chosen, families=families)
    log = log or SearchLog()
    coordinator = Coordinator(
        chosen,
        agents,
        folds(config),
        budget=budget,
        finalists=finalists,
        model_seed=0,
        search_seed=seed,
        log=log,
        control=control,
        baseline="logistic_regression",
    )
    return coordinator.search(), log


def test_the_screening_tries_each_model_once_with_its_defaults_baseline_first() -> None:
    outcome, _ = search(budget=3)

    assert [c.model for c in outcome.candidates] == ["logistic_regression", "knn", "decision_tree"]
    assert all(c.phase is SearchPhase.SCREENING and not c.changes for c in outcome.candidates)
    assert all(c.mean is not None and c.mean > 0.85 for c in outcome.candidates)


def test_the_budget_is_never_exceeded() -> None:
    outcome, _ = search(budget=7)

    assert len(outcome.candidates) == 7
    assert [c.number for c in outcome.candidates] == list(range(1, 8))


def test_only_the_finalists_are_tuned_taking_turns() -> None:
    outcome, _ = search(budget=9, finalists=2)

    screening = [c for c in outcome.candidates if c.phase is SearchPhase.SCREENING]
    tuning = [c for c in outcome.candidates if c.phase is SearchPhase.TUNING]
    best_two = sorted(screening, key=lambda c: (-(c.mean or 0), c.std or 0, c.number))[:2]
    assert {c.model for c in tuning} == {c.model for c in best_two}
    assert [c.model for c in tuning[:2]] == [c.model for c in best_two]  # à vez
    assert all(c.changes for c in tuning)  # cada proposta muda alguma coisa


def test_the_winner_has_the_best_mean_and_ties_go_to_the_steadiest() -> None:
    outcome, _ = search(budget=9)

    ranked = [next(c for c in outcome.candidates if c.number == n) for n in outcome.ranking]
    keys = [(-(c.mean or 0.0), c.std or 0.0, c.number) for c in ranked]
    assert keys == sorted(keys)
    assert outcome.winner is ranked[0]


def test_a_lower_is_better_objective_ranks_the_other_way() -> None:
    regression = PreparationConfig("sepal_length", task_type=TaskType.REGRESSION)

    outcome, _ = search(
        budget=6,
        families=(ModelFamily.LINEAR, ModelFamily.TREE),
        task_type=TaskType.REGRESSION,
        objective="rmse",
        config=regression,
    )

    means = [
        mean
        for n in outcome.ranking
        for c in outcome.candidates
        if c.number == n and (mean := c.mean) is not None
    ]
    assert len(means) == len(outcome.ranking)  # o quadro só tem candidatos medidos
    assert means == sorted(means)  # na RMSE, o melhor é o mais pequeno
    assert outcome.winner is not None
    assert outcome.winner.mean == min(means)


def test_the_same_seed_gives_the_same_search() -> None:
    first, _ = search(budget=9, seed=3)
    second, _ = search(budget=9, seed=3)

    def board(outcome: SearchOutcome) -> list[tuple[Any, ...]]:
        return [(c.model, c.hyperparameters, c.mean, c.std) for c in outcome.candidates]

    assert board(first) == board(second)
    assert first.ranking == second.ranking


def test_another_seed_draws_other_combinations() -> None:
    first, _ = search(budget=9, seed=3)
    second, _ = search(budget=9, seed=4)

    def tuned(outcome: SearchOutcome) -> list[dict[str, Any]]:
        return [c.changes for c in outcome.candidates if c.phase is SearchPhase.TUNING]

    assert tuned(first) != tuned(second)


def test_the_coordinator_only_ever_sees_the_training_rows() -> None:
    # O coordenador recebe os folds, e é tudo o que vê: nenhuma linha do teste lá está.
    whole = DatasetService().prepared(IRIS, SPECIES, 7).dataset

    seen = {int(row) for fold in folds() for split in (fold.train, fold.test) for row in split.rows}

    assert seen == {int(row) for row in whole.train.rows}
    assert not seen & {int(row) for row in whole.test.rows}


def test_a_failing_candidate_stays_on_the_board_with_its_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Uma proposta válida para o KNN, mas impossível aqui: 100 vizinhos, e cada fold só treina
    # com 80 linhas.
    def impossible(self: ModelAgent, model: type[LabModel[Any]], rng: Any) -> dict[str, Any]:
        return {**self.defaults(model), "n_neighbors": 100}

    monkeypatch.setattr(ModelAgent, "propose", impossible)

    outcome, log = search(budget=2, finalists=1, families=(ModelFamily.INSTANCE_BASED,))

    first, second = outcome.candidates
    assert not first.failed
    assert second.failed
    assert second.error is not None
    assert "100 vizinhos" in second.error
    assert outcome.winner is first  # a procura continuou e o vencedor é o que funcionou
    assert outcome.ranking == (1,)
    assert any("falhou" in message.text for message in log.messages())


def test_a_stop_request_ends_the_search_and_the_best_so_far_wins() -> None:
    control = TrainingControl()

    def stop_after_two(candidate: Candidate) -> None:
        if candidate.number == 2:
            control.stop()

    outcome, log = search(budget=8, log=SearchLog(stop_after_two), control=control)

    assert len(outcome.candidates) == 2
    assert outcome.stopped
    assert outcome.winner is not None
    assert outcome.winner.number in (1, 2)
    assert any("Pediste para parar" in message.text for message in log.messages())


def test_a_pause_holds_the_search_until_it_resumes() -> None:
    control = TrainingControl()
    log = SearchLog()
    control.pause()
    worker = threading.Thread(target=lambda: search(budget=3, log=log, control=control))
    worker.start()
    time.sleep(0.3)

    assert log.candidates() == ()  # em pausa antes do primeiro candidato
    control.resume()
    worker.join(timeout=30)
    assert len(log.candidates()) == 3


def test_the_conversation_tells_the_story_of_the_search() -> None:
    _, log = search(budget=6, finalists=2)

    messages = log.messages()
    texts = [message.text for message in messages]
    assert texts[0].startswith("Ronda 1, triagem: 3 modelos de 3 famílias")
    assert any(text.startswith("Passam à afinação:") for text in texts)
    assert texts[-1].startswith("O melhor na validação cruzada é ")
    agents = [m for m in messages if m.role is AgentRole.MODEL]
    assert {m.author for m in agents} == {
        "Agente · Modelos lineares",
        "Agente · Baseados em instâncias",
        "Agente · Árvores de decisão",
    }
    assert all(m.candidate is not None for m in agents)
    assert [m.number for m in messages] == list(range(1, len(messages) + 1))
    assert {m.phase for m in messages} == {SearchPhase.SCREENING, SearchPhase.TUNING}


class FoldWatcher(SearchLog):
    """Uma conversa que fotografa o trabalho em curso a cada fold (como o fluxograma o vê)."""

    def __init__(self) -> None:
        super().__init__()
        self.seen: list[WorkInProgress] = []

    def progress(self, fold: int) -> None:
        super().progress(fold)
        assert self.working is not None
        self.seen.append(self.working)


def test_whoever_follows_sees_which_agent_works_and_on_which_fold() -> None:
    log = FoldWatcher()

    outcome, _ = search(budget=2, log=log)

    assert [(work.candidate, work.fold) for work in log.seen] == [
        (1, 1),
        (1, 2),
        (1, 3),
        (2, 1),
        (2, 2),
        (2, 3),
    ]
    assert {work.folds for work in log.seen} == {3}
    first, second = outcome.candidates
    assert log.seen[0].model == first.model
    assert log.seen[0].author == "Agente · Modelos lineares"
    assert log.seen[3].model == second.model
    assert log.working is None  # entre candidatos (e no fim) não há trabalho em curso
