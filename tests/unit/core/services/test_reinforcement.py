"""O serviço de reinforcement learning: treinar, acompanhar, e o run no histórico."""

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from ml_lab.config import Settings, get_settings
from ml_lab.core.foundation.errors import (
    ComponentNotFoundError,
    HyperparameterError,
    InvalidParameterError,
    TrainingInProgressError,
)
from ml_lab.core.foundation.types import Partition, TaskType
from ml_lab.core.schemas.reinforcement import AgentSpec, EnvironmentSpec, ReinforcementConfig
from ml_lab.core.schemas.serialization import to_json
from ml_lab.core.services.lab import Lab
from ml_lab.core.tracking.store import InMemoryStore


def settings(folder: Path) -> Settings:
    """As definições do processo, com o registo e os artefactos numa pasta só deste teste."""
    return replace(
        get_settings(), artifacts_dir=folder / "artifacts", database_path=folder / "registo.db"
    )


def config(
    agent: str = "q_learning", layout: str = "simples", seed: int = 1, **hyperparameters: Any
) -> ReinforcementConfig:
    return ReinforcementConfig(
        environment=EnvironmentSpec(parameters={"layout": layout}),
        agent=AgentSpec(name=agent, hyperparameters={"episodes": 200, **hyperparameters}),
        seed=seed,
    )


@pytest.fixture
def lab() -> Lab:
    return Lab.create(store=InMemoryStore())


def test_the_cards_describe_environments_and_agents(lab: Lab) -> None:
    environments = lab.reinforcement.environments()
    agents = lab.reinforcement.agents()

    assert [card.name for card in environments] == ["gridworld"]
    assert "layout" in environments[0].parameters_schema["properties"]
    assert [card.name for card in agents] == ["q_learning", "dqn"]  # o didático primeiro
    assert agents[0].doc.startswith("# Q-learning")
    assert agents[1].defaults["target_update"] == 100


def test_a_trained_agent_crosses_the_grid_and_the_result_is_plain_json(lab: Lab) -> None:
    result = lab.reinforcement.run(config())

    assert result.primary is not None
    assert (result.primary.name, result.primary.value) == ("success_rate", 1.0)
    assert result.learned.reached
    assert len(result.learned.path) - 1 == 8
    assert result.steps == 200
    assert len(result.rewards.rewards) == 200
    assert result.trace is not None
    assert len(result.trace.frames) > 10
    assert result.reports[Partition.TRAIN].n_samples == 100
    json.dumps(to_json(result), allow_nan=False)


def test_exploring_costs_reward_on_the_cliff(lab: Lab) -> None:
    result = lab.reinforcement.run(config(layout="penhasco", episodes=500))

    train = result.reports[Partition.TRAIN].metric("success_rate").value
    test = result.reports[Partition.TEST].metric("success_rate").value
    assert train is not None
    assert test == 1.0
    assert train < test  # a explorar, cai nas armadilhas; a política aprendida não cai
    assert "exploration_cost" in [notice.code for notice in result.notices]


def test_problems_in_the_configuration_fail_before_training(lab: Lab) -> None:
    with pytest.raises(ComponentNotFoundError, match="Querias dizer 'q_learning'"):
        lab.reinforcement.run(config(agent="q_lerning"))
    with pytest.raises(InvalidParameterError, match="ambiente 'gridworld'"):
        lab.reinforcement.run(config(layout="lua"))
    with pytest.raises(HyperparameterError, match="Desconto"):
        lab.reinforcement.run(config(discount=2.0))


def test_a_background_training_is_followed_paused_and_stopped(lab: Lab) -> None:
    job = lab.reinforcement.start(config(layout="labirinto", episodes=5_000))
    run_id = job.run_id

    paused = lab.reinforcement.pause(run_id)
    with pytest.raises(TrainingInProgressError):
        lab.reinforcement.result(run_id)
    lab.reinforcement.cancel(run_id)
    result = job.wait(30)

    assert paused.status.value == "paused"
    assert result.stopped_early
    assert result.steps < 5_000
    assert result.notices[0].code == "stopped_early"
    update = lab.reinforcement.updates(run_id)
    assert update.state.status.value == "finished"
    assert [step.step for step in update.steps] == list(range(1, result.steps + 1))
    assert lab.reinforcement.updates(run_id, after_step=result.steps).steps == ()


def test_the_run_is_recorded_and_opens_as_reinforcement_learning(lab: Lab) -> None:
    result = lab.reinforcement.run(config())

    entry = lab.history.entry(result.run_id)

    assert entry.task_type is TaskType.CONTROL
    assert entry.name == "Q-learning em Grelha · simples"
    assert entry.primary_value == 1.0
    assert lab.history.open_reinforcement(result.run_id) is result  # desta sessão: em memória
    with pytest.raises(InvalidParameterError, match="reinforcement learning"):
        lab.history.open(result.run_id)


def test_a_saved_run_reopens_with_the_policy_after_closing_the_app(tmp_path: Path) -> None:
    before = Lab.create(settings(tmp_path))
    saved = before.reinforcement.run(config())
    only_recorded = before.reinforcement.run(config(seed=2))
    before.history.save(saved.run_id)

    after = Lab.create(settings(tmp_path))  # outro Lab, a mesma base de dados: reabrir a app
    reopened = after.history.open_reinforcement(saved.run_id)
    recorded = after.history.open_reinforcement(only_recorded.run_id)

    assert reopened == saved
    assert recorded.trace is None  # só registado: sem o que o agente sabia em cada snapshot
    assert recorded.learned == only_recorded.learned
    assert (tmp_path / "artifacts" / "runs" / saved.run_id / "trace.json").is_file()


def test_runs_of_agents_are_compared_and_reproduced(lab: Lab) -> None:
    near = lab.reinforcement.run(config(discount=0.5))
    far = lab.reinforcement.run(config(discount=0.95))

    comparison = lab.history.compare([near.run_id, far.run_id])
    again = lab.history.reproduce(far.run_id).wait(30)

    assert [d.label for d in comparison.differences] == ["Hiperparâmetro · Desconto (γ)"]
    assert again.run_id != far.run_id
    assert again.reports == far.reports
    assert again.learned == far.learned
