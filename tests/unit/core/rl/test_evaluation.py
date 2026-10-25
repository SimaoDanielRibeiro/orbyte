"""As métricas de um agente: o treino (a explorar) e o teste (a política aprendida)."""

import pytest

from ml_lab.core.foundation.types import Partition, TaskType
from ml_lab.core.rl.agents.q_learning import QLearningAgent, QLearningParams
from ml_lab.core.rl.environments.gridworld import GridWorld
from ml_lab.core.rl.evaluation import (
    EpisodeOutcome,
    episodes_report,
    play,
    reading,
    training_outcomes,
)
from ml_lab.core.training.recorder import StepRecord


def report(*outcomes: EpisodeOutcome, partition: Partition = Partition.TEST):  # type: ignore[no-untyped-def]
    return episodes_report(outcomes, partition)


def test_the_report_has_the_success_the_reward_and_the_length() -> None:
    result = report(EpisodeOutcome(0.9, 10, True), EpisodeOutcome(-1.05, 6, False))

    assert result.task_type is TaskType.CONTROL
    assert result.n_samples == 2
    assert result.values() == pytest.approx(
        {"success_rate": 0.5, "mean_reward": -0.075, "mean_length": 8.0}
    )
    assert result.metric("success_rate").as_percent


def test_the_learned_policy_plays_without_exploring() -> None:
    world = GridWorld()
    trained = QLearningAgent(QLearningParams(episodes=200), seed=0).train(world)

    outcomes = play(trained, world, episodes=5, seed=1)

    assert all(outcome.reached and outcome.length == 8 for outcome in outcomes)


def test_the_training_report_uses_the_last_episodes() -> None:
    history = [
        StepRecord(step, None, {"reward": 0.5, "length": 9.0, "success": float(step > 150)})
        for step in range(1, 201)
    ]

    outcomes = training_outcomes(history)

    assert len(outcomes) == 100
    assert sum(outcome.reached for outcome in outcomes) == 50


def test_the_reading_explains_the_cost_of_exploring_and_an_unsolved_grid() -> None:
    train = report(*[EpisodeOutcome(0.5, 10, n < 7) for n in range(10)], partition=Partition.TRAIN)
    solved = report(EpisodeOutcome(0.9, 10, True))
    unsolved = report(EpisodeOutcome(-0.5, 50, False), EpisodeOutcome(0.9, 10, True))

    assert [n.code for n in reading(train, solved, slip=0.0)] == ["exploration_cost"]
    assert "not_solved" in [n.code for n in reading(train, unsolved, slip=0.0)]
    assert "not_solved" not in [n.code for n in reading(train, unsolved, slip=0.2)]
