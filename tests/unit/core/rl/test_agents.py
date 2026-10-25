"""Os agentes: o ε-greedy, a regra do Q-learning, o caminho aprendido e as peças do DQN."""

import numpy as np
import pytest

from ml_lab.core.rl.agents.base import greedy_path
from ml_lab.core.rl.agents.dqn import DQNAgent, DQNParams, _ReplayMemory
from ml_lab.core.rl.agents.q_learning import QLearningAgent, QLearningParams
from ml_lab.core.rl.environments.base import Transition
from ml_lab.core.rl.environments.gridworld import GridWorld, GridWorldParams


def grid(layout: str = "simples") -> GridWorld:
    return GridWorld(GridWorldParams.parse({"layout": layout}))


def test_exploration_goes_down_in_a_straight_line_and_then_stays() -> None:
    agent = QLearningAgent(
        QLearningParams(episodes=100, epsilon_start=1.0, epsilon_end=0.1, exploration_fraction=0.5)
    )

    assert agent.epsilon(1) == pytest.approx(1.0)
    assert agent.epsilon(26) == pytest.approx(0.55)
    assert agent.epsilon(51) == pytest.approx(0.1)
    assert agent.epsilon(100) == pytest.approx(0.1)


def test_q_learning_moves_q_a_fraction_of_the_surprise() -> None:
    agent = QLearningAgent(QLearningParams(learning_rate=0.5, discount=0.9))
    agent._start(grid())  # o teste mexe-lhe por dentro, para ver uma só correção
    agent._table[7] = [1.0, 2.0, 0.0, 0.0]  # o melhor Q do estado seguinte é 2

    error = agent._learn(6, 1, Transition(state=7, reward=-0.01))

    target = -0.01 + 0.9 * 2.0
    assert agent._table[6, 1] == pytest.approx(0.5 * target)
    assert error == pytest.approx(target)


def test_an_end_of_episode_has_no_future_but_running_out_of_steps_does() -> None:
    agent = QLearningAgent(QLearningParams(learning_rate=1.0, discount=0.9))
    agent._start(grid())
    agent._table[7] = [5.0, 5.0, 5.0, 5.0]

    agent._learn(6, 0, Transition(state=7, reward=1.0, done=True, reached_goal=True))
    agent._learn(6, 1, Transition(state=7, reward=0.0))  # só se parou de jogar: há futuro

    assert agent._table[6, 0] == pytest.approx(1.0)
    assert agent._table[6, 1] == pytest.approx(0.9 * 5.0)


def test_q_learning_finds_the_short_way_around_the_cliff() -> None:
    world = grid("penhasco")

    trained = QLearningAgent(QLearningParams(episodes=500), seed=0).train(world)

    path, reached = greedy_path(trained.q_values(), world)
    assert reached
    assert len(path) - 1 <= 15  # o mais curto tem 13 passos, colado às armadilhas


def test_the_greedy_path_stops_when_the_policy_goes_round_in_circles() -> None:
    world = grid()
    table = np.zeros((world.n_states, world.n_actions))
    table[:, 0] = 1.0  # subir sempre: na primeira linha, bate na borda

    path, reached = greedy_path(table, world)

    assert path == [0, 0]
    assert not reached


def test_the_replay_memory_keeps_the_most_recent_steps() -> None:
    memory = _ReplayMemory(size=3, n_states=10)

    for state in range(5):
        memory.add(state, 0, Transition(state=state + 1, reward=float(state)))
    states, _, rewards, next_states, done = memory.sample(50, np.random.default_rng(0))

    assert len(memory) == 3
    assert set(states.tolist()) == {2, 3, 4}  # os dois primeiros saíram
    assert np.array_equal(next_states.numpy(), states.numpy() + 1)
    assert np.array_equal(rewards.numpy(), states.numpy().astype(np.float32))
    assert not done.any()


def test_the_dqn_network_gives_one_q_per_action_and_counts_its_weights() -> None:
    world = grid()

    trained = DQNAgent(DQNParams(episodes=20, hidden_units=8), seed=0).train(world)

    assert trained.q_values().shape == (world.n_states, world.n_actions)
    state = trained.state()
    assert state["n_parameters"] == (25 * 8 + 8) + (8 * 8 + 8) + (8 * 4 + 4)
    assert state["updates"] > 0
