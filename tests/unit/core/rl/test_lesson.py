"""A lição do ecrã: um Q-learning no penhasco, passo a passo, sempre igual."""

from itertools import pairwise

import pytest

from ml_lab.core.rl.environments.base import CellKind
from ml_lab.core.rl.environments.gridworld import GOAL_REWARD, STEP_REWARD, TRAP_REWARD
from ml_lab.core.rl.lesson import LESSON_EPISODES, ReinforcementLesson, lesson


@pytest.fixture(scope="module")
def taught() -> ReinforcementLesson:
    return lesson()


def test_the_lesson_is_always_the_same(taught: ReinforcementLesson) -> None:
    assert lesson() == taught


def test_it_keeps_the_first_and_the_last_episode_and_the_first_success(
    taught: ReinforcementLesson,
) -> None:
    numbers = [episode.episode for episode in taught.kept]
    first_success = next(
        number for number, reward in enumerate(taught.rewards, start=1) if reward > 0
    )

    assert numbers == sorted(set(numbers))
    assert (numbers[0], numbers[-1]) == (1, LESSON_EPISODES)
    assert first_success in numbers
    assert not any(episode.reached for episode in taught.kept if episode.episode < first_success)
    assert len(taught.rewards) == taught.episodes == LESSON_EPISODES


def test_each_step_starts_where_the_last_one_ended_and_pays_what_the_cell_pays(
    taught: ReinforcementLesson,
) -> None:
    layout = taught.layout
    rewards = {CellKind.GOAL: GOAL_REWARD, CellKind.TRAP: TRAP_REWARD}
    for episode in taught.kept:
        steps = episode.steps
        assert layout.kind(steps[0].state) is CellKind.START
        for step, following in pairwise(steps):
            assert following.state == step.next_state
        for step in steps:
            assert step.reward == rewards.get(layout.kind(step.next_state), STEP_REWARD)
        assert episode.reward == pytest.approx(sum(step.reward for step in steps))
        assert (layout.kind(steps[-1].next_state) is CellKind.GOAL) == episode.reached


def test_the_first_episode_explores_every_step_and_learns_from_its_fall(
    taught: ReinforcementLesson,
) -> None:
    first = taught.kept[0]
    fall = first.steps[-1]

    assert first.epsilon == pytest.approx(1.0)
    assert all(step.explored for step in first.steps)
    assert all(value in (0.0, None) for value in first.before)  # ainda não sabe nada
    # Do zero a −1: a surpresa é toda a queda, e o Q anda metade dela (α = 0,5).
    assert fall.surprise == pytest.approx(TRAP_REWARD)
    assert fall.value <= 0.0


def test_the_last_episode_walks_the_short_way_along_the_cliff(
    taught: ReinforcementLesson,
) -> None:
    last = taught.kept[-1]

    assert last.reached
    assert len(last.steps) == 13
    assert not any(step.explored for step in last.steps)


def test_cells_without_a_choice_have_no_value(taught: ReinforcementLesson) -> None:
    layout = taught.layout
    for episode in taught.kept:
        for kind, value in zip(layout.cells, episode.before, strict=True):
            assert (value is None) == (kind in (CellKind.WALL, CellKind.GOAL, CellKind.TRAP))


def test_every_episode_says_what_happened(taught: ReinforcementLesson) -> None:
    first_success = next(episode for episode in taught.kept if episode.reached)

    assert all(episode.story for episode in taught.kept)
    assert "armadilha" in taught.kept[0].story
    assert "pela primeira vez" in first_success.story
