"""A grelha: paredes, bordas, recompensas e o escorregar."""

import numpy as np
import pytest

from ml_lab.core.foundation.errors import HyperparameterError
from ml_lab.core.rl.environments.base import CellKind
from ml_lab.core.rl.environments.gridworld import (
    GOAL_REWARD,
    STEP_REWARD,
    TRAP_REWARD,
    GridWorld,
    GridWorldParams,
)

UP, RIGHT, DOWN, LEFT = range(4)


def grid(layout: str = "simples", slip: float = 0.0) -> GridWorld:
    return GridWorld(GridWorldParams.parse({"layout": layout, "slip": slip}))


def cell(world: GridWorld, row: int, column: int) -> int:
    return row * world.layout().columns + column


def test_the_simple_map_is_read_row_by_row() -> None:
    world = grid()
    layout = world.layout()

    assert (layout.rows, layout.columns) == (5, 5)
    assert world.reset(np.random.default_rng(0)) == 0
    assert layout.kind(cell(world, 4, 4)) is CellKind.GOAL
    assert layout.kind(cell(world, 1, 1)) is CellKind.WALL
    assert world.title() == "Grid world · Simple"


def test_borders_and_walls_keep_the_agent_in_place_and_cost_a_step() -> None:
    world = grid()

    against_border = world.step(0, UP, None)
    against_wall = world.step(cell(world, 0, 1), DOWN, None)  # por baixo está uma parede

    assert (against_border.state, against_border.reward) == (0, STEP_REWARD)
    assert against_wall.state == cell(world, 0, 1)
    assert not against_border.done


def test_the_goal_and_the_traps_end_the_episode() -> None:
    simple, cliff = grid(), grid("penhasco")

    goal = simple.step(cell(simple, 3, 4), DOWN, None)
    trap = cliff.step(cliff.reset(np.random.default_rng(0)), RIGHT, None)

    assert (goal.reward, goal.done, goal.reached_goal) == (GOAL_REWARD, True, True)
    assert (trap.reward, trap.done, trap.reached_goal) == (TRAP_REWARD, True, False)


def test_slipping_goes_sideways_and_never_backwards() -> None:
    world = grid(slip=0.5)
    rng = np.random.default_rng(0)
    # Livre à direita, em cima e em baixo; atrás (à esquerda) há uma parede, por isso
    # ficar no mesmo sítio só podia ser ter andado para trás.
    centre = cell(world, 3, 2)

    landed = [world.step(centre, RIGHT, rng).state for _ in range(4000)]

    right, up, down = cell(world, 3, 3), cell(world, 2, 2), cell(world, 4, 2)
    counts = {target: landed.count(target) / len(landed) for target in set(landed)}
    assert counts[right] == pytest.approx(0.5, abs=0.03)
    assert counts[up] == pytest.approx(0.25, abs=0.03)
    assert counts[down] == pytest.approx(0.25, abs=0.03)
    assert centre not in counts  # para trás, nunca


def test_without_randomness_there_is_no_slipping() -> None:
    world = grid(slip=0.5)

    assert {world.step(cell(world, 3, 2), RIGHT, None).state for _ in range(50)} == {
        cell(world, 3, 3)
    }


def test_invalid_parameters_are_explained() -> None:
    with pytest.raises(HyperparameterError, match="Slip"):
        GridWorldParams.parse({"slip": 0.9})
    with pytest.raises(HyperparameterError, match="Map"):
        GridWorldParams.parse({"layout": "lua"})
