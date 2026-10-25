"""A política na grelha: o valor e a ação de cada célula, o caminho e a recompensa por episódio."""

import numpy as np
import pytest

from ml_lab.core.rl.environments.base import CellKind, GridLayout
from ml_lab.core.training.recorder import StepRecord
from ml_lab.core.visualization_data.policy import policy_frame, policy_trace, reward_curve

# Uma grelha 1 × 4: partida, livre, parede, objetivo.
LAYOUT = GridLayout(
    rows=1,
    columns=4,
    cells=(CellKind.START, CellKind.EMPTY, CellKind.WALL, CellKind.GOAL),
    action_labels=("esquerda", "direita"),
    action_moves=((0, -1), (0, 1)),
)


def state(right: float = 0.5) -> dict[str, object]:
    return {"q_table": [[0.1, right], [0.0, 0.9], [0.0, 0.0], [0.0, 0.0]], "path": [0, 1]}


def test_each_cell_has_its_value_and_its_action_where_there_is_a_choice() -> None:
    frame = policy_frame(7, state(), LAYOUT)

    assert frame.episode == 7
    assert frame.values == (0.5, 0.9, None, None)  # na parede e no objetivo não se escolhe
    assert frame.actions == (1, 1, None, None)
    assert frame.q_values[1] == (0.0, 0.9)
    assert (frame.path, frame.reached) == ((0, 1), False)


def test_the_trace_follows_the_episodes_and_keeps_the_last_of_a_repeated_one() -> None:
    trace = policy_trace(LAYOUT, [(5, state(0.2)), (1, state(0.1)), (5, state(0.3))])

    assert trace is not None
    assert [frame.episode for frame in trace.frames] == [1, 5]
    assert trace.frames[1].values[0] == pytest.approx(0.3)
    assert policy_trace(LAYOUT, [(1, {"probe": []})]) is None  # não é um agente


def test_the_reward_curve_has_moving_averages() -> None:
    history = [
        StepRecord(step, None, {"reward": float(step), "success": float(step % 2)})
        for step in range(1, 31)
    ]

    curve = reward_curve(history)

    assert curve.window == 3  # um décimo dos episódios
    assert curve.episodes[:3] == (1, 2, 3)
    assert curve.average[:4] == pytest.approx((1.0, 1.5, 2.0, 3.0))
    assert curve.success[3] == pytest.approx(np.mean([0.0, 1.0, 0.0]))
