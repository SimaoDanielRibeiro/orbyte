"""O que um agente sabe, pronto a desenhar na grelha (ecrã Reinforcement learning).

Cada fotografia do treino de um agente (a tabela Q e o caminho que ele faria) vira uma
``PolicyFrame``:

- o valor de cada célula, o maior Q das suas ações: quanto se espera ganhar a partir dali;
- a ação que a política escolhe em cada célula (as setas);
- o caminho que o agente faria a partir da partida.

Com as fotografias por ordem, vê-se o valor a espalhar-se a partir do objetivo, episódio a
episódio. É assim que o Q-learning aprende: o +1 do objetivo passa para as células vizinhas,
e destas para as seguintes, até chegar à partida.

``reward_curve`` prepara a curva da recompensa por episódio, com a média móvel: cada
episódio, sozinho, é ruidoso, porque a exploração escolhe ao acaso.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

from ml_lab.core.rl.environments.base import CellKind, GridLayout
from ml_lab.core.training.recorder import StepRecord

MAX_WINDOW: Final = 50
"""A janela da média móvel, no máximo (em episódios)."""

_NO_POLICY: Final = frozenset({CellKind.WALL, CellKind.GOAL, CellKind.TRAP})


@dataclass(frozen=True, slots=True)
class PolicyFrame:
    """O que o agente sabia num episódio.

    Attributes:
        episode: O episódio da fotografia.
        values: Por célula (linha a linha), o maior Q; ``None`` nas paredes, no objetivo e
            nas armadilhas, onde o episódio acaba e não há nada a escolher.
        actions: Por célula, a ação que a política escolhe (o número da ação); ``None`` onde
            não há escolha.
        q_values: Por célula, o Q de cada ação (a tabela Q, linha a linha).
        path: As células por onde o agente passaria, a partir da partida, sem escorregar.
        reached: Se esse caminho chega ao objetivo.
    """

    episode: int
    values: tuple[float | None, ...]
    actions: tuple[int | None, ...]
    q_values: tuple[tuple[float, ...], ...]
    path: tuple[int, ...]
    reached: bool


@dataclass(frozen=True, slots=True)
class PolicyTrace:
    """O que o agente sabia ao longo do treino.

    Attributes:
        layout: A grelha.
        frames: Uma fotografia por snapshot, por ordem de episódio.
    """

    layout: GridLayout
    frames: tuple[PolicyFrame, ...]


@dataclass(frozen=True, slots=True)
class RewardCurve:
    """A recompensa de cada episódio do treino.

    Attributes:
        episodes: Os episódios.
        rewards: A recompensa de cada um.
        average: A média móvel da recompensa (nos últimos ``window`` episódios).
        success: A proporção de episódios que chegaram ao objetivo, na mesma janela.
        window: O tamanho da janela da média móvel.
    """

    episodes: tuple[int, ...]
    rewards: tuple[float, ...]
    average: tuple[float, ...]
    success: tuple[float, ...]
    window: int


def policy_frame(episode: int, state: Mapping[str, Any], layout: GridLayout) -> PolicyFrame:
    """A fotografia ``state`` de um agente (com ``q_table``, ``path`` e ``reached``) na grelha."""
    table = np.asarray(state["q_table"], dtype=np.float64)
    chooses = [kind not in _NO_POLICY for kind in layout.cells]
    return PolicyFrame(
        episode=episode,
        values=tuple(
            float(row.max()) if choice else None for row, choice in zip(table, chooses, strict=True)
        ),
        actions=tuple(
            int(row.argmax()) if choice else None
            for row, choice in zip(table, chooses, strict=True)
        ),
        q_values=tuple(tuple(float(value) for value in row) for row in table),
        path=tuple(int(cell) for cell in state.get("path", ())),
        reached=bool(state.get("reached", False)),
    )


def policy_trace(
    layout: GridLayout, snapshots: Sequence[tuple[int, Mapping[str, Any]]]
) -> PolicyTrace | None:
    """As fotografias do treino, por ordem de episódio (com o mesmo episódio, fica a última)."""
    by_episode = {step: state for step, state in snapshots if "q_table" in state}
    if not by_episode:
        return None
    frames = tuple(policy_frame(step, by_episode[step], layout) for step in sorted(by_episode))
    return PolicyTrace(layout, frames)


def reward_curve(history: Sequence[StepRecord]) -> RewardCurve:
    """A recompensa por episódio e as médias móveis (a janela cresce com o treino, até 50)."""
    rewards = np.array([record.metrics.get("reward", 0.0) for record in history])
    success = np.array([record.metrics.get("success", 0.0) for record in history])
    window = int(np.clip(len(history) // 10, 1, MAX_WINDOW))
    return RewardCurve(
        episodes=tuple(record.step for record in history),
        rewards=tuple(float(value) for value in rewards),
        average=_moving_average(rewards, window),
        success=_moving_average(success, window),
        window=window,
    )


def _moving_average(values: np.ndarray[Any, Any], window: int) -> tuple[float, ...]:
    """A média dos últimos ``window`` valores em cada ponto (no início, dos que houver)."""
    if values.size == 0:
        return ()
    totals = np.cumsum(np.insert(values.astype(np.float64), 0, 0.0))
    ends = np.arange(1, values.size + 1)
    starts = np.maximum(0, ends - window)
    return tuple(float(value) for value in (totals[ends] - totals[starts]) / (ends - starts))
