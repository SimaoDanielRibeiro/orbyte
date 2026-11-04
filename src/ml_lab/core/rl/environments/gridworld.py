"""A grelha: um agente atravessa um mapa até ao objetivo, a evitar paredes e armadilhas.

É o ambiente clássico para aprender reinforcement learning, porque tudo se vê. Cada célula
é um estado; em cada uma, o agente pode subir, ir para a direita, descer ou ir para a
esquerda. As recompensas são:

- **+1** ao chegar ao objetivo (o episódio acaba);
- **−1** ao cair numa armadilha (o episódio também acaba);
- **−0,01** em cada passo, para o caminho mais curto valer mais do que um longo.

Contra uma parede, ou contra a borda, o agente fica onde está (e paga o passo na mesma).
Com ``slip`` acima de zero, o ambiente deixa de ser determinista: em cada passo, o agente
escorrega para um dos lados com essa probabilidade. No penhasco, isso muda o melhor
caminho: o mais curto passa colado às armadilhas, e quem escorrega cai.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final, Literal, override

import numpy as np
from pydantic import Field

from ml_lab.core.rl.environments.base import (
    ENVIRONMENTS,
    CellKind,
    Environment,
    EnvironmentInfo,
    GridLayout,
    Transition,
)
from ml_lab.core.schemas.hyperparameters import Hyperparameters

GOAL_REWARD: Final = 1.0
TRAP_REWARD: Final = -1.0
STEP_REWARD: Final = -0.01

ACTION_LABELS: Final = ("up", "right", "down", "left")
ACTION_MOVES: Final = ((-1, 0), (0, 1), (1, 0), (0, -1))

MAPS: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        # S: partida · G: objetivo · #: parede · X: armadilha · .: livre
        "simples": (
            "S....",
            ".###.",
            "...#.",
            ".#...",
            "...#G",
        ),
        "penhasco": (
            "............",
            "............",
            "............",
            "SXXXXXXXXXXG",
        ),
        "labirinto": (
            "S..#...",
            ".#.#.#.",
            ".#...#.",
            ".###.X.",
            "...#.#.",
            ".X.#.#.",
            ".#...#G",
        ),
    }
)
"""Os mapas, linha a linha."""

MAP_LABELS: Final[Mapping[str, str]] = MappingProxyType(
    {"simples": "Simple", "penhasco": "Cliff", "labirinto": "Maze"}
)
"""O nome de cada mapa no formulário (os identificadores ficam: os runs guardados usam-nos)."""

_KINDS: Final[Mapping[str, CellKind]] = MappingProxyType(
    {
        ".": CellKind.EMPTY,
        "#": CellKind.WALL,
        "S": CellKind.START,
        "G": CellKind.GOAL,
        "X": CellKind.TRAP,
    }
)


class GridWorldParams(Hyperparameters):
    """Os parâmetros da grelha."""

    layout: Literal["simples", "penhasco", "labirinto"] = Field(
        default="simples",
        title="Map",
        description=(
            "The simple one has walls; the cliff has a row of traps between the start and the "
            "goal; the maze has dead ends and traps."
        ),
        json_schema_extra={"x-labels": dict(MAP_LABELS)},
    )
    slip: float = Field(
        default=0.0,
        ge=0.0,
        le=0.5,
        title="Slip",
        description=(
            "The probability that the agent slips to one side instead of going where it wants. "
            "At 0, the environment is deterministic."
        ),
    )


@ENVIRONMENTS.register("gridworld")
class GridWorld(Environment[GridWorldParams]):
    """A grelha (ver o módulo)."""

    info = EnvironmentInfo(
        name="gridworld",
        display_name="Grid world",
        summary=(
            "An agent crosses a grid to the goal, avoiding the walls and the traps: each cell "
            "is a state, and there are four actions."
        ),
        parameters=GridWorldParams,
    )

    def __init__(self, parameters: GridWorldParams | None = None) -> None:
        super().__init__(parameters)
        rows = MAPS[self.parameters.layout]
        self._layout = GridLayout(
            rows=len(rows),
            columns=len(rows[0]),
            cells=tuple(_KINDS[symbol] for row in rows for symbol in row),
            action_labels=ACTION_LABELS,
            action_moves=ACTION_MOVES,
        )
        self._start = self._layout.cells.index(CellKind.START)

    @property
    @override
    def n_states(self) -> int:
        return len(self._layout.cells)

    @property
    @override
    def n_actions(self) -> int:
        return len(ACTION_MOVES)

    @property
    @override
    def max_steps(self) -> int:
        # Tempo de sobra para atravessar o mapa várias vezes, mesmo às voltas.
        return 4 * self.n_states

    @override
    def title(self) -> str:
        return f"{self.info.display_name} · {MAP_LABELS[self.parameters.layout]}"

    @override
    def reset(self, rng: np.random.Generator) -> int:
        return self._start

    @override
    def step(self, state: int, action: int, rng: np.random.Generator | None) -> Transition:
        moved = self._move(state, self._slipped(action, rng))
        kind = self._layout.kind(moved)
        if kind is CellKind.GOAL:
            return Transition(moved, GOAL_REWARD, done=True, reached_goal=True)
        if kind is CellKind.TRAP:
            return Transition(moved, TRAP_REWARD, done=True)
        return Transition(moved, STEP_REWARD)

    @override
    def layout(self) -> GridLayout:
        return self._layout

    def _slipped(self, action: int, rng: np.random.Generator | None) -> int:
        """A ação que acontece mesmo: a pedida, ou uma das do lado, se escorregar."""
        if rng is None or self.parameters.slip == 0 or rng.random() >= self.parameters.slip:
            return action
        side = 1 if rng.random() < 0.5 else -1
        return (action + side) % len(ACTION_MOVES)

    def _move(self, state: int, action: int) -> int:
        """Onde se vai parar: contra uma parede ou a borda, fica-se no mesmo sítio."""
        row, column = self._layout.position(state)
        d_row, d_column = ACTION_MOVES[action]
        row, column = row + d_row, column + d_column
        if not (0 <= row < self._layout.rows and 0 <= column < self._layout.columns):
            return state
        target = row * self._layout.columns + column
        return state if self._layout.kind(target) is CellKind.WALL else target
