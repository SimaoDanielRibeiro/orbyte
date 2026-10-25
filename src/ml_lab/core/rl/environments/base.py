"""Contrato dos ambientes de reinforcement learning: o mundo onde um agente aprende a agir.

Num ambiente há estados, ações e recompensas. Em cada passo, o agente vê o estado em que
está e escolhe uma ação; o ambiente responde com o estado seguinte, a recompensa e se o
episódio acabou. Um episódio vai do início (``reset``) até ao fim: o objetivo, uma armadilha,
ou o limite de passos (``max_steps``).

Os ambientes do ML Lab são pequenos e discretos: os estados e as ações contam-se
(``n_states`` e ``n_actions``), o que deixa guardar numa tabela o que o agente aprendeu e
desenhar tudo. Cada ambiente diz como se desenha (``layout``).

Um ambiente não guarda o estado do episódio: ``step`` recebe o estado atual e devolve o
seguinte. Quem guarda onde o agente está é o ciclo de treino. Assim, o mesmo ambiente serve
o treino, a avaliação e o desenho do caminho aprendido, sem nada escondido entre eles.

Para acrescentar um ambiente: um ficheiro nesta pasta com uma subclasse de ``Environment``
decorada com ``@ENVIRONMENTS.register("nome")``.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any, ClassVar, Final, cast

import numpy as np

from ml_lab.core.foundation.errors import InvalidParameterError
from ml_lab.core.foundation.registry import Registry
from ml_lab.core.schemas.hyperparameters import Hyperparameters


@dataclass(frozen=True, slots=True)
class Transition:
    """O que o ambiente responde a uma ação.

    Attributes:
        state: O estado seguinte.
        reward: A recompensa desta ação.
        done: Se o episódio acabou: chegou ao objetivo ou caiu numa armadilha. (Esgotar os
            passos não conta: o estado não é final, só se parou de jogar.)
        reached_goal: Se acabou no objetivo.
    """

    state: int
    reward: float
    done: bool = False
    reached_goal: bool = False


class CellKind(StrEnum):
    """O que há numa célula da grelha."""

    EMPTY = "empty"
    WALL = "wall"
    START = "start"
    GOAL = "goal"
    TRAP = "trap"

    @property
    def label(self) -> str:
        """Nome a mostrar na interface, em português."""
        return _CELL_LABELS[self]


_CELL_LABELS: Final[Mapping[CellKind, str]] = MappingProxyType(
    {
        CellKind.EMPTY: "Livre",
        CellKind.WALL: "Parede",
        CellKind.START: "Partida",
        CellKind.GOAL: "Objetivo",
        CellKind.TRAP: "Armadilha",
    }
)


@dataclass(frozen=True, slots=True)
class GridLayout:
    """Como se desenha um ambiente em grelha.

    O estado de cada célula é o seu número, linha a linha: ``linha × colunas + coluna``.

    Attributes:
        rows: Quantas linhas tem a grelha.
        columns: Quantas colunas.
        cells: O que há em cada célula, linha a linha.
        action_labels: O nome de cada ação, pela ordem dos números das ações.
        action_moves: Para onde cada ação leva, em (linhas, colunas): ``(-1, 0)`` é subir.
    """

    rows: int
    columns: int
    cells: tuple[CellKind, ...]
    action_labels: tuple[str, ...]
    action_moves: tuple[tuple[int, int], ...]

    def kind(self, state: int) -> CellKind:
        """O que há na célula do estado ``state``."""
        return self.cells[state]

    def position(self, state: int) -> tuple[int, int]:
        """A linha e a coluna do estado ``state``."""
        return divmod(state, self.columns)


@dataclass(frozen=True, slots=True)
class EnvironmentInfo:
    """Ficha de um ambiente.

    Attributes:
        name: Identificador, igual ao nome no registo ``ENVIRONMENTS``.
        display_name: Nome a mostrar.
        summary: Uma frase sobre o ambiente.
        parameters: A classe dos parâmetros (o mapa, a dificuldade…).
    """

    name: str
    display_name: str
    summary: str
    parameters: type[Hyperparameters]


class Environment[P: Hyperparameters](ABC):
    """Um ambiente discreto e pequeno, que se desenha numa grelha.

    Args:
        parameters: Os parâmetros, já validados (por omissão, os valores por omissão).
    """

    info: ClassVar[EnvironmentInfo]

    def __init__(self, parameters: P | None = None) -> None:
        expected = self.info.parameters
        chosen = parameters if parameters is not None else expected()
        if not isinstance(chosen, expected):
            raise InvalidParameterError(
                f"'{self.info.name}' recebe parâmetros do tipo {expected.__name__}, "
                f"não {type(chosen).__name__}."
            )
        self.parameters = cast("P", chosen)

    @property
    @abstractmethod
    def n_states(self) -> int:
        """Quantos estados há (incluindo os que não se alcançam, como as paredes)."""

    @property
    @abstractmethod
    def n_actions(self) -> int:
        """Quantas ações há em cada estado."""

    @property
    @abstractmethod
    def max_steps(self) -> int:
        """Quantos passos tem um episódio, no máximo (depois disso, pára-se de jogar)."""

    @abstractmethod
    def reset(self, rng: np.random.Generator) -> int:
        """O estado onde começa um episódio."""

    @abstractmethod
    def step(self, state: int, action: int, rng: np.random.Generator | None) -> Transition:
        """O que acontece ao fazer ``action`` no estado ``state``.

        Args:
            state: Onde o agente está.
            action: O que escolheu fazer.
            rng: O acaso do ambiente (escorregar, por exemplo); ``None`` dá sempre o que o
                agente pediu, para desenhar o caminho que ele quer seguir.
        """

    @abstractmethod
    def layout(self) -> GridLayout:
        """Como se desenha o ambiente."""

    def title(self) -> str:
        """O nome a mostrar deste ambiente, com o que o distingue (o mapa, por exemplo)."""
        return self.info.display_name


ENVIRONMENTS: Registry[type[Environment[Any]]] = Registry(
    "environments", discover="ml_lab.core.rl.environments"
)
"""Todos os ambientes, pelo nome. A descoberta importa cada ficheiro de ``rl/environments/``."""
