"""A lição do ecrã Reinforcement learning: um Q-learning pequeno, visto passo a passo.

Antes de se treinar um agente, o ecrã mostra como um agente aprende, numa animação: o ciclo de
cada passo (o estado, a ação, a recompensa e a correção do Q) e o que muda de episódio para
episódio. Para isso, treina-se um Q-learning no penhasco, sempre com a mesma seed, e guardam-se
alguns episódios inteiros:

1. o primeiro: o agente ainda não sabe nada e escolhe ao acaso;
2. uma queda dos primeiros episódios: cada armadilha deixa um Q negativo;
3. o primeiro que chega ao objetivo, quase por acaso: é aí que o +1 entra na tabela;
4. um do meio, com o valor já a espalhar-se a partir do objetivo;
5. o último, com o caminho aprendido.

Cada passo diz onde estava o agente, a ação, se a escolheu ao acaso (explorar) ou por ser a
melhor que conhecia (aproveitar), a recompensa, a surpresa (o erro de diferença temporal) e o
valor da célula depois de aprender. Cada episódio traz o valor de cada célula no início e uma
frase sobre o que nele se vê. A interface só anima o que aqui se calcula.
"""

import math
from dataclasses import dataclass
from typing import Any, Final, override

import numpy as np
from numpy.typing import NDArray

from ml_lab.core.rl.agents.q_learning import QLearningAgent, QLearningParams
from ml_lab.core.rl.environments.base import CellKind, Environment, GridLayout, Transition
from ml_lab.core.rl.environments.gridworld import GridWorld, GridWorldParams
from ml_lab.core.visualization_data.policy import policy_frame
from ml_lab.formatting import format_number, format_percent, format_quantity

LESSON_LAYOUT: Final = "penhasco"
"""O mapa da lição: o penhasco tem as três recompensas (o passo, a armadilha e o objetivo)."""

LESSON_EPISODES: Final = 160
"""Episódios que chegam para o Q-learning aprender o caminho, com os valores por omissão."""

LESSON_SEED: Final = 7

EARLY_EPISODES: Final = 15
"""Onde se procura a queda dos primeiros episódios."""

LONGEST_FALL: Final = 30
"""A queda escolhida é a mais comprida até este número de passos: vê-se o agente a andar antes
de cair, sem a animação se arrastar."""


@dataclass(frozen=True, slots=True)
class LessonStep:
    """Um passo de um episódio da lição.

    Attributes:
        state: A célula onde o agente estava (linha a linha, como em ``GridLayout.cells``).
        action: A ação que escolheu (o número dela em ``GridLayout.action_labels``).
        explored: Se a escolheu ao acaso (explorar) ou por ser a melhor que conhecia
            (aproveitar).
        reward: A recompensa que recebeu.
        next_state: A célula onde foi parar.
        surprise: O erro de diferença temporal: o que viu (a recompensa e o melhor Q da
            célula seguinte, descontado) menos o que esperava, ``Q(estado, ação)``.
        value: O valor da célula ``state`` depois de aprender com este passo (o maior Q dela).
    """

    state: int
    action: int
    explored: bool
    reward: float
    next_state: int
    surprise: float
    value: float


@dataclass(frozen=True, slots=True)
class LessonEpisode:
    """Um episódio da lição, inteiro.

    Attributes:
        episode: O número do episódio no treino (a começar em 1).
        epsilon: A exploração nesse episódio: a probabilidade de escolher ao acaso.
        before: O valor de cada célula no início do episódio (o maior Q); ``None`` onde não
            há nada a escolher (as paredes, o objetivo e as armadilhas).
        steps: Os passos, por ordem.
        reward: A recompensa total do episódio.
        reached: Se chegou ao objetivo.
        story: O que se vê neste episódio, numa ou duas frases.
    """

    episode: int
    epsilon: float
    before: tuple[float | None, ...]
    steps: tuple[LessonStep, ...]
    reward: float
    reached: bool
    story: str


@dataclass(frozen=True, slots=True)
class ReinforcementLesson:
    """A lição: o ambiente, o agente e os episódios escolhidos, passo a passo.

    Attributes:
        environment_name: O ambiente ("Grelha · penhasco").
        agent_display_name: O agente ("Q-learning").
        layout: A grelha.
        learning_rate: A taxa de aprendizagem (α): a fração da surpresa que corrige o Q.
        discount: O desconto (γ) de uma recompensa futura.
        episodes: Quantos episódios teve o treino.
        rewards: A recompensa de cada episódio do treino, por ordem.
        kept: Os episódios guardados, por ordem.
    """

    environment_name: str
    agent_display_name: str
    layout: GridLayout
    learning_rate: float
    discount: float
    episodes: int
    rewards: tuple[float, ...]
    kept: tuple[LessonEpisode, ...]


@dataclass(frozen=True, slots=True)
class _Played:
    """Um episódio do treino, como o gravador o viu (antes de se escolherem os guardados)."""

    epsilon: float
    table: NDArray[np.float64]
    steps: tuple[LessonStep, ...]
    reward: float
    reached: bool


class _Recorder(QLearningAgent):
    """O Q-learning de sempre, a tomar nota de cada passo e de cada episódio."""

    def __init__(self, hyperparameters: QLearningParams, *, seed: int) -> None:
        super().__init__(hyperparameters, seed=seed)
        self.played: list[_Played] = []
        self._steps: list[LessonStep] = []
        self._explored = False

    @override
    def _choose(self, state: int, n_actions: int) -> int:
        action, self._explored = self._decide(state, n_actions)
        return action

    @override
    def _learn(self, state: int, action: int, transition: Transition) -> float:
        expected = float(self._table[state, action])
        error = super()._learn(state, action, transition)
        learned = float(self._table[state, action])
        self._steps.append(
            LessonStep(
                state=state,
                action=action,
                explored=self._explored,
                reward=transition.reward,
                next_state=transition.state,
                # O Q andou na direção da surpresa: o sinal dela é o da mudança.
                surprise=math.copysign(error, learned - expected),
                value=float(self._table[state].max()),
            )
        )
        return error

    @override
    def _episode(self, environment: Environment[Any]) -> tuple[float, int, bool, list[float]]:
        self._steps = []
        table = self._table.copy()
        reward, length, reached, errors = super()._episode(environment)
        self.played.append(_Played(self._epsilon, table, tuple(self._steps), reward, reached))
        return reward, length, reached, errors


def lesson() -> ReinforcementLesson:
    """Treina o Q-learning da lição e escolhe os episódios a mostrar (ver o módulo)."""
    environment = GridWorld(GridWorldParams(layout=LESSON_LAYOUT))
    params = QLearningParams(episodes=LESSON_EPISODES)
    agent = _Recorder(params, seed=LESSON_SEED)
    agent.train(environment)
    layout = environment.layout()
    played = agent.played
    first_success = next(
        (number for number, episode in enumerate(played, start=1) if episode.reached), None
    )
    # O fim da exploração: o primeiro episódio com o ε mais baixo (a conta do ε arredonda).
    explored_until = next(
        (
            number
            for number in range(1, len(played) + 1)
            if math.isclose(agent.epsilon(number), params.epsilon_end)
        ),
        len(played),
    )
    kept = sorted(set(_chosen(played, layout, first_success, explored_until)))
    return ReinforcementLesson(
        environment_name=environment.title(),
        agent_display_name=agent.info.display_name,
        layout=layout,
        learning_rate=params.learning_rate,
        discount=params.discount,
        episodes=len(played),
        rewards=tuple(episode.reward for episode in played),
        kept=tuple(
            _kept(number, played[number - 1], layout, first_success, len(played)) for number in kept
        ),
    )


def _chosen(
    played: list[_Played], layout: GridLayout, first_success: int | None, explored_until: int
) -> list[int]:
    """Os episódios a guardar: o primeiro, uma queda, o primeiro êxito, um do meio, o último.

    O do meio fica entre o primeiro êxito e o fim da exploração (o primeiro, a partir daí, que
    chega ao objetivo): o agente já conhece parte do caminho, mas ainda explora.
    """
    last = len(played)
    falls = [
        number
        for number in range(2, min(EARLY_EPISODES, last) + 1)
        if _fell(played[number - 1], layout)
    ]
    short = [number for number in falls if len(played[number - 1].steps) <= LONGEST_FALL]
    fall = max(short or falls, key=lambda number: len(played[number - 1].steps), default=None)
    if first_success is None:
        middle = last // 2
    elif first_success < explored_until:
        middle = (first_success + explored_until) // 2
    else:
        middle = (first_success + last) // 2
    # De preferência um que chegue ao objetivo: vê-se o caminho a formar-se.
    middle = next((number for number in range(middle, last) if played[number - 1].reached), middle)
    return [number for number in (1, fall, first_success, middle, last) if number is not None]


def _kept(
    number: int,
    played: _Played,
    layout: GridLayout,
    first_success: int | None,
    last: int,
) -> LessonEpisode:
    """Um episódio guardado, pronto a mostrar."""
    return LessonEpisode(
        episode=number,
        epsilon=played.epsilon,
        before=policy_frame(number, {"q_table": played.table}, layout).values,
        steps=played.steps,
        reward=played.reward,
        reached=played.reached,
        story=_story(number, played, layout, first_success, last),
    )


def _fell(played: _Played, layout: GridLayout) -> bool:
    """Se o episódio acabou numa armadilha."""
    return bool(played.steps) and layout.kind(played.steps[-1].next_state) is CellKind.TRAP


def _story(
    number: int,
    played: _Played,
    layout: GridLayout,
    first_success: int | None,
    last: int,
) -> str:
    """O que se vê num episódio: como escolhe as ações, e como acaba."""
    steps = len(played.steps)
    explored = sum(step.explored for step in played.steps)
    walked = format_quantity(steps, "passo", "passos")
    if number == 1:
        opening = "Ainda não sabe nada: todos os Q são zero, e por isso escolhe ao acaso."
    elif explored == 0:
        opening = (
            f"Já quase não explora (ε = {format_number(played.epsilon, 2)}): neste episódio, "
            "escolheu sempre a ação de maior Q."
        )
    else:
        opening = (
            f"Escolhe ao acaso em {format_percent(explored / steps, 0)} dos passos "
            f"(ε = {format_number(played.epsilon, 2)})."
        )
    if played.reached and number == first_success:
        ending = (
            f"Chega ao objetivo pela primeira vez, ao fim de {walked}: o +1 entra na tabela, "
            "no Q da última ação, e começa a espalhar-se para trás."
        )
    elif played.reached and number == last:
        ending = (
            f"Aprendeu o caminho: chega ao objetivo em {walked}, pelo mais curto, colado às "
            "armadilhas."
        )
    elif played.reached:
        ending = (
            f"Chega ao objetivo em {walked}: o valor já se espalhou do objetivo para trás, "
            "pelas células do caminho."
        )
    elif _fell(played, layout):
        ending = (
            f"Cai numa armadilha ao fim de {walked} e recebe −1: o Q da ação que o levou lá "
            "fica negativo."
        )
    else:
        ending = f"Anda {walked} às voltas, e o episódio acaba no limite de passos."
    return f"{opening} {ending}"
