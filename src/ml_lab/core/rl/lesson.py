"""A lição do ecrã Reinforcement learning: um agente a aprender, visto passo a passo.

Antes de se treinar um agente, o ecrã mostra como um agente aprende, numa animação: o ciclo de
cada passo (o estado, a ação, a recompensa e a correção do Q) e o que muda de episódio para
episódio. Para isso, treina-se o agente no penhasco, sempre com a mesma seed, e guardam-se
alguns episódios inteiros:

1. o primeiro: o agente ainda não sabe nada e escolhe ao acaso;
2. uma queda dos primeiros episódios: cada armadilha deixa um Q negativo;
3. o primeiro que chega ao objetivo, quase por acaso: é aí que o +1 entra no que aprende;
4. um do meio, com o valor já a espalhar-se a partir do objetivo;
5. o último, com o caminho aprendido.

Há uma lição para cada agente (ADR 0050): o Q-learning corrige uma tabela a cada passo; o DQN
treina uma rede com lotes da memória de repetição, e vê-se o valor espalhar-se mais devagar. A
lição grava o treino de verdade: junta ao agente um gravador (`_Recording`), por herança, que
toma nota de cada passo e de cada episódio sem mudar o que o agente faz.

Cada passo diz onde estava o agente, a ação, se a escolheu ao acaso (explorar) ou por ser a
melhor que conhecia (aproveitar), a recompensa, a surpresa (o erro de diferença temporal) e o
valor da célula depois de aprender. Cada episódio traz o valor de cada célula no início e uma
frase sobre o que nele se vê. A interface só anima o que aqui se calcula.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final, cast, override

import numpy as np
from numpy.typing import NDArray

from ml_lab.core.foundation.errors import InvalidParameterError
from ml_lab.core.rl.agents.base import AGENTS, EpisodicAgent, EpisodicParams
from ml_lab.core.rl.agents.q_learning import QLearningAgent
from ml_lab.core.rl.environments.base import CellKind, Environment, GridLayout, Transition
from ml_lab.core.rl.environments.gridworld import GridWorld, GridWorldParams
from ml_lab.core.visualization_data.policy import policy_frame
from ml_lab.formatting import format_count, format_number, format_percent, format_quantity

LESSON_LAYOUT: Final = "penhasco"
"""O mapa da lição: o penhasco tem as três recompensas (o passo, a armadilha e o objetivo)."""

LESSON_AGENT: Final = "q_learning"
"""O agente da lição por omissão: o Q-learning, o mais simples de seguir."""

LESSON_EPISODES: Final = 160
"""Episódios que chegam para o Q-learning aprender o caminho, com os valores por omissão."""

EPISODES: Final[Mapping[str, int]] = MappingProxyType({LESSON_AGENT: LESSON_EPISODES})
"""Os episódios de cada lição, quando não são os do agente (o DQN fica com os seus, 300: uma
rede aprende mais devagar do que uma tabela)."""

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
        agent: O agente, pelo nome no registo ``AGENTS``.
        environment_name: O ambiente ("Grelha · penhasco").
        agent_display_name: O agente ("Q-learning").
        layout: A grelha.
        learning_rate: A taxa de aprendizagem: a fração da surpresa que corrige o Q (α), no
            Q-learning; o passo do otimizador da rede, no DQN.
        discount: O desconto (γ) de uma recompensa futura.
        episodes: Quantos episódios teve o treino.
        rewards: A recompensa de cada episódio do treino, por ordem.
        kept: Os episódios guardados, por ordem.
        update_topic: O tema que explica a correção do Q deste agente (o último cartão do
            ciclo): numa tabela não é o mesmo que numa rede.
        update_rule: O que acontece ao Q depois da surpresa, numa frase curta, sem ponto final
            ("Q moves α = 0.5 of it").
    """

    agent: str
    environment_name: str
    agent_display_name: str
    layout: GridLayout
    learning_rate: float
    discount: float
    episodes: int
    rewards: tuple[float, ...]
    kept: tuple[LessonEpisode, ...]
    update_topic: str
    update_rule: str


@dataclass(frozen=True, slots=True)
class _Played:
    """Um episódio do treino, como o gravador o viu (antes de se escolherem os guardados)."""

    epsilon: float
    table: NDArray[np.float64]
    steps: tuple[LessonStep, ...]
    reward: float
    reached: bool


class _Recording(EpisodicAgent[Any]):
    """Toma nota de cada passo e de cada episódio, por cima de um agente episódico qualquer.

    Junta-se ao agente por herança (``_recorder``): ``_choose``, ``_learn`` e ``_episode``
    passam por aqui e seguem para os do agente, que treina como sempre.

    A surpresa de cada passo calcula-se como o agente a vê antes de aprender: a recompensa, mais
    o melhor Q da célula seguinte descontado, menos o Q da ação. No Q-learning é o erro que
    corrige a tabela; no DQN, a rede aprende com um lote da memória, e esta é a surpresa do
    passo que acabou de lá entrar.
    """

    def __init__(self, hyperparameters: Any = None, *, seed: int = 0) -> None:
        super().__init__(hyperparameters, seed=seed)
        self.played: list[_Played] = []
        self._steps: list[LessonStep] = []
        self._explored = False

    @override
    def _choose(self, state: int, n_actions: int) -> int:
        action, self._explored = self._decide(state, n_actions)
        return action

    @override
    def _learn(self, state: int, action: int, transition: Transition) -> float | None:
        expected = float(self._q(state)[action])
        future = (
            0.0
            if transition.done
            else self.hyperparameters.discount * float(self._q(transition.state).max())
        )
        # O gravador nunca existe sozinho (`_recorder` junta-o a um agente): o `super()` chega
        # sempre ao `_learn` do agente, que o mypy não consegue ver daqui.
        error = super()._learn(state, action, transition)  # type: ignore[safe-super]
        self._steps.append(
            LessonStep(
                state=state,
                action=action,
                explored=self._explored,
                reward=transition.reward,
                next_state=transition.state,
                surprise=transition.reward + future - expected,
                value=float(self._q(state).max()),
            )
        )
        return error

    @override
    def _episode(self, environment: Environment[Any]) -> tuple[float, int, bool, list[float]]:
        self._steps = []
        table = self._q_table()
        reward, length, reached, errors = super()._episode(environment)
        self.played.append(_Played(self._epsilon, table, tuple(self._steps), reward, reached))
        return reward, length, reached, errors


def _recorder(agent_class: type[EpisodicAgent[Any]]) -> type[_Recording]:
    """O agente com o gravador por cima: uma classe nova, que herda dos dois."""
    return cast(
        "type[_Recording]", type(f"{agent_class.__name__}Lesson", (_Recording, agent_class), {})
    )


def lesson(agent: str = LESSON_AGENT) -> ReinforcementLesson:
    """Treina o agente da lição e escolhe os episódios a mostrar (ver o módulo).

    Args:
        agent: O agente, pelo nome no registo ``AGENTS``.

    Raises:
        ComponentNotFoundError: Se não houver um agente com esse nome.
        InvalidParameterError: Se o agente não aprender episódio a episódio.
    """
    agent_class = AGENTS.get(agent)
    if not issubclass(agent_class, EpisodicAgent):
        raise InvalidParameterError(
            f"{agent_class.info.display_name} doesn't learn episode by episode: there's no "
            "lesson to watch."
        )
    defaults: EpisodicParams = agent_class.info.hyperparameters()
    params = defaults.model_copy(update={"episodes": EPISODES.get(agent, defaults.episodes)})
    environment = GridWorld(GridWorldParams(layout=LESSON_LAYOUT))
    recorder = _recorder(agent_class)(params, seed=LESSON_SEED)
    recorder.train(environment)
    tabular = isinstance(recorder, QLearningAgent)
    layout = environment.layout()
    played = recorder.played
    first_success = next(
        (number for number, episode in enumerate(played, start=1) if episode.reached), None
    )
    # O fim da exploração: o primeiro episódio com o ε mais baixo (a conta do ε arredonda).
    explored_until = next(
        (
            number
            for number in range(1, len(played) + 1)
            if math.isclose(recorder.epsilon(number), params.epsilon_end)
        ),
        len(played),
    )
    kept = sorted(set(_chosen(played, layout, first_success, explored_until)))
    learning_rate = float(getattr(params, "learning_rate", 0.0))
    return ReinforcementLesson(
        agent=agent,
        environment_name=environment.title(),
        agent_display_name=agent_class.info.display_name,
        layout=layout,
        learning_rate=learning_rate,
        discount=params.discount,
        episodes=len(played),
        rewards=tuple(episode.reward for episode in played),
        kept=tuple(
            _kept(number, played[number - 1], layout, first_success, len(played), tabular=tabular)
            for number in kept
        ),
        update_topic="rl_update" if tabular else "rl_update_network",
        update_rule=_update_rule(params, learning_rate, tabular=tabular),
    )


def _update_rule(params: EpisodicParams, learning_rate: float, *, tabular: bool) -> str:
    """O que acontece ao Q depois da surpresa, numa frase (o último cartão do ciclo)."""
    if tabular:
        return f"Q moves α = {format_number(learning_rate, 1)} of it"
    batch = getattr(params, "batch_size", None)
    drawn = "a batch of steps" if batch is None else f"{format_count(int(batch))} steps"
    return f"the network takes a training step on {drawn} drawn from its memory"


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
    *,
    tabular: bool,
) -> LessonEpisode:
    """Um episódio guardado, pronto a mostrar."""
    return LessonEpisode(
        episode=number,
        epsilon=played.epsilon,
        before=policy_frame(number, {"q_table": played.table}, layout).values,
        steps=played.steps,
        reward=played.reward,
        reached=played.reached,
        story=_story(number, played, layout, first_success, last, tabular=tabular),
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
    *,
    tabular: bool,
) -> str:
    """O que se vê num episódio: como escolhe as ações, e como acaba.

    Uma tabela e uma rede aprendem o mesmo de maneiras diferentes, e a frase di-lo: o Q-learning
    começa com tudo a zero e corrige uma célula de cada vez; o DQN começa com uma rede ao acaso
    e aprende com lotes da memória.
    """
    steps = len(played.steps)
    explored = sum(step.explored for step in played.steps)
    walked = format_quantity(steps, "step", "steps")
    if number == 1:
        opening = (
            "It knows nothing yet: every Q is zero, so it picks at random."
            if tabular
            else "It knows nothing yet: the network starts with random weights, so its Q values "
            "mean nothing, and it picks at random."
        )
    elif explored == 0:
        opening = (
            f"It barely explores any more (ε = {format_number(played.epsilon, 2)}): in this "
            "episode, it always picked the action with the highest Q."
        )
    else:
        opening = (
            f"It picks at random in {format_percent(explored / steps, 0)} of the steps "
            f"(ε = {format_number(played.epsilon, 2)})."
        )
    if played.reached and number == first_success:
        learns = (
            "the +1 goes into the table, in the Q of the last action, and starts spreading "
            "backwards"
            if tabular
            else "the step with the +1 goes into the replay memory, and the network starts to "
            "learn it from the batches it draws"
        )
        ending = f"It reaches the goal for the first time, after {walked}: {learns}."
    elif played.reached and number == last:
        # O Q-learning aprende o caminho mais curto, colado às armadilhas (é off-policy); a rede
        # do DQN chega lá, mas nem sempre pelo mais curto.
        ending = (
            f"It has learned the path: it reaches the goal in {walked}, by the shortest one, "
            "right along the traps."
            if tabular
            else f"It has learned a way to the goal: it gets there in {walked}."
        )
    elif played.reached:
        ending = (
            f"It reaches the goal in {walked}: the value has already spread backwards from the "
            "goal, through the cells of the path."
        )
    elif _fell(played, layout):
        learns = (
            "the Q of the action that took it there turns negative"
            if tabular
            else "batch after batch, the network learns that the action that took it there is "
            "worth less"
        )
        ending = f"It falls into a trap after {walked} and gets −1: {learns}."
    else:
        ending = f"It wanders around for {walked}, and the episode ends at the step limit."
    return f"{opening} {ending}"
