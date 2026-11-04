"""Q-learning: uma tabela com o valor de cada ação em cada estado, corrigida a cada passo.

Didático, em NumPy. A tabela ``Q`` começa a zeros. Depois de cada passo (estado ``s``, ação
``a``, recompensa ``r``, estado seguinte ``s'``), o valor de ``Q(s, a)`` aproxima-se do que
o passo mostrou:

    alvo = r + γ · max_a' Q(s', a')          (só r, se o episódio acabou)
    Q(s, a) ← Q(s, a) + α · (alvo − Q(s, a))

O ``alvo − Q(s, a)`` é o **erro de diferença temporal**: a surpresa entre o que se esperava
e o que se viu, mais o que se espera daí em diante. O ``max`` é o que torna o Q-learning
*off-policy*: aprende o valor da melhor política, mesmo enquanto explora ao acaso. É por isso
que, no penhasco, aprende o caminho colado às armadilhas, o mais curto, embora durante o
treino caia lá muitas vezes.
"""

from typing import Any, override

import numpy as np
from numpy.typing import NDArray
from pydantic import Field

from ml_lab.core.models.base import Implementation
from ml_lab.core.rl.agents.base import AGENTS, AgentInfo, EpisodicAgent, EpisodicParams
from ml_lab.core.rl.environments.base import Environment, Transition


class QLearningParams(EpisodicParams):
    """Os hiperparâmetros do Q-learning."""

    learning_rate: float = Field(
        default=0.5,
        gt=0.0,
        le=1.0,
        title="Learning rate (α)",
        description=(
            "How much each step corrects Q: at 1, Q becomes only what was just seen; at 0.1, it "
            "changes slowly, which helps when the environment is random (with slipping)."
        ),
    )


@AGENTS.register("q_learning")
class QLearningAgent(EpisodicAgent[QLearningParams]):
    """O Q-learning tabular (ver o módulo)."""

    info = AgentInfo(
        name="q_learning",
        display_name="Q-learning",
        summary=(
            "Learns a table with the value of each action in each state, corrected at every "
            "step by what it has just seen."
        ),
        implementation=Implementation.DIDACTIC,
        hyperparameters=QLearningParams,
        doc="q_learning.md",
    )

    def __init__(self, hyperparameters: QLearningParams | None = None, *, seed: int = 0) -> None:
        super().__init__(hyperparameters, seed=seed)
        self._table = np.zeros((0, 0))

    @override
    def _start(self, environment: Environment[Any]) -> None:
        self._table = np.zeros((environment.n_states, environment.n_actions))

    @override
    def _learn(self, state: int, action: int, transition: Transition) -> float:
        params = self.hyperparameters
        future = 0.0 if transition.done else params.discount * self._table[transition.state].max()
        error = transition.reward + future - self._table[state, action]
        self._table[state, action] += params.learning_rate * error
        return abs(float(error))

    @override
    def _q(self, state: int) -> NDArray[np.float64]:
        values: NDArray[np.float64] = self._table[state]
        return values

    @override
    def _q_table(self) -> NDArray[np.float64]:
        return self._table.copy()
