"""Contrato dos agentes de reinforcement learning, e o ciclo de episódios que partilham.

Um agente aprende uma **política**: que ação escolher em cada estado. Os agentes do ML Lab
aprendem-na através dos **valores Q**: ``Q(s, a)`` é a recompensa que se espera ganhar até
ao fim do episódio ao fazer a ação ``a`` no estado ``s`` (e depois seguir a melhor
política). A política é escolher, em cada estado, a ação de maior Q.

O ciclo de cada episódio é igual para todos (``EpisodicAgent``):

1. o ambiente põe o agente na partida;
2. em cada passo, o agente escolhe uma ação com **ε-greedy**: com probabilidade ε, uma ao
   acaso (explorar); senão, a de maior Q (aproveitar o que já sabe). O ε começa alto e desce
   ao longo do treino, porque no início não se sabe nada e no fim já se sabe;
3. o ambiente responde com o estado seguinte e a recompensa, e o agente aprende com isso
   (``_learn``: é aqui que o Q-learning e o DQN diferem);
4. o episódio acaba no objetivo, numa armadilha, ou ao fim de ``max_steps`` passos.

Para o treino, cada episódio é um passo: o contexto recebe a recompensa, quantos passos
durou e se chegou ao objetivo, e tira fotografias (a tabela Q, a política e o caminho que o
agente faria agora). Como os modelos, o agente nunca sabe quem o observa (regra 1 dos
contratos), e por isso pausa e pára como qualquer modelo.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Self, cast

import numpy as np
from numpy.typing import NDArray
from pydantic import Field

from ml_lab.core.foundation.errors import ModelError, NotFittedError
from ml_lab.core.foundation.notices import Notice
from ml_lab.core.foundation.random import validate_seed
from ml_lab.core.foundation.registry import Registry
from ml_lab.core.models.base import Implementation
from ml_lab.core.rl.environments.base import Environment, Transition
from ml_lab.core.schemas.hyperparameters import Hyperparameters
from ml_lab.core.training.context import NullTrainingContext, TrainingContext
from ml_lab.formatting import format_count


class EpisodicParams(Hyperparameters):
    """Os hiperparâmetros que todos os agentes têm: quantos episódios, o desconto e a exploração."""

    episodes: int = Field(
        default=500,
        ge=10,
        le=5_000,
        title="Episódios",
        description="Quantas vezes o agente atravessa o ambiente, do início ao fim.",
    )
    discount: float = Field(
        default=0.95,
        ge=0.0,
        le=0.999,
        title="Desconto (γ)",
        description=(
            "Quanto vale uma recompensa futura comparada com uma já: com 0,9, uma recompensa "
            "daqui a 10 passos vale 0,9¹⁰ ≈ 0,35. Perto de 1, o agente pensa a longo prazo."
        ),
    )
    epsilon_start: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        title="Exploração no início (ε)",
        description="A probabilidade de escolher uma ação ao acaso no primeiro episódio.",
    )
    epsilon_end: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        title="Exploração no fim (ε)",
        description="A probabilidade de escolher ao acaso depois de a exploração descer.",
    )
    exploration_fraction: float = Field(
        default=0.5,
        ge=0.05,
        le=1.0,
        title="Fração a explorar",
        description="Em que parte dos episódios o ε desce do início até ao fim.",
    )


@dataclass(frozen=True, slots=True)
class AgentInfo:
    """Ficha técnica de um agente.

    Attributes:
        name: Identificador, igual ao nome no registo ``AGENTS``.
        display_name: Nome a mostrar.
        summary: Uma frase sobre o que o agente faz.
        implementation: Didático (NumPy) ou de uma biblioteca (PyTorch).
        hyperparameters: A classe dos hiperparâmetros.
        doc: A ficha didática, em ``docs/algorithms/``.
        step_limit: O hiperparâmetro que limita os passos do treino (os episódios).
    """

    name: str
    display_name: str
    summary: str
    implementation: Implementation
    hyperparameters: type[EpisodicParams]
    doc: str
    step_limit: str = "episodes"


class Agent[P: EpisodicParams](ABC):
    """Contrato de um agente.

    Args:
        hyperparameters: Os hiperparâmetros, da classe ``info.hyperparameters``; por
            omissão, os valores por omissão.
        seed: A seed: a mesma seed dá o mesmo treino (a mesma exploração e o mesmo acaso do
            ambiente).
    """

    info: ClassVar[AgentInfo]

    def __init__(self, hyperparameters: P | None = None, *, seed: int = 0) -> None:
        expected = self.info.hyperparameters
        chosen = hyperparameters if hyperparameters is not None else expected()
        if not isinstance(chosen, expected):
            raise ModelError(
                f"'{self.info.name}' recebe hiperparâmetros do tipo {expected.__name__}, "
                f"não {type(chosen).__name__}."
            )
        self.hyperparameters = cast("P", chosen)
        self.seed = validate_seed(seed)
        self._notices: list[Notice] = []
        self._trained_on: Environment[Any] | None = None

    @property
    def notices(self) -> tuple[Notice, ...]:
        """Avisos do treino."""
        return tuple(self._notices)

    @property
    def is_trained(self) -> bool:
        """Se o agente já treinou."""
        return self._trained_on is not None

    def train(self, environment: Environment[Any], context: TrainingContext | None = None) -> Self:
        """Treina o agente no ambiente.

        Args:
            environment: O ambiente.
            context: Para onde anunciar os episódios (e de onde vêm as ordens de parar).
        """
        self._notices = []
        self._train(environment, context or NullTrainingContext())
        self._trained_on = environment
        return self

    def act(self, state: int) -> int:
        """A ação da política aprendida no estado ``state`` (a de maior Q).

        Raises:
            NotFittedError: Se o agente ainda não treinou.
        """
        self._check_trained()
        return int(np.argmax(self._q(state)))

    def q_values(self) -> NDArray[np.float64]:
        """Os valores Q de todos os estados (linhas) e ações (colunas).

        Raises:
            NotFittedError: Se o agente ainda não treinou.
        """
        self._check_trained()
        return self._q_table()

    def state(self) -> dict[str, Any]:
        """O que o agente aprendeu, em valores de JSON (a tabela Q e a política)."""
        if self._trained_on is None:
            return {}
        table = self._q_table()
        return {
            "q_table": table.tolist(),
            "policy": [int(action) for action in table.argmax(axis=1)],
        }

    def snapshot_state(self, environment: Environment[Any]) -> dict[str, Any]:
        """A fotografia de agora: a tabela Q e o caminho que o agente faria a partir da partida."""
        table = self._q_table()
        path, reached = greedy_path(table, environment)
        return {"q_table": table.tolist(), "path": path, "reached": reached}

    def _note(self, notice: Notice) -> None:
        self._notices.append(notice)

    def _check_trained(self) -> None:
        if self._trained_on is None:
            raise NotFittedError(
                f"'{self.info.display_name}' ainda não treinou: treina-o primeiro."
            )

    @abstractmethod
    def _train(self, environment: Environment[Any], context: TrainingContext) -> None:
        """O treino (ver ``EpisodicAgent``)."""

    @abstractmethod
    def _q(self, state: int) -> NDArray[np.float64]:
        """Os valores Q das ações do estado ``state``."""

    @abstractmethod
    def _q_table(self) -> NDArray[np.float64]:
        """Os valores Q de todos os estados e ações."""


class EpisodicAgent[P: EpisodicParams](Agent[P]):
    """Base dos agentes que aprendem episódio a episódio, com ε-greedy (ver o módulo)."""

    def __init__(self, hyperparameters: P | None = None, *, seed: int = 0) -> None:
        super().__init__(hyperparameters, seed=seed)
        self._rng = np.random.default_rng(self.seed)
        self._epsilon = self.hyperparameters.epsilon_start

    @abstractmethod
    def _start(self, environment: Environment[Any]) -> None:
        """Prepara o que o agente vai aprender (a tabela, a rede), antes do primeiro episódio."""

    @abstractmethod
    def _learn(self, state: int, action: int, transition: Transition) -> float | None:
        """Aprende com um passo; devolve o erro (ou a loss), se aprendeu alguma coisa."""

    def epsilon(self, episode: int) -> float:
        """O ε do episódio ``episode`` (a começar em 1): desce em linha reta e depois fica."""
        params = self.hyperparameters
        decay = max(1.0, params.exploration_fraction * params.episodes)
        progress = min(1.0, (episode - 1) / decay)
        return params.epsilon_start + progress * (params.epsilon_end - params.epsilon_start)

    def _train(self, environment: Environment[Any], context: TrainingContext) -> None:
        self._rng = np.random.default_rng(self.seed)
        self._start(environment)
        reached_any = False
        episode = 0
        for episode in range(1, self.hyperparameters.episodes + 1):
            self._epsilon = self.epsilon(episode)
            reward, length, reached, errors = self._episode(environment)
            reached_any = reached_any or reached
            context.report_step(
                episode,
                loss=float(np.mean(errors)) if errors else None,
                metrics={
                    "reward": reward,
                    "length": float(length),
                    "success": float(reached),
                    "epsilon": self._epsilon,
                },
            )
            if context.wants_snapshot(episode):
                context.report_snapshot(episode, self.snapshot_state(environment))
            if context.should_stop():
                break
        if not reached_any:
            self._note(
                Notice.warning(
                    "never_reached",
                    f"Em {format_count(episode)} episódios, o agente nunca chegou ao objetivo: "
                    "não teve nada de bom para aprender. Experimenta mais episódios ou mais "
                    "exploração.",
                )
            )

    def _episode(self, environment: Environment[Any]) -> tuple[float, int, bool, list[float]]:
        """Um episódio: a recompensa total, quantos passos, se chegou ao objetivo e os erros."""
        state = environment.reset(self._rng)
        total, errors = 0.0, []
        for length in range(1, environment.max_steps + 1):
            action = self._choose(state, environment.n_actions)
            transition = environment.step(state, action, self._rng)
            error = self._learn(state, action, transition)
            if error is not None:
                errors.append(error)
            total += transition.reward
            state = transition.state
            if transition.done:
                return total, length, transition.reached_goal, errors
        return total, environment.max_steps, False, errors

    def _choose(self, state: int, n_actions: int) -> int:
        """ε-greedy: ao acaso com probabilidade ε; senão, a ação de maior Q."""
        if self._rng.random() < self._epsilon:
            return int(self._rng.integers(n_actions))
        values = self._q(state)
        # Num empate (no início, todos os Q são iguais), escolhe-se ao acaso entre os melhores:
        # escolher sempre o primeiro faria o agente ir sempre para cima.
        best = np.flatnonzero(values == values.max())
        return int(best[0] if len(best) == 1 else self._rng.choice(best))

    def state(self) -> dict[str, Any]:
        """O que o agente aprendeu, e o ε com que acabou."""
        learned = super().state()
        return {**learned, "epsilon": self._epsilon} if learned else learned


def greedy_path(
    q_table: NDArray[np.float64], environment: Environment[Any]
) -> tuple[list[int], bool]:
    """O caminho que a política de ``q_table`` faz a partir da partida, sem escorregar.

    Pára no fim do episódio, ou quando volta a um estado onde já esteve (a política anda às
    voltas, ou contra uma parede, e nunca mais sai dali).

    Returns:
        Os estados por onde passa, e se chega ao objetivo.
    """
    state = environment.reset(np.random.default_rng(0))
    path, seen = [state], {state}
    for _ in range(environment.max_steps):
        transition = environment.step(state, int(np.argmax(q_table[state])), None)
        path.append(transition.state)
        if transition.done:
            return path, transition.reached_goal
        if transition.state in seen:
            break
        seen.add(transition.state)
        state = transition.state
    return path, False


AGENTS: Registry[type[Agent[Any]]] = Registry("agents", discover="ml_lab.core.rl.agents")
"""Todos os agentes, pelo nome. A descoberta importa cada ficheiro de ``rl/agents/``."""
