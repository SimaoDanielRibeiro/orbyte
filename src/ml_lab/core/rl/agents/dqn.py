"""DQN (Deep Q-Network): os valores Q dados por uma rede neuronal, em vez de uma tabela.

Em PyTorch. A ideia é a do Q-learning, mas o ``Q(s, ·)`` sai de uma rede que recebe o estado
(aqui, *one-hot*: um vetor de zeros com um 1 na posição do estado). Numa grelha pequena,
uma tabela chegava. A rede é o que permite, noutros problemas, estados demasiados para uma
tabela (as imagens de um jogo, por exemplo). Treinar uma rede com o erro de diferença
temporal é instável, e o DQN junta duas peças para o acalmar:

- **memória de repetição** (*replay buffer*): os passos guardam-se numa memória, e a rede
  aprende com lotes tirados dela ao acaso. Passos seguidos são muito parecidos entre si;
  misturá-los quebra essa correlação, que faria a rede esquecer o que aprendeu antes;
- **rede-alvo**: o alvo ``r + γ · max Q(s', ·)`` é calculado por uma cópia da rede que só
  se atualiza de ``target_update`` em ``target_update`` passos. Sem ela, o alvo mexia-se a
  cada passo de treino, como perseguir a própria sombra.

A loss é a de Huber (*smooth L1*): quadrática para erros pequenos e linear para os grandes,
para um erro grande (uma armadilha inesperada) não dar um gradiente enorme.
"""

import copy
from typing import Any, override

import numpy as np
import torch
from numpy.typing import NDArray
from pydantic import Field
from torch import nn

from ml_lab.core.models.base import Implementation
from ml_lab.core.rl.agents.base import AGENTS, AgentInfo, EpisodicAgent, EpisodicParams
from ml_lab.core.rl.environments.base import Environment, Transition


class DQNParams(EpisodicParams):
    """Os hiperparâmetros do DQN."""

    episodes: int = Field(
        default=300,
        ge=10,
        le=2_000,
        title="Episodes",
        description="How many times the agent crosses the environment, from start to finish.",
    )
    learning_rate: float = Field(
        default=0.001,
        ge=0.00001,
        le=0.1,
        title="Learning rate",
        description="The size of each step of the network's optimiser (Adam).",
        json_schema_extra={"x-scale": "log"},
    )
    hidden_units: int = Field(
        default=64,
        ge=8,
        le=256,
        title="Neurons per layer",
        description="The size of the network's two hidden layers.",
    )
    batch_size: int = Field(
        default=32,
        ge=8,
        le=256,
        title="Batch size",
        description="How many steps from the memory go into each training step of the network.",
    )
    replay_size: int = Field(
        default=5_000,
        ge=100,
        le=100_000,
        title="Replay memory",
        description=(
            "How many steps the memory keeps (the oldest leave first). The network learns from "
            "batches drawn from it at random."
        ),
    )
    target_update: int = Field(
        default=100,
        ge=1,
        le=5_000,
        title="Update the target network every",
        description=(
            "Every how many training steps the target network (the one that computes the "
            "targets) becomes a copy of the network. With 1, there's no target network."
        ),
    )


@AGENTS.register("dqn")
class DQNAgent(EpisodicAgent[DQNParams]):
    """O DQN (ver o módulo)."""

    info = AgentInfo(
        name="dqn",
        display_name="DQN (Deep Q-Network)",
        summary=(
            "Q-learning with a neural network in place of the table, a replay memory and a "
            "target network to keep the training stable."
        ),
        implementation=Implementation.PYTORCH,
        hyperparameters=DQNParams,
        doc="dqn.md",
    )

    def __init__(self, hyperparameters: DQNParams | None = None, *, seed: int = 0) -> None:
        super().__init__(hyperparameters, seed=seed)
        self._network = nn.Sequential()
        self._target = nn.Sequential()
        self._optimizer: torch.optim.Optimizer | None = None
        self._identity = torch.zeros(0, 0)
        self._memory = _ReplayMemory(1, 0)
        self._updates = 0

    @override
    def _start(self, environment: Environment[Any]) -> None:
        params = self.hyperparameters
        generator = torch.Generator().manual_seed(self.seed)
        self._network = _network(
            environment.n_states, environment.n_actions, params.hidden_units, generator
        )
        self._target = copy.deepcopy(self._network)
        self._optimizer = torch.optim.Adam(self._network.parameters(), lr=params.learning_rate)
        # Cada estado é uma linha da matriz identidade: o vetor one-hot que entra na rede.
        self._identity = torch.eye(environment.n_states)
        self._memory = _ReplayMemory(params.replay_size, environment.n_states)
        self._updates = 0

    @override
    def _learn(self, state: int, action: int, transition: Transition) -> float | None:
        params = self.hyperparameters
        self._memory.add(state, action, transition)
        if len(self._memory) < params.batch_size or self._optimizer is None:
            return None  # ainda há poucos passos na memória para um lote
        states, actions, rewards, next_states, done = self._memory.sample(
            params.batch_size, self._rng
        )
        with torch.no_grad():
            future = self._target(self._identity[next_states]).max(dim=1).values
            targets = rewards + params.discount * future * (1.0 - done)
        predicted = self._network(self._identity[states]).gather(1, actions[:, None])[:, 0]
        loss = nn.functional.smooth_l1_loss(predicted, targets)
        self._optimizer.zero_grad()
        torch.autograd.backward(loss)
        self._optimizer.step()
        self._updates += 1
        if self._updates % params.target_update == 0:
            self._target.load_state_dict(self._network.state_dict())
        return float(loss.item())

    @override
    def _q(self, state: int) -> NDArray[np.float64]:
        with torch.no_grad():
            values: NDArray[np.float64] = (
                self._network(self._identity[state : state + 1])[0].double().numpy()
            )
        return values

    @override
    def _q_table(self) -> NDArray[np.float64]:
        with torch.no_grad():
            table: NDArray[np.float64] = self._network(self._identity).double().numpy()
        return table

    @override
    def state(self) -> dict[str, Any]:
        learned = super().state()
        if not learned:
            return learned
        parameters = sum(parameter.numel() for parameter in self._network.parameters())
        return {**learned, "n_parameters": parameters, "updates": self._updates}


class _ReplayMemory:
    """A memória de repetição: os últimos ``size`` passos, numa fila circular.

    Args:
        size: Quantos passos guarda.
        n_states: Quantos estados há (só para validar).
    """

    def __init__(self, size: int, n_states: int) -> None:
        self._states = np.zeros(size, dtype=np.int64)
        self._actions = np.zeros(size, dtype=np.int64)
        self._rewards = np.zeros(size, dtype=np.float32)
        self._next_states = np.zeros(size, dtype=np.int64)
        self._done = np.zeros(size, dtype=np.float32)
        self._size, self._count, self._position = size, 0, 0
        self._n_states = n_states

    def __len__(self) -> int:
        return self._count

    def add(self, state: int, action: int, transition: Transition) -> None:
        """Guarda um passo (por cima do mais antigo, se a memória estiver cheia)."""
        i = self._position
        self._states[i], self._actions[i] = state, action
        self._rewards[i], self._next_states[i] = transition.reward, transition.state
        self._done[i] = float(transition.done)
        self._position = (i + 1) % self._size
        self._count = min(self._count + 1, self._size)

    def sample(
        self, n: int, rng: np.random.Generator
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """``n`` passos ao acaso: os estados, as ações, as recompensas, os seguintes e os finais."""
        chosen = rng.integers(self._count, size=n)
        return (
            torch.from_numpy(self._states[chosen]),
            torch.from_numpy(self._actions[chosen]),
            torch.from_numpy(self._rewards[chosen]),
            torch.from_numpy(self._next_states[chosen]),
            torch.from_numpy(self._done[chosen]),
        )


def _network(
    n_states: int, n_actions: int, hidden: int, generator: torch.Generator
) -> nn.Sequential:
    """A rede: o estado (one-hot) → duas camadas escondidas com ReLU → um Q por ação."""
    network = nn.Sequential(
        nn.Linear(n_states, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, n_actions),
    )
    for layer in network:
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu", generator=generator)
            nn.init.zeros_(layer.bias)
    return network
