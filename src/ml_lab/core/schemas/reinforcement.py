"""Configuração de um treino de reinforcement learning: o ambiente, o agente e a seed.

É o que entra no serviço de reinforcement learning e o que fica guardado com cada run: com
a mesma configuração obtém-se o mesmo treino. Como o ``ExperimentConfig``, vai e volta de
JSON sem perder nada, e é o que a API recebe.

Exemplo:
    >>> config = ReinforcementConfig.model_validate(
    ...     {
    ...         "environment": {"name": "gridworld", "parameters": {"layout": "penhasco"}},
    ...         "agent": {"name": "q_learning", "hyperparameters": {"episodes": 300}},
    ...         "seed": 7,
    ...     }
    ... )
    >>> config.agent.hyperparameters["episodes"]
    300
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ml_lab.core.foundation.random import MAX_SEED


class EnvironmentSpec(BaseModel):
    """O ambiente escolhido e os seus parâmetros.

    Attributes:
        name: Nome do ambiente no registo, por exemplo ``"gridworld"``.
        parameters: Valores escolhidos; os que faltam usam o valor por omissão.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(default="gridworld", min_length=1)
    parameters: dict[str, Any] = Field(default_factory=dict)


class AgentSpec(BaseModel):
    """O agente escolhido e os seus hiperparâmetros.

    Attributes:
        name: Nome do agente no registo, por exemplo ``"q_learning"``.
        hyperparameters: Valores escolhidos; os que faltam usam o valor por omissão.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    hyperparameters: dict[str, Any] = Field(default_factory=dict)


class ReinforcementConfig(BaseModel):
    """Tudo o que define um treino de reinforcement learning.

    Attributes:
        environment: O ambiente.
        agent: O agente e os hiperparâmetros.
        seed: Seed raiz; ``None`` gera uma nova, que fica registada no resultado.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    environment: EnvironmentSpec = Field(default_factory=EnvironmentSpec)
    agent: AgentSpec
    seed: int | None = Field(default=None, ge=0, le=MAX_SEED)
