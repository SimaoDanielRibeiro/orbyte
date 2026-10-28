"""Perceptrão multicamada (MLP) em PyTorch: camadas densas com uma ativação não linear.

Uma camada densa calcula, para cada neurónio, uma soma pesada das entradas mais uma
ordenada, e passa-a por uma **ativação** (ReLU, tanh ou sigmoide). Sem a ativação, várias
camadas seguidas seriam só uma conta linear, e a rede nunca desenharia uma fronteira
curva; com ela, e neurónios suficientes, aproxima qualquer forma. É isso que se vê no
treino passo a passo: a fronteira começa quase reta e vai-se curvando à volta das luas.

Os pesos iniciais são tirados ao acaso com a escala certa para a ativação (He para a ReLU,
Glorot para as outras), para os sinais não morrerem nem explodirem de camada para camada.
O ciclo de treino e a fotografia das camadas vêm do adaptador do PyTorch.

Ficha didática: ``docs/algorithms/mlp.md``.
"""

from collections.abc import Callable
from itertools import pairwise
from typing import Final, Literal, override

import torch
from pydantic import Field
from torch import nn

from ml_lab.core.foundation.types import TaskType
from ml_lab.core.models.adapters.torch_adapter import (
    TorchClassifier,
    TorchRegressor,
    TrainingParams,
)
from ml_lab.core.models.base import MODELS, Implementation, ModelFamily, ModelInfo

type Activation = Literal["relu", "tanh", "sigmoid"]

_ACTIVATIONS: Final[dict[str, Callable[[], nn.Module]]] = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
}


class MLPParams(TrainingParams):
    """Hiperparâmetros da MLP: a forma da rede e, herdados, os do treino."""

    hidden_layers: int = Field(
        default=2,
        ge=1,
        le=4,
        title="Camadas escondidas",
        description="Quantas camadas entre a entrada e a saída. Mais camadas, formas mais ricas.",
        json_schema_extra={"x-search": {"values": [1, 2, 3]}},
    )
    units: int = Field(
        default=16,
        ge=2,
        le=128,
        title="Neurónios por camada",
        description="Quantos neurónios tem cada camada escondida. Poucos: fronteira mais simples.",
        json_schema_extra={"x-search": {"values": [8, 16, 32, 64]}},
    )
    activation: Activation = Field(
        default="relu",
        title="Ativação",
        description=(
            "A não linearidade de cada neurónio. A ReLU é a habitual; a tanh e a sigmoide "
            "saturam, e em redes fundas os gradientes podem desaparecer."
        ),
        # A sigmoide fica fora da procura: satura, e quase nunca ganha às outras duas.
        json_schema_extra={
            "x-labels": {"relu": "ReLU", "tanh": "tanh", "sigmoid": "Sigmoide"},
            "x-search": {"values": ["relu", "tanh"]},
        },
    )


def mlp_network(
    n_inputs: int, n_outputs: int, params: MLPParams, generator: torch.Generator
) -> nn.Sequential:
    """As camadas da MLP: ``hidden_layers`` camadas densas com a ativação, e a saída."""
    sizes = [n_inputs, *[params.units] * params.hidden_layers]
    modules: list[nn.Module] = []
    for fan_in, fan_out in pairwise(sizes):
        layer = nn.Linear(fan_in, fan_out)
        _initialize(layer, params.activation, generator)
        modules += [layer, _ACTIVATIONS[params.activation]()]
    output = nn.Linear(sizes[-1], n_outputs)
    _initialize(output, None, generator)
    return nn.Sequential(*modules, output)


def _initialize(
    layer: nn.Linear, activation: Activation | None, generator: torch.Generator
) -> None:
    """Pesos ao acaso, na escala da ativação (He na ReLU, Glorot nas outras); ordenadas a zero."""
    if activation == "relu":
        nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu", generator=generator)
    else:
        gain = nn.init.calculate_gain(activation or "linear")
        nn.init.xavier_uniform_(layer.weight, gain=gain, generator=generator)
    nn.init.zeros_(layer.bias)


@MODELS.register("mlp")
class MLPClassifier(TorchClassifier[MLPParams]):
    """Rede neuronal densa que classifica."""

    info = ModelInfo(
        name="mlp",
        display_name="Rede neuronal (MLP)",
        summary=(
            "Camadas de neurónios com uma ativação não linear: aprende fronteiras curvas, "
            "como as das luas. Vês a fronteira a curvar-se e o que cada camada faz."
        ),
        task_types=frozenset({TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION}),
        family=ModelFamily.NEURAL,
        implementation=Implementation.PYTORCH,
        iterative=True,
        supports_proba=True,
        hyperparameters=MLPParams,
        doc="mlp.md",
        step_limit="max_epochs",
    )

    @override
    def build_network(
        self, n_inputs: int, n_outputs: int, generator: torch.Generator
    ) -> nn.Sequential:
        return mlp_network(n_inputs, n_outputs, self.hyperparameters, generator)


@MODELS.register("mlp_regression")
class MLPRegressor(TorchRegressor[MLPParams]):
    """Rede neuronal densa que prevê um número."""

    info = ModelInfo(
        name="mlp_regression",
        display_name="Rede neuronal (MLP, regressão)",
        summary=(
            "A mesma rede, com um neurónio de saída sem ativação: aprende relações não "
            "lineares entre as features e o alvo."
        ),
        task_types=frozenset({TaskType.REGRESSION}),
        family=ModelFamily.NEURAL,
        implementation=Implementation.PYTORCH,
        iterative=True,
        supports_proba=False,
        hyperparameters=MLPParams,
        doc="mlp.md",
        step_limit="max_epochs",
    )

    @override
    def build_network(
        self, n_inputs: int, n_outputs: int, generator: torch.Generator
    ) -> nn.Sequential:
        return mlp_network(n_inputs, n_outputs, self.hyperparameters, generator)
