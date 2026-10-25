"""Adaptador do contrato LabModel para as redes do PyTorch: o ciclo de treino, escrito uma vez.

O scikit-learn treina de uma vez, por dentro. O PyTorch deixa (e obriga a) escrever o
ciclo de treino à mão, e é por isso que o ML Lab o usa (ADR 0001). Em cada época:

1. baralha o treino com o gerador do modelo (a mesma seed repete tudo) e percorre-o em
   lotes (*mini-batches*). Em cada lote: as previsões (*forward*), a loss, os gradientes
   de todos os pesos (*backward*, a retropropagação) e um passo do otimizador;
2. mede a loss média da época e, com *early stopping*, a loss numa parte do treino posta
   de lado (a validação), com que a rede nunca aprende;
3. anuncia a época ao contexto de treino e, quando ele pede, tira uma fotografia: as
   previsões na sonda e o que cada camada está a fazer, ou seja, a ativação de cada
   neurónio e o tamanho dos gradientes.

Com *early stopping*, o treino pára quando a validação deixa de melhorar durante
``patience`` épocas, e a rede fica com os pesos da melhor época (ADR 0021).

Uma rede concreta só diz que camadas tem (``build_network``): uma sequência de camadas
lineares, cada uma seguida da sua ativação, e uma camada linear de saída. As bases de
classificação e de regressão dizem que loss usam e como se lê a saída.
"""

import copy
import math
from abc import abstractmethod
from typing import Any, Final, Literal, override

import numpy as np
import torch
from numpy.typing import NDArray
from pydantic import Field
from torch import nn

from ml_lab.core.foundation.errors import ModelError
from ml_lab.core.foundation.notices import Notice
from ml_lab.core.models.base import Features, LabModel, Target
from ml_lab.core.models.supervised.gradient_descent import diverged_notice, has_diverged
from ml_lab.core.schemas.hyperparameters import Hyperparameters
from ml_lab.core.training.context import TrainingContext
from ml_lab.formatting import format_count

SAMPLE_ROWS: Final = 256
"""Linhas de treino usadas para medir as ativações de cada camada nas fotografias."""

STUCK_STD: Final = 1e-3
"""Um neurónio cuja saída quase não varia entre exemplos (desvio abaixo disto) está parado."""

HISTOGRAM_BINS: Final = 20

MIN_IMPROVEMENT: Final = 1e-4
"""O early stopping só conta como melhoria uma descida da loss de validação maior do que isto."""

_ACTIVATION_LABELS: Final = {nn.ReLU: "ReLU", nn.Tanh: "tanh", nn.Sigmoid: "sigmoide"}


class TrainingParams(Hyperparameters):
    """Os hiperparâmetros do treino, comuns a todas as redes."""

    optimizer: Literal["adam", "sgd"] = Field(
        default="adam",
        title="Otimizador",
        description=(
            "Como se dá cada passo. O SGD com momento segue o gradiente com alguma inércia; o "
            "Adam ajusta o passo de cada peso sozinho, e costuma funcionar à primeira."
        ),
        json_schema_extra={"x-labels": {"adam": "Adam", "sgd": "SGD com momento"}},
    )
    learning_rate: float = Field(
        default=0.01,
        ge=0.0001,
        le=1.0,
        title="Taxa de aprendizagem",
        description="O tamanho de cada passo. Grande demais, a loss salta ou explode.",
        json_schema_extra={"x-scale": "log", "x-search": {"range": [0.001, 0.03]}},
    )
    batch_size: int = Field(
        default=32,
        ge=1,
        le=1024,
        title="Tamanho do lote",
        description=(
            "Quantos exemplos em cada passo. Lotes pequenos dão passos ruidosos (que ajudam a "
            "sair de mínimos maus); lotes grandes, passos certeiros mas poucos por época."
        ),
        json_schema_extra={"x-search": {"values": [16, 32, 64]}},
    )
    max_epochs: int = Field(
        default=300,
        ge=5,
        le=5_000,
        title="Máximo de épocas",
        description="Quantas passagens pelos dados de treino, no máximo.",
    )
    alpha: float = Field(
        default=0.0001,
        ge=0.000001,
        le=0.1,
        title="Regularização L2 (alpha)",
        description="Puxa os pesos para zero em cada passo: quanto maior, mais suave a fronteira.",
        json_schema_extra={"x-scale": "log", "x-search": {"range": [0.00001, 0.01]}},
    )
    early_stopping: bool = Field(
        default=True,
        title="Early stopping",
        description=(
            "Põe de lado uma parte do treino (a validação) e pára quando a loss nela deixa de "
            "melhorar; a rede fica com os pesos da melhor época."
        ),
    )
    patience: int = Field(
        default=20,
        ge=1,
        le=500,
        title="Paciência",
        description="Quantas épocas sem melhorar na validação até parar.",
    )
    validation_fraction: float = Field(
        default=0.1,
        ge=0.05,
        le=0.5,
        title="Fração para validação",
        description="Que parte do treino fica de lado para o early stopping.",
    )


class TorchModel[P: TrainingParams](LabModel[P]):
    """Base das redes do PyTorch: o ciclo de treino, o early stopping e a fotografia das camadas."""

    def __init__(self, hyperparameters: P | None = None, *, seed: int = 0) -> None:
        super().__init__(hyperparameters, seed=seed)
        self._network = nn.Sequential()
        self._sample = torch.zeros(0)
        self._gradients: list[float] = []
        self._epochs = 0
        self._best_epoch: int | None = None
        self._early_stopped = False

    @abstractmethod
    def build_network(
        self, n_inputs: int, n_outputs: int, generator: torch.Generator
    ) -> nn.Sequential:
        """As camadas da rede, com os pesos iniciais tirados de ``generator``."""

    @abstractmethod
    def _targets(self, y: Target) -> tuple[torch.Tensor, int]:
        """O alvo pronto para a loss, e quantos neurónios de saída a rede precisa."""

    @abstractmethod
    def _loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """A loss de um lote."""

    @property
    @abstractmethod
    def output_activation(self) -> str:
        """O que a camada de saída faz aos números (para mostrar: softmax, linear)."""

    def _progress(self, inputs: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
        """Métricas parciais de cada época (a exatidão, na classificação)."""
        return {}

    @override
    def _fit(self, X: Features, y: Target | None, context: TrainingContext) -> None:
        if y is None:
            raise ModelError(f"'{self.info.display_name}' precisa do alvo para aprender.")
        params = self.hyperparameters
        generator = torch.Generator().manual_seed(self.seed)
        inputs = _float_tensor(X)
        targets, n_outputs = self._targets(y)
        train, validation = self._split(len(X), generator)
        self._network = self.build_network(X.shape[1], n_outputs, generator)
        self._sample = inputs[train[:SAMPLE_ROWS]]
        optimizer = self._optimizer()
        self._epochs, self._best_epoch, self._early_stopped = 0, None, False
        best_loss, best_weights, waiting = math.inf, None, 0
        first_loss: float | None = None
        for epoch in range(1, params.max_epochs + 1):
            loss = self._epoch(inputs, targets, train, optimizer, generator)
            first_loss = loss if first_loss is None else first_loss
            self._epochs = epoch
            metrics = self._progress(inputs[train], targets[train])
            validation_loss = None
            if validation is not None:
                validation_loss = self._validation_loss(inputs[validation], targets[validation])
                metrics["val_loss"] = validation_loss
            context.report_step(epoch, loss=loss, metrics=metrics)
            if context.wants_snapshot(epoch):
                context.report_snapshot(epoch, self.snapshot_state(context.probe))
            if has_diverged(loss, first_loss):
                self._note(diverged_notice(params.learning_rate))
                break
            if validation_loss is not None:
                if validation_loss < best_loss - MIN_IMPROVEMENT:
                    best_loss, waiting, self._best_epoch = validation_loss, 0, epoch
                    best_weights = copy.deepcopy(self._network.state_dict())
                else:
                    waiting += 1
                    if waiting >= params.patience:
                        self._early_stopped = True
                        break
            if context.should_stop():
                break
        if best_weights is not None:
            self._network.load_state_dict(best_weights)
        self._note_the_ending(validation is not None)

    def _split(
        self, n_rows: int, generator: torch.Generator
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """As linhas de treino e as de validação (para o early stopping), ao acaso com a seed."""
        params = self.hyperparameters
        order = torch.randperm(n_rows, generator=generator)
        n_validation = round(n_rows * params.validation_fraction)
        if not params.early_stopping or n_validation < 1 or n_rows - n_validation < 2:
            return order, None
        return order[n_validation:], order[:n_validation]

    def _optimizer(self) -> torch.optim.Optimizer:
        params = self.hyperparameters
        weights = self._network.parameters()
        if params.optimizer == "sgd":
            return torch.optim.SGD(
                weights, lr=params.learning_rate, momentum=0.9, weight_decay=params.alpha
            )
        return torch.optim.Adam(weights, lr=params.learning_rate, weight_decay=params.alpha)

    def _epoch(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        rows: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        generator: torch.Generator,
    ) -> float:
        """Uma passagem pelo treino, em lotes; devolve a loss média e guarda os gradientes."""
        self._network.train()
        order = rows[torch.randperm(len(rows), generator=generator)]
        size = self.hyperparameters.batch_size
        layers = self._linear_layers()
        gradients = [0.0] * len(layers)
        total, batches = 0.0, 0
        for start in range(0, len(order), size):
            batch = order[start : start + size]
            optimizer.zero_grad()
            loss = self._loss(self._network(inputs[batch]), targets[batch])
            torch.autograd.backward(loss)  # o mesmo que loss.backward(): a retropropagação
            for index, layer in enumerate(layers):
                if layer.weight.grad is not None:
                    gradients[index] += layer.weight.grad.norm().item()
            optimizer.step()
            total += loss.item() * len(batch)
            batches += 1
        # O tamanho do gradiente de cada camada, em média na época: é o que se fotografa.
        self._gradients = [gradient / batches for gradient in gradients]
        return total / len(order)

    def _validation_loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> float:
        with torch.no_grad():
            return self._loss(self._outputs(inputs), targets).item()

    def _outputs(self, inputs: torch.Tensor) -> torch.Tensor:
        """A saída da rede, sem calcular gradientes (para prever e para as métricas)."""
        self._network.eval()
        with torch.no_grad():
            outputs: torch.Tensor = self._network(inputs)
        return outputs

    def _forward(self, X: Features) -> torch.Tensor:
        return self._outputs(_float_tensor(X))

    def _linear_layers(self) -> list[nn.Linear]:
        return [module for module in self._network if isinstance(module, nn.Linear)]

    def _note_the_ending(self, validated: bool) -> None:
        """O aviso sobre como o treino acabou (parou cedo, ou fez as épocas todas)."""
        params = self.hyperparameters
        if self._early_stopped:
            self._note(
                Notice.info(
                    "early_stopping",
                    f"Parou na época {format_count(self._epochs)}: a loss na validação não "
                    f"melhorava há {format_count(params.patience)} épocas. A rede ficou com os "
                    f"pesos da época {format_count(self._best_epoch or self._epochs)}, a melhor "
                    "na validação.",
                )
            )
        elif self._epochs == params.max_epochs:
            ending = (
                f" Ficou com os pesos da época {format_count(self._best_epoch)}, a melhor na "
                "validação."
                if validated and self._best_epoch is not None
                else ""
            )
            self._note(
                Notice.info(
                    "max_epochs",
                    f"A rede treinou as {format_count(params.max_epochs)} épocas todas: se a loss "
                    f"ainda descia, mais épocas podem ajudar.{ending}",
                )
            )
        if params.early_stopping and not validated:
            self._note(
                Notice.info(
                    "no_validation",
                    "Com tão poucas linhas não sobra nenhuma para a validação: o early stopping "
                    "ficou desligado.",
                )
            )

    @override
    def state(self) -> dict[str, Any]:
        layers = self._linear_layers()
        if not layers:
            return {}
        sizes = [layers[0].in_features, *(layer.out_features for layer in layers)]
        hidden = self._hidden_reports()
        return {
            "architecture": " → ".join(str(size) for size in sizes),
            "activation": hidden[0]["activation"] if hidden else self.output_activation,
            "n_parameters": sum(parameter.numel() for parameter in self._network.parameters()),
            "iterations": self._epochs,
            "best_epoch": self._best_epoch,
            "early_stopped": self._early_stopped,
            "layers": [*hidden, self._output_report(layers[-1])],
        }

    def _hidden_reports(self) -> list[dict[str, Any]]:
        """Cada camada escondida: os neurónios, os parâmetros, o gradiente e as ativações."""
        modules = list(self._network)
        reports: list[dict[str, Any]] = []
        values = self._sample
        self._network.eval()
        with torch.no_grad():
            for number, (linear, activation) in enumerate(
                zip(modules[:-1:2], modules[1:-1:2], strict=True), start=1
            ):
                assert isinstance(linear, nn.Linear)  # a forma que build_network promete
                values = activation(linear(values))
                outputs = values.numpy().astype(np.float64)
                stuck = outputs.std(axis=0) < STUCK_STD
                counts, edges = np.histogram(outputs.ravel(), bins=HISTOGRAM_BINS)
                reports.append(
                    {
                        "name": f"Camada {number}",
                        "units": linear.out_features,
                        "activation": _ACTIVATION_LABELS.get(type(activation), "outra"),
                        "parameters": _parameters(linear),
                        "gradient": self._gradient(number - 1),
                        "activations": [float(value) for value in outputs.mean(axis=0)],
                        "stuck_units": [bool(flag) for flag in stuck],
                        "stuck": float(np.mean(stuck)),
                        "histogram": {
                            "edges": [float(edge) for edge in edges],
                            "counts": [int(count) for count in counts],
                        },
                    }
                )
        return reports

    def _output_report(self, layer: nn.Linear) -> dict[str, Any]:
        return {
            "name": "Saída",
            "units": layer.out_features,
            "activation": self.output_activation,
            "parameters": _parameters(layer),
            "gradient": self._gradient(len(self._linear_layers()) - 1),
        }

    def _gradient(self, index: int) -> float | None:
        return self._gradients[index] if index < len(self._gradients) else None


class TorchClassifier[P: TrainingParams](TorchModel[P]):
    """Base das redes que classificam: um neurónio de saída por classe, com softmax."""

    def __init__(self, hyperparameters: P | None = None, *, seed: int = 0) -> None:
        super().__init__(hyperparameters, seed=seed)
        self._classes = np.zeros(0, dtype=np.int64)

    @property
    @override
    def output_activation(self) -> str:
        return "softmax"

    @override
    def classes(self) -> NDArray[np.int64] | None:
        return self._classes if self._classes.size else None

    @override
    def _targets(self, y: Target) -> tuple[torch.Tensor, int]:
        self._classes = np.unique(y).astype(np.int64)
        if len(self._classes) < 2:
            raise ModelError(
                f"'{self.info.display_name}' precisa de pelo menos duas classes no treino."
            )
        codes = np.searchsorted(self._classes, y)
        return torch.as_tensor(codes, dtype=torch.long), len(self._classes)

    @override
    def _loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # A cross-entropy aplica o softmax por dentro: é a log-loss da regressão logística.
        return nn.functional.cross_entropy(outputs, targets)

    @override
    def _progress(self, inputs: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
        predicted = self._outputs(inputs).argmax(dim=1)
        return {"accuracy": float((predicted == targets).float().mean().item())}

    @override
    def _predict(self, X: Features) -> NDArray[Any]:
        predicted: NDArray[np.int64] = self._classes[self._forward(X).argmax(dim=1).numpy()]
        return predicted

    @override
    def _predict_proba(self, X: Features) -> NDArray[np.float64]:
        # O softmax em float64: em float32, as probabilidades somam 1 só a menos de 1e-7, e o
        # scikit-learn (na log-loss) avisa que não são probabilidades.
        probabilities: NDArray[np.float64] = torch.softmax(self._forward(X).double(), dim=1).numpy()
        return probabilities

    @override
    def state(self) -> dict[str, Any]:
        state = super().state()
        return {**state, "classes": self._classes.tolist()} if state else state


class TorchRegressor[P: TrainingParams](TorchModel[P]):
    """Base das redes que preveem um número: um neurónio de saída, sem ativação.

    O alvo é padronizado (média 0, desvio 1) antes de treinar: com valores grandes (preços
    de casas, por exemplo), os gradientes do início seriam enormes e o treino instável. As
    previsões voltam à escala original.
    """

    def __init__(self, hyperparameters: P | None = None, *, seed: int = 0) -> None:
        super().__init__(hyperparameters, seed=seed)
        self._mean, self._std = 0.0, 1.0

    @property
    @override
    def output_activation(self) -> str:
        return "linear"

    @override
    def _targets(self, y: Target) -> tuple[torch.Tensor, int]:
        values = np.asarray(y, dtype=np.float64)
        self._mean = float(values.mean())
        self._std = float(values.std()) or 1.0
        scaled = (values - self._mean) / self._std
        return torch.as_tensor(scaled, dtype=torch.float32).reshape(-1, 1), 1

    @override
    def _loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return nn.functional.mse_loss(outputs, targets)

    @override
    def _predict(self, X: Features) -> NDArray[Any]:
        scaled = self._forward(X)[:, 0].numpy().astype(np.float64)
        return scaled * self._std + self._mean

    @override
    def state(self) -> dict[str, Any]:
        state = super().state()
        return {**state, "target_mean": self._mean, "target_std": self._std} if state else state


def _parameters(layer: nn.Linear) -> int:
    return sum(parameter.numel() for parameter in layer.parameters())


def _float_tensor(values: Features) -> torch.Tensor:
    """As features num tensor de float32: sempre uma cópia, nunca a mesma memória.

    As matrizes do ``Dataset`` são só de leitura (para ninguém as mudar sem querer), e um
    tensor que partilhasse essa memória podia ser escrito por baixo dela: o PyTorch avisa.
    """
    return torch.from_numpy(np.array(values, dtype=np.float32))
