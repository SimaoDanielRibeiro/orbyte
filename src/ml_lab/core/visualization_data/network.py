"""A rede ao longo do treino: o que cada camada faz em cada fotografia (treino passo a passo).

Cada fotografia de uma rede neuronal (``state["layers"]``, do adaptador do PyTorch) diz,
para cada camada, o tamanho do gradiente dos seus pesos e, nas camadas escondidas, a
ativação média de cada neurónio e quantos estão parados. Aqui juntam-se as fotografias
numa sequência pronta a desenhar, e é nela que se leem dois problemas clássicos:

- um neurónio **parado** dá sempre o mesmo valor, seja qual for o exemplo (uma ReLU que
  só dá zero, uma tanh sempre em ±1). Não ajuda a distinguir nada e, como o gradiente que
  passa por ele é zero ou quase, tende a ficar assim;
- um gradiente que encolhe de camada para camada, até à primeira, é o **gradiente que
  desaparece**: as primeiras camadas quase não aprendem (acontece com a sigmoide em redes
  fundas).
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class LayerInfo:
    """Uma camada da rede.

    Attributes:
        name: "Camada 1", "Camada 2"… ou "Saída".
        units: Quantos neurónios tem.
        activation: A ativação (ReLU, tanh, sigmoide; softmax ou linear na saída).
        parameters: Quantos pesos e ordenadas tem.
        hidden: Se é uma camada escondida (as que têm as ativações fotografadas).
    """

    name: str
    units: int
    activation: str
    parameters: int
    hidden: bool


@dataclass(frozen=True, slots=True)
class LayerFrame:
    """Uma camada numa fotografia.

    Attributes:
        activations: A ativação média de cada neurónio (vazio na saída).
        stuck_units: Se cada neurónio está parado (vazio na saída).
        stuck: A proporção de neurónios parados (``None`` na saída).
        gradient: O tamanho (norma) do gradiente dos pesos, em média na época.
        edges: Os limites dos intervalos do histograma das ativações.
        counts: Quantas ativações caíram em cada intervalo.
    """

    activations: tuple[float, ...] = ()
    stuck_units: tuple[bool, ...] = ()
    stuck: float | None = None
    gradient: float | None = None
    edges: tuple[float, ...] = ()
    counts: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class NetworkFrame:
    """Todas as camadas num passo do treino.

    Attributes:
        step: O passo do treino.
        layers: As camadas, pela ordem de ``NetworkTrace.layers``.
        stuck: A proporção de neurónios parados, contados todos os das camadas escondidas.
    """

    step: int
    layers: tuple[LayerFrame, ...]
    stuck: float


@dataclass(frozen=True, slots=True)
class NetworkTrace:
    """A rede ao longo do treino.

    Attributes:
        inputs: Quantas features entram na rede.
        layers: As camadas, da primeira escondida à saída.
        frames: Uma fotografia por snapshot, por ordem de passo.
    """

    inputs: int
    layers: tuple[LayerInfo, ...]
    frames: tuple[NetworkFrame, ...]


def network_trace(
    inputs: int, snapshots: Sequence[tuple[int, Mapping[str, Any]]]
) -> NetworkTrace | None:
    """A rede em cada fotografia, ou ``None`` se o modelo não for uma rede neuronal.

    Args:
        inputs: Quantas features entram na rede.
        snapshots: Os snapshots do treino, ``(passo, estado)``; com o mesmo passo repetido,
            fica o último.
    """
    by_step = {step: state["layers"] for step, state in snapshots if state.get("layers")}
    if not by_step:
        return None
    last = by_step[max(by_step)]
    layers = tuple(
        LayerInfo(
            name=str(layer["name"]),
            units=int(layer["units"]),
            activation=str(layer["activation"]),
            parameters=int(layer["parameters"]),
            hidden="activations" in layer,
        )
        for layer in last
    )
    frames = tuple(_network_frame(step, by_step[step]) for step in sorted(by_step))
    return NetworkTrace(inputs, layers, frames)


def _network_frame(step: int, layers: Sequence[Mapping[str, Any]]) -> NetworkFrame:
    frames = tuple(_frame(layer) for layer in layers)
    # Por neurónio, não pela média das camadas: assim uma camada pequena não pesa o mesmo
    # que uma grande.
    units = [flag for frame in frames for flag in frame.stuck_units]
    return NetworkFrame(step, frames, stuck=sum(units) / len(units) if units else 0.0)


def _frame(layer: Mapping[str, Any]) -> LayerFrame:
    histogram = layer.get("histogram") or {}
    gradient = layer.get("gradient")
    stuck = layer.get("stuck")
    return LayerFrame(
        activations=tuple(float(value) for value in layer.get("activations", ())),
        stuck_units=tuple(bool(flag) for flag in layer.get("stuck_units", ())),
        stuck=None if stuck is None else float(stuck),
        gradient=None if gradient is None else float(gradient),
        edges=tuple(float(edge) for edge in histogram.get("edges", ())),
        counts=tuple(int(count) for count in histogram.get("counts", ())),
    )
