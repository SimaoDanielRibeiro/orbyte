"""Os números de que as previsões de alguns modelos dependem, em matrizes do NumPy.

Servem para levar um modelo para fora do laboratório (o ONNX, em ``core/export``) sem depender
de como ele os guarda por dentro: o PyTorch fica nas redes, e as linhas que um modelo memoriza
ficam com ele. Os modelos lineares e os de centróides não precisam disto: o que aprenderam já
está todo no ``state()``.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

type DenseActivation = Literal["relu", "tanh", "sigmoid"]


@dataclass(frozen=True, slots=True, eq=False)
class DenseLayer:
    """Uma camada densa de uma rede: ``ativação(x · pesosᵀ + ordenada)``.

    Attributes:
        weights: Os pesos, ``(saídas, entradas)``, em ``float32`` (como a rede os usa).
        bias: A ordenada de cada saída.
        activation: A ativação depois da soma (``None`` na camada de saída).
    """

    weights: NDArray[np.float32]
    bias: NDArray[np.float32]
    activation: DenseActivation | None


@dataclass(frozen=True, slots=True, eq=False)
class Memory:
    """As linhas que um modelo guarda para prever, e o que cada uma é.

    Attributes:
        rows: As linhas, já com as features preparadas.
        labels: A classe ou o grupo de cada linha.
    """

    rows: NDArray[np.float64]
    labels: NDArray[np.int64]
