"""Dataset: o objeto que sai da camada de dados (arrays NumPy + metadados).

É tudo o que o resto do sistema recebe, venham os dados de onde vierem: as
features de cada parte (treino, validação, teste) em matrizes ``float64``, o alvo,
os nomes das features (para a explicabilidade e a interface), o esquema das
colunas, a linhagem de tudo o que foi feito aos dados e a projeção 2D para os
gráficos. Não tem nada do pandas (ADR 0002).
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from ml_lab.core.data.lineage import Lineage
from ml_lab.core.data.projection.projection_2d import Projection
from ml_lab.core.data.schema import TableSchema
from ml_lab.core.foundation.types import Partition, TaskType

if TYPE_CHECKING:
    from ml_lab.core.data.serving import FittedPreparation

type Features = NDArray[np.float64]
"""Matriz ``(n_linhas, n_features)`` de ``float64``."""

type Target = NDArray[np.float64] | NDArray[np.int64]
"""Alvo: ``float64`` na regressão; na classificação, o número da classe (``int64``)."""


@dataclass(frozen=True, slots=True, eq=False)
class DataSplit:
    """Uma parte dos dados.

    Attributes:
        X: As features, ``(n_linhas, n_features)``.
        y: O alvo, ``(n_linhas,)``, ou ``None`` sem alvo (aprendizagem não supervisionada).
        rows: O número de cada linha na tabela original, para a reencontrar.
    """

    X: Features
    y: Target | None
    rows: NDArray[np.intp]

    def __len__(self) -> int:
        return int(self.X.shape[0])


@dataclass(frozen=True, slots=True, eq=False)
class Dataset:
    """Os dados prontos para os modelos.

    Attributes:
        name: Nome do dataset, para mostrar.
        task_type: O tipo de tarefa.
        feature_names: Nome de cada coluna de ``X``, pela ordem.
        target_name: A coluna-alvo, ou ``None``.
        class_labels: Na classificação, o nome de cada classe: a classe ``k`` do alvo é
            ``class_labels[k]``. Vazio nas outras tarefas.
        splits: As partes dos dados (treino e teste sempre; validação se foi pedida).
        schema: O esquema das colunas antes das transformações.
        lineage: Todas as operações aplicadas, da entrada à saída.
        projection: As coordenadas 2D de cada parte, para os gráficos.
        preparation: A preparação já treinada, que transforma linhas novas como estas foram
            transformadas (é o que vai para o pacote de um modelo, ADR 0039).
    """

    name: str
    task_type: TaskType
    feature_names: tuple[str, ...]
    target_name: str | None
    class_labels: tuple[str, ...]
    splits: Mapping[Partition, DataSplit]
    schema: TableSchema
    lineage: Lineage
    projection: Projection | None = None
    preparation: "FittedPreparation | None" = None

    @property
    def train(self) -> DataSplit:
        """As linhas de treino."""
        return self.splits[Partition.TRAIN]

    @property
    def test(self) -> DataSplit:
        """As linhas de teste."""
        return self.splits[Partition.TEST]

    @property
    def validation(self) -> DataSplit | None:
        """As linhas de validação, se a divisão as criou."""
        return self.splits.get(Partition.VALIDATION)

    @property
    def n_features(self) -> int:
        """Número de features."""
        return len(self.feature_names)

    @property
    def n_classes(self) -> int:
        """Número de classes (0 se não for classificação)."""
        return len(self.class_labels)
