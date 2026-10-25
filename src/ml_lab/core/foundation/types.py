"""Tipos partilhados por todo o projeto: paradigmas, tipos de tarefa e partições dos dados.

O tipo de tarefa (``TaskType``) organiza o ML Lab inteiro: decide que algoritmos
aparecem no catálogo, que métricas se calculam e que snapshots fazem sentido
durante o treino.
"""

from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType
from typing import Final


class Paradigm(StrEnum):
    """Paradigma de aprendizagem: de onde vem o que o modelo aprende.

    - ``SUPERVISED``: aprende a prever um alvo a partir de exemplos já rotulados.
    - ``UNSUPERVISED``: procura estrutura em dados sem alvo (grupos, direções, anomalias).
    - ``REINFORCEMENT``: não há dataset; um agente aprende a agir num ambiente, pelas
      recompensas que recebe (fase 8).

    Deep learning não é um paradigma: uma rede neuronal que classifica faz
    aprendizagem supervisionada; o que muda é a família do modelo.
    """

    SUPERVISED = "supervised"
    UNSUPERVISED = "unsupervised"
    REINFORCEMENT = "reinforcement"

    @property
    def label(self) -> str:
        """Nome a mostrar na interface, em português."""
        return _PARADIGM_LABELS[self]


class TaskType(StrEnum):
    """Lista fechada dos tipos de tarefa que o ML Lab sabe resolver.

    Os valores são estáveis porque ficam guardados em JSON e na base de dados.
    Para acrescentar um tipo novo, segue "Adicionar um tipo de tarefa" em
    docs/architecture.md.
    """

    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    DIMENSIONALITY_REDUCTION = "dimensionality_reduction"
    ANOMALY_DETECTION = "anomaly_detection"
    CONTROL = "control"

    @property
    def label(self) -> str:
        """Nome a mostrar na interface, em português."""
        return _LABELS[self]

    @property
    def paradigm(self) -> Paradigm:
        """Paradigma a que o tipo de tarefa pertence."""
        return _PARADIGMS[self]

    @property
    def requires_target(self) -> bool:
        """Se a tarefa precisa de uma coluna-alvo (só as supervisionadas precisam)."""
        return self.paradigm is Paradigm.SUPERVISED

    @property
    def learns_from_data(self) -> bool:
        """Se a tarefa aprende com um dataset (todas, menos o controlo: aprende num ambiente)."""
        return self.paradigm is not Paradigm.REINFORCEMENT

    @property
    def is_classification(self) -> bool:
        """Se a tarefa é de classificação, binária ou multiclasse."""
        return self in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION)


class Partition(StrEnum):
    """Parte dos dados depois da divisão.

    O modelo aprende com o treino; a validação serve para afinar escolhas (por
    exemplo, quando parar o treino); o teste só se usa no fim, para medir o
    desempenho em dados que o modelo nunca viu.
    """

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"

    @property
    def label(self) -> str:
        """Nome a mostrar na interface, em português."""
        return _PARTITION_LABELS[self]


_PARTITION_LABELS: Final[Mapping[Partition, str]] = MappingProxyType(
    {Partition.TRAIN: "Treino", Partition.VALIDATION: "Validação", Partition.TEST: "Teste"}
)

_PARADIGM_LABELS: Final[Mapping[Paradigm, str]] = MappingProxyType(
    {
        Paradigm.SUPERVISED: "Supervisionada",
        Paradigm.UNSUPERVISED: "Não supervisionada",
        Paradigm.REINFORCEMENT: "Reinforcement learning",
    }
)

_LABELS: Final[Mapping[TaskType, str]] = MappingProxyType(
    {
        TaskType.BINARY_CLASSIFICATION: "Classificação binária",
        TaskType.MULTICLASS_CLASSIFICATION: "Classificação multiclasse",
        TaskType.REGRESSION: "Regressão",
        TaskType.CLUSTERING: "Clustering",
        TaskType.DIMENSIONALITY_REDUCTION: "Redução de dimensionalidade",
        TaskType.ANOMALY_DETECTION: "Deteção de anomalias",
        TaskType.CONTROL: "Controlo",
    }
)

_PARADIGMS: Final[Mapping[TaskType, Paradigm]] = MappingProxyType(
    {
        TaskType.BINARY_CLASSIFICATION: Paradigm.SUPERVISED,
        TaskType.MULTICLASS_CLASSIFICATION: Paradigm.SUPERVISED,
        TaskType.REGRESSION: Paradigm.SUPERVISED,
        TaskType.CLUSTERING: Paradigm.UNSUPERVISED,
        TaskType.DIMENSIONALITY_REDUCTION: Paradigm.UNSUPERVISED,
        TaskType.ANOMALY_DETECTION: Paradigm.UNSUPERVISED,
        TaskType.CONTROL: Paradigm.REINFORCEMENT,
    }
)
