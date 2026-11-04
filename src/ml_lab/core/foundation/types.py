"""Tipos partilhados por todo o projeto: paradigmas, problemas, tipos de tarefa e partições.

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
        """Nome a mostrar na interface, em inglês."""
        return _PARADIGM_LABELS[self]


class Problem(StrEnum):
    """O problema que uma tarefa resolve: a pergunta que se faz aos dados.

    Várias tarefas podem responder à mesma pergunta: a classificação binária e a multiclasse
    são as duas uma classificação, e os mesmos modelos fazem as duas. É pelo problema que o
    ecrã Modelos arruma os modelos e lhes dá cor (ADR 0048).
    """

    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    DIMENSIONALITY_REDUCTION = "dimensionality_reduction"
    ANOMALY_DETECTION = "anomaly_detection"
    CONTROL = "control"

    @property
    def label(self) -> str:
        """Nome a mostrar na interface, em inglês."""
        return _PROBLEM_LABELS[self]


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
        """Nome a mostrar na interface, em inglês."""
        return _LABELS[self]

    @property
    def paradigm(self) -> Paradigm:
        """Paradigma a que o tipo de tarefa pertence."""
        return _PARADIGMS[self]

    @property
    def problem(self) -> Problem:
        """O problema que a tarefa resolve (as duas classificações são o mesmo)."""
        return _PROBLEMS[self]

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
        """Nome a mostrar na interface, em inglês."""
        return _PARTITION_LABELS[self]


_PARTITION_LABELS: Final[Mapping[Partition, str]] = MappingProxyType(
    {Partition.TRAIN: "Training", Partition.VALIDATION: "Validation", Partition.TEST: "Test"}
)

_PARADIGM_LABELS: Final[Mapping[Paradigm, str]] = MappingProxyType(
    {
        Paradigm.SUPERVISED: "Supervised",
        Paradigm.UNSUPERVISED: "Unsupervised",
        Paradigm.REINFORCEMENT: "Reinforcement learning",
    }
)

_LABELS: Final[Mapping[TaskType, str]] = MappingProxyType(
    {
        TaskType.BINARY_CLASSIFICATION: "Binary classification",
        TaskType.MULTICLASS_CLASSIFICATION: "Multiclass classification",
        TaskType.REGRESSION: "Regression",
        TaskType.CLUSTERING: "Clustering",
        TaskType.DIMENSIONALITY_REDUCTION: "Dimensionality reduction",
        TaskType.ANOMALY_DETECTION: "Anomaly detection",
        TaskType.CONTROL: "Control",
    }
)

_PROBLEM_LABELS: Final[Mapping[Problem, str]] = MappingProxyType(
    {
        Problem.CLASSIFICATION: "Classification",
        Problem.REGRESSION: "Regression",
        Problem.CLUSTERING: "Clustering",
        Problem.DIMENSIONALITY_REDUCTION: "Dimensionality reduction",
        Problem.ANOMALY_DETECTION: "Anomaly detection",
        Problem.CONTROL: "Control",
    }
)

_PROBLEMS: Final[Mapping[TaskType, Problem]] = MappingProxyType(
    {
        TaskType.BINARY_CLASSIFICATION: Problem.CLASSIFICATION,
        TaskType.MULTICLASS_CLASSIFICATION: Problem.CLASSIFICATION,
        TaskType.REGRESSION: Problem.REGRESSION,
        TaskType.CLUSTERING: Problem.CLUSTERING,
        TaskType.DIMENSIONALITY_REDUCTION: Problem.DIMENSIONALITY_REDUCTION,
        TaskType.ANOMALY_DETECTION: Problem.ANOMALY_DETECTION,
        TaskType.CONTROL: Problem.CONTROL,
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
