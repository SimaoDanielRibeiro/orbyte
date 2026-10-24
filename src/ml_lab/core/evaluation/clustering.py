"""Avaliador de clustering: a forma dos grupos e, se houver, a comparação com uma referência.

Sem alvo não há respostas certas: mede-se se os grupos são compactos e separados. Se a
preparação tiver uma coluna de referência (uma coluna-alvo escolhida com a tarefa
clustering), ela não entra no treino, mas serve para comparar: a tabela cruza as
classes da referência com os grupos encontrados, e o Rand ajustado resume o acordo.
"""

from typing import Any, Final, override

import numpy as np
from numpy.typing import NDArray

from ml_lab.core.evaluation.base import EVALUATORS, Evaluator, measure, undefined_notices
from ml_lab.core.evaluation.metrics.base import MetricInputs
from ml_lab.core.evaluation.report import ConfusionMatrix, EvaluationReport
from ml_lab.core.foundation.notices import Notice
from ml_lab.core.foundation.types import Partition, TaskType
from ml_lab.core.models.base import Features, LabModel, Target
from ml_lab.formatting import format_percent

DISPLAY_ORDER: Final = (
    "silhouette",
    "n_clusters",
    "adjusted_rand",
    "davies_bouldin",
    "calinski_harabasz",
    "noise_ratio",
)

NOISE_LABEL: Final = "Ruído"
"""O nome dos pontos que não ficaram em nenhum grupo."""


def group_name(group: int) -> str:
    """O nome de um grupo: "Grupo 1" para o grupo 0, e o ruído (-1) pelo seu nome."""
    return NOISE_LABEL if group < 0 else f"Grupo {group + 1}"


def group_names(labels: NDArray[np.int64]) -> tuple[str, ...]:
    """O nome de cada grupo, pela ordem dos números (o ruído, se houver, no fim)."""
    groups = sorted(set(labels.tolist()) - {-1})
    names = tuple(group_name(group) for group in groups)
    return (*names, NOISE_LABEL) if -1 in labels else names


@EVALUATORS.register("clustering")
class ClusteringEvaluator(Evaluator):
    """Avalia modelos de clustering."""

    task_types = frozenset({TaskType.CLUSTERING})

    @override
    def evaluate(
        self,
        model: LabModel[Any],
        X: Features,
        y: Target | None,
        *,
        partition: Partition,
        task_type: TaskType,
        class_labels: tuple[str, ...] = (),
    ) -> EvaluationReport:
        labels = np.asarray(model.predict(X), dtype=np.int64)
        reference = None if y is None else y.astype(np.int64)
        inputs = MetricInputs(y_true=reference, y_pred=labels, X=X)
        results, undefined = measure(task_type, inputs, order=DISPLAY_ORDER)
        where = f"no {partition.label.lower()}"
        notices: list[Notice] = []
        groups = set(labels.tolist()) - {-1}
        noise = float(np.mean(labels == -1))
        if len(groups) < 2:
            notices.append(
                Notice.warning(
                    "single_cluster",
                    f"Ficou tudo num só grupo {where} (ou em nenhum): não há estrutura para "
                    "medir. No DBSCAN, muda o raio; nos outros, confirma o número de grupos.",
                )
            )
        if noise > 0.5:
            notices.append(
                Notice.warning(
                    "mostly_noise",
                    f"{format_percent(noise)} dos pontos ficaram como ruído {where}: o raio "
                    "(eps) é pequeno demais para a densidade destes dados.",
                )
            )
        notices += undefined_notices(
            undefined, partition, "precisa de pelo menos dois grupos com pontos"
        )
        matrix = None
        if reference is not None and class_labels:
            matrix = _contingency(reference, labels, class_labels)
        return EvaluationReport(
            partition=partition,
            task_type=task_type,
            n_samples=len(labels),
            metrics=results,
            confusion_matrix=matrix,
            notices=tuple(notices),
        )


def _contingency(
    reference: NDArray[np.int64], labels: NDArray[np.int64], class_labels: tuple[str, ...]
) -> ConfusionMatrix:
    """Quantos pontos de cada classe da referência caíram em cada grupo."""
    groups = sorted(set(labels.tolist()) - {-1}) + ([-1] if -1 in labels else [])
    counts = tuple(
        tuple(int(np.sum((reference == code) & (labels == group))) for group in groups)
        for code in range(len(class_labels))
    )
    return ConfusionMatrix(class_labels, counts, group_names(labels))
