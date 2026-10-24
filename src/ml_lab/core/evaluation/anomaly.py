"""Avaliador de deteção de anomalias: quantos pontos foram marcados e, com referência, se acertou.

Se a preparação tiver uma coluna de referência com duas classes, a mais rara conta como
"anomalia" (é o que as anomalias são: raras), e a preparação já lhe deu o número 1. Ela
não entra no treino, mas diz se o detetor encontrou as anomalias verdadeiras.
"""

from typing import Any, Final, override

import numpy as np

from ml_lab.core.evaluation.base import EVALUATORS, Evaluator, measure, undefined_notices
from ml_lab.core.evaluation.metrics.base import MetricInputs
from ml_lab.core.evaluation.report import ConfusionMatrix, EvaluationReport
from ml_lab.core.foundation.notices import Notice
from ml_lab.core.foundation.types import Partition, TaskType
from ml_lab.core.models.base import Features, LabModel, Target

DISPLAY_ORDER: Final = (
    "anomaly_rate",
    "anomaly_roc_auc",
    "anomaly_precision",
    "anomaly_recall",
)

FLAG_LABELS: Final = ("Normal", "Anomalia")


@EVALUATORS.register("anomaly")
class AnomalyEvaluator(Evaluator):
    """Avalia detetores de anomalias."""

    task_types = frozenset({TaskType.ANOMALY_DETECTION})

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
        flags = np.asarray(model.predict(X), dtype=np.int64)
        scores = np.asarray(model.anomaly_scores(X), dtype=np.float64)
        notices: list[Notice] = []
        truth = None
        matrix = None
        if y is not None and len(class_labels) == 2:
            truth = y.astype(np.int64)  # a preparação dá o 1 à classe mais rara: a anomalia
            notices.append(
                Notice.info(
                    "reference",
                    f"A coluna de referência não entrou no treino; serve para comparar, com "
                    f"«{class_labels[1]}» (a classe mais rara) como anomalia.",
                )
            )
            counts = tuple(
                tuple(int(np.sum((truth == real) & (flags == flag))) for flag in (0, 1))
                for real in (0, 1)
            )
            matrix = ConfusionMatrix(FLAG_LABELS, counts)
        elif y is not None:
            notices.append(
                Notice.info(
                    "reference_not_binary",
                    "A coluna de referência tem mais de duas classes: para dizer quais são as "
                    "anomalias verdadeiras, precisa de duas (normal e anomalia).",
                )
            )
        inputs = MetricInputs(y_true=truth, y_pred=flags, scores=scores)
        results, undefined = measure(task_type, inputs, order=DISPLAY_ORDER)
        notices += undefined_notices(
            undefined, partition, "faltam anomalias verdadeiras, ou pontos marcados"
        )
        return EvaluationReport(
            partition=partition,
            task_type=task_type,
            n_samples=len(flags),
            metrics=results,
            confusion_matrix=matrix,
            notices=tuple(notices),
        )
