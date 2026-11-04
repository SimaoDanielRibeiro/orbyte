"""Avaliador de classificação: métricas, matriz de confusão e comparação com o acaso.

Além das métricas, compara o modelo com a referência mais simples possível: prever
sempre a classe mais frequente. Um modelo que não bate essa referência não aprendeu
nada de útil, por mais alta que pareça a exatidão.
"""

from typing import Any, Final, override

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import confusion_matrix

from ml_lab.core.evaluation.base import EVALUATORS, Evaluator, measure, undefined_notices
from ml_lab.core.evaluation.metrics.base import MetricInputs
from ml_lab.core.evaluation.report import ConfusionMatrix, EvaluationReport
from ml_lab.core.foundation.errors import ModelError
from ml_lab.core.foundation.notices import Notice
from ml_lab.core.foundation.types import Partition, TaskType
from ml_lab.core.models.base import Features, LabModel, Target, class_probabilities
from ml_lab.formatting import format_percent

DISPLAY_ORDER: Final = (
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "log_loss",
)
"""Ordem dos cartões de métricas; as métricas novas aparecem depois, por ordem alfabética."""

_BASELINE_MARGIN: Final = 0.01


@EVALUATORS.register("classification")
class ClassificationEvaluator(Evaluator):
    """Avalia classificadores (binários e multiclasse)."""

    task_types = frozenset({TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION})

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
        if y is None or not class_labels:
            raise ModelError("Evaluating a classification needs the target and the classes.")
        n_classes = len(class_labels)
        y_true = y.astype(np.int64)
        y_pred = np.asarray(model.predict(X), dtype=np.int64)
        y_proba = class_probabilities(model, X, n_classes) if model.info.supports_proba else None
        inputs = MetricInputs(y_true, y_pred, y_proba, n_classes)

        results, undefined = measure(task_type, inputs, order=DISPLAY_ORDER)
        counts = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
        matrix = ConfusionMatrix(class_labels, tuple(tuple(int(c) for c in row) for row in counts))
        notices = [
            _baseline_notice(y_true, y_pred, class_labels, partition),
            *undefined_notices(undefined, partition, "at least one class is missing"),
        ]
        return EvaluationReport(
            partition=partition,
            task_type=task_type,
            n_samples=len(y_true),
            metrics=tuple(results),
            confusion_matrix=matrix,
            notices=tuple(notices),
        )


def _baseline_notice(
    y_true: NDArray[np.int64],
    y_pred: NDArray[np.int64],
    class_labels: tuple[str, ...],
    partition: Partition,
) -> Notice:
    counts = np.bincount(y_true, minlength=len(class_labels))
    majority = int(counts.argmax())
    baseline = float(counts[majority] / len(y_true))
    accuracy = float((y_true == y_pred).mean())
    where = f"on the {partition.label.lower()} data"
    if accuracy <= baseline + _BASELINE_MARGIN:
        return Notice.warning(
            "below_baseline",
            f"Always predicting “{class_labels[majority]}” would get {format_percent(baseline)} "
            f"right {where}, and the model gets {format_percent(accuracy)}: it learned nothing "
            "that helps. Check the features and the preparation.",
        )
    return Notice.info(
        "baseline",
        f"For comparison: always predicting “{class_labels[majority]}” would get "
        f"{format_percent(baseline)} right {where}; the model gets {format_percent(accuracy)}.",
    )
