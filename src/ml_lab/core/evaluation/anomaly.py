"""Avaliador de deteção de anomalias: quantos pontos foram marcados e, com referência, se acertou.

Se a preparação tiver uma coluna de referência com duas classes, a mais rara conta como
"anomalia" (é o que as anomalias são: raras), e a preparação já lhe deu o número 1. Ela
não entra no treino, mas diz se o detetor encontrou as anomalias verdadeiras.

Como nas outras tarefas, um detetor que não serve leva um aviso: o que não marcou nada (o
limiar ficou alto de mais para estes dados) e o que, com referência, não apanhou nenhuma
anomalia verdadeira.
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
from ml_lab.formatting import format_count

DISPLAY_ORDER: Final = (
    "anomaly_rate",
    "anomaly_roc_auc",
    "anomaly_precision",
    "anomaly_recall",
)

FLAG_LABELS: Final = ("Normal", "Anomaly")


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
                    f"The reference column didn't go into the training; it's there to compare "
                    f"with, with “{class_labels[1]}” (the rarer class) as the anomaly.",
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
                    "The reference column has more than two classes: to tell which are the true "
                    "anomalies, it needs two (normal and anomaly).",
                )
            )
        inputs = MetricInputs(y_true=truth, y_pred=flags, scores=scores)
        results, undefined = measure(task_type, inputs, order=DISPLAY_ORDER)
        notices += _useless_detector_notices(flags, truth, partition)
        notices += undefined_notices(
            undefined, partition, "there are no true anomalies, or no flagged points"
        )
        return EvaluationReport(
            partition=partition,
            task_type=task_type,
            n_samples=len(flags),
            metrics=results,
            confusion_matrix=matrix,
            notices=tuple(notices),
        )


def _useless_detector_notices(
    flags: NDArray[np.int64], truth: NDArray[np.int64] | None, partition: Partition
) -> list[Notice]:
    """O aviso de um detetor que não serve nesta parte dos dados (se for o caso).

    Args:
        flags: O que o detetor marcou (1 para anomalia).
        truth: As anomalias verdadeiras, da coluna de referência (``None`` sem ela).
        partition: A parte avaliada.
    """
    where = f"on the {partition.label.lower()} data"
    real = int(truth.sum()) if truth is not None else 0
    missed = {1: "the one real anomaly", 2: "both real anomalies"}.get(
        real, f"all {format_count(real)} real anomalies"
    )
    nothing = f"It flagged none of the {format_count(len(flags))} rows {where}"
    raise_it = "raise the expected share of anomalies (Nu, in the One-Class SVM)"
    if len(flags) > 0 and not flags.any():
        if truth is None:  # sem referência, não se sabe se havia o que marcar
            return [
                Notice.info(
                    "nothing_flagged",
                    f"{nothing}: with this threshold, nothing looked unusual enough. If you "
                    f"expected anomalies here, {raise_it}.",
                )
            ]
        if real > 0:  # se a referência não tem anomalias aqui, não marcar nada está certo
            return [
                Notice.warning(
                    "nothing_flagged",
                    f"{nothing}, so it missed {missed}: with this threshold, nothing looked "
                    f"unusual enough. {raise_it.capitalize()} and train again.",
                )
            ]
        return []
    if truth is not None and real > 0 and not np.any((truth == 1) & (flags == 1)):
        flagged = int(flags.sum())
        wrong = (
            f"The one row it flagged {where} isn't a real anomaly"
            if flagged == 1
            else f"None of the {format_count(flagged)} rows it flagged {where} is a real anomaly"
        )
        return [
            Notice.warning(
                "no_anomaly_caught",
                f"{wrong}, and it missed {missed}: its scores don't single out the anomalies in "
                "this data. Try another detector, or check the features and the preparation.",
            )
        ]
    return []
