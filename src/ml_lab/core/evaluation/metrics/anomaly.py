"""Métricas de deteção de anomalias.

Sem respostas certas, só se pode descrever quantos pontos foram marcados. Com uma coluna
de referência que diga quais são as anomalias verdadeiras (1 = anomalia), mede-se se o
detetor as encontrou: a precisão, a recall e a ROC AUC da pontuação.
"""

import math

import numpy as np
from sklearn import metrics as sk

from ml_lab.core.evaluation.metrics.base import MetricInputs, metric
from ml_lab.core.foundation.types import TaskType

ANOMALY = (TaskType.ANOMALY_DETECTION,)


@metric(
    "anomaly_rate",
    display_name="Anomalias marcadas",
    task_types=ANOMALY,
    higher_is_better=None,
    needs=("y_pred",),
    as_percent=True,
)
def anomaly_rate(inputs: MetricInputs) -> float:
    """A proporção de pontos marcados como anomalias."""
    flags = np.asarray(inputs.predicted)
    return float(np.mean(flags == 1)) if len(flags) else math.nan


@metric(
    "anomaly_precision",
    display_name="Precisão das anomalias",
    task_types=ANOMALY,
    higher_is_better=True,
    as_percent=True,
)
def anomaly_precision(inputs: MetricInputs) -> float:
    """Dos pontos marcados como anomalias, quantos o eram de facto (indefinida sem marcados)."""
    flags = np.asarray(inputs.predicted)
    if not np.any(flags == 1):
        return math.nan
    return float(sk.precision_score(inputs.truth, flags))


@metric(
    "anomaly_recall",
    display_name="Recall das anomalias",
    task_types=ANOMALY,
    higher_is_better=True,
    as_percent=True,
)
def anomaly_recall(inputs: MetricInputs) -> float:
    """Das anomalias verdadeiras, quantas foram marcadas (indefinida sem anomalias)."""
    truth = np.asarray(inputs.truth)
    if not np.any(truth == 1):
        return math.nan
    return float(sk.recall_score(truth, inputs.predicted))


@metric(
    "anomaly_roc_auc",
    display_name="ROC AUC da pontuação",
    task_types=ANOMALY,
    higher_is_better=True,
    needs=("y_true", "scores"),
)
def anomaly_roc_auc(inputs: MetricInputs) -> float:
    """A probabilidade de uma anomalia verdadeira ter mais pontuação do que um ponto normal."""
    truth = np.asarray(inputs.truth)
    if inputs.scores is None or len(np.unique(truth)) < 2:
        return math.nan
    return float(sk.roc_auc_score(truth, inputs.scores))
