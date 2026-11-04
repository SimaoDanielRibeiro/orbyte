"""Métricas de deteção de anomalias.

Sem respostas certas, só se pode descrever quantos pontos foram marcados. Com uma coluna
de referência que diga quais são as anomalias verdadeiras (1 = anomalia), mede-se se o
detetor as encontrou: a precisão, a recall e a ROC AUC da pontuação.
"""

import math

import numpy as np
from sklearn import metrics as sk

from ml_lab.core.evaluation.metrics.base import MetricInputs, MetricTerm, metric
from ml_lab.core.foundation.types import TaskType

ANOMALY = (TaskType.ANOMALY_DETECTION,)


# --- os termos das fórmulas ----------------------------------------------------------------


def _counts(inputs: MetricInputs) -> tuple[int, int, int]:
    """As anomalias verdadeiras marcadas, os normais marcados e as anomalias por marcar."""
    flagged, anomalous = np.asarray(inputs.predicted) == 1, np.asarray(inputs.truth) == 1
    return (
        int(np.sum(flagged & anomalous)),
        int(np.sum(flagged & ~anomalous)),
        int(np.sum(~flagged & anomalous)),
    )


def _rate_terms(inputs: MetricInputs) -> tuple[MetricTerm, ...]:
    flags = np.asarray(inputs.predicted)
    return (
        MetricTerm(r"n_{\text{flagged}}", "Points flagged", float(np.sum(flags == 1))),
        MetricTerm("n", "Points evaluated", float(len(flags))),
    )


def _precision_terms(inputs: MetricInputs) -> tuple[MetricTerm, ...]:
    true_positives, false_positives, _ = _counts(inputs)
    return (
        MetricTerm(r"\mathrm{TP}", "True anomalies flagged", true_positives),
        MetricTerm(r"\mathrm{FP}", "Normal points flagged", false_positives),
    )


def _recall_terms(inputs: MetricInputs) -> tuple[MetricTerm, ...]:
    true_positives, _, false_negatives = _counts(inputs)
    return (
        MetricTerm(r"\mathrm{TP}", "True anomalies flagged", true_positives),
        MetricTerm(r"\mathrm{FN}", "Anomalies left unflagged", false_negatives),
    )


# --- as métricas ---------------------------------------------------------------------------


@metric(
    "anomaly_rate",
    display_name="Anomalies flagged",
    task_types=ANOMALY,
    higher_is_better=None,
    needs=("y_pred",),
    as_percent=True,
    breakdown=_rate_terms,
)
def anomaly_rate(inputs: MetricInputs) -> float:
    """A proporção de pontos marcados como anomalias."""
    flags = np.asarray(inputs.predicted)
    return float(np.mean(flags == 1)) if len(flags) else math.nan


@metric(
    "anomaly_precision",
    display_name="Anomaly precision",
    task_types=ANOMALY,
    higher_is_better=True,
    as_percent=True,
    breakdown=_precision_terms,
)
def anomaly_precision(inputs: MetricInputs) -> float:
    """Dos pontos marcados como anomalias, quantos o eram de facto (indefinida sem marcados)."""
    flags = np.asarray(inputs.predicted)
    if not np.any(flags == 1):
        return math.nan
    return float(sk.precision_score(inputs.truth, flags))


@metric(
    "anomaly_recall",
    display_name="Anomaly recall",
    task_types=ANOMALY,
    higher_is_better=True,
    as_percent=True,
    breakdown=_recall_terms,
)
def anomaly_recall(inputs: MetricInputs) -> float:
    """Das anomalias verdadeiras, quantas foram marcadas (indefinida sem anomalias)."""
    truth = np.asarray(inputs.truth)
    if not np.any(truth == 1):
        return math.nan
    return float(sk.recall_score(truth, inputs.predicted))


@metric(
    "anomaly_roc_auc",
    display_name="Score ROC AUC",
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
