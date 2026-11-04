"""Métricas de classificação (binária e multiclasse), com o scikit-learn.

Na classificação binária, a classe "positiva" é a classe 1 (por exemplo, sobreviveu);
precisão, recall e F1 medem o desempenho nessa classe. Na multiclasse, calculam-se
para cada classe e faz-se a média simples (``macro``), para as classes raras pesarem
tanto como as frequentes.

Quando a conta não é possível (por exemplo, a ROC AUC com uma só classe no teste), a
métrica devolve NaN e o relatório explica porquê.

Os termos de cada fórmula (``breakdown``) contam-se com as mesmas regras: na binária, os
da classe positiva; na multiclasse, os de cada classe (a métrica é a média deles).
"""

import math
from typing import Final, Literal

import numpy as np
from sklearn import metrics as sk

from ml_lab.core.evaluation.metrics.base import MetricInputs, MetricTerm, metric
from ml_lab.core.foundation.types import TaskType

CLASSIFICATION: Final = (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION)

_TRUE_POSITIVES: Final = (r"\mathrm{TP}", "True positives")
_FALSE_POSITIVES: Final = (r"\mathrm{FP}", "False positives")
_FALSE_NEGATIVES: Final = (r"\mathrm{FN}", "False negatives")


def _average(inputs: MetricInputs) -> Literal["binary", "macro"]:
    return "binary" if inputs.n_classes == 2 else "macro"


# --- os termos das fórmulas ----------------------------------------------------------------


def _classes(inputs: MetricInputs) -> list[int]:
    """As classes em que se faz a conta: só a positiva (a 1) na binária, todas na multiclasse."""
    return [1] if inputs.n_classes == 2 else inputs.labels


def _group(inputs: MetricInputs, code: int) -> int | None:
    """Na binária, os termos são da métrica inteira; na multiclasse, de cada classe."""
    return None if inputs.n_classes == 2 else code


def _counts(inputs: MetricInputs, code: int) -> tuple[int, int, int]:
    """Os verdadeiros positivos, os falsos positivos e os falsos negativos da classe ``code``."""
    truth, predicted = np.asarray(inputs.truth), np.asarray(inputs.predicted)
    said, was = predicted == code, truth == code
    return int(np.sum(said & was)), int(np.sum(said & ~was)), int(np.sum(~said & was))


def _ratio(part: float, whole: float) -> float:
    """``part / whole``, ou 0 sem nada a dividir (o ``zero_division=0`` do scikit-learn)."""
    return part / whole if whole else 0.0


def _accuracy_terms(inputs: MetricInputs) -> tuple[MetricTerm, ...]:
    truth, predicted = np.asarray(inputs.truth), np.asarray(inputs.predicted)
    return (
        MetricTerm(r"\text{correct}", "Right predictions", float(np.sum(truth == predicted))),
        MetricTerm("n", "Examples evaluated", float(len(truth))),
    )


def _balanced_accuracy_terms(inputs: MetricInputs) -> tuple[MetricTerm, ...]:
    seen = set(np.asarray(inputs.truth).tolist())
    terms = []
    for code in inputs.labels:
        if code in seen:  # as classes que não aparecem nos dados avaliados não entram na média
            true_positives, _, false_negatives = _counts(inputs, code)
            recall = _ratio(true_positives, true_positives + false_negatives)
            terms.append(MetricTerm(r"\text{recall}_c", "Recall of the class", recall, code))
    return tuple(terms)


def _precision_terms(inputs: MetricInputs) -> tuple[MetricTerm, ...]:
    terms = []
    for code in _classes(inputs):
        true_positives, false_positives, _ = _counts(inputs, code)
        group = _group(inputs, code)
        terms.append(MetricTerm(*_TRUE_POSITIVES, true_positives, group))
        terms.append(MetricTerm(*_FALSE_POSITIVES, false_positives, group))
    return tuple(terms)


def _recall_terms(inputs: MetricInputs) -> tuple[MetricTerm, ...]:
    terms = []
    for code in _classes(inputs):
        true_positives, _, false_negatives = _counts(inputs, code)
        group = _group(inputs, code)
        terms.append(MetricTerm(*_TRUE_POSITIVES, true_positives, group))
        terms.append(MetricTerm(*_FALSE_NEGATIVES, false_negatives, group))
    return tuple(terms)


def _f1_terms(inputs: MetricInputs) -> tuple[MetricTerm, ...]:
    terms = []
    for code in _classes(inputs):
        true_positives, false_positives, false_negatives = _counts(inputs, code)
        group = _group(inputs, code)
        precision = _ratio(true_positives, true_positives + false_positives)
        recall = _ratio(true_positives, true_positives + false_negatives)
        terms.append(MetricTerm("P", "Precision", precision, group))
        terms.append(MetricTerm("R", "Recall", recall, group))
    return tuple(terms)


def _log_loss_terms(inputs: MetricInputs) -> tuple[MetricTerm, ...]:
    if inputs.y_proba is None:
        return ()
    truth = np.asarray(inputs.truth, dtype=np.int64)
    # A probabilidade dada à classe certa; o mínimo evita log(0), como no scikit-learn.
    right = np.clip(inputs.y_proba[np.arange(len(truth)), truth], np.finfo(np.float64).eps, 1.0)
    return (
        MetricTerm(
            r"\sum_{i=1}^{n} -\log p_i", "Sum of the penalties", float(np.sum(-np.log(right)))
        ),
        MetricTerm("n", "Examples evaluated", float(len(truth))),
    )


# --- as métricas ---------------------------------------------------------------------------


@metric(
    "accuracy",
    display_name="Accuracy",
    task_types=CLASSIFICATION,
    higher_is_better=True,
    as_percent=True,
    breakdown=_accuracy_terms,
)
def accuracy(inputs: MetricInputs) -> float:
    """Proporção de previsões certas."""
    return float(sk.accuracy_score(inputs.truth, inputs.predicted))


@metric(
    "balanced_accuracy",
    display_name="Balanced accuracy",
    task_types=CLASSIFICATION,
    higher_is_better=True,
    as_percent=True,
    breakdown=_balanced_accuracy_terms,
)
def balanced_accuracy(inputs: MetricInputs) -> float:
    """Média da recall de cada classe: cada classe pesa o mesmo, seja rara ou frequente.

    É a definição do ``balanced_accuracy_score`` do scikit-learn, escrita à mão para não
    avisar quando o modelo prevê uma classe que não aparece no conjunto avaliado.
    """
    seen = set(inputs.truth.tolist())
    recalls = sk.recall_score(
        inputs.truth, inputs.predicted, labels=inputs.labels, average=None, zero_division=0
    )
    present = [float(value) for code, value in enumerate(recalls) if code in seen]
    return sum(present) / len(present) if present else math.nan


@metric(
    "precision",
    display_name="Precision",
    task_types=CLASSIFICATION,
    higher_is_better=True,
    as_percent=True,
    breakdown=_precision_terms,
)
def precision(inputs: MetricInputs) -> float:
    """Das previsões de uma classe, quantas estavam certas."""
    return float(
        sk.precision_score(
            inputs.truth,
            inputs.predicted,
            labels=inputs.labels,
            average=_average(inputs),
            zero_division=0,
        )
    )


@metric(
    "recall",
    display_name="Recall (sensitivity)",
    task_types=CLASSIFICATION,
    higher_is_better=True,
    as_percent=True,
    breakdown=_recall_terms,
)
def recall(inputs: MetricInputs) -> float:
    """Dos exemplos de uma classe, quantos o modelo encontrou."""
    return float(
        sk.recall_score(
            inputs.truth,
            inputs.predicted,
            labels=inputs.labels,
            average=_average(inputs),
            zero_division=0,
        )
    )


@metric(
    "f1",
    display_name="F1",
    task_types=CLASSIFICATION,
    higher_is_better=True,
    as_percent=True,
    breakdown=_f1_terms,
)
def f1(inputs: MetricInputs) -> float:
    """Média harmónica da precisão e da recall."""
    return float(
        sk.f1_score(
            inputs.truth,
            inputs.predicted,
            labels=inputs.labels,
            average=_average(inputs),
            zero_division=0,
        )
    )


@metric(
    "log_loss",
    display_name="Log-loss",
    task_types=CLASSIFICATION,
    higher_is_better=False,
    needs_proba=True,
    breakdown=_log_loss_terms,
)
def log_loss(inputs: MetricInputs) -> float:
    """Penaliza as probabilidades erradas, sobretudo as erradas e confiantes."""
    if inputs.y_proba is None:
        return math.nan
    return float(sk.log_loss(inputs.truth, inputs.y_proba, labels=inputs.labels))


@metric(
    "roc_auc",
    display_name="ROC AUC",
    task_types=CLASSIFICATION,
    higher_is_better=True,
    needs_proba=True,
)
def roc_auc(inputs: MetricInputs) -> float:
    """Probabilidade de um exemplo positivo ter mais pontuação do que um negativo."""
    if inputs.y_proba is None or len(set(inputs.truth.tolist())) < inputs.n_classes:
        # Sem todas as classes no conjunto avaliado, a curva ROC não está definida.
        return math.nan
    if inputs.n_classes == 2:
        return float(sk.roc_auc_score(inputs.truth, inputs.y_proba[:, 1]))
    return float(
        sk.roc_auc_score(inputs.truth, inputs.y_proba, multi_class="ovr", labels=inputs.labels)
    )
