"""Métricas de classificação (binária e multiclasse), com o scikit-learn.

Na classificação binária, a classe "positiva" é a classe 1 (por exemplo, sobreviveu);
precisão, recall e F1 medem o desempenho nessa classe. Na multiclasse, calculam-se
para cada classe e faz-se a média simples (``macro``), para as classes raras pesarem
tanto como as frequentes.

Quando a conta não é possível (por exemplo, a ROC AUC com uma só classe no teste), a
métrica devolve NaN e o relatório explica porquê.
"""

import math
from typing import Final, Literal

from sklearn import metrics as sk

from ml_lab.core.evaluation.metrics.base import MetricInputs, metric
from ml_lab.core.foundation.types import TaskType

CLASSIFICATION: Final = (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION)


def _average(inputs: MetricInputs) -> Literal["binary", "macro"]:
    return "binary" if inputs.n_classes == 2 else "macro"


@metric(
    "accuracy",
    display_name="Exatidão",
    task_types=CLASSIFICATION,
    higher_is_better=True,
    as_percent=True,
)
def accuracy(inputs: MetricInputs) -> float:
    """Proporção de previsões certas."""
    return float(sk.accuracy_score(inputs.y_true, inputs.y_pred))


@metric(
    "balanced_accuracy",
    display_name="Exatidão equilibrada",
    task_types=CLASSIFICATION,
    higher_is_better=True,
    as_percent=True,
)
def balanced_accuracy(inputs: MetricInputs) -> float:
    """Média da recall de cada classe: cada classe pesa o mesmo, seja rara ou frequente.

    É a definição do ``balanced_accuracy_score`` do scikit-learn, escrita à mão para não
    avisar quando o modelo prevê uma classe que não aparece no conjunto avaliado.
    """
    seen = set(inputs.y_true.tolist())
    recalls = sk.recall_score(
        inputs.y_true, inputs.y_pred, labels=inputs.labels, average=None, zero_division=0
    )
    present = [float(value) for code, value in enumerate(recalls) if code in seen]
    return sum(present) / len(present) if present else math.nan


@metric(
    "precision",
    display_name="Precisão",
    task_types=CLASSIFICATION,
    higher_is_better=True,
    as_percent=True,
)
def precision(inputs: MetricInputs) -> float:
    """Das previsões de uma classe, quantas estavam certas."""
    return float(
        sk.precision_score(
            inputs.y_true,
            inputs.y_pred,
            labels=inputs.labels,
            average=_average(inputs),
            zero_division=0,
        )
    )


@metric(
    "recall",
    display_name="Recall (sensibilidade)",
    task_types=CLASSIFICATION,
    higher_is_better=True,
    as_percent=True,
)
def recall(inputs: MetricInputs) -> float:
    """Dos exemplos de uma classe, quantos o modelo encontrou."""
    return float(
        sk.recall_score(
            inputs.y_true,
            inputs.y_pred,
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
)
def f1(inputs: MetricInputs) -> float:
    """Média harmónica da precisão e da recall."""
    return float(
        sk.f1_score(
            inputs.y_true,
            inputs.y_pred,
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
)
def log_loss(inputs: MetricInputs) -> float:
    """Penaliza as probabilidades erradas, sobretudo as erradas e confiantes."""
    if inputs.y_proba is None:
        return math.nan
    return float(sk.log_loss(inputs.y_true, inputs.y_proba, labels=inputs.labels))


@metric(
    "roc_auc",
    display_name="ROC AUC",
    task_types=CLASSIFICATION,
    higher_is_better=True,
    needs_proba=True,
)
def roc_auc(inputs: MetricInputs) -> float:
    """Probabilidade de um exemplo positivo ter mais pontuação do que um negativo."""
    if inputs.y_proba is None or len(set(inputs.y_true.tolist())) < inputs.n_classes:
        # Sem todas as classes no conjunto avaliado, a curva ROC não está definida.
        return math.nan
    if inputs.n_classes == 2:
        return float(sk.roc_auc_score(inputs.y_true, inputs.y_proba[:, 1]))
    return float(
        sk.roc_auc_score(inputs.y_true, inputs.y_proba, multi_class="ovr", labels=inputs.labels)
    )
