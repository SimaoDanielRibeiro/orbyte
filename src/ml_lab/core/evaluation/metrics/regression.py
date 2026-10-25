"""Métricas de regressão: quão longe as previsões ficam dos valores reais.

Todas comparam ``y_true`` com ``y_pred``. O erro quadrático médio (MSE) e a sua raiz
(RMSE) castigam muito os erros grandes (estão ao quadrado); o erro absoluto médio (MAE)
trata todos os erros por igual; o R² compara o modelo com a referência mais simples,
prever sempre a média. Os nomes a mostrar são as siglas habituais (cabem nos cartões);
as explicações dizem-nas por extenso.
"""

import math
from typing import Final

import numpy as np
from numpy.typing import NDArray
from sklearn import metrics as sk

from ml_lab.core.evaluation.metrics.base import MetricInputs, MetricTerm, metric
from ml_lab.core.foundation.types import TaskType

REGRESSION = (TaskType.REGRESSION,)

_COUNT: Final = ("n", "Exemplos avaliados")


# --- os termos das fórmulas ----------------------------------------------------------------


def _errors(inputs: MetricInputs) -> NDArray[np.float64]:
    """O erro de cada linha: o valor real menos a previsão."""
    truth = np.asarray(inputs.truth, dtype=np.float64)
    return truth - np.asarray(inputs.predicted, dtype=np.float64)


def _squared_terms(inputs: MetricInputs) -> tuple[MetricTerm, ...]:
    errors = _errors(inputs)
    return (
        MetricTerm(
            r"\sum_{i=1}^{n} (y_i - \hat{y}_i)^2",
            "Soma dos erros ao quadrado",
            float(np.sum(errors**2)),
        ),
        MetricTerm(*_COUNT, float(len(errors))),
    )


def _absolute_terms(inputs: MetricInputs) -> tuple[MetricTerm, ...]:
    errors = _errors(inputs)
    return (
        MetricTerm(
            r"\sum_{i=1}^{n} |y_i - \hat{y}_i|",
            "Soma dos erros absolutos",
            float(np.sum(np.abs(errors))),
        ),
        MetricTerm(*_COUNT, float(len(errors))),
    )


def _r2_terms(inputs: MetricInputs) -> tuple[MetricTerm, ...]:
    truth = np.asarray(inputs.truth, dtype=np.float64)
    return (
        MetricTerm(
            r"\mathrm{SQ}_{\text{res}}",
            "Soma dos quadrados dos resíduos",
            float(np.sum(_errors(inputs) ** 2)),
        ),
        MetricTerm(
            r"\mathrm{SQ}_{\text{tot}}",
            "Soma dos quadrados à volta da média",
            float(np.sum((truth - truth.mean()) ** 2)),
        ),
    )


# --- as métricas ---------------------------------------------------------------------------


@metric("r2", display_name="R²", task_types=REGRESSION, higher_is_better=True, breakdown=_r2_terms)
def r2(inputs: MetricInputs) -> float:
    """Proporção da variação do alvo que o modelo explica (1 é perfeito, 0 é prever a média)."""
    if len(inputs.truth) < 2 or float(np.var(inputs.truth)) == 0.0:
        return math.nan  # com um só valor (ou todos iguais), não há variação para explicar
    return float(sk.r2_score(inputs.truth, inputs.predicted))


@metric(
    "rmse",
    display_name="RMSE",
    task_types=REGRESSION,
    higher_is_better=False,
    breakdown=_squared_terms,
)
def rmse(inputs: MetricInputs) -> float:
    """Raiz do MSE: um erro "típico", nas unidades do alvo."""
    return float(np.sqrt(sk.mean_squared_error(inputs.truth, inputs.predicted)))


@metric(
    "mae",
    display_name="MAE",
    task_types=REGRESSION,
    higher_is_better=False,
    breakdown=_absolute_terms,
)
def mae(inputs: MetricInputs) -> float:
    """Média do valor absoluto dos erros, nas unidades do alvo."""
    return float(sk.mean_absolute_error(inputs.truth, inputs.predicted))


@metric(
    "mse",
    display_name="MSE",
    task_types=REGRESSION,
    higher_is_better=False,
    breakdown=_squared_terms,
)
def mse(inputs: MetricInputs) -> float:
    """Média dos erros ao quadrado: a loss que a regressão linear minimiza."""
    return float(sk.mean_squared_error(inputs.truth, inputs.predicted))
