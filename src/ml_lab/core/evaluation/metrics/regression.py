"""Métricas de regressão: quão longe as previsões ficam dos valores reais.

Todas comparam ``y_true`` com ``y_pred``. O erro quadrático médio (MSE) e a sua raiz
(RMSE) castigam muito os erros grandes (estão ao quadrado); o erro absoluto médio (MAE)
trata todos os erros por igual; o R² compara o modelo com a referência mais simples,
prever sempre a média. Os nomes a mostrar são as siglas habituais (cabem nos cartões);
as explicações dizem-nas por extenso.
"""

import math

import numpy as np
from sklearn import metrics as sk

from ml_lab.core.evaluation.metrics.base import MetricInputs, metric
from ml_lab.core.foundation.types import TaskType

REGRESSION = (TaskType.REGRESSION,)


@metric("r2", display_name="R²", task_types=REGRESSION, higher_is_better=True)
def r2(inputs: MetricInputs) -> float:
    """Proporção da variação do alvo que o modelo explica (1 é perfeito, 0 é prever a média)."""
    if len(inputs.truth) < 2 or float(np.var(inputs.truth)) == 0.0:
        return math.nan  # com um só valor (ou todos iguais), não há variação para explicar
    return float(sk.r2_score(inputs.truth, inputs.predicted))


@metric("rmse", display_name="RMSE", task_types=REGRESSION, higher_is_better=False)
def rmse(inputs: MetricInputs) -> float:
    """Raiz do MSE: um erro "típico", nas unidades do alvo."""
    return float(np.sqrt(sk.mean_squared_error(inputs.truth, inputs.predicted)))


@metric("mae", display_name="MAE", task_types=REGRESSION, higher_is_better=False)
def mae(inputs: MetricInputs) -> float:
    """Média do valor absoluto dos erros, nas unidades do alvo."""
    return float(sk.mean_absolute_error(inputs.truth, inputs.predicted))


@metric("mse", display_name="MSE", task_types=REGRESSION, higher_is_better=False)
def mse(inputs: MetricInputs) -> float:
    """Média dos erros ao quadrado: a loss que a regressão linear minimiza."""
    return float(sk.mean_squared_error(inputs.truth, inputs.predicted))
